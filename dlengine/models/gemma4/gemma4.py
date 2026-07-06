import math
from collections import UserDict

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

from dlengine.context_v2.batch import get_batch_context
from dlengine.context_v2.cache.plan import gqa_cache_plan, gqa_hisparse_cache_plan
from dlengine.context_v2.distributed import get_dist_context
from dlengine.layers import get_backend
from dlengine.layers.base_backend import ColumnParallelLinearBase, RowParallelLinearBase
from dlengine.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from dlengine.layers.layernorm import RMSNorm


def _gelu_pytorch_tanh(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


def _activation(name: str):
    if name in ("gelu_pytorch_tanh", "gelu_fast", "gelu"):
        return _gelu_pytorch_tanh
    if name == "silu":
        return F.silu
    raise NotImplementedError(f"Gemma4 activation {name!r} is not supported yet")


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Gemma4RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        base: float,
        partial_rotary_factor: float = 1.0,
        proportional: bool = False,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        if proportional:
            rope_angles = int(partial_rotary_factor * head_dim // 2)
            inv_freq_rotated = 1.0 / (
                base
                ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32) / head_dim)
            )
            nope_angles = head_dim // 2 - rope_angles
            inv_freq = (
                torch.cat((inv_freq_rotated, torch.zeros(nope_angles)))
                if nope_angles > 0
                else inv_freq_rotated
            )
        else:
            inv_freq = 1.0 / (
                base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
            )
        self.max_position_embeddings = max_position_embeddings
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.to(device=positions.device)
        freqs = torch.einsum("i,j->ij", positions.float(), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(1).to(dtype=query.dtype)
        sin = emb.sin().unsqueeze(1).to(dtype=query.dtype)
        query = query * cos + _rotate_half(query) * sin
        key = key * cos + _rotate_half(key) * sin
        return query, key


class ScaledVocabEmbedding(VocabParallelEmbedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, embed_scale: float):
        super().__init__(num_embeddings, embedding_dim)
        self.embed_scale = embed_scale

    def forward(self, x: torch.Tensor):
        return super().forward(x) * self.embed_scale


class CpuScaledVocabEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, embed_scale: float):
        super().__init__()
        dist = get_dist_context()
        self.tp_rank = dist.attn_tp_rank
        self.tp_size = dist.attn_tp_world_size
        if self.tp_size != 1:
            raise NotImplementedError(
                "Gemma4 CPU PLE embedding currently requires TP=1"
            )
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device="cpu")
        )
        self.embed_scale = embed_scale

    def forward(self, x: torch.Tensor, device: torch.device, dtype: torch.dtype):
        y = F.embedding(x.cpu(), self.weight)
        return y.to(device=device, dtype=dtype, non_blocking=True) * self.embed_scale


class TiedParallelLMHead(nn.Module):
    def __init__(self, embedding: VocabParallelEmbedding):
        super().__init__()
        if embedding.tp_size != 1:
            raise NotImplementedError("Gemma4 tied lm_head currently requires TP=1")
        self.weight = embedding.weight

    def forward(self, x: torch.Tensor):
        context = get_batch_context()
        if context.is_prefill:
            if context.sampling_token_indices is not None:
                x = x[context.sampling_token_indices].contiguous()
            else:
                last_indices = context.cu_seqlens_q[1:] - 1
                x = x[last_indices].contiguous()
        return F.linear(x, self.weight)


class Gemma4Attention(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"
        if getattr(config, "attention_k_eq_v", False):
            raise NotImplementedError("Gemma4 attention_k_eq_v is not wired yet")
        first_kv_shared_layer_idx = config.num_hidden_layers - getattr(
            config, "num_kv_shared_layers", 0
        )
        self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        prev_layers = config.layer_types[:first_kv_shared_layer_idx]
        self.store_full_length_kv = not self.is_kv_shared_layer and layer_idx == len(
            prev_layers
        ) - 1 - prev_layers[::-1].index(config.layer_types[layer_idx])

        tp_size = get_dist_context().attn_tp_world_size
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = (
            config.head_dim
            if self.is_sliding or not getattr(config, "global_head_dim", None)
            else config.global_head_dim
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = 1.0

        backend = get_backend()
        self.q_proj: ColumnParallelLinearBase = backend.get_column_parallel_linear(
            config.hidden_size,
            self.total_num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        if not self.is_kv_shared_layer:
            self.k_proj: ColumnParallelLinearBase = backend.get_column_parallel_linear(
                config.hidden_size,
                self.total_num_kv_heads * self.head_dim,
                bias=config.attention_bias,
            )
            self.v_proj: ColumnParallelLinearBase = backend.get_column_parallel_linear(
                config.hidden_size,
                self.total_num_kv_heads * self.head_dim,
                bias=config.attention_bias,
            )
        self.o_proj: RowParallelLinearBase = backend.get_row_parallel_linear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        if not self.is_kv_shared_layer:
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.v_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        rope_params = config.rope_parameters[self.layer_type]
        self.rotary_emb = Gemma4RotaryEmbedding(
            self.head_dim,
            config.max_position_embeddings,
            rope_params["rope_theta"],
            partial_rotary_factor=rope_params.get("partial_rotary_factor", 1.0),
            proportional=rope_params.get("rope_type") == "proportional",
        )
        self.attn = backend.get_attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            self.head_dim,
            "GQA",
            sliding_window=config.sliding_window if self.is_sliding else None,
        )
        if self.is_sliding and getattr(config, "enable_hisparse", False):
            self.attn.use_paged_kv_cache = False
        if self.is_kv_shared_layer:
            self.attn.use_paged_kv_cache = False

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        q = self.q_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
        q = self.q_norm(q)
        if self.is_kv_shared_layer:
            k, v = shared_kv_states[self.layer_type]
        else:
            k = self.k_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
            v = self.v_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
            k = self.k_norm(k)
            v = self.v_norm(v)
            _, k = self.rotary_emb(positions, q, k)
            if self.store_full_length_kv:
                shared_kv_states[self.layer_type] = (k, v)
        q, _ = self.rotary_emb(positions, q, k)
        out = self.attn(q, k, v)
        return self.o_proj(out.flatten(1, -1))


class Gemma4MLP(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int) -> None:
        super().__init__()
        backend = get_backend()
        first_kv_shared_layer_idx = config.num_hidden_layers - getattr(
            config, "num_kv_shared_layers", 0
        )
        is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        use_double_wide_mlp = (
            getattr(config, "use_double_wide_mlp", False) and is_kv_shared_layer
        )
        intermediate_size = config.intermediate_size * (2 if use_double_wide_mlp else 1)
        self.gate_proj = backend.get_column_parallel_linear(
            config.hidden_size, intermediate_size, bias=False
        )
        self.up_proj = backend.get_column_parallel_linear(
            config.hidden_size, intermediate_size, bias=False
        )
        self.down_proj = backend.get_row_parallel_linear(
            intermediate_size, config.hidden_size, bias=False
        )
        self.act_fn = _activation(config.hidden_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Gemma4DecoderLayer(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int) -> None:
        super().__init__()
        if getattr(config, "enable_moe_block", False):
            raise NotImplementedError("Gemma4 MoE block is not wired yet")
        self.self_attn = Gemma4Attention(config, layer_idx)
        self.mlp = Gemma4MLP(config, layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.layer_scalar = nn.Parameter(torch.ones(1), requires_grad=False)
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            backend = get_backend()
            self.per_layer_input_gate = backend.get_replicated_linear(
                config.hidden_size, self.hidden_size_per_layer_input, bias=False
            )
            self.per_layer_projection = backend.get_replicated_linear(
                self.hidden_size_per_layer_input, config.hidden_size, bias=False
            )
            self.post_per_layer_input_norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.act_fn = _activation(config.hidden_activation)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor | None = None,
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        if shared_kv_states is None:
            shared_kv_states = {}
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states, shared_kv_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        if self.hidden_size_per_layer_input:
            if per_layer_input is None:
                raise RuntimeError("Gemma4 PLE expected per_layer_input")
            residual = hidden_states
            hidden_states = self.per_layer_input_gate(hidden_states)
            hidden_states = self.act_fn(hidden_states) * per_layer_input
            hidden_states = self.per_layer_projection(hidden_states)
            hidden_states = self.post_per_layer_input_norm(hidden_states)
            hidden_states = residual + hidden_states

        return hidden_states * self.layer_scalar


class Gemma4Model(nn.Module):
    def __init__(self, config: Gemma4TextConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = ScaledVocabEmbedding(
            config.vocab_size, config.hidden_size, math.sqrt(config.hidden_size)
        )
        self.layers = nn.ModuleList(
            [Gemma4DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = CpuScaledVocabEmbedding(
                config.vocab_size_per_layer_input,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                math.sqrt(config.hidden_size_per_layer_input),
            )
            backend = get_backend()
            self.per_layer_model_projection = backend.get_replicated_linear(
                config.hidden_size,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                bias=False,
            )
            self.per_layer_projection_norm = RMSNorm(
                config.hidden_size_per_layer_input, eps=config.rms_norm_eps
            )
            self.per_layer_input_scale = 2.0**-0.5
            self.per_layer_model_projection_scale = config.hidden_size**-0.5

    def wire_shared_kv_caches(self) -> None:
        source_by_type: dict[str, Gemma4Attention] = {}
        for layer in self.layers:
            attn = layer.self_attn
            if attn.store_full_length_kv:
                source_by_type[attn.layer_type] = attn
        for layer in self.layers:
            attn = layer.self_attn
            if not attn.is_kv_shared_layer:
                continue
            source = source_by_type.get(attn.layer_type)
            if source is None:
                raise RuntimeError(
                    f"Gemma4 shared KV source missing for layer_type={attn.layer_type!r}"
                )
            attn.attn.k_cache = source.attn.k_cache
            attn.attn.v_cache = source.attn.v_cache
            attn.attn.hisparse_k_cache = source.attn.hisparse_k_cache
            attn.attn.hisparse_v_cache = source.attn.hisparse_v_cache

    def _per_layer_inputs(
        self, input_ids: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor | None:
        if not self.hidden_size_per_layer_input:
            return None
        token_part = self.embed_tokens_per_layer(
            input_ids, hidden_states.device, hidden_states.dtype
        )
        token_part = token_part.view(
            -1,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        model_part = self.per_layer_model_projection(hidden_states)
        model_part = model_part * self.per_layer_model_projection_scale
        model_part = model_part.view(
            -1,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        model_part = self.per_layer_projection_norm(model_part)
        return (model_part + token_part) * self.per_layer_input_scale

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        per_layer_inputs = self._per_layer_inputs(input_ids, hidden_states)
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]] = UserDict()
        for i, layer in enumerate(self.layers):
            pli = per_layer_inputs[:, i, :] if per_layer_inputs is not None else None
            hidden_states = layer(positions, hidden_states, pli, shared_kv_states)
        return self.norm(hidden_states)


class Gemma4ForCausalLM(nn.Module):
    def __init__(self, config: Gemma4TextConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Gemma4Model(config)
        if config.tie_word_embeddings:
            self.lm_head = TiedParallelLMHead(self.model.embed_tokens)
        else:
            self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        softcap = getattr(self.config, "final_logit_softcapping", None)
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        return logits

    def get_cache_plan(self):
        if getattr(self.config, "enable_hisparse", False):
            return gqa_hisparse_cache_plan()
        return gqa_cache_plan()

    def load_weights(self, weights):
        from .gemma4_loader import load_weights

        load_weights(self, weights)
