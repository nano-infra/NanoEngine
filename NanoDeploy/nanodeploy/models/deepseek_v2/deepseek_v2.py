import math
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from transformers import DeepseekV3Config

from nanodeploy.backends import get_backend
from nanodeploy.backends.base_backend import (
    ColumnParallelLinearBase,
    DistributedRoutedExpertsBase,
    MergedColumnParallelLinearBase,
    RowParallelLinearBase,
)
from nanodeploy.backends.hopper.layers.attention import (
    _gather_cache_cached_only,
    _interleave_cached_fresh,
)
from nanodeploy.context.context import get_context
from nanodeploy.context.distributed import get_dist_context
from nanodeploy.layers.activation import SiluAndMul
from nanodeploy.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanodeploy.layers.layernorm import RMSNorm
from nanodeploy.layers.parallelism_transition import (
    AttnToFfnTransition,
    FfnToAttnTransition,
)
from nanodeploy.layers.rotary_embedding import get_rope
from nanodeploy.logging import get_logger
from nanodeploy.worker.runner_config import get_runner_config
from ..quant_config import QuantizationConfig

logger = get_logger()


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _interleaved_to_half(x: torch.Tensor) -> torch.Tensor:
    """Convert RoPE dims from interleaved to half format.

    Interleaved pairs: (d0,d1), (d2,d3), ...
    Half pairs:        (d0,d32), (d1,d33), ... (first half paired with second half)

    This is needed because DeepseekV3 checkpoints use rope_interleave=True,
    meaning projection weights produce PE dims in interleaved layout, but
    our apply_rotary_emb uses the half-rotation layout.
    """
    *leading, d = x.shape
    return x.unflatten(-1, (-1, 2)).transpose(-1, -2).contiguous().flatten(-2)


def compute_topk_ids(topk_ids, ranks, num_experts):
    shape = topk_ids.shape
    step = num_experts // ranks
    topk_ids = (
        (
            torch.arange(
                0, topk_ids.numel(), dtype=topk_ids.dtype, device=topk_ids.device
            )
            // ranks
        )
        % step
        + (
            torch.arange(
                0, topk_ids.numel(), dtype=topk_ids.dtype, device=topk_ids.device
            )
            % ranks
        )
        * step
    ) % num_experts
    topk_ids = topk_ids.reshape(shape)
    return topk_ids


# 已改


class DeepseekV2MoE(nn.Module):
    """Deepseek v2/v3 MoE."""

    def __init__(
        self, config: DeepseekV3Config, quantization_config: QuantizationConfig
    ):
        super().__init__()
        self.config = config
        self.quantization_config = quantization_config

        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok

        # Routing config
        self.n_group = getattr(config, "n_group", 1)
        self.topk_group = getattr(config, "topk_group", 1)
        self.norm_topk_prob = getattr(config, "norm_topk_prob", True)
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.scoring_func = getattr(config, "scoring_func", "sigmoid")

        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        # e_score_correction_bias: loaded from safetensors, registered as param for easy loading
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.zeros(self.num_experts, dtype=torch.float32, device="cuda"),
            requires_grad=False,
        )

        self.ep_group = get_dist_context().ffn_ep_group
        self.ep_size = get_dist_context().ffn_ep_world_size
        self.tp_group = get_dist_context().ffn_tp_group
        self.tp_size = get_dist_context().ffn_tp_world_size

        self.routed_experts: (
            DistributedRoutedExpertsBase
        ) = get_backend().get_distributed_routed_experts(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            num_experts=self.num_experts,
            top_k=self.top_k,
            ep_size=self.ep_size,
            tp_size=self.tp_size,
            ep_group=self.ep_group,
            tp_group=self.tp_group,
            n_group=self.n_group,
            topk_group=self.topk_group,
            norm_topk_prob=self.norm_topk_prob,
            routed_scaling_factor=self.routed_scaling_factor,
            scoring_func=self.scoring_func,
        )

        self.shared_experts = None
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                meta=False,
                config=config,
                quantization_config=quantization_config,
            )

    @property
    def num_experts_per_rank(self):
        ep_world_size = get_dist_context().ffn_ep_world_size
        return self.num_experts // ep_world_size

    def route_tokens_to_experts(self, router_logits: torch.Tensor):
        """Sigmoid routing with group-limited topk (noaux_tc).

        Returns:
            topk_indices: (batch, top_k)
            topk_weights: (batch, top_k)
        """
        if self.scoring_func == "sigmoid":
            scores = router_logits.float().sigmoid()
        else:
            scores = F.softmax(router_logits.float(), dim=-1)

        scores_for_choice = scores + self.gate.e_score_correction_bias.float()

        # Group-limited topk selection
        if self.n_group > 1:
            group_scores = (
                scores_for_choice.view(
                    -1, self.n_group, self.num_experts // self.n_group
                )
                .topk(2, dim=-1)[0]
                .sum(dim=-1)
            )
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(-1, self.n_group, self.num_experts // self.n_group)
                .reshape(-1, self.num_experts)
            )
            scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)

        topk_indices = torch.topk(
            scores_for_choice, k=self.top_k, dim=-1, sorted=False
        )[1]
        topk_weights = scores.gather(1, topk_indices)

        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator
        topk_weights = topk_weights * self.routed_scaling_factor

        return topk_indices, topk_weights

    def forward(self, hidden_states: torch.Tensor):
        """forward."""
        batch_size, hidden_dim = hidden_states.shape
        residual = hidden_states
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states)
        topk_idx, topk_weights = self.route_tokens_to_experts(router_logits)

        context = get_context()
        is_prefill = context.is_prefill

        final_hidden_states = self.routed_experts(
            hidden_states, topk_idx, topk_weights, is_prefill=is_prefill
        )

        if self.shared_experts is not None:
            shared_states = self.shared_experts(residual)
            final_hidden_states = final_hidden_states + shared_states.view(
                -1, hidden_dim
            )

        final_hidden_states = final_hidden_states.reshape(batch_size, -1)
        return final_hidden_states


class DeepseekV2MLP(nn.Module):
    """Deepseek v2/v3 mlp."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int = None,
        hidden_act: str = "silu",
        gate_up_proj_tensor: torch.Tensor | None = None,
        down_proj_tensor: torch.Tensor | None = None,
        gate_up_scale_inv_tensor: torch.Tensor | None = None,
        down_scale_inv_tensor: torch.Tensor | None = None,
        meta: bool = False,
        config: DeepseekV3Config | None = None,
        quantization_config: QuantizationConfig | None = None,
    ):
        super().__init__()

        self.config = config
        self.quantization_config = quantization_config

        self.gate_up_proj: (
            MergedColumnParallelLinearBase
        ) = get_backend().get_merged_column_parallel_linear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            meta=meta,
            weight_tensor=gate_up_proj_tensor,
            scale_tensor=gate_up_scale_inv_tensor,
            tp_group=get_dist_context().ffn_tp_group,
        )

        self.down_proj: RowParallelLinearBase = get_backend().get_row_parallel_linear(
            intermediate_size,
            hidden_size,
            bias=False,
            meta=meta,
            weight_tensor=down_proj_tensor,
            scale_tensor=down_scale_inv_tensor,
            tp_group=get_dist_context().ffn_tp_group,
        )

        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        """forward."""
        gate_up = self.gate_up_proj(x)
        act = self.act_fn(gate_up)
        x = self.down_proj(act)
        return x


class DeepseekV2DecoderLayer(nn.Module):
    """Deepseekv2 decoder layer."""

    def __init__(
        self,
        config: DeepseekV3Config,
        quantization_config: QuantizationConfig,
        layer_idx: int,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.self_attn = DeepseekV2Attention(
            config, quantization_config, layer_idx=layer_idx
        )

        if (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
        ):
            self.mlp = DeepseekV2MoE(
                config=config, quantization_config=quantization_config
            )
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                config=config,
                quantization_config=quantization_config,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Parallelism transition
        attn_tp = get_dist_context().attn_tp_world_size
        ffn_ep = get_dist_context().ffn_ep_world_size
        if attn_tp > 1 and ffn_ep > 1:
            self.attn_to_ffn = AttnToFfnTransition()
            self.ffn_to_attn = FfnToAttnTransition(scatter_layer=self.attn_to_ffn)
        else:
            self.attn_to_ffn = nn.Identity()
            self.ffn_to_attn = nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        # Self Attention

        hidden_states = self.self_attn(positions, hidden_states)

        # Fully Connected

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.attn_to_ffn(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.ffn_to_attn(hidden_states)

        outputs = (hidden_states, residual)
        return outputs


# 已改


class DeepseekV2Model(nn.Module):
    """Deepseek v2 model."""

    def __init__(
        self, config: DeepseekV3Config, quantization_config: QuantizationConfig
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV2DecoderLayer(config, quantization_config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        positions: Optional[torch.LongTensor] = None,
    ):
        """forward."""
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for idx, decoder_layer in enumerate(self.layers):
            hidden_states, residual = decoder_layer(hidden_states, positions, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class DeepseekV2ForCausalLM(nn.Module):
    """Mixture model for causalLM."""

    def __init__(self, config: DeepseekV3Config):
        super().__init__()
        self.config = config
        self.quantization_config = QuantizationConfig(
            **getattr(config, "quantization_config", dict())
        )
        self.model = DeepseekV2Model(config, self.quantization_config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor):
        """Compute logits of the model output."""
        return self.lm_head(hidden_states)

    def load_weights(self, weights):
        """Load weights using per-model loader."""
        from .deepseek_v2_loader import load_weights

        load_weights(self, weights)


class DeepseekV2BMM(nn.Module):
    """Wrapped bmm."""

    def __init__(self, batch: int, in_features: int, out_features: int):
        super().__init__()

        weight = self.create_weight(batch, in_features, out_features)
        self.weight = torch.nn.Parameter(weight, requires_grad=False)

    def create_weight(self, batch: int, in_features: int, out_features: int):
        """Create weight."""
        return torch.empty((batch, in_features, out_features))

    def forward(self, x: torch.Tensor, output: torch.Tensor):
        """forward."""
        torch.bmm(x.transpose(0, 1), self.weight, out=output.transpose(0, 1))


class DeepseekV2Attention(nn.Module):
    """Deepseekv2 attention."""

    def __init__(
        self,
        config: DeepseekV3Config,
        quantization_config: QuantizationConfig | None = None,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.q_lora_rank = config.q_lora_rank
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        # For MLA, effective num_kv_heads is 1 (single compressed KV representation)
        num_key_value_heads = 1
        self.is_v32 = hasattr(config, "index_topk")

        if self.q_lora_rank is None:
            self.q_proj: (
                ColumnParallelLinearBase
            ) = get_backend().get_column_parallel_linear(
                self.hidden_size,
                self.num_heads * self.q_head_dim,
                tp_group=get_dist_context().attn_tp_group,
            )
        else:
            self.q_a_proj: (
                ColumnParallelLinearBase
            ) = get_backend().get_column_parallel_linear(
                self.hidden_size,
                config.q_lora_rank,
                bias=config.attention_bias,
                tp_group=get_dist_context().attn_tp_group,
            )
            self.q_a_layernorm = RMSNorm(hidden_size=config.q_lora_rank, eps=1e-6)
            self.q_b_proj: (
                ColumnParallelLinearBase
            ) = get_backend().get_column_parallel_linear(
                config.q_lora_rank,
                self.num_heads * self.q_head_dim,
                bias=False,
                tp_group=get_dist_context().attn_tp_group,
            )
        self.kv_a_proj_with_mqa: (
            ColumnParallelLinearBase
        ) = get_backend().get_column_parallel_linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
            tp_group=get_dist_context().attn_tp_group,
        )
        self.kv_a_layernorm = RMSNorm(
            config.kv_lora_rank,
            1e-6,
        )
        self.kc = DeepseekV2BMM(
            self.num_heads,
            config.qk_nope_head_dim,
            config.kv_lora_rank,
        )

        rope_dim = (
            config.qk_rope_head_dim
            if getattr(config, "use_mla", True)
            else (config.hidden_size // config.num_attention_heads)
        )

        # Extract rope_theta and rope_scaling from config.
        # Built-in DeepseekV3Config stores these inside rope_parameters dict;
        # older custom configs may have rope_theta / rope_scaling as top-level attrs.
        rope_params = getattr(config, "rope_parameters", None) or getattr(
            config, "rope_scaling", None
        )
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None:
            rope_theta = (rope_params or {}).get(
                "rope_theta", getattr(config, "default_theta", 10000.0)
            )

        self.rotary_emb = get_rope(
            rope_dim,
            rotary_dim=rope_dim,
            max_position=config.max_position_embeddings,
            base=float(rope_theta),
            rope_scaling=rope_params,
        )

        self.softmax_scale = self.q_head_dim ** (-0.5)

        if rope_params is not None:
            mscale_all_dim = rope_params.get("mscale_all_dim", 0)
            scaling_factor = rope_params.get("factor", 1.0)
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale
        self.attn_fwd = get_backend().get_attention(
            self.num_heads,
            config.kv_lora_rank + self.qk_rope_head_dim,
            scale=self.softmax_scale,
            num_kv_heads=num_key_value_heads,
            v_head_dim=config.kv_lora_rank,
            attention_type="MLA",
            nsa_index_topk=getattr(config, "index_topk", 0),
        )

        self.vc = DeepseekV2BMM(self.num_heads, config.kv_lora_rank, self.v_head_dim)

        self.o_proj: RowParallelLinearBase = get_backend().get_row_parallel_linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
            tp_group=get_dist_context().attn_tp_group,
        )

        # NSA Indexer (V3.2 only)
        if self.is_v32:
            from nanodeploy.layers.indexer import Indexer

            self.indexer = Indexer(
                hidden_size=config.hidden_size,
                index_n_heads=config.index_n_heads,
                index_head_dim=config.index_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                q_lora_rank=config.q_lora_rank,
                index_topk=config.index_topk,
                max_position_embeddings=config.max_position_embeddings,
                rope_theta=float(rope_theta),
                rope_scaling=rope_params,
                layer_id=layer_idx,
            )
        else:
            self.indexer = None

    def _q_proj_absorbed(self, hidden_states, num_heads: int):
        """Q proj with W_UK absorption (for decode).

        Returns:
            query_states: (q_len, H, kv_lora_rank + qk_rope_head_dim) = (q_len, H, 576)
            q_pe: (q_len, H, qk_rope_head_dim)
            q_lora: (q_len, q_lora_rank) — intermediate for indexer (or None if no q_lora_rank)
        """
        q_len = hidden_states.size(0)
        nope_size = self.kv_lora_rank  # 512
        pe_size = self.qk_rope_head_dim  # 64

        query_states = hidden_states.new_empty([q_len, num_heads, nope_size + pe_size])

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
            q_lora = None
        else:
            q_lora = self.q_a_layernorm(self.q_a_proj(hidden_states))
            q = self.q_b_proj(q_lora)
        q = q.view(q_len, num_heads, self.q_head_dim)

        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # Absorb W_UK: q_nope (q_len, H, D) @ kc (H, D, R) -> (q_len, H, R)
        q_nope_out = query_states[..., :nope_size]
        self.kc(q_nope, q_nope_out)
        return query_states, q_pe, q_lora

    def _q_proj_raw(self, hidden_states, num_heads: int):
        """Q proj without absorption (for prefill)."""
        q_len = hidden_states.size(0)

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(q_len, num_heads, self.q_head_dim)
        # q: (q_len, num_heads, qk_nope_head_dim + qk_rope_head_dim) = (q_len, H, 192)

        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        return q, q_nope, q_pe

    def _kv_proj(self, hidden_states):
        """Kv proj: returns compressed KV and k_pe."""
        nope_size = self.kv_lora_rank

        key_states = self.kv_a_proj_with_mqa(hidden_states)
        # key_states: (q_len, kv_lora_rank + qk_rope_head_dim)

        k_pe = key_states[..., nope_size:]
        # k_pe: (q_len, qk_rope_head_dim)

        value_states = key_states[..., :nope_size]
        value_states = self.kv_a_layernorm(value_states)
        key_states[..., :nope_size] = value_states
        # key_states: (q_len, kv_lora_rank + qk_rope_head_dim) with normalized latent
        # value_states: (q_len, kv_lora_rank) — normalized compressed latent
        return key_states, value_states, k_pe

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ):
        """Forward with separate prefill (non-absorbed) and decode (absorbed) paths."""
        num_heads = self.num_heads
        q_len = hidden_states.size(0)
        is_prefill = get_context().is_prefill

        # KV projection (shared between prefill and decode)
        key_states, compressed_kv, k_pe = self._kv_proj(hidden_states)
        # key_states: (q_len, kv_lora_rank + qk_rope_head_dim) = (q_len, 576)
        # compressed_kv: (q_len, kv_lora_rank) = (q_len, 512)
        # k_pe: (q_len, qk_rope_head_dim) = (q_len, 64)

        if is_prefill:
            # === Non-absorbed prefill path ===
            # Q: original (not absorbed), shape (q_len, H, qk_nope+qk_rope) = (q_len, H, 192)
            q_full, q_nope, q_pe = self._q_proj_raw(hidden_states, num_heads)

            # Convert PE dims from interleaved to half format before RoPE
            # (DeepseekV3 uses rope_interleave=True; projections produce interleaved layout)
            q_pe = _interleaved_to_half(q_pe)
            k_pe_3d = k_pe.unsqueeze(1)  # (q_len, 1, rope_dim)
            k_pe_3d = _interleaved_to_half(k_pe_3d)
            q_pe, k_pe_3d = self.rotary_emb(positions, q_pe, k_pe_3d)
            q_full[..., self.qk_nope_head_dim :] = q_pe  # write RoPE'd q_pe back

            # Also write RoPE'd k_pe into key_states for KV cache storage
            key_states_3d = key_states.unsqueeze(1)  # (q_len, 1, 576)
            key_states_3d[..., self.kv_lora_rank :] = k_pe_3d

            # Store compressed KV (576 dims) into cache for future decode
            k_cache = self.attn_fwd.k_cache
            context = get_context()
            if (
                k_cache.numel()
                and not context.is_dummy
                and context.slot_mapping is not None
            ):
                slot_mapping = context.slot_mapping
                if slot_mapping.numel() != key_states_3d.shape[0]:
                    raise RuntimeError(
                        f"MLA prefill store_kcache shape mismatch: "
                        f"key_states_3d.shape={key_states_3d.shape}, "
                        f"slot_mapping.shape={slot_mapping.shape} "
                        f"(numel={slot_mapping.numel()}), "
                        f"is_dummy={context.is_dummy}"
                    )
                if k_cache.dtype == torch.float8_e4m3fn:
                    from nanodeploy.backends.hopper.kernels.fp8_utils import (
                        store_kcache_fp8,
                    )

                    store_kcache_fp8(key_states_3d, k_cache, slot_mapping)
                else:
                    from nanodeploy.backends.gpu_generic.kernels.kv_store import (
                        store_kcache,
                    )

                    store_kcache(key_states_3d, k_cache, slot_mapping)

            # Store indexer keys during prefill (NSA V3.2 only)
            if (
                self.indexer is not None
                and self.indexer.indexer_cache is not None
                and not context.is_dummy
                and context.slot_mapping is not None
            ):
                self.indexer.store_prefill_keys(
                    hidden_states, positions, context.slot_mapping
                )

            # Weight matrices for K/V expansion (shared by both paths)
            kc_t = self.kc.weight.reshape(
                num_heads * self.qk_nope_head_dim, self.kv_lora_rank
            ).T  # (R, H*D)
            vc_reshaped = self.vc.weight.permute(1, 0, 2).reshape(
                self.kv_lora_rank, num_heads * self.v_head_dim
            )  # (R, H*V)

            # Expand fresh tokens (needed for both paths)
            k_nope_fresh = (compressed_kv @ kc_t).view(
                q_len, num_heads, self.qk_nope_head_dim
            )
            k_expanded_fresh = torch.cat(
                [k_nope_fresh, k_pe_3d.expand(-1, num_heads, -1)], dim=-1
            )
            v_expanded_fresh = (compressed_kv @ vc_reshaped).view(
                q_len, num_heads, self.v_head_dim
            )

            if context.block_tables is not None:

                sp_rank = get_dist_context().attn_sp_rank
                num_seqs = context.cu_seqlens_k.shape[0] - 1
                bt = context.block_tables[sp_rank, :num_seqs, :]
                block_size = k_cache.shape[1]

                k_cached_raw, cached_lens, cu_cached = _gather_cache_cached_only(
                    k_cache,
                    bt,
                    context.cu_seqlens_q,
                    context.cu_seqlens_k,
                    block_size,
                )

                total_cached = int(cu_cached[-1].item())
                if total_cached > 0:
                    k_cached_raw = k_cached_raw.squeeze(1)  # [total_cached, 576]
                    comp_cached = k_cached_raw[:, : self.kv_lora_rank]
                    kpe_cached = k_cached_raw[:, self.kv_lora_rank :]

                    k_nope_cached = (comp_cached @ kc_t).view(
                        -1, num_heads, self.qk_nope_head_dim
                    )
                    k_expanded_cached = torch.cat(
                        [
                            k_nope_cached,
                            kpe_cached.unsqueeze(1).expand(-1, num_heads, -1),
                        ],
                        dim=-1,
                    )
                    v_expanded_cached = (comp_cached @ vc_reshaped).view(
                        -1, num_heads, self.v_head_dim
                    )

                    k_expanded = _interleave_cached_fresh(
                        k_expanded_cached,
                        k_expanded_fresh,
                        cached_lens,
                        cu_cached,
                        context.cu_seqlens_q,
                        context.cu_seqlens_k,
                    )
                    v_expanded = _interleave_cached_fresh(
                        v_expanded_cached,
                        v_expanded_fresh,
                        cached_lens,
                        cu_cached,
                        context.cu_seqlens_q,
                        context.cu_seqlens_k,
                    )
                else:
                    k_expanded = k_expanded_fresh
                    v_expanded = v_expanded_fresh
            else:
                k_expanded = k_expanded_fresh
                v_expanded = v_expanded_fresh

            from flash_attn_interface import flash_attn_varlen_func

            attn_output = flash_attn_varlen_func(
                q_full,  # (q_len, H, 192)
                k_expanded,  # (total_k or q_len, H, 192)
                v_expanded,  # (total_k or q_len, H, 128)
                cu_seqlens_q=context.cu_seqlens_q,
                cu_seqlens_k=context.cu_seqlens_k,
                max_seqlen_q=context.max_seqlen_q,
                max_seqlen_k=context.max_seqlen_k,
                softmax_scale=self.softmax_scale,
                causal=True,
            )
            # attn_output: (T, H, v_head_dim=128) — no vc BMM needed
            attn_output = attn_output.reshape(q_len, -1)  # (T, H*128)
            attn_output = self.o_proj(attn_output)
            return attn_output

        else:
            # === Absorbed decode path ===
            context = get_context()
            # Q absorbed: q_nope @ W_UK -> (q_len, H, kv_lora_rank=512), concat with q_pe -> 576
            query_states, q_pe, q_lora = self._q_proj_absorbed(hidden_states, num_heads)

            key_states_3d = key_states.unsqueeze(1)  # (q_len, 1, 576)
            # Convert PE dims from interleaved to half format before RoPE
            q_pe = _interleaved_to_half(q_pe)
            k_pe_3d = k_pe.unsqueeze(1)  # (q_len, 1, rope_dim)
            k_pe_3d = _interleaved_to_half(k_pe_3d)
            q_pe, k_pe_3d = self.rotary_emb(positions, q_pe, k_pe_3d)
            query_states[..., self.kv_lora_rank :] = q_pe
            key_states_3d[..., self.kv_lora_rank :] = k_pe_3d

            # Run NSA Indexer (V3.2 only) — compute topk block indices
            sparse_indices = None
            if (
                self.indexer is not None
                and self.indexer.indexer_cache is not None
                and self.attn_fwd.k_cache.dtype == torch.float8_e4m3fn
            ):
                from nanodeploy.backends.hopper.layers.attention import (
                    topk_indices_to_physical,
                )
                from nanodeploy.context.distributed import get_dist_context

                sp_rank = get_dist_context().attn_sp_rank
                ntps = context.num_tokens_per_seq
                total_tokens = hidden_states.size(0)
                bs = total_tokens // ntps
                ctx_lens = context.context_lens[0, :bs]
                bt = context.block_tables[sp_rank, :bs]
                topk_indices = self.indexer(
                    hidden_states, q_lora, positions, ctx_lens, bt, context.slot_mapping
                )
                # Convert logical token indices → physical paged indices
                k_cache = self.attn_fwd.k_cache
                block_size = k_cache.shape[1]
                # topk_indices: (bs*ntps, topk), bt: (bs, max_blocks)
                # For ntps>1 (lazy verify), repeat bt per token
                if ntps > 1:
                    bt_expanded = bt.repeat_interleave(ntps, dim=0)
                else:
                    bt_expanded = bt
                sparse_indices = topk_indices_to_physical(
                    topk_indices, bt_expanded, block_size
                )

            # value_states for MLA decode: same compressed latent (unused by FlashMLA decode)
            value_states = compressed_kv.unsqueeze(1)  # (q_len, 1, 512)

            attn_output = self.attn_fwd(
                query_states,  # (q_len, H, 576)
                key_states_3d,  # (q_len, 1, 576)
                value_states,  # (q_len, 1, 512)
                sparse_indices=sparse_indices,
            )

            # Post-multiply by W_UV (vc BMM)
            attn_bmm_out = attn_output.new_empty(q_len, num_heads, self.v_head_dim)
            self.vc(attn_output, attn_bmm_out)
            attn_output = attn_bmm_out.reshape(q_len, -1)
            attn_output = self.o_proj(attn_output)
            return attn_output
