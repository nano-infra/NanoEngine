"""Qwen3.5 dense text model implementation for DLEngine.

Qwen3.5 dense shares the Qwen3.5 mixed-attention stack with Qwen3.5-MoE:
linear-attention layers use GatedDeltaNet, while full-attention layers use
GQA with partial RoPE and an attention output gate. The FFN is dense SwiGLU.
"""

from typing import Optional

import torch
from torch import nn

from dlengine.context_v2.cache.plan import qwen35_cache_plan
from dlengine.context_v2.distributed import get_dist_context
from dlengine.layers import get_backend
from dlengine.layers.base_backend import QKVParallelLinearBase, RowParallelLinearBase
from dlengine.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from dlengine.layers.layernorm import RMSNorm
from dlengine.layers.rotary_embedding import get_rope
from dlengine.models.pp_utils import (
    get_pp_layer_range,
    make_pp_layers,
    pp_recv_hidden,
    pp_send_hidden,
)
from dlengine.models.quant_config import QuantizationConfig
from dlengine.models.qwen3_5_moe.qwen3_5_moe import Qwen3_5MoeMLP


class Qwen3_5FullAttention(nn.Module):
    """Full attention with partial RoPE and output gating."""

    def __init__(
        self,
        layer_idx: int,
        config,
        quantization_config: QuantizationConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.quantization_config = quantization_config
        self.layer_idx = layer_idx

        tp_size = get_dist_context().attn_tp_world_size
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.attn_output_gate = getattr(config, "attn_output_gate", True)

        rope_params = getattr(config, "rope_parameters", {}) or {}
        self.partial_rotary_factor = rope_params.get("partial_rotary_factor", 1.0)
        self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        rope_theta = rope_params.get("rope_theta", 10000000.0)

        q_heads_for_proj = self.total_num_heads * (1 + int(self.attn_output_gate))
        self.qkv_proj: QKVParallelLinearBase = get_backend().get_qkv_parallel_linear(
            config.hidden_size,
            self.head_dim,
            q_heads_for_proj,
            self.total_num_kv_heads,
            bias=getattr(config, "attention_bias", False),
            tp_group=get_dist_context().attn_tp_group,
        )

        self.o_proj: RowParallelLinearBase = get_backend().get_row_parallel_linear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=getattr(config, "attention_bias", False),
            tp_group=get_dist_context().attn_tp_group,
        )

        self.rotary_emb = get_rope(
            self.rotary_dim,
            rotary_dim=self.rotary_dim,
            max_position=getattr(config, "max_position_embeddings", 262144),
            base=rope_theta,
            rope_scaling=None,
        )

        self.attn = get_backend().get_attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            self.head_dim,
            "GQA",
        )

        self.q_norm = RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, add_unit_offset=True
        )
        self.k_norm = RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, add_unit_offset=True
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)

        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            q_gate = q_gate.view(-1, self.num_heads, self.head_dim * 2)
            q, gate = q_gate.chunk(2, dim=-1)
            gate = gate.reshape(-1, self.q_size)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q = q.view(-1, self.num_heads, self.head_dim)
            gate = None

        q = self.q_norm(q.contiguous())
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim))
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        if self.rotary_dim < self.head_dim:
            q_rot = q[..., : self.rotary_dim].contiguous()
            q_pass = q[..., self.rotary_dim :]
            k_rot = k[..., : self.rotary_dim].contiguous()
            k_pass = k[..., self.rotary_dim :]
            q_rot, k_rot = self.rotary_emb(positions, q_rot, k_rot)
            q = torch.cat([q_rot, q_pass], dim=-1)
            k = torch.cat([k_rot, k_pass], dim=-1)
        else:
            q, k = self.rotary_emb(positions, q, k)

        o = self.attn(q, k, v)
        attn_output = o.flatten(1, -1)
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)

        return self.o_proj(attn_output)


class Qwen3_5DecoderLayer(nn.Module):
    """Decoder layer with Qwen3.5 mixed attention and dense MLP."""

    def __init__(
        self,
        config,
        quantization_config: QuantizationConfig,
        layer_idx: int = -1,
        state_layer_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        layer_types = getattr(config, "layer_types", [])
        if layer_idx < len(layer_types):
            self.layer_type = layer_types[layer_idx]
        else:
            interval = getattr(config, "full_attention_interval", 1)
            self.layer_type = (
                "full_attention"
                if interval > 0 and (layer_idx + 1) % interval == 0
                else "linear_attention"
            )

        if layer_idx in getattr(config, "mlp_only_layers", []):
            self.layer_type = "mlp_only"

        if self.layer_type == "full_attention":
            self.self_attn = Qwen3_5FullAttention(
                layer_idx=layer_idx,
                config=config,
                quantization_config=quantization_config,
            )
        elif self.layer_type == "linear_attention":
            self.linear_attn = get_backend().get_gated_delta_net(
                layer_idx=layer_idx if state_layer_idx is None else state_layer_idx,
                config=config,
                quantization_config=quantization_config,
            )
        elif self.layer_type != "mlp_only":
            raise ValueError(f"Unsupported Qwen3.5 layer type: {self.layer_type}")

        self.mlp = Qwen3_5MoeMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quantization_config=quantization_config,
        )

        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, add_unit_offset=True
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, add_unit_offset=True
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self.layer_type == "full_attention":
            hidden_states = self.self_attn(positions, hidden_states)
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
        elif self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states)
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
        elif self.layer_type == "mlp_only":
            hidden_states = self.post_attention_layernorm(residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3_5Model(nn.Module):
    """Qwen3.5 dense text backbone."""

    def __init__(self, config, quantization_config: QuantizationConfig) -> None:
        super().__init__()
        ctx = get_dist_context()
        self.is_first_pp_stage = ctx.is_first_pp_stage
        self.is_last_pp_stage = ctx.is_last_pp_stage
        self.hidden_size = config.hidden_size
        self.hidden_dtype = getattr(config, "dtype", None) or torch.get_default_dtype()

        if self.is_first_pp_stage:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size, config.hidden_size
            )
        else:
            self.embed_tokens = None

        pp_start, _ = get_pp_layer_range(config.num_hidden_layers)
        self.start_layer, self.end_layer, self.layers = make_pp_layers(
            config.num_hidden_layers,
            lambda layer_idx: Qwen3_5DecoderLayer(
                config,
                quantization_config,
                layer_idx,
                state_layer_idx=layer_idx - pp_start,
            ),
        )

        if self.is_last_pp_stage:
            self.norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps, add_unit_offset=True
            )
        else:
            self.norm = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.is_first_pp_stage:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            hidden_states = pp_recv_hidden(
                positions.size(0), self.hidden_size, self.hidden_dtype
            )
            residual = None

        for idx in range(self.start_layer, self.end_layer):
            hidden_states, residual = self.layers[idx](
                positions, hidden_states, residual
            )

        if not self.is_last_pp_stage:
            if residual is not None:
                hidden_states = hidden_states + residual
            pp_send_hidden(hidden_states)
            return hidden_states

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3_5ForConditionalGeneration(nn.Module):
    """Qwen3.5 dense language model."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        quantization_config = QuantizationConfig(
            **getattr(config, "quantization_config", dict())
        )
        self.quantization_config = quantization_config

        self.model = Qwen3_5Model(config, quantization_config)
        if get_dist_context().is_last_pp_stage:
            self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
            if (
                getattr(config, "tie_word_embeddings", False)
                and self.model.embed_tokens is not None
            ):
                self.lm_head.weight.data = self.model.embed_tokens.weight.data
        else:
            self.lm_head = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds=inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def get_cache_plan(self):
        return qwen35_cache_plan()

    def load_weights(self, weights):
        """Load weights using per-model loader."""
        from .qwen3_5_loader import load_weights

        load_weights(self, weights)
