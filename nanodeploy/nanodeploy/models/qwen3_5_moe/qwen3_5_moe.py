"""Qwen3.5-MoE model implementation for NanoDeploy.

Supports mixed attention (full_attention GQA + GatedDeltaNet linear_attention)
and sparse MoE with shared expert.

Key design decisions (Phase 1):
  - attention_tp = 1 (no tensor parallelism for now)
  - flash-linear-attention kernels for GatedDeltaNet
  - Fixed-size state buffers for GDN conv & recurrent states
  - Text-only (no vision module)
"""

import logging
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from nanodeploy.backends import get_backend
from nanodeploy.backends.base_backend import (
    ColumnParallelLinearBase,
    DistributedRoutedExpertsBase,
    MergedColumnParallelLinearBase,
    QKVParallelLinearBase,
    ReplicatedLinearBase,
    RowParallelLinearBase,
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


# ---------------------------------------------------------------------------
# Full Attention (GQA with partial RoPE + attention output gate)
# ---------------------------------------------------------------------------
class Qwen3_5MoeFullAttention(nn.Module):
    """Full attention with partial RoPE and output gating.

    Used for ~25% of layers (every 4th layer).
    The q_proj output is doubled: [q, gate] interleaved per-head.
    attn_output = o_proj(attn(q, k, v) * sigmoid(gate))
    """

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
        self.total_num_heads = config.num_attention_heads  # 32
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size  # 32 for tp=1
        self.total_num_kv_heads = config.num_key_value_heads  # 2
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size  # 2 for tp=1
        self.head_dim = config.head_dim  # 256
        self.q_size = self.num_heads * self.head_dim  # 8192
        self.kv_size = self.num_kv_heads * self.head_dim  # 512
        self.scaling = self.head_dim**-0.5

        # Attention output gate: q_proj outputs 2x (interleaved q and gate per-head)
        self.attn_output_gate = getattr(config, "attn_output_gate", True)

        # Partial rotary factor
        rope_params = getattr(config, "rope_parameters", {}) or {}
        self.partial_rotary_factor = rope_params.get("partial_rotary_factor", 1.0)
        self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)  # 64
        rope_theta = rope_params.get("rope_theta", 10000000.0)

        # QKV projection (packed)
        # When attn_output_gate=True, q_proj output is doubled (includes gate)
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

        # RoPE: use rotary_dim as head_size for the rotary embedding
        self.rotary_emb = get_rope(
            self.rotary_dim,
            rotary_dim=self.rotary_dim,
            max_position=getattr(config, "max_position_embeddings", 262144),
            base=rope_theta,
            rope_scaling=None,  # rope_type='default' means no special scaling
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
            # q_gate portion: q_size * 2
            # k portion: kv_size
            # v portion: kv_size
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            # Split q and gate: interleaved per-head [num_heads, head_dim*2]
            q_gate = q_gate.view(-1, self.num_heads, self.head_dim * 2)
            q, gate = q_gate.chunk(2, dim=-1)  # each [-1, num_heads, head_dim]
            gate = gate.reshape(-1, self.q_size)  # [-1, q_size]
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q = q.view(-1, self.num_heads, self.head_dim)
            gate = None

        q = self.q_norm(q.contiguous())
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim))
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        # Partial RoPE: apply to first rotary_dim, pass through rest
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

        # Apply output gate
        attn_output = o.flatten(1, -1)  # [-1, q_size]
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)

        output = self.o_proj(attn_output)
        return output


# ---------------------------------------------------------------------------
# MoE MLP (single expert)
# ---------------------------------------------------------------------------
class Qwen3_5MoeMLP(nn.Module):
    """Single expert MLP (SwiGLU)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        gate_up_proj_tensor=None,
        down_proj_tensor=None,
        gate_up_scale_inv_tensor=None,
        down_scale_inv_tensor=None,
        meta: bool = False,
        quantization_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
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
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


# ---------------------------------------------------------------------------
# Sparse MoE Block with shared expert
# ---------------------------------------------------------------------------
class Qwen3_5MoeSparseMoeBlock(nn.Module):
    """Sparse MoE block with TopK routing and shared expert."""

    def __init__(self, config, quantization_config: QuantizationConfig) -> None:
        super().__init__()
        self.config = config
        self.quantization_config = quantization_config

        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size  # 1024
        self.num_experts = config.num_experts  # 512
        self.top_k = config.num_experts_per_tok  # 10

        # Router
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        weight_dtype = quantization_config.dtype or config.dtype

        self.tp_size = get_dist_context().ffn_tp_world_size
        self.tp_group = get_dist_context().ffn_tp_group

        # EP setup
        self.ep_group = get_dist_context().ffn_ep_group
        self.ep_size = get_dist_context().ffn_ep_world_size

        self.routed_experts: (
            DistributedRoutedExpertsBase
        ) = get_backend().get_distributed_routed_experts(
            hidden_size=config.hidden_size,
            intermediate_size=self.moe_intermediate_size,
            num_experts=self.num_experts,
            top_k=self.top_k,
            ep_size=self.ep_size,
            tp_size=self.tp_size,
            ep_group=self.ep_group,
            tp_group=self.tp_group,
            n_group=getattr(config, "n_group", 1),
            topk_group=getattr(config, "topk_group", 1),
            norm_topk_prob=getattr(config, "norm_topk_prob", True),
            routed_scaling_factor=getattr(config, "routed_scaling_factor", 1.0),
            scoring_func="softmax",
        )

        # Shared expert
        shared_intermediate = getattr(
            config, "shared_expert_intermediate_size", self.moe_intermediate_size
        )
        self.shared_expert = Qwen3_5MoeMLP(
            hidden_size=config.hidden_size,
            intermediate_size=shared_intermediate,
            hidden_act=config.hidden_act,
            quantization_config=quantization_config,
        )

        # Shared expert gate (sigmoid, scalar per token)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

        self.act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor):
        orig_shape = hidden_states.shape
        hidden_dim = orig_shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)
        num_tokens = hidden_states.shape[0]

        # Shared expert forward
        shared_out = self.shared_expert(hidden_states)
        shared_gate = torch.sigmoid(self.shared_expert_gate(hidden_states))
        shared_out = shared_out * shared_gate

        router_logits = self.gate(hidden_states)

        # Softmax and routing
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        context = get_context()
        is_prefill = context.is_prefill

        final_hidden_states = self.routed_experts(
            hidden_states, selected_experts, routing_weights, is_prefill=is_prefill
        )

        final_hidden_states = final_hidden_states + shared_out
        return final_hidden_states.view(orig_shape)


# ---------------------------------------------------------------------------
# Decoder Layer (mixed attention type)
# ---------------------------------------------------------------------------
class Qwen3_5MoeDecoderLayer(nn.Module):
    """Decoder layer supporting both full_attention and linear_attention."""

    def __init__(
        self,
        config,
        quantization_config: QuantizationConfig,
        layer_idx: int = -1,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        # Determine attention type for this layer
        layer_types = getattr(config, "layer_types", [])
        if layer_idx < len(layer_types):
            self.layer_type = layer_types[layer_idx]
        else:
            self.layer_type = "full_attention"

        if self.layer_type == "full_attention":
            self.self_attn = Qwen3_5MoeFullAttention(
                layer_idx=layer_idx,
                config=config,
                quantization_config=quantization_config,
            )
        else:  # linear_attention
            self.linear_attn = get_backend().get_gated_delta_net(
                layer_idx=layer_idx,
                config=config,
                quantization_config=quantization_config,
            )

        # MoE MLP (all layers)
        self.mlp = Qwen3_5MoeSparseMoeBlock(
            config=config, quantization_config=quantization_config
        )

        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, add_unit_offset=True
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, add_unit_offset=True
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
        else:
            hidden_states = self.linear_attn(hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        hidden_states = self.attn_to_ffn(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.ffn_to_attn(hidden_states)

        return hidden_states, residual


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class Qwen3_5MoeModel(nn.Module):
    """Qwen3.5-MoE text model backbone."""

    def __init__(self, config, quantization_config: QuantizationConfig) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                Qwen3_5MoeDecoderLayer(config, quantization_config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, add_unit_offset=True
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        residual = None
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(positions, hidden_states, residual)

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


# ---------------------------------------------------------------------------
# Top-level ForConditionalGeneration (text-only)
# ---------------------------------------------------------------------------
class Qwen3_5MoeForConditionalGeneration(nn.Module):
    """Qwen3.5-MoE for conditional generation.

    Checkpoint weight prefix: model.language_model.* → model.*
    The loader strips `language_model.` before parameter lookup.
    """

    def __init__(self, config) -> None:
        super().__init__()
        # The config might have text_config nested (VLM) or be flat
        # After config.py flattening, all text_config attrs are on config
        self.config = config

        quantization_config = QuantizationConfig(
            **getattr(config, "quantization_config", dict())
        )
        self.quantization_config = quantization_config

        self.model = Qwen3_5MoeModel(config, quantization_config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, inputs_embeds=inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits

    def load_weights(self, weights):
        """Load weights using per-model loader."""
        from .qwen3_5_moe_loader import load_weights

        load_weights(self, weights)
