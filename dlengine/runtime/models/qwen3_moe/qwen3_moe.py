import logging
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3MoeConfig

from dlengine.logging import get_logger
from dlengine.runtime.context.batch import get_batch_context
from dlengine.runtime.context.distributed import get_dist_context
from dlengine.runtime.layers import get_backend
from dlengine.runtime.layers.activation import SiluAndMul
from dlengine.runtime.layers.base_backend import (
    DistributedRoutedExpertsBase,
    MergedColumnParallelLinearBase,
    PrequantizedActivation,
    QKVParallelLinearBase,
    RowParallelLinearBase,
)
from dlengine.runtime.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from dlengine.runtime.layers.layernorm import RMSNorm
from dlengine.runtime.layers.parallelism_transition import (
    AttnToFfnTransition,
    FfnToAttnTransition,
)
from dlengine.runtime.layers.rotary_embedding import get_rope
from dlengine.runtime.models.pp_utils import (
    make_pp_layers,
    pp_recv_hidden,
    pp_send_hidden,
)
from dlengine.runtime.runner.runner_config import get_runner_config
from ..quant_config import QuantizationConfig

try:
    from dlengine.runtime.kernel.triton.generic.fused_topk import (
        can_use_fused_softmax_topk,
        fused_softmax_topk,
    )
except ImportError:
    can_use_fused_softmax_topk = None
    fused_softmax_topk = None

try:
    from dlengine.runtime.kernel.triton.hopper.rmsnorm_quant_fp8 import (
        add_rms_norm_quant_fp8,
        can_use_add_rms_norm_quant_fp8,
    )
except ImportError:
    add_rms_norm_quant_fp8 = None
    can_use_add_rms_norm_quant_fp8 = None

logger = get_logger()


class Qwen3MoeAttention(nn.Module):

    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
        config: Qwen3MoeConfig | None = None,
        quantization_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.quantization_config = quantization_config

        self.layer_idx = layer_idx

        tp_size = get_dist_context().attn_tp_world_size
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj: QKVParallelLinearBase = get_backend().get_qkv_parallel_linear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            tp_group=get_dist_context().attn_tp_group,
        )

        self.o_proj: RowParallelLinearBase = get_backend().get_row_parallel_linear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            tp_group=get_dist_context().attn_tp_group,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        self.attn = get_backend().get_attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            self.head_dim,
            "GQA",
        )

        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim))
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim))
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        q, k = self.rotary_emb(positions, q, k)

        o = self.attn(q, k, v)

        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MoeMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        gate_up_proj_tensor: torch.Tensor | None = None,
        down_proj_tenosr: torch.Tensor | None = None,
        gate_up_scale_inv_tensor: torch.Tensor | None = None,
        down_scale_inv_tensor: torch.Tensor | None = None,
        meta: bool = False,
        config: Qwen3MoeConfig | None = None,
        quantization_config: QuantizationConfig | None = None,
    ) -> None:
        # by now, all FFN layers are SparseMLP
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
            weight_tensor=down_proj_tenosr,
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


class Qwen3MoeSparseMoeBlock(nn.Module):

    def __init__(
        self, config: Qwen3MoeConfig, quantization_config: QuantizationConfig
    ) -> None:
        super().__init__()

        self.config = config
        self.quantization_config = quantization_config

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.hidden_act = config.hidden_act

        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        # gating
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
            intermediate_size=config.moe_intermediate_size,
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

        self.act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor):
        orig_shape = hidden_states.shape
        hidden_dim = orig_shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)
        num_tokens = hidden_states.shape[0]

        router_logits = self.gate(hidden_states)

        # Softmax and routing
        if fused_softmax_topk is not None and can_use_fused_softmax_topk(
            router_logits, self.top_k
        ):
            # Single fused kernel; emits fp32 weights + int64 ids so the
            # DeepEP dispatcher-side casts become no-ops.
            routing_weights, selected_experts = fused_softmax_topk(
                router_logits, self.top_k, renormalize=True
            )
        else:
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(
                routing_weights, self.top_k, dim=-1
            )
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            routing_weights = routing_weights.to(hidden_states.dtype)

        context = get_batch_context()
        is_prefill = context.is_prefill

        # Workaround for Qwen3 MoE FP8 scales: DeepGEMM expects 3D scale tensors
        # (num_groups, channels, 1) when grouped FP8 GEMM is used.
        if getattr(self.routed_experts, "is_fp8", False):
            if (
                self.routed_experts.gate_up_scale_inv is not None
                and self.routed_experts.gate_up_scale_inv.dim() == 2
            ):
                self.routed_experts.gate_up_scale_inv.data = (
                    self.routed_experts.gate_up_scale_inv.data.unsqueeze(-1)
                )
            if (
                self.routed_experts.down_scale_inv is not None
                and self.routed_experts.down_scale_inv.dim() == 2
            ):
                self.routed_experts.down_scale_inv.data = (
                    self.routed_experts.down_scale_inv.data.unsqueeze(-1)
                )

        final_hidden_states = self.routed_experts(
            hidden_states, selected_experts, routing_weights, is_prefill=is_prefill
        )

        return final_hidden_states.view(orig_shape)


class Qwen3MoeDecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3MoeConfig,
        quantization_config: QuantizationConfig,
        layer_idx: int = -1,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3MoeAttention(
            layer_idx=layer_idx,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
            config=config,
            quantization_config=quantization_config,
        )
        mlp_only_layers = getattr(config, "mlp_only_layers", [])
        if (layer_idx not in mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3MoeSparseMoeBlock(
                config=config, quantization_config=quantization_config
            )
        else:
            self.mlp = Qwen3MoeMLP(
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
        self.layer_idx = layer_idx

        # Fuse input_layernorm + FP8 activation quant into one kernel and
        # hand the prequantized activation straight to qkv_proj's DeepGEMM
        # (skips one kernel launch + a bf16 HBM round-trip per layer).
        # Only valid on the Hopper FP8 path with 128-blocked scales.
        self._fuse_norm_quant_qkv = (
            add_rms_norm_quant_fp8 is not None
            and quantization_config.quant_method == "fp8"
            and list(quantization_config.block_size or [])[:1] == [128]
            and can_use_add_rms_norm_quant_fp8(config.hidden_size)
            and hasattr(self.self_attn.qkv_proj, "_fp8_forward")
            and not self.input_layernorm.add_unit_offset
        )
        self._norm_quant_ue8m0 = (
            getattr(quantization_config, "scale_fmt", None) == "ue8m0"
        )

        # Parallelism transition: when attn uses TP and FFN uses EP,
        # we must redistribute tokens between the two phases.
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
        elif (
            self._fuse_norm_quant_qkv
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
            and residual.dtype == torch.bfloat16
            and residual.is_contiguous()
            and residual.shape == hidden_states.shape
        ):
            q, scales, residual = add_rms_norm_quant_fp8(
                hidden_states,
                residual,
                self.input_layernorm.weight,
                self.input_layernorm.eps,
                dtype=self.self_attn.qkv_proj.weight.dtype,
                round_ue8m0=self._norm_quant_ue8m0,
            )
            hidden_states = PrequantizedActivation(q, scales, hidden_states.shape[0])
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        hidden_states = self.attn_to_ffn(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.ffn_to_attn(hidden_states)

        return hidden_states, residual


class Qwen3MoeModel(nn.Module):

    def __init__(
        self, config: Qwen3MoeConfig, quantization_config: QuantizationConfig
    ) -> None:
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

        self.start_layer, self.end_layer, self.layers = make_pp_layers(
            config.num_hidden_layers,
            lambda layer_idx: Qwen3MoeDecoderLayer(
                config, quantization_config, layer_idx
            ),
        )

        if self.is_last_pp_stage:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
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


class Qwen3MoeForCausalLM(nn.Module):

    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.config = config
        quantization_config = QuantizationConfig(
            **getattr(config, "quantization_config", dict())
        )
        self.model = Qwen3MoeModel(config, quantization_config)
        if get_dist_context().is_last_pp_stage:
            self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
            if config.tie_word_embeddings and self.model.embed_tokens is not None:
                self.lm_head.weight.data = self.model.embed_tokens.weight.data
        else:
            self.lm_head = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
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
        from .qwen3_moe_loader import load_weights

        load_weights(self, weights)
