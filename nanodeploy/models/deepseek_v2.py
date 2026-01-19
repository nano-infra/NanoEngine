import math
from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from dlblas.layers.moe.ep_moe import build_deepep_moe
from lmdeploy.pytorch.nn import build_rotary_embedding, RopeType
from lmdeploy.pytorch.nn.rotary_embedding import YarnParameters
from nanodeploy.layers.activation import SiluAndMul
from nanodeploy.layers.attention import Attention
from nanodeploy.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanodeploy.layers.layernorm import RMSNorm
from nanodeploy.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from nanodeploy.layers.rotary_embedding import get_rope
from nanodeploy.logging import get_logger
from nanodeploy.worker.context import get_context
from nanodeploy.worker.distributed import get_dist_context
from nanodeploy.worker.runner_config import get_runner_config
from torch import nn
from transformers import DeepseekV3Config

from .quant_config import QuantizationConfig

logger = get_logger()


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def compute_topk_ids(topk_ids, ranks, num_experts):
    """Optimized version: compute expert IDs for perfect load balancing.
    
    This function redistributes expert IDs to ensure perfect load balancing
    across expert parallel ranks. Optimized to use a single torch.arange call.
    """
    shape = topk_ids.shape
    numel = topk_ids.numel()
    step = num_experts // ranks
    
    # Single arange call instead of two
    indices = torch.arange(
        0, numel, dtype=topk_ids.dtype, device=topk_ids.device
    )
    
    # Compute both components from the same indices
    div_ranks = indices // ranks
    mod_ranks = indices % ranks
    
    # Compute the remapped expert IDs
    topk_ids = (div_ranks % step + mod_ranks * step) % num_experts
    topk_ids = topk_ids.reshape(shape)
    return topk_ids


# 已改


class DeepseekV2MoE(nn.Module):
    """Deepseek v2 MoE."""

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
        self.distribution = "uniform"

        # Use optimized Linear layer for gate
        # For gate, we don't need quantization, so use standard Linear
        # but we can optimize it by using F.linear directly in forward
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        weight_dtype = quantization_config.dtype or config.dtype

        # global parameter for DeepGEMM

        self.gate_up_proj = nn.Parameter(
            torch.ones(
                self.num_experts_per_rank,
                config.moe_intermediate_size * 2,
                config.hidden_size,
                dtype=weight_dtype,
                device="cuda",
            ),
        )

        self.down_proj = nn.Parameter(
            torch.ones(
                self.num_experts_per_rank,
                config.hidden_size,
                config.moe_intermediate_size,
                dtype=weight_dtype,
                device="cuda",
            )
        )

        if quantization_config.quant_method == "fp8":
            self.gate_up_scale_inv = nn.Parameter(
                torch.ones(
                    self.num_experts_per_rank,
                    config.moe_intermediate_size
                    * 2
                    // quantization_config.block_size[0],
                    config.hidden_size // quantization_config.block_size[1],
                    dtype=torch.float32,
                    device="cuda",
                )
            )

            self.down_scale_inv = (
                nn.Parameter(
                    torch.ones(
                        self.num_experts_per_rank,
                        config.hidden_size // quantization_config.block_size[0],
                        config.moe_intermediate_size
                        // quantization_config.block_size[1],
                        dtype=torch.float32,
                        device="cuda",
                    )
                )
                if quantization_config.quant_method == "fp8"
                else None
            )
        self.experts = nn.ModuleList(
            [
                DeepseekV2MLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.moe_intermediate_size,
                    hidden_act=config.hidden_act,
                    meta=True,
                    config=config,
                    quantization_config=quantization_config,
                )
                for i in range(self.num_experts)
            ]
        )

        self.ep_group = get_dist_context().ffn_ep_group
        self.ep_size = get_dist_context().ffn_ep_world_size

        if self.ep_size > 1:
            self.moe = build_deepep_moe(
                low_latency_mode=True,
                ep_size=self.ep_size,
                ep_group=self.ep_group,
                num_experts=self.num_experts,
                hidden_dim=self.hidden_size,
                block_size=self.quantization_config.block_size[0],
                top_k=self.top_k,
                out_dtype=torch.bfloat16,
                layer_idx=0,
                chunk_size=16 * 1024,
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

    def fusedmoe_build(self, low_latency_mode):
        fusedmoe = build_deepep_moe(
            low_latency_mode=low_latency_mode,
            ep_size=self.ep_size,
            ep_group=self.ep_group,
            num_experts=self.num_experts,
            hidden_dim=self.hidden_size,
            block_size=self.quantization_config.block_size[0],
            top_k=self.top_k,
            out_dtype=torch.bfloat16,
            layer_idx=0,
            chunk_size=16 * 1024,
        )
        return fusedmoe

    @property
    def expert_list_this_rank(self):
        ep_group = get_dist_context().ffn_ep_group
        ep_rank = dist.get_rank(group=ep_group)

        expert_id_begin = ep_rank * self.num_experts_per_rank
        expert_id_end = (ep_rank + 1) * self.num_experts_per_rank

        return list(range(expert_id_begin, expert_id_end))

    def forward(self, hidden_states: torch.Tensor):
        """forward."""
        if self.ep_size > 1:
            assert (
                self.quantization_config.quant_method == "fp8"
            ), "Only FP8 EP is supported by now"
            batch_size, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_dim)

            context = get_context()
            moe = self.fusedmoe_build(not context.is_prefill)
            
            # Optimized gate computation: use F.linear for better performance
            # F.linear is more efficient than nn.Linear forward for inference
            router_logits = F.linear(hidden_states, self.gate.weight, None)

            # Optimized softmax: use torch.softmax instead of F.softmax with dtype conversion
            # This avoids unnecessary type conversion and is more efficient
            # torch.softmax automatically handles numerical stability
            routing_weights = torch.softmax(router_logits, dim=-1)
            
            # Optimized topk: use sorted=False for better performance when order doesn't matter
            # In decode phase, we typically don't need sorted results
            sorted_topk = context.is_prefill if hasattr(context, 'is_prefill') else True
            routing_weights, selected_experts = torch.topk(
                routing_weights, self.top_k, dim=-1, sorted=sorted_topk
            )

            if self.distribution == "uniform":
                # Uniform random sampling
                selected_experts = torch.randint(
                    low=0,
                    high=self.num_experts,
                    size=(hidden_states.shape[0], self.top_k),
                    dtype=selected_experts.dtype,
                    device=hidden_states.device,
                )
            elif get_runner_config().perfect_eplb:
                ep_size = get_dist_context().ffn_ep_world_size
                selected_experts = compute_topk_ids(
                    selected_experts, ep_size, self.num_experts
                )
            final_hidden_states = moe.forward(
                hidden_states,
                routing_weights,
                selected_experts,
                self.gate_up_proj,
                self.gate_up_scale_inv,
                self.down_proj,
                self.down_scale_inv,
                expert_list=self.expert_list_this_rank,
            )
        if self.shared_experts is not None:
            shared_states = self.shared_experts(final_hidden_states)
            final_hidden_states += shared_states
        final_hidden_states = final_hidden_states.reshape(batch_size, -1)

        return final_hidden_states


# 已改


class DeepseekV2MLP(nn.Module):
    """Deepseek v2 mlp."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int = None,
        hidden_act: str = "silu",
        meta: bool = False,
        config: DeepseekV3Config | None = None,
        quantization_config: QuantizationConfig | None = None,
    ):
        super().__init__()

        self.config = config
        self.quantization_config = quantization_config

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            meta=meta,
            quantization_config=quantization_config,
        )

        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            meta=meta,
            quantization_config=quantization_config,
        )

        # silu and mul

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
        self.self_attn = DeepseekV2Attention(config, quantization_config)

        # mlp

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
        # build input layer norm

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # build attention layer norm

        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

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
        hidden_states = self.mlp(hidden_states)

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


# 已改


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
        num_key_value_heads = getattr(config, "num_key_value_heads", 1)

        if self.q_lora_rank is None:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.q_head_dim,
                quantization_config=quantization_config,
            )
            self.kv_a_proj_with_mqa = ColumnParallelLinear(
                self.hidden_size,
                config.kv_lora_rank + config.qk_rope_head_dim,
                bias=config.attention_bias,
                quantization_config=quantization_config,
            )
            self.kv_a_layernorm = RMSNorm(
                config.kv_lora_rank,
                1e-6,
            )
        else:
            # Fused QKV projection: fuse q_a_proj and kv_a_proj_with_mqa
            # This reduces one GEMM call and improves memory locality
            self.fused_qkv_a_proj = MergedColumnParallelLinear(
                self.hidden_size,
                [config.q_lora_rank, config.kv_lora_rank + config.qk_rope_head_dim],
                bias=False,
                meta=False,
                quantization_config=quantization_config,
            )
            self.q_a_layernorm = RMSNorm(hidden_size=config.q_lora_rank, eps=1e-6)
            self.q_b_proj = ColumnParallelLinear(
                config.q_lora_rank,
                self.num_heads * self.q_head_dim,
                bias=False,
                quantization_config=quantization_config,
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

        emb_type = RopeType.LinearScaling
        rope_dim = (
            config.qk_rope_head_dim
            if getattr(config, "use_mla", True)
            else (config.hidden_size // config.num_attention_heads)
        )
        rope_max_pos_emb = config.max_position_embeddings
        rope_base = config.rope_theta
        scaling_factor = 1.0
        other_params = dict()
        if config.rope_scaling is not None:
            scaling_type = config.rope_scaling["type"]
            scaling_factor = config.rope_scaling["factor"]
            if scaling_type == "dynamic":
                emb_type = RopeType.DynamicNTKScaling
            elif scaling_type == "yarn":
                emb_type = RopeType.Yarn
                rope_max_pos_emb = config.rope_scaling.get(
                    "original_max_position_embeddings", 4096
                )
                kwargs = {
                    key: config.rope_scaling[key]
                    for key in [
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if key in config.rope_scaling
                }
                yarn_params = YarnParameters(**kwargs)
                other_params["yarn_params"] = yarn_params
        self.rotary_emb = get_rope(
            config.head_dim,
            rotary_dim=config.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            # rope_scaling=config.rope_scaling,
        )

        self.softmax_scale = self.q_head_dim ** (-0.5)

        if config.rope_scaling is not None:
            mscale_all_dim = config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale
        self.attn_fwd = Attention(
            self.num_heads,
            config.kv_lora_rank + self.qk_rope_head_dim,
            scale=self.softmax_scale,
            num_kv_heads=num_key_value_heads,
            v_head_dim=config.kv_lora_rank,
            attention_type="MLA",
        )

        self.vc = DeepseekV2BMM(self.num_heads, config.kv_lora_rank, self.v_head_dim)

        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
            quantization_config=quantization_config,
        )

    def _q_proj(self, hidden_states, num_heads: int, nope_size: int, pe_size: int):
        """Q proj."""
        q_len = hidden_states.size(0)

        query_states = hidden_states.new_empty([q_len, num_heads, nope_size + pe_size])

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            # This path should not be called when using fused projection
            # Use _q_proj_from_fused instead
            fused_output = self.fused_qkv_a_proj(hidden_states)
            q_a = fused_output[..., :self.q_lora_rank]
            q = self.q_b_proj(self.q_a_layernorm(q_a))
        q = q.view(q_len, num_heads, self.q_head_dim)
        # q_pe: (q_len, num_heads, qk_rope_head_dim)

        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        # q_nope: (q_len, num_heads, kv_lora_rank)

        q_nope_out = query_states[..., :nope_size]
        self.kc(q_nope, q_nope_out)
        return query_states, q_pe

    def _q_proj_from_fused(self, q_a, num_heads: int, nope_size: int, pe_size: int):
        """Q proj from pre-computed fused output."""
        q_len = q_a.size(0)
        query_states = q_a.new_empty([q_len, num_heads, nope_size + pe_size])
        
        q = self.q_b_proj(self.q_a_layernorm(q_a))
        q = q.view(q_len, num_heads, self.q_head_dim)
        
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        
        q_nope_out = query_states[..., :nope_size]
        self.kc(q_nope, q_nope_out)
        return query_states, q_pe

    def _kv_proj(self, hidden_states, nope_size: int):
        """Kv proj."""
        # Original implementation: separate kv_a_proj_with_mqa
        key_states = self.kv_a_proj_with_mqa(hidden_states)
        k_pe = key_states[..., nope_size:]
        value_states = key_states[..., :nope_size]
        value_states = self.kv_a_layernorm(value_states)
        key_states[..., :nope_size] = value_states
        return key_states, value_states, k_pe

    def _kv_proj_from_fused(self, kv_a_full, nope_size: int):
        """Kv proj from pre-computed fused output."""
        # Extract kv_a and k_pe from fused output
        kv_a, k_pe = kv_a_full.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        value_states = self.kv_a_layernorm(kv_a)
        key_states = torch.cat([value_states, k_pe], dim=-1)
        return key_states, value_states, k_pe

    def _qkv_proj(self, hidden_states: torch.Tensor, num_heads: int):
        """Qkv proj."""
        nope_size = self.kv_lora_rank
        pe_size = self.qk_rope_head_dim
        
        # Optimize: compute fused_qkv_a_proj once if using fused projection
        # This reduces one GEMM call compared to separate q_a_proj and kv_a_proj_with_mqa
        if self.q_lora_rank is not None:
            fused_output = self.fused_qkv_a_proj(hidden_states)
            q_a = fused_output[..., :self.q_lora_rank]
            kv_a_full = fused_output[..., self.q_lora_rank:]
            query_states, q_pe = self._q_proj_from_fused(q_a, num_heads, nope_size, pe_size)
            key_states, value_states, k_pe = self._kv_proj_from_fused(kv_a_full, nope_size)
        else:
            query_states, q_pe = self._q_proj(hidden_states, num_heads, nope_size, pe_size)
            key_states, value_states, k_pe = self._kv_proj(hidden_states, nope_size)

        return query_states, key_states, value_states, q_pe, k_pe

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ):
        """Rewrite of LlamaAttention.forward."""
        num_heads = self.num_heads
        nope_size = self.kv_lora_rank
        q_len = hidden_states.size(0)

        # qkv_proj

        query_states, key_states, value_states, q_pe, k_pe = self._qkv_proj(
            hidden_states, num_heads=num_heads
        )

        key_states = key_states.unsqueeze(1)
        value_states = value_states.unsqueeze(1)

        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        # query_states[..., nope_size:] = q_pe
        # key_states[..., nope_size:] = k_pe

        attn_output = self.attn_fwd(
            query_states,
            key_states,
            value_states,
        )

        attn_bmm_out = attn_output.new_empty(q_len, num_heads, self.v_head_dim)

        self.vc(attn_output, attn_bmm_out)
        attn_output = attn_bmm_out.reshape(attn_bmm_out.size(0), -1)
        attn_output = self.o_proj(attn_output)
        return attn_output
