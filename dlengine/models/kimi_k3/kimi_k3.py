"""Text-only Kimi K3 model for NanoDeploy."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from dlengine.context_v2.batch import get_batch_context
from dlengine.context_v2.cache.plan import kimi_k3_cache_plan
from dlengine.context_v2.distributed import get_dist_context
from dlengine.layers import get_backend
from dlengine.layers.activation import SituAndMul
from dlengine.layers.backends.kda import FlashInferKDA
from dlengine.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from dlengine.layers.layernorm import RMSNorm
from dlengine.layers.parallelism_transition import AttnToFfnTransition, FfnToAttnTransition
from dlengine.models.deepseek_v2.deepseek_v2 import DeepseekV2Attention
from dlengine.models.pp_utils import get_pp_layer_range, make_pp_layers, pp_recv_hidden, pp_send_hidden
from dlengine.models.quant_config import QuantizationConfig


class KimiMLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int, *, replicated: bool = False):
        super().__init__()
        backend = get_backend()
        if replicated:
            self.gate_proj = backend.get_replicated_linear(hidden, intermediate)
            self.up_proj = backend.get_replicated_linear(hidden, intermediate)
            self.down_proj = backend.get_replicated_linear(intermediate, hidden)
        else:
            group = get_dist_context().ffn_tp_group
            self.gate_proj = backend.get_column_parallel_linear(hidden, intermediate, tp_group=group)
            self.up_proj = backend.get_column_parallel_linear(hidden, intermediate, tp_group=group)
            self.down_proj = backend.get_row_parallel_linear(intermediate, hidden, tp_group=group)
        self.act = SituAndMul(4.0, 25.0)
        self.register_buffer("fused_gate_up_weight", None, persistent=False)

    def prepare_fused_gate_up(self) -> None:
        weight = torch.cat((self.gate_proj.weight, self.up_proj.weight), dim=0).contiguous()
        split = self.gate_proj.weight.shape[0]
        self.fused_gate_up_weight = weight
        self.gate_proj.weight.data = weight[:split]
        self.up_proj.weight.data = weight[split:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused_gate_up_weight is None:
            raise RuntimeError("K3 gate/up weights were not fused after loading")
        from dlengine.kernel.cutedsl_bf16_gemm import blackwell_bf16_linear

        gate_up = blackwell_bf16_linear(x, self.fused_gate_up_weight)
        return self.down_proj(self.act(gate_up))


class KimiMLAAttention(DeepseekV2Attention):
    def __init__(self, config, layer_idx: int, cache_layer_idx: int):
        super().__init__(
            config, QuantizationConfig(), layer_idx, cache_layer_idx, skip_rope=True
        )
        self.g_proj = get_backend().get_column_parallel_linear(
            config.hidden_size,
            config.num_attention_heads * config.v_head_dim,
            tp_group=get_dist_context().attn_tp_group,
        )
        self._gate_input = None
        inner = self.o_proj.forward

        def gated_o_proj(x, *args, **kwargs):
            gate_input, self._gate_input = self._gate_input, None
            if gate_input is not None:
                from dlengine.kernel.triton.generic.sigmoid_mul import (
                    sigmoid_mul_triton,
                )

                x = sigmoid_mul_triton(x, self.g_proj(gate_input))
            return inner(x, *args, **kwargs)

        self.o_proj.forward = gated_o_proj

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        self._gate_input = hidden_states
        return super().forward(positions, hidden_states)


class AttentionResidual:
    def __init__(self, hidden: torch.Tensor, blocks: int):
        self.bank = hidden.new_empty(hidden.shape[0], blocks, hidden.shape[1])
        self.valid = 0

    @staticmethod
    def _combined_score_weight(proj, norm):
        cached = getattr(proj, "_k3_residual_score_weight", None)
        if cached is None:
            cached = (norm.weight.float() * proj.weight.squeeze().float()).to(norm.weight.dtype).contiguous()
            proj._k3_residual_score_weight = cached
        return cached

    def aggregate(
        self, prefix, delta, proj, score_norm, out_norm, *, write=False, rows=None
    ):
        bank = self.bank if rows is None else self.bank[rows]
        if bank.shape[0] < delta.shape[0]:
            bank = F.pad(bank, (0, 0, 0, 0, 0, delta.shape[0] - bank.shape[0]))
        if prefix is not None and prefix.shape[0] != delta.shape[0]:
            prefix = prefix[rows]
            if prefix.shape[0] < delta.shape[0]:
                prefix = F.pad(prefix, (0, 0, 0, delta.shape[0] - prefix.shape[0]))
        prefix = delta if prefix is None else prefix + delta
        if self.valid:
            from dlengine.kernel.jit.sgl.attn_res import (
                fused_attention_residual_tma,
            )

            output = fused_attention_residual_tma(
                prefix,
                bank,
                self.valid,
                self._combined_score_weight(proj, score_norm),
                out_norm.weight,
                score_norm.eps,
                write_prefix=write and rows is None,
            )
        else:
            output = out_norm(prefix)
        if write:
            if self.valid == 0 or rows is not None:
                if rows is None:
                    self.bank[:, self.valid].copy_(prefix)
                else:
                    target = self.bank[rows, self.valid]
                    if target.shape[0]:
                        target.copy_(prefix[: target.shape[0]])
            self.valid += 1
        return output, prefix


class KimiMoE(nn.Module):
    def __init__(self, config, quant: QuantizationConfig, layer_idx: int):
        super().__init__()
        self.hidden = config.hidden_size
        self.latent = config.routed_expert_hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.gate = nn.Linear(self.hidden, self.num_experts, bias=False)
        self.e_score_correction_bias = nn.Parameter(torch.zeros(self.num_experts, dtype=torch.float32))
        backend = get_backend()
        self.routed_expert_down_proj = backend.get_replicated_linear(self.hidden, self.latent)
        self.routed_expert_norm = RMSNorm(self.latent, eps=config.rms_norm_eps)
        self.routed_expert_up_proj = backend.get_replicated_linear(self.latent, self.hidden)
        ctx = get_dist_context()
        self.experts = backend.get_distributed_routed_experts(
            hidden_size=self.latent,
            intermediate_size=config.moe_intermediate_size,
            num_experts=self.num_experts,
            top_k=self.top_k,
            ep_size=ctx.ffn_ep_world_size,
            tp_size=ctx.ffn_tp_world_size,
            ep_group=ctx.ffn_ep_group,
            tp_group=ctx.ffn_tp_group,
            quantization_config=quant,
            layer_idx=layer_idx,
            activation="situ",
            activation_situ_beta=config.activation_situ_beta,
            activation_situ_linear_beta=config.activation_situ_linear_beta,
            routed_scaling_factor=config.routed_scaling_factor,
        )
        self.shared_experts = KimiMLP(
            self.hidden, config.moe_intermediate_size * config.num_shared_experts,
            replicated=True,
        )
        self.register_buffer("fused_front_weight", None, persistent=False)
        self._shared_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self._shared_event = torch.cuda.Event() if torch.cuda.is_available() else None

    def prepare_fused_front(self) -> None:
        """Merge router and latent-down weights once, outside graph capture."""
        gate_weight = self.gate.weight
        down_weight = self.routed_expert_down_proj.weight
        self.fused_front_weight = torch.cat((gate_weight, down_weight), dim=0).contiguous()
        # Keep loader-visible parameters as views of the merged allocation so
        # the unfused copies can be released rather than costing ~6 GiB/GPU.
        self.gate.weight.data = self.fused_front_weight[: self.num_experts]
        self.routed_expert_down_proj.weight.data = self.fused_front_weight[
            self.num_experts :
        ]

    def forward(self, x: torch.Tensor, prefix: Optional[torch.Tensor] = None):
        if self.fused_front_weight is None:
            raise RuntimeError("K3 fused MoE front was not prepared after weight loading")
        from dlengine.kernel.jit.sgl.moe_front import fused_front

        renormalize = getattr(self.experts, "routed_scaling_factor", 1.0) == 1.0
        weights, ids, latent = fused_front(
            x,
            self.fused_front_weight,
            self.e_score_correction_bias,
            self.latent,
            self.top_k,
            renormalize,
        )

        shared = None
        shared_event = None
        if self._shared_stream is not None and x.numel():
            current = torch.cuda.current_stream(x.device)
            self._shared_stream.wait_stream(current)
            with torch.cuda.stream(self._shared_stream):
                shared = self.shared_experts(x)
                self._shared_event.record(self._shared_stream)
                shared_event = self._shared_event
            x.record_stream(self._shared_stream)
        else:
            shared = self.shared_experts(x)
        routed = self.experts(
            latent, ids, weights, is_prefill=get_batch_context().is_prefill
        )
        routed = self.routed_expert_up_proj(self.routed_expert_norm(routed))
        if shared_event is not None:
            torch.cuda.current_stream(x.device).wait_event(shared_event)
        if prefix is not None:
            from dlengine.kernel.jit.sgl.add3 import add3, covered

            if covered(routed, shared, prefix):
                return add3(routed, shared, prefix)
        out = routed + shared
        return out if prefix is None else out + prefix


class KimiDecoderLayer(nn.Module):
    def __init__(self, config, quant, layer_idx: int, state_idx: int, cache_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_kda = config.layer_types[layer_idx] == "linear_attention"
        self.self_attn = (
            FlashInferKDA(layer_idx, state_idx, config)
            if self.is_kda
            else KimiMLAAttention(config, layer_idx, cache_idx)
        )
        self.is_moe = layer_idx >= config.first_k_dense_replace and layer_idx % config.moe_layer_freq == 0
        self.mlp = KimiMoE(config, quant, layer_idx) if self.is_moe else KimiMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.use_res = config.attn_res_block_size is not None
        if self.use_res:
            self.write_block = layer_idx % config.attn_res_block_size == 0
            self.self_attention_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.mlp_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.self_attention_res_proj = get_backend().get_replicated_linear(config.hidden_size, 1)
            self.mlp_res_proj = get_backend().get_replicated_linear(config.hidden_size, 1)
        # SP-MoE: shard TP-replicated rows before MegaMoE and gather afterward.
        self.attn_to_ffn = AttnToFfnTransition()
        self.ffn_to_attn = FfnToAttnTransition(self.attn_to_ffn)
        if self.use_res and get_dist_context().attn_tp_world_size > 1:
            # K3 residual aggregation is token-local, so reduce-scatter the
            # TP-partial o_proj result before the aggregation and MoE tail.
            self.self_attn.o_proj.defer_reduce = True

    def forward(self, positions, hidden, prefix, residual_bank):
        if residual_bank is None:
            if prefix is None:
                prefix, hidden = hidden, self.input_layernorm(hidden)
            else:
                hidden, prefix = self.input_layernorm(hidden, prefix)
            hidden = self.self_attn(hidden) if self.is_kda else self.self_attn(positions, hidden)
            hidden, prefix = self.post_attention_layernorm(hidden, prefix)
            hidden = self.attn_to_ffn(hidden)
            hidden = self.mlp(hidden)
            return self.ffn_to_attn(hidden), prefix

        hidden, prefix = residual_bank.aggregate(
            prefix, hidden, self.self_attention_res_proj,
            self.self_attention_res_norm, self.input_layernorm,
            write=self.write_block,
        )
        if self.write_block:
            prefix = None
        hidden = self.self_attn(hidden) if self.is_kda else self.self_attn(positions, hidden)
        rows = self.attn_to_ffn.local_rows(hidden)
        hidden = self.attn_to_ffn.reduce_scatter(hidden)
        hidden, prefix = residual_bank.aggregate(
            prefix, hidden, self.mlp_res_proj, self.mlp_res_norm,
            self.post_attention_layernorm, rows=rows,
        )
        # Attention-residual consumes prefix in the FFN tail. Under TP8/EP8
        # hidden has already been token-scattered, so prefix must follow the
        # identical row mapping before the local MoE/dense tail add.
        hidden = self.mlp(hidden, prefix) if self.is_moe else self.mlp(hidden) + prefix
        return self.ffn_to_attn(hidden), None


class KimiK3Model(nn.Module):
    def __init__(self, config, quant):
        super().__init__()
        ctx = get_dist_context()
        self.is_first_pp_stage = ctx.is_first_pp_stage
        self.is_last_pp_stage = ctx.is_last_pp_stage
        self.hidden_size = config.hidden_size
        self.hidden_dtype = config.dtype
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size) if self.is_first_pp_stage else None
        pp_start, _ = get_pp_layer_range(config.num_hidden_layers)
        local_types = config.layer_types[pp_start:]
        state_prefix = [0]
        cache_prefix = [0]
        for kind in local_types:
            state_prefix.append(state_prefix[-1] + (kind == "linear_attention"))
            cache_prefix.append(cache_prefix[-1] + (kind == "full_attention"))
        self.start_layer, self.end_layer, self.layers = make_pp_layers(
            config.num_hidden_layers,
            lambda i: KimiDecoderLayer(
                config, quant, i,
                state_prefix[i - pp_start], cache_prefix[i - pp_start],
            ),
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if self.is_last_pp_stage else None
        self.output_attn_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if self.is_last_pp_stage else None
        self.output_attn_res_proj = get_backend().get_replicated_linear(config.hidden_size, 1) if self.is_last_pp_stage else None
        self.blocks = math.ceil(config.num_hidden_layers / config.attn_res_block_size)

    def forward(self, input_ids, positions, inputs_embeds=None):
        hidden = (inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)) if self.is_first_pp_stage else pp_recv_hidden(positions.numel(), self.hidden_size, self.hidden_dtype)
        prefix = None
        bank = AttentionResidual(hidden, self.blocks)
        for i in range(self.start_layer, self.end_layer):
            hidden, prefix = self.layers[i](positions, hidden, prefix, bank)
        if not self.is_last_pp_stage:
            pp_send_hidden(hidden if prefix is None else hidden + prefix)
            return hidden
        hidden, _ = bank.aggregate(prefix, hidden, self.output_attn_res_proj, self.output_attn_res_norm, self.norm)
        return hidden


class KimiK3ForConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.quantization_config = QuantizationConfig(**getattr(config, "quantization_config", {}))
        self.model = KimiK3Model(config, self.quantization_config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size) if get_dist_context().is_last_pp_stage else None

    def forward(self, input_ids, positions, inputs_embeds=None):
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)

    def get_cache_plan(self):
        return kimi_k3_cache_plan()

    def load_weights(self, weights):
        from .kimi_k3_loader import load_weights
        load_weights(self, weights)
