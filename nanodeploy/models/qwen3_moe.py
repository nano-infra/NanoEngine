from typing import Dict

import torch
import torch.distributed as dist
import torch.nn.functional as F

from dlblas.layers.moe.ep_moe import build_deepep_moe

from nanodeploy.layers.activation import SiluAndMul
from nanodeploy.layers.attention import Attention
from nanodeploy.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanodeploy.layers.layernorm import RMSNorm
from nanodeploy.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from nanodeploy.layers.rotary_embedding import get_rope
from nanodeploy.worker.context import get_context
from nanodeploy.worker.distributed import get_dist_context
from nanodeploy.worker.runner_config import get_runner_config

from torch import nn
from transformers import Qwen3MoeConfig

from .quant_config import QuantizationConfig


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

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quantization_config=quantization_config,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quantization_config=quantization_config,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
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

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            meta=meta,
            weight_tensor=gate_up_proj_tensor,
            scale_tensor=gate_up_scale_inv_tensor,
            quantization_config=quantization_config,
        )

        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            meta=meta,
            weight_tensor=down_proj_tenosr,
            scale_tensor=down_scale_inv_tensor,
            quantization_config=quantization_config,
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


class MoEGate(nn.Module):
    """MoE Gate module for computing topk_idx and topk_weights."""

    def __init__(
        self,
        config: Qwen3MoeConfig,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.renormalize = getattr(config, "norm_topk_prob", True)

        # gating weight - shape: (num_experts, hidden_size) to match nn.Linear
        self.weight = nn.Parameter(
            torch.empty(
                (self.num_experts, self.hidden_size),
                dtype=dtype,
                device=device,
            )
        )
        # No bias for gate
        self.register_parameter("bias", None)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute topk_weights and topk_idx from hidden_states.

        Args:
            hidden_states: Input tensor of shape (num_tokens, hidden_size)

        Returns:
            topk_weights: Tensor of shape (num_tokens, top_k)
            topk_idx: Tensor of shape (num_tokens, top_k)
        """
        # Compute router logits: (num_tokens, num_experts)
        router_logits = F.linear(hidden_states, self.weight.to( hidden_states.dtype))

        # Compute routing weights via softmax
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)

        # Select top-k experts
        topk_weights, topk_idx = torch.topk(
            routing_weights, self.top_k, dim=-1
        )

        # Apply EPLB if enabled
        if get_runner_config().perfect_eplb:
            ep_size = get_dist_context().ffn_ep_world_size
            topk_idx = compute_topk_ids(topk_idx, ep_size, self.num_experts)

        # Renormalize weights if needed
        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            if not topk_weights.is_contiguous():
                topk_weights = topk_weights.contiguous()

        return topk_weights, topk_idx


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

        weight_dtype = quantization_config.dtype or config.dtype

        # gating - use separate MoEGate module
        self.gate = MoEGate(config, dtype=weight_dtype, device="cuda")

        self.tp_size = get_dist_context().ffn_tp_world_size

        # global parameter for DeepGEMM
        self.gate_up_proj = nn.Parameter(
            torch.ones(
                self.num_experts_per_rank,
                config.moe_intermediate_size * 2 // self.tp_size,
                config.hidden_size,
                dtype=weight_dtype,
                device="cuda",
            ),
        )

        self.down_proj = nn.Parameter(
            torch.ones(
                self.num_experts_per_rank,
                config.hidden_size,
                config.moe_intermediate_size // self.tp_size,
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
                    // quantization_config.block_size[0]
                    // self.tp_size,
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
                        // quantization_config.block_size[1]
                        // self.tp_size,
                        dtype=torch.float32,
                        device="cuda",
                    )
                )
                if quantization_config.quant_method == "fp8"
                else None
            )

        local_expert_id = lambda i: i - self.expert_list_this_rank[0]
        is_local_expert = lambda i: i in self.expert_list_this_rank

        def _get_data_for_expert(i, t):
            return t[local_expert_id(i)] if is_local_expert(i) else None

        self.experts = nn.ModuleList(
            [
                Qwen3MoeMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.moe_intermediate_size,
                    hidden_act=config.hidden_act,
                    meta=not is_local_expert(i),
                    gate_up_proj_tensor=_get_data_for_expert(i, self.gate_up_proj),
                    down_proj_tenosr=_get_data_for_expert(i, self.down_proj),
                    gate_up_scale_inv_tensor=(
                        None
                        if quantization_config.quant_method != "fp8"
                        else _get_data_for_expert(i, self.gate_up_scale_inv)
                    ),
                    down_scale_inv_tensor=(
                        None
                        if quantization_config.quant_method != "fp8"
                        else (_get_data_for_expert(i, self.down_scale_inv))
                    ),
                    quantization_config=self.quantization_config,
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

        self.act_fn = config.hidden_act

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
    def num_experts_per_rank(self):
        ep_world_size = get_dist_context().ffn_ep_world_size
        return self.num_experts // ep_world_size

    @property
    def expert_list_this_rank(self):
        ep_group = get_dist_context().ffn_ep_group
        ep_rank = dist.get_rank(group=ep_group)

        expert_id_begin = ep_rank * self.num_experts_per_rank
        expert_id_end = (ep_rank + 1) * self.num_experts_per_rank

        return list(range(expert_id_begin, expert_id_end))

    def forward(self, hidden_states: torch.Tensor):
        if self.ep_size > 1:
            assert (
                self.quantization_config.quant_method == "fp8"
            ), "Only FP8 EP is supported by now"
            context = get_context()
            moe = self.fusedmoe_build(not context.is_prefill)
            
            # Use MoEGate to compute topk_weights and topk_idx
            routing_weights, selected_experts = self.gate(hidden_states)

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
        else:
            _, hidden_dim = hidden_states.shape
            
            # Use MoEGate to compute topk_weights and topk_idx
            routing_weights, selected_experts = self.gate(hidden_states)
            
            # we cast back to the input dtype
            routing_weights = routing_weights.to(hidden_states.dtype)

            final_hidden_states = torch.zeros(
                hidden_states.shape,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

            # One hot encode the selected experts to create an expert mask
            # this will be used to easily index which expert is going to be sollicitated
            expert_mask = torch.nn.functional.one_hot(
                selected_experts, num_classes=self.num_experts
            ).permute(2, 1, 0)

            expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in expert_hitted:
                expert_layer = self.experts[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
                # Index the correct hidden states and compute the expert hidden state for
                # the current expert. We need to make sure to multiply the output hidden
                # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
                current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                current_hidden_states = (
                    expert_layer(current_state) * routing_weights[top_x, idx, None]
                )

                # However `index_add_` only support torch tensors for indexing so we'll use
                # the `top_x` tensor here.
                final_hidden_states.index_add_(
                    0, top_x, current_hidden_states.to(hidden_states.dtype)
                )
        return final_hidden_states


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
        self.ep_size = get_dist_context().ffn_ep_world_size


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

        # all_gather
        hidden_states = self.self_attn(positions, hidden_states)
        # all_to_all

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        # print(f"{hidden_states.shape=}")
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
    # 把本来 Qwen3MoeSparseMoeBlock 要做的工作分步骤拆解到这里
    def forward_yield(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) :
        # stage 0
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        yield
        # stage 1
        
        # all_gather
        hidden_states = self.self_attn(positions, hidden_states)
        # all_to_all
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        
        if isinstance(self.mlp, Qwen3MoeSparseMoeBlock) and self.mlp.ep_size > 1:
            assert (
                self.mlp.quantization_config.quant_method == "fp8"
            ), "Only FP8 EP is supported by now"
            
            context = get_context()
            # Assuming decode mode (low latency)
            moe = self.mlp.fusedmoe_build(low_latency_mode=not context.is_prefill)
            
            # Use MoEGate to compute topk_weights and topk_idx
            topk_weights, topk_idx = self.mlp.gate(hidden_states)
            
            # print(f"{hidden_states.shape=}, {topk_weights.shape=}, {topk_idx.shape=}")
            if context.is_prefill:
                hs_quant, hs_scale = moe.per_token_group_quant_fp8(hidden_states)
                x, recv_topk_ids, recv_topk_weights, recv_tokens_per_expert = moe.token_dispatcher.dispatch(
                    (hs_quant, hs_scale),
                    topk_idx,
                    topk_weights,
                    self.mlp.expert_list_this_rank,
                )
                yield
                state = {
                    "recv_hidden_states": x,
                    "recv_topk_idx": recv_topk_ids,
                    "recv_topk_weights": recv_topk_weights,
                    "recv_tokens_per_expert": recv_tokens_per_expert,
                }
                out_states = moe.fusedmoe_forward(
                    state,
                    self.mlp.gate_up_proj,
                    self.mlp.gate_up_scale_inv,
                    self.mlp.down_proj,
                    self.mlp.down_scale_inv,
                )
                yield
                final_hidden_states = moe.token_dispatcher.combine(out_states)
                yield
            else:
                # Low latency decode mode
                (recv_hidden_states, recv_expert_count, handle, event,
                 hook) = moe.token_dispatcher.dispatch_async(
                    hidden_states,
                    topk_idx,
                    use_fp8=True, # 这样子 recv_hidden_states 是一个元组
                    async_finish=False
                )
                hook()

                yield
                hidden_shape = hidden_states.shape
                expected_m = (hidden_shape[0] * self.ep_size  * topk_idx.shape[1] +
                       moe.token_dispatcher.num_experts) //  moe.token_dispatcher.num_experts
                # out_states = moe.experts(
                out_states = moe.experts(
                    recv_hidden_states, # 传入的需要是 权重和对应的 scale，需要前面开 Fp8
                    # recv_hidden_states,
                    self.mlp.gate_up_proj,
                    self.mlp.gate_up_scale_inv,
                    self.mlp.down_proj,
                    self.mlp.down_scale_inv,
                    recv_expert_count,
                    expected_m,
                )

                yield
                # print(f"{out_states.shape=};{topk_idx.shape=};{topk_weights.shape=}")
                final_hidden_states, event, hook = moe.token_dispatcher.combine_async(
                    out_states, topk_idx, topk_weights, handle, async_finish=False
                    # out_states, topk_idx, topk_weights.to(torch.float32), handle, async_finish=False
                )
                
                hook()

                yield
        else:
            final_hidden_states = self.mlp(hidden_states)
            yield
            yield
            yield

        outputs = (final_hidden_states, residual)
        return outputs


class Qwen3MoeModel(nn.Module):

    def __init__(
        self, config: Qwen3MoeConfig, quantization_config: QuantizationConfig
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                Qwen3MoeDecoderLayer(config, quantization_config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            Y = 1
            if Y:
                # print(f"{hidden_states.shape=}, entering layer {layer}")
                runner = layer.forward_yield(positions, hidden_states, residual)
                try:
                    while True:
                        next(runner)
                except StopIteration as e:
                    hidden_states, residual = e.value
            else:
                # print(f"{hidden_states.shape=}, entering layer {layer}")
                hidden_states, residual = layer.forward(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3MoeForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
        "gate_scale_inv": ("gate_up_scale_inv", 0),
        "up_scale_inv": ("gate_up_scale_inv", 1),
    }

    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        quantization_config = QuantizationConfig(
            **getattr(config, "quantization_config", dict())
        )
        self.model = Qwen3MoeModel(config, quantization_config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits
