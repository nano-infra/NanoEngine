import torch
import math
import torch.nn.functional as F
from torch import nn
from typing import Any, List, Optional, Tuple, Iterable
from lmdeploy.pytorch.nn import SiluAndMul, RMSNorm, RopeType, ApplyRotaryEmb, build_rotary_embedding
from lmdeploy.pytorch.nn.moe import SoftmaxTopK
from lmdeploy.pytorch.nn.rotary_embedding import YarnParameters
from nanolmdeploy.layers.moe import build_fused_moe
from nanolmdeploy.layers.linear import build_gateup_linear, build_down_linear, build_rowwise_linear, build_colwise_linear, build_o_proj
from nanolmdeploy.layers.attention import Attention

def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0

def compute_topk_ids(topk_ids, ranks, num_experts):
    shape = topk_ids.shape
    step = num_experts // ranks
    topk_ids = (
        (torch.arange(0, topk_ids.numel(), dtype=topk_ids.dtype, device=topk_ids.device) // ranks) % step
        + (torch.arange(0, topk_ids.numel(), dtype=topk_ids.dtype, device=topk_ids.device) % ranks) * step
    ) % num_experts
    topk_ids = topk_ids.reshape(shape)
    return topk_ids

class MoEGate(nn.Module):
    """Deepseek Gate."""

    def __init__(self,
                 config: Any,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.renormalize = self.top_k > 1 and self.norm_topk_prob

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.randn((self.n_routed_experts, self.gating_dim), dtype=dtype, device=device))
        if self.topk_method == 'noaux_tc':
            self.e_score_correction_bias = nn.Parameter(
                torch.randn((self.n_routed_experts, ), dtype=dtype, device=device))
        self.softmax_topk = SoftmaxTopK(self.top_k)

    def _compute_scores(self, logits: torch.Tensor):
        """Compute scores."""
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1, dtype=torch.float32)
        elif self.scoring_func == 'sigmoid':
            scores = logits.sigmoid()
        else:
            raise NotImplementedError('insupportable scoring function '
                                      f'for MoE gating: {self.scoring_func}')
        return scores

    def forward(self, hidden_states: torch.Tensor):
        """forward."""
        sequence_length, hidden_dim = hidden_states.shape
        router_logits = F.linear(hidden_states, self.weight)

        if self.topk_method == 'greedy':
            topk_weight, topk_idx = self.softmax_topk(router_logits)
        elif self.topk_method == 'group_limited_greedy':
            scores = router_logits
            grouped_logits = scores.unflatten(-1, (self.n_group, -1))
            group_scores = (grouped_logits.max(-1).values)
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            group_mask = ~group_mask.bool()[..., None]
            grouped_logits = grouped_logits.masked_fill(group_mask, 0.0)
            scores = grouped_logits.flatten(1, 2)
            topk_weight, topk_idx = self.softmax_topk(scores)
        elif self.topk_method == 'noaux_tc':
            scores = self._compute_scores(router_logits)
            scores_for_choice = scores.view(sequence_length, -1) + self.e_score_correction_bias[None]
            group_scores = (scores_for_choice.view(sequence_length, self.n_group,
                                                   -1).topk(2, dim=-1)[0].sum(dim=-1))  # [n, n_group]
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            score_mask = (group_mask.unsqueeze(-1).expand(sequence_length, self.n_group,
                                                          self.n_routed_experts // self.n_group).reshape(
                                                              sequence_length, -1))  # [n, e]
            tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
            topk_weight = scores.gather(1, topk_idx)
        else:
            raise RuntimeError(f'Unsupported topk_method: {self.topk_method}')

        if self.renormalize:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
            if not topk_weight.is_contiguous():
                topk_weight = topk_weight.contiguous()

        if not self.renormalize or self.topk_method == 'noaux_tc':
            topk_weight = topk_weight * self.routed_scaling_factor

        return topk_weight, topk_idx

class DeepseekV2MoE(nn.Module):
    """Deepseek v2 MoE."""

    def __init__(self, config: Any, layer_idx, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.moe_intermediate_size
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.renormalize = self.top_k > 1 and self.norm_topk_prob
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        
        self.gate = MoEGate(config, dtype=dtype, device=device)
        self.experts = build_fused_moe(
            self.hidden_dim,
            self.ffn_dim,
            self.num_experts,
            top_k=self.top_k,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            layer_idx=layer_idx,
        )
        self.shared_experts = None
        if config.n_shared_experts is not None:
            intermediate_size = (config.moe_intermediate_size * config.n_shared_experts)
            self.shared_experts = DeepseekV2MLP(
                config=config,
                intermediate_size=intermediate_size,
                dtype=dtype,
                device=device,
            )

    def forward(self, hidden_states: torch.Tensor):
        """forward."""

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        topk_weights, topk_ids = self.gate(hidden_states)

        if True:
            ranks = torch.distributed.get_world_size()
            topk_ids = compute_topk_ids(topk_ids, ranks, self.num_experts)

        out_states = self.experts(
            hidden_states,
            topk_weights,
            topk_ids,
        )

        if self.shared_experts is not None:
            shared_states = self.shared_experts(hidden_states)
            out_states += shared_states
        out_states = out_states.reshape(batch_size, sequence_length, -1)

        return out_states


class DeepseekV2MLP(nn.Module):
    """Deepseek v2 mlp."""

    def __init__(self,
                 config: Any,
                 intermediate_size: int = None,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)

        # gate up
        if intermediate_size is None:
            intermediate_size = config.intermediate_size

        self.gate_up_proj = build_gateup_linear(
            config.hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
        )

        # silu and mul
        self.act_fn = SiluAndMul(inplace=True)

        # down
        self.down_proj = build_down_linear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
        )

    def forward(self, x):
        """forward."""
        gate_up = self.gate_up_proj(x)
        act = self.act_fn(gate_up)
        return self.down_proj(act)

class DeepseekV2DecoderLayer(nn.Module):
    """Deepseekv2 decoder layer."""

    def __init__(self, config: Any, layer_idx: int, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.layer_idx = layer_idx
        quantization_config = None

        self.self_attn = DeepseekV2Attention(config, dtype=dtype, device=device)

        # mlp
        self.mlp = (DeepseekV2MoE(config, layer_idx, dtype=dtype, device=device) if
                    (config.n_routed_experts is not None and layer_idx >= config.first_k_dense_replace
                     and layer_idx % config.moe_layer_freq == 0) else DeepseekV2MLP(config, dtype=dtype, device=device))

        # build input layer norm
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       config.rms_norm_eps,
                                       quant_config=quantization_config,
                                       dtype=dtype,
                                       device=device)

        # build attention layer norm
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[List[torch.FloatTensor]],
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        outputs = (hidden_states, residual)
        return outputs
    
class DeepseekV2Model(nn.Module):
    """Mixtral model."""

    def __init__(self, config: Any, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config
        self.padding_idx = getattr(config, 'pad_token_id', None)
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size,
                                         config.hidden_size,
                                         self.padding_idx,
                                         dtype=dtype,
                                         device=device)
        self.layers = nn.ModuleList([
            DeepseekV2DecoderLayer(config, layer_idx, dtype=dtype, device=device)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # build norm
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, quant_config=None, dtype=dtype, device=device)

        emb_type = RopeType.LinearScaling
        rope_dim = config.qk_rope_head_dim if getattr(config, 'use_mla', True) else (config.hidden_size //
                                                                                     config.num_attention_heads)
        rope_max_pos_emb = config.max_position_embeddings
        rope_base = config.rope_theta
        scaling_factor = 1.0
        other_params = dict()
        if config.rope_scaling is not None:
            scaling_type = config.rope_scaling['type']
            scaling_factor = config.rope_scaling['factor']
            if scaling_type == 'dynamic':
                emb_type = RopeType.DynamicNTKScaling
            elif scaling_type == 'yarn':
                emb_type = RopeType.Yarn
                rope_max_pos_emb = config.rope_scaling.get('original_max_position_embeddings', 4096)
                kwargs = {
                    key: config.rope_scaling[key]
                    for key in [
                        'beta_fast',
                        'beta_slow',
                        'mscale',
                        'mscale_all_dim',
                    ] if key in self.config.rope_scaling
                }
                yarn_params = YarnParameters(**kwargs)
                other_params['yarn_params'] = yarn_params

        self.rotary_emb = build_rotary_embedding(rope_dim,
                                                 rope_max_pos_emb,
                                                 rope_base,
                                                 scaling_factor,
                                                 emb_type=emb_type,
                                                 **other_params)
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
    ):
        """forward."""
        inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        residual = None
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos, sin = cos[0], sin[0]
        rotary_pos_emb = (cos, sin)
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx]
            with torch.profiler.record_function(f"decoder_layer_{idx}"):
                hidden_states, residual = decoder_layer(
                    hidden_states,
                    rotary_pos_emb=rotary_pos_emb,
                    past_key_value=past_key_value,
                    residual=residual,
                )

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states

class DeepseekV2ForCausalLM(nn.Module):
    """Mixture model for causalLM."""
    def __init__(self,
                 config: Any,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.quantization_config = getattr(config, 'quantization_config', None)
        self.dtype = dtype
        self.device = device
        self.model = DeepseekV2Model(config, dtype=self.dtype, device=self.device)
        self.lm_head = build_rowwise_linear(config.hidden_size,
                                            config.vocab_size,
                                            bias=False,
                                            dtype=dtype,
                                            device=device)
        
    
        
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
    ):
        hidden_states = self.model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        return hidden_states
        
    def compute_logits(self, hidden_states: torch.Tensor):
        """Compute logits of the model output."""
        return self.lm_head(hidden_states)

class DeepseekV2BMM(nn.Module):
    """Wrapped bmm."""

    def __init__(self, batch: int, in_features: int, out_features: int, dtype: torch.dtype, device: torch.device):
        super().__init__()

        weight = self.create_weight(batch, in_features, out_features, dtype=dtype, device=device)
        self.weight = torch.nn.Parameter(weight, requires_grad=False)

        self.batch = batch
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.device = device

    def create_weight(self, batch: int, in_features: int, out_features: int, dtype: torch.dtype, device: torch.device):
        """Create weight."""
        return torch.empty((batch, in_features, out_features), dtype=dtype, device=device)

    def forward(self, x: torch.Tensor, output: torch.Tensor):
        """forward."""
        torch.bmm(x.transpose(0, 1), self.weight, out=output.transpose(0, 1))

class DeepseekV2Attention(nn.Module):
    """Deepseekv2 attention."""

    def __init__(self, config: Any, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        self.q_lora_rank = config.q_lora_rank
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        num_key_value_heads = getattr(config, 'num_key_value_heads', 1)

        if self.q_lora_rank is None:
            self.q_proj = build_colwise_linear(
                self.hidden_size,
                self.num_heads * self.q_head_dim,
                bias=False,
                dtype=dtype,
                device=device,
                quant_config=quantization_config,
            )
        else:
            self.q_a_proj = build_colwise_linear(
                self.hidden_size,
                config.q_lora_rank,
                bias=config.attention_bias,
                dtype=dtype,
                device=device,
                quant_config=quantization_config,
            )
            self.q_a_layernorm = RMSNorm(config.q_lora_rank,
                                         1e-6,
                                         quant_config=quantization_config,
                                         dtype=dtype,
                                         device=device)
            self.q_b_proj = build_colwise_linear(
                config.q_lora_rank,
                self.num_heads * self.q_head_dim,
                bias=False,
                dtype=dtype,
                device=device,
                quant_config=quantization_config,
            )

        self.kv_a_proj_with_mqa = build_colwise_linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
        )
        self.kv_a_layernorm = RMSNorm(config.kv_lora_rank,
                                      1e-6,
                                      quant_config=quantization_config,
                                      dtype=dtype,
                                      device=device)
        self.kc = DeepseekV2BMM(self.num_heads,
                                config.qk_nope_head_dim,
                                config.kv_lora_rank,
                                dtype=dtype,
                                device=device)

        self.apply_rotary_pos_emb = ApplyRotaryEmb()

        self.softmax_scale = self.q_head_dim**(-0.5)

        if config.rope_scaling is not None:
            mscale_all_dim = config.rope_scaling.get('mscale_all_dim', 0)
            scaling_factor = config.rope_scaling['factor']
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        self.attn_fwd = Attention(self.num_heads,
                                  config.kv_lora_rank + self.qk_rope_head_dim,
                                  scale=self.softmax_scale,
                                  num_kv_heads=num_key_value_heads,
                                  v_head_size=config.kv_lora_rank)

        self.vc = DeepseekV2BMM(self.num_heads, config.kv_lora_rank, self.v_head_dim, dtype=dtype, device=device)
        self.o_proj = build_o_proj(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
        )

    def _q_proj(self, hidden_states, num_heads: int, nope_size: int, pe_size: int):
        """Q proj."""
        q_len = hidden_states.size(1)

        query_states = hidden_states.new_empty(q_len, num_heads, nope_size + pe_size)

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(q_len, num_heads, self.q_head_dim)
        # q_pe: (q_len, num_heads, qk_rope_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        # q_nope: (q_len, num_heads, kv_lora_rank)
        q_nope_out = query_states[..., :nope_size]
        self.kc(q_nope, q_nope_out)
        return query_states, q_pe

    def _kv_proj(self, hidden_states, nope_size: int):
        """Kv proj."""
        # (q_len, 1, nope_size + pe_size)
        key_states = self.kv_a_proj_with_mqa(hidden_states[0, :, None])
        # (q_len, 1, pe_size)
        k_pe = key_states[..., nope_size:]
        # kv_a_layernorm
        value_states = key_states[..., :nope_size]
        value_states = self.kv_a_layernorm(value_states)
        key_states[..., :nope_size] = value_states
        return key_states, value_states, k_pe

    def _qkv_proj(self, hidden_states: torch.Tensor, num_heads: int):
        """Qkv proj."""
        nope_size = self.kv_lora_rank
        pe_size = self.qk_rope_head_dim
        query_states, q_pe = self._q_proj(hidden_states, num_heads, nope_size, pe_size)
        key_states, value_states, k_pe = self._kv_proj(hidden_states, nope_size)

        return query_states, key_states, value_states, q_pe, k_pe

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ):
        """Rewrite of LlamaAttention.forward."""
        num_heads = self.num_heads
        nope_size = self.kv_lora_rank
        q_len = hidden_states.size(1)

        # qkv_proj
        query_states, key_states, value_states, q_pe, k_pe = self._qkv_proj(hidden_states, num_heads=num_heads)

        cos, sin = rotary_pos_emb
        q_pe, k_pe = self.apply_rotary_pos_emb(
            q_pe,
            k_pe,
            cos,
            sin,
            inplace=False,
        )
        query_states[..., nope_size:] = q_pe
        key_states[..., nope_size:] = k_pe

        attn_output = self.attn_fwd(
            query_states,
            past_key_value[0],
        )

        # print(f"finish attention !!! attn_output shape: {attn_output.shape}",flush=True)

        attn_bmm_out = attn_output.new_empty(q_len, num_heads, self.v_head_dim)

        self.vc(attn_output, attn_bmm_out)
        attn_output = attn_bmm_out.flatten(-2, -1)[None]
        attn_output = self.o_proj(attn_output)

        return attn_output
