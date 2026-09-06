"""GLM-5.3-Flash hybrid decoder and checkpoint loader.

The backbone alternates KDA linear-attention and DeepSeek sparse-attention
blocks as declared by ``layer_types``.  The hidden stream is four-way mHC
through the decoder and is contracted before the LM head.
"""
from __future__ import annotations
import re
import torch
import torch.nn.functional as F
from torch import nn

from dlengine.runtime.context.distributed import get_dist_context
from dlengine.runtime.context.cache.plan import glm5_next_cache_plan
from dlengine.runtime.layers import get_backend
from dlengine.runtime.layers.backends.kda import FlashInferKDA
from dlengine.runtime.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from dlengine.runtime.layers.activation import SiluAndMul
from dlengine.runtime.layers.layernorm import RMSNorm
from dlengine.runtime.layers.parallelism_transition import AttnToFfnTransition, FfnToAttnTransition
from dlengine.runtime.models.deepseek_v2.deepseek_v2 import DeepseekV2Attention, DeepseekV2MLP, DeepseekV2MoE
from dlengine.runtime.models.deepseek_v2.deepseek_v2_mtp import DeepSeekMTP
from dlengine.runtime.models.pp_utils import get_pp_layer_range, make_pp_layers, pp_recv_hidden, pp_send_hidden
from dlengine.runtime.models.quant_config import QuantizationConfig


class Glm5NextHCProjector(nn.Module):
    """Reference mHC pre-mix used by every GLM-5.3 decoder block."""
    def __init__(self, hidden_size: int, hc_mult: int, sinkhorn_iters: int, eps: float):
        super().__init__()
        self.hidden_size, self.hc_mult = hidden_size, hc_mult
        self.sinkhorn_iters, self.eps = sinkhorn_iters, eps
        dim = hc_mult * hidden_size
        self.fn = nn.Parameter(torch.empty((2 + hc_mult) * hc_mult, dim, dtype=torch.float32))
        self.base = nn.Parameter(torch.empty((2 + hc_mult) * hc_mult, dtype=torch.float32))
        self.scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            # NanoDeploy's flattened token layout: [tokens, streams, hidden].
            flat = x.flatten(1).float()
            stream_dim = 1
        elif x.ndim == 4:
            # Transformers reference layout: [batch, sequence, streams, hidden].
            flat = x.flatten(start_dim=2).float()
            stream_dim = 2
        else:
            raise ValueError(
                "GLM-5.3 mHC expects [tokens, streams, hidden] or "
                f"[batch, sequence, streams, hidden], got {tuple(x.shape)}"
            )
        dtype = x.dtype
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.eps)
        mixes = F.linear(flat, self.fn) * rsqrt
        h = self.hc_mult
        pre = torch.sigmoid(mixes[..., :h] * self.scale[0] + self.base[:h]) + self.eps
        post = 2 * torch.sigmoid(
            mixes[..., h : 2 * h] * self.scale[1] + self.base[h : 2 * h]
        )
        comb = mixes[..., 2 * h :].view(*mixes.shape[:-1], h, h) * self.scale[2]
        comb = comb + self.base[2 * h :].view(h, h)
        comb = comb.softmax(-1) + self.eps
        comb = comb / (comb.sum(-2, keepdim=True) + self.eps)
        for _ in range(max(0, self.sinkhorn_iters - 1)):
            comb = comb / (comb.sum(-1, keepdim=True) + self.eps)
            comb = comb / (comb.sum(-2, keepdim=True) + self.eps)
        y = torch.sum(pre.unsqueeze(-1) * x, dim=stream_dim)
        return y.to(dtype), post.to(dtype), comb.to(dtype)


class Glm5NextHyperHead(nn.Module):
    """Reference GLM-5.3 final mHC stream contraction.

    GLM-5.3 deliberately uses an *unweighted mean* at the output boundary.
    This is different from DeepSeek-V4's learned hyper-head and is important
    for keeping the hidden-state scale (and therefore the logits) aligned with
    the checkpoint.
    """

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        if hidden_streams.ndim == 3:
            # NanoDeploy flattens [batch, sequence] into the leading token
            # dimension, so the stream axis is dim=1 ([tokens, hc_mult, H]).
            return hidden_streams.mean(dim=1)
        if hidden_streams.ndim >= 4:
            # Transformers reference layout is [batch, sequence, hc_mult, H].
            return hidden_streams.mean(dim=2)
        raise ValueError(
            "GLM-5.3 hyper-head expects [tokens, streams, hidden] or "
            f"[batch, sequence, streams, hidden], got {tuple(hidden_streams.shape)}"
        )


def _hc_post(x, residual, post, comb):
    return post.unsqueeze(-1) * x.unsqueeze(1) + torch.einsum("tij,tjd->tid", comb.transpose(1, 2), residual)


class Glm5NextDecoderLayer(nn.Module):
    def __init__(self, config, quantization_config, layer_idx: int, state_idx: int, cache_idx: int):
        super().__init__(); self.config = config; self.layer_idx = layer_idx
        self.is_kda = config.layer_types[layer_idx] == "linear_attention"
        self.self_attn = (FlashInferKDA(layer_idx, state_idx, config) if self.is_kda else
                          DeepseekV2Attention(config, quantization_config, layer_idx, cache_idx))
        dense = (getattr(config, "mlp_layer_types", ["sparse"] * config.num_hidden_layers)[layer_idx] == "dense")
        self.mlp = (DeepseekV2MLP(hidden_size=config.hidden_size, intermediate_size=config.intermediate_size,
                                  hidden_act=config.hidden_act, config=config, quantization_config=quantization_config,
                                  swiglu_limit=float(getattr(config, "swiglu_limit", 10.0)))
                    if dense else DeepseekV2MoE(config, quantization_config))
        if not dense:
            # DeepseekV2MoE is shared with models whose SwiGLU is unclamped.
            # GLM-5.3 clamps both routed and shared experts at 10.0, matching
            # Glm5NextTextExperts/Glm5NextTextMLP in the reference model.
            routed = getattr(self.mlp, "routed_experts", None)
            if routed is not None and hasattr(routed, "_swiglu_limit_runtime"):
                routed._swiglu_limit_runtime = float(
                    getattr(config, "swiglu_limit", 10.0)
                )
            shared = getattr(self.mlp, "shared_experts", None)
            if shared is not None:
                shared.act_fn = SiluAndMul(
                    swiglu_limit=float(getattr(config, "swiglu_limit", 10.0))
                )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_attn = Glm5NextHCProjector(config.hidden_size, config.hc_mult, config.hc_sinkhorn_iters, config.hc_eps)
        self.hc_ffn = Glm5NextHCProjector(config.hidden_size, config.hc_mult, config.hc_sinkhorn_iters, config.hc_eps)
        if get_dist_context().attn_tp_world_size > 1 and get_dist_context().ffn_ep_world_size > 1:
            self.attn_to_ffn = AttnToFfnTransition(); self.ffn_to_attn = FfnToAttnTransition(scatter_layer=self.attn_to_ffn)
        else:
            self.attn_to_ffn = nn.Identity(); self.ffn_to_attn = nn.Identity()

    def forward(self, hidden_states, positions, residual=None, indexer_state=None, reuse_indexer_topk=False):
        if hidden_states.ndim == 2:
            hidden_states = hidden_states.view(-1, self.config.hc_mult, self.config.hidden_size)
        residual_w = hidden_states if residual is None else residual.view_as(hidden_states)
        x, post, comb = self.hc_attn(hidden_states); x = self.input_layernorm(x)
        if self.is_kda: x = self.self_attn(x)
        else: x = self.self_attn(positions, x, indexer_state=indexer_state, reuse_indexer_topk=reuse_indexer_topk)
        hidden_states = _hc_post(x, residual_w, post, comb)
        x, post, comb = self.hc_ffn(hidden_states); x = self.post_attention_layernorm(x)
        x = self.ffn_to_attn(self.mlp(self.attn_to_ffn(x)))
        return _hc_post(x, hidden_states, post, comb), None


class Glm5NextModel(nn.Module):
    def __init__(self, config, quantization_config):
        super().__init__(); self.config = config
        ctx = get_dist_context(); self.is_first_pp_stage = ctx.is_first_pp_stage; self.is_last_pp_stage = ctx.is_last_pp_stage
        self.hidden_size = config.hidden_size; self.hc_mult = int(config.hc_mult)
        self.hidden_dtype = getattr(config, "dtype", torch.bfloat16)
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size) if self.is_first_pp_stage else None
        pp_start, _ = get_pp_layer_range(config.num_hidden_layers)
        self.start_layer, self.end_layer, self.layers = make_pp_layers(config.num_hidden_layers, lambda i: Glm5NextDecoderLayer(
            config, quantization_config, i,
            sum(t == "linear_attention" for t in config.layer_types[:i]),
            sum(t != "linear_attention" for t in config.layer_types[:i])))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if self.is_last_pp_stage else None
        self.hc_head = Glm5NextHyperHead() if self.is_last_pp_stage else None

    def forward(self, input_ids, positions, inputs_embeds=None):
        if self.is_first_pp_stage:
            hidden = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
            hidden = hidden.unsqueeze(1).expand(-1, self.hc_mult, -1).contiguous()
        else:
            hidden = pp_recv_hidden(positions.numel(), self.hc_mult * self.hidden_size, self.hidden_dtype).view(-1, self.hc_mult, self.hidden_size)
        residual = None
        for i in range(self.start_layer, self.end_layer):
            hidden, residual = self.layers[i](hidden, positions, residual)
        if not self.is_last_pp_stage:
            pp_send_hidden(hidden); return hidden
        # Decoder layers carry [tokens, hc_mult, hidden].  The reference
        # hyper-head contracts that stream axis with a mean immediately before
        # the final RMSNorm (not a sum).
        hidden = self.hc_head(hidden)
        return self.norm(hidden)


class Glm5NextForConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__(); self.config = config
        self.quantization_config = QuantizationConfig(**getattr(config, "quantization_config", {}))
        self.model = Glm5NextModel(config, self.quantization_config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size) if get_dist_context().is_last_pp_stage else None
    def forward(self, input_ids, positions, inputs_embeds=None): return self.model(input_ids, positions, inputs_embeds)
    def compute_logits(self, hidden_states): return self.lm_head(hidden_states)
    def get_cache_plan(self): return glm5_next_cache_plan()
    def load_weights(self, weights):
        from dlengine.runtime.models.deepseek_v2.deepseek_v2_loader import load_weights
        load_weights(self, _normalize_glm5_weights(weights))


def _normalize_glm5_weights(weights):
    """Normalize split GLM KDA tensors into NanoDeploy parameter names.

    This is intentionally a streaming generator: a GLM checkpoint is too large
    to materialize in memory merely to combine the three depthwise-conv shards.
    """
    conv, gates = {}, {}
    for name, raw, tensor in weights:
        # ``iterate_weights`` already strips VLM prefixes, but keeping the
        # normalizer self-contained also makes direct checkpoint/meta-device
        # tests and alternative loaders behave identically.
        if name.startswith("model.language_model."):
            name = "model." + name[len("model.language_model.") :]
        elif name.startswith("language_model."):
            name = "model." + name[len("language_model.") :]
        m = re.search(r"model\.layers\.(\d+)\.self_attn\.([qkv])_conv1d\.weight$", name)
        if m:
            conv.setdefault(int(m.group(1)), {})[m.group(2)] = tensor
            continue
        m = re.search(r"model\.layers\.(\d+)\.self_attn\.g_([ab])_proj\.weight$", name)
        if m:
            gates.setdefault(int(m.group(1)), {})[m.group(2)] = tensor
            continue
        name = re.sub(r"(layers\.\d+)\.(hc_attn|hc_ffn)_(fn|base|scale)$", r"\1.\2.\3", name)
        yield name, raw, tensor
    for layer, vals in conv.items():
        if set(vals) == {"q", "k", "v"}:
            yield f"model.layers.{layer}.self_attn.conv1d.weight", "", torch.cat((vals["q"], vals["k"], vals["v"]), 0)
    for layer, vals in gates.items():
        if set(vals) == {"a", "b"}:
            yield f"model.layers.{layer}.self_attn.g_proj.weight", "", vals["b"] @ vals["a"]



class Glm5NextForConditionalGenerationNextN(DeepSeekMTP):
    pass
