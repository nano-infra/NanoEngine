import json
import types

import pytest
import torch

from dlengine.runtime.models.glm5_next.glm5_next import (
    Glm5NextHCProjector,
    Glm5NextHyperHead,
    _hc_post,
    _normalize_glm5_weights,
)
from dlengine.runtime.context.cache.plan import glm5_next_cache_plan


def test_glm5_layer_schedule_from_checkpoint_config():
    cfg = types.SimpleNamespace()
    cfg.layer_types = ["linear_attention" if i % 4 != 3 else "deepseek_sparse_attention" for i in range(45)]
    assert sum(x == "linear_attention" for x in cfg.layer_types) == 34
    assert sum(x == "deepseek_sparse_attention" for x in cfg.layer_types) == 11


def test_mhc_projector_matches_reference_formula():
    torch.manual_seed(0)
    h, mult, tokens = 3, 4, 2
    p = Glm5NextHCProjector(h, mult, 3, 1e-6)
    x = torch.randn(tokens, mult, h, dtype=torch.bfloat16)
    y, post, comb = p(x)
    flat = x.flatten(1).float()
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + p.eps)
    mixes = torch.nn.functional.linear(flat, p.fn) * rsqrt
    pre = torch.sigmoid(mixes[:, :mult] * p.scale[0] + p.base[:mult]) + p.eps
    ref_y = torch.sum(pre.unsqueeze(-1) * x, dim=1).to(torch.bfloat16)
    assert torch.allclose(y, ref_y, atol=2e-2, rtol=2e-2)
    assert post.shape == (tokens, mult)
    assert comb.shape == (tokens, mult, mult)


def test_mhc_post_uses_transposed_combination_like_reference():
    x = torch.tensor([[1.0, 2.0]])
    residual = torch.tensor([[[10.0], [20.0]]])
    post = torch.ones(1, 2)
    comb = torch.tensor([[[0.25, 0.75], [0.6, 0.4]]])
    out = _hc_post(x, residual, post, comb)
    expected = post.unsqueeze(-1) * x + torch.einsum("tij,tjd->tid", comb.transpose(1, 2), residual)
    assert torch.equal(out, expected)


def test_glm_kda_weight_normalization_is_streaming_and_shape_correct():
    q = torch.randn(8, 2, 3)
    k = torch.randn(8, 2, 3)
    v = torch.randn(8, 2, 3)
    ga = torch.randn(2, 3)
    gb = torch.randn(8, 2)
    src = [
        ("model.language_model.layers.0.self_attn.q_conv1d.weight", "", q),
        ("model.language_model.layers.0.self_attn.k_conv1d.weight", "", k),
        ("model.language_model.layers.0.self_attn.v_conv1d.weight", "", v),
        ("model.language_model.layers.0.self_attn.g_a_proj.weight", "", ga),
        ("model.language_model.layers.0.self_attn.g_b_proj.weight", "", gb),
        ("model.language_model.layers.0.hc_attn_fn", "", torch.empty(24, 12)),
    ]
    out = list(_normalize_glm5_weights(iter(src)))
    names = {name: tensor for name, _, tensor in out}
    assert names["model.layers.0.self_attn.conv1d.weight"].shape == (24, 2, 3)
    assert torch.allclose(names["model.layers.0.self_attn.g_proj.weight"], gb @ ga)
    assert "model.layers.0.hc_attn.fn" in names


def test_glm_final_hyper_head_is_unweighted_mean():
    head = Glm5NextHyperHead()
    streams = torch.arange(2 * 3 * 4 * 1, dtype=torch.float32).view(2, 3, 4, 1)
    out = head(streams)
    assert out.shape == (2, 3, 1)
    assert torch.equal(out, streams.mean(dim=2))
    flat_streams = streams.view(6, 4, 1)
    assert torch.equal(head(flat_streams), flat_streams.mean(dim=1))


def test_glm_cache_plan_includes_mla_gdn_and_indexer():
    plan = glm5_next_cache_plan()
    assert plan.has_mla()
    assert plan.has_gdn()
    assert plan.has_indexer()


def _write_minimal_glm_config(tmp_path):
    config = {
        "model_type": "glm5_next",
        "architectures": ["Glm5NextForConditionalGeneration"],
        "hidden_size": 8, "num_hidden_layers": 1,
        "num_attention_heads": 1, "num_key_value_heads": 1,
        "vocab_size": 32, "max_position_embeddings": 16,
        "layer_types": ["linear_attention"],
        "mlp_layer_types": ["dense"],
        "index_head_dim": 2, "index_n_heads": 1,
        "index_topk": 4, "index_kpool": 2,
        "n_routed_experts": 8, "num_experts_per_tok": 1,
        "num_nextn_predict_layers": 1,
        "kv_lora_rank": 2, "qk_rope_head_dim": 0,
        "qk_nope_head_dim": 2, "v_head_dim": 2, "q_lora_rank": 2,
        "n_shared_experts": 1, "moe_intermediate_size": 4,
        "intermediate_size": 4, "hidden_act": "silu",
        "rms_norm_eps": 1e-6, "attention_bias": False,
        "linear_attn_config": {"num_heads": 1, "head_dim": 2, "short_conv_kernel_size": 2},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return tmp_path


def test_glm_config_accepts_target_dp_ep_and_mtp_range(tmp_path, monkeypatch):
    from dlengine import config as config_module

    monkeypatch.setattr(
        config_module.AutoConfig, "from_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("unknown model")),
    )
    path = _write_minimal_glm_config(tmp_path)
    for steps in range(0, 6):
        cfg = config_module.Config(
            model=str(path), attention_dp=4, ffn_ep=4,
            num_speculative_tokens=steps, max_num_seqs=1,
        )
        assert cfg.kvcache_block_size == 64


def test_glm_config_rejects_mismatched_dp_ep_and_mtp_overflow(tmp_path, monkeypatch):
    from dlengine import config as config_module

    monkeypatch.setattr(
        config_module.AutoConfig, "from_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("unknown model")),
    )
    path = _write_minimal_glm_config(tmp_path)
    with pytest.raises(Exception, match="attention_dp must equal ffn_ep"):
        config_module.Config(model=str(path), attention_dp=2, ffn_ep=4)
    with pytest.raises(Exception, match="at most 5"):
        config_module.Config(
            model=str(path), attention_dp=4, ffn_ep=4, num_speculative_tokens=6
        )
