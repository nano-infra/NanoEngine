import types
import torch

from dlengine.runtime.models.glm5_next.glm5_next import (
    Glm5NextHCProjector,
    _hc_post,
    _normalize_glm5_weights,
)


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
        ("model.layers.0.self_attn.q_conv1d.weight", "", q),
        ("model.layers.0.self_attn.k_conv1d.weight", "", k),
        ("model.layers.0.self_attn.v_conv1d.weight", "", v),
        ("model.layers.0.self_attn.g_a_proj.weight", "", ga),
        ("model.layers.0.self_attn.g_b_proj.weight", "", gb),
        ("model.layers.0.hc_attn_fn", "", torch.empty(24, 12)),
    ]
    out = list(_normalize_glm5_weights(iter(src)))
    names = {name: tensor for name, _, tensor in out}
    assert names["model.layers.0.self_attn.conv1d.weight"].shape == (24, 2, 3)
    assert torch.allclose(names["model.layers.0.self_attn.g_proj.weight"], gb @ ga)
    assert "model.layers.0.hc_attn.fn" in names
