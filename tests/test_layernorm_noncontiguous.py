import torch

from nanodeploy.layers.layernorm import RMSNorm


def test_rms_norm_accepts_noncontiguous_last_dimension_view() -> None:
    source = torch.randn(7, 13, dtype=torch.float32)
    x = source[:, :8]
    assert not x.is_contiguous()

    layer = RMSNorm(8, eps=1e-6)
    layer.weight.data.copy_(torch.linspace(0.5, 1.5, 8))

    expected = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)
    expected = expected * layer.weight
    actual = layer(x)

    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected)
