import pytest
import torch

from dlengine.kernel.triton.generic.k3_causal_conv import k3_causal_conv_update
from dlengine.kernel.triton.generic.k3_output_norm import k3_output_norm


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


@torch.inference_mode()
def test_k3_causal_conv_updates_indexed_state_and_is_graph_safe():
    batch, dim, width, slots_count = 4, 768, 4, 9
    x = torch.randn(batch, dim, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(dim, width, device="cuda", dtype=torch.float32)
    slots = torch.tensor([7, 2, 8, 1], device="cuda", dtype=torch.int32)
    pool = torch.randn(slots_count, dim, width, device="cuda", dtype=torch.bfloat16)
    expected_pool = pool.clone()
    selected = expected_pool[slots.long()].clone()
    selected[:, :, :-1] = selected[:, :, 1:].clone()
    selected[:, :, -1] = x
    expected = torch.nn.functional.silu(
        (selected.float() * weight.unsqueeze(0)).sum(-1)
    ).to(x.dtype)
    expected_pool[slots.long()] = selected

    actual = k3_causal_conv_update(x, pool, weight, slots)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-6)
    assert torch.equal(pool, expected_pool)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = k3_causal_conv_update(x, pool, weight, slots)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(replayed).all()


@torch.inference_mode()
def test_k3_output_norm_accepts_strided_gate_and_is_graph_safe():
    batch, heads, dim = 4, 12, 128
    x = torch.randn(batch, heads, dim, device="cuda", dtype=torch.bfloat16)
    backing = torch.randn(
        batch, 4 * heads * dim, device="cuda", dtype=torch.bfloat16
    )
    gate = backing[:, 3 * heads * dim :].view(batch, heads, dim)
    assert not gate.is_contiguous()
    weight = torch.randn(dim, device="cuda", dtype=torch.bfloat16)
    eps = 1e-6
    xf = x.float()
    expected = (
        xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
        * weight.float()
        * torch.sigmoid(gate.float())
    ).to(x.dtype)

    actual = k3_output_norm(x, gate, weight, eps)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=3e-2)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = k3_output_norm(x, gate, weight, eps)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(replayed, expected, rtol=2e-2, atol=3e-2)
