import pytest
import torch
from dlengine.runtime.kernel.triton.generic.fused_topk import fused_sigmoid_biased_topk
from dlengine.runtime.kernel.triton.generic.rmsnorm_gated import (
    sigmoid_rms_norm_gated_triton,
)
from dlengine.runtime.kernel.triton.generic.sigmoid_mul import sigmoid_mul_triton
from dlengine.runtime.kernel.triton.generic.situ import situ_and_mul_triton

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


@torch.inference_mode()
def test_sigmoid_rms_norm_gated_matches_reference():
    x = torch.randn(7, 128, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(x)
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    eps = 1e-6

    actual = sigmoid_rms_norm_gated_triton(x, gate, weight, eps)
    xf = x.float()
    expected = (
        xf
        * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
        * weight.float()
        * torch.sigmoid(gate.float())
    ).to(x.dtype)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=3e-2)


@torch.inference_mode()
@pytest.mark.parametrize("linear_beta", [None, 7.0])
def test_situ_and_mul_matches_reference(linear_beta):
    x = torch.randn(5, 64, device="cuda", dtype=torch.bfloat16)
    beta = 3.0
    gate, up = x.float().chunk(2, dim=-1)
    expected_gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    expected_up = (
        up if linear_beta is None else linear_beta * torch.tanh(up / linear_beta)
    )
    expected = (expected_gate * expected_up).to(x.dtype)

    actual = situ_and_mul_triton(x, beta=beta, linear_beta=linear_beta)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@torch.inference_mode()
def test_sigmoid_mul_is_cuda_graph_safe():
    x = torch.randn(4, 96, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(x)
    sigmoid_mul_triton(x, gate)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = sigmoid_mul_triton(x, gate)
    graph.replay()
    torch.cuda.synchronize()

    expected = (x.float() * torch.sigmoid(gate.float())).to(x.dtype)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@torch.inference_mode()
@pytest.mark.parametrize("renormalize", [False, True])
def test_fused_sigmoid_biased_topk_matches_reference(renormalize):
    logits = torch.randn(11, 64, device="cuda", dtype=torch.float32)
    bias = torch.randn(64, device="cuda", dtype=torch.float32) * 0.1
    top_k = 8

    actual_weights, actual_ids = fused_sigmoid_biased_topk(
        logits, bias, top_k, renormalize
    )
    probabilities = torch.sigmoid(logits)
    expected_ids = torch.topk(probabilities + bias, top_k, dim=-1).indices
    expected_weights = torch.gather(probabilities, 1, expected_ids)
    if renormalize:
        expected_weights /= expected_weights.sum(dim=-1, keepdim=True)

    assert torch.equal(actual_ids, expected_ids)
    torch.testing.assert_close(actual_weights, expected_weights)


@torch.inference_mode()
def test_fused_sigmoid_biased_topk_keeps_dummy_nan_ids_in_range():
    logits = torch.full((8, 896), float("nan"), device="cuda")
    bias = torch.zeros(896, device="cuda")

    _, ids = fused_sigmoid_biased_topk(logits, bias, top_k=16, renormalize=True)
    torch.cuda.synchronize()

    assert torch.all((ids >= 0) & (ids < logits.shape[1]))
