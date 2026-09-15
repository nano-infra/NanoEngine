import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="the fused MoE gate requires a Hopper GPU",
)


@pytest.mark.parametrize("num_tokens", [1, 16, 513])
def test_moe_fused_gate_matches_single_group_reference(num_tokens):
    pytest.importorskip("tvm_ffi")
    from dlengine.runtime.kernel.jit.sgl.moe_fused_gate import moe_fused_gate

    torch.manual_seed(0)
    logits = torch.randn(num_tokens, 256, device="cuda", dtype=torch.float32)
    bias = torch.randn(256, device="cuda", dtype=torch.float32) * 0.1

    indices, weights = moe_fused_gate(
        logits,
        bias,
        topk=8,
        scoring_func="sigmoid",
        renormalize=True,
        routed_scaling_factor=2.5,
    )

    scores = logits.sigmoid()
    reference_indices = torch.topk(scores + bias, 8, dim=-1, sorted=False).indices
    reference_weights = scores.gather(1, reference_indices)
    reference_weights = (
        reference_weights / reference_weights.sum(dim=-1, keepdim=True) * 2.5
    )

    torch.testing.assert_close(
        indices.sort(dim=-1).values,
        reference_indices.to(torch.int32).sort(dim=-1).values,
        rtol=0,
        atol=0,
    )
    sorted_weights = weights.gather(1, indices.argsort(dim=-1))
    sorted_reference_weights = reference_weights.gather(
        1, reference_indices.argsort(dim=-1)
    )
    torch.testing.assert_close(
        sorted_weights, sorted_reference_weights, rtol=1e-6, atol=1e-6
    )
