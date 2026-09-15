import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_kimi_k3_triton_residual_matches_reference():
    from dlengine.runtime.kernel.triton.kimi_k3 import fused_attention_residual

    torch.manual_seed(0)
    tokens, hidden, valid = 2, 7168, 5
    prefix = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    bank = torch.randn(tokens, 8, hidden, device="cuda", dtype=torch.bfloat16)
    combined_weight = torch.randn(hidden, device="cuda", dtype=torch.float32)
    actual = fused_attention_residual(prefix, bank, valid, combined_weight, 1e-6)
    candidates = torch.cat((bank[:, :valid], prefix[:, None]), dim=1).float()
    scores = (candidates * combined_weight).sum(-1) * torch.rsqrt(
        candidates.square().mean(-1) + 1e-6
    )
    expected = (scores.softmax(-1)[..., None] * candidates).sum(1).bfloat16()
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
