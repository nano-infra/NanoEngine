import pytest
import torch

from dlengine.runtime.kernel.jit.sgl.attn_res import fused_attention_residual_tma

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="Blackwell CUDA device required",
)

_HIDDEN = 7168
_EPS = 1e-6


def _inputs(tokens):
    torch.manual_seed(42)
    prefix = torch.randn(tokens, _HIDDEN, device="cuda", dtype=torch.bfloat16)
    bank = torch.randn(tokens, 8, _HIDDEN, device="cuda", dtype=torch.bfloat16)
    cw = (torch.randn(_HIDDEN, device="cuda") / _HIDDEN**0.5).bfloat16()
    ow = torch.randn(_HIDDEN, device="cuda", dtype=torch.bfloat16)
    return prefix, bank, cw, ow


def _reference(prefix, bank, valid, cw, ow):
    candidates = torch.cat((bank[:, :valid], prefix[:, None]), dim=1).float()
    scores = (candidates * cw.float()).sum(-1) * torch.rsqrt(
        candidates.square().mean(-1) + _EPS
    )
    mixed = (scores.softmax(-1)[..., None] * candidates).sum(1)
    return (
        mixed * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + _EPS) * ow.float()
    ).bfloat16()


def _assert_close(actual, expected):
    # Allow one BF16 ULP across fused and reference reduction orders.
    torch.testing.assert_close(
        actual, expected, rtol=torch.finfo(torch.bfloat16).eps, atol=2e-3
    )


@torch.inference_mode()
@pytest.mark.parametrize("valid", range(1, 9))
@pytest.mark.parametrize("tokens", [1, 7])
def test_attn_res_tma_matches_reference_and_writes_bank(valid, tokens):
    pytest.importorskip("tvm_ffi.cpp")
    prefix, bank, cw, ow = _inputs(tokens)
    expected = _reference(prefix, bank, valid, cw, ow)
    for write in ((False, True) if valid < 8 else (False,)):
        actual = fused_attention_residual_tma(
            prefix, bank, valid, cw, ow, _EPS, write_prefix=write
        )
        _assert_close(actual, expected)
        if write:
            torch.testing.assert_close(bank[:, valid], prefix, rtol=0, atol=0)


@torch.inference_mode()
def test_attn_res_tma_cuda_graph_replays_updated_inputs():
    pytest.importorskip("tvm_ffi.cpp")
    prefix, bank, cw, ow = _inputs(7)
    valid = 5
    fused_attention_residual_tma(prefix, bank, valid, cw, ow, _EPS)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = fused_attention_residual_tma(prefix, bank, valid, cw, ow, _EPS)
    for _ in range(2):
        prefix.normal_()
        expected = _reference(prefix, bank, valid, cw, ow)
        graph.replay()
        _assert_close(actual, expected)
