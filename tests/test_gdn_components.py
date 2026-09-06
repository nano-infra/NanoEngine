"""Component-level tests for the decomposed GDN implementation.

These exercise the extracted mixins in isolation (no full model, no GPU
kernels) and assert that the naive PyTorch reference recurrence is internally
consistent between its prefill (chunked scan) and decode (single-step) paths —
the reference-vs-reference numerical anchor the kernel backends must match.
"""

from types import SimpleNamespace

import torch
from torch import nn

from dlengine.runtime.layers.backends.delta_net.generic import GenericGatedDeltaNet
from dlengine.runtime.layers.backends.delta_net.components.output import RMSNormGated
from dlengine.runtime.layers.backends.delta_net.components.state import StateMixin


def _bare_layer(num_v_heads=2, head_v_dim=4, head_k_dim=3):
    """Construct a GenericGatedDeltaNet shell without running __init__."""
    layer = GenericGatedDeltaNet.__new__(GenericGatedDeltaNet)
    nn.Module.__init__(layer)
    layer.layer_idx = 0
    layer.num_k_heads = num_v_heads
    layer.num_v_heads = num_v_heads
    layer.head_k_dim = head_k_dim
    layer.head_v_dim = head_v_dim
    layer.A_log = nn.Parameter(torch.zeros(num_v_heads))
    layer.dt_bias = nn.Parameter(torch.zeros(num_v_heads))
    # GenericGatedDeltaNet is the naive reference backend (NaiveRecurrenceMixin),
    # so prefill/decode already use the pure-PyTorch scan; no flags to set.
    return layer


def test_state_continuation_keep_mask_marks_only_continuations():
    context = SimpleNamespace(
        cu_seqlens_q=torch.tensor([0, 3, 5]),
        cu_seqlens_k=torch.tensor([0, 3, 10]),  # seq0 fresh, seq1 has cache
    )
    keep = StateMixin._continuation_keep_mask(context, num_seqs=2, dtype=torch.float32)
    assert torch.equal(keep, torch.tensor([0.0, 1.0]))


def test_rms_norm_gated_matches_manual_reference():
    torch.manual_seed(0)
    norm = RMSNormGated(hidden_size=8, eps=1e-6)
    x = torch.randn(5, 8)
    gate = torch.randn(5, 8)

    out = norm(x, gate)

    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    ref = norm.weight * (xf * torch.rsqrt(var + norm.eps)).to(x.dtype)
    ref = ref * torch.nn.functional.silu(gate.float()).to(x.dtype)
    assert torch.allclose(out, ref, atol=1e-5)


def test_naive_prefill_and_decode_recurrence_agree():
    """A single-token prefill scan must equal one decode step from zero state.

    This pins the reference recurrence: running the chunked-prefill scan over a
    length-1 sequence produces the same output and final state as taking one
    naive decode step, so kernel backends can be compared against either path.
    """
    torch.manual_seed(0)
    layer = _bare_layer()
    H, V, K = layer.num_v_heads, layer.head_v_dim, layer.head_k_dim
    scale = K**-0.5

    q = torch.randn(1, H, K)
    k = torch.randn(1, H, K)
    v = torch.randn(1, H, V)
    g = torch.zeros(1, H)  # log-decay 0 -> decay factor 1
    beta = torch.rand(1, H)

    cu = torch.tensor([0, 1])
    prefill_out, prefill_state = layer._naive_gdn_prefill(
        q, k, v, g, beta, scale, cu, initial_state=None
    )

    zero_state = torch.zeros(1, H, V, K)
    decode_out, decode_state = layer._naive_gdn_decode(
        q[:1], k[:1], v[:1], g[:1], beta[:1], scale, zero_state
    )

    assert torch.allclose(prefill_out, decode_out.squeeze(0), atol=1e-5)
    assert torch.allclose(prefill_state, decode_state, atol=1e-5)
