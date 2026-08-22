from types import SimpleNamespace

import dlengine.runtime.layers.generic.gated_delta_net as gdn_module
import torch
from dlengine.runtime.layers.generic.gated_delta_net import GenericGatedDeltaNet
from torch import nn


def test_nontranspose_decode_fallback_updates_v_major_state_pool(monkeypatch):
    layer = GenericGatedDeltaNet.__new__(GenericGatedDeltaNet)
    nn.Module.__init__(layer)
    layer.layer_idx = 0
    layer._has_flashinfer_pretranspose = False
    layer._has_flashinfer_nontranspose = True
    layer._has_fla = False
    layer.A_log = nn.Parameter(torch.zeros(2))
    layer.dt_bias = nn.Parameter(torch.zeros(2))

    state_pool = torch.arange(1 * 3 * 2 * 4 * 3, dtype=torch.float32).reshape(
        1, 3, 2, 4, 3
    )
    original = state_pool.clone()
    slots = torch.tensor([2, 0])
    context = SimpleNamespace(
        gdn_recurrent_states=state_pool,
        gdn_state_slots=slots,
    )

    def fake_nontranspose_decode(**kwargs):
        state = kwargs["state"]
        expected = original[0, slots].transpose(-1, -2)
        assert torch.equal(state, expected)
        state.add_(10)
        batch, _, heads, _ = kwargs["v"].shape
        output = torch.zeros(batch, 1, heads, 4)
        return output, state

    monkeypatch.setattr(gdn_module, "gated_delta_rule_decode", fake_nontranspose_decode)

    output = layer._gdn_decode(
        q=torch.zeros(2, 2, 3),
        k=torch.zeros(2, 2, 3),
        v=torch.zeros(2, 2, 4),
        a=torch.zeros(2, 2),
        b=torch.zeros(2, 2),
        scale=1.0,
        context=context,
    )

    assert output.shape == (2, 2, 4)
    assert torch.equal(state_pool[0, slots], original[0, slots] + 10)
    assert torch.equal(state_pool[0, 1], original[0, 1])
