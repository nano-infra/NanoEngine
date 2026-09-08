import pytest
import torch
import torch.nn.functional as F


def _reference(x, weight, state, cu_seqlens, slots, has_initial_state):
    output = torch.empty_like(x)
    expected_state = state.clone()
    width = weight.shape[1]
    for seq_idx, slot in enumerate(slots.tolist()):
        start = int(cu_seqlens[seq_idx])
        end = int(cu_seqlens[seq_idx + 1])
        prefix = (
            state[slot].float()
            if bool(has_initial_state[seq_idx])
            else torch.zeros_like(state[slot], dtype=torch.float32)
        )
        sequence = x[:, start:end].T.float()
        history = torch.cat((prefix.T, sequence), dim=0)
        values = []
        for token_idx in range(sequence.shape[0]):
            window = history[token_idx : token_idx + width].T
            values.append(F.silu((window * weight.float()).sum(dim=1)))
        output[:, start:end] = torch.stack(values).T.to(x.dtype)
        expected_state[slot] = history[-(width - 1) :].T.to(state.dtype)
    return output, expected_state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("lengths", ([5, 3], [1, 2]))
def test_k3_ragged_causal_conv_matches_reference(lengths):
    from dlengine.runtime.kernel.triton.generic.k3_causal_conv1d_prefill import (
        causal_conv1d_fn,
    )

    torch.manual_seed(7)
    dim, width = 257, 4
    total = sum(lengths)
    x = torch.randn(total, dim, device="cuda", dtype=torch.bfloat16).T
    weight = torch.randn(dim, width, device="cuda", dtype=torch.float32)
    state = torch.randn(4, dim, width - 1, device="cuda", dtype=torch.bfloat16)
    initial_state = state.clone()
    cu_seqlens = torch.tensor([0, lengths[0], total], device="cuda", dtype=torch.int32)
    slots = torch.tensor([1, 3], device="cuda", dtype=torch.int32)
    has_initial_state = torch.tensor([False, True], device="cuda")

    expected_output, expected_state = _reference(
        x, weight, initial_state, cu_seqlens, slots, has_initial_state
    )
    output = causal_conv1d_fn(
        x,
        weight,
        None,
        conv_states=state,
        query_start_loc=cu_seqlens,
        seq_lens_cpu=lengths,
        cache_indices=slots,
        has_initial_state=has_initial_state,
        activation="silu",
        validate_data=True,
    )

    torch.testing.assert_close(output, expected_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
