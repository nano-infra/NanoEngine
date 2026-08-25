import pytest
import torch

from nanodeploy.worker.token_carry import GpuTokenCarry


def test_gpu_token_carry_follows_request_identity_across_row_reordering():
    carry = GpuTokenCarry()
    q0_keys = ((10, 0), (20, 3), (30, -1))
    q0_payload = torch.tensor([101, 201, 301])

    selected, hits = carry.select_inputs(
        q0_payload,
        q0_keys,
        wave_id=1,
        quantum_id=0,
    )
    assert selected is q0_payload
    assert hits == 0
    carry.record(
        torch.tensor([102, 202, 302]),
        q0_keys,
        wave_id=1,
        quantum_id=0,
    )

    q1_payload = torch.tensor([999, 888, 777])
    selected, hits = carry.select_inputs(
        q1_payload,
        ((30, -1), (10, 0), (40, 0)),
        wave_id=1,
        quantum_id=1,
    )

    assert selected.tolist() == [302, 102, 777]
    assert hits == 2
    assert q1_payload.tolist() == [999, 888, 777]


def test_gpu_token_carry_rejects_stale_epoch_and_nonconsecutive_quantum():
    carry = GpuTokenCarry()
    carry.record(
        torch.tensor([42]),
        ((10, 0),),
        wave_id=2,
        quantum_id=4,
    )

    changed_epoch, changed_hits = carry.select_inputs(
        torch.tensor([7]),
        ((10, 1),),
        wave_id=2,
        quantum_id=5,
    )
    skipped_quantum, skipped_hits = carry.select_inputs(
        torch.tensor([8]),
        ((10, 0),),
        wave_id=2,
        quantum_id=6,
    )

    assert changed_epoch.tolist() == [7]
    assert changed_hits == 0
    assert skipped_quantum.tolist() == [8]
    assert skipped_hits == 0


@pytest.mark.parametrize(
    ("tokens", "keys", "message"),
    [
        (torch.tensor([[1]]), ((1, 0),), "one-dimensional"),
        (torch.tensor([1, 2]), ((1, 0),), "row count"),
        (torch.tensor([1, 2]), ((1, 0), (1, 0)), "unique"),
    ],
)
def test_gpu_token_carry_validates_row_contract(tokens, keys, message):
    carry = GpuTokenCarry()

    with pytest.raises(ValueError, match=message):
        carry.record(tokens, keys, wave_id=1, quantum_id=0)
