from __future__ import annotations

import pickle
import struct

import pytest
import torch

from nanodeploy._cpp import (
    BlockContextSlot,
    Sequence,
    SequenceMetric,
    SequenceStatus,
    deserialize,
    serialize,
)


_RAW_HEADER_BYTES = 12  # uint64 magic + uint32 version


def _serialize_raw(
    seq: Sequence,
    *,
    is_prefill: bool = True,
    sp_rank: int = -1,
    sp_size: int = -1,
) -> tuple[torch.Tensor, int]:
    buffer = torch.empty(1 << 20, dtype=torch.int8)
    data_len = serialize(
        buffer.data_ptr(),
        buffer.numel(),
        [seq],
        is_prefill,
        sp_rank,
        sp_size,
    )
    return buffer, data_len


def _raw_roundtrip(seq: Sequence) -> Sequence:
    buffer, data_len = _serialize_raw(seq)
    return deserialize(buffer.data_ptr(), data_len)[0]


def test_sequence_status_ordinals_preserve_existing_values():
    assert SequenceStatus.WAITING.value == 0
    assert SequenceStatus.RUNNING.value == 1
    assert SequenceStatus.FINISHED.value == 2
    assert SequenceStatus.TO_BE_MIGRATED.value == 3
    assert SequenceStatus.PAUSED_OFFLOAD.value == 4


def test_assigned_dp_defaults_to_unassigned_and_locks_python_seq_id():
    seq = Sequence([1, 2, 3])
    assert seq.assigned_dp == -1

    seq.seq_id = 123
    assert seq.seq_id == 123

    seq.assigned_dp = 2
    with pytest.raises(ValueError, match="cannot modify seq_id after assigned_dp is set"):
        seq.seq_id = 456
    with pytest.raises(ValueError, match="cannot modify assigned_dp after it is set"):
        seq.assigned_dp = -1
    assert seq.seq_id == 123
    assert seq.assigned_dp == 2


def test_assigned_dp_is_restored_when_active_context_is_reset():
    seq = Sequence([1])
    seq.assigned_dp = 3
    seq.active("ls-engine", 8, 4)
    assert seq.block_ctx(BlockContextSlot.ACTIVE).dp_idx == 3

    non_ls_seq = Sequence([1])
    non_ls_seq.active("legacy-engine", 2, 1)
    assert non_ls_seq.block_ctx(BlockContextSlot.ACTIVE).dp_idx == 0


def test_raw_schema_roundtrips_assigned_dp_and_paused_offload():
    seq = Sequence([1, 2, 3])
    seq.seq_id = 321
    seq.assigned_dp = 1
    seq.status = SequenceStatus.PAUSED_OFFLOAD
    seq.active("ls-offloaded", 2, 2)
    active = seq.block_ctx(BlockContextSlot.ACTIVE)
    active.master_sp_idx = -1

    restored = _raw_roundtrip(seq)

    assert restored.seq_id == 321
    assert restored.assigned_dp == 1
    assert restored.status == SequenceStatus.PAUSED_OFFLOAD
    restored_active = restored.block_ctx(BlockContextSlot.ACTIVE)
    assert restored_active.dp_idx == 1
    assert restored_active.master_sp_idx == -1
    assert list(restored_active.block_location) == []
    assert restored_active.num_dispatched_tokens == [0, 0]
    assert list(restored_active.sp_block_table[0]) == []
    assert list(restored_active.sp_block_table[1]) == []


def test_raw_decode_optimized_schema_preserves_skeleton_and_trims_only_target_heavy_fields():
    seq = Sequence([10, 11, 12], 0.25, 9, True)
    seq.seq_id = 654321
    seq.assigned_dp = 1
    seq.status = SequenceStatus.RUNNING
    seq.token_ids = [10, 11, 12, 40]
    seq.last_token = 40
    seq.num_tokens = 4
    seq.num_prompt_tokens = 3
    seq.num_checkpointed_tokens = 3
    seq.num_cached_tokens = 2
    seq.active("ls-optimized-raw", 2, 2)
    active = seq.block_ctx(BlockContextSlot.ACTIVE)
    active.master_sp_idx = 0
    active.pending_token_present = True
    active.pending_token_target_sp = 0
    active.num_dispatched_tokens = [3, 1]
    active.sp_block_table[0] = [7]
    active.sp_block_table[1] = [9]
    active.block_location.append((0, 7))
    active.block_location.append((1, 9))

    buffer, data_len = _serialize_raw(
        seq,
        is_prefill=False,
        sp_rank=1,
        sp_size=2,
    )
    restored = deserialize(buffer.data_ptr(), data_len)[0]

    assert restored.seq_id == seq.seq_id
    assert restored.status == SequenceStatus.RUNNING
    assert restored.assigned_dp == 1
    assert restored.temperature == 0.25
    assert restored.max_tokens == 9
    assert restored.ignore_eos is True
    assert restored.last_token == 40
    assert restored.num_tokens == 4
    assert restored.num_prompt_tokens == 3
    assert restored.num_checkpointed_tokens == 3
    assert restored.num_cached_tokens == 2
    assert restored.token_ids == []
    restored_active = restored.block_ctx(BlockContextSlot.ACTIVE)
    assert restored_active.engine_id == "ls-optimized-raw"
    assert restored_active.dp_idx == restored.assigned_dp == 1
    assert restored_active.master_sp_idx == 0
    assert restored_active.pending_token_present is True
    assert restored_active.pending_token_target_sp == 0
    assert restored_active.num_dispatched_tokens == [3, 1]
    assert list(restored_active.block_location) == [(1, 9)]
    assert list(restored_active.sp_block_table[0]) == []
    assert list(restored_active.sp_block_table[1]) == [9]


def test_raw_deserialize_high_id_advances_process_high_watermark():
    restored_id = 1 << 50
    seq = Sequence([1, 2, 3])
    seq.seq_id = restored_id

    restored = _raw_roundtrip(seq)

    assert restored.seq_id == restored_id
    assert Sequence([4]).seq_id > restored_id


def test_raw_schema_rejects_malformed_block_context_state():
    seq = Sequence([1, 2, 3])
    seq.active("ls-invalid-context", 2, 1)
    active = seq.block_ctx(BlockContextSlot.ACTIVE)

    active.num_dispatched_tokens = [0]
    with pytest.raises(RuntimeError, match="vector dimensions"):
        _serialize_raw(seq)

    active.reset("ls-invalid-context", 2, 1)
    active.sp_block_table[0] = [7]
    with pytest.raises(RuntimeError, match="tables and locations disagree"):
        _serialize_raw(seq)

    active.reset("ls-invalid-context", 2, 1)
    active.master_sp_idx = -1
    active.num_dispatched_tokens = [1, 0]
    with pytest.raises(RuntimeError, match="inactive BlockContext retains live state"):
        _serialize_raw(seq)


def test_pickle_schema_roundtrips_assigned_paused_sequence_complete_state():
    restored_id = 10_000_000
    seq = Sequence([101, 102, 103], 0.25, 7, True)
    seq.seq_id = restored_id
    seq.assigned_dp = 2
    seq.status = SequenceStatus.PAUSED_OFFLOAD
    seq.active("ls-pickle", 8, 4)
    seq.token_ids = [101, 102, 103, 0, 55]
    seq.last_token = 55
    seq.num_tokens = 5
    seq.num_prompt_tokens = 3
    seq.num_checkpointed_tokens = 4
    seq.num_cached_tokens = 2
    metric = SequenceMetric(restored_id, 3)
    metric.arrival_time = 1.0
    metric.first_token_time = 2.0
    metric.last_token_time = 3.0
    metric.num_generated_tokens = 2
    metric.itl_samples = [4.0, 5.0]
    seq.metric = metric

    restored = pickle.loads(pickle.dumps(seq))

    assert restored.seq_id == restored_id
    assert restored.assigned_dp == 2
    assert restored.status == SequenceStatus.PAUSED_OFFLOAD
    assert restored.token_ids == [101, 102, 103, 0, 55]
    assert restored.completion_token_ids == [0, 55]
    assert restored.last_token == 55
    assert restored.num_tokens == 5
    assert restored.num_prompt_tokens == 3
    assert restored.num_checkpointed_tokens == 4
    assert restored.num_cached_tokens == 2
    assert restored.temperature == 0.25
    assert restored.max_tokens == 7
    assert restored.ignore_eos is True
    assert restored.block_ctx(BlockContextSlot.ACTIVE).dp_idx == 2
    assert restored.metric is not None
    assert restored.metric.seq_id == restored_id
    assert restored.metric.num_prompt_tokens == 3
    assert restored.metric.num_generated_tokens == 2
    assert restored.metric.itl_samples == [4.0, 5.0]
    with pytest.raises(ValueError, match="cannot modify seq_id after assigned_dp is set"):
        restored.seq_id = restored.seq_id + 1

    # Unpickling a high ID advances the allocator rather than allowing a later
    # locally constructed Sequence to reuse the restored identity.
    assert Sequence([9]).seq_id > restored_id


def test_pickle_schema_rejects_inconsistent_complete_token_history():
    seq = Sequence([1, 2, 3])

    wrong_length = list(seq.__getstate__())
    wrong_length[7] = 4
    with pytest.raises(ValueError, match="complete token history"):
        Sequence.__new__(Sequence).__setstate__(tuple(wrong_length))

    wrong_last_token = list(seq.__getstate__())
    wrong_last_token[6] = 99
    with pytest.raises(ValueError, match="last_token"):
        Sequence.__new__(Sequence).__setstate__(tuple(wrong_last_token))


def test_pickle_schema_explicitly_rejects_legacy_magic_and_version():
    seq = Sequence([1, 2, 3])
    state = list(seq.__getstate__())

    legacy_state = tuple(state[2:])
    with pytest.raises(ValueError, match="unsupported Sequence pickle schema"):
        Sequence.__new__(Sequence).__setstate__(legacy_state)

    wrong_magic = list(state)
    wrong_magic[0] = 0
    with pytest.raises(ValueError, match="unsupported Sequence pickle schema"):
        Sequence.__new__(Sequence).__setstate__(tuple(wrong_magic))

    wrong_version = list(state)
    wrong_version[1] += 1
    with pytest.raises(ValueError, match="unsupported Sequence pickle schema"):
        Sequence.__new__(Sequence).__setstate__(tuple(wrong_version))


def test_pickle_schema_rejects_malformed_block_context_state():
    seq = Sequence([1, 2, 3])
    seq.active("ls-invalid-pickle-context", 2, 1)
    seq.block_ctx(BlockContextSlot.ACTIVE).num_dispatched_tokens = [0]

    with pytest.raises(ValueError, match="BlockContext.*vector dimensions"):
        pickle.dumps(seq)


def test_assigned_dp_context_invariant_rejects_initialized_mismatch_but_allows_detached_waiting():
    detached = Sequence([1, 2, 3])
    detached.assigned_dp = 3
    assert detached.status == SequenceStatus.WAITING
    assert detached.block_ctx(BlockContextSlot.ACTIVE).dp_idx == -1
    assert _raw_roundtrip(detached).assigned_dp == 3
    assert pickle.loads(pickle.dumps(detached)).assigned_dp == 3

    seq = Sequence([1, 2, 3])
    seq.assigned_dp = 1
    seq.active("ls-owner-invariant", 2, 2)
    valid_buffer, valid_len = _serialize_raw(seq)
    active = seq.block_ctx(BlockContextSlot.ACTIVE)
    active.dp_idx = 0

    with pytest.raises(RuntimeError, match="assigned_dp.*ACTIVE BlockContext"):
        _serialize_raw(seq)
    with pytest.raises(ValueError, match="assigned_dp.*ACTIVE BlockContext"):
        pickle.dumps(seq)

    # A hostile raw payload can disagree even when the producer serialized a
    # valid object. assigned_dp follows seq_id/status in the fixed raw skeleton.
    assigned_dp_offset = _RAW_HEADER_BYTES + struct.calcsize("P") + 8 + 4
    valid_buffer[assigned_dp_offset : assigned_dp_offset + 4] = 0
    with pytest.raises(RuntimeError, match="assigned_dp.*ACTIVE BlockContext"):
        deserialize(valid_buffer.data_ptr(), valid_len)

    # Exercise the pickle load boundary independently of __getstate__.
    active.dp_idx = 1
    invalid_state = list(seq.__getstate__())
    invalid_state[11][0].dp_idx = 0
    with pytest.raises(ValueError, match="assigned_dp.*ACTIVE BlockContext"):
        Sequence.__new__(Sequence).__setstate__(tuple(invalid_state))


def test_raw_schema_rejects_legacy_mixed_and_trailing_payloads():
    seq = Sequence([1, 2, 3])
    buffer, data_len = _serialize_raw(seq)

    # A legacy payload starts directly at the sequence count, i.e. immediately
    # after the new fixed-size header.
    with pytest.raises(RuntimeError, match="legacy or invalid.*magic"):
        deserialize(
            buffer.data_ptr() + _RAW_HEADER_BYTES,
            data_len - _RAW_HEADER_BYTES,
        )

    mixed_version = buffer.clone()
    mixed_version[8:12] = 0
    with pytest.raises(RuntimeError, match="payload version"):
        deserialize(mixed_version.data_ptr(), data_len)

    with pytest.raises(RuntimeError, match="Trailing bytes"):
        deserialize(buffer.data_ptr(), data_len + 1)


def test_raw_schema_rejects_invalid_status_ordinal():
    seq = Sequence([1, 2, 3])
    buffer, data_len = _serialize_raw(seq)
    status_offset = _RAW_HEADER_BYTES + struct.calcsize("P") + 8
    buffer[status_offset : status_offset + 4] = 0x7F

    with pytest.raises(RuntimeError, match="SequenceStatus ordinal"):
        deserialize(buffer.data_ptr(), data_len)
