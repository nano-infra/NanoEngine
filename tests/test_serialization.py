import ctypes

import pytest
from dlengine._rust.config import CachePlan, SchedulerConfig
from dlengine._rust.core import Scheduler
from dlengine._rust.proto import RequestIn, RunnerIn, SamplingParams
from dlengine.engine.dlslime_protocol import (
    decode_run_request,
    decode_run_result,
    encode_run_request,
    encode_run_result,
)


@pytest.mark.parametrize("is_prefill", [False, True])
def test_run_request_phase_envelope(is_prefill):
    encoded = encode_run_request(b"\x00LDMD\x00", is_prefill)
    buffer = ctypes.create_string_buffer(encoded)

    payload, decoded_is_prefill = decode_run_request(
        ctypes.addressof(buffer), len(encoded)
    )

    assert payload == b"\x00LDMD\x00"
    assert decoded_is_prefill is is_prefill


def test_run_request_rejects_missing_or_invalid_phase():
    with pytest.raises(ValueError, match="missing its phase byte"):
        decode_run_request(0, 0)

    buffer = ctypes.create_string_buffer(b"\x02payload")
    with pytest.raises(ValueError, match="invalid run request phase byte 2"):
        decode_run_request(ctypes.addressof(buffer), 8)


def _scheduler() -> Scheduler:
    return Scheduler(
        SchedulerConfig(
            engine_id="serialization-test",
            max_num_seqs=4,
            max_num_batched_tokens=16,
            max_model_len=64,
            attention_dp=1,
            group_size=1,
            num_kvcache_blocks=16,
            kvcache_block_size=4,
            cache_plan=CachePlan(1),
        )
    )


def test_request_to_prefill_batch_protocol():
    sched = _scheduler()
    payload = RequestIn(999, [101, 102, 103], SamplingParams(), 0, None).to_bytes()
    assert sched.add_request_bytes(payload) == [(999, 3)]

    result = sched.schedule()
    batch = sched.serialize_run_batches_for_result(result, 1)[0]
    meta = RunnerIn.from_bytes(batch).prefill(0, 1, 4, 4, 16)
    assert meta.input_ids == [101, 102, 103]
    assert meta.positions == [0, 1, 2]


def test_decode_batch_and_run_result_protocol():
    sched = _scheduler()
    payload = RequestIn(12345, [1, 2, 3], SamplingParams(), 0, None).to_bytes()
    sched.add_request_bytes(payload)

    result = sched.schedule()
    sched.postprocess(result.filtered_dp_group_seq_ids, [[[4]]], None, None)

    result = sched.schedule()
    batch = sched.serialize_run_batches_for_result(result, 1)[0]
    meta = RunnerIn.from_bytes(batch).decode(0, 1, 4, 4, 16)
    assert meta.input_ids == [4]
    assert meta.positions == [3]

    encoded = encode_run_result([[5]], 0)
    token_ids, logprobs = decode_run_result(encoded)
    assert token_ids == [[5]]
    assert logprobs is None
