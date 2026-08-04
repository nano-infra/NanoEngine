from dlengine._rust.config import CachePlan, SchedulerConfig
from dlengine._rust.core import Scheduler
from dlengine._rust.proto import RequestIn, RunnerIn, SamplingParams
from dlengine.engine.dlslime_protocol import decode_run_result, encode_run_result


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


def test_json_schema_reaches_worker_batch_aux():
    sched = _scheduler()
    schema = '{"type":"object"}'
    payload = RequestIn(
        1000,
        [101, 102],
        SamplingParams(json_schema=schema),
        0,
        None,
    ).to_bytes()
    sched.add_request_bytes(payload)

    result = sched.schedule()
    batch = sched.serialize_run_batches_for_result(result, 1)[0]
    aux = RunnerIn.from_bytes(batch).aux()

    assert aux.seq_ids == [1000]
    assert aux.json_schemas == [schema]


def test_structural_tag_reaches_worker_batch_aux():
    sched = _scheduler()
    structural_tag = '{"type":"structural_tag","format":{"type":"any_text"}}'
    payload = RequestIn(
        1001,
        [101, 102],
        SamplingParams(structural_tag=structural_tag),
        0,
        None,
    ).to_bytes()
    sched.add_request_bytes(payload)

    result = sched.schedule()
    batch = sched.serialize_run_batches_for_result(result, 1)[0]
    aux = RunnerIn.from_bytes(batch).aux()

    assert aux.seq_ids == [1001]
    assert aux.structural_tags == [structural_tag]


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
