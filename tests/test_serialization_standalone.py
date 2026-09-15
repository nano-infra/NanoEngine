from dlengine._rust.config import CachePlan, SchedulerConfig
from dlengine._rust.core import Scheduler
from dlengine._rust.proto import RequestIn, RunnerIn, SamplingParams
from dlengine.executor.dlslime_protocol import decode_run_result, encode_run_result


def test_standalone_rust_protocol_serialization():
    sched = Scheduler(
        SchedulerConfig(
            engine_id="serialization-standalone",
            max_num_seqs=2,
            max_num_batched_tokens=8,
            max_model_len=32,
            attention_dp=1,
            group_size=1,
            num_kvcache_blocks=8,
            kvcache_block_size=4,
            cache_plan=CachePlan(1),
        )
    )
    payload = RequestIn(999, [101, 202, 303, 404], SamplingParams(), 0, None).to_bytes()
    assert sched.add_request_bytes(payload) == [(999, 4)]

    result = sched.schedule()
    batch = sched.serialize_run_batches_for_result(result, 1)[0]
    meta = RunnerIn.from_bytes(batch).prefill(0, 1, 4, 2, 8)
    assert meta.input_ids == [101, 202, 303, 404]
    assert meta.positions == [0, 1, 2, 3]

    encoded = encode_run_result([[42]], 123)
    token_ids, logprobs = decode_run_result(encoded)
    assert token_ids == [[42]]
    assert logprobs is None
