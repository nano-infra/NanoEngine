"""Public scheduler protocol boundary sanity checks.

Python submits request bytes and receives protocol batches, while sequence
state remains owned by the Rust scheduler.
"""

from dlengine._rust.config import CachePlan, SchedulerConfig
from dlengine._rust.core import Scheduler
from dlengine._rust.proto import RequestIn, RunnerIn, SamplingParams


def test_python_uses_protocol_not_sequence_container_proxies():
    sched = Scheduler(
        SchedulerConfig(
            engine_id="proxy-boundary-test",
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
    payload = RequestIn(1, [7, 8, 9], SamplingParams(), 0, None).to_bytes()
    assert sched.add_request_bytes(payload) == [(1, 3)]

    result = sched.schedule()
    batch = sched.serialize_run_batches_for_result(result, 1)[0]
    meta = RunnerIn.from_bytes(batch).prefill(0, 1, 4, 2, 8)
    assert meta.input_ids == [7, 8, 9]
