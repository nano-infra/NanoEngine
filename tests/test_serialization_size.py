from dlengine._rust.config import CachePlan, SchedulerConfig
from dlengine._rust.core import Scheduler
from dlengine._rust.proto import RequestIn, RunnerIn, SamplingParams


def test_protocol_batch_size_with_scheduler_state():
    sched = Scheduler(
        SchedulerConfig(
            engine_id="serialization-size",
            max_num_seqs=8,
            max_num_batched_tokens=4096,
            max_model_len=4096,
            attention_dp=1,
            group_size=2,
            num_kvcache_blocks=512,
            kvcache_block_size=64,
            cache_plan=CachePlan(1),
        )
    )
    for seq_id in range(8):
        tokens = list(range(seq_id * 128, seq_id * 128 + 256))
        payload = RequestIn(
            seq_id + 1, tokens, SamplingParams(), seq_id, None
        ).to_bytes()
        assert sched.add_request_bytes(payload) == [(seq_id + 1, len(tokens))]

    result = sched.schedule()
    batches = sched.serialize_run_batches(result.dp_group_seqs, result.is_prefill, 1)
    assert batches
    assert sum(len(batch) for batch in batches) > 0

    first = next(batch for batch in batches if len(batch) > 0)
    meta = RunnerIn.from_bytes(first).prefill(0, 2, 64, 8, 512)
    assert meta.input_ids
    assert len(meta.input_ids) == len(meta.positions)
