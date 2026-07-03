import ctypes
import math

from dlengine._cpp import (
    CachePlan,
    decode_run_result,
    encode_run_result,
    parse_migrate_batch,
    prepare_decode_from_bytes,
    prepare_prefill_from_bytes,
    SamplingParams,
    Scheduler,
    SchedulerConfig,
    Sequence,
    SequenceMetric,
    serialize,
    serialize_migrate_batch,
    serialize_run_batch,
)


def _scheduler() -> Scheduler:
    return Scheduler(
        SchedulerConfig(
            engine_id="strong-type-test",
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


def test_scheduler_sequence_metric_and_wire_roundtrip():
    sched = _scheduler()
    seq = Sequence(
        [1, 2, 3], SamplingParams(max_tokens=2, return_completion_logprobs=True)
    )
    seq.metric = SequenceMetric(seq.seq_id, seq.num_prompt_tokens)

    sched.add(seq)
    result = sched.schedule()
    assert result.is_prefill
    assert len(result.dp_seqs) == 1
    assert len(result.dp_seqs[0]) == 1

    batch = serialize_run_batch(result.dp_seqs[0], True)
    meta = prepare_prefill_from_bytes(batch, 0, 1, 4, 4, 16)
    assert meta.input_ids == [1, 2, 3]
    assert meta.positions == [0, 1, 2]

    sched.postprocess(result.dp_group_seqs, [[[10]]], None, [[[0.5]]])
    assert seq.token_ids == [1, 2, 3, 10]
    assert seq.metric.num_generated_tokens == 1

    encoded = encode_run_result([[11]], [[0.1]], 7)
    token_ids, logprobs = decode_run_result(encoded)
    assert token_ids == [[11]]
    assert math.isclose(logprobs[0][0], 0.1, rel_tol=1e-6)

    buffer = (ctypes.c_ubyte * 4096)()
    nbytes = serialize(ctypes.addressof(buffer), len(buffer), [seq], True)
    restored = __import__("dlengine._cpp", fromlist=["deserialize"]).deserialize(
        ctypes.addressof(buffer), nbytes
    )
    assert len(restored) == 1
    assert restored[0].seq_id == seq.seq_id
    assert restored[0].token_ids == seq.token_ids

    migrated = parse_migrate_batch(serialize_migrate_batch([seq]))
    assert len(migrated) == 1
    assert migrated[0].seq_id == seq.seq_id


def test_decode_batch_uses_typed_sequence_fields():
    seq = Sequence([4, 5, 6], SamplingParams())
    seq.set_active_group_block_table(0, [7])
    batch = serialize_run_batch([seq], False)
    meta = prepare_decode_from_bytes(batch, 0, 1, 4, 4, 16)
    assert meta.input_ids == [6]
    assert meta.positions == [2]
