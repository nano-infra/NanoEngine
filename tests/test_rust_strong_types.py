import math
import pickle

from dlengine._rust.config import CachePlan, SchedulerConfig
from dlengine._rust.core import Scheduler, SequenceMetric
from dlengine._rust.proto import MigrationIn, RequestIn, RunnerIn, SamplingParams
from dlengine.engine.dlslime_protocol import decode_run_result, encode_run_result


def _scheduler() -> Scheduler:
    return Scheduler(
        SchedulerConfig(
            engine_id="protocol-test",
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


def _add_request(sched: Scheduler, seq_id: int, tokens: list[int]) -> None:
    added = sched.add_request(
        seq_id,
        tokens,
        SamplingParams(max_tokens=2, return_completion_logprobs=True),
        0,
        None,
    )
    assert added == (seq_id, len(tokens))
    assert sched.set_sequence_metric(seq_id, SequenceMetric(seq_id, len(tokens)))


def test_sampling_params_pickle_roundtrip_for_ray_actor_args():
    params = SamplingParams(
        temperature=0.1,
        max_tokens=1024,
        ignore_eos=True,
        return_completion_logprobs=True,
        json_schema='{"type":"object"}',
        structural_tag='{"type":"structural_tag"}',
    )
    restored = pickle.loads(pickle.dumps(params))
    assert restored.temperature == 0.1
    assert restored.max_tokens == 1024
    assert restored.ignore_eos is True
    assert restored.return_completion_logprobs is True
    assert restored.json_schema == '{"type":"object"}'
    assert restored.structural_tag == '{"type":"structural_tag"}'


def test_ipc_add_request_bytes_roundtrip():
    sched = _scheduler()
    request = RequestIn(3003, [8, 9], SamplingParams(), 0, None)
    payload = request.to_bytes()
    assert sched.add_request_bytes(payload) == [(3003, 2)]
    decoded = RequestIn.from_bytes(payload)
    assert decoded.seq_id == 3003
    assert decoded.prompt_token_ids == [8, 9]


def test_scheduler_protocol_roundtrip_without_public_sequence_construction():
    sched = _scheduler()
    _add_request(sched, 1001, [1, 2, 3])

    result = sched.schedule()
    assert result.is_prefill
    assert result.dp_seq_ids == [[1001]]
    assert result.dp_group_seq_ids == [[1001]]

    batch = sched.serialize_run_batches_for_result(result, 1)[0]
    meta = RunnerIn.from_bytes(batch).prefill(0, 1, 4, 4, 16)
    assert meta.input_ids == [1, 2, 3]
    assert meta.positions == [0, 1, 2]

    sched.postprocess(result.filtered_dp_group_seq_ids, [[[10]]], None, [[[0.5]]])

    result = sched.schedule()
    assert not result.is_prefill
    assert result.dp_seq_ids == [[1001]]
    batch = sched.serialize_run_batches_for_result(result, 1)[0]
    meta = RunnerIn.from_bytes(batch).decode(0, 1, 4, 4, 16)
    assert meta.input_ids == [10]
    assert meta.positions == [3]

    encoded = encode_run_result(([[11]], [[0.1]]), 7)
    token_ids, logprobs = decode_run_result(encoded)
    assert token_ids == [[11]]
    assert math.isclose(logprobs[0][0], 0.1, rel_tol=1e-6)


def test_migration_batch_is_a_protocol_product():
    sched = _scheduler()
    _add_request(sched, 2002, [4, 5, 6])

    result = sched.schedule()
    sched.postprocess(result.filtered_dp_group_seq_ids, [[[-2]]], None, None)

    migrate_bytes = sched.serialize_migrate_batches_for_result(result, 1)[0]
    migrated = MigrationIn.from_bytes(migrate_bytes)
    assert len(migrated) == 1
    assert migrated[0].seq_id == 2002
