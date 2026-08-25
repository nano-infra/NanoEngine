from types import SimpleNamespace

import pytest

from nanodeploy.engine.local_executor import LocalExecutor
from nanodeploy.engine.topology import EngineTopology


class _RemoteMethod:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def remote(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class _Worker:
    def __init__(self):
        self.run = _RemoteMethod(
            (
                [
                    list(range(16)),
                    [0] * 16,
                    list(range(100, 116)),
                ],
                1.0,
            )
        )


class _Endpoint:
    def __init__(self, *_args, **_kwargs):
        self.sent = None

    def send_seqs(self, sequences, *, is_prefill):
        self.sent = (sequences, is_prefill)


class _Sequence:
    def __init__(self, seq_id):
        self.seq_id = seq_id

    @staticmethod
    def block_ctx():
        return SimpleNamespace(master_sp_idx=0)


class _Batch:
    engine_id = 0
    wave_id = 1
    quantum_id = 0
    engine_has_real = True
    first_real = _Sequence(7)
    control_dummy = _Sequence(-1)
    second_real = _Sequence(9)
    per_rank_sequences = {
        0: [first_real, control_dummy, second_real]
    }

    @staticmethod
    def is_control_dummy(sequence):
        return sequence.seq_id == -1

    @staticmethod
    def expected_request_ids(_global_rank):
        return (7, 9)


@pytest.mark.parametrize("result_fastpath", [False, True])
def test_local_executor_preserves_real_rows_around_control_dummy(
    monkeypatch, result_fastpath
):
    monkeypatch.setattr(
        "nanodeploy.engine.local_executor.RPCServerEndpoint",
        _Endpoint,
    )
    monkeypatch.setattr(
        "nanodeploy.engine.local_executor.ray.get",
        lambda refs, timeout: refs,
    )
    config = SimpleNamespace(
        optimize_decode_block_table=True,
        hierarchical_execution_trace=False,
        hierarchical_quantum_diagnostics=False,
        hierarchical_result_fastpath=result_fastpath,
    )
    topology = EngineTopology(
        engine_id=0,
        global_dp_idx=0,
        global_ranks=(0,),
        attention_sp=1,
        attention_tp=1,
    )
    worker = _Worker()
    executor = LocalExecutor(config, topology, [worker])

    results = executor.run(_Batch(), timeout=1.0)

    assert results[0].mastered_request_ids == (7, 9)
    assert results[0].sampled_token_ids == (
        tuple(range(16)),
        tuple(range(100, 116)),
    )
    assert executor.endpoint.sent == (
        [_Batch.per_rank_sequences[0]],
        False,
    )

