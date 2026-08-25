import time
from types import SimpleNamespace

import pytest

from nanodeploy.engine.local_executor import LocalExecutor
from nanodeploy.engine.topology import EngineTopology


class _Sequence:
    def __init__(self, seq_id: int, *, control_dummy: bool = False):
        self.seq_id = seq_id
        self.control_dummy = control_dummy

    @staticmethod
    def block_ctx():
        return SimpleNamespace(master_sp_idx=0)


class _Batch:
    def __init__(self, quantum_id: int):
        self.engine_id = 0
        self.wave_id = 1
        self.quantum_id = quantum_id
        self.engine_has_real = True
        self.real = _Sequence(7)
        self.dummy = _Sequence(-1, control_dummy=True)
        self.per_rank_sequences = {0: [self.real, self.dummy]}
        self.per_request_epoch = {7: 0}
        self.per_request_output_offset = {7: quantum_id}

    @staticmethod
    def expected_request_ids(_global_rank):
        return (7,)

    @staticmethod
    def is_control_dummy(sequence):
        return sequence.control_dummy


class _RemoteRun:
    def __init__(self):
        self.calls = []

    def remote(self, **kwargs):
        self.calls.append(kwargs)
        return ([[7], [0]], time.time())


class _Endpoint:
    def __init__(self, *_args, **kwargs):
        self.num_slots = kwargs["num_slots"]
        self.sent_slots = []

    def send_seqs(
        self, _sequences, *, is_prefill, transport_slot=0
    ):
        assert not is_prefill
        self.sent_slots.append(transport_slot)


def _executor(monkeypatch):
    monkeypatch.setattr(
        "nanodeploy.engine.local_executor.RPCServerEndpoint", _Endpoint
    )
    monkeypatch.setattr(
        "nanodeploy.engine.local_executor.ray.get",
        lambda refs, timeout: refs,
    )
    worker = SimpleNamespace(run=_RemoteRun())
    config = SimpleNamespace(
        optimize_decode_block_table=True,
        hierarchical_execution_trace=False,
        hierarchical_quantum_diagnostics=False,
        hierarchical_result_fastpath=True,
        hierarchical_worker_transport="ray",
    )
    topology = EngineTopology(
        engine_id=0,
        global_dp_idx=0,
        global_ranks=(0,),
        attention_sp=1,
        attention_tp=1,
    )
    return LocalExecutor(config, topology, [worker]), worker


def test_depth_two_submit_collect_preserves_fifo_and_alternates_slots(
    monkeypatch,
):
    executor, worker = _executor(monkeypatch)
    first_batch = _Batch(0)

    flight = executor.submit(first_batch, timeout=1.0)

    assert flight.frozen_batch is first_batch
    assert flight.transport_slot == 0
    assert executor.endpoint.num_slots == 2
    assert executor.endpoint.sent_slots == [0]
    assert worker.run.calls[0]["transport_slot"] == 0
    assert worker.run.calls[0]["hierarchical_wave_id"] == 1
    assert worker.run.calls[0]["hierarchical_quantum_id"] == 0
    assert worker.run.calls[0]["hierarchical_sequence_epochs"] == (
        (7, 0),
        (-1, -1),
    )
    second = executor.submit(_Batch(1), timeout=1.0)
    assert second.transport_slot == 1
    assert executor.endpoint.sent_slots == [0, 1]
    assert worker.run.calls[1]["transport_slot"] == 1
    with pytest.raises(RuntimeError, match="capacity"):
        executor.submit(_Batch(2), timeout=1.0)
    with pytest.raises(RuntimeError, match="FIFO"):
        executor.collect(second)

    results = executor.collect(flight)
    assert results[0].mastered_request_ids == (7,)
    assert results[0].sampled_token_ids == ((7,),)
    with pytest.raises(RuntimeError, match="not pending"):
        executor.collect(flight)
    assert executor.collect(second)[0].sampled_token_ids == ((7,),)


def test_run_remains_a_synchronous_submit_collect_wrapper(monkeypatch):
    executor, _worker = _executor(monkeypatch)

    results = executor.run(_Batch(0), timeout=1.0)

    assert results[0].sampled_token_ids == ((7,),)
    assert not executor._active_flights
