from __future__ import annotations

import threading
import time

import msgspec
import pytest

from nanodeploy.engine.worker_transport import (
    MAX_COMMAND_BYTES,
    PROTOCOL_VERSION,
    DecodeCommand,
    DecodeSuccess,
    RemoteWorkerError,
    WorkerExecutionOutput,
    WorkerTransportError,
    WorkerTransportProtocolError,
    WorkerTransportTimeout,
    ZmqWorkerClient,
    ZmqWorkerServer,
    decode_command,
    decode_response,
    encode_command,
    encode_response,
)


def command_for(
    server: ZmqWorkerServer,
    rank: int,
    *,
    wave_id: int = 1,
    quantum_id: int = 0,
    deployment_epoch: str | None = None,
    transport_slot: int | None = None,
) -> DecodeCommand:
    return DecodeCommand(
        protocol_version=PROTOCOL_VERSION,
        deployment_epoch=deployment_epoch or server.deployment_epoch,
        engine_id=server.engine_id,
        global_rank=rank,
        wave_id=wave_id,
        quantum_id=quantum_id,
        send_timestamp=time.time(),
        transport_slot=(
            quantum_id % 2 if transport_slot is None else transport_slot
        ),
    )


def start_client(
    server: ZmqWorkerServer,
    rank: int,
    execute,
    errors: list[BaseException],
) -> threading.Thread:
    config = server.worker_config(
        rank,
        startup_timeout_s=2.0,
        quantum_timeout_s=2.0,
    )

    def run() -> None:
        try:
            ZmqWorkerClient(config).run(execute)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def test_worker_protocol_round_trip_and_strict_unknown_fields():
    command = DecodeCommand(
        protocol_version=PROTOCOL_VERSION,
        deployment_epoch="epoch",
        engine_id=2,
        global_rank=7,
        wave_id=3,
        quantum_id=4,
        send_timestamp=5.0,
        hierarchical_trace={"forwards": (1, 2)},
        hierarchical_quantum_diagnostics=True,
    )
    decoded_command = decode_command(encode_command(command))
    assert decoded_command.global_rank == 7
    assert decoded_command.transport_slot == 0
    assert decoded_command.hierarchical_trace == {"forwards": [1, 2]}

    response = DecodeSuccess(
        protocol_version=PROTOCOL_VERSION,
        deployment_epoch="epoch",
        engine_id=2,
        global_rank=7,
        wave_id=3,
        quantum_id=4,
        token_rows=[[10, 11]],
        worker_end_time=6.0,
    )
    assert decode_response(encode_response(response)) == response

    unknown_field = msgspec.msgpack.encode(
        {
            "kind": "decode",
            "protocol_version": PROTOCOL_VERSION,
            "deployment_epoch": "epoch",
            "engine_id": 2,
            "global_rank": 7,
            "wave_id": 3,
            "quantum_id": 4,
            "send_timestamp": 5.0,
            "unknown": True,
        }
    )
    with pytest.raises(
        WorkerTransportProtocolError, match="unknown field"
    ):
        decode_command(unknown_field)


def test_worker_protocol_rejects_oversized_command_before_decode():
    with pytest.raises(WorkerTransportProtocolError, match="exceeds"):
        decode_command(b"x" * (MAX_COMMAND_BYTES + 1))


def test_server_rejects_transport_slot_quantum_mismatch(monkeypatch):
    server = ZmqWorkerServer(
        address="inproc://unused",
        deployment_epoch="epoch",
        engine_id=0,
        expected_ranks=(0,),
    )
    monkeypatch.setattr(server, "_require_active", lambda: None)

    with pytest.raises(
        WorkerTransportProtocolError, match="slot/quantum mismatch"
    ):
        server.send_decode_commands(
            {
                0: command_for(
                    server,
                    0,
                    quantum_id=1,
                    transport_slot=0,
                )
            },
            deadline=time.monotonic() + 1.0,
        )


def test_router_dealer_success_is_returned_in_topology_order():
    ranks = (7, 3)
    server = ZmqWorkerServer.create_ipc(
        engine_id=1, expected_ranks=ranks
    )
    errors: list[BaseException] = []
    observed: dict[int, list[tuple[int, int]]] = {
        rank: [] for rank in ranks
    }
    threads = []
    for rank in ranks:

        def execute(command, rank=rank):
            observed[rank].append((command.wave_id, command.quantum_id))
            return WorkerExecutionOutput(
                token_rows=[[rank, command.quantum_id]],
                worker_end_time=time.time(),
            )

        threads.append(start_client(server, rank, execute, errors))

    try:
        server.activate(timeout_s=2.0)
        server.send_decode_commands(
            {rank: command_for(server, rank) for rank in ranks},
            deadline=time.monotonic() + 2.0,
        )
        results = server.receive_decode_results(
            deadline=time.monotonic() + 2.0
        )
        assert tuple(result.global_rank for result in results) == ranks
        assert [result.token_rows for result in results] == [
            [[7, 0]],
            [[3, 0]],
        ]
        server.send_decode_commands(
            {
                rank: command_for(server, rank, quantum_id=1)
                for rank in ranks
            },
            deadline=time.monotonic() + 2.0,
        )
        server.receive_decode_results(deadline=time.monotonic() + 2.0)
        server.send_decode_commands(
            {
                rank: command_for(
                    server, rank, wave_id=2, quantum_id=0
                )
                for rank in ranks
            },
            deadline=time.monotonic() + 2.0,
        )
        server.receive_decode_results(deadline=time.monotonic() + 2.0)
        assert observed == {
            7: [(1, 0), (1, 1), (2, 0)],
            3: [(1, 0), (1, 1), (2, 0)],
        }
        assert server.close(graceful=True, timeout_s=2.0) == ()
    finally:
        if server.cleanup_dir is not None:
            server.close(graceful=False, timeout_s=0.0)
    for thread in threads:
        thread.join(timeout=2.0)
        assert not thread.is_alive()
    assert errors == []


def test_worker_execution_failure_is_fail_stop_and_not_retried():
    server = ZmqWorkerServer.create_ipc(
        engine_id=0, expected_ranks=(0,)
    )
    client_errors: list[BaseException] = []
    call_count = 0

    def execute(_command):
        nonlocal call_count
        call_count += 1
        raise ValueError("decode exploded")

    thread = start_client(server, 0, execute, client_errors)
    try:
        server.activate(timeout_s=2.0)
        server.send_decode_commands(
            {0: command_for(server, 0)},
            deadline=time.monotonic() + 2.0,
        )
        with pytest.raises(RemoteWorkerError, match="decode exploded"):
            server.receive_decode_results(
                deadline=time.monotonic() + 2.0
            )
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert call_count == 1
        assert len(client_errors) == 1
        assert isinstance(client_errors[0], ValueError)
    finally:
        server.close(graceful=False, timeout_s=0.0)


def test_server_rejects_wrong_epoch_before_sending_command():
    server = ZmqWorkerServer.create_ipc(
        engine_id=0, expected_ranks=(0,)
    )
    client_errors: list[BaseException] = []
    thread = start_client(
        server,
        0,
        lambda _command: WorkerExecutionOutput([], time.time()),
        client_errors,
    )
    try:
        server.activate(timeout_s=2.0)
        with pytest.raises(
            WorkerTransportProtocolError, match="deployment epoch"
        ):
            server.send_decode_commands(
                {
                    0: command_for(
                        server, 0, deployment_epoch="stale-epoch"
                    )
                },
                deadline=time.monotonic() + 2.0,
            )
        assert server.close(graceful=True, timeout_s=2.0) == ()
    finally:
        if server.cleanup_dir is not None:
            server.close(graceful=False, timeout_s=0.0)
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert client_errors == []


def test_worker_rejects_skipped_quantum_and_exits():
    server = ZmqWorkerServer.create_ipc(
        engine_id=0, expected_ranks=(0,)
    )
    client_errors: list[BaseException] = []
    thread = start_client(
        server,
        0,
        lambda command: WorkerExecutionOutput(
            [[command.quantum_id]], time.time()
        ),
        client_errors,
    )
    try:
        server.activate(timeout_s=2.0)
        server.send_decode_commands(
            {0: command_for(server, 0)},
            deadline=time.monotonic() + 2.0,
        )
        server.receive_decode_results(deadline=time.monotonic() + 2.0)
        server.send_decode_commands(
            {0: command_for(server, 0, quantum_id=2)},
            deadline=time.monotonic() + 2.0,
        )
        with pytest.raises(RemoteWorkerError, match="out-of-order"):
            server.receive_decode_results(
                deadline=time.monotonic() + 2.0
            )
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert len(client_errors) == 1
        assert isinstance(
            client_errors[0], WorkerTransportProtocolError
        )
    finally:
        server.close(graceful=False, timeout_s=0.0)


def test_server_rejects_socket_use_from_non_owner_thread():
    server = ZmqWorkerServer.create_ipc(
        engine_id=0, expected_ranks=(0,)
    )
    client_errors: list[BaseException] = []
    thread = start_client(
        server,
        0,
        lambda _command: WorkerExecutionOutput([], time.time()),
        client_errors,
    )
    foreign_errors: list[BaseException] = []

    def foreign_send() -> None:
        try:
            server.send_decode_commands(
                {0: command_for(server, 0)},
                deadline=time.monotonic() + 1.0,
            )
        except BaseException as exc:
            foreign_errors.append(exc)

    try:
        server.activate(timeout_s=2.0)
        foreign = threading.Thread(target=foreign_send)
        foreign.start()
        foreign.join(timeout=2.0)
        assert len(foreign_errors) == 1
        assert isinstance(foreign_errors[0], WorkerTransportError)
        assert "owner thread" in str(foreign_errors[0])
        assert server.close(graceful=True, timeout_s=2.0) == ()
    finally:
        if server.cleanup_dir is not None:
            server.close(graceful=False, timeout_s=0.0)
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert client_errors == []


def test_server_ready_uses_one_absolute_timeout():
    server = ZmqWorkerServer.create_ipc(
        engine_id=0, expected_ranks=(0,)
    )
    begin = time.monotonic()
    with pytest.raises(WorkerTransportTimeout, match="READY"):
        server.activate(timeout_s=0.05)
    elapsed = time.monotonic() - begin
    assert elapsed < 0.5
    assert server.cleanup_dir is None


def test_partial_command_fanout_is_fatal_and_reports_sent_ranks(
    monkeypatch,
):
    server = ZmqWorkerServer(
        address="inproc://unused",
        deployment_epoch="epoch",
        engine_id=0,
        expected_ranks=(0, 1),
    )
    sent: list[int] = []

    def fake_send(rank, _frame, _deadline, _operation):
        if rank == 1:
            raise WorkerTransportError("rank disappeared")
        sent.append(rank)

    monkeypatch.setattr(server, "_require_active", lambda: None)
    monkeypatch.setattr(server, "_send", fake_send)
    commands = {
        rank: command_for(server, rank) for rank in server.expected_ranks
    }

    with pytest.raises(
        WorkerTransportError, match=r"after ranks \(0,\)"
    ):
        server.send_decode_commands(
            commands, deadline=time.monotonic() + 1.0
        )
    assert sent == [0]
    with pytest.raises(WorkerTransportError, match="is failed"):
        server.send_decode_commands(
            commands, deadline=time.monotonic() + 1.0
        )
