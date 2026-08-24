from __future__ import annotations

import time

import msgspec
import pytest
import zmq

from nanodeploy._cpp import (
    BlockContextSlot,
    Sequence,
    SequenceStatus,
    serialize_sequence_payload,
)
from nanodeploy.engine.frontend_transport import (
    FRONTEND_PROTOCOL_VERSION,
    FrontendPing,
    FrontendTransportProtocolError,
    FrontendTransportTimeout,
    RemoteFrontendError,
    ZmqFrontendClient,
    ZmqFrontendServer,
    decode_frontend_request,
    encode_frontend_request,
)
from nanodeploy.engine.hierarchical_contract import (
    AddCommand,
    AddResult,
    AdmissionReservation,
    IngressAck,
)


def _command(request_id: int) -> AddCommand:
    sequence = Sequence(
        [request_id, request_id + 1], 0.0, 16, True
    )
    sequence.seq_id = request_id
    return AddCommand(
        request_id=request_id,
        prompt_len=2,
        num_tokens=2,
        max_tokens=16,
        temperature=0.0,
        ignore_eos=True,
        wave_id=3,
        sequence_payload=serialize_sequence_payload(sequence),
    )


def _start_pair(
    *,
    add=None,
    enqueue=None,
    admit=None,
):
    observed: list[tuple] = []

    def default_add(command: AddCommand, sequence: Sequence) -> AddResult:
        observed.append(("add", command, sequence))
        return AddResult(command.request_id, True, 2)

    def default_enqueue(
        commands: tuple[AddCommand, ...],
        sequences: tuple[Sequence, ...],
    ) -> tuple[IngressAck, ...]:
        observed.append(("enqueue", commands, sequences))
        return tuple(
            IngressAck(command.request_id, 2, True)
            for command in commands
        )

    def default_admit(
        commands: tuple[AddCommand, ...],
        reservations: tuple[AdmissionReservation, ...] | None,
        sequences: tuple[Sequence, ...],
    ) -> tuple[IngressAck, ...]:
        observed.append(("admit", commands, reservations, sequences))
        return tuple(
            IngressAck(
                command.request_id,
                2,
                True,
                admission_version=9,
            )
            for command in commands
        )

    server = ZmqFrontendServer(
        engine_id=2,
        advertised_host="127.0.0.1",
        queue_capacity=8,
        add=add or default_add,
        enqueue_batch=enqueue or default_enqueue,
        admit_batch=admit or default_admit,
    )
    server.start(2.0)
    context = zmq.Context(io_threads=1)
    client = ZmqFrontendClient(
        context=context,
        address=server.address,
        deployment_epoch=server.deployment_epoch,
        engine_id=2,
        queue_capacity=8,
        startup_timeout_s=2.0,
        request_timeout_s=2.0,
    )
    return server, context, client, observed


def _wait_ingress(client, flight):
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        ready, acks = client.poll_ingress(flight)
        if ready:
            return acks
        time.sleep(0.001)
    raise AssertionError("frontend ingress response did not arrive")


def test_frontend_protocol_round_trip_and_strict_unknown_fields():
    ping = FrontendPing(
        protocol_version=FRONTEND_PROTOCOL_VERSION,
        deployment_epoch="epoch",
        engine_id=1,
        transport_id=0,
    )
    assert decode_frontend_request(encode_frontend_request(ping)) == ping

    unknown_field = msgspec.msgpack.encode(
        {
            "kind": "ping",
            "protocol_version": FRONTEND_PROTOCOL_VERSION,
            "deployment_epoch": "epoch",
            "engine_id": 1,
            "transport_id": 0,
            "unknown": True,
        }
    )
    with pytest.raises(
        FrontendTransportProtocolError, match="unknown field"
    ):
        decode_frontend_request(unknown_field)


def test_frontend_zmq_routes_add_enqueue_and_planned_admission():
    server, context, client, observed = _start_pair()
    try:
        command1 = _command(1)
        command2 = _command(2)
        assert client.add(command1) == AddResult(1, True, 2)

        enqueue_flight = client.enqueue((command1,))
        reservation = AdmissionReservation(
            request_id=2,
            engine_id=2,
            master_sp_idx=1,
            dispatched_tokens=(2, 3),
        )
        admission_flight = client.admit(
            (command2,), (reservation,)
        )

        admission_acks = _wait_ingress(client, admission_flight)
        enqueue_acks = _wait_ingress(client, enqueue_flight)
        assert admission_acks == (
            IngressAck(2, 2, True, admission_version=9),
        )
        assert enqueue_acks == (IngressAck(1, 2, True),)
        assert [entry[0] for entry in observed] == [
            "add",
            "enqueue",
            "admit",
        ]
        assert observed[0][1] == command1
        assert observed[1][1] == (command1,)
        assert observed[2][1:3] == ((command2,), (reservation,))
        assert observed[0][2].token_ids == [1, 2]
        assert observed[1][2][0].token_ids == [1, 2]
        assert observed[2][3][0].token_ids == [2, 3]
    finally:
        client.close()
        context.term()
        server.close(2.0)


def test_frontend_zmq_propagates_handler_failure():
    def fail_enqueue(commands, sequences):
        raise ValueError(f"rejected {commands[0].request_id}")

    server, context, client, _ = _start_pair(
        enqueue=fail_enqueue
    )
    try:
        flight = client.enqueue((_command(7),))
        with pytest.raises(
            RemoteFrontendError, match="ValueError: rejected 7"
        ):
            _wait_ingress(client, flight)
    finally:
        client.close()
        context.term()
        server.close(2.0)


def test_frontend_zmq_preserves_migrating_sequence_state():
    received: list[Sequence] = []

    def enqueue(commands, sequences):
        received.extend(sequences)
        return (IngressAck(commands[0].request_id, 2, True),)

    sequence = Sequence([11, 12, 13, 42], 0.1, 16, True)
    sequence.seq_id = 99
    sequence.num_prompt_tokens = 3
    sequence.last_token = 42
    sequence.status = SequenceStatus.TO_BE_MIGRATED
    sequence.active("prefill-engine", 2, 1)
    active = sequence.block_ctx(BlockContextSlot.ACTIVE)
    active.dp_idx = 1
    active.master_sp_idx = 1
    active.num_dispatched_tokens = [2, 2]
    active.sp_block_table[0] = [7]
    active.sp_block_table[1] = [8, 9]
    active.block_location.append((0, 7))
    active.block_location.append((1, 8))
    sequence.migrate()
    command = AddCommand(
        request_id=99,
        prompt_len=3,
        num_tokens=4,
        max_tokens=16,
        temperature=0.1,
        ignore_eos=True,
        wave_id=3,
        sequence_payload=serialize_sequence_payload(sequence),
    )

    server, context, client, _ = _start_pair(enqueue=enqueue)
    try:
        assert _wait_ingress(client, client.enqueue((command,))) == (
            IngressAck(99, 2, True),
        )
        assert len(received) == 1
        restored = received[0]
        assert restored.seq_id == 99
        assert restored.status == SequenceStatus.TO_BE_MIGRATED
        assert restored.token_ids == [11, 12, 13, 42]
        assert restored.num_prompt_tokens == 3
        assert restored.last_token == 42
        migrate = restored.block_ctx(BlockContextSlot.MIGRATE)
        assert migrate.engine_id == "prefill-engine"
        assert migrate.dp_idx == 1
        assert migrate.master_sp_idx == 1
        assert list(migrate.num_dispatched_tokens) == [2, 2]
        assert list(migrate.sp_block_table[0]) == [7]
        assert list(migrate.sp_block_table[1]) == [8, 9]
        assert list(migrate.block_location) == [(0, 7), (1, 8)]
    finally:
        client.close()
        context.term()
        server.close(2.0)


def test_frontend_zmq_connect_is_bounded_and_server_close_is_idempotent():
    address_context = zmq.Context(io_threads=1)
    probe = address_context.socket(zmq.ROUTER)
    port = probe.bind_to_random_port("tcp://127.0.0.1")
    probe.close(linger=0)
    context = zmq.Context(io_threads=1)
    try:
        with pytest.raises(FrontendTransportTimeout):
            ZmqFrontendClient(
                context=context,
                address=f"tcp://127.0.0.1:{port}",
                deployment_epoch="missing",
                engine_id=0,
                queue_capacity=4,
                startup_timeout_s=0.05,
                request_timeout_s=0.05,
            )
    finally:
        context.term()
        address_context.term()

    server, context, client, _ = _start_pair()
    client.close()
    context.term()
    server.close(2.0)
    server.close(2.0)
