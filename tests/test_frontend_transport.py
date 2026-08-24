from __future__ import annotations

import time

import msgspec
import pytest
import zmq

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
    return AddCommand(
        request_id=request_id,
        prompt_token_ids=(request_id, request_id + 1),
        max_tokens=16,
        temperature=0.0,
        ignore_eos=True,
        wave_id=3,
    )


def _start_pair(
    *,
    add=None,
    enqueue=None,
    admit=None,
):
    observed: list[tuple] = []

    def default_add(command: AddCommand) -> AddResult:
        observed.append(("add", command))
        return AddResult(command.request_id, True, 2)

    def default_enqueue(
        commands: tuple[AddCommand, ...]
    ) -> tuple[IngressAck, ...]:
        observed.append(("enqueue", commands))
        return tuple(
            IngressAck(command.request_id, 2, True)
            for command in commands
        )

    def default_admit(
        commands: tuple[AddCommand, ...],
        reservations: tuple[AdmissionReservation, ...] | None,
    ) -> tuple[IngressAck, ...]:
        observed.append(("admit", commands, reservations))
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
        assert observed == [
            ("add", command1),
            ("enqueue", (command1,)),
            ("admit", (command2,), (reservation,)),
        ]
    finally:
        client.close()
        context.term()
        server.close(2.0)


def test_frontend_zmq_propagates_handler_failure():
    def fail_enqueue(commands):
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
