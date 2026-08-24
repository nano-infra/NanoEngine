from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable, TypeAlias

import msgspec
import zmq

from nanodeploy.engine.hierarchical_contract import (
    AddCommand,
    AddResult,
    AdmissionReservation,
    IngressAck,
)


FRONTEND_PROTOCOL_VERSION = 1
MAX_FRONTEND_REQUEST_BYTES = 1 << 30
MAX_FRONTEND_RESPONSE_BYTES = 16 << 20
_MAX_FAILURE_MESSAGE_CHARS = 8 << 10
_IDLE_POLL_MS = 100


class FrontendTransportError(RuntimeError):
    """Base error for RequestRouter-to-LocalEngine ZMQ traffic."""


class FrontendTransportProtocolError(FrontendTransportError):
    """A frontend transport peer violated the wire contract."""


class FrontendTransportTimeout(TimeoutError, FrontendTransportError):
    """A bounded frontend transport operation timed out."""


class RemoteFrontendError(FrontendTransportError):
    """A LocalEngine frontend handler failed."""


class _WireAddCommand(msgspec.Struct, array_like=True, frozen=True):
    request_id: int
    prompt_token_ids: tuple[int, ...]
    max_tokens: int
    temperature: float
    ignore_eos: bool
    wave_id: int


class _WireAdmissionReservation(
    msgspec.Struct, array_like=True, frozen=True
):
    request_id: int
    engine_id: int
    master_sp_idx: int
    dispatched_tokens: tuple[int, ...]


class _WireAddResult(msgspec.Struct, array_like=True, frozen=True):
    request_id: int
    accepted: bool
    engine_id: int | None
    reason: str | None


class _WireIngressAck(msgspec.Struct, array_like=True, frozen=True):
    request_id: int
    engine_id: int
    enqueued: bool
    reason: str | None
    admission_version: int | None
    capacity_epoch: int | None
    router_pending_ms: float | None
    admission_rpc_ms: float | None
    local_command_queue_ms: float | None
    local_admission_ms: float | None


class FrontendPing(
    msgspec.Struct,
    tag="ping",
    tag_field="kind",
    forbid_unknown_fields=True,
    frozen=True,
):
    protocol_version: int
    deployment_epoch: str
    engine_id: int
    transport_id: int


class FrontendAddRequest(
    msgspec.Struct,
    tag="add",
    tag_field="kind",
    forbid_unknown_fields=True,
    frozen=True,
):
    protocol_version: int
    deployment_epoch: str
    engine_id: int
    transport_id: int
    command: _WireAddCommand


class FrontendEnqueueRequest(
    msgspec.Struct,
    tag="enqueue",
    tag_field="kind",
    forbid_unknown_fields=True,
    frozen=True,
):
    protocol_version: int
    deployment_epoch: str
    engine_id: int
    transport_id: int
    commands: tuple[_WireAddCommand, ...]


class FrontendAdmissionRequest(
    msgspec.Struct,
    tag="admit",
    tag_field="kind",
    forbid_unknown_fields=True,
    frozen=True,
):
    protocol_version: int
    deployment_epoch: str
    engine_id: int
    transport_id: int
    commands: tuple[_WireAddCommand, ...]
    reservations: tuple[_WireAdmissionReservation, ...] | None


class FrontendReady(
    msgspec.Struct,
    tag="ready",
    tag_field="kind",
    forbid_unknown_fields=True,
    frozen=True,
):
    protocol_version: int
    deployment_epoch: str
    engine_id: int
    transport_id: int


class FrontendAddResponse(
    msgspec.Struct,
    tag="add_result",
    tag_field="kind",
    forbid_unknown_fields=True,
    frozen=True,
):
    protocol_version: int
    deployment_epoch: str
    engine_id: int
    transport_id: int
    result: _WireAddResult


class FrontendIngressResponse(
    msgspec.Struct,
    tag="ingress",
    tag_field="kind",
    forbid_unknown_fields=True,
    frozen=True,
):
    protocol_version: int
    deployment_epoch: str
    engine_id: int
    transport_id: int
    acks: tuple[_WireIngressAck, ...]


class FrontendFailure(
    msgspec.Struct,
    tag="failure",
    tag_field="kind",
    forbid_unknown_fields=True,
    frozen=True,
):
    protocol_version: int
    deployment_epoch: str
    engine_id: int
    transport_id: int
    exception_type: str
    message: str


FrontendRequest: TypeAlias = (
    FrontendPing
    | FrontendAddRequest
    | FrontendEnqueueRequest
    | FrontendAdmissionRequest
)
FrontendResponse: TypeAlias = (
    FrontendReady
    | FrontendAddResponse
    | FrontendIngressResponse
    | FrontendFailure
)

_REQUEST_DECODER = msgspec.msgpack.Decoder(type=FrontendRequest)
_RESPONSE_DECODER = msgspec.msgpack.Decoder(type=FrontendResponse)


@dataclass(frozen=True, slots=True)
class FrontendFlight:
    engine_id: int
    transport_id: int
    response_kind: str


def _wire_command(command: AddCommand) -> _WireAddCommand:
    return _WireAddCommand(
        command.request_id,
        command.prompt_token_ids,
        command.max_tokens,
        command.temperature,
        command.ignore_eos,
        command.wave_id,
    )


def _command_from_wire(command: _WireAddCommand) -> AddCommand:
    return AddCommand(
        request_id=command.request_id,
        prompt_token_ids=command.prompt_token_ids,
        max_tokens=command.max_tokens,
        temperature=command.temperature,
        ignore_eos=command.ignore_eos,
        wave_id=command.wave_id,
    )


def _wire_reservation(
    reservation: AdmissionReservation,
) -> _WireAdmissionReservation:
    return _WireAdmissionReservation(
        reservation.request_id,
        reservation.engine_id,
        reservation.master_sp_idx,
        reservation.dispatched_tokens,
    )


def _reservation_from_wire(
    reservation: _WireAdmissionReservation,
) -> AdmissionReservation:
    return AdmissionReservation(
        request_id=reservation.request_id,
        engine_id=reservation.engine_id,
        master_sp_idx=reservation.master_sp_idx,
        dispatched_tokens=reservation.dispatched_tokens,
    )


def _wire_result(result: AddResult) -> _WireAddResult:
    return _WireAddResult(
        result.request_id,
        result.accepted,
        result.engine_id,
        result.reason,
    )


def _result_from_wire(result: _WireAddResult) -> AddResult:
    return AddResult(
        request_id=result.request_id,
        accepted=result.accepted,
        engine_id=result.engine_id,
        reason=result.reason,
    )


def _wire_ack(ack: IngressAck) -> _WireIngressAck:
    return _WireIngressAck(
        ack.request_id,
        ack.engine_id,
        ack.enqueued,
        ack.reason,
        ack.admission_version,
        ack.capacity_epoch,
        ack.router_pending_ms,
        ack.admission_rpc_ms,
        ack.local_command_queue_ms,
        ack.local_admission_ms,
    )


def _ack_from_wire(ack: _WireIngressAck) -> IngressAck:
    return IngressAck(
        request_id=ack.request_id,
        engine_id=ack.engine_id,
        enqueued=ack.enqueued,
        reason=ack.reason,
        admission_version=ack.admission_version,
        capacity_epoch=ack.capacity_epoch,
        router_pending_ms=ack.router_pending_ms,
        admission_rpc_ms=ack.admission_rpc_ms,
        local_command_queue_ms=ack.local_command_queue_ms,
        local_admission_ms=ack.local_admission_ms,
    )


def encode_frontend_request(message: FrontendRequest) -> bytes:
    return _encode_message(
        message, MAX_FRONTEND_REQUEST_BYTES, "request"
    )


def decode_frontend_request(frame: bytes) -> FrontendRequest:
    return _decode_message(
        _REQUEST_DECODER,
        frame,
        MAX_FRONTEND_REQUEST_BYTES,
        "request",
    )


def encode_frontend_response(message: FrontendResponse) -> bytes:
    return _encode_message(
        message, MAX_FRONTEND_RESPONSE_BYTES, "response"
    )


def decode_frontend_response(frame: bytes) -> FrontendResponse:
    return _decode_message(
        _RESPONSE_DECODER,
        frame,
        MAX_FRONTEND_RESPONSE_BYTES,
        "response",
    )


def _encode_message(message, maximum: int, label: str) -> bytes:
    try:
        frame = msgspec.msgpack.encode(message)
    except (TypeError, ValueError, msgspec.EncodeError) as exc:
        raise FrontendTransportProtocolError(
            f"could not encode frontend {label}: {exc}"
        ) from exc
    if len(frame) > maximum:
        raise FrontendTransportProtocolError(
            f"frontend {label} exceeds {maximum} bytes: {len(frame)}"
        )
    return frame


def _decode_message(
    decoder: msgspec.msgpack.Decoder,
    frame: bytes,
    maximum: int,
    label: str,
):
    if len(frame) > maximum:
        raise FrontendTransportProtocolError(
            f"frontend {label} exceeds {maximum} bytes: {len(frame)}"
        )
    try:
        return decoder.decode(frame)
    except msgspec.DecodeError as exc:
        raise FrontendTransportProtocolError(
            f"could not decode frontend {label}: {exc}"
        ) from exc


def _identity(deployment_epoch: str, engine_id: int) -> bytes:
    return (
        f"nanodeploy-frontend-v{FRONTEND_PROTOCOL_VERSION}:"
        f"{deployment_epoch}:e{engine_id}"
    ).encode("ascii")


def _remaining_ms(deadline: float, operation: str) -> int:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise FrontendTransportTimeout(f"{operation} timed out")
    return max(1, math.ceil(remaining * 1000))


def _configure_socket(
    socket: zmq.Socket,
    *,
    high_water_mark: int,
    max_message_bytes: int,
) -> None:
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SNDHWM, high_water_mark)
    socket.setsockopt(zmq.RCVHWM, high_water_mark)
    socket.setsockopt(zmq.MAXMSGSIZE, max_message_bytes)


def _validate_common(
    message,
    *,
    deployment_epoch: str,
    engine_id: int,
) -> None:
    if message.protocol_version != FRONTEND_PROTOCOL_VERSION:
        raise FrontendTransportProtocolError(
            "frontend protocol version mismatch: "
            f"expected={FRONTEND_PROTOCOL_VERSION}, "
            f"got={message.protocol_version}"
        )
    if message.deployment_epoch != deployment_epoch:
        raise FrontendTransportProtocolError(
            "frontend deployment epoch mismatch"
        )
    if message.engine_id != engine_id:
        raise FrontendTransportProtocolError(
            f"frontend engine mismatch: expected={engine_id}, "
            f"got={message.engine_id}"
        )


class ZmqFrontendServer:
    """Dedicated TCP ROUTER serving one LocalEngine's ingress methods."""

    def __init__(
        self,
        *,
        engine_id: int,
        advertised_host: str,
        queue_capacity: int,
        add: Callable[[AddCommand], AddResult],
        enqueue_batch: Callable[
            [tuple[AddCommand, ...]], tuple[IngressAck, ...]
        ],
        admit_batch: Callable[
            [
                tuple[AddCommand, ...],
                tuple[AdmissionReservation, ...] | None,
            ],
            tuple[IngressAck, ...],
        ],
    ) -> None:
        if queue_capacity <= 0:
            raise ValueError("frontend ZMQ queue capacity must be positive")
        self.engine_id = engine_id
        self.advertised_host = advertised_host
        self.deployment_epoch = uuid.uuid4().hex
        self._high_water_mark = max(4, queue_capacity)
        self._add = add
        self._enqueue_batch = enqueue_batch
        self._admit_batch = admit_batch
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._address: str | None = None
        self._error: BaseException | None = None

    @property
    def address(self) -> str:
        if self._address is None:
            raise FrontendTransportError(
                "frontend ZMQ server is not active"
            )
        return self._address

    def start(self, timeout_s: float) -> None:
        if timeout_s <= 0:
            raise ValueError("frontend ZMQ startup timeout must be positive")
        if self._thread is not None:
            raise FrontendTransportError(
                "frontend ZMQ server is already started"
            )
        self._thread = threading.Thread(
            target=self._run,
            name=f"nanodeploy-frontend-zmq-{self.engine_id}",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout_s):
            self._stop.set()
            raise FrontendTransportTimeout(
                f"engine {self.engine_id} frontend ZMQ bind timed out"
            )
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise FrontendTransportError(
                f"engine {self.engine_id} frontend ZMQ server failed: "
                f"{type(self._error).__name__}: {self._error}"
            ) from self._error
        if self._thread is not None and not self._thread.is_alive():
            raise FrontendTransportError(
                f"engine {self.engine_id} frontend ZMQ server stopped"
            )

    def close(self, timeout_s: float) -> None:
        self._stop.set()
        if self._thread is None:
            return
        self._thread.join(timeout=max(timeout_s, 0.001))
        if self._thread.is_alive():
            raise FrontendTransportTimeout(
                f"engine {self.engine_id} frontend ZMQ server did not stop"
            )

    def _run(self) -> None:
        context: zmq.Context | None = None
        socket: zmq.Socket | None = None
        poller: zmq.Poller | None = None
        try:
            context = zmq.Context(io_threads=1)
            socket = context.socket(zmq.ROUTER)
            _configure_socket(
                socket,
                high_water_mark=self._high_water_mark,
                max_message_bytes=MAX_FRONTEND_REQUEST_BYTES,
            )
            socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
            port = socket.bind_to_random_port("tcp://0.0.0.0")
            self._address = (
                f"tcp://{self.advertised_host}:{port}"
            )
            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)
            self._ready.set()
            expected_identity = _identity(
                self.deployment_epoch, self.engine_id
            )
            while not self._stop.is_set():
                events = dict(poller.poll(_IDLE_POLL_MS))
                if not events.get(socket, 0) & zmq.POLLIN:
                    continue
                frames = socket.recv_multipart(flags=zmq.NOBLOCK)
                if len(frames) != 2:
                    continue
                identity, frame = frames
                if identity != expected_identity:
                    continue
                response = self._handle(frame)
                self._send(socket, identity, response)
        except BaseException as exc:
            self._error = exc
            self._ready.set()
        finally:
            if poller is not None and socket is not None:
                try:
                    poller.unregister(socket)
                except (KeyError, zmq.ZMQError):
                    pass
            if socket is not None:
                socket.close(linger=0)
            if context is not None:
                context.term()

    def _handle(self, frame: bytes) -> FrontendResponse:
        transport_id = -1
        try:
            request = decode_frontend_request(frame)
            transport_id = request.transport_id
            _validate_common(
                request,
                deployment_epoch=self.deployment_epoch,
                engine_id=self.engine_id,
            )
            if request.transport_id < 0:
                raise FrontendTransportProtocolError(
                    "frontend transport id must be non-negative"
                )
            if isinstance(request, FrontendPing):
                return FrontendReady(
                    FRONTEND_PROTOCOL_VERSION,
                    self.deployment_epoch,
                    self.engine_id,
                    request.transport_id,
                )
            if isinstance(request, FrontendAddRequest):
                result = self._add(_command_from_wire(request.command))
                return FrontendAddResponse(
                    FRONTEND_PROTOCOL_VERSION,
                    self.deployment_epoch,
                    self.engine_id,
                    request.transport_id,
                    _wire_result(result),
                )
            commands = tuple(
                _command_from_wire(command)
                for command in request.commands
            )
            if isinstance(request, FrontendEnqueueRequest):
                acks = self._enqueue_batch(commands)
            else:
                reservations = (
                    tuple(
                        _reservation_from_wire(reservation)
                        for reservation in request.reservations
                    )
                    if request.reservations is not None
                    else None
                )
                acks = self._admit_batch(commands, reservations)
            return FrontendIngressResponse(
                FRONTEND_PROTOCOL_VERSION,
                self.deployment_epoch,
                self.engine_id,
                request.transport_id,
                tuple(_wire_ack(ack) for ack in acks),
            )
        except BaseException as exc:
            return FrontendFailure(
                FRONTEND_PROTOCOL_VERSION,
                self.deployment_epoch,
                self.engine_id,
                transport_id,
                type(exc).__name__[:256],
                str(exc)[:_MAX_FAILURE_MESSAGE_CHARS],
            )

    def _send(
        self,
        socket: zmq.Socket,
        identity: bytes,
        response: FrontendResponse,
    ) -> None:
        frame = encode_frontend_response(response)
        deadline = time.monotonic() + 1.0
        while True:
            try:
                socket.send_multipart(
                    [identity, frame], flags=zmq.NOBLOCK
                )
                return
            except zmq.Again:
                poller = zmq.Poller()
                poller.register(socket, zmq.POLLOUT)
                if not poller.poll(
                    _remaining_ms(deadline, "frontend response")
                ):
                    raise FrontendTransportTimeout(
                        "frontend response timed out"
                    )


class ZmqFrontendClient:
    """Persistent DEALER owned by the LLMEngine frontend thread."""

    def __init__(
        self,
        *,
        context: zmq.Context,
        address: str,
        deployment_epoch: str,
        engine_id: int,
        queue_capacity: int,
        startup_timeout_s: float,
        request_timeout_s: float,
    ) -> None:
        if startup_timeout_s <= 0 or request_timeout_s <= 0:
            raise ValueError("frontend ZMQ timeouts must be positive")
        self.engine_id = engine_id
        self.deployment_epoch = deployment_epoch
        self.request_timeout_s = request_timeout_s
        self._socket = context.socket(zmq.DEALER)
        self._owner_thread_id = threading.get_ident()
        self._poller = zmq.Poller()
        self._next_transport_id = 1
        self._pending: dict[int, FrontendFlight] = {}
        self._completed: dict[int, FrontendResponse] = {}
        self._closed = False
        try:
            _configure_socket(
                self._socket,
                high_water_mark=max(4, queue_capacity),
                max_message_bytes=MAX_FRONTEND_RESPONSE_BYTES,
            )
            self._socket.setsockopt(
                zmq.IDENTITY,
                _identity(deployment_epoch, engine_id),
            )
            self._socket.setsockopt(zmq.IMMEDIATE, 1)
            self._poller.register(self._socket, zmq.POLLIN)
            self._socket.connect(address)
            deadline = time.monotonic() + startup_timeout_s
            self._wait_writable(deadline, "frontend READY")
            ping = FrontendPing(
                FRONTEND_PROTOCOL_VERSION,
                deployment_epoch,
                engine_id,
                0,
            )
            self._send_frame(
                encode_frontend_request(ping), deadline, "frontend READY"
            )
            response = self._receive_response(deadline, "frontend READY")
            self._validate_response(response, expected_transport_id=0)
            if isinstance(response, FrontendFailure):
                self._raise_remote(response)
            if not isinstance(response, FrontendReady):
                raise FrontendTransportProtocolError(
                    "frontend handshake expected READY, got "
                    f"{type(response).__name__}"
                )
        except BaseException:
            self.close()
            raise

    @property
    def socket(self) -> zmq.Socket:
        return self._socket

    def add(self, command: AddCommand) -> AddResult:
        flight = self._start(
            "add",
            lambda transport_id: FrontendAddRequest(
                FRONTEND_PROTOCOL_VERSION,
                self.deployment_epoch,
                self.engine_id,
                transport_id,
                _wire_command(command),
            ),
        )
        response = self._wait(flight)
        if not isinstance(response, FrontendAddResponse):
            raise FrontendTransportProtocolError(
                "frontend add returned "
                f"{type(response).__name__}"
            )
        return _result_from_wire(response.result)

    def enqueue(
        self, commands: tuple[AddCommand, ...]
    ) -> FrontendFlight:
        return self._start(
            "ingress",
            lambda transport_id: FrontendEnqueueRequest(
                FRONTEND_PROTOCOL_VERSION,
                self.deployment_epoch,
                self.engine_id,
                transport_id,
                tuple(_wire_command(command) for command in commands),
            ),
        )

    def admit(
        self,
        commands: tuple[AddCommand, ...],
        reservations: tuple[AdmissionReservation, ...] | None,
    ) -> FrontendFlight:
        return self._start(
            "ingress",
            lambda transport_id: FrontendAdmissionRequest(
                FRONTEND_PROTOCOL_VERSION,
                self.deployment_epoch,
                self.engine_id,
                transport_id,
                tuple(_wire_command(command) for command in commands),
                (
                    tuple(
                        _wire_reservation(reservation)
                        for reservation in reservations
                    )
                    if reservations is not None
                    else None
                ),
            ),
        )

    def poll_ingress(
        self, flight: FrontendFlight
    ) -> tuple[bool, tuple[IngressAck, ...] | None]:
        response = self._poll(flight)
        if response is None:
            return False, None
        if not isinstance(response, FrontendIngressResponse):
            raise FrontendTransportProtocolError(
                "frontend ingress returned "
                f"{type(response).__name__}"
            )
        return True, tuple(_ack_from_wire(ack) for ack in response.acks)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._poller.unregister(self._socket)
        except (KeyError, zmq.ZMQError):
            pass
        self._socket.close(linger=0)
        self._pending.clear()
        self._completed.clear()

    def _start(
        self,
        response_kind: str,
        build: Callable[[int], FrontendRequest],
    ) -> FrontendFlight:
        self._assert_owner_thread()
        if self._closed:
            raise FrontendTransportError("frontend ZMQ client is closed")
        transport_id = self._next_transport_id
        flight = FrontendFlight(
            engine_id=self.engine_id,
            transport_id=transport_id,
            response_kind=response_kind,
        )
        message = build(transport_id)
        deadline = time.monotonic() + self.request_timeout_s
        self._send_frame(
            encode_frontend_request(message), deadline, "frontend request"
        )
        self._next_transport_id += 1
        self._pending[transport_id] = flight
        return flight

    def _wait(self, flight: FrontendFlight) -> FrontendResponse:
        deadline = time.monotonic() + self.request_timeout_s
        while True:
            response = self._pop_completed(flight)
            if response is not None:
                return response
            self._receive_and_buffer(deadline, "frontend response")

    def _poll(self, flight: FrontendFlight) -> FrontendResponse | None:
        self._assert_flight(flight)
        response = self._pop_completed(flight)
        if response is not None:
            return response
        events = dict(self._poller.poll(0))
        while events.get(self._socket, 0) & zmq.POLLIN:
            self._receive_and_buffer(
                time.monotonic() + self.request_timeout_s,
                "frontend response",
                nonblocking=True,
            )
            response = self._pop_completed(flight)
            if response is not None:
                return response
            events = dict(self._poller.poll(0))
        return None

    def _receive_and_buffer(
        self,
        deadline: float,
        operation: str,
        *,
        nonblocking: bool = False,
    ) -> None:
        if nonblocking:
            try:
                frames = self._socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                return
            response = self._decode_frames(frames, operation)
        else:
            response = self._receive_response(deadline, operation)
        self._validate_response(response)
        if response.transport_id not in self._pending:
            raise FrontendTransportProtocolError(
                "frontend response has no pending request: "
                f"transport_id={response.transport_id}"
            )
        if response.transport_id in self._completed:
            raise FrontendTransportProtocolError(
                "duplicate frontend response: "
                f"transport_id={response.transport_id}"
            )
        self._completed[response.transport_id] = response

    def _pop_completed(
        self, flight: FrontendFlight
    ) -> FrontendResponse | None:
        self._assert_flight(flight)
        response = self._completed.pop(flight.transport_id, None)
        if response is None:
            return None
        self._pending.pop(flight.transport_id)
        if isinstance(response, FrontendFailure):
            self._raise_remote(response)
        return response

    def _assert_flight(self, flight: FrontendFlight) -> None:
        self._assert_owner_thread()
        if flight.engine_id != self.engine_id:
            raise FrontendTransportProtocolError(
                f"frontend flight belongs to engine {flight.engine_id}, "
                f"not {self.engine_id}"
            )
        pending = self._pending.get(flight.transport_id)
        if pending != flight:
            raise FrontendTransportProtocolError(
                "frontend flight is not pending: "
                f"transport_id={flight.transport_id}"
            )

    def _validate_response(
        self,
        response: FrontendResponse,
        *,
        expected_transport_id: int | None = None,
    ) -> None:
        _validate_common(
            response,
            deployment_epoch=self.deployment_epoch,
            engine_id=self.engine_id,
        )
        if (
            expected_transport_id is not None
            and response.transport_id != expected_transport_id
        ):
            raise FrontendTransportProtocolError(
                "frontend response correlation mismatch: "
                f"expected={expected_transport_id}, "
                f"got={response.transport_id}"
            )

    def _raise_remote(self, response: FrontendFailure) -> None:
        raise RemoteFrontendError(
            f"LocalEngine {self.engine_id} frontend request failed: "
            f"{response.exception_type}: {response.message}"
        )

    def _send_frame(
        self, frame: bytes, deadline: float, operation: str
    ) -> None:
        while True:
            try:
                self._socket.send(frame, flags=zmq.NOBLOCK)
                return
            except zmq.Again:
                self._wait_writable(deadline, operation)

    def _wait_writable(self, deadline: float, operation: str) -> None:
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLOUT)
        if not poller.poll(_remaining_ms(deadline, operation)):
            raise FrontendTransportTimeout(f"{operation} timed out")

    def _receive_response(
        self, deadline: float, operation: str
    ) -> FrontendResponse:
        events = dict(
            self._poller.poll(_remaining_ms(deadline, operation))
        )
        if not events.get(self._socket, 0) & zmq.POLLIN:
            raise FrontendTransportTimeout(f"{operation} timed out")
        try:
            frames = self._socket.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again as exc:
            raise FrontendTransportError(
                f"{operation} became unreadable after poll"
            ) from exc
        return self._decode_frames(frames, operation)

    def _decode_frames(
        self, frames: list[bytes], operation: str
    ) -> FrontendResponse:
        if len(frames) != 1:
            raise FrontendTransportProtocolError(
                f"{operation} expected one frame, got {len(frames)}"
            )
        return decode_frontend_response(frames[0])

    def _assert_owner_thread(self) -> None:
        if threading.get_ident() != self._owner_thread_id:
            raise FrontendTransportError(
                "frontend ZMQ socket used outside its owner thread"
            )
