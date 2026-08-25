from __future__ import annotations

import math
import os
import tempfile
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

import msgspec
import zmq


PROTOCOL_VERSION = 2
TRANSPORT_SLOTS = 2
MAX_COMMAND_BYTES = 1 << 20
MAX_RESPONSE_BYTES = 16 << 20
MAX_FAILURE_MESSAGE_CHARS = 8 << 10
MAX_FAILURE_TRACEBACK_CHARS = 64 << 10
_SOCKET_HWM = 4
_IDLE_POLL_MS = 1_000
_MAX_IPC_PATH_BYTES = 100


class WorkerTransportError(RuntimeError):
    """Base error for the persistent hierarchical worker transport."""


class WorkerTransportProtocolError(WorkerTransportError):
    """A peer sent a message that violates the transport contract."""


class WorkerTransportTimeout(TimeoutError, WorkerTransportError):
    """A bounded transport operation exceeded its absolute deadline."""


class RemoteWorkerError(WorkerTransportError):
    """A persistent worker reported an execution failure."""


class WorkerReady(
    msgspec.Struct,
    tag="ready",
    tag_field="kind",
    forbid_unknown_fields=True,
    frozen=True,
):
    protocol_version: int
    deployment_epoch: str
    engine_id: int
    global_rank: int


class DecodeCommand(
    msgspec.Struct,
    tag="decode",
    tag_field="kind",
    forbid_unknown_fields=True,
    frozen=True,
):
    protocol_version: int
    deployment_epoch: str
    engine_id: int
    global_rank: int
    wave_id: int
    quantum_id: int
    send_timestamp: float
    transport_slot: int = 0
    hierarchical_trace: dict[str, Any] | None = None
    hierarchical_quantum_diagnostics: bool = False


class StopCommand(
    msgspec.Struct,
    tag="stop",
    tag_field="kind",
    forbid_unknown_fields=True,
    frozen=True,
):
    protocol_version: int
    deployment_epoch: str
    engine_id: int
    global_rank: int
    reason: str = "shutdown"


class DecodeSuccess(
    msgspec.Struct,
    tag="success",
    tag_field="kind",
    forbid_unknown_fields=True,
    frozen=True,
):
    protocol_version: int
    deployment_epoch: str
    engine_id: int
    global_rank: int
    wave_id: int
    quantum_id: int
    token_rows: list[list[int]]
    worker_end_time: float
    hierarchical_trace: dict[str, Any] | None = None
    diagnostic: dict[str, Any] | None = None


class DecodeFailure(
    msgspec.Struct,
    tag="failure",
    tag_field="kind",
    forbid_unknown_fields=True,
    frozen=True,
):
    protocol_version: int
    deployment_epoch: str
    engine_id: int
    global_rank: int
    wave_id: int
    quantum_id: int
    exception_type: str
    message: str
    traceback: str


class WorkerStopped(
    msgspec.Struct,
    tag="stopped",
    tag_field="kind",
    forbid_unknown_fields=True,
    frozen=True,
):
    protocol_version: int
    deployment_epoch: str
    engine_id: int
    global_rank: int


WorkerCommand: TypeAlias = DecodeCommand | StopCommand
WorkerResponse: TypeAlias = (
    WorkerReady | DecodeSuccess | DecodeFailure | WorkerStopped
)

_COMMAND_DECODER = msgspec.msgpack.Decoder(type=WorkerCommand)
_RESPONSE_DECODER = msgspec.msgpack.Decoder(type=WorkerResponse)


@dataclass(frozen=True, slots=True)
class WorkerZmqConfig:
    address: str
    deployment_epoch: str
    engine_id: int
    global_rank: int
    startup_timeout_s: float
    quantum_timeout_s: float


@dataclass(slots=True)
class WorkerExecutionOutput:
    token_rows: list[list[int]]
    worker_end_time: float
    hierarchical_trace: dict[str, Any] | None = None
    diagnostic: dict[str, Any] | None = None


def encode_command(message: WorkerCommand) -> bytes:
    return _encode_message(message, MAX_COMMAND_BYTES, "command")


def encode_response(message: WorkerResponse) -> bytes:
    return _encode_message(message, MAX_RESPONSE_BYTES, "response")


def decode_command(frame: bytes) -> WorkerCommand:
    return _decode_message(
        _COMMAND_DECODER,
        frame,
        MAX_COMMAND_BYTES,
        "command",
    )


def decode_response(frame: bytes) -> WorkerResponse:
    return _decode_message(
        _RESPONSE_DECODER,
        frame,
        MAX_RESPONSE_BYTES,
        "response",
    )


def _encode_message(message: Any, maximum: int, label: str) -> bytes:
    try:
        frame = msgspec.msgpack.encode(message)
    except (TypeError, ValueError, msgspec.EncodeError) as exc:
        raise WorkerTransportProtocolError(
            f"could not encode worker {label}: {exc}"
        ) from exc
    if len(frame) > maximum:
        raise WorkerTransportProtocolError(
            f"worker {label} exceeds {maximum} bytes: {len(frame)}"
        )
    return frame


def _decode_message(
    decoder: msgspec.msgpack.Decoder,
    frame: bytes,
    maximum: int,
    label: str,
) -> Any:
    if len(frame) > maximum:
        raise WorkerTransportProtocolError(
            f"worker {label} exceeds {maximum} bytes: {len(frame)}"
        )
    try:
        return decoder.decode(frame)
    except msgspec.DecodeError as exc:
        raise WorkerTransportProtocolError(
            f"could not decode worker {label}: {exc}"
        ) from exc


def _identity(deployment_epoch: str, engine_id: int, global_rank: int) -> bytes:
    return (
        f"nanodeploy-worker-v{PROTOCOL_VERSION}:"
        f"{deployment_epoch}:e{engine_id}:r{global_rank}"
    ).encode("ascii")


def _remaining_ms(deadline: float, operation: str) -> int:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise WorkerTransportTimeout(f"{operation} timed out")
    return max(1, math.ceil(remaining * 1000))


def _configure_socket(socket: zmq.Socket, max_message_bytes: int) -> None:
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SNDHWM, _SOCKET_HWM)
    socket.setsockopt(zmq.RCVHWM, _SOCKET_HWM)
    socket.setsockopt(zmq.MAXMSGSIZE, max_message_bytes)


def _validate_common(
    message: Any,
    *,
    deployment_epoch: str,
    engine_id: int,
    global_rank: int,
) -> None:
    if message.protocol_version != PROTOCOL_VERSION:
        raise WorkerTransportProtocolError(
            "worker protocol version mismatch: "
            f"expected={PROTOCOL_VERSION}, got={message.protocol_version}"
        )
    if message.deployment_epoch != deployment_epoch:
        raise WorkerTransportProtocolError("worker deployment epoch mismatch")
    if message.engine_id != engine_id:
        raise WorkerTransportProtocolError(
            f"worker engine mismatch: expected={engine_id}, "
            f"got={message.engine_id}"
        )
    if message.global_rank != global_rank:
        raise WorkerTransportProtocolError(
            f"worker rank mismatch: expected={global_rank}, "
            f"got={message.global_rank}"
        )


def _failure_response(
    config: WorkerZmqConfig,
    exc: BaseException,
    *,
    wave_id: int = -1,
    quantum_id: int = -1,
) -> DecodeFailure:
    return DecodeFailure(
        protocol_version=PROTOCOL_VERSION,
        deployment_epoch=config.deployment_epoch,
        engine_id=config.engine_id,
        global_rank=config.global_rank,
        wave_id=wave_id,
        quantum_id=quantum_id,
        exception_type=type(exc).__name__[:256],
        message=str(exc)[:MAX_FAILURE_MESSAGE_CHARS],
        traceback="".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )[:MAX_FAILURE_TRACEBACK_CHARS],
    )


class ZmqWorkerServer:
    """Event-loop-owned ROUTER for one strict-packed LocalEngine."""

    def __init__(
        self,
        *,
        address: str,
        deployment_epoch: str,
        engine_id: int,
        expected_ranks: tuple[int, ...],
        cleanup_dir: str | None = None,
    ) -> None:
        if not expected_ranks or len(set(expected_ranks)) != len(expected_ranks):
            raise ValueError("expected worker ranks must be non-empty and unique")
        self.address = address
        self.deployment_epoch = deployment_epoch
        self.engine_id = engine_id
        self.expected_ranks = expected_ranks
        self.cleanup_dir = cleanup_dir
        self._identities = {
            rank: _identity(deployment_epoch, engine_id, rank)
            for rank in expected_ranks
        }
        self._rank_by_identity = {
            identity: rank for rank, identity in self._identities.items()
        }
        self._context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None
        self._poller: zmq.Poller | None = None
        self._owner_thread_id: int | None = None
        self._ready_ranks: set[int] = set()
        self._inflight: tuple[int, int] | None = None
        self._failed = False

    @classmethod
    def create_ipc(
        cls,
        *,
        engine_id: int,
        expected_ranks: tuple[int, ...],
    ) -> ZmqWorkerServer:
        deployment_epoch = uuid.uuid4().hex
        endpoint_dir = tempfile.mkdtemp(
            prefix=f"nanodeploy-zmq-e{engine_id}-"
        )
        os.chmod(endpoint_dir, 0o700)
        socket_path = os.path.join(endpoint_dir, "worker.sock")
        if len(socket_path.encode()) > _MAX_IPC_PATH_BYTES:
            os.rmdir(endpoint_dir)
            raise WorkerTransportError(
                f"ZMQ IPC path is too long: {socket_path!r}"
            )
        return cls(
            address=f"ipc://{socket_path}",
            deployment_epoch=deployment_epoch,
            engine_id=engine_id,
            expected_ranks=expected_ranks,
            cleanup_dir=endpoint_dir,
        )

    def worker_config(
        self,
        global_rank: int,
        *,
        startup_timeout_s: float,
        quantum_timeout_s: float,
    ) -> WorkerZmqConfig:
        if global_rank not in self._identities:
            raise ValueError(f"unknown worker rank {global_rank}")
        return WorkerZmqConfig(
            address=self.address,
            deployment_epoch=self.deployment_epoch,
            engine_id=self.engine_id,
            global_rank=global_rank,
            startup_timeout_s=startup_timeout_s,
            quantum_timeout_s=quantum_timeout_s,
        )

    def activate(
        self,
        timeout_s: float,
        *,
        liveness_check: Callable[[], None] | None = None,
    ) -> None:
        if self._socket is not None:
            raise WorkerTransportError("worker ZMQ server is already active")
        self._owner_thread_id = threading.get_ident()
        self._context = zmq.Context(io_threads=1)
        self._socket = self._context.socket(zmq.ROUTER)
        _configure_socket(self._socket, MAX_RESPONSE_BYTES)
        self._socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self._poller = zmq.Poller()
        self._poller.register(self._socket, zmq.POLLIN)
        deadline = time.monotonic() + timeout_s
        try:
            self._socket.bind(self.address)
            pending = set(self.expected_ranks)
            while pending:
                identity, frame = self._receive(
                    deadline,
                    "worker READY",
                    liveness_check=liveness_check,
                )
                rank = self._rank_for_identity(identity)
                if rank not in pending:
                    raise WorkerTransportProtocolError(
                        f"duplicate worker READY from rank {rank}"
                    )
                message = decode_response(frame)
                if not isinstance(message, WorkerReady):
                    raise WorkerTransportProtocolError(
                        f"expected worker READY from rank {rank}, "
                        f"got {type(message).__name__}"
                    )
                _validate_common(
                    message,
                    deployment_epoch=self.deployment_epoch,
                    engine_id=self.engine_id,
                    global_rank=rank,
                )
                pending.remove(rank)
                self._ready_ranks.add(rank)
        except BaseException:
            self._failed = True
            self.close(graceful=False, timeout_s=0.0)
            raise

    def send_decode_commands(
        self,
        commands: Mapping[int, DecodeCommand],
        *,
        deadline: float,
    ) -> None:
        self._require_active()
        if self._failed:
            raise WorkerTransportError("worker ZMQ server is failed")
        if self._inflight is not None:
            raise WorkerTransportProtocolError(
                f"worker quantum {self._inflight} is already in flight"
            )
        if set(commands) != set(self.expected_ranks):
            raise WorkerTransportProtocolError(
                "decode command ranks do not match LocalEngine topology"
            )

        encoded: dict[int, bytes] = {}
        quantum_keys: set[tuple[int, int]] = set()
        for rank in self.expected_ranks:
            command = commands[rank]
            _validate_common(
                command,
                deployment_epoch=self.deployment_epoch,
                engine_id=self.engine_id,
                global_rank=rank,
            )
            if command.wave_id <= 0 or command.quantum_id < 0:
                raise WorkerTransportProtocolError(
                    f"invalid worker quantum ({command.wave_id}, "
                    f"{command.quantum_id})"
                )
            if not 0 <= command.transport_slot < TRANSPORT_SLOTS:
                raise WorkerTransportProtocolError(
                    f"invalid worker transport slot {command.transport_slot}"
                )
            expected_slot = command.quantum_id % TRANSPORT_SLOTS
            if command.transport_slot != expected_slot:
                raise WorkerTransportProtocolError(
                    "worker transport slot/quantum mismatch: "
                    f"expected={expected_slot}, got={command.transport_slot}"
                )
            quantum_keys.add((command.wave_id, command.quantum_id))
            encoded[rank] = encode_command(command)
        if len(quantum_keys) != 1:
            raise WorkerTransportProtocolError(
                "workers received inconsistent wave/quantum commands"
            )

        self._inflight = next(iter(quantum_keys))
        sent: list[int] = []
        try:
            for rank in self.expected_ranks:
                self._send(rank, encoded[rank], deadline, "decode command")
                sent.append(rank)
        except BaseException as exc:
            self._failed = True
            raise WorkerTransportError(
                "worker command fan-out failed after ranks "
                f"{tuple(sent)}: {exc}"
            ) from exc

    def receive_decode_results(
        self,
        *,
        deadline: float,
    ) -> tuple[DecodeSuccess, ...]:
        self._require_active()
        if self._inflight is None:
            raise WorkerTransportProtocolError(
                "no worker quantum is in flight"
            )
        wave_id, quantum_id = self._inflight
        pending = set(self.expected_ranks)
        results: dict[int, DecodeSuccess] = {}
        try:
            while pending:
                identity, frame = self._receive(
                    deadline, "worker decode results"
                )
                rank = self._rank_for_identity(identity)
                if rank not in pending:
                    raise WorkerTransportProtocolError(
                        f"duplicate worker result from rank {rank}"
                    )
                message = decode_response(frame)
                _validate_common(
                    message,
                    deployment_epoch=self.deployment_epoch,
                    engine_id=self.engine_id,
                    global_rank=rank,
                )
                if isinstance(message, DecodeFailure):
                    if (message.wave_id, message.quantum_id) not in {
                        (-1, -1),
                        (wave_id, quantum_id),
                    }:
                        raise WorkerTransportProtocolError(
                            f"worker {rank} failure belongs to stale quantum "
                            f"({message.wave_id}, {message.quantum_id})"
                        )
                    raise RemoteWorkerError(
                        f"worker {rank} failed in quantum "
                        f"({wave_id}, {quantum_id}): "
                        f"{message.exception_type}: {message.message}\n"
                        f"{message.traceback}"
                    )
                if not isinstance(message, DecodeSuccess):
                    raise WorkerTransportProtocolError(
                        f"expected decode result from rank {rank}, "
                        f"got {type(message).__name__}"
                    )
                if (
                    message.wave_id != wave_id
                    or message.quantum_id != quantum_id
                ):
                    raise WorkerTransportProtocolError(
                        f"worker {rank} returned quantum "
                        f"({message.wave_id}, {message.quantum_id}); "
                        f"expected ({wave_id}, {quantum_id})"
                    )
                pending.remove(rank)
                results[rank] = message
        except BaseException:
            self._failed = True
            raise
        self._inflight = None
        return tuple(results[rank] for rank in self.expected_ranks)

    def mark_failed(self) -> None:
        self._failed = True

    def close(self, *, graceful: bool, timeout_s: float) -> tuple[str, ...]:
        errors: list[str] = []
        if self._socket is not None:
            self._assert_owner_thread()
            if (
                graceful
                and not self._failed
                and self._inflight is None
                and self._ready_ranks == set(self.expected_ranks)
            ):
                try:
                    self._graceful_stop(timeout_s)
                except BaseException as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
            try:
                if self._poller is not None:
                    self._poller.unregister(self._socket)
            except (KeyError, zmq.ZMQError):
                pass
            self._socket.close(linger=0)
            self._socket = None
            self._poller = None
        if self._context is not None:
            self._context.term()
            self._context = None
        self._cleanup_endpoint()
        return tuple(errors)

    def cleanup_unbound(self) -> None:
        if self._socket is not None:
            raise WorkerTransportError(
                "active worker ZMQ server must be closed by its owner thread"
            )
        self._cleanup_endpoint()

    def _graceful_stop(self, timeout_s: float) -> None:
        deadline = time.monotonic() + max(timeout_s, 0.001)
        for rank in self.expected_ranks:
            frame = encode_command(
                StopCommand(
                    protocol_version=PROTOCOL_VERSION,
                    deployment_epoch=self.deployment_epoch,
                    engine_id=self.engine_id,
                    global_rank=rank,
                )
            )
            self._send(rank, frame, deadline, "worker STOP")
        pending = set(self.expected_ranks)
        while pending:
            identity, frame = self._receive(deadline, "worker STOP ACK")
            rank = self._rank_for_identity(identity)
            if rank not in pending:
                raise WorkerTransportProtocolError(
                    f"duplicate worker STOP ACK from rank {rank}"
                )
            message = decode_response(frame)
            _validate_common(
                message,
                deployment_epoch=self.deployment_epoch,
                engine_id=self.engine_id,
                global_rank=rank,
            )
            if not isinstance(message, WorkerStopped):
                raise WorkerTransportProtocolError(
                    f"expected STOP ACK from rank {rank}, "
                    f"got {type(message).__name__}"
                )
            pending.remove(rank)

    def _require_active(self) -> None:
        if self._socket is None or self._poller is None:
            raise WorkerTransportError("worker ZMQ server is not active")
        self._assert_owner_thread()

    def _assert_owner_thread(self) -> None:
        if self._owner_thread_id != threading.get_ident():
            raise WorkerTransportError(
                "worker ZMQ socket used outside its owner thread"
            )

    def _rank_for_identity(self, identity: bytes) -> int:
        try:
            return self._rank_by_identity[identity]
        except KeyError as exc:
            raise WorkerTransportProtocolError(
                f"unknown worker ZMQ identity {identity!r}"
            ) from exc

    def _send(
        self,
        rank: int,
        frame: bytes,
        deadline: float,
        operation: str,
    ) -> None:
        assert self._socket is not None
        while True:
            try:
                self._socket.send_multipart(
                    [self._identities[rank], frame], flags=zmq.NOBLOCK
                )
                return
            except zmq.Again:
                poller = zmq.Poller()
                poller.register(self._socket, zmq.POLLOUT)
                if not poller.poll(_remaining_ms(deadline, operation)):
                    raise WorkerTransportTimeout(f"{operation} timed out")
            except zmq.ZMQError as exc:
                raise WorkerTransportError(
                    f"{operation} failed for rank {rank}: {exc}"
                ) from exc

    def _receive(
        self,
        deadline: float,
        operation: str,
        *,
        liveness_check: Callable[[], None] | None = None,
    ) -> tuple[bytes, bytes]:
        assert self._socket is not None
        assert self._poller is not None
        while True:
            remaining_ms = _remaining_ms(deadline, operation)
            poll_ms = (
                min(remaining_ms, _IDLE_POLL_MS)
                if liveness_check is not None
                else remaining_ms
            )
            events = dict(self._poller.poll(poll_ms))
            if events.get(self._socket, 0) & zmq.POLLIN:
                break
            if liveness_check is not None:
                liveness_check()
                continue
            raise WorkerTransportTimeout(f"{operation} timed out")
        try:
            frames = self._socket.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again as exc:
            raise WorkerTransportError(
                f"{operation} became unreadable after poll"
            ) from exc
        if len(frames) != 2:
            raise WorkerTransportProtocolError(
                f"{operation} expected identity and payload, got "
                f"{len(frames)} frames"
            )
        return frames[0], frames[1]

    def _cleanup_endpoint(self) -> None:
        if self.cleanup_dir is None:
            return
        socket_path = self.address.removeprefix("ipc://")
        try:
            os.unlink(socket_path)
        except FileNotFoundError:
            pass
        try:
            os.rmdir(self.cleanup_dir)
        except FileNotFoundError:
            pass
        except OSError:
            # A hard-killed peer cannot make this UUID path collide with a
            # future deployment. Leave unexpected residue for inspection.
            pass
        self.cleanup_dir = None


class ZmqWorkerClient:
    """Persistent DEALER loop owned by one ModelRunner actor method."""

    def __init__(self, config: WorkerZmqConfig) -> None:
        if config.startup_timeout_s <= 0 or config.quantum_timeout_s <= 0:
            raise ValueError("worker ZMQ timeouts must be positive")
        self.config = config

    def run(
        self,
        execute: Callable[[DecodeCommand], WorkerExecutionOutput],
    ) -> None:
        context = zmq.Context(io_threads=1)
        socket = context.socket(zmq.DEALER)
        _configure_socket(socket, MAX_COMMAND_BYTES)
        socket.setsockopt(
            zmq.IDENTITY,
            _identity(
                self.config.deployment_epoch,
                self.config.engine_id,
                self.config.global_rank,
            ),
        )
        socket.setsockopt(zmq.IMMEDIATE, 1)
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        last_quantum: tuple[int, int] | None = None
        try:
            socket.connect(self.config.address)
            startup_deadline = (
                time.monotonic() + self.config.startup_timeout_s
            )
            self._wait_writable(socket, startup_deadline, "worker READY")
            self._send(
                socket,
                WorkerReady(
                    protocol_version=PROTOCOL_VERSION,
                    deployment_epoch=self.config.deployment_epoch,
                    engine_id=self.config.engine_id,
                    global_rank=self.config.global_rank,
                ),
                startup_deadline,
                "worker READY",
            )
            while True:
                events = dict(poller.poll(_IDLE_POLL_MS))
                if not events.get(socket, 0) & zmq.POLLIN:
                    continue
                command: WorkerCommand | None = None
                try:
                    frames = socket.recv_multipart(flags=zmq.NOBLOCK)
                    if len(frames) != 1:
                        raise WorkerTransportProtocolError(
                            "worker command expected one frame, got "
                            f"{len(frames)}"
                        )
                    command = decode_command(frames[0])
                    _validate_common(
                        command,
                        deployment_epoch=self.config.deployment_epoch,
                        engine_id=self.config.engine_id,
                        global_rank=self.config.global_rank,
                    )
                    if isinstance(command, StopCommand):
                        self._send(
                            socket,
                            WorkerStopped(
                                protocol_version=PROTOCOL_VERSION,
                                deployment_epoch=(
                                    self.config.deployment_epoch
                                ),
                                engine_id=self.config.engine_id,
                                global_rank=self.config.global_rank,
                            ),
                            time.monotonic()
                            + self.config.quantum_timeout_s,
                            "worker STOP ACK",
                        )
                        return
                    self._validate_quantum(command, last_quantum)
                    output = execute(command)
                    if not isinstance(output, WorkerExecutionOutput):
                        raise TypeError(
                            "worker execution callback must return "
                            "WorkerExecutionOutput"
                        )
                    self._send(
                        socket,
                        DecodeSuccess(
                            protocol_version=PROTOCOL_VERSION,
                            deployment_epoch=self.config.deployment_epoch,
                            engine_id=self.config.engine_id,
                            global_rank=self.config.global_rank,
                            wave_id=command.wave_id,
                            quantum_id=command.quantum_id,
                            token_rows=output.token_rows,
                            worker_end_time=output.worker_end_time,
                            hierarchical_trace=(
                                output.hierarchical_trace
                            ),
                            diagnostic=output.diagnostic,
                        ),
                        time.monotonic()
                        + self.config.quantum_timeout_s,
                        "worker decode result",
                    )
                    last_quantum = (command.wave_id, command.quantum_id)
                except BaseException as exc:
                    wave_id = (
                        command.wave_id
                        if isinstance(command, DecodeCommand)
                        else -1
                    )
                    quantum_id = (
                        command.quantum_id
                        if isinstance(command, DecodeCommand)
                        else -1
                    )
                    self._try_send_failure(
                        socket,
                        exc,
                        wave_id=wave_id,
                        quantum_id=quantum_id,
                    )
                    raise
        finally:
            try:
                poller.unregister(socket)
            except (KeyError, zmq.ZMQError):
                pass
            socket.close(linger=0)
            context.term()

    def _validate_quantum(
        self,
        command: DecodeCommand,
        last_quantum: tuple[int, int] | None,
    ) -> None:
        current = (command.wave_id, command.quantum_id)
        if not 0 <= command.transport_slot < TRANSPORT_SLOTS:
            raise WorkerTransportProtocolError(
                f"invalid worker transport slot {command.transport_slot}"
            )
        expected_slot = command.quantum_id % TRANSPORT_SLOTS
        if command.transport_slot != expected_slot:
            raise WorkerTransportProtocolError(
                "worker transport slot/quantum mismatch: "
                f"expected={expected_slot}, got={command.transport_slot}"
            )
        if command.wave_id <= 0 or command.quantum_id < 0:
            raise WorkerTransportProtocolError(
                f"invalid worker quantum {current}"
            )
        if last_quantum is None:
            if command.quantum_id != 0:
                raise WorkerTransportProtocolError(
                    f"first worker quantum must end in 0, got {current}"
                )
            return
        last_wave, last_id = last_quantum
        if command.wave_id == last_wave:
            expected = (last_wave, last_id + 1)
        elif command.wave_id > last_wave:
            expected = (command.wave_id, 0)
        else:
            expected = (last_wave, last_id + 1)
        if current != expected:
            raise WorkerTransportProtocolError(
                f"out-of-order worker quantum {current}; expected {expected}"
            )

    def _send(
        self,
        socket: zmq.Socket,
        message: WorkerResponse,
        deadline: float,
        operation: str,
    ) -> None:
        frame = encode_response(message)
        while True:
            try:
                socket.send(frame, flags=zmq.NOBLOCK)
                return
            except zmq.Again:
                self._wait_writable(socket, deadline, operation)
            except zmq.ZMQError as exc:
                raise WorkerTransportError(
                    f"{operation} failed: {exc}"
                ) from exc

    def _try_send_failure(
        self,
        socket: zmq.Socket,
        exc: BaseException,
        *,
        wave_id: int,
        quantum_id: int,
    ) -> None:
        try:
            self._send(
                socket,
                _failure_response(
                    self.config,
                    exc,
                    wave_id=wave_id,
                    quantum_id=quantum_id,
                ),
                time.monotonic() + self.config.quantum_timeout_s,
                "worker failure result",
            )
        except BaseException:
            pass

    @staticmethod
    def _wait_writable(
        socket: zmq.Socket,
        deadline: float,
        operation: str,
    ) -> None:
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLOUT)
        events = dict(poller.poll(_remaining_ms(deadline, operation)))
        if not events.get(socket, 0) & zmq.POLLOUT:
            raise WorkerTransportTimeout(f"{operation} timed out")
