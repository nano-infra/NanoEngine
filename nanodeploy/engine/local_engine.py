from __future__ import annotations

import queue
import threading
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import timedelta
from time import perf_counter, time as wall_time, time_ns as wall_time_ns
from typing import Any

import ray
import torch
import torch.distributed as dist

from nanodeploy._cpp import Sequence
from nanodeploy.config import Config
from nanodeploy.engine.hierarchical_contract import (
    AddCommand,
    AddResult,
    AddResultEvent,
    AdmissionReservation,
    AbortResult,
    DecodeITLSample,
    EngineReady,
    FrontendEventBatch,
    FirstScheduleEvent,
    FirstTokenEvent,
    FinishEvent,
    IngressAck,
    HIERARCHICAL_LOOP_COUNT,
    LoadSnapshot,
)
from nanodeploy.engine.frontend_transport import ZmqFrontendServer
from nanodeploy.engine.local_executor import LocalExecutor
from nanodeploy.engine.local_scheduler import LocalScheduler
from nanodeploy.engine.topology import EngineTopology


@dataclass(slots=True)
class _LoopCommand:
    kind: str
    payload: Any = None
    enqueued_at: float = field(default_factory=perf_counter)
    completed: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None


@dataclass(frozen=True, slots=True)
class _IngressAdd:
    command: AddCommand
    sequence: Sequence
    reservation: AdmissionReservation | None = None
    ingress_version: int | None = None
    enqueued_at: float = field(default_factory=perf_counter)


@dataclass(frozen=True, slots=True)
class _PlannedAdmission:
    command: AddCommand
    sequence: Sequence
    reservation: AdmissionReservation | None


@dataclass(slots=True)
class _PendingConsensus:
    reduced: torch.Tensor | None
    work: Any | None
    local_unfinished: bool
    wave_id: int
    quantum_id: int
    arrival_perf_s: float
    arrival_unix_ns: int
    leader_arrival_skew_ms: float = 0.0
    leader_rendezvous_ms: float = 0.0
    late_participant_collective_ms: float | None = None


def _rank_load_payload(snapshot: LoadSnapshot) -> tuple[dict[str, int], ...]:
    return tuple(
        {
            "global_rank": rank_load.global_rank,
            "sp_idx": rank_load.sp_idx,
            "master_batch_size": rank_load.master_batch_size,
            "active_master_requests": rank_load.active_master_requests,
            "active_receiver_requests": (
                rank_load.active_receiver_requests
            ),
            "active_dispatched_tokens": (
                rank_load.active_dispatched_tokens
            ),
            "free_blocks": rank_load.free_blocks,
            "total_blocks": rank_load.total_blocks,
            "control_dummy_blocks": rank_load.control_dummy_blocks,
        }
        for rank_load in snapshot.rank_loads
    )


@ray.remote(num_cpus=0.1, max_concurrency=64)
class LocalEngineCore:
    """Single-writer LocalScheduler plus one DP-group LocalExecutor."""

    def __init__(
        self,
        config: Config,
        topology: EngineTopology,
        workers: list[Any],
    ) -> None:
        self.config = config
        self.topology = topology
        self.engine_id = topology.engine_id
        self.scheduler = LocalScheduler(config, topology)
        self.executor = LocalExecutor(config, topology, workers)

        self._normal_commands: queue.Queue[_LoopCommand] = queue.Queue()
        self._abort_commands: queue.Queue[_LoopCommand] = queue.Queue()
        self._ingress_adds: queue.Queue[_IngressAdd] = queue.Queue()
        self._ingress_head: _IngressAdd | None = None
        self._ingress_replay: deque[_IngressAdd] = deque()
        self._ingress_lock = threading.Lock()
        self._reserved_request_ids: set[int] = set()
        self._ingress_pending_ids: set[int] = set()
        self._admission_pending_ids: set[int] = set()
        self._cancelled_ingress_ids: set[int] = set()
        self._reserved_slots = 0
        self._ingress_version = 0
        self._admission_version = 0
        self._capacity_epoch = 0
        self._add_result_events: deque[AddResultEvent] = deque()
        self._first_schedule_events: deque[FirstScheduleEvent] = deque()
        self._first_token_events: deque[FirstTokenEvent] = deque()
        self._terminal_events: deque[FinishEvent] = deque()
        self._events_lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._state_cv = threading.Condition()
        self._wave_running = False
        self._wave_id = 0
        self._quantum_id = 0
        self._pending_start_wave: int | None = None
        self._stop = False
        self._failure: str | None = None
        self._loop_thread: threading.Thread | None = None
        self._frontend_server: ZmqFrontendServer | None = None
        self._worker_transport_ready = threading.Event()
        self._coordinator: Any | None = None
        self._control_group_initialized = False
        self._worker_identities: tuple[dict[str, Any], ...] = ()
        self._command_count = 0
        self._command_queue_delay_ms_total = 0.0
        self._decode_quantum_count = 0
        self._admission_latency_ms_total = 0.0
        self._schedule_latency_ms_total = 0.0
        self._coordination_latency_ms_total = 0.0
        self._execute_latency_ms_total = 0.0
        self._postprocess_latency_ms_total = 0.0
        self._ingress_queue_delay_ms_total = 0.0
        self._scheduler_add_ms_total = 0.0
        self._local_transient_retries = 0
        self._staged_ingress_depth_max = 0
        self._planned_commit_attempts = 0
        self._decode_itl_ms_weighted_total = 0.0
        self._decode_itl_token_count = 0
        self._decode_itl_samples: list[DecodeITLSample] = []
        self._quantum_diagnostics_enabled = bool(
            config.hierarchical_quantum_diagnostics
        )
        self._quantum_diagnostics: list[dict[str, Any]] = []
        self._quantum_diagnostics_lock = threading.Lock()
        self._cached_load_snapshot = self._build_load_snapshot()

    def initialize(
        self,
        *,
        control_init_method: str | None,
        coordinator: Any | None,
    ) -> EngineReady:
        if self._loop_thread is not None:
            raise RuntimeError("LocalEngineCore is already initialized")
        self.executor.initialize_endpoint(self.config.startup_timeout_s)
        self._worker_identities = self.executor.worker_identities(
            self.config.startup_timeout_s
        )
        expected_ranks = self.topology.global_ranks
        actual_ranks = tuple(
            identity["global_rank"] for identity in self._worker_identities
        )
        if actual_ranks != expected_ranks:
            raise RuntimeError(
                f"engine {self.engine_id} worker rank slice mismatch: "
                f"expected={expected_ranks}, got={actual_ranks}"
            )
        actual_local_ranks = tuple(
            identity["engine_local_rank"]
            for identity in self._worker_identities
        )
        expected_local_ranks = tuple(range(self.topology.world_size))
        if actual_local_ranks != expected_local_ranks:
            raise RuntimeError(
                f"engine {self.engine_id} worker local-rank mismatch: "
                f"expected={expected_local_ranks}, got={actual_local_ranks}"
            )
        expected_fingerprint = self.config.collective_fingerprint()
        worker_fingerprints = {
            identity["config_fingerprint"]
            for identity in self._worker_identities
        }
        if worker_fingerprints != {expected_fingerprint}:
            raise RuntimeError(
                f"engine {self.engine_id} worker config fingerprint mismatch"
            )
        if any(
            len(identity["gpu_ids"]) != 1
            for identity in self._worker_identities
        ):
            raise RuntimeError(
                f"engine {self.engine_id} worker does not own exactly one GPU"
            )
        worker_nodes = tuple(
            str(identity["node_id"]) for identity in self._worker_identities
        )
        if len(set(worker_nodes)) != 1:
            raise RuntimeError(
                f"engine {self.engine_id} is not strict-packed on one node"
            )

        self._coordinator = coordinator
        if self.config.attention_dp > 1:
            if control_init_method is None or coordinator is None:
                raise ValueError(
                    "multi-DP LocalEngineCore requires Gloo and coordinator"
                )
            dist.init_process_group(
                backend="gloo",
                init_method=control_init_method,
                world_size=self.config.attention_dp,
                rank=self.topology.global_dp_idx,
                timeout=timedelta(seconds=self.config.quantum_timeout_s),
            )
            self._control_group_initialized = True
        elif control_init_method is not None or coordinator is not None:
            raise ValueError("DP1 fast path must not create a coordinator")

        actor_node_id = str(ray.get_runtime_context().get_node_id())
        if actor_node_id != worker_nodes[0]:
            raise RuntimeError(
                f"engine {self.engine_id} actor/worker node mismatch"
            )

        self.executor.prepare_worker_transport(self.config.startup_timeout_s)
        self._loop_thread = threading.Thread(
            target=self._event_loop,
            name=f"nanodeploy-local-engine-{self.engine_id}",
            daemon=True,
        )
        self._loop_thread.start()
        if not self._worker_transport_ready.wait(
            timeout=self.config.startup_timeout_s
        ):
            raise TimeoutError(
                f"LocalEngineCore {self.engine_id} worker transport "
                "did not become ready"
            )
        self._raise_if_failed()
        self._frontend_server = ZmqFrontendServer(
            engine_id=self.engine_id,
            advertised_host=ray.util.get_node_ip_address(),
            queue_capacity=self.config.hierarchical_queue_capacity,
            add=self.submit_add,
            enqueue_batch=self.enqueue_add_batch,
            admit_batch=self.admit_add_batch,
        )
        self._frontend_server.start(self.config.startup_timeout_s)
        return EngineReady(
            engine_id=self.engine_id,
            global_ranks=self.topology.global_ranks,
            config_fingerprint=self.config.collective_fingerprint(),
            node_id=actor_node_id,
            worker_node_ids=worker_nodes,
            frontend_address=self._frontend_server.address,
            frontend_epoch=self._frontend_server.deployment_epoch,
        )

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            raise RuntimeError(
                f"LocalEngineCore {self.engine_id} failed: {self._failure}"
            )

    def _submit(
        self, command: _LoopCommand, *, abort_priority: bool = False
    ) -> Any:
        self._raise_if_failed()
        target = (
            self._abort_commands if abort_priority else self._normal_commands
        )
        target.put(command)
        with self._state_cv:
            self._state_cv.notify_all()
        if not command.completed.wait(timeout=self.config.quantum_timeout_s):
            raise TimeoutError(
                f"LocalEngineCore {self.engine_id} command {command.kind} "
                "timed out"
            )
        if command.error is not None:
            raise command.error
        self._raise_if_failed()
        return command.result

    def submit_add(
        self, command: AddCommand, sequence: Sequence
    ) -> AddResult:
        return self._submit(
            _LoopCommand("add", _IngressAdd(command, sequence))
        )

    def enqueue_add(
        self, command: AddCommand, sequence: Sequence
    ) -> IngressAck:
        """Reserve capacity and enqueue without touching LocalScheduler."""
        return self.enqueue_add_batch((command,), (sequence,))[0]

    def enqueue_add_batch(
        self,
        commands: tuple[AddCommand, ...],
        sequences: tuple[Sequence, ...],
    ) -> tuple[IngressAck, ...]:
        """Reserve and enqueue one ingress batch with a single wakeup.

        The receipt only means that LocalEngine owns a bounded lifecycle slot
        and the command is visible in its ingress queue. LocalScheduler state
        is left exclusively to the event-loop thread at the next drain point.
        """
        return self._stage_add_batch(commands, sequences, reservations=None)

    def _stage_add_batch(
        self,
        commands: tuple[AddCommand, ...],
        sequences: tuple[Sequence, ...],
        *,
        reservations: tuple[AdmissionReservation, ...] | None,
    ) -> tuple[IngressAck, ...]:
        self._raise_if_failed()
        if len(commands) != len(sequences):
            raise ValueError("ingress command/Sequence count mismatch")
        if reservations is not None and len(reservations) != len(commands):
            raise ValueError("ingress command/reservation count mismatch")
        if self.config.attention_dp > 1 and self._coordinator is None:
            raise RuntimeError("LocalEngine coordinator is not initialized")
        acks: list[IngressAck] = []
        enqueued_any = False
        with self._ingress_lock:
            for index, (command, sequence) in enumerate(
                zip(commands, sequences, strict=True)
            ):
                reservation = (
                    reservations[index]
                    if reservations is not None
                    else None
                )
                if reservation is not None and (
                    reservation.request_id != command.request_id
                    or reservation.engine_id != self.engine_id
                    or not 0
                    <= reservation.master_sp_idx
                    < self.config.attention_sp
                    or len(reservation.dispatched_tokens)
                    != self.config.attention_sp
                    or any(
                        token_count < 0
                        for token_count in reservation.dispatched_tokens
                    )
                    or sum(reservation.dispatched_tokens)
                    != command.num_tokens
                ):
                    acks.append(
                        IngressAck(
                            request_id=command.request_id,
                            engine_id=self.engine_id,
                            enqueued=False,
                            reason="invalid_admission_reservation",
                        )
                    )
                    continue
                if command.request_id in self._cancelled_ingress_ids:
                    self._cancelled_ingress_ids.discard(command.request_id)
                    acks.append(
                        IngressAck(
                            request_id=command.request_id,
                            engine_id=self.engine_id,
                            enqueued=False,
                            reason="aborted",
                        )
                    )
                    continue
                if (
                    command.request_id in self._reserved_request_ids
                    or command.request_id in self._admission_pending_ids
                ):
                    acks.append(
                        IngressAck(
                            request_id=command.request_id,
                            engine_id=self.engine_id,
                            enqueued=False,
                            reason="duplicate_request_id",
                        )
                    )
                    continue
                if (
                    self._reserved_slots + len(self._admission_pending_ids)
                    >= self.config.hierarchical_queue_capacity
                ):
                    acks.append(
                        IngressAck(
                            request_id=command.request_id,
                            engine_id=self.engine_id,
                            enqueued=False,
                            reason="queue_full",
                        )
                    )
                    continue
                self._reserved_request_ids.add(command.request_id)
                self._ingress_pending_ids.add(command.request_id)
                self._staged_ingress_depth_max = max(
                    self._staged_ingress_depth_max,
                    len(self._ingress_pending_ids),
                )
                self._reserved_slots += 1
                self._ingress_version += 1
                ingress_version = self._ingress_version
                self._ingress_adds.put_nowait(
                    _IngressAdd(
                        command,
                        sequence,
                        reservation=reservation,
                        ingress_version=ingress_version,
                    )
                )
                enqueued_any = True
                acks.append(
                    IngressAck(
                        request_id=command.request_id,
                        engine_id=self.engine_id,
                        enqueued=True,
                        ingress_version=(
                            ingress_version
                            if reservation is not None
                            else None
                        ),
                    )
                )

        # The request is visible in ingress before any wakeup is triggered.
        if enqueued_any and self.config.attention_dp == 1:
            with self._state_cv:
                if not self._wave_running:
                    self._wave_id += 1
                    self._quantum_id = 0
                    self._wave_running = True
                self._state_cv.notify_all()
        elif enqueued_any:
            self._coordinator.first_request.remote(
                self.engine_id, self._wave_id
            )
            with self._state_cv:
                self._state_cv.notify_all()
        return tuple(acks)

    def admit_add(
        self, command: AddCommand, sequence: Sequence
    ) -> IngressAck:
        """Run local SP admission in the scheduler's single-writer loop."""
        return self.admit_add_batch((command,), None, (sequence,))[0]

    def admit_add_batch(
        self,
        commands: tuple[AddCommand, ...],
        reservations: tuple[AdmissionReservation, ...] | None,
        sequences: tuple[Sequence, ...],
    ) -> tuple[IngressAck, ...]:
        """Admit one frontend batch with one transport request."""
        self._raise_if_failed()
        if len(commands) != len(sequences):
            raise ValueError("admission command/Sequence count mismatch")
        if self.config.attention_dp > 1 and self._coordinator is None:
            raise RuntimeError("LocalEngine coordinator is not initialized")
        if reservations is not None and len(reservations) != len(commands):
            raise ValueError(
                "admission command/reservation count mismatch"
            )
        if reservations is not None:
            return self._stage_add_batch(
                commands,
                sequences,
                reservations=reservations,
            )

        immediate: list[IngressAck | None] = [None] * len(commands)
        active_commands: list[AddCommand] = []
        active_sequences: list[Sequence] = []
        active_reservations: list[AdmissionReservation | None] = []
        active_indexes: list[int] = []
        with self._ingress_lock:
            for index, (command, sequence) in enumerate(
                zip(commands, sequences, strict=True)
            ):
                reservation = (
                    reservations[index]
                    if reservations is not None
                    else None
                )
                if (
                    command.request_id in self._reserved_request_ids
                    or command.request_id in self._ingress_pending_ids
                    or command.request_id in self._admission_pending_ids
                ):
                    immediate[index] = IngressAck(
                        request_id=command.request_id,
                        engine_id=self.engine_id,
                        enqueued=False,
                        reason="duplicate_request_id",
                    )
                    continue
                if (
                    self._reserved_slots + len(self._admission_pending_ids)
                    >= self.config.hierarchical_queue_capacity
                ):
                    immediate[index] = IngressAck(
                        request_id=command.request_id,
                        engine_id=self.engine_id,
                        enqueued=False,
                        reason="queue_full",
                    )
                    continue
                self._admission_pending_ids.add(command.request_id)
                active_commands.append(command)
                active_sequences.append(sequence)
                active_reservations.append(reservation)
                active_indexes.append(index)

        if active_commands:
            active_acks = self._submit(
                _LoopCommand(
                    "admit_batch",
                    tuple(
                        _PlannedAdmission(command, sequence, reservation)
                        for command, sequence, reservation in zip(
                            active_commands,
                            active_sequences,
                            active_reservations,
                            strict=True,
                        )
                    ),
                )
            )
            if len(active_acks) != len(active_commands):
                raise RuntimeError(
                    "LocalEngine admission batch result count mismatch"
                )
            for index, ack in zip(
                active_indexes, active_acks, strict=True
            ):
                immediate[index] = ack
        if any(ack is None for ack in immediate):
            raise RuntimeError("LocalEngine admission batch result is missing")
        return tuple(ack for ack in immediate if ack is not None)

    def submit_abort(
        self,
        request_id: int,
        *,
        allow_future_ingress: bool = False,
    ) -> AbortResult:
        with self._ingress_lock:
            if allow_future_ingress:
                # A concurrent abort may reach this method before the
                # corresponding fast enqueue. Keep a bounded cancellation
                # tombstone so that enqueue and abort have an atomic outcome
                # under the same lock regardless of actor-call ordering.
                self._cancelled_ingress_ids.add(request_id)
            else:
                # A post-receipt reconciliation call proves that the enqueue
                # RPC has completed, so an obsolete future-ingress tombstone
                # can no longer be needed.
                self._cancelled_ingress_ids.discard(request_id)
            if (
                request_id in self._ingress_pending_ids
                or request_id in self._admission_pending_ids
            ):
                self._cancelled_ingress_ids.add(request_id)
                return AbortResult(
                    request_id=request_id, status="abort_pending"
                )
        result = self._submit(
            _LoopCommand("abort", request_id), abort_priority=True
        )
        if allow_future_ingress and result.status != "not_found":
            with self._ingress_lock:
                self._cancelled_ingress_ids.discard(request_id)
        if allow_future_ingress and result.status == "not_found":
            return AbortResult(request_id=request_id, status="abort_pending")
        return result

    def clear_ingress_abort(self, request_id: int) -> None:
        """Forget a future-ingress tombstone after a negative receipt."""
        with self._ingress_lock:
            self._cancelled_ingress_ids.discard(request_id)

    def get_load(self) -> LoadSnapshot:
        return self._submit(_LoopCommand("load"))

    def get_cached_load(self) -> LoadSnapshot:
        self._raise_if_failed()
        with self._load_lock:
            return self._cached_load_snapshot

    def drain_add_results(self) -> tuple[AddResultEvent, ...]:
        self._raise_if_failed()
        with self._events_lock:
            events = tuple(self._add_result_events)
            self._add_result_events.clear()
        return events

    def drain_first_token_events(self) -> tuple[FirstTokenEvent, ...]:
        self._raise_if_failed()
        with self._events_lock:
            events = tuple(self._first_token_events)
            self._first_token_events.clear()
        return events

    def drain_first_schedule_events(
        self,
    ) -> tuple[FirstScheduleEvent, ...]:
        self._raise_if_failed()
        with self._events_lock:
            events = tuple(self._first_schedule_events)
            self._first_schedule_events.clear()
        return events

    def drain_events(self) -> tuple[FinishEvent, ...]:
        self._raise_if_failed()
        with self._events_lock:
            events = tuple(self._terminal_events)
            self._terminal_events.clear()
        return events

    def drain_frontend_events(self) -> FrontendEventBatch:
        """Drain all frontend events and attach the latest cached load."""
        self.health()
        with self._events_lock:
            add_results = tuple(self._add_result_events)
            first_schedule_events = tuple(self._first_schedule_events)
            first_token_events = tuple(self._first_token_events)
            finish_events = tuple(self._terminal_events)
            self._add_result_events.clear()
            self._first_schedule_events.clear()
            self._first_token_events.clear()
            self._terminal_events.clear()
        with self._load_lock:
            load = self._cached_load_snapshot
        return FrontendEventBatch(
            engine_id=self.engine_id,
            load=load,
            add_results=add_results,
            first_schedule_events=first_schedule_events,
            first_token_events=first_token_events,
            finish_events=finish_events,
        )

    def drain_execution_traces(self) -> tuple[dict[str, Any], ...]:
        self._raise_if_failed()
        return self.executor.drain_execution_traces()

    def get_decode_itl_samples(self) -> tuple[DecodeITLSample, ...]:
        self._raise_if_failed()
        return tuple(self._decode_itl_samples)

    def drain_quantum_diagnostics(self) -> tuple[dict[str, Any], ...]:
        self._raise_if_failed()
        if not self._quantum_diagnostics_enabled:
            return ()
        with self._quantum_diagnostics_lock:
            samples = tuple(self._quantum_diagnostics)
            self._quantum_diagnostics.clear()
        return samples

    def get_execution_boundary_metrics(self) -> dict[str, float | int]:
        self._raise_if_failed()
        return self.executor.execution_boundary_metrics()

    def reset_execution_boundary_metrics(self) -> None:
        self._raise_if_failed()
        self.executor.reset_execution_boundary_metrics()

    def health(self) -> bool:
        self._raise_if_failed()
        self.executor.check_worker_liveness()
        if self._frontend_server is None:
            raise RuntimeError(
                f"LocalEngineCore {self.engine_id} frontend is not ready"
            )
        self._frontend_server.raise_if_failed()
        if self._loop_thread is None or not self._loop_thread.is_alive():
            raise RuntimeError(
                f"LocalEngineCore {self.engine_id} event loop is not alive"
            )
        return True

    def start_wave(self, wave_id: int) -> None:
        if wave_id <= 0:
            raise ValueError("wave_id must be positive")
        self._raise_if_failed()
        with self._state_cv:
            if wave_id <= self._wave_id:
                return
            if self._wave_running:
                self._pending_start_wave = max(
                    wave_id, self._pending_start_wave or wave_id
                )
            else:
                self._wave_id = wave_id
                self._quantum_id = 0
                self._wave_running = True
            self._state_cv.notify_all()

    def _complete_command(self, command: _LoopCommand) -> None:
        if command.kind == "admit":
            self._complete_admission_commands((command,))
            return
        if command.kind == "admit_batch":
            children = tuple(
                _LoopCommand(
                    "admit",
                    planned_admission,
                    enqueued_at=command.enqueued_at,
                )
                for planned_admission in command.payload
            )
            self._complete_admission_commands(children)
            error = next(
                (
                    child.error
                    for child in children
                    if child.error is not None
                ),
                None,
            )
            if error is not None:
                command.error = error
            else:
                command.result = tuple(
                    child.result for child in children
                )
            command.completed.set()
            return
        self._command_count += 1
        self._command_queue_delay_ms_total += (
            perf_counter() - command.enqueued_at
        ) * 1000
        try:
            if command.kind == "add":
                ingress = command.payload
                command.result = self.scheduler.add(
                    ingress.command, ingress.sequence
                )
                if (
                    command.result.accepted
                    and self.config.attention_dp == 1
                    and not self._wave_running
                ):
                    with self._state_cv:
                        self._wave_id += 1
                        self._quantum_id = 0
                        self._wave_running = True
            elif command.kind == "abort":
                command.result = self.scheduler.abort(command.payload)
                self._publish_events(self.scheduler.drain_terminal_events())
            elif command.kind == "load":
                command.result = self._build_load_snapshot()
            else:
                raise RuntimeError(f"unknown LocalEngine command {command.kind}")
        except BaseException as exc:
            command.error = exc
        finally:
            command.completed.set()

    def _complete_admission_commands(
        self, commands: tuple[_LoopCommand, ...]
    ) -> None:
        if not commands:
            return
        now = perf_counter()
        command_queue_ms = tuple(
            (now - command.enqueued_at) * 1000
            for command in commands
        )
        self._command_count += len(commands)
        self._command_queue_delay_ms_total += sum(command_queue_ms)
        planned_admissions = tuple(command.payload for command in commands)
        add_commands = tuple(
            planned.command for planned in planned_admissions
        )
        sequences = tuple(
            planned.sequence for planned in planned_admissions
        )
        reservations = tuple(
            planned.reservation for planned in planned_admissions
        )
        try:
            with self._ingress_lock:
                cancelled_ids = {
                    add_command.request_id
                    for add_command in add_commands
                    if add_command.request_id
                    in self._cancelled_ingress_ids
                }
                self._cancelled_ingress_ids.difference_update(
                    cancelled_ids
                )
            active_add_commands = tuple(
                add_command
                for add_command in add_commands
                if add_command.request_id not in cancelled_ids
            )
            active_indexes = tuple(
                index
                for index, add_command in enumerate(add_commands)
                if add_command.request_id not in cancelled_ids
            )
            active_reservations = tuple(
                reservations[index] for index in active_indexes
            )
            active_sequences = tuple(
                sequences[index] for index in active_indexes
            )
            if any(
                reservation is not None
                for reservation in active_reservations
            ):
                raise RuntimeError(
                    "planned reservations must use staged ingress"
                )
            active_results = tuple(
                self.scheduler.try_admit_batch(
                    active_add_commands, active_sequences
                )
            )
            if len(active_results) != len(active_add_commands):
                raise RuntimeError(
                    "LocalScheduler active admission result count mismatch: "
                    f"commands={len(active_add_commands)}, "
                    f"results={len(active_results)}"
                )
            active_results_by_id = {
                result.request_id: result for result in active_results
            }
            if len(active_results_by_id) != len(active_results):
                raise RuntimeError(
                    "LocalScheduler returned duplicate admission results"
                )
            results = tuple(
                AddResult(
                    request_id=add_command.request_id,
                    accepted=False,
                    engine_id=self.engine_id,
                    reason="aborted",
                )
                if add_command.request_id in cancelled_ids
                else active_results_by_id[add_command.request_id]
                for add_command in add_commands
            )
            if len(results) != len(commands):
                raise RuntimeError(
                    "LocalScheduler admission result count mismatch: "
                    f"commands={len(commands)}, results={len(results)}"
                )

            admission_versions: list[int | None] = []
            accepted_events: list[AddResultEvent] = []
            with self._ingress_lock:
                for add_command, result in zip(
                    add_commands, results, strict=True
                ):
                    if result.request_id != add_command.request_id:
                        raise RuntimeError(
                            "LocalScheduler admission request mismatch: "
                            f"expected={add_command.request_id}, "
                            f"got={result.request_id}"
                        )
                    self._admission_pending_ids.discard(
                        add_command.request_id
                    )
                    if not result.accepted:
                        admission_versions.append(None)
                        continue
                    if (
                        add_command.request_id
                        in self._reserved_request_ids
                    ):
                        raise RuntimeError(
                            "duplicate lifecycle reservation after "
                            f"admission: {add_command.request_id}"
                        )
                    self._reserved_request_ids.add(
                        add_command.request_id
                    )
                    self._reserved_slots += 1
                    self._admission_version += 1
                    admission_versions.append(self._admission_version)
                    accepted_events.append(
                        AddResultEvent(
                            request_id=add_command.request_id,
                            engine_id=self.engine_id,
                            accepted=True,
                            admission_version=self._admission_version,
                        )
                    )

            local_admission_ms = (perf_counter() - now) * 1000
            if accepted_events:
                queue_ms_by_id = {
                    add_command.request_id: queue_ms
                    for add_command, queue_ms in zip(
                        add_commands, command_queue_ms, strict=True
                    )
                }
                accepted_events = [
                    replace(
                        event,
                        local_planned_queue_ms=queue_ms_by_id[
                            event.request_id
                        ],
                        local_admission_ms=local_admission_ms,
                    )
                    for event in accepted_events
                ]
            for (
                command,
                add_command,
                result,
                admission_version,
                queue_ms,
            ) in zip(
                commands,
                add_commands,
                results,
                admission_versions,
                command_queue_ms,
                strict=True,
            ):
                command.result = IngressAck(
                    request_id=add_command.request_id,
                    engine_id=self.engine_id,
                    enqueued=result.accepted,
                    reason=result.reason,
                    admission_version=admission_version,
                    capacity_epoch=self._capacity_epoch,
                    local_command_queue_ms=queue_ms,
                    local_admission_ms=local_admission_ms,
                )
            if accepted_events:
                with self._events_lock:
                    self._add_result_events.extend(accepted_events)
                if self.config.attention_dp == 1:
                    with self._state_cv:
                        if not self._wave_running:
                            self._wave_id += 1
                            self._quantum_id = 0
                            self._wave_running = True
                        self._state_cv.notify_all()
                else:
                    self._coordinator.first_request.remote(
                        self.engine_id, self._wave_id
                    )
        except BaseException as exc:
            for command in commands:
                command.error = exc
        finally:
            with self._ingress_lock:
                for add_command in add_commands:
                    self._admission_pending_ids.discard(
                        add_command.request_id
                    )
            for command in commands:
                command.completed.set()

    def _drain_queue(self, source: queue.Queue[_LoopCommand]) -> None:
        admission_commands: list[_LoopCommand] = []
        while True:
            try:
                command = source.get_nowait()
            except queue.Empty:
                break
            if command.kind == "admit":
                admission_commands.append(command)
                continue
            if admission_commands:
                self._complete_admission_commands(
                    tuple(admission_commands)
                )
                admission_commands.clear()
            self._complete_command(command)
        if admission_commands:
            self._complete_admission_commands(tuple(admission_commands))

    def _release_reservation(self, request_id: int) -> None:
        with self._ingress_lock:
            self._ingress_pending_ids.discard(request_id)
            if request_id not in self._reserved_request_ids:
                return
            self._reserved_request_ids.remove(request_id)
            self._reserved_slots -= 1
            self._admission_version += 1
            self._capacity_epoch += 1
            if self._reserved_slots < 0:
                raise RuntimeError("negative LocalEngine ingress reservation")

    def _take_ingress(self) -> _IngressAdd | None:
        if self._ingress_head is not None:
            ingress = self._ingress_head
            self._ingress_head = None
            return ingress
        if self._ingress_replay:
            return self._ingress_replay.popleft()
        try:
            return self._ingress_adds.get_nowait()
        except queue.Empty:
            return None

    def _restore_ingress_prefix(
        self, ingresses: tuple[_IngressAdd, ...]
    ) -> None:
        if not ingresses:
            return
        if self._ingress_head is not None:
            raise RuntimeError("LocalEngine ingress retry head is already occupied")
        self._ingress_head = ingresses[0]
        self._ingress_replay.extendleft(reversed(ingresses[1:]))

    def _cancel_uncommitted_ingress(
        self,
        ingress: _IngressAdd,
        cancelled_events: list[FinishEvent],
    ) -> bool:
        request_id = ingress.command.request_id
        with self._ingress_lock:
            if request_id not in self._cancelled_ingress_ids:
                return False
            self._cancelled_ingress_ids.discard(request_id)
            self._ingress_pending_ids.discard(request_id)
        cancelled_events.append(
            FinishEvent(
                request_id=request_id,
                generated_count=0,
                status="ABORTED",
                engine_id=self.engine_id,
            )
        )
        return True

    def _reconcile_committed_ingress_abort(
        self,
        ingress: _IngressAdd,
        result: AddResult,
        cancelled_events: list[FinishEvent],
    ) -> None:
        request_id = ingress.command.request_id
        if result.accepted:
            abort_result = self.scheduler.abort(request_id)
            scheduler_events = self.scheduler.drain_terminal_events()
            if (
                abort_result.status != "aborted"
                or len(scheduler_events) != 1
                or scheduler_events[0].request_id != request_id
                or scheduler_events[0].status != "ABORTED"
            ):
                raise RuntimeError(
                    "LocalScheduler failed to reconcile an ingress abort "
                    f"after commit: request={request_id}, "
                    f"result={abort_result}, events={scheduler_events}"
                )
            cancelled_events.extend(scheduler_events)
            return
        cancelled_events.append(
            FinishEvent(
                request_id=request_id,
                generated_count=0,
                status="ABORTED",
                engine_id=self.engine_id,
            )
        )

    def _commit_planned_ingress_batch(
        self,
        ingresses: tuple[_IngressAdd, ...],
        results: list[AddResultEvent],
        cancelled_events: list[FinishEvent],
    ) -> tuple[int, tuple[_IngressAdd, ...]]:
        active_ingresses: list[_IngressAdd] = []
        processed = 0
        for ingress in ingresses:
            if self._cancel_uncommitted_ingress(
                ingress, cancelled_events
            ):
                processed += 1
            else:
                active_ingresses.append(ingress)
        if not active_ingresses:
            return processed, ()

        active = tuple(active_ingresses)
        add_begin = perf_counter()
        self._planned_commit_attempts += len(active)
        committed = tuple(
            self.scheduler.commit_planned_batch(
                tuple(ingress.command for ingress in active),
                tuple(
                    ingress.reservation
                    for ingress in active
                    if ingress.reservation is not None
                ),
                tuple(ingress.sequence for ingress in active),
            )
        )
        if len(committed) != len(active):
            raise RuntimeError(
                "LocalScheduler planned ingress result count mismatch: "
                f"commands={len(active)}, results={len(committed)}"
            )
        local_admission_ms = (perf_counter() - add_begin) * 1000
        self._scheduler_add_ms_total += local_admission_ms

        cancelled_after_commit: set[int] = set()
        retry_index: int | None = None
        admission_versions: list[int | None] = [None] * len(active)
        with self._ingress_lock:
            for index, (ingress, result) in enumerate(
                zip(active, committed, strict=True)
            ):
                request_id = ingress.command.request_id
                if result.request_id != request_id:
                    raise RuntimeError(
                        "LocalScheduler ingress result request mismatch: "
                        f"expected={request_id}, got={result.request_id}"
                    )
                cancelled = request_id in self._cancelled_ingress_ids
                if cancelled:
                    self._cancelled_ingress_ids.discard(request_id)
                    cancelled_after_commit.add(request_id)
                transient_mismatch = (
                    not result.accepted
                    and result.reason == "admission_state_mismatch"
                )
                if (
                    retry_index is None
                    and transient_mismatch
                    and not cancelled
                ):
                    retry_index = index
                deferred = (
                    retry_index is not None
                    and index >= retry_index
                    and not cancelled
                )
                if deferred and result.accepted:
                    raise RuntimeError(
                        "LocalScheduler accepted planned ingress after a "
                        "state mismatch"
                    )
                if not deferred:
                    self._ingress_pending_ids.discard(request_id)
                if not deferred and not cancelled and result.accepted:
                    self._admission_version += 1
                    admission_versions[index] = self._admission_version

        retry_suffix: tuple[_IngressAdd, ...] = ()
        if retry_index is not None:
            self._local_transient_retries += 1
            retry_suffix = tuple(
                ingress
                for ingress in active[retry_index:]
                if ingress.command.request_id
                not in cancelled_after_commit
            )

        for index, (ingress, result) in enumerate(
            zip(active, committed, strict=True)
        ):
            request_id = ingress.command.request_id
            cancelled = request_id in cancelled_after_commit
            deferred = (
                retry_index is not None
                and index >= retry_index
                and not cancelled
            )
            if deferred:
                continue
            processed += 1
            staged_queue_ms = max(
                0.0, (add_begin - ingress.enqueued_at) * 1000
            )
            self._ingress_queue_delay_ms_total += staged_queue_ms
            if cancelled:
                self._reconcile_committed_ingress_abort(
                    ingress, result, cancelled_events
                )
                continue
            event = AddResultEvent(
                request_id=result.request_id,
                engine_id=self.engine_id,
                accepted=result.accepted,
                reason=result.reason,
                admission_version=admission_versions[index],
                local_planned_queue_ms=staged_queue_ms,
                local_admission_ms=local_admission_ms,
            )
            results.append(event)
            if not result.accepted:
                self._release_reservation(request_id)
        return processed, retry_suffix

    def _drain_ingress(self) -> None:
        begin = perf_counter()
        drain_budget_ms = self.config.max_ingress_drain_ms
        results: list[AddResultEvent] = []
        cancelled_events: list[FinishEvent] = []
        processed = 0
        while processed < self.config.max_ingress_batch_requests:
            if (
                processed > 0
                and drain_budget_ms > 0
                and (perf_counter() - begin) * 1000
                >= drain_budget_ms
            ):
                break
            ingress = self._take_ingress()
            if ingress is None:
                break
            command = ingress.command
            if ingress.reservation is not None:
                planned = [ingress]
                while processed + len(planned) < (
                    self.config.max_ingress_batch_requests
                ):
                    if (
                        drain_budget_ms > 0
                        and (perf_counter() - begin) * 1000
                        >= drain_budget_ms
                    ):
                        break
                    next_ingress = self._take_ingress()
                    if next_ingress is None:
                        break
                    if next_ingress.reservation is None:
                        self._restore_ingress_prefix((next_ingress,))
                        break
                    planned.append(next_ingress)
                committed_count, retry_suffix = (
                    self._commit_planned_ingress_batch(
                        tuple(planned), results, cancelled_events
                    )
                )
                processed += committed_count
                if retry_suffix:
                    self._restore_ingress_prefix(retry_suffix)
                    break
                continue

            if self._cancel_uncommitted_ingress(
                ingress, cancelled_events
            ):
                processed += 1
                continue
            add_begin = perf_counter()
            try:
                result = self.scheduler.add(command, ingress.sequence)
            except BaseException as exc:
                result = AddResult(
                    request_id=command.request_id,
                    accepted=False,
                    engine_id=self.engine_id,
                    reason=f"{type(exc).__name__}: {exc}",
                )
            if result.request_id != command.request_id:
                raise RuntimeError(
                    "LocalScheduler ingress result request mismatch: "
                    f"expected={command.request_id}, got={result.request_id}"
                )
            local_admission_ms = (perf_counter() - add_begin) * 1000
            self._scheduler_add_ms_total += local_admission_ms
            with self._ingress_lock:
                cancelled = (
                    command.request_id in self._cancelled_ingress_ids
                )
                if cancelled:
                    self._cancelled_ingress_ids.discard(command.request_id)
                # This is the atomic ingress-to-scheduler ownership handoff.
                # An abort after this point cannot leave a stale tombstone:
                # submit_abort will enqueue a normal scheduler abort.
                self._ingress_pending_ids.discard(command.request_id)
                admission_version = None

            processed += 1
            staged_queue_ms = max(
                0.0, (add_begin - ingress.enqueued_at) * 1000
            )
            self._ingress_queue_delay_ms_total += staged_queue_ms
            if cancelled:
                self._reconcile_committed_ingress_abort(
                    ingress, result, cancelled_events
                )
                continue
            event = AddResultEvent(
                request_id=result.request_id,
                engine_id=self.engine_id,
                accepted=result.accepted,
                reason=result.reason,
                admission_version=admission_version,
                local_planned_queue_ms=(
                    staged_queue_ms
                    if ingress.reservation is not None
                    else None
                ),
                local_admission_ms=local_admission_ms,
            )
            results.append(event)
            if not result.accepted:
                self._release_reservation(command.request_id)
        if results:
            with self._events_lock:
                self._add_result_events.extend(results)
        self._publish_events(tuple(cancelled_events))

    def _build_load_snapshot(
        self,
        scheduler_snapshot: LoadSnapshot | None = None,
    ) -> LoadSnapshot:
        snapshot = scheduler_snapshot
        if snapshot is None:
            snapshot = self.scheduler.load_snapshot(
                wave_id=self._wave_id,
                quantum_id=self._quantum_id,
            )
        elif (
            snapshot.engine_id != self.engine_id
            or snapshot.wave_id != self._wave_id
            or snapshot.quantum_id != self._quantum_id
        ):
            raise RuntimeError(
                "frozen LocalScheduler load identity mismatch: "
                f"expected=({self.engine_id}, {self._wave_id}, "
                f"{self._quantum_id}), got=({snapshot.engine_id}, "
                f"{snapshot.wave_id}, {snapshot.quantum_id})"
            )
        with self._ingress_lock:
            pending_ingress = (
                len(self._ingress_pending_ids)
                + len(self._admission_pending_ids)
            )
            reserved_slots = self._reserved_slots
            ingress_version = self._ingress_version
        with self._events_lock:
            pending_add_results = len(self._add_result_events)
        return replace(
            snapshot,
            command_count=self._command_count,
            command_queue_delay_ms_total=(
                self._command_queue_delay_ms_total
            ),
            decode_quantum_count=self._decode_quantum_count,
            admission_latency_ms_total=self._admission_latency_ms_total,
            schedule_latency_ms_total=self._schedule_latency_ms_total,
            coordination_latency_ms_total=(
                self._coordination_latency_ms_total
            ),
            execute_latency_ms_total=self._execute_latency_ms_total,
            ray_get_latency_ms_total=(
                self.executor.ray_get_latency_ms_total
            ),
            ray_get_latency_ms_max=(
                self.executor.ray_get_latency_ms_max
            ),
            worker_result_wait_latency_ms_total=(
                self.executor.worker_result_wait_latency_ms_total
            ),
            worker_result_wait_latency_ms_max=(
                self.executor.worker_result_wait_latency_ms_max
            ),
            result_rebuild_latency_ms_total=(
                self.executor.result_rebuild_latency_ms_total
            ),
            result_rebuild_latency_ms_max=(
                self.executor.result_rebuild_latency_ms_max
            ),
            result_rebuild_sample_count=(
                self.executor.result_rebuild_sample_count
            ),
            result_index_latency_ms_total=(
                self.executor.result_index_latency_ms_total
            ),
            result_validate_latency_ms_total=(
                self.executor.result_validate_latency_ms_total
            ),
            result_pack_latency_ms_total=(
                self.executor.result_pack_latency_ms_total
            ),
            postprocess_latency_ms_total=self._postprocess_latency_ms_total,
            pending_ingress=pending_ingress,
            pending_add_results=pending_add_results,
            reserved_slots=reserved_slots,
            ingress_version=ingress_version,
            admission_version=self._admission_version,
            capacity_epoch=self._capacity_epoch,
            ingress_queue_delay_ms_total=(
                self._ingress_queue_delay_ms_total
            ),
            scheduler_add_ms_total=self._scheduler_add_ms_total,
            local_transient_retries=self._local_transient_retries,
            staged_ingress_depth_max=self._staged_ingress_depth_max,
            planned_commit_attempts=self._planned_commit_attempts,
            payload_retry_bytes=0,
            decode_itl_ms_weighted_total=(
                self._decode_itl_ms_weighted_total
            ),
            decode_itl_token_count=self._decode_itl_token_count,
            decode_itl_sample_count=len(self._decode_itl_samples),
        )

    def _refresh_cached_load(
        self,
        scheduler_snapshot: LoadSnapshot | None = None,
    ) -> LoadSnapshot:
        snapshot = self._build_load_snapshot(scheduler_snapshot)
        with self._load_lock:
            self._cached_load_snapshot = snapshot
        return snapshot

    def _start_consensus(
        self,
        local_unfinished: bool,
        *,
        wave_id: int,
        quantum_id: int,
    ) -> _PendingConsensus:
        arrival_perf_s = perf_counter()
        arrival_unix_ns = wall_time_ns()
        if self.config.attention_dp == 1:
            return _PendingConsensus(
                reduced=None,
                work=None,
                local_unfinished=local_unfinished,
                wave_id=wave_id,
                quantum_id=quantum_id,
                arrival_perf_s=arrival_perf_s,
                arrival_unix_ns=arrival_unix_ns,
                late_participant_collective_ms=0.0,
            )
        unfinished = int(local_unfinished)
        # MAX over x and -x yields the global maximum and negative minimum.
        local = torch.tensor(
            [
                wave_id,
                -wave_id,
                quantum_id,
                -quantum_id,
                unfinished,
                -unfinished,
                arrival_unix_ns,
                -arrival_unix_ns,
            ],
            dtype=torch.int64,
            device="cpu",
        )
        reduced = local.clone()
        work = dist.all_reduce(
            reduced,
            op=dist.ReduceOp.MAX,
            async_op=True,
        )
        return _PendingConsensus(
            reduced=reduced,
            work=work,
            local_unfinished=local_unfinished,
            wave_id=wave_id,
            quantum_id=quantum_id,
            arrival_perf_s=arrival_perf_s,
            arrival_unix_ns=arrival_unix_ns,
        )

    def _finish_consensus(self, pending: _PendingConsensus) -> bool:
        if pending.work is None:
            return pending.local_unfinished
        pending.work.wait()
        consensus_finished = perf_counter()
        reduced = pending.reduced
        if reduced is None:
            raise RuntimeError("distributed consensus is missing its result")
        minimum = [-reduced[1].item(), -reduced[3].item(), -reduced[5].item()]
        maximum = [reduced[0].item(), reduced[2].item(), reduced[4].item()]
        if minimum[0] != maximum[0] or minimum[1] != maximum[1]:
            raise RuntimeError(
                "LocalEngine leader wave/quantum mismatch: "
                f"min={minimum}, max={maximum}"
            )
        latest_arrival_ns = int(reduced[6].item())
        earliest_arrival_ns = -int(reduced[7].item())
        if earliest_arrival_ns > latest_arrival_ns:
            raise RuntimeError(
                "LocalEngine leader arrival range is invalid: "
                f"min={earliest_arrival_ns}, max={latest_arrival_ns}"
            )
        pending.leader_arrival_skew_ms = (
            latest_arrival_ns - earliest_arrival_ns
        ) / 1_000_000
        pending.leader_rendezvous_ms = max(
            0.0, (consensus_finished - pending.arrival_perf_s) * 1000
        )
        if pending.arrival_unix_ns == latest_arrival_ns:
            pending.late_participant_collective_ms = (
                pending.leader_rendezvous_ms
            )
        return bool(maximum[2])

    def _consensus(self, local_unfinished: bool) -> bool:
        return self._finish_consensus(
            self._start_consensus(
                local_unfinished,
                wave_id=self._wave_id,
                quantum_id=self._quantum_id,
            )
        )

    def _pause_wave(self) -> None:
        with self._state_cv:
            self._wave_running = False
            if (
                self._pending_start_wave is not None
                and self._pending_start_wave > self._wave_id
            ):
                self._wave_id = self._pending_start_wave
                self._quantum_id = 0
                self._wave_running = True
            self._pending_start_wave = None
            self._state_cv.notify_all()
        self._refresh_cached_load()

    def _publish_events(self, events: tuple[FinishEvent, ...]) -> None:
        if not events:
            return
        for event in events:
            self._release_reservation(event.request_id)
        with self._events_lock:
            self._terminal_events.extend(events)

    def _publish_first_token_events(
        self, events: tuple[FirstTokenEvent, ...]
    ) -> None:
        if not events:
            return
        with self._events_lock:
            self._first_token_events.extend(events)

    def _publish_first_schedule_events(
        self, events: tuple[FirstScheduleEvent, ...]
    ) -> None:
        if not events:
            return
        with self._events_lock:
            self._first_schedule_events.extend(events)

    def _fail_pending_commands(self, error: RuntimeError) -> None:
        for source in (self._abort_commands, self._normal_commands):
            while True:
                try:
                    command = source.get_nowait()
                except queue.Empty:
                    break
                command.error = error
                command.completed.set()

    def _event_loop(self) -> None:
        try:
            self.executor.activate_worker_transport(
                self.config.startup_timeout_s
            )
            self._worker_transport_ready.set()
            while True:
                self._drain_queue(self._abort_commands)
                ingress_drain_begin = perf_counter()
                self._drain_ingress()
                ingress_drain_ms = (
                    perf_counter() - ingress_drain_begin
                ) * 1000
                self._drain_queue(self._normal_commands)
                # Close the ingress-drain/plan gap for aborts submitted after
                # the first priority drain but before scheduler admission.
                self._drain_queue(self._abort_commands)
                with self._state_cv:
                    if self._stop:
                        return
                    wave_running = self._wave_running
                    if wave_running:
                        wave_id = self._wave_id
                        quantum_id = self._quantum_id
                if not wave_running:
                    # Idle refreshes publish newly drained ingress. Active
                    # quantums publish the snapshot frozen by plan_decode(),
                    # avoiding another full live-record traversal here.
                    self._refresh_cached_load()
                    with self._state_cv:
                        if self._stop:
                            return
                        if not self._wave_running:
                            self._state_cv.wait(timeout=0.1)
                            continue
                        wave_id = self._wave_id
                        quantum_id = self._quantum_id

                quantum_begin = perf_counter()
                quantum_started_at_unix_s = wall_time()
                begin = perf_counter()
                self.scheduler.admit()
                admission_latency_ms = (perf_counter() - begin) * 1000
                self._admission_latency_ms_total += admission_latency_ms
                local_unfinished = not self.scheduler.is_finished()
                consensus_started = perf_counter()
                pending_consensus = self._start_consensus(
                    local_unfinished,
                    wave_id=wave_id,
                    quantum_id=quantum_id,
                )
                try:
                    begin = perf_counter()
                    batch = self.scheduler.plan_decode(
                        wave_id=wave_id, quantum_id=quantum_id
                    )
                    schedule_latency_ms = (perf_counter() - begin) * 1000
                    self._schedule_latency_ms_total += schedule_latency_ms
                    frozen_load_snapshot = batch.frozen_load_snapshot
                    if frozen_load_snapshot is None:
                        raise RuntimeError(
                            "planned decode batch is missing its frozen load"
                        )
                    pre_execute_snapshot = self._refresh_cached_load(
                        frozen_load_snapshot
                    )
                except BaseException:
                    # Both leaders have already entered the same collective.
                    # Reap it before fail-stop teardown so no Gloo Work is
                    # abandoned while preserving the original planning error.
                    try:
                        self._finish_consensus(pending_consensus)
                    except BaseException:
                        pass
                    raise
                consensus_wait_begin = perf_counter()
                global_unfinished = self._finish_consensus(
                    pending_consensus
                )
                consensus_finished = perf_counter()
                coordination_latency_ms = (
                    consensus_finished - consensus_wait_begin
                ) * 1000
                if pending_consensus.work is None:
                    consensus_overlap_window_ms = 0.0
                else:
                    consensus_overlap_window_ms = (
                        consensus_wait_begin - consensus_started
                    ) * 1000
                self._coordination_latency_ms_total += coordination_latency_ms
                if not global_unfinished:
                    self._pause_wave()
                    if (
                        self.config.attention_dp > 1
                        and self.topology.global_dp_idx == 0
                    ):
                        ray.get(
                            self._coordinator.wave_complete.remote(wave_id),
                            timeout=self.config.quantum_timeout_s,
                        )
                    continue

                self._publish_first_schedule_events(
                    self.scheduler.mark_first_forward_started(batch)
                )
                begin = perf_counter()
                worker_results = self.executor.run(
                    batch, timeout=self.config.quantum_timeout_s
                )
                execute_latency_ms = (perf_counter() - begin) * 1000
                self._execute_latency_ms_total += execute_latency_ms
                # ABORT is the only command allowed to mutate scheduler state
                # after batch freeze and before canonical token commit.
                self._drain_queue(self._abort_commands)
                begin = perf_counter()
                events = self.scheduler.postprocess(
                    batch,
                    worker_results,
                    execute_latency_ms=execute_latency_ms,
                )
                postprocess_latency_ms = (perf_counter() - begin) * 1000
                self._postprocess_latency_ms_total += postprocess_latency_ms
                itl_token_count = self.scheduler.last_itl_token_slots
                if itl_token_count > 0:
                    itl_ms = (
                        execute_latency_ms / HIERARCHICAL_LOOP_COUNT
                    )
                    self._decode_itl_samples.append(
                        DecodeITLSample(
                            engine_id=self.engine_id,
                            wave_id=wave_id,
                            quantum_id=quantum_id,
                            itl_ms=itl_ms,
                            token_count=itl_token_count,
                        )
                    )
                    self._decode_itl_ms_weighted_total += (
                        itl_ms * itl_token_count
                    )
                    self._decode_itl_token_count += itl_token_count
                self._decode_quantum_count += 1
                self._publish_first_token_events(
                    self.scheduler.drain_first_token_events()
                )
                self._publish_events(events)
                post_execute_snapshot = self._refresh_cached_load()
                if self._quantum_diagnostics_enabled:
                    executor_diagnostic = (
                        self.executor.last_quantum_diagnostic
                    )
                    if (
                        executor_diagnostic is None
                        or executor_diagnostic.get("engine_id")
                        != self.engine_id
                        or executor_diagnostic.get("wave_id") != wave_id
                        or executor_diagnostic.get("quantum_id") != quantum_id
                    ):
                        raise RuntimeError(
                            "LocalExecutor quantum diagnostic identity mismatch"
                        )
                    sample = {
                        "schema_version": 3,
                        "scheduler_arch": "hierarchical",
                        "engine_id": self.engine_id,
                        "wave_id": wave_id,
                        "quantum_id": quantum_id,
                        "started_at_unix_s": quantum_started_at_unix_s,
                        "engine_has_real": batch.engine_has_real,
                        "waiting_before": pre_execute_snapshot.waiting,
                        "running_before": pre_execute_snapshot.running,
                        "useful_real_batch_size": (
                            pre_execute_snapshot.useful_real_batch_size
                        ),
                        "control_dummy_count": (
                            pre_execute_snapshot.control_dummy_count
                        ),
                        "attention_work_tokens": sum(
                            rank_load.active_dispatched_tokens
                            for rank_load in pre_execute_snapshot.rank_loads
                        ),
                        "free_blocks_min_before": (
                            pre_execute_snapshot.free_blocks_min
                        ),
                        "free_blocks_min_after": (
                            post_execute_snapshot.free_blocks_min
                        ),
                        "rank_loads_before": _rank_load_payload(
                            pre_execute_snapshot
                        ),
                        "rank_loads_after": _rank_load_payload(
                            post_execute_snapshot
                        ),
                        "admission_ms": admission_latency_ms,
                        "schedule_ms": schedule_latency_ms,
                        "ingress_drain_ms": ingress_drain_ms,
                        "consensus_exposed_wait_ms": coordination_latency_ms,
                        "consensus_overlap_window_ms": (
                            consensus_overlap_window_ms
                        ),
                        "leader_arrival_unix_ns": (
                            pending_consensus.arrival_unix_ns
                        ),
                        "leader_arrival_skew_ms": (
                            pending_consensus.leader_arrival_skew_ms
                        ),
                        "leader_rendezvous_ms": (
                            pending_consensus.leader_rendezvous_ms
                        ),
                        "late_participant_collective_ms": (
                            pending_consensus.late_participant_collective_ms
                        ),
                        "execute_ms": execute_latency_ms,
                        "postprocess_ms": postprocess_latency_ms,
                        "quantum_total_ms": (
                            perf_counter() - quantum_begin
                        )
                        * 1000,
                        "itl_ms": (
                            execute_latency_ms
                            / HIERARCHICAL_LOOP_COUNT
                        ),
                        "itl_token_count": itl_token_count,
                        "useful_decode_tokens": (
                            post_execute_snapshot.useful_decode_tokens
                            - pre_execute_snapshot.useful_decode_tokens
                        ),
                        "raw_token_slots": (
                            post_execute_snapshot.raw_token_slots
                            - pre_execute_snapshot.raw_token_slots
                        ),
                        "control_dummy_slots": (
                            post_execute_snapshot.control_dummy_slots
                            - pre_execute_snapshot.control_dummy_slots
                        ),
                        "preemption_count": (
                            post_execute_snapshot.preemption_count
                        ),
                        "executor": dict(executor_diagnostic),
                    }
                    with self._quantum_diagnostics_lock:
                        self._quantum_diagnostics.append(sample)
                with self._state_cv:
                    self._quantum_id += 1
        except BaseException as exc:
            self._failure = f"{type(exc).__name__}: {exc}"
            error = RuntimeError(
                f"LocalEngineCore {self.engine_id} failed: {self._failure}"
            )
            self._fail_pending_commands(error)
            with self._state_cv:
                self._stop = True
                self._state_cv.notify_all()
        finally:
            self._worker_transport_ready.set()
            self.executor.shutdown_worker_transport(
                timeout=self.config.quantum_timeout_s,
                failed=self._failure is not None,
            )

    def shutdown(self) -> None:
        if self._frontend_server is not None:
            self._frontend_server.close(self.config.quantum_timeout_s)
        with self._state_cv:
            self._stop = True
            self._state_cv.notify_all()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=self.config.quantum_timeout_s)
            if self._loop_thread.is_alive():
                raise TimeoutError(
                    f"LocalEngineCore {self.engine_id} event loop did not stop"
                )
        if self._control_group_initialized and dist.is_initialized():
            dist.destroy_process_group()
            self._control_group_initialized = False
