from __future__ import annotations

import queue
import threading
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import timedelta
from time import perf_counter
from typing import Any

import ray
import torch
import torch.distributed as dist

from nanodeploy.config import Config
from nanodeploy.engine.hierarchical_contract import (
    AddCommand,
    AddResult,
    AbortResult,
    EngineReady,
    FinishEvent,
    LoadSnapshot,
)
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


@ray.remote(num_cpus=0.1, max_concurrency=32)
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
        self._terminal_events: deque[FinishEvent] = deque()
        self._events_lock = threading.Lock()
        self._state_cv = threading.Condition()
        self._wave_running = False
        self._wave_id = 0
        self._quantum_id = 0
        self._pending_start_wave: int | None = None
        self._stop = False
        self._failure: str | None = None
        self._loop_thread: threading.Thread | None = None
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

        self._loop_thread = threading.Thread(
            target=self._event_loop,
            name=f"nanodeploy-local-engine-{self.engine_id}",
            daemon=True,
        )
        self._loop_thread.start()
        return EngineReady(
            engine_id=self.engine_id,
            global_ranks=self.topology.global_ranks,
            config_fingerprint=self.config.collective_fingerprint(),
            node_id=actor_node_id,
            worker_node_ids=worker_nodes,
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

    def submit_add(self, command: AddCommand) -> AddResult:
        return self._submit(_LoopCommand("add", command))

    def submit_abort(self, request_id: int) -> AbortResult:
        return self._submit(
            _LoopCommand("abort", request_id), abort_priority=True
        )

    def get_load(self) -> LoadSnapshot:
        return self._submit(_LoopCommand("load"))

    def drain_events(self) -> tuple[FinishEvent, ...]:
        self._raise_if_failed()
        with self._events_lock:
            events = tuple(self._terminal_events)
            self._terminal_events.clear()
        return events

    def drain_execution_traces(self) -> tuple[dict[str, Any], ...]:
        self._raise_if_failed()
        return self.executor.drain_execution_traces()

    def health(self) -> bool:
        self._raise_if_failed()
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
        self._command_count += 1
        self._command_queue_delay_ms_total += (
            perf_counter() - command.enqueued_at
        ) * 1000
        try:
            if command.kind == "add":
                command.result = self.scheduler.add(command.payload)
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
                snapshot = self.scheduler.load_snapshot(
                    wave_id=self._wave_id,
                    quantum_id=self._quantum_id,
                )
                command.result = replace(
                    snapshot,
                    command_count=self._command_count,
                    command_queue_delay_ms_total=(
                        self._command_queue_delay_ms_total
                    ),
                    decode_quantum_count=self._decode_quantum_count,
                    admission_latency_ms_total=(
                        self._admission_latency_ms_total
                    ),
                    schedule_latency_ms_total=self._schedule_latency_ms_total,
                    coordination_latency_ms_total=(
                        self._coordination_latency_ms_total
                    ),
                    execute_latency_ms_total=self._execute_latency_ms_total,
                    postprocess_latency_ms_total=(
                        self._postprocess_latency_ms_total
                    ),
                )
            else:
                raise RuntimeError(f"unknown LocalEngine command {command.kind}")
        except BaseException as exc:
            command.error = exc
        finally:
            command.completed.set()

    def _drain_queue(self, source: queue.Queue[_LoopCommand]) -> None:
        while True:
            try:
                command = source.get_nowait()
            except queue.Empty:
                return
            self._complete_command(command)

    def _consensus(self, local_unfinished: bool) -> bool:
        if self.config.attention_dp == 1:
            return local_unfinished
        local = torch.tensor(
            [self._wave_id, self._quantum_id, int(local_unfinished)],
            dtype=torch.int64,
            device="cpu",
        )
        minimum = local.clone()
        maximum = local.clone()
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        if (
            minimum[0].item() != maximum[0].item()
            or minimum[1].item() != maximum[1].item()
        ):
            raise RuntimeError(
                "LocalEngine leader wave/quantum mismatch: "
                f"min={minimum.tolist()}, max={maximum.tolist()}"
            )
        return bool(maximum[2].item())

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

    def _publish_events(self, events: tuple[FinishEvent, ...]) -> None:
        if not events:
            return
        with self._events_lock:
            self._terminal_events.extend(events)

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
            while True:
                self._drain_queue(self._abort_commands)
                self._drain_queue(self._normal_commands)
                with self._state_cv:
                    if self._stop:
                        return
                    if not self._wave_running:
                        self._state_cv.wait(timeout=0.1)
                        continue
                    wave_id = self._wave_id
                    quantum_id = self._quantum_id

                begin = perf_counter()
                self.scheduler.admit()
                self._admission_latency_ms_total += (
                    perf_counter() - begin
                ) * 1000
                begin = perf_counter()
                batch = self.scheduler.plan_decode(
                    wave_id=wave_id, quantum_id=quantum_id
                )
                self._schedule_latency_ms_total += (
                    perf_counter() - begin
                ) * 1000
                local_unfinished = not self.scheduler.is_finished()
                begin = perf_counter()
                global_unfinished = self._consensus(local_unfinished)
                self._coordination_latency_ms_total += (
                    perf_counter() - begin
                ) * 1000
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

                begin = perf_counter()
                worker_results = self.executor.run(
                    batch, timeout=self.config.quantum_timeout_s
                )
                self._execute_latency_ms_total += (
                    perf_counter() - begin
                ) * 1000
                # ABORT is the only command allowed to mutate scheduler state
                # after batch freeze and before canonical token commit.
                self._drain_queue(self._abort_commands)
                begin = perf_counter()
                events = self.scheduler.postprocess(batch, worker_results)
                self._postprocess_latency_ms_total += (
                    perf_counter() - begin
                ) * 1000
                self._decode_quantum_count += 1
                self._publish_events(events)
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

    def shutdown(self) -> None:
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
