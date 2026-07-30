from __future__ import annotations

import contextlib
import os
import threading
from dataclasses import dataclass
from math import ceil
from typing import Any, Iterator, Mapping

import ray
from ray.util.placement_group import (
    placement_group,
    remove_placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from nanodeploy.config import Config
from nanodeploy.engine.decode_coordinator import DecodeCoordinator
from nanodeploy.engine.hierarchical_contract import (
    AddCommand,
    AddResult,
    AddResultEvent,
    AdmissionReservation,
    AbortResult,
    CoordinatorStatus,
    DecodeITLSample,
    EngineReady,
    FrontendEventBatch,
    FirstScheduleEvent,
    FirstTokenEvent,
    FinishEvent,
    IngressAck,
    LoadSnapshot,
)
from nanodeploy.engine.local_engine import LocalEngineCore
from nanodeploy.engine.ray_executor import (
    get_available_nodes_with_master_first,
)
from nanodeploy.worker.model_runner import ModelRunner


_PROXY_ENV_NAMES = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
)
_PROXY_ENV_LOCK = threading.RLock()


@contextlib.contextmanager
def _without_proxy_env() -> Iterator[None]:
    with _PROXY_ENV_LOCK:
        saved = {
            name: os.environ[name]
            for name in _PROXY_ENV_NAMES
            if name in os.environ
        }
        for name in _PROXY_ENV_NAMES:
            os.environ.pop(name, None)
        try:
            yield
        finally:
            for name in _PROXY_ENV_NAMES:
                os.environ.pop(name, None)
            os.environ.update(saved)


def _control_init_method(config: Config) -> str:
    raw = config.hierarchical_control_address or config.master_address
    if "://" in raw:
        _scheme, raw = raw.split("://", 1)
    try:
        host, port_text = raw.rsplit(":", 1)
        port = int(port_text)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"invalid hierarchical control/master address {raw!r}"
        ) from exc
    if config.hierarchical_control_address is None:
        port = port + 1 if port < 65535 else port - 1
    if not host or not 1 <= port <= 65535:
        raise ValueError(f"invalid hierarchical control address {raw!r}")
    return f"tcp://{host}:{port}"


@dataclass(slots=True)
class RayEngineTransport:
    engine_id: int
    actor: Any
    timeout: float

    def _get(self, ref):
        with _without_proxy_env():
            return ray.get(ref, timeout=self.timeout)

    def add(self, command: AddCommand) -> AddResult:
        return self._get(self.actor.submit_add.remote(command))

    def enqueue_async(self, command: AddCommand):
        with _without_proxy_env():
            return self.actor.enqueue_add.remote(command)

    def admit_async(self, command: AddCommand):
        with _without_proxy_env():
            return self.actor.admit_add.remote(command)

    def admit_batch_async(
        self,
        commands: tuple[AddCommand, ...],
        reservations: tuple[AdmissionReservation, ...],
    ):
        with _without_proxy_env():
            return self.actor.admit_add_batch.remote(
                commands, reservations
            )

    def enqueue_batch_async(self, commands: tuple[AddCommand, ...]):
        with _without_proxy_env():
            return self.actor.enqueue_add_batch.remote(commands)

    def poll_enqueue(
        self, handle
    ) -> tuple[bool, IngressAck | None]:
        with _without_proxy_env():
            ready, _ = ray.wait([handle], num_returns=1, timeout=0)
            if not ready:
                return False, None
            return True, ray.get(ready[0])

    def poll_admission_batch(
        self, handle
    ) -> tuple[bool, tuple[IngressAck, ...] | None]:
        with _without_proxy_env():
            ready, _ = ray.wait([handle], num_returns=1, timeout=0)
            if not ready:
                return False, None
            return True, tuple(ray.get(ready[0]))

    def abort(self, request_id: int) -> AbortResult:
        return self._get(self.actor.submit_abort.remote(request_id))

    def load(self) -> LoadSnapshot:
        return self._get(self.actor.get_cached_load.remote())


class DeploymentManager:
    """Owns hierarchical actors, exact DP placement groups, and cleanup."""

    _GPUS_PER_NODE = 8

    def __init__(self, config: Config) -> None:
        if config.scheduler_arch != "hierarchical":
            raise ValueError(
                "DeploymentManager requires scheduler_arch='hierarchical'"
            )
        self.config = config
        self.topology = config.hierarchical_topology
        self._lock = threading.Lock()
        self._closed = False
        self.placement_groups: list[Any] = []
        self.workers_by_engine: dict[int, list[Any]] = {}
        self.engines: dict[int, Any] = {}
        self.coordinator: Any | None = None
        self.engine_clients: dict[int, RayEngineTransport] = {}
        self.ready: tuple[EngineReady, ...] = ()
        self._expected_node_by_engine: dict[int, str] = {}

        try:
            self._start()
        except BaseException:
            self.close()
            raise

    def _runtime_env(self) -> dict[str, dict[str, str]]:
        # Actor creation happens inside _without_proxy_env(), so proxy variables
        # are absent rather than inherited with empty string values.
        env_vars: dict[str, str] = {}
        if "SLIME_QP_NUM" in os.environ:
            env_vars["SLIME_QP_NUM"] = os.environ["SLIME_QP_NUM"]
        return {"env_vars": env_vars}

    def _start(self) -> None:
        with _without_proxy_env():
            ray.init(
                address=self.config.ray_address,
                ignore_reinit_error=True,
            )
            nodes = get_available_nodes_with_master_first(
                self.config.master_address
            )
            engine_world_size = self.topology.engines[0].world_size
            if engine_world_size > self._GPUS_PER_NODE:
                raise ValueError(
                    "a LocalEngine DP group cannot exceed one 8-GPU node"
                )
            engines_per_node = self._GPUS_PER_NODE // engine_world_size
            nodes_needed = ceil(
                len(self.topology.engines) / engines_per_node
            )
            if nodes_needed > len(nodes):
                raise RuntimeError(
                    f"hierarchical deployment needs {nodes_needed} free nodes, "
                    f"found {len(nodes)}"
                )

            runtime_env = self._runtime_env()
            for topology in self.topology.engines:
                node_index = topology.engine_id // engines_per_node
                target_node_id = nodes[node_index]["NodeID"]
                self._expected_node_by_engine[topology.engine_id] = str(
                    target_node_id
                )
                bundles = [
                    {"CPU": 0.1, "GPU": 1.0}
                    for _ in range(topology.world_size)
                ]
                bundles.append({"CPU": 0.1})
                pg = placement_group(
                    bundles=bundles,
                    strategy="STRICT_PACK",
                    name=(
                        f"nanodeploy-{self.config.engine_id}-"
                        f"dp{topology.engine_id}"
                    ),
                    _soft_target_node_id=target_node_id,
                )
                # Track pending groups before waiting so startup timeout cleanup
                # also removes a group that never became schedulable.
                self.placement_groups.append(pg)
                ray.get(
                    pg.ready(), timeout=self.config.startup_timeout_s
                )

                workers = []
                self.workers_by_engine[topology.engine_id] = workers
                for local_rank, global_rank in enumerate(
                    topology.global_ranks
                ):
                    strategy = PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=local_rank,
                        placement_group_capture_child_tasks=True,
                    )
                    worker = ModelRunner.options(
                        scheduling_strategy=strategy,
                        runtime_env=runtime_env,
                    ).remote(self.config, global_rank, local_rank)
                    workers.append(worker)

            all_workers = [
                worker
                for engine_id in sorted(self.workers_by_engine)
                for worker in self.workers_by_engine[engine_id]
            ]
            block_counts = ray.get(
                [worker.num_kvcache_blocks.remote() for worker in all_workers],
                timeout=self.config.startup_timeout_s,
            )
            self.config.num_kvcache_blocks = min(block_counts)
            ray.get(
                [
                    worker.allocate_kvcache.remote(
                        self.config.num_kvcache_blocks
                    )
                    for worker in all_workers
                ],
                timeout=self.config.startup_timeout_s,
            )
            # Hierarchical dummy admission assumes deterministic zero-backed KV
            # pages. This is a one-time READY prerequisite, not request work.
            ray.get(
                [worker.zero_kvcache.remote() for worker in all_workers],
                timeout=self.config.startup_timeout_s,
            )

            if self.config.attention_dp > 1:
                self.coordinator = DecodeCoordinator.remote(
                    self.config.attention_dp
                )

            for topology, pg in zip(
                self.topology.engines,
                self.placement_groups,
                strict=True,
            ):
                strategy = PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=topology.world_size,
                    placement_group_capture_child_tasks=True,
                )
                actor = LocalEngineCore.options(
                    scheduling_strategy=strategy,
                    runtime_env=runtime_env,
                ).remote(
                    self.config,
                    topology,
                    self.workers_by_engine[topology.engine_id],
                )
                self.engines[topology.engine_id] = actor

            if self.coordinator is not None:
                ray.get(
                    self.coordinator.attach_engines.remote(self.engines),
                    timeout=self.config.startup_timeout_s,
                )

            control_method = (
                _control_init_method(self.config)
                if self.config.attention_dp > 1
                else None
            )
            ready = ray.get(
                [
                    actor.initialize.remote(
                        control_init_method=control_method,
                        coordinator=self.coordinator,
                    )
                    for actor in self.engines.values()
                ],
                timeout=self.config.startup_timeout_s,
            )
            self.ready = tuple(sorted(ready, key=lambda item: item.engine_id))
            fingerprints = {
                item.config_fingerprint for item in self.ready
            }
            if len(fingerprints) != 1:
                raise RuntimeError(
                    "LocalEngine collective fingerprint mismatch"
                )
            for item in self.ready:
                expected_node_id = self._expected_node_by_engine[item.engine_id]
                if item.node_id != expected_node_id:
                    raise RuntimeError(
                        f"engine {item.engine_id} placement mismatch: "
                        f"expected node {expected_node_id}, got {item.node_id}"
                    )

            if self.coordinator is not None:
                for item in self.ready:
                    ray.get(
                        self.coordinator.register.remote(
                            item.engine_id, item.config_fingerprint
                        ),
                        timeout=self.config.startup_timeout_s,
                    )
                statuses = [
                    ray.get(
                        self.coordinator.mark_ready.remote(item.engine_id),
                        timeout=self.config.startup_timeout_s,
                    )
                    for item in self.ready
                ]
                if not statuses[-1].ready:
                    raise RuntimeError(
                        "DecodeCoordinator READY barrier did not complete"
                    )

            self.engine_clients = {
                engine_id: RayEngineTransport(
                    engine_id=engine_id,
                    actor=actor,
                    timeout=self.config.quantum_timeout_s,
                )
                for engine_id, actor in self.engines.items()
            }

    def notify_request(
        self, target_engine_id: int, observed_wave_id: int
    ) -> int:
        if self.coordinator is None:
            # DP1 is woken directly by LocalEngineCore's ADD command. It has no
            # deployment-wide wave state to synchronize.
            return observed_wave_id
        with _without_proxy_env():
            status: CoordinatorStatus = ray.get(
                self.coordinator.first_request.remote(
                    target_engine_id, observed_wave_id
                ),
                timeout=self.config.quantum_timeout_s,
            )
        return status.wave_id

    def poll_admission_batches(
        self, handles: Mapping[int, Any]
    ) -> dict[int, tuple[IngressAck, ...]]:
        """Resolve all ready DP admission flights with one nonblocking wait."""
        if not handles:
            return {}
        unknown = set(handles).difference(self.engines)
        if unknown:
            raise ValueError(
                f"admission handles contain unknown engines {sorted(unknown)}"
            )
        if len(handles) > len(self.engines):
            raise RuntimeError("more than one admission flight per engine")
        ref_to_engine = {
            handle: engine_id for engine_id, handle in handles.items()
        }
        with _without_proxy_env():
            ready, _ = ray.wait(
                list(ref_to_engine),
                num_returns=len(ref_to_engine),
                timeout=0,
            )
            if not ready:
                return {}
            results = ray.get(
                ready, timeout=self.config.quantum_timeout_s
            )
        return {
            ref_to_engine[ref]: tuple(acks)
            for ref, acks in zip(ready, results, strict=True)
        }

    def poll_frontend_events(self) -> tuple[FrontendEventBatch, ...]:
        """Fetch health, load, and all lifecycle events in one RPC per DP."""
        engine_items = sorted(self.engines.items())
        with _without_proxy_env():
            try:
                batches = ray.get(
                    [
                        actor.drain_frontend_events.remote()
                        for _, actor in engine_items
                    ],
                    timeout=self.config.quantum_timeout_s,
                )
            except BaseException:
                self.close()
                raise
        for (engine_id, _), batch in zip(
            engine_items, batches, strict=True
        ):
            if batch.engine_id != engine_id:
                raise RuntimeError(
                    "LocalEngine frontend event batch owner mismatch: "
                    f"expected={engine_id}, got={batch.engine_id}"
                )
        return tuple(batches)

    def poll_events(self) -> tuple[FinishEvent, ...]:
        with _without_proxy_env():
            try:
                health_refs = [
                    actor.health.remote() for actor in self.engines.values()
                ]
                event_refs = [
                    actor.drain_events.remote()
                    for actor in self.engines.values()
                ]
                ray.get(health_refs, timeout=self.config.quantum_timeout_s)
                per_engine = ray.get(
                    event_refs, timeout=self.config.quantum_timeout_s
                )
            except BaseException:
                self.close()
                raise
        return tuple(
            event for engine_events in per_engine for event in engine_events
        )

    def poll_add_results(self) -> tuple[AddResultEvent, ...]:
        with _without_proxy_env():
            per_engine = ray.get(
                [
                    actor.drain_add_results.remote()
                    for actor in self.engines.values()
                ],
                timeout=self.config.quantum_timeout_s,
            )
        return tuple(
            event for engine_events in per_engine for event in engine_events
        )

    def poll_first_token_events(self) -> tuple[FirstTokenEvent, ...]:
        with _without_proxy_env():
            per_engine = ray.get(
                [
                    actor.drain_first_token_events.remote()
                    for actor in self.engines.values()
                ],
                timeout=self.config.quantum_timeout_s,
            )
        return tuple(
            event for engine_events in per_engine for event in engine_events
        )

    def poll_first_schedule_events(
        self,
    ) -> tuple[FirstScheduleEvent, ...]:
        with _without_proxy_env():
            per_engine = ray.get(
                [
                    actor.drain_first_schedule_events.remote()
                    for actor in self.engines.values()
                ],
                timeout=self.config.quantum_timeout_s,
            )
        return tuple(
            event for engine_events in per_engine for event in engine_events
        )

    def load_snapshots(self) -> tuple[LoadSnapshot, ...]:
        with _without_proxy_env():
            snapshots = ray.get(
                [
                    actor.get_cached_load.remote()
                    for actor in self.engines.values()
                ],
                timeout=self.config.quantum_timeout_s,
            )
        return tuple(sorted(snapshots, key=lambda item: item.engine_id))

    def execution_traces(self) -> tuple[dict[str, Any], ...]:
        if not self.config.hierarchical_execution_trace:
            return ()
        with _without_proxy_env():
            per_engine = ray.get(
                [
                    actor.drain_execution_traces.remote()
                    for actor in self.engines.values()
                ],
                timeout=self.config.quantum_timeout_s,
            )
        return tuple(trace for traces in per_engine for trace in traces)

    def decode_itl_samples(self) -> tuple[DecodeITLSample, ...]:
        with _without_proxy_env():
            per_engine = ray.get(
                [
                    actor.get_decode_itl_samples.remote()
                    for actor in self.engines.values()
                ],
                timeout=self.config.quantum_timeout_s,
            )
        return tuple(
            sample for engine_samples in per_engine for sample in engine_samples
        )

    def quantum_diagnostics(self) -> tuple[dict[str, Any], ...]:
        if not self.config.hierarchical_quantum_diagnostics:
            return ()
        with _without_proxy_env():
            per_engine = ray.get(
                [
                    actor.drain_quantum_diagnostics.remote()
                    for actor in self.engines.values()
                ],
                timeout=self.config.quantum_timeout_s,
            )
        samples = tuple(
            sample
            for engine_samples in per_engine
            for sample in engine_samples
        )
        return tuple(
            sorted(
                samples,
                key=lambda sample: (
                    int(sample["wave_id"]),
                    int(sample["quantum_id"]),
                    int(sample["engine_id"]),
                ),
            )
        )

    def execution_boundary_metrics(
        self,
    ) -> dict[int, dict[str, float | int]]:
        engine_items = sorted(self.engines.items())
        with _without_proxy_env():
            per_engine = ray.get(
                [
                    actor.get_execution_boundary_metrics.remote()
                    for _, actor in engine_items
                ],
                timeout=self.config.quantum_timeout_s,
            )
        return {
            engine_id: metrics
            for (engine_id, _), metrics in zip(
                engine_items, per_engine, strict=True
            )
        }

    def reset_execution_boundary_metrics(self) -> None:
        with _without_proxy_env():
            ray.get(
                [
                    actor.reset_execution_boundary_metrics.remote()
                    for actor in self.engines.values()
                ],
                timeout=self.config.quantum_timeout_s,
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        with _without_proxy_env():
            if ray.is_initialized():
                if self.engines:
                    try:
                        ray.get(
                            [
                                actor.shutdown.remote()
                                for actor in self.engines.values()
                            ],
                            timeout=self.config.quantum_timeout_s,
                        )
                    except BaseException:
                        pass
                for actor in self.engines.values():
                    try:
                        ray.kill(actor, no_restart=True)
                    except BaseException:
                        pass
                if self.coordinator is not None:
                    try:
                        ray.kill(self.coordinator, no_restart=True)
                    except BaseException:
                        pass
                for workers in self.workers_by_engine.values():
                    for worker in workers:
                        try:
                            ray.kill(worker, no_restart=True)
                        except BaseException:
                            pass
                for pg in self.placement_groups:
                    try:
                        remove_placement_group(pg)
                    except BaseException:
                        pass

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass
