from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from math import ceil
from time import perf_counter
from typing import Any, Callable, Iterable, Literal, Mapping, Protocol

from nanodeploy.engine.hierarchical_contract import (
    AddCommand,
    AddResult,
    AddResultEvent,
    AdmissionReservation,
    AbortResult,
    FirstScheduleEvent,
    FinishEvent,
    IngressAck,
    LoadSnapshot,
    OwnerState,
    round_up,
)
from nanodeploy.router.admission_planner import (
    AdmissionPlanner,
    AdmissionPlannerConfig,
    AdmissionShadow,
)

RouterPolicy = Literal[
    "round_robin",
    "least_batch",
    "least_batch_v2",
    "least_cache",
]
_ROUTER_POLICIES = frozenset(
    {"round_robin", "least_batch", "least_batch_v2", "least_cache"}
)
_GLOBAL_QUEUE_POLICIES = frozenset({"least_batch", "least_batch_v2"})


class EngineTransport(Protocol):
    """Synchronous control-plane view of one LocalEngineCore."""

    engine_id: int

    def add(self, command: AddCommand) -> AddResult: ...

    def enqueue_async(self, command: AddCommand) -> Any: ...

    def enqueue_batch_async(
        self, commands: tuple[AddCommand, ...]
    ) -> Any: ...

    def admit_async(self, command: AddCommand) -> Any: ...

    def admit_batch_async(
        self,
        commands: tuple[AddCommand, ...],
        reservations: tuple[AdmissionReservation, ...],
    ) -> Any: ...

    def poll_enqueue(
        self, handle: Any
    ) -> tuple[bool, IngressAck | None]: ...

    def poll_admission_batch(
        self, handle: Any
    ) -> tuple[bool, tuple[IngressAck, ...] | None]: ...

    def abort(
        self,
        request_id: int,
        *,
        allow_future_ingress: bool = False,
    ) -> AbortResult: ...

    def clear_ingress_abort(self, request_id: int) -> None: ...

    def load(self) -> LoadSnapshot: ...


@dataclass(slots=True)
class RequestOwner:
    state: OwnerState
    engine_id: int | None = None


@dataclass(slots=True)
class _PendingIngress:
    command: AddCommand
    candidate_engine_ids: tuple[int, ...]
    candidate_index: int
    handle: Any
    rpc_started_at: float
    router_pending_ms: float = 0.0
    admission_rpc_ms: float = 0.0


@dataclass(slots=True)
class _GlobalPending:
    command: AddCommand
    queue_seq: int
    router_queued_at: float
    router_pending_ms: float = 0.0
    admission_rpc_ms: float = 0.0
    capacity_blocked_since: float | None = None


@dataclass(slots=True)
class _AdmissionBatchFlight:
    engine_id: int
    pending: tuple[_GlobalPending, ...]
    handle: Any
    rpc_started_at: float
    capacity_epoch: int


@dataclass(slots=True)
class _IngressBatchFlight:
    engine_id: int
    pending: tuple[_GlobalPending, ...]
    handle: Any
    rpc_started_at: float


@dataclass(frozen=True, slots=True)
class _KvIngressCharge:
    engine_id: int
    blocks: int


@dataclass(slots=True)
class _LeastBatchCharge:
    engine_id: int
    reservation: AdmissionReservation
    admission_version: int | None = None


WakeupCallback = Callable[[int, int], int]
AdmissionBatchPoller = Callable[
    [Mapping[int, Any]], Mapping[int, tuple[IngressAck, ...]]
]


class RequestRouter:
    """READY-gated load routing with sticky request ownership."""

    def __init__(
        self,
        engines: Mapping[int, EngineTransport],
        *,
        router_policy: RouterPolicy = "least_batch",
        kvcache_block_size: int | None = None,
        wakeup: WakeupCallback | None = None,
        admission_batch_size: int = 256,
        poll_admission_batches: AdmissionBatchPoller | None = None,
        admission_planner_config: AdmissionPlannerConfig | None = None,
        max_unscheduled_requests: int | None = None,
        initial_wave_id: int = 0,
    ) -> None:
        if not engines:
            raise ValueError("RequestRouter requires at least one engine")
        self._engines = dict(sorted(engines.items()))
        if any(
            engine_id != transport.engine_id
            for engine_id, transport in self._engines.items()
        ):
            raise ValueError("engine registry key/id mismatch")
        if router_policy not in _ROUTER_POLICIES:
            raise ValueError(f"unsupported router policy {router_policy!r}")
        if router_policy == "least_cache" and (
            kvcache_block_size is None or kvcache_block_size <= 0
        ):
            raise ValueError(
                "least_cache routing requires a positive kvcache_block_size"
            )
        if admission_batch_size <= 0:
            raise ValueError("admission_batch_size must be positive")
        if (
            max_unscheduled_requests is not None
            and max_unscheduled_requests <= 0
        ):
            raise ValueError("max_unscheduled_requests must be positive")
        self._ready_engine_ids = tuple(self._engines)
        self.router_policy = router_policy
        self._kvcache_block_size = kvcache_block_size
        self._owners: dict[int, RequestOwner] = {}
        self._terminal: dict[int, FinishEvent] = {}
        self._rejected_request_ids: set[int] = set()
        self._pending_ingress: dict[int, _PendingIngress] = {}
        self._global_pending: deque[_GlobalPending] = deque()
        self._admission_flights: dict[int, _AdmissionBatchFlight] = {}
        self._ingress_batch_flights: dict[int, _IngressBatchFlight] = {}
        self._admission_batch_size = admission_batch_size
        self._poll_admission_batches = poll_admission_batches
        self._admission_planner = (
            AdmissionPlanner(admission_planner_config)
            if (
                admission_planner_config is not None
                and router_policy == "least_batch"
            )
            else None
        )
        if router_policy == "least_batch_v2":
            if admission_planner_config is None:
                raise ValueError(
                    "least_batch_v2 requires admission_planner_config"
                )
            if (
                kvcache_block_size is None
                or kvcache_block_size <= 0
                or kvcache_block_size
                != admission_planner_config.kvcache_block_size
            ):
                raise ValueError(
                    "least_batch_v2 requires a matching positive "
                    "kvcache_block_size"
                )
            if max_unscheduled_requests is None:
                max_unscheduled_requests = (
                    admission_planner_config.max_num_seqs
                )
        self._max_unscheduled_requests = max_unscheduled_requests or 0
        self._kv_attention_sp = (
            admission_planner_config.attention_sp
            if admission_planner_config is not None
            else 1
        )
        self._kv_reserved_blocks_per_req = (
            admission_planner_config.reserved_blocks_per_req
            if admission_planner_config is not None
            else 0.0
        )
        self._kv_credit_generation: dict[int, tuple[Any, ...]] = {}
        self._kv_remaining_blocks: dict[int, int] = {}
        self._kv_ingress_charges: dict[int, _KvIngressCharge] = {}
        self._kv_unscheduled_counts = {
            engine_id: 0 for engine_id in self._ready_engine_ids
        }
        self._kv_projected_waiting_counts = {
            engine_id: 0 for engine_id in self._ready_engine_ids
        }
        self._kv_outstanding_blocks = {
            engine_id: 0 for engine_id in self._ready_engine_ids
        }
        self._future_ingress_aborts: set[int] = set()
        self._kv_blocked_generation: dict[
            int, tuple[Any, ...] | None
        ] = {
            engine_id: None for engine_id in self._ready_engine_ids
        }
        self._engine_blocked_capacity_epoch: dict[int, int | None] = {
            engine_id: None for engine_id in self._ready_engine_ids
        }
        self._next_global_queue_seq = 0
        self._immediate_ingress_acks: deque[IngressAck] = deque()
        self._early_add_results: dict[int, AddResultEvent] = {}
        self._early_terminal_events: dict[int, FinishEvent] = {}
        self._global_capacity_queue_ms: dict[int, float] = {}
        self._loads: dict[int, LoadSnapshot] = {}
        self._estimated_free_blocks: dict[int, int] = {}
        self._cache_charges: dict[int, tuple[int, int]] = {}
        self._least_batch_charges: dict[int, _LeastBatchCharge] = {}
        self._least_batch_tentative_counts = {
            engine_id: 0 for engine_id in self._ready_engine_ids
        }
        self._admission_attempts = {
            engine_id: 0 for engine_id in self._ready_engine_ids
        }
        self._admission_commits = {
            engine_id: 0 for engine_id in self._ready_engine_ids
        }
        self._admission_deferred = {
            engine_id: 0 for engine_id in self._ready_engine_ids
        }
        self._admission_queue_full = {
            engine_id: 0 for engine_id in self._ready_engine_ids
        }
        self._admission_local_state_mismatch = {
            engine_id: 0 for engine_id in self._ready_engine_ids
        }
        self._admission_fallbacks = 0
        self._admission_global_retries = 0
        self._admission_state_mismatches = 0
        self._kv_ingress_attempts = {
            engine_id: 0 for engine_id in self._ready_engine_ids
        }
        self._kv_ingress_receipts = {
            engine_id: 0 for engine_id in self._ready_engine_ids
        }
        self._kv_ingress_queue_full = {
            engine_id: 0 for engine_id in self._ready_engine_ids
        }
        self._kv_gate_blocked = 0
        self._unscheduled_gate_blocked = 0
        self._rr_cursor = 0
        self._wakeup = wakeup
        self._wave_id = initial_wave_id

    @property
    def wave_id(self) -> int:
        return self._wave_id

    @property
    def active_count(self) -> int:
        return len(self._owners)

    @property
    def pending_ingress_count(self) -> int:
        return sum(
            owner.state
            in {OwnerState.PENDING_GLOBAL, OwnerState.PENDING_INGRESS}
            for owner in self._owners.values()
        )

    @property
    def pending_global_count(self) -> int:
        return sum(
            owner.state == OwnerState.PENDING_GLOBAL
            for owner in self._owners.values()
        )

    @property
    def pending_add_count(self) -> int:
        return sum(
            owner.state == OwnerState.PENDING_ADD
            for owner in self._owners.values()
        )

    @property
    def is_idle(self) -> bool:
        return self.active_count == 0

    def owner(self, request_id: int) -> RequestOwner | None:
        return self._owners.get(request_id)

    def terminal_event(self, request_id: int) -> FinishEvent | None:
        return self._terminal.get(request_id)

    def _round_robin_candidates(self) -> tuple[int, ...]:
        engine_ids = self._ready_engine_ids
        start = self._rr_cursor
        self._rr_cursor = (self._rr_cursor + 1) % len(engine_ids)
        return tuple(
            engine_ids[(start + offset) % len(engine_ids)]
            for offset in range(len(engine_ids))
        )

    def _projected_waiting(self, engine_id: int) -> int:
        """Return observed waiting plus assignments since that observation."""
        return self._kv_projected_waiting_counts[engine_id]

    def _projected_batch(self, engine_id: int) -> int:
        snapshot = self._loads.get(engine_id)
        running = snapshot.running if snapshot is not None else 0
        if self.router_policy == "least_batch_v2":
            return running + self._projected_waiting(engine_id)
        tentative = self._least_batch_tentative_counts[engine_id]
        return running + tentative

    def _estimate_request_blocks(
        self,
        prompt_token_ids: tuple[int, ...],
        max_tokens: int,
    ) -> int:
        if self._kvcache_block_size is None:
            return 0
        if max_tokens < 1:
            return 0
        total_tokens = len(prompt_token_ids) + round_up(max_tokens)
        return (
            total_tokens + self._kvcache_block_size - 1
        ) // self._kvcache_block_size

    def _estimate_kv_admission_blocks(self, command: AddCommand) -> int:
        """Conservative scalar estimate of the immediate KV footprint.

        This deliberately does not mirror LocalScheduler placement. The SP
        rounding guard is the maximum extra block rounding introduced by
        splitting the prompt across the ranks that can receive at least one
        token. Full padded decode lifetime remains a local static-validity
        and scheduling concern rather than live frontend KV reservation.
        """
        block_size = self._kvcache_block_size
        if block_size is None or block_size <= 0:
            raise RuntimeError(
                "KV-credit routing requires a positive block size"
            )
        prompt_tokens = len(command.prompt_token_ids)
        prompt_blocks = (prompt_tokens + block_size - 1) // block_size
        participating_ranks = min(
            self._kv_attention_sp, max(1, prompt_tokens)
        )
        sp_rounding_guard = participating_ranks - 1
        decode_reserve = ceil(self._kv_reserved_blocks_per_req)
        return prompt_blocks + sp_rounding_guard + decode_reserve

    def _kv_snapshot_state(
        self, snapshot: LoadSnapshot
    ) -> tuple[tuple[Any, ...], int] | None:
        ranks = {rank.sp_idx: rank for rank in snapshot.rank_loads}
        expected = set(range(self._kv_attention_sp))
        if set(ranks) != expected:
            return None
        free_blocks = tuple(
            ranks[sp_idx].free_blocks
            for sp_idx in range(self._kv_attention_sp)
        )
        generation = (
            snapshot.wave_id,
            snapshot.quantum_id,
            snapshot.capacity_epoch,
            free_blocks,
        )
        total_free_blocks = sum(free_blocks)
        return generation, total_free_blocks

    def _record_kv_credit_snapshot(
        self,
        snapshot: LoadSnapshot,
        *,
        refresh_load_projection: bool = False,
    ) -> None:
        if self.router_policy != "least_batch_v2":
            return
        engine_id = snapshot.engine_id
        if refresh_load_projection:
            observed_waiting = snapshot.waiting + snapshot.pending_ingress
            # Outstanding charges cover requests between frontend dispatch and
            # first schedule. max() prevents a load snapshot captured before
            # a fast ingress receipt from making ACK look like free compute
            # capacity. New assignments increment this projection below.
            self._kv_projected_waiting_counts[engine_id] = max(
                observed_waiting,
                self._kv_unscheduled_counts[engine_id],
            )
        state = self._kv_snapshot_state(snapshot)
        if state is None:
            self._kv_credit_generation.pop(engine_id, None)
            self._kv_remaining_blocks.pop(engine_id, None)
            return
        generation, free_blocks = state
        self._kv_credit_generation[engine_id] = generation
        # A charge stays live until the request reaches first schedule. Before
        # that point a newer quantum can still report KV free blocks that do
        # not reflect an ingress command waiting inside LocalEngine. Rebuilding
        # credit from snapshot minus this ledger prevents quantum-by-quantum
        # re-granting. Queue and placement limits remain authoritative in the
        # LocalEngine; the frontend v2 eligibility gate intentionally models
        # only KV credit plus its bounded unscheduled window.
        self._kv_remaining_blocks[engine_id] = max(
            0,
            free_blocks - self._kv_outstanding_blocks[engine_id],
        )

    def _charge_kv_ingress(
        self, *, request_id: int, engine_id: int, blocks: int
    ) -> None:
        if request_id in self._kv_ingress_charges:
            raise RuntimeError(
                f"duplicate KV ingress charge for {request_id}"
            )
        self._kv_ingress_charges[request_id] = _KvIngressCharge(
            engine_id=engine_id,
            blocks=blocks,
        )
        self._kv_unscheduled_counts[engine_id] += 1
        self._kv_projected_waiting_counts[engine_id] += 1
        self._kv_outstanding_blocks[engine_id] += blocks

    def _release_kv_ingress_charge(self, request_id: int) -> None:
        charge = self._kv_ingress_charges.pop(request_id, None)
        if charge is None:
            return
        engine_id = charge.engine_id
        self._kv_unscheduled_counts[engine_id] -= 1
        self._kv_projected_waiting_counts[engine_id] -= 1
        self._kv_outstanding_blocks[engine_id] -= charge.blocks
        if (
            self._kv_unscheduled_counts[engine_id] < 0
            or self._kv_projected_waiting_counts[engine_id] < 0
            or self._kv_outstanding_blocks[engine_id] < 0
        ):
            raise RuntimeError(
                "KV ingress charge ledger became negative: "
                f"engine={engine_id}, request={request_id}"
            )
        snapshot = self._loads.get(engine_id)
        if snapshot is not None:
            self._record_kv_credit_snapshot(snapshot)

    def _kv_static_capacity_reason(
        self, command: AddCommand
    ) -> str | None:
        """Reject only requests that cannot fit an otherwise empty engine."""
        block_size = self._kvcache_block_size
        if (
            block_size is None
            or block_size <= 0
            or command.max_tokens < 0
        ):
            return None
        padded_completion = round_up(command.max_tokens)
        lifetime_blocks = (
            len(command.prompt_token_ids)
            + 1
            + padded_completion
            + block_size
            - 1
        ) // block_size
        master_lifetime_blocks = (
            1 + padded_completion + block_size - 1
        ) // block_size
        observed_complete_engine = False
        for snapshot in self._loads.values():
            ranks = {rank.sp_idx: rank for rank in snapshot.rank_loads}
            if set(ranks) != set(range(self._kv_attention_sp)):
                continue
            observed_complete_engine = True
            service_blocks = [
                ranks[sp_idx].total_blocks
                - ranks[sp_idx].control_dummy_blocks
                for sp_idx in range(self._kv_attention_sp)
            ]
            if (
                service_blocks
                and min(service_blocks) > 0
                and lifetime_blocks <= sum(service_blocks)
                and master_lifetime_blocks <= max(service_blocks)
            ):
                return None
        if not observed_complete_engine:
            return None
        return "request padded lifetime exceeds every LocalEngine KV capacity"

    def _candidate_engine_ids(
        self,
        *,
        prompt_token_ids: tuple[int, ...],
        max_tokens: int,
    ) -> tuple[int, ...]:
        if self.router_policy in _GLOBAL_QUEUE_POLICIES:
            # Match the centralized scheduler's node ordering:
            # (running + tentative admissions, dp_idx).
            return tuple(
                sorted(
                    self._ready_engine_ids,
                    key=lambda engine_id: (
                        self._projected_batch(engine_id),
                        engine_id,
                    ),
                )
            )

        # Round-robin also supplies deterministic tie-breaking for least-cache.
        candidates = self._round_robin_candidates()
        if self.router_policy == "round_robin":
            return candidates

        if any(
            engine_id not in self._estimated_free_blocks
            for engine_id in candidates
        ):
            # Startup remains available before the first complete load report.
            return candidates
        request_blocks = self._estimate_request_blocks(
            prompt_token_ids, max_tokens
        )
        return tuple(
            sorted(
                candidates,
                key=lambda engine_id: (
                    self._estimated_free_blocks[engine_id] - request_blocks
                ),
                reverse=True,
            )
        )

    def _charge_cache(
        self,
        *,
        request_id: int,
        engine_id: int,
        prompt_token_ids: tuple[int, ...],
        max_tokens: int,
    ) -> None:
        if (
            self.router_policy != "least_cache"
            or engine_id not in self._estimated_free_blocks
        ):
            return
        if request_id in self._cache_charges:
            raise RuntimeError(f"duplicate cache charge for {request_id}")
        request_blocks = self._estimate_request_blocks(
            prompt_token_ids, max_tokens
        )
        self._estimated_free_blocks[engine_id] -= request_blocks
        self._cache_charges[request_id] = (engine_id, request_blocks)

    def _refund_cache(self, request_id: int) -> None:
        charge = self._cache_charges.pop(request_id, None)
        if charge is None:
            return
        engine_id, request_blocks = charge
        if engine_id in self._estimated_free_blocks:
            self._estimated_free_blocks[engine_id] += request_blocks

    def _charge_least_batch(
        self,
        *,
        request_id: int,
        engine_id: int,
        reservation: AdmissionReservation,
    ) -> None:
        if self.router_policy != "least_batch":
            return
        if request_id in self._least_batch_charges:
            raise RuntimeError(
                f"duplicate least-batch charge for {request_id}"
            )
        if (
            reservation.request_id != request_id
            or reservation.engine_id != engine_id
        ):
            raise RuntimeError(
                "admission reservation ownership mismatch: "
                f"request={request_id}, engine={engine_id}, "
                f"reservation={reservation}"
            )
        self._least_batch_charges[request_id] = _LeastBatchCharge(
            engine_id, reservation
        )
        self._least_batch_tentative_counts[engine_id] += 1

    def _refund_least_batch(self, request_id: int) -> None:
        charge = self._least_batch_charges.pop(request_id, None)
        if charge is None:
            return
        self._least_batch_tentative_counts[charge.engine_id] -= 1
        if self._least_batch_tentative_counts[charge.engine_id] < 0:
            raise RuntimeError(
                "negative least-batch tentative admission count"
            )

    def _commit_least_batch(
        self, request_id: int, admission_version: int | None
    ) -> None:
        charge = self._least_batch_charges.get(request_id)
        if charge is None:
            raise RuntimeError(
                f"missing least-batch charge for request {request_id}"
            )
        if admission_version is None:
            raise RuntimeError(
                "successful centralized admission ACK did not carry an "
                f"admission version: request={request_id}"
            )
        charge.admission_version = admission_version
        snapshot = self._loads.get(charge.engine_id)
        if (
            snapshot is not None
            and snapshot.admission_version >= admission_version
        ):
            # The consolidated load/event poll can observe the commit before
            # the admission ObjectRef is resolved. Do not apply the same
            # reservation twice during the next dispatch in this cycle.
            self._refund_least_batch(request_id)

    def add(
        self,
        *,
        request_id: int,
        prompt_token_ids: tuple[int, ...],
        max_tokens: int,
        temperature: float,
        ignore_eos: bool,
    ) -> AddResult:
        if (
            request_id in self._owners
            or request_id in self._terminal
            or request_id in self._rejected_request_ids
        ):
            return AddResult(
                request_id=request_id,
                accepted=False,
                reason="duplicate_request_id",
            )

        self._owners[request_id] = RequestOwner(
            OwnerState.PENDING_INGRESS
        )
        engine_ids = self._candidate_engine_ids(
            prompt_token_ids=prompt_token_ids,
            max_tokens=max_tokens,
        )
        last_result: AddResult | None = None

        for engine_id in engine_ids:
            command = AddCommand(
                request_id=request_id,
                prompt_token_ids=prompt_token_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                ignore_eos=ignore_eos,
                wave_id=self._wave_id,
            )
            result = self._engines[engine_id].add(command)
            if result.request_id != request_id:
                self._owners.pop(request_id, None)
                raise RuntimeError(
                    "LocalEngine returned an inconsistent request id: "
                    f"expected={request_id}, got={result.request_id}"
                )
            if result.accepted:
                if result.engine_id != engine_id:
                    self._owners.pop(request_id, None)
                    raise RuntimeError(
                        "LocalEngine returned an inconsistent owner: "
                        f"expected={engine_id}, got={result.engine_id}"
                    )
                self._owners[request_id] = RequestOwner(
                    OwnerState.OWNED, engine_id
                )
                self._charge_cache(
                    request_id=request_id,
                    engine_id=engine_id,
                    prompt_token_ids=prompt_token_ids,
                    max_tokens=max_tokens,
                )
                if self._wakeup is not None:
                    self._wave_id = self._wakeup(engine_id, self._wave_id)
                return result

            last_result = result
            if result.reason != "queue_full":
                break

        self._owners.pop(request_id, None)
        self._rejected_request_ids.add(request_id)
        if last_result is None:
            raise RuntimeError("RequestRouter had no READY engine to try")
        return last_result

    def submit_async(
        self,
        *,
        request_id: int,
        prompt_token_ids: tuple[int, ...],
        max_tokens: int,
        temperature: float,
        ignore_eos: bool,
    ) -> int:
        """Queue a request without waiting for a LocalEngine result."""
        submitted_at = perf_counter()
        if (
            request_id in self._owners
            or request_id in self._terminal
            or request_id in self._rejected_request_ids
        ):
            self._immediate_ingress_acks.append(
                IngressAck(
                    request_id=request_id,
                    engine_id=-1,
                    enqueued=False,
                    reason="duplicate_request_id",
                )
            )
            return request_id

        command = AddCommand(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            ignore_eos=ignore_eos,
            wave_id=self._wave_id,
        )
        if self.router_policy == "least_batch_v2":
            if reason := self._kv_static_capacity_reason(command):
                self._rejected_request_ids.add(request_id)
                self._immediate_ingress_acks.append(
                    IngressAck(
                        request_id=request_id,
                        engine_id=-1,
                        enqueued=False,
                        reason=reason,
                    )
                )
                return request_id
        if self.router_policy in _GLOBAL_QUEUE_POLICIES:
            self._owners[request_id] = RequestOwner(
                OwnerState.PENDING_GLOBAL
            )
            if self.router_policy == "least_batch":
                queue_is_capacity_blocked = (
                    (
                        bool(self._global_pending)
                        and any(
                            pending.capacity_blocked_since is not None
                            for pending in self._global_pending
                        )
                    )
                    or not any(
                        self._engine_has_admission_window(engine_id)
                        for engine_id in self._ready_engine_ids
                    )
                )
            else:
                queue_is_capacity_blocked = bool(
                    self._global_pending
                    and self._global_pending[0].capacity_blocked_since
                    is not None
                )
            capacity_blocked_since = (
                submitted_at if queue_is_capacity_blocked else None
            )
            self._global_pending.append(
                _GlobalPending(
                    command,
                    queue_seq=self._next_global_queue_seq,
                    router_queued_at=submitted_at,
                    capacity_blocked_since=capacity_blocked_since,
                )
            )
            self._next_global_queue_seq += 1
            self._global_capacity_queue_ms[request_id] = 0.0
            return request_id

        candidates = self._candidate_engine_ids(
            prompt_token_ids=prompt_token_ids,
            max_tokens=max_tokens,
        )
        engine_id = candidates[0]
        self._owners[request_id] = RequestOwner(
            OwnerState.PENDING_INGRESS, engine_id
        )
        self._charge_cache(
            request_id=request_id,
            engine_id=engine_id,
            prompt_token_ids=prompt_token_ids,
            max_tokens=max_tokens,
        )
        try:
            rpc_started_at = perf_counter()
            handle = self._engines[engine_id].enqueue_async(command)
        except BaseException:
            self._refund_cache(request_id)
            self._owners.pop(request_id, None)
            raise
        self._pending_ingress[request_id] = _PendingIngress(
            command=command,
            candidate_engine_ids=candidates,
            candidate_index=0,
            handle=handle,
            rpc_started_at=rpc_started_at,
            router_pending_ms=(
                rpc_started_at - submitted_at
            )
            * 1000,
        )
        return request_id

    def _capacity_epoch(self, engine_id: int) -> int:
        snapshot = self._loads.get(engine_id)
        return snapshot.capacity_epoch if snapshot is not None else 0

    def _engine_has_admission_window(self, engine_id: int) -> bool:
        blocked_epoch = self._engine_blocked_capacity_epoch[engine_id]
        return (
            engine_id not in self._admission_flights
            and (
                blocked_epoch is None
                or self._capacity_epoch(engine_id) > blocked_epoch
            )
        )

    def _mark_global_capacity_blocked(self, blocked_at: float) -> None:
        for pending in self._global_pending:
            if pending.capacity_blocked_since is None:
                pending.capacity_blocked_since = blocked_at

    def _requeue_global_pending(
        self, pendings: Iterable[_GlobalPending]
    ) -> None:
        """Merge retries with the queue in original global submission order."""
        combined = tuple(self._global_pending) + tuple(pendings)
        request_ids = tuple(
            pending.command.request_id for pending in combined
        )
        if len(request_ids) != len(set(request_ids)):
            raise RuntimeError("duplicate request while rebuilding global FIFO")
        self._global_pending = deque(
            sorted(combined, key=lambda pending: pending.queue_seq)
        )

    def _admission_shadows(self) -> dict[int, AdmissionShadow]:
        planner = self._admission_planner
        if planner is None:
            raise RuntimeError(
                "least_batch async admission requires "
                "admission_planner_config"
            )
        shadows: dict[int, AdmissionShadow] = {}
        for engine_id in self._ready_engine_ids:
            snapshot = self._loads.get(engine_id)
            if snapshot is None:
                continue
            shadow = planner.shadow_from_snapshot(snapshot)
            if shadow is None:
                continue
            for charge in self._least_batch_charges.values():
                if charge.engine_id == engine_id:
                    planner.apply_reservation(
                        shadow, charge.reservation
                    )
            shadows[engine_id] = shadow
        return shadows

    def _dispatch_global_pending(self) -> None:
        if not self._global_pending:
            return
        planner = self._admission_planner
        if planner is None:
            raise RuntimeError(
                "least_batch async admission requires "
                "admission_planner_config"
            )
        shadows = self._admission_shadows()
        available = {
            engine_id
            for engine_id in self._ready_engine_ids
            if (
                engine_id in shadows
                and self._engine_has_admission_window(engine_id)
            )
        }
        batches: dict[
            int, list[tuple[_GlobalPending, AdmissionReservation]]
        ] = {}
        assignment_order: list[_GlobalPending] = []
        capacity_blocked = False
        while self._global_pending and available:
            pending_global = self._global_pending[0]
            candidates = sorted(
                available,
                key=lambda candidate: (
                    self._projected_batch(candidate)
                    + len(batches.get(candidate, ())),
                    candidate,
                ),
            )
            selected: tuple[int, AdmissionReservation] | None = None
            for engine_id in candidates:
                candidate_shadow = shadows[engine_id].copy()
                reservation = planner.plan(
                    candidate_shadow, pending_global.command
                )
                if reservation is None:
                    continue
                selected = (engine_id, reservation)
                shadows[engine_id] = candidate_shadow
                break
            if selected is None:
                capacity_blocked = True
                break
            engine_id, reservation = selected
            self._global_pending.popleft()
            batches.setdefault(engine_id, []).append(
                (pending_global, reservation)
            )
            assignment_order.append(pending_global)
            if len(batches[engine_id]) >= self._admission_batch_size:
                available.remove(engine_id)

        dispatched_request_ids: set[int] = set()
        for engine_id, planned_batch in batches.items():
            batch = [pending for pending, _ in planned_batch]
            for pending_global, reservation in planned_batch:
                command = pending_global.command
                owner = self._owners.get(command.request_id)
                if (
                    owner is None
                    or owner.state != OwnerState.PENDING_GLOBAL
                    or owner.engine_id is not None
                ):
                    raise RuntimeError(
                        "global admission owner mismatch: "
                        f"request={command.request_id}, owner={owner}"
                    )
                self._owners[command.request_id] = RequestOwner(
                    OwnerState.PENDING_INGRESS, engine_id
                )
                self._charge_least_batch(
                    request_id=command.request_id,
                    engine_id=engine_id,
                    reservation=reservation,
                )
                self._admission_attempts[engine_id] += 1

            rpc_started_at = perf_counter()
            for pending_global in batch:
                command = pending_global.command
                pending_global.router_pending_ms += (
                    rpc_started_at - pending_global.router_queued_at
                ) * 1000
                if pending_global.capacity_blocked_since is not None:
                    self._global_capacity_queue_ms[
                        command.request_id
                    ] += (
                        rpc_started_at
                        - pending_global.capacity_blocked_since
                    ) * 1000
                    pending_global.capacity_blocked_since = None
            try:
                handle = self._engines[
                    engine_id
                ].admit_batch_async(
                    tuple(item.command for item in batch),
                    tuple(
                        reservation
                        for _, reservation in planned_batch
                    ),
                )
            except BaseException:
                for pending_global in batch:
                    request_id = pending_global.command.request_id
                    self._refund_least_batch(request_id)
                    self._owners[request_id] = RequestOwner(
                        OwnerState.PENDING_GLOBAL
                    )
                    pending_global.router_queued_at = rpc_started_at
                    pending_global.capacity_blocked_since = rpc_started_at
                restore = [
                    pending
                    for pending in assignment_order
                    if pending.command.request_id
                    not in dispatched_request_ids
                ]
                self._requeue_global_pending(restore)
                raise
            self._admission_flights[engine_id] = _AdmissionBatchFlight(
                engine_id=engine_id,
                pending=tuple(batch),
                handle=handle,
                rpc_started_at=rpc_started_at,
                capacity_epoch=self._capacity_epoch(engine_id),
            )
            dispatched_request_ids.update(
                pending.command.request_id for pending in batch
            )
        if self._global_pending:
            if capacity_blocked or (
                not self._admission_flights
                and any(
                    self._engine_blocked_capacity_epoch[engine_id]
                    is not None
                    for engine_id in self._ready_engine_ids
                )
            ):
                self._mark_global_capacity_blocked(perf_counter())

    def _ready_admission_batches(
        self,
    ) -> Mapping[int, tuple[IngressAck, ...]]:
        handles = {
            engine_id: flight.handle
            for engine_id, flight in self._admission_flights.items()
        }
        if not handles:
            return {}
        if self._poll_admission_batches is not None:
            return self._poll_admission_batches(handles)
        ready_batches: dict[int, tuple[IngressAck, ...]] = {}
        for engine_id, handle in handles.items():
            ready, acks = self._engines[
                engine_id
            ].poll_admission_batch(handle)
            if ready:
                if acks is None:
                    raise RuntimeError(
                        f"engine {engine_id} returned no admission batch"
                    )
                ready_batches[engine_id] = acks
        return ready_batches

    def _poll_centralized_admission(self) -> tuple[IngressAck, ...]:
        self._dispatch_global_pending()
        ready_batches = self._ready_admission_batches()
        unknown = set(ready_batches).difference(self._admission_flights)
        if unknown:
            raise RuntimeError(
                f"admission poller returned unknown engines {sorted(unknown)}"
            )

        acks: list[IngressAck] = []
        requeued: list[_GlobalPending] = []
        for engine_id in sorted(ready_batches):
            flight = self._admission_flights.pop(engine_id)
            engine_acks = tuple(ready_batches[engine_id])
            if len(engine_acks) != len(flight.pending):
                raise RuntimeError(
                    "LocalEngine admission batch result count mismatch: "
                    f"engine={engine_id}, commands={len(flight.pending)}, "
                    f"acks={len(engine_acks)}"
                )
            ack_observed_at = perf_counter()
            rpc_ms = (
                ack_observed_at - flight.rpc_started_at
            ) * 1000
            capacity_blocked = False
            blocked_capacity_epoch = flight.capacity_epoch
            for pending_global, ack in zip(
                flight.pending, engine_acks, strict=True
            ):
                command = pending_global.command
                request_id = command.request_id
                pending_global.admission_rpc_ms += rpc_ms
                if (
                    ack.request_id != request_id
                    or ack.engine_id != engine_id
                ):
                    self._refund_least_batch(request_id)
                    self._owners.pop(request_id, None)
                    raise RuntimeError(
                        "LocalEngine returned an inconsistent admission ACK: "
                        f"request={request_id}, engine={engine_id}, ack={ack}"
                    )

                transient = (
                    not ack.enqueued
                    and ack.reason
                    in {
                        "queue_full",
                        "admission_deferred",
                        "admission_state_mismatch",
                    }
                )
                if transient:
                    self._admission_state_mismatches += 1
                    capacity_blocked = True
                    if ack.capacity_epoch is not None:
                        blocked_capacity_epoch = max(
                            blocked_capacity_epoch,
                            ack.capacity_epoch,
                        )
                    if ack.reason == "admission_state_mismatch":
                        self._admission_local_state_mismatch[
                            engine_id
                        ] += 1
                    elif ack.reason == "admission_deferred":
                        self._admission_deferred[engine_id] += 1
                    else:
                        self._admission_queue_full[engine_id] += 1
                    self._refund_least_batch(request_id)
                    self._owners[request_id] = RequestOwner(
                        OwnerState.PENDING_GLOBAL
                    )
                    pending_global.router_queued_at = ack_observed_at
                    pending_global.capacity_blocked_since = ack_observed_at
                    requeued.append(pending_global)
                    self._admission_global_retries += 1
                    continue

                if ack.enqueued:
                    self._commit_least_batch(
                        request_id, ack.admission_version
                    )
                    self._admission_commits[engine_id] += 1
                    self._owners[request_id] = RequestOwner(
                        OwnerState.PENDING_ADD, engine_id
                    )
                    if self._wakeup is not None:
                        self._wave_id = self._wakeup(
                            engine_id, self._wave_id
                        )
                else:
                    self._refund_least_batch(request_id)
                    self._global_capacity_queue_ms.pop(request_id, None)
                    self._owners.pop(request_id, None)
                    self._rejected_request_ids.add(request_id)
                acks.append(
                    replace(
                        ack,
                        router_pending_ms=(
                            pending_global.router_pending_ms
                        ),
                        admission_rpc_ms=(
                            pending_global.admission_rpc_ms
                        ),
                    )
                )
            self._engine_blocked_capacity_epoch[engine_id] = (
                blocked_capacity_epoch if capacity_blocked else None
            )

        if requeued:
            ordered = sorted(
                requeued, key=lambda pending: pending.queue_seq
            )
            self._requeue_global_pending(ordered)
        self._dispatch_global_pending()
        return tuple(acks)

    def _kv_engine_has_dispatch_window(self, engine_id: int) -> bool:
        generation = self._kv_credit_generation.get(engine_id)
        return (
            generation is not None
            and engine_id not in self._ingress_batch_flights
            and self._kv_blocked_generation[engine_id] != generation
        )

    def _dispatch_kv_global_pending(self) -> None:
        if not self._global_pending:
            return

        available = {
            engine_id
            for engine_id in self._ready_engine_ids
            if self._kv_engine_has_dispatch_window(engine_id)
        }
        batches: dict[int, list[tuple[_GlobalPending, int]]] = {}
        assignment_order: list[_GlobalPending] = []
        kv_blocked = False
        unscheduled_blocked = False

        while self._global_pending and available:
            pending_global = self._global_pending[0]
            request_blocks = self._estimate_kv_admission_blocks(
                pending_global.command
            )
            kv_candidates = [
                engine_id
                for engine_id in available
                if self._kv_remaining_blocks.get(engine_id, 0)
                >= request_blocks
            ]
            if not kv_candidates:
                kv_blocked = True
                break
            candidates = [
                engine_id
                for engine_id in kv_candidates
                if self._projected_waiting(engine_id)
                < self._max_unscheduled_requests
            ]
            if not candidates:
                unscheduled_blocked = True
                break

            # KV is an eligibility gate, not the balancing score. Among
            # engines that can hold the FIFO head, choose the smallest
            # projected running+waiting load. Remaining KV is only a
            # tie-break so a roomy engine wins equal-load choices.
            engine_id = min(
                candidates,
                key=lambda candidate: (
                    self._projected_batch(candidate),
                    -(
                        self._kv_remaining_blocks[candidate]
                        - request_blocks
                    ),
                    candidate,
                ),
            )
            self._kv_remaining_blocks[engine_id] -= request_blocks

            self._global_pending.popleft()
            batches.setdefault(engine_id, []).append(
                (pending_global, request_blocks)
            )
            assignment_order.append(pending_global)
            request_id = pending_global.command.request_id
            owner = self._owners.get(request_id)
            if (
                owner is None
                or owner.state != OwnerState.PENDING_GLOBAL
                or owner.engine_id is not None
            ):
                raise RuntimeError(
                    "KV-gated global owner mismatch: "
                    f"request={request_id}, owner={owner}"
                )
            self._charge_kv_ingress(
                request_id=request_id,
                engine_id=engine_id,
                blocks=request_blocks,
            )
            self._owners[request_id] = RequestOwner(
                OwnerState.PENDING_INGRESS, engine_id
            )
            self._kv_ingress_attempts[engine_id] += 1
            if len(batches[engine_id]) >= self._admission_batch_size:
                available.remove(engine_id)

        dispatched_request_ids: set[int] = set()
        for engine_id, planned_batch in batches.items():
            batch = [pending for pending, _ in planned_batch]
            rpc_started_at = perf_counter()
            for pending_global in batch:
                request_id = pending_global.command.request_id
                pending_global.router_pending_ms += (
                    rpc_started_at - pending_global.router_queued_at
                ) * 1000
                if pending_global.capacity_blocked_since is not None:
                    self._global_capacity_queue_ms[request_id] += (
                        rpc_started_at
                        - pending_global.capacity_blocked_since
                    ) * 1000
                    pending_global.capacity_blocked_since = None
            try:
                handle = self._engines[engine_id].enqueue_batch_async(
                    tuple(item.command for item in batch)
                )
            except BaseException:
                restore = [
                    pending
                    for pending in assignment_order
                    if pending.command.request_id
                    not in dispatched_request_ids
                ]
                for pending_global in restore:
                    request_id = pending_global.command.request_id
                    self._release_kv_ingress_charge(request_id)
                    self._owners[request_id] = RequestOwner(
                        OwnerState.PENDING_GLOBAL
                    )
                    pending_global.router_queued_at = rpc_started_at
                self._requeue_global_pending(restore)
                raise

            self._ingress_batch_flights[engine_id] = _IngressBatchFlight(
                engine_id=engine_id,
                pending=tuple(batch),
                handle=handle,
                rpc_started_at=rpc_started_at,
            )
            dispatched_request_ids.update(
                pending.command.request_id for pending in batch
            )

        if self._global_pending and (kv_blocked or unscheduled_blocked):
            if kv_blocked:
                self._kv_gate_blocked += 1
            if unscheduled_blocked:
                self._unscheduled_gate_blocked += 1
            self._mark_global_capacity_blocked(perf_counter())

    def _ready_ingress_batches(
        self,
    ) -> Mapping[int, tuple[IngressAck, ...]]:
        handles = {
            engine_id: flight.handle
            for engine_id, flight in self._ingress_batch_flights.items()
        }
        if not handles:
            return {}
        if self._poll_admission_batches is not None:
            return self._poll_admission_batches(handles)
        ready_batches: dict[int, tuple[IngressAck, ...]] = {}
        for engine_id, handle in handles.items():
            ready, acks = self._engines[
                engine_id
            ].poll_admission_batch(handle)
            if ready:
                if acks is None:
                    raise RuntimeError(
                        f"engine {engine_id} returned no ingress batch"
                    )
                ready_batches[engine_id] = acks
        return ready_batches

    def _poll_kv_ingress_batches(self) -> tuple[IngressAck, ...]:
        self._dispatch_kv_global_pending()
        ready_batches = self._ready_ingress_batches()
        unknown = set(ready_batches).difference(
            self._ingress_batch_flights
        )
        if unknown:
            raise RuntimeError(
                f"ingress poller returned unknown engines {sorted(unknown)}"
            )

        acks: list[IngressAck] = []
        requeued: list[_GlobalPending] = []
        queue_full_engines: set[int] = set()
        for engine_id in sorted(ready_batches):
            flight = self._ingress_batch_flights.pop(engine_id)
            engine_acks = tuple(ready_batches[engine_id])
            if len(engine_acks) != len(flight.pending):
                raise RuntimeError(
                    "LocalEngine ingress batch result count mismatch: "
                    f"engine={engine_id}, commands={len(flight.pending)}, "
                    f"acks={len(engine_acks)}"
                )
            ack_observed_at = perf_counter()
            rpc_ms = (
                ack_observed_at - flight.rpc_started_at
            ) * 1000
            for pending_global, ack in zip(
                flight.pending,
                engine_acks,
                strict=True,
            ):
                request_id = pending_global.command.request_id
                pending_global.admission_rpc_ms += rpc_ms
                if (
                    ack.request_id != request_id
                    or ack.engine_id != engine_id
                ):
                    self._future_ingress_aborts.discard(request_id)
                    self._release_kv_ingress_charge(request_id)
                    self._owners.pop(request_id, None)
                    raise RuntimeError(
                        "LocalEngine returned an inconsistent ingress ACK: "
                        f"request={request_id}, engine={engine_id}, ack={ack}"
                    )

                future_abort = request_id in self._future_ingress_aborts
                if future_abort:
                    self._future_ingress_aborts.discard(request_id)
                    if ack.enqueued:
                        # The receipt establishes that enqueue is no longer a
                        # future actor call. Reconcile the tombstone with the
                        # concrete ingress/scheduler state and clean it up if
                        # LocalEngine had already rejected the request.
                        self._engines[engine_id].abort(request_id)
                    else:
                        self._engines[engine_id].clear_ingress_abort(
                            request_id
                        )
                        ack = replace(ack, reason="aborted")

                if not ack.enqueued and ack.reason == "queue_full":
                    self._release_kv_ingress_charge(request_id)
                    self._kv_ingress_queue_full[engine_id] += 1
                    queue_full_engines.add(engine_id)
                    self._admission_global_retries += 1
                    self._owners[request_id] = RequestOwner(
                        OwnerState.PENDING_GLOBAL
                    )
                    pending_global.router_queued_at = ack_observed_at
                    pending_global.capacity_blocked_since = ack_observed_at
                    requeued.append(pending_global)
                    continue

                if ack.enqueued:
                    self._kv_ingress_receipts[engine_id] += 1
                    self._owners[request_id] = RequestOwner(
                        OwnerState.PENDING_ADD, engine_id
                    )
                    if self._wakeup is not None:
                        self._wave_id = self._wakeup(
                            engine_id, self._wave_id
                        )
                else:
                    self._release_kv_ingress_charge(request_id)
                    self._owners.pop(request_id, None)
                    capacity_queue_ms = self._global_capacity_queue_ms.pop(
                        request_id, 0.0
                    )
                    if ack.reason == "aborted":
                        self._terminal[request_id] = FinishEvent(
                            request_id=request_id,
                            generated_count=0,
                            status="ABORTED",
                            engine_id=engine_id,
                            global_capacity_queue_ms=capacity_queue_ms,
                        )
                    else:
                        self._rejected_request_ids.add(request_id)
                acks.append(
                    replace(
                        ack,
                        router_pending_ms=pending_global.router_pending_ms,
                        admission_rpc_ms=pending_global.admission_rpc_ms,
                    )
                )

        if requeued:
            # A stale queue-slot snapshot blocks only the engine that rejected
            # the batch. Healthy engines may immediately take the requeued
            # global FIFO head using their remaining ledger-backed credit.
            for engine_id in queue_full_engines:
                self._kv_blocked_generation[engine_id] = (
                    self._kv_credit_generation.get(engine_id)
                )
            ordered = sorted(
                requeued, key=lambda pending: pending.queue_seq
            )
            self._requeue_global_pending(ordered)
            self._mark_global_capacity_blocked(perf_counter())
        if ready_batches:
            # Receipts free only the bounded transport flight. The request's
            # KV and unscheduled-load charge remains live until first schedule,
            # so ACK latency cannot become compute dispatch credit.
            self._dispatch_kv_global_pending()
        return tuple(acks)

    def poll_ingress_acks(self) -> tuple[IngressAck, ...]:
        acks = list(self._immediate_ingress_acks)
        self._immediate_ingress_acks.clear()
        if self.router_policy == "least_batch":
            acks.extend(self._poll_centralized_admission())
            return tuple(acks)
        if self.router_policy == "least_batch_v2":
            acks.extend(self._poll_kv_ingress_batches())
            return tuple(acks)
        for request_id, pending in tuple(self._pending_ingress.items()):
            engine_id = pending.candidate_engine_ids[
                pending.candidate_index
            ]
            ready, ack = self._engines[engine_id].poll_enqueue(
                pending.handle
            )
            if not ready:
                continue
            ack_observed_at = perf_counter()
            pending.admission_rpc_ms += (
                ack_observed_at - pending.rpc_started_at
            ) * 1000
            if ack is None:
                raise RuntimeError(
                    f"engine {engine_id} returned no ready ingress ACK"
                )
            if ack.request_id != request_id or ack.engine_id != engine_id:
                self._refund_cache(request_id)
                self._refund_least_batch(request_id)
                self._owners.pop(request_id, None)
                self._pending_ingress.pop(request_id, None)
                raise RuntimeError(
                    "LocalEngine returned an inconsistent ingress ACK: "
                    f"request={request_id}, engine={engine_id}, ack={ack}"
                )
            transient_rejection = (
                not ack.enqueued and ack.reason == "queue_full"
            )
            if (
                transient_rejection
                and pending.candidate_index + 1
                < len(pending.candidate_engine_ids)
            ):
                self._refund_cache(request_id)
                pending.candidate_index += 1
                fallback_engine_id = pending.candidate_engine_ids[
                    pending.candidate_index
                ]
                self._owners[request_id] = RequestOwner(
                    OwnerState.PENDING_INGRESS, fallback_engine_id
                )
                self._charge_cache(
                    request_id=request_id,
                    engine_id=fallback_engine_id,
                    prompt_token_ids=pending.command.prompt_token_ids,
                    max_tokens=pending.command.max_tokens,
                )
                try:
                    transport = self._engines[fallback_engine_id]
                    fallback_started_at = perf_counter()
                    pending.router_pending_ms += (
                        fallback_started_at - ack_observed_at
                    ) * 1000
                    pending.handle = transport.enqueue_async(
                        pending.command
                    )
                    pending.rpc_started_at = fallback_started_at
                except BaseException:
                    self._refund_cache(request_id)
                    self._owners.pop(request_id, None)
                    self._pending_ingress.pop(request_id, None)
                    raise
                continue

            self._pending_ingress.pop(request_id, None)
            if ack.enqueued:
                self._owners[request_id] = RequestOwner(
                    OwnerState.PENDING_ADD, engine_id
                )
                if self._wakeup is not None:
                    self._wave_id = self._wakeup(
                        engine_id, self._wave_id
                    )
            else:
                self._refund_cache(request_id)
                self._refund_least_batch(request_id)
                self._global_capacity_queue_ms.pop(request_id, None)
                self._owners.pop(request_id, None)
                self._rejected_request_ids.add(request_id)
            acks.append(
                replace(
                    ack,
                    router_pending_ms=pending.router_pending_ms,
                    admission_rpc_ms=pending.admission_rpc_ms,
                )
            )
        return tuple(acks)

    def record_add_results(
        self, events: Iterable[AddResultEvent]
    ) -> tuple[AddResultEvent, ...]:
        ready_events: list[AddResultEvent] = []
        incoming = tuple(events)
        for event in incoming:
            owner = self._owners.get(event.request_id)
            if (
                owner is not None
                and owner.state == OwnerState.PENDING_INGRESS
                and owner.engine_id == event.engine_id
            ):
                if event.request_id in self._early_add_results:
                    raise RuntimeError(
                        f"duplicate early ADD result for {event.request_id}"
                    )
                self._early_add_results[event.request_id] = event
                continue
            ready_events.append(event)

        for request_id, event in tuple(self._early_add_results.items()):
            owner = self._owners.get(request_id)
            if owner is not None and owner.state == OwnerState.PENDING_ADD:
                ready_events.append(event)
                self._early_add_results.pop(request_id)

        for event in ready_events:
            owner = self._owners.get(event.request_id)
            if (
                owner is None
                or owner.state != OwnerState.PENDING_ADD
                or owner.engine_id != event.engine_id
            ):
                raise RuntimeError(
                    "ADD result owner mismatch: "
                    f"request={event.request_id}, "
                    f"engine={event.engine_id}, owner={owner}"
                )
            if event.accepted:
                self._owners[event.request_id] = RequestOwner(
                    OwnerState.OWNED, event.engine_id
                )
            else:
                self._future_ingress_aborts.discard(event.request_id)
                self._release_kv_ingress_charge(event.request_id)
                self._refund_cache(event.request_id)
                self._refund_least_batch(event.request_id)
                self._global_capacity_queue_ms.pop(event.request_id, None)
                self._owners.pop(event.request_id)
                self._rejected_request_ids.add(event.request_id)
        return tuple(ready_events)

    def record_first_schedule_events(
        self, events: Iterable[FirstScheduleEvent]
    ) -> tuple[FirstScheduleEvent, ...]:
        def capacity_queue_ms(event: FirstScheduleEvent) -> float:
            if event.request_id in self._global_capacity_queue_ms:
                return self._global_capacity_queue_ms[event.request_id]
            terminal = self._terminal.get(event.request_id)
            if terminal is not None:
                return terminal.global_capacity_queue_ms
            return 0.0

        ready_events = []
        for event in events:
            charge = self._kv_ingress_charges.get(event.request_id)
            if charge is not None and charge.engine_id != event.engine_id:
                raise RuntimeError(
                    "first-schedule KV charge owner mismatch: "
                    f"request={event.request_id}, engine={event.engine_id}, "
                    f"charge_engine={charge.engine_id}"
                )
            self._release_kv_ingress_charge(event.request_id)
            ready_events.append(
                replace(
                    event,
                    global_capacity_queue_ms=capacity_queue_ms(event),
                )
            )
        if ready_events and self.router_policy == "least_batch_v2":
            # First schedule is the compute-credit boundary. Refill directly
            # instead of waiting for the next frontend poll or load snapshot.
            self._dispatch_kv_global_pending()
        return tuple(ready_events)

    def abort(self, request_id: int) -> AbortResult:
        if request_id in self._terminal:
            return AbortResult(
                request_id=request_id, status="already_terminal"
            )
        owner = self._owners.get(request_id)
        if owner is None:
            return AbortResult(request_id=request_id, status="not_found")
        if owner.state == OwnerState.PENDING_GLOBAL:
            self._global_pending = deque(
                pending
                for pending in self._global_pending
                if pending.command.request_id != request_id
            )
            self._owners.pop(request_id)
            self._future_ingress_aborts.discard(request_id)
            self._release_kv_ingress_charge(request_id)
            self._global_capacity_queue_ms.pop(request_id, None)
            self._terminal[request_id] = FinishEvent(
                request_id=request_id,
                generated_count=0,
                status="ABORTED",
                engine_id=-1,
            )
            return AbortResult(request_id=request_id, status="aborted")
        if owner.engine_id is None:
            return AbortResult(request_id=request_id, status="abort_pending")
        allow_future_ingress = (
            self.router_policy == "least_batch_v2"
            and owner.state == OwnerState.PENDING_INGRESS
        )
        result = self._engines[owner.engine_id].abort(
            request_id,
            allow_future_ingress=allow_future_ingress,
        )
        if result.request_id != request_id:
            raise RuntimeError(
                "LocalEngine returned an inconsistent abort request id: "
                f"expected={request_id}, got={result.request_id}"
            )
        if allow_future_ingress and result.status == "abort_pending":
            self._future_ingress_aborts.add(request_id)
        return result

    def finish(self, event: FinishEvent) -> FinishEvent:
        if event.status not in {"FINISHED", "ABORTED"}:
            raise ValueError(f"invalid terminal status {event.status!r}")
        if event.request_id in self._terminal:
            raise RuntimeError(
                f"duplicate terminal event for request {event.request_id}"
            )
        owner = self._owners.get(event.request_id)
        if (
            owner is None
            or owner.state != OwnerState.OWNED
            or owner.engine_id != event.engine_id
        ):
            raise RuntimeError(
                "terminal event owner mismatch: "
                f"request={event.request_id}, engine={event.engine_id}, "
                f"owner={owner}"
            )
        self._owners.pop(event.request_id)
        self._future_ingress_aborts.discard(event.request_id)
        self._release_kv_ingress_charge(event.request_id)
        self._refund_cache(event.request_id)
        self._refund_least_batch(event.request_id)
        event = replace(
            event,
            global_capacity_queue_ms=self._global_capacity_queue_ms.pop(
                event.request_id, 0.0
            ),
        )
        self._terminal[event.request_id] = event
        return event

    def record_finish_events(
        self, events: Iterable[FinishEvent]
    ) -> tuple[FinishEvent, ...]:
        ready_events: list[FinishEvent] = []
        for event in events:
            if (
                event.request_id in self._terminal
                or event.request_id in self._early_terminal_events
            ):
                raise RuntimeError(
                    f"duplicate terminal event for request {event.request_id}"
                )
            owner = self._owners.get(event.request_id)
            if (
                owner is not None
                and owner.state
                in {OwnerState.PENDING_INGRESS, OwnerState.PENDING_ADD}
                and owner.engine_id == event.engine_id
            ):
                self._early_terminal_events[event.request_id] = event
                continue
            ready_events.append(event)

        for request_id, event in tuple(
            self._early_terminal_events.items()
        ):
            owner = self._owners.get(request_id)
            if (
                owner is not None
                and owner.state == OwnerState.OWNED
                and owner.engine_id == event.engine_id
            ):
                ready_events.append(event)
                self._early_terminal_events.pop(request_id)
                continue
            if (
                owner is None
                or owner.engine_id != event.engine_id
                or owner.state
                not in {
                    OwnerState.PENDING_INGRESS,
                    OwnerState.PENDING_ADD,
                }
            ):
                self._early_terminal_events.pop(request_id)
                ready_events.append(event)

        return tuple(self.finish(event) for event in ready_events)

    def refresh_loads(self) -> dict[int, LoadSnapshot]:
        snapshots = {
            engine_id: engine.load()
            for engine_id, engine in self._engines.items()
        }
        self.record_loads(snapshots.values())
        return dict(snapshots)

    def record_loads(self, snapshots: Iterable[LoadSnapshot]) -> None:
        snapshots = tuple(snapshots)
        loads = {snapshot.engine_id: snapshot for snapshot in snapshots}
        if len(loads) != len(snapshots):
            raise ValueError("load report contains duplicate engine ids")
        unknown = set(loads).difference(self._engines)
        if unknown:
            raise ValueError(
                f"load report contains unknown engines {sorted(unknown)}"
            )
        previous_loads = {
            engine_id: self._loads.get(engine_id) for engine_id in loads
        }
        self._loads.update(loads)
        for engine_id, snapshot in loads.items():
            self._record_kv_credit_snapshot(
                snapshot, refresh_load_projection=True
            )
            for request_id, charge in tuple(
                self._least_batch_charges.items()
            ):
                if (
                    charge.engine_id == engine_id
                    and charge.admission_version is not None
                    and snapshot.admission_version
                    >= charge.admission_version
                ):
                    self._refund_least_batch(request_id)
            previous = previous_loads[engine_id]
            if (
                previous is not None
                and previous.wave_id == snapshot.wave_id
                and previous.quantum_id == snapshot.quantum_id
                and previous.free_blocks_min == snapshot.free_blocks_min
            ):
                # Re-reading an unchanged cached snapshot must not erase
                # deductions for submissions made since that snapshot.
                continue
            self._estimated_free_blocks[engine_id] = snapshot.free_blocks_min
            # A changed snapshot is authoritative for prior optimistic
            # deductions against that engine.
            for request_id, (
                charged_engine_id,
                _request_blocks,
            ) in tuple(self._cache_charges.items()):
                if charged_engine_id == engine_id:
                    self._cache_charges.pop(request_id)

    def last_loads(self) -> dict[int, LoadSnapshot]:
        return dict(self._loads)

    def admission_metrics(self) -> dict[str, Any]:
        per_engine = {}
        for engine_id in self._ready_engine_ids:
            snapshot = self._loads.get(engine_id)
            running = snapshot.running if snapshot is not None else 0
            waiting = snapshot.waiting if snapshot is not None else 0
            pending_ingress = (
                snapshot.pending_ingress if snapshot is not None else 0
            )
            tentative = self._least_batch_tentative_counts[engine_id]
            engine_metrics = {
                "running_snapshot": running,
                "tentative_admissions": tentative,
                "projected_batch": self._projected_batch(engine_id),
                "attempts": self._admission_attempts[engine_id],
                "commits": self._admission_commits[engine_id],
                "deferred": self._admission_deferred[engine_id],
                "queue_full": self._admission_queue_full[engine_id],
                "state_mismatches": (
                    self._admission_local_state_mismatch[engine_id]
                ),
            }
            if self.router_policy == "least_batch_v2":
                engine_metrics.update(
                    {
                        "waiting_snapshot": waiting,
                        "pending_ingress_snapshot": pending_ingress,
                        "projected_waiting": self._projected_waiting(
                            engine_id
                        ),
                        "max_unscheduled_requests": (
                            self._max_unscheduled_requests
                        ),
                        "unscheduled_window_remaining": max(
                            0,
                            self._max_unscheduled_requests
                            - self._projected_waiting(engine_id),
                        ),
                        "kv_free_blocks_snapshot": (
                            sum(
                                rank.free_blocks
                                for rank in snapshot.rank_loads
                            )
                            if snapshot is not None
                            else None
                        ),
                        "kv_credit_remaining": (
                            self._kv_remaining_blocks.get(engine_id)
                        ),
                        "kv_outstanding_charges": (
                            self._kv_unscheduled_counts[engine_id]
                        ),
                        "kv_outstanding_blocks": (
                            self._kv_outstanding_blocks[engine_id]
                        ),
                        "ingress_attempts": self._kv_ingress_attempts[
                            engine_id
                        ],
                        "ingress_receipts": self._kv_ingress_receipts[
                            engine_id
                        ],
                        "ingress_queue_full": (
                            self._kv_ingress_queue_full[engine_id]
                        ),
                    }
                )
            per_engine[str(engine_id)] = engine_metrics
        flights: Mapping[int, _AdmissionBatchFlight | _IngressBatchFlight]
        flights = (
            self._ingress_batch_flights
            if self.router_policy == "least_batch_v2"
            else self._admission_flights
        )
        return {
            "policy": self.router_policy,
            "global_pending": self.pending_global_count,
            "pending_rpc": len(flights),
            "pending_rpc_requests": sum(
                len(flight.pending)
                for flight in flights.values()
            ),
            "admission_batch_size": self._admission_batch_size,
            "max_unscheduled_requests": (
                self._max_unscheduled_requests
                if self.router_policy == "least_batch_v2"
                else None
            ),
            "fallbacks": self._admission_fallbacks,
            "global_retries": self._admission_global_retries,
            "state_mismatches": self._admission_state_mismatches,
            "kv_gate_blocked": self._kv_gate_blocked,
            "unscheduled_gate_blocked": (
                self._unscheduled_gate_blocked
            ),
            "future_ingress_aborts": len(self._future_ingress_aborts),
            "per_engine": per_engine,
        }
