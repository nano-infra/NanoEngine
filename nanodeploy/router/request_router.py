from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, Callable, Iterable, Literal, Mapping, Protocol

from nanodeploy.engine.hierarchical_contract import (
    AddCommand,
    AddResult,
    AddResultEvent,
    AbortResult,
    FirstScheduleEvent,
    FinishEvent,
    IngressAck,
    LoadSnapshot,
    OwnerState,
    round_up,
)

RouterPolicy = Literal["round_robin", "least_batch", "least_cache"]
_ROUTER_POLICIES = frozenset(
    {"round_robin", "least_batch", "least_cache"}
)


class EngineTransport(Protocol):
    """Synchronous control-plane view of one LocalEngineCore."""

    engine_id: int

    def add(self, command: AddCommand) -> AddResult: ...

    def enqueue_async(self, command: AddCommand) -> Any: ...

    def admit_async(self, command: AddCommand) -> Any: ...

    def admit_batch_async(
        self, commands: tuple[AddCommand, ...]
    ) -> Any: ...

    def poll_enqueue(
        self, handle: Any
    ) -> tuple[bool, IngressAck | None]: ...

    def poll_admission_batch(
        self, handle: Any
    ) -> tuple[bool, tuple[IngressAck, ...] | None]: ...

    def abort(self, request_id: int) -> AbortResult: ...

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
    dispatch_order: int


@dataclass(slots=True)
class _LeastBatchCharge:
    engine_id: int
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
        self._ready_engine_ids = tuple(self._engines)
        self.router_policy = router_policy
        self._kvcache_block_size = kvcache_block_size
        self._owners: dict[int, RequestOwner] = {}
        self._terminal: dict[int, FinishEvent] = {}
        self._rejected_request_ids: set[int] = set()
        self._pending_ingress: dict[int, _PendingIngress] = {}
        self._global_pending: deque[_GlobalPending] = deque()
        self._admission_flights: dict[int, _AdmissionBatchFlight] = {}
        self._admission_batch_size = admission_batch_size
        self._poll_admission_batches = poll_admission_batches
        self._engine_blocked_capacity_epoch: dict[int, int | None] = {
            engine_id: None for engine_id in self._ready_engine_ids
        }
        self._next_batch_dispatch_order = 0
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
        self._admission_fallbacks = 0
        self._admission_global_retries = 0
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

    def _projected_batch(self, engine_id: int) -> int:
        snapshot = self._loads.get(engine_id)
        running = snapshot.running if snapshot is not None else 0
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

    def _candidate_engine_ids(
        self,
        *,
        prompt_token_ids: tuple[int, ...],
        max_tokens: int,
    ) -> tuple[int, ...]:
        if self.router_policy == "least_batch":
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
        self, *, request_id: int, engine_id: int
    ) -> None:
        if self.router_policy != "least_batch":
            return
        if request_id in self._least_batch_charges:
            raise RuntimeError(
                f"duplicate least-batch charge for {request_id}"
            )
        self._least_batch_charges[request_id] = _LeastBatchCharge(engine_id)
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
        if self.router_policy == "least_batch":
            self._owners[request_id] = RequestOwner(
                OwnerState.PENDING_GLOBAL
            )
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
            capacity_blocked_since = (
                submitted_at if queue_is_capacity_blocked else None
            )
            self._global_pending.append(
                _GlobalPending(
                    command,
                    router_queued_at=submitted_at,
                    capacity_blocked_since=capacity_blocked_since,
                )
            )
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

    def _dispatch_global_pending(self) -> None:
        available = {
            engine_id
            for engine_id in self._ready_engine_ids
            if self._engine_has_admission_window(engine_id)
        }
        batches: dict[int, list[_GlobalPending]] = {}
        assignment_order: list[_GlobalPending] = []
        while self._global_pending and available:
            engine_id = min(
                available,
                key=lambda candidate: (
                    self._projected_batch(candidate)
                    + len(batches.get(candidate, ())),
                    candidate,
                ),
            )
            pending_global = self._global_pending.popleft()
            batches.setdefault(engine_id, []).append(pending_global)
            assignment_order.append(pending_global)
            if len(batches[engine_id]) >= self._admission_batch_size:
                available.remove(engine_id)

        dispatched_request_ids: set[int] = set()
        for engine_id, batch in batches.items():
            for pending_global in batch:
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
                    tuple(item.command for item in batch)
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
                self._global_pending.extendleft(reversed(restore))
                raise
            self._admission_flights[engine_id] = _AdmissionBatchFlight(
                engine_id=engine_id,
                pending=tuple(batch),
                handle=handle,
                rpc_started_at=rpc_started_at,
                capacity_epoch=self._capacity_epoch(engine_id),
                dispatch_order=self._next_batch_dispatch_order,
            )
            self._next_batch_dispatch_order += 1
            dispatched_request_ids.update(
                pending.command.request_id for pending in batch
            )
        if self._global_pending:
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
        requeued: list[tuple[int, _GlobalPending]] = []
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
                    and ack.reason in {"queue_full", "admission_deferred"}
                )
                if transient:
                    capacity_blocked = True
                    if ack.capacity_epoch is not None:
                        blocked_capacity_epoch = max(
                            blocked_capacity_epoch,
                            ack.capacity_epoch,
                        )
                    if ack.reason == "admission_deferred":
                        self._admission_deferred[engine_id] += 1
                    else:
                        self._admission_queue_full[engine_id] += 1
                    self._refund_least_batch(request_id)
                    self._owners[request_id] = RequestOwner(
                        OwnerState.PENDING_GLOBAL
                    )
                    pending_global.router_queued_at = ack_observed_at
                    pending_global.capacity_blocked_since = ack_observed_at
                    requeued.append(
                        (flight.dispatch_order, pending_global)
                    )
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
            ordered = [
                pending
                for _, pending in sorted(
                    requeued,
                    key=lambda item: item[0],
                )
            ]
            self._global_pending.extendleft(reversed(ordered))
        self._dispatch_global_pending()
        return tuple(acks)

    def poll_ingress_acks(self) -> tuple[IngressAck, ...]:
        acks = list(self._immediate_ingress_acks)
        self._immediate_ingress_acks.clear()
        if self.router_policy == "least_batch":
            acks.extend(self._poll_centralized_admission())
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
                self._refund_cache(event.request_id)
                self._refund_least_batch(event.request_id)
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

        return tuple(
            replace(
                event,
                global_capacity_queue_ms=capacity_queue_ms(event),
            )
            for event in events
        )

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
        result = self._engines[owner.engine_id].abort(request_id)
        if result.request_id != request_id:
            raise RuntimeError(
                "LocalEngine returned an inconsistent abort request id: "
                f"expected={request_id}, got={result.request_id}"
            )
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
            tentative = self._least_batch_tentative_counts[engine_id]
            per_engine[str(engine_id)] = {
                "running_snapshot": running,
                "tentative_admissions": tentative,
                "projected_batch": running + tentative,
                "attempts": self._admission_attempts[engine_id],
                "commits": self._admission_commits[engine_id],
                "deferred": self._admission_deferred[engine_id],
                "queue_full": self._admission_queue_full[engine_id],
            }
        return {
            "policy": self.router_policy,
            "global_pending": self.pending_global_count,
            "pending_rpc": len(self._admission_flights),
            "pending_rpc_requests": sum(
                len(flight.pending)
                for flight in self._admission_flights.values()
            ),
            "admission_batch_size": self._admission_batch_size,
            "fallbacks": self._admission_fallbacks,
            "global_retries": self._admission_global_retries,
            "per_engine": per_engine,
        }
