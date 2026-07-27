from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Mapping, Protocol

from nanodeploy.engine.hierarchical_contract import (
    AddCommand,
    AddResult,
    AddResultEvent,
    AbortResult,
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

    def poll_enqueue(
        self, handle: Any
    ) -> tuple[bool, IngressAck | None]: ...

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


WakeupCallback = Callable[[int, int], int]


class RequestRouter:
    """READY-gated load routing with sticky request ownership."""

    def __init__(
        self,
        engines: Mapping[int, EngineTransport],
        *,
        router_policy: RouterPolicy = "least_batch",
        kvcache_block_size: int | None = None,
        wakeup: WakeupCallback | None = None,
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
        self._ready_engine_ids = tuple(self._engines)
        self.router_policy = router_policy
        self._kvcache_block_size = kvcache_block_size
        self._owners: dict[int, RequestOwner] = {}
        self._terminal: dict[int, FinishEvent] = {}
        self._rejected_request_ids: set[int] = set()
        self._pending_ingress: dict[int, _PendingIngress] = {}
        self._immediate_ingress_acks: deque[IngressAck] = deque()
        self._early_add_results: dict[int, AddResultEvent] = {}
        self._loads: dict[int, LoadSnapshot] = {}
        self._estimated_free_blocks: dict[int, int] = {}
        self._cache_charges: dict[int, tuple[int, int]] = {}
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
            owner.state == OwnerState.PENDING_INGRESS
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

    def _owner_counts(self) -> dict[int, int]:
        counts = {engine_id: 0 for engine_id in self._ready_engine_ids}
        for owner in self._owners.values():
            if owner.engine_id is not None:
                counts[owner.engine_id] += 1
        return counts

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
        # The rotating base order supplies deterministic, starvation-free
        # tie-breaking for every policy and advances exactly once per request.
        candidates = self._round_robin_candidates()
        if self.router_policy == "round_robin":
            return candidates
        if self.router_policy == "least_batch":
            owner_counts = self._owner_counts()
            return tuple(
                sorted(candidates, key=lambda engine_id: owner_counts[engine_id])
            )

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
        """Start ingress submission without waiting for a Ray result."""
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

        candidates = self._candidate_engine_ids(
            prompt_token_ids=prompt_token_ids,
            max_tokens=max_tokens,
        )
        engine_id = candidates[0]
        command = AddCommand(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            ignore_eos=ignore_eos,
            wave_id=self._wave_id,
        )
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
        )
        return request_id

    def poll_ingress_acks(self) -> tuple[IngressAck, ...]:
        acks = list(self._immediate_ingress_acks)
        self._immediate_ingress_acks.clear()
        for request_id, pending in tuple(self._pending_ingress.items()):
            engine_id = pending.candidate_engine_ids[
                pending.candidate_index
            ]
            ready, ack = self._engines[engine_id].poll_enqueue(
                pending.handle
            )
            if not ready:
                continue
            if ack is None:
                raise RuntimeError(
                    f"engine {engine_id} returned no ready ingress ACK"
                )
            if ack.request_id != request_id or ack.engine_id != engine_id:
                self._refund_cache(request_id)
                self._owners.pop(request_id, None)
                self._pending_ingress.pop(request_id, None)
                raise RuntimeError(
                    "LocalEngine returned an inconsistent ingress ACK: "
                    f"request={request_id}, engine={engine_id}, ack={ack}"
                )
            if (
                not ack.enqueued
                and ack.reason == "queue_full"
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
                    pending.handle = self._engines[
                        fallback_engine_id
                    ].enqueue_async(pending.command)
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
                self._owners.pop(request_id, None)
                self._rejected_request_ids.add(request_id)
            acks.append(ack)
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
                self._owners.pop(event.request_id)
                self._rejected_request_ids.add(event.request_id)
        return tuple(ready_events)

    def abort(self, request_id: int) -> AbortResult:
        if request_id in self._terminal:
            return AbortResult(
                request_id=request_id, status="already_terminal"
            )
        owner = self._owners.get(request_id)
        if owner is None:
            return AbortResult(request_id=request_id, status="not_found")
        if owner.engine_id is None:
            return AbortResult(request_id=request_id, status="abort_pending")
        result = self._engines[owner.engine_id].abort(request_id)
        if result.request_id != request_id:
            raise RuntimeError(
                "LocalEngine returned an inconsistent abort request id: "
                f"expected={request_id}, got={result.request_id}"
            )
        return result

    def finish(self, event: FinishEvent) -> None:
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
        self._terminal[event.request_id] = event

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
