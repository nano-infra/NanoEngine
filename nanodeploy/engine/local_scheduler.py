from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from math import ceil
from time import perf_counter

from nanodeploy._cpp import BlockContextSlot
from nanodeploy.config import Config
from nanodeploy.engine.hierarchical_contract import (
    AddCommand,
    AddResult,
    AdmissionReservation,
    AbortResult,
    FirstScheduleEvent,
    FirstTokenEvent,
    FinishEvent,
    HIERARCHICAL_LOOP_COUNT,
    LoadSnapshot,
    LocalDecodeBatch,
    RankLoad,
    RequestState,
    WorkerDecodeResult,
    round_up,
    validate_add_request,
)
from nanodeploy.engine.scheduler import Scheduler
from nanodeploy.engine.sequence import Sequence
from nanodeploy.engine.sequence import SequenceStatus
from nanodeploy.engine.topology import EngineTopology
from nanodeploy.sampling_params import SamplingParams


@dataclass(slots=True)
class LocalRequestRecord:
    sequence: Sequence
    state: RequestState
    original_prompt_len: int
    padded_completion_len: int
    scheduler_enqueued_at: float
    first_forward_started_at: float | None = None
    first_token_emitted: bool = False
    terminal_emitted: bool = False


@dataclass(frozen=True, slots=True)
class _ActiveLoadState:
    waiting: int
    running: int
    master_counts: tuple[int, ...]
    receiver_counts: tuple[int, ...]
    dispatched_tokens: tuple[int, ...]


class LocalScheduler:
    """Single-writer scheduler state for exactly one attention DP group."""

    def __init__(
        self,
        config: Config,
        topology: EngineTopology,
        *,
        bootstrap_token_id: int = 0,
    ) -> None:
        if config.scheduler_arch != "hierarchical":
            raise ValueError("LocalScheduler requires scheduler_arch='hierarchical'")
        if topology.global_dp_idx >= config.attention_dp:
            raise ValueError("LocalScheduler topology is outside the deployment")
        if topology.attention_sp != config.attention_sp:
            raise ValueError("LocalScheduler topology/config SP mismatch")
        if topology.attention_tp != config.attention_tp:
            raise ValueError("LocalScheduler topology/config TP mismatch")
        if not 0 <= bootstrap_token_id < config.hf_config.vocab_size:
            raise ValueError("bootstrap_token_id is outside the model vocabulary")

        self.config = config
        self.topology = topology
        self.engine_id = topology.engine_id
        self.bootstrap_token_id = bootstrap_token_id
        self._scheduler = Scheduler(
            config,
            attention_dp_override=1,
            engine_id_override=f"{config.engine_id or 'hierarchical'}:dp{self.engine_id}",
        )
        self._state_manager = self._scheduler.worker_state[0]
        self._capacity_probe: Scheduler | None = None
        self._capacity_probe_state = None
        self._capacity_probe_num_blocks = 0
        self._capacity_validation_cache: dict[
            tuple[int, int, int], str | None
        ] = {}
        # Keep only live requests in the hot record table. Terminal request
        # IDs retain duplicate/abort semantics through compact tombstones
        # without keeping full Sequence objects in every subsequent scan.
        self._records: dict[int, LocalRequestRecord] = {}
        self._terminal_states: dict[int, RequestState] = {}
        self._terminal_events: list[FinishEvent] = []
        self._first_token_events: list[FirstTokenEvent] = []
        self._inflight_ids: set[int] = set()
        self._all_dummy_engine_quantums = 0
        self._useful_decode_tokens = 0
        self._raw_token_slots = 0
        self._control_dummy_slots = 0
        self._total_rank_forwards = 0
        self._all_dummy_rank_forwards = 0
        self._preemption_count = 0
        self._last_master_batch_sizes = [
            0 for _ in range(self.topology.attention_sp)
        ]
        self._master_assignments = [
            0 for _ in range(self.topology.attention_sp)
        ]
        self._mastered_decode_tokens = [
            0 for _ in range(self.topology.attention_sp)
        ]
        self._last_itl_token_slots = 0

    @property
    def cpp_scheduler(self) -> Scheduler:
        return self._scheduler

    @property
    def state_manager(self):
        return self._state_manager

    @property
    def useful_decode_tokens(self) -> int:
        return self._useful_decode_tokens

    @property
    def last_itl_token_slots(self) -> int:
        return self._last_itl_token_slots

    def _active_request_count(self) -> int:
        return len(self._records)

    def _active_load_state(self) -> _ActiveLoadState:
        attention_sp = self.topology.attention_sp
        master_counts = [0] * attention_sp
        receiver_counts = [0] * attention_sp
        dispatched_tokens = [0] * attention_sp
        waiting = 0
        running = 0
        for record in self._records.values():
            if record.state == RequestState.WAITING_ADMISSION:
                waiting += 1
                continue
            if record.state not in {
                RequestState.RUNNING_DECODE,
                RequestState.ABORT_PENDING,
            }:
                raise RuntimeError(
                    "LocalScheduler live record has invalid state: "
                    f"request_id={record.sequence.seq_id}, "
                    f"state={record.state.value}"
                )
            running += 1
            block_ctx = record.sequence.block_ctx(BlockContextSlot.ACTIVE)
            master_sp_idx = block_ctx.master_sp_idx
            if not 0 <= master_sp_idx < attention_sp:
                raise RuntimeError(
                    "LocalScheduler live request has invalid master rank: "
                    f"request_id={record.sequence.seq_id}, "
                    f"sp_idx={master_sp_idx}"
                )
            master_counts[master_sp_idx] += 1
            for sp_idx, token_count in enumerate(
                block_ctx.num_dispatched_tokens
            ):
                dispatched_tokens[sp_idx] += token_count
                if token_count > 0 and sp_idx != master_sp_idx:
                    receiver_counts[sp_idx] += 1
        return _ActiveLoadState(
            waiting=waiting,
            running=running,
            master_counts=tuple(master_counts),
            receiver_counts=tuple(receiver_counts),
            dispatched_tokens=tuple(dispatched_tokens),
        )

    def _ensure_capacity_probe(
        self, *, prompt_len: int, total_capacity_len: int
    ) -> None:
        block_size = self.config.kvcache_block_size
        control_blocks = max(
            self._state_manager.num_control_dummy_blocks(sp_idx)
            for sp_idx in self._state_manager.block_manager
        )
        reservation_blocks = ceil(self.config.reserved_blocks_per_req)
        lifetime_blocks = (total_capacity_len + block_size - 1) // block_size
        admission_blocks = (
            (prompt_len + block_size - 1) // block_size
        ) + reservation_blocks
        required_num_blocks = control_blocks + max(
            lifetime_blocks, admission_blocks
        )
        probe_num_blocks = min(
            self.config.num_kvcache_blocks, required_num_blocks
        )
        if (
            self._capacity_probe is not None
            and self._capacity_probe_num_blocks >= probe_num_blocks
        ):
            return

        probe_config = copy(self.config)
        probe_config.num_kvcache_blocks = probe_num_blocks
        self._capacity_probe_state = None
        self._capacity_probe = None
        self._capacity_probe = Scheduler(
            probe_config,
            attention_dp_override=1,
            engine_id_override=(
                f"{self.config.engine_id or 'hierarchical'}:"
                f"dp{self.engine_id}:capacity-probe"
            ),
        )
        self._capacity_probe_state = self._capacity_probe.worker_state[0]
        self._capacity_probe_num_blocks = probe_num_blocks

    def _validate_exclusive_lifetime(
        self,
        *,
        prompt_len: int,
        max_tokens: int,
        total_capacity_len: int,
        padded_completion_len: int,
    ) -> None:
        block_size = self.config.kvcache_block_size
        service_blocks = [
            block_manager.num_blocks
            - self._state_manager.num_control_dummy_blocks(sp_idx)
            for sp_idx, block_manager in self._state_manager.block_manager.items()
        ]
        if not service_blocks or min(service_blocks) <= 0:
            raise ValueError("control dummy reservation leaves no service KV blocks")

        master_only_tokens = 1 + padded_completion_len
        master_only_blocks = (
            master_only_tokens + block_size - 1
        ) // block_size
        if master_only_blocks > max(service_blocks):
            raise ValueError(
                "request padded decode lifetime cannot fit on any master SP rank"
            )

        minimum_total_blocks = (
            total_capacity_len + block_size - 1
        ) // block_size
        if minimum_total_blocks > sum(service_blocks):
            raise ValueError(
                "request padded lifetime exceeds the LocalEngine KV capacity"
            )

        cache_key = (prompt_len, max_tokens, padded_completion_len)
        if cache_key in self._capacity_validation_cache:
            cached_reason = self._capacity_validation_cache[cache_key]
            if cached_reason is not None:
                raise ValueError(cached_reason)
            return

        self._ensure_capacity_probe(
            prompt_len=prompt_len,
            total_capacity_len=total_capacity_len,
        )
        if self._capacity_probe is None or self._capacity_probe_state is None:
            raise RuntimeError("exclusive-capacity probe was not initialized")

        probe = Sequence(
            [0] * prompt_len,
            sampling_params=SamplingParams(
                temperature=1.0,
                max_tokens=max_tokens,
                ignore_eos=True,
            ),
        )
        if self._capacity_probe.running(0):
            raise RuntimeError("exclusive-capacity probe retained running state")
        if self._capacity_probe.waiting_migration:
            raise RuntimeError("exclusive-capacity probe retained waiting state")

        self._capacity_probe.add(probe)
        admitted: list[Sequence] = []
        try:
            admitted = list(self._capacity_probe.admit()[0])
            fits = (
                len(admitted) == 1
                and admitted[0] is probe
                and self._capacity_probe_state.can_fit_lifetime(
                    probe, 1 + padded_completion_len
                )
            )
        finally:
            if probe in self._capacity_probe_state.running:
                self._capacity_probe_state.running.remove(probe)
                self._capacity_probe_state.deallocate(
                    probe, BlockContextSlot.ACTIVE
                )
            if probe in self._capacity_probe.waiting_migration:
                self._capacity_probe.waiting_migration.remove(probe)

        reason = None
        if not fits:
            reason = (
                "request padded lifetime cannot fit its exclusive "
                "LocalEngine SP placement"
            )
        self._capacity_validation_cache[cache_key] = reason
        if reason is not None:
            raise ValueError(reason)

    def add(self, command: AddCommand) -> AddResult:
        existing = self._records.get(command.request_id)
        if existing is not None:
            return AddResult(
                request_id=command.request_id,
                accepted=False,
                engine_id=self.engine_id,
                reason=f"duplicate request in state {existing.state.value}",
            )
        terminal_state = self._terminal_states.get(command.request_id)
        if terminal_state is not None:
            return AddResult(
                request_id=command.request_id,
                accepted=False,
                engine_id=self.engine_id,
                reason=(
                    "duplicate request in state "
                    f"{terminal_state.value}"
                ),
            )
        if self._active_request_count() >= self.config.hierarchical_queue_capacity:
            return AddResult(
                request_id=command.request_id,
                accepted=False,
                engine_id=self.engine_id,
                reason="queue_full",
            )
        if command.temperature <= 1e-10:
            return AddResult(
                request_id=command.request_id,
                accepted=False,
                engine_id=self.engine_id,
                reason="temperature must be positive",
            )

        try:
            validation = validate_add_request(
                request_id=command.request_id,
                prompt_token_ids=command.prompt_token_ids,
                max_tokens=command.max_tokens,
                ignore_eos=command.ignore_eos,
                max_model_len=self.config.max_model_len,
                vocab_size=self.config.hf_config.vocab_size,
            )
            self._validate_exclusive_lifetime(
                prompt_len=validation.original_prompt_len,
                max_tokens=command.max_tokens,
                total_capacity_len=validation.total_capacity_len,
                padded_completion_len=validation.padded_completion_len,
            )
        except ValueError as exc:
            return AddResult(
                request_id=command.request_id,
                accepted=False,
                engine_id=self.engine_id,
                reason=str(exc),
            )

        sequence = Sequence(
            list(command.prompt_token_ids),
            sampling_params=SamplingParams(
                temperature=command.temperature,
                max_tokens=command.max_tokens,
                ignore_eos=command.ignore_eos,
            ),
        )
        sequence.seq_id = command.request_id
        self._scheduler.add(sequence)
        self._records[command.request_id] = LocalRequestRecord(
            sequence=sequence,
            state=RequestState.WAITING_ADMISSION,
            original_prompt_len=validation.original_prompt_len,
            padded_completion_len=validation.padded_completion_len,
            scheduler_enqueued_at=perf_counter(),
        )
        return AddResult(
            request_id=command.request_id,
            accepted=True,
            engine_id=self.engine_id,
        )

    def _discard_waiting_request(self, request_id: int) -> None:
        record = self._records.get(request_id)
        if record is None:
            return
        if record.state != RequestState.WAITING_ADMISSION:
            raise RuntimeError(
                "cannot discard a non-waiting admission candidate: "
                f"request_id={request_id}, state={record.state.value}"
            )
        if record.sequence in self._state_manager.running:
            raise RuntimeError(
                "waiting admission candidate unexpectedly entered running: "
                f"request_id={request_id}"
            )
        self._scheduler.waiting_migration.remove(record.sequence)
        self._records.pop(request_id)

    def try_admit_batch(
        self, commands: tuple[AddCommand, ...]
    ) -> tuple[AddResult, ...]:
        """Compatibility path that plans a candidate batch locally.

        `Scheduler.admit()` invokes the same C++ SP placement planner used by
        the legacy centralized scheduler. Normal global-FIFO admission uses
        `commit_planned_batch()` so LocalEngine does not revise the LB's
        placement decision.
        """
        if not commands:
            return ()

        results = [self.add(command) for command in commands]
        candidate_ids = {
            command.request_id
            for command, result in zip(commands, results, strict=True)
            if result.accepted
        }
        if not candidate_ids:
            return tuple(results)

        admitted_ids = set(self.admit())
        for index, (command, result) in enumerate(
            zip(commands, results, strict=True)
        ):
            if not result.accepted or command.request_id in admitted_ids:
                continue
            self._discard_waiting_request(command.request_id)
            results[index] = AddResult(
                request_id=command.request_id,
                accepted=False,
                engine_id=self.engine_id,
                reason="admission_deferred",
            )
        return tuple(results)

    def try_admit(self, command: AddCommand) -> AddResult:
        return self.try_admit_batch((command,))[0]

    def _planned_admission_fits(
        self,
        sequence: Sequence,
        reservation: AdmissionReservation,
        *,
        active_master_counts: tuple[int, ...],
        active_receiver_counts: tuple[int, ...],
        batch_master_counts: list[int],
        batch_receiver_counts: list[int],
        batch_tokens: list[int],
        padded_completion_len: int,
    ) -> bool:
        attention_sp = self.topology.attention_sp
        master = reservation.master_sp_idx
        dispatched = reservation.dispatched_tokens
        if (
            reservation.engine_id != self.engine_id
            or not 0 <= master < attention_sp
            or len(dispatched) != attention_sp
            or any(token_count < 0 for token_count in dispatched)
            or sum(dispatched) != sequence.num_tokens
        ):
            return False

        if (
            active_master_counts[master]
            + batch_master_counts[master]
            + 1
            > self.config.max_num_seqs
        ):
            return False
        if (
            batch_tokens[master] + sequence.num_tokens
            >= self.config.max_num_batched_tokens
        ):
            return False

        self._state_manager.apply_planned_placement(
            sequence, master, dispatched
        )
        block_size = self.config.kvcache_block_size
        for sp_idx, token_count in enumerate(dispatched):
            if token_count <= 0 and sp_idx != master:
                continue
            if (
                self.config.fixed_sp_size == 0
                and sp_idx != master
                and token_count > 0
                and (
                    active_receiver_counts[sp_idx]
                    + batch_receiver_counts[sp_idx]
                )
                >= self.config.max_num_recv_seqs
            ):
                return False
            prefill_blocks = (
                token_count + block_size - 1
            ) // block_size
            projected_masters = (
                active_master_counts[sp_idx]
                + batch_master_counts[sp_idx]
                + (1 if sp_idx == master else 0)
            )
            reserved_blocks = ceil(
                projected_masters
                * self.config.reserved_blocks_per_req
            )
            block_manager = self._state_manager.block_manager[sp_idx]
            if (
                block_manager.num_free_blocks
                < prefill_blocks + reserved_blocks
                or not block_manager.can_allocate(sequence)
            ):
                return False
        return self._state_manager.can_fit_lifetime(
            sequence, 1 + padded_completion_len
        )

    def commit_planned_batch(
        self,
        commands: tuple[AddCommand, ...],
        reservations: tuple[AdmissionReservation, ...],
    ) -> tuple[AddResult, ...]:
        """Validate and commit LB-selected placements without replanning."""
        if len(commands) != len(reservations):
            raise ValueError(
                "planned admission command/reservation count mismatch"
            )
        active_load = self._active_load_state()
        results = [self.add(command) for command in commands]
        batch_master_counts = [0] * self.topology.attention_sp
        batch_receiver_counts = [0] * self.topology.attention_sp
        batch_tokens = [0] * self.topology.attention_sp
        admitted = []
        state_mismatch = False
        for index, (command, reservation, result) in enumerate(
            zip(commands, reservations, results, strict=True)
        ):
            if reservation.request_id != command.request_id:
                raise ValueError(
                    "planned admission request mismatch: "
                    f"command={command.request_id}, "
                    f"reservation={reservation.request_id}"
                )
            if not result.accepted:
                continue
            record = self._records[command.request_id]
            sequence = record.sequence
            waiting = self._scheduler.waiting_migration
            fits = (
                not state_mismatch
                and bool(waiting)
                and waiting[0].seq_id == sequence.seq_id
                and self._planned_admission_fits(
                    sequence,
                    reservation,
                    active_master_counts=active_load.master_counts,
                    active_receiver_counts=active_load.receiver_counts,
                    batch_master_counts=batch_master_counts,
                    batch_receiver_counts=batch_receiver_counts,
                    batch_tokens=batch_tokens,
                    padded_completion_len=record.padded_completion_len,
                )
            )
            if not fits:
                state_mismatch = True
                self._discard_waiting_request(command.request_id)
                results[index] = AddResult(
                    request_id=command.request_id,
                    accepted=False,
                    engine_id=self.engine_id,
                    reason="admission_state_mismatch",
                )
                continue

            popped = waiting.popleft()
            if popped.seq_id != sequence.seq_id:
                raise RuntimeError(
                    "planned admission lost local FIFO ownership"
                )
            self._state_manager.allocate(sequence)
            sequence.status = SequenceStatus.RUNNING
            self._state_manager.running.append(sequence)
            batch_master_counts[reservation.master_sp_idx] += 1
            for sp_idx, token_count in enumerate(
                reservation.dispatched_tokens
            ):
                if (
                    token_count > 0
                    and sp_idx != reservation.master_sp_idx
                ):
                    batch_receiver_counts[sp_idx] += 1
            batch_tokens[reservation.master_sp_idx] += sequence.num_tokens
            admitted.append(sequence)

        admitted_ids = set(self._finalize_admitted(admitted))
        for index, (command, result) in enumerate(
            zip(commands, results, strict=True)
        ):
            if not result.accepted:
                continue
            if command.request_id not in admitted_ids:
                self._discard_waiting_request(command.request_id)
                results[index] = AddResult(
                    request_id=command.request_id,
                    accepted=False,
                    engine_id=self.engine_id,
                    reason="admission_state_mismatch",
                )
        return tuple(results)

    def _defer_admission(self, sequence: Sequence) -> None:
        if sequence not in self._state_manager.running:
            raise RuntimeError("cannot defer a sequence that is not running")
        self._state_manager.running.remove(sequence)
        self._scheduler.preempt(0, sequence)

    def _finalize_admitted(
        self, admitted: list[Sequence]
    ) -> tuple[int, ...]:
        admitted_ids: list[int] = []
        for sequence in admitted:
            record = self._records[sequence.seq_id]
            if not self._state_manager.can_fit_lifetime(
                sequence, 1 + record.padded_completion_len
            ):
                self._defer_admission(sequence)
                continue
            if not self._state_manager.can_append(sequence, 1):
                self._defer_admission(sequence)
                continue
            if not self._state_manager.may_append(sequence, 1):
                raise RuntimeError(
                    "bootstrap allocation failed after successful admission: "
                    f"request_id={sequence.seq_id}"
                )
            sequence.append_token(
                self.bootstrap_token_id,
                BlockContextSlot.ACTIVE,
            )
            self._state_manager.add_running_tokens(
                sequence.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx,
                1,
            )
            sequence.num_bootstrap_tokens = 1
            sequence.num_checkpointed_tokens = sequence.num_tokens
            sequence.block_ctx(BlockContextSlot.ACTIVE).dp_idx = (
                self.topology.global_dp_idx
            )
            master_sp_idx = sequence.block_ctx(
                BlockContextSlot.ACTIVE
            ).master_sp_idx
            if not 0 <= master_sp_idx < self.topology.attention_sp:
                raise RuntimeError(
                    "admitted request has invalid master SP rank: "
                    f"request_id={sequence.seq_id}, sp_idx={master_sp_idx}"
                )
            self._master_assignments[master_sp_idx] += 1
            record.state = RequestState.RUNNING_DECODE
            admitted_ids.append(sequence.seq_id)
        return tuple(admitted_ids)

    def admit(self) -> tuple[int, ...]:
        return self._finalize_admitted(
            list(self._scheduler.admit()[0])
        )

    def _reconcile_preemptions(self) -> None:
        waiting_ids = {
            sequence.seq_id for sequence in self._scheduler.waiting_migration
        }
        running_ids = {
            sequence.seq_id for sequence in self._state_manager.running
        }
        for request_id, record in self._records.items():
            if record.state.is_terminal or record.state == RequestState.ABORT_PENDING:
                continue
            if request_id in waiting_ids:
                if record.sequence.num_bootstrap_tokens != 0:
                    raise RuntimeError(
                        "preempted request retained a bootstrap token"
                    )
                if record.state != RequestState.WAITING_ADMISSION:
                    self._preemption_count += 1
                record.state = RequestState.WAITING_ADMISSION
            elif request_id in running_ids:
                record.state = RequestState.RUNNING_DECODE

    def plan_decode(self, *, wave_id: int, quantum_id: int) -> LocalDecodeBatch:
        if self._inflight_ids:
            raise RuntimeError("cannot freeze a second batch while one is in flight")

        sequences = list(self._scheduler.plan_decode()[0])
        self._reconcile_preemptions()
        control_dummy_ids = frozenset(
            sequence.seq_id
            for sequence in sequences
            if self._state_manager.is_control_dummy(sequence)
        )
        control_dummy_object_ids = frozenset(
            id(sequence)
            for sequence in sequences
            if self._state_manager.is_control_dummy(sequence)
        )
        real_sequences = [
            sequence
            for sequence in sequences
            if id(sequence) not in control_dummy_object_ids
        ]
        self._inflight_ids = {sequence.seq_id for sequence in real_sequences}

        request_master_global_rank = {
            sequence.seq_id: self.topology.global_rank(
                sequence.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx
            )
            for sequence in real_sequences
        }
        frozen_request_order = {
            global_rank: tuple(
                sequence.seq_id
                for sequence in real_sequences
                if request_master_global_rank[sequence.seq_id] == global_rank
            )
            for global_rank in self.topology.global_ranks
        }
        self._last_master_batch_sizes = [
            len(
                frozen_request_order[
                    self.topology.global_rank(sp_idx=sp_idx)
                ]
            )
            for sp_idx in range(self.topology.attention_sp)
        ]
        per_rank_sequences = {
            global_rank: list(sequences)
            for global_rank in self.topology.global_ranks
        }
        return LocalDecodeBatch(
            wave_id=wave_id,
            quantum_id=quantum_id,
            engine_id=self.engine_id,
            engine_has_real=bool(real_sequences),
            per_rank_sequences=per_rank_sequences,
            request_master_global_rank=request_master_global_rank,
            frozen_request_order=frozen_request_order,
            control_dummy_ids=control_dummy_ids,
            _all_sequences=sequences,
            _control_dummy_object_ids=control_dummy_object_ids,
        )

    def mark_first_forward_started(
        self, batch: LocalDecodeBatch
    ) -> tuple[FirstScheduleEvent, ...]:
        """Mark requests immediately before their first executor.run call."""
        if batch.engine_id != self.engine_id:
            raise RuntimeError(
                "cannot start a batch owned by another LocalScheduler"
            )
        batch_request_ids = {
            request_id
            for request_ids in batch.frozen_request_order.values()
            for request_id in request_ids
        }
        if batch_request_ids != self._inflight_ids:
            raise RuntimeError(
                "first-forward batch does not match frozen inflight requests"
            )

        started_at = perf_counter()
        events: list[FirstScheduleEvent] = []
        for request_id in sorted(batch_request_ids):
            record = self._records[request_id]
            if record.first_forward_started_at is not None:
                continue
            record.first_forward_started_at = started_at
            events.append(
                FirstScheduleEvent(
                    request_id=request_id,
                    engine_id=self.engine_id,
                    local_scheduler_queue_ms=max(
                        0.0,
                        (started_at - record.scheduler_enqueued_at) * 1000,
                    ),
                )
            )
        return tuple(events)

    def _finish_aborted(self, request_id: int) -> None:
        record = self._records[request_id]
        sequence = record.sequence
        if sequence in self._state_manager.running:
            self._state_manager.running.remove(sequence)
            self._state_manager.deallocate(sequence, BlockContextSlot.ACTIVE)
        record.state = RequestState.ABORTED
        self._emit_terminal(record, "ABORTED")

    def abort(self, request_id: int) -> AbortResult:
        record = self._records.get(request_id)
        if record is None:
            if request_id in self._terminal_states:
                return AbortResult(
                    request_id=request_id, status="already_terminal"
                )
            return AbortResult(request_id=request_id, status="not_found")
        if request_id in self._inflight_ids:
            record.state = RequestState.ABORT_PENDING
            return AbortResult(request_id=request_id, status="abort_pending")
        if record.state == RequestState.WAITING_ADMISSION:
            self._scheduler.waiting_migration.remove(record.sequence)
            record.state = RequestState.ABORTED
            self._emit_terminal(record, "ABORTED")
            return AbortResult(request_id=request_id, status="aborted")

        self._finish_aborted(request_id)
        return AbortResult(request_id=request_id, status="aborted")

    def _emit_terminal(
        self,
        record: LocalRequestRecord,
        status: str,
        *,
        final_quantum_execute_ms: float | None = None,
    ) -> None:
        if record.terminal_emitted:
            raise RuntimeError(
                f"duplicate terminal event for request {record.sequence.seq_id}"
            )
        if not record.state.is_terminal:
            raise RuntimeError(
                "cannot emit a terminal event for a live request: "
                f"request_id={record.sequence.seq_id}, "
                f"state={record.state.value}"
            )
        request_id = record.sequence.seq_id
        if self._records.get(request_id) is not record:
            raise RuntimeError(
                "terminal request is missing from LocalScheduler live records: "
                f"request_id={request_id}"
            )
        first_forward_to_terminal_ms = None
        if record.first_forward_started_at is not None:
            first_forward_to_terminal_ms = max(
                0.0,
                (
                    perf_counter() - record.first_forward_started_at
                )
                * 1000,
            )
        elif status == "FINISHED":
            raise RuntimeError(
                "finished hierarchical request has no first-forward "
                f"timestamp: request_id={record.sequence.seq_id}"
            )
        record.terminal_emitted = True
        self._terminal_events.append(
            FinishEvent(
                request_id=request_id,
                generated_count=record.sequence.num_completed_tokens,
                status=status,
                engine_id=self.engine_id,
                first_forward_to_terminal_ms=(
                    first_forward_to_terminal_ms
                ),
                final_quantum_execute_ms=final_quantum_execute_ms,
            )
        )
        self._records.pop(request_id)
        self._terminal_states[request_id] = record.state

    def postprocess(
        self,
        batch: LocalDecodeBatch,
        worker_results: list[WorkerDecodeResult],
        *,
        execute_latency_ms: float | None = None,
    ) -> tuple[FinishEvent, ...]:
        if batch.engine_id != self.engine_id:
            raise ValueError("decode batch belongs to a different LocalScheduler")
        if execute_latency_ms is not None and execute_latency_ms < 0:
            raise ValueError("execute_latency_ms must be non-negative")
        results_by_rank = batch.validate_worker_results(worker_results)
        self._raw_token_slots += (
            len(batch._all_sequences) * HIERARCHICAL_LOOP_COUNT
        )
        self._control_dummy_slots += (
            len(batch._control_dummy_object_ids) * HIERARCHICAL_LOOP_COUNT
        )
        rank_forwards = self.topology.world_size * HIERARCHICAL_LOOP_COUNT
        self._total_rank_forwards += rank_forwards
        if not batch.engine_has_real:
            self._all_dummy_engine_quantums += 1
            self._all_dummy_rank_forwards += rank_forwards

        aborted_ids = {
            request_id
            for request_id in self._inflight_ids
            if self._records[request_id].state == RequestState.ABORT_PENDING
        }
        completed_before = {
            request_id: self._records[request_id].sequence.num_completed_tokens
            for request_id in self._inflight_ids.difference(aborted_ids)
        }
        master_sp_by_request = {
            request_id: (
                self._records[request_id]
                .sequence.block_ctx(BlockContextSlot.ACTIVE)
                .master_sp_idx
            )
            for request_id in completed_before
        }
        self._last_itl_token_slots = 0
        for request_id in sorted(aborted_ids):
            self._finish_aborted(request_id)

        dp_sp_sequences: list[list[Sequence]] = []
        dp_sp_token_ids: list[list[list[int]]] = []
        for sp_idx in range(self.topology.attention_sp):
            global_rank = self.topology.global_rank(sp_idx)
            result = results_by_rank[global_rank]
            token_by_request = dict(
                zip(
                    result.mastered_request_ids,
                    result.sampled_token_ids,
                    strict=True,
                )
            )
            rank_sequences: list[Sequence] = []
            rank_token_ids: list[list[int]] = []
            for sequence in batch._all_sequences:
                if (
                    sequence.block_ctx(BlockContextSlot.ACTIVE).master_sp_idx
                    != sp_idx
                ):
                    continue
                if (
                    sequence.seq_id in aborted_ids
                    and not batch.is_control_dummy(sequence)
                ):
                    continue
                rank_sequences.append(sequence)
                if batch.is_control_dummy(sequence):
                    rank_token_ids.append([0] * HIERARCHICAL_LOOP_COUNT)
                else:
                    rank_token_ids.append(
                        list(token_by_request[sequence.seq_id])
                    )
            dp_sp_sequences.append(rank_sequences)
            dp_sp_token_ids.append(rank_token_ids)

        self._scheduler.postprocess(
            dp_sp_sequences,
            dp_sp_token_ids,
            metrics_manager=None,
            loop_count=HIERARCHICAL_LOOP_COUNT,
        )
        for request_id, previous_tokens in completed_before.items():
            completed_tokens = self._records[
                request_id
            ].sequence.num_completed_tokens
            generated_tokens = completed_tokens - previous_tokens
            if generated_tokens < 0:
                raise RuntimeError(
                    "hierarchical completed-token counter moved backwards: "
                    f"request_id={request_id}"
                )
            master_sp_idx = master_sp_by_request[request_id]
            if not 0 <= master_sp_idx < self.topology.attention_sp:
                raise RuntimeError(
                    "decoded request has invalid master SP rank: "
                    f"request_id={request_id}, sp_idx={master_sp_idx}"
                )
            self._useful_decode_tokens += generated_tokens
            self._mastered_decode_tokens[master_sp_idx] += generated_tokens
            self._last_itl_token_slots += (
                generated_tokens
                if previous_tokens > 0
                else max(0, generated_tokens - 1)
            )
        for request_id in sorted(self._inflight_ids.difference(aborted_ids)):
            record = self._records[request_id]
            if (
                not record.first_token_emitted
                and record.sequence.num_completed_tokens > 0
            ):
                record.first_token_emitted = True
                self._first_token_events.append(
                    FirstTokenEvent(
                        request_id=request_id,
                        engine_id=self.engine_id,
                        generated_count=(
                            record.sequence.num_completed_tokens
                        ),
                    )
                )
            if record.sequence.is_finished:
                record.state = RequestState.FINISHED
                self._emit_terminal(
                    record,
                    "FINISHED",
                    final_quantum_execute_ms=execute_latency_ms,
                )
            else:
                record.state = RequestState.RUNNING_DECODE
        self._inflight_ids.clear()
        return self.drain_terminal_events()

    def drain_terminal_events(self) -> tuple[FinishEvent, ...]:
        events = tuple(self._terminal_events)
        self._terminal_events.clear()
        return events

    def drain_first_token_events(self) -> tuple[FirstTokenEvent, ...]:
        events = tuple(self._first_token_events)
        self._first_token_events.clear()
        return events

    def is_finished(self) -> bool:
        return not self._records

    def load_snapshot(self, *, wave_id: int, quantum_id: int) -> LoadSnapshot:
        free_blocks = [
            self._state_manager.block_manager[sp_idx].num_free_blocks
            for sp_idx in range(self.topology.attention_sp)
        ]
        active_load = self._active_load_state()
        rank_loads = tuple(
            RankLoad(
                global_rank=self.topology.global_rank(sp_idx=sp_idx),
                sp_idx=sp_idx,
                tp_idx=0,
                master_batch_size=self._last_master_batch_sizes[sp_idx],
                active_master_requests=(
                    active_load.master_counts[sp_idx]
                ),
                free_blocks=free_blocks[sp_idx],
                total_blocks=self._state_manager.block_manager[
                    sp_idx
                ].num_blocks,
                master_assignments=self._master_assignments[sp_idx],
                mastered_decode_tokens=self._mastered_decode_tokens[sp_idx],
                active_receiver_requests=(
                    active_load.receiver_counts[sp_idx]
                ),
                active_dispatched_tokens=(
                    active_load.dispatched_tokens[sp_idx]
                ),
                control_dummy_blocks=(
                    self._state_manager.num_control_dummy_blocks(sp_idx)
                ),
            )
            for sp_idx in range(self.topology.attention_sp)
        )
        return LoadSnapshot(
            engine_id=self.engine_id,
            ready=True,
            waiting=active_load.waiting,
            running=active_load.running,
            free_blocks_min=min(free_blocks, default=0),
            wave_id=wave_id,
            quantum_id=quantum_id,
            useful_real_batch_size=len(self._inflight_ids),
            control_dummy_count=len(self._state_manager.dummy_seqs),
            all_dummy_engine_quantums=self._all_dummy_engine_quantums,
            useful_decode_tokens=self._useful_decode_tokens,
            raw_token_slots=self._raw_token_slots,
            control_dummy_slots=self._control_dummy_slots,
            total_rank_forwards=self._total_rank_forwards,
            all_dummy_rank_forwards=self._all_dummy_rank_forwards,
            preemption_count=self._preemption_count,
            rank_loads=rank_loads,
        )
