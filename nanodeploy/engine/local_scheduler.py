from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from time import perf_counter

from nanodeploy._cpp import BlockContextSlot, Sequence
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
from nanodeploy.engine.sequence import SequenceStatus
from nanodeploy.engine.topology import EngineTopology
from nanodeploy.router.admission_planner import (
    AdmissionPlanner,
    AdmissionPlannerConfig,
    AdmissionShadow,
)


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
        self._capacity_planner = AdmissionPlanner(
            AdmissionPlannerConfig.from_config(config)
        )
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
        self._sp_idx_by_global_rank = {
            global_rank: sp_idx
            for sp_idx, global_rank in enumerate(self.topology.global_ranks)
        }
        self._active_load_cache: _ActiveLoadState | None = _ActiveLoadState(
            waiting=0,
            running=0,
            master_counts=(0,) * self.topology.attention_sp,
            receiver_counts=(0,) * self.topology.attention_sp,
            dispatched_tokens=(0,) * self.topology.attention_sp,
        )

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

    def _invalidate_active_load_state(self) -> None:
        self._active_load_cache = None

    def _active_load_state(self) -> _ActiveLoadState:
        cached = self._active_load_cache
        if cached is not None:
            return cached
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
        active_load = _ActiveLoadState(
            waiting=waiting,
            running=running,
            master_counts=tuple(master_counts),
            receiver_counts=tuple(receiver_counts),
            dispatched_tokens=tuple(dispatched_tokens),
        )
        self._active_load_cache = active_load
        return active_load

    def _active_load_with_added_running(
        self,
        active_load: _ActiveLoadState,
        sequences: tuple[Sequence, ...],
    ) -> _ActiveLoadState:
        master_counts = list(active_load.master_counts)
        receiver_counts = list(active_load.receiver_counts)
        dispatched_tokens = list(active_load.dispatched_tokens)
        for sequence in sequences:
            block_ctx = sequence.block_ctx(BlockContextSlot.ACTIVE)
            master_sp_idx = block_ctx.master_sp_idx
            if not 0 <= master_sp_idx < self.topology.attention_sp:
                raise RuntimeError(
                    "admitted request has invalid master SP rank: "
                    f"request_id={sequence.seq_id}, sp_idx={master_sp_idx}"
                )
            master_counts[master_sp_idx] += 1
            for sp_idx, token_count in enumerate(
                block_ctx.num_dispatched_tokens
            ):
                dispatched_tokens[sp_idx] += token_count
                if token_count > 0 and sp_idx != master_sp_idx:
                    receiver_counts[sp_idx] += 1
        return _ActiveLoadState(
            waiting=active_load.waiting,
            running=active_load.running + len(sequences),
            master_counts=tuple(master_counts),
            receiver_counts=tuple(receiver_counts),
            dispatched_tokens=tuple(dispatched_tokens),
        )

    def _validate_exclusive_capacity(self, command: AddCommand) -> None:
        total_blocks = []
        control_dummy_blocks = []
        service_blocks = []
        for sp_idx in range(self.topology.attention_sp):
            block_manager = self._state_manager.block_manager[sp_idx]
            control_blocks = self._state_manager.num_control_dummy_blocks(
                sp_idx
            )
            total_blocks.append(block_manager.num_blocks)
            control_dummy_blocks.append(control_blocks)
            service_blocks.append(block_manager.num_blocks - control_blocks)
        if not service_blocks or min(service_blocks) <= 0:
            raise ValueError("control dummy reservation leaves no service KV blocks")

        attention_sp = self.topology.attention_sp
        empty_shadow = AdmissionShadow(
            engine_id=self.engine_id,
            free_blocks=list(service_blocks),
            total_blocks=total_blocks,
            control_dummy_blocks=control_dummy_blocks,
            master_counts=[0] * attention_sp,
            receiver_counts=[0] * attention_sp,
            dispatched_tokens=[0] * attention_sp,
            batch_master_counts=[0] * attention_sp,
            batch_tokens=[0] * attention_sp,
            queue_slots=0,
            rr_cursor=0,
        )
        if self._capacity_planner.plan(empty_shadow, command) is None:
            raise ValueError(
                "request padded lifetime cannot fit its exclusive "
                "LocalEngine SP placement"
            )

    def add(self, command: AddCommand, sequence: Sequence) -> AddResult:
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
                prompt_len=command.prompt_len,
                max_tokens=command.max_tokens,
                ignore_eos=command.ignore_eos,
                max_model_len=self.config.max_model_len,
                vocab_size=self.config.hf_config.vocab_size,
            )
            if sequence.seq_id != command.request_id:
                raise ValueError("Sequence request_id does not match ADD metadata")
            if sequence.num_prompt_tokens != command.prompt_len:
                raise ValueError("Sequence prompt length does not match ADD metadata")
            if sequence.num_tokens != command.num_tokens:
                raise ValueError("Sequence token count does not match ADD metadata")
            if sequence.materialized_token_count != command.num_tokens:
                raise ValueError("Sequence payload does not contain all token ids")
            if sequence.max_tokens != command.max_tokens:
                raise ValueError("Sequence max_tokens does not match ADD metadata")
            if sequence.temperature != command.temperature:
                raise ValueError("Sequence temperature does not match ADD metadata")
            if sequence.ignore_eos != command.ignore_eos:
                raise ValueError("Sequence ignore_eos does not match ADD metadata")
            invalid_token = sequence.first_invalid_prompt_token(
                self.config.hf_config.vocab_size
            )
            if invalid_token is not None:
                raise ValueError(
                    f"prompt token id {invalid_token} is outside "
                    f"[0, {self.config.hf_config.vocab_size})"
                )
            self._validate_exclusive_capacity(command)
        except ValueError as exc:
            return AddResult(
                request_id=command.request_id,
                accepted=False,
                engine_id=self.engine_id,
                reason=str(exc),
            )

        self._scheduler.add(sequence)
        self._records[command.request_id] = LocalRequestRecord(
            sequence=sequence,
            state=RequestState.WAITING_ADMISSION,
            original_prompt_len=validation.original_prompt_len,
            padded_completion_len=validation.padded_completion_len,
            scheduler_enqueued_at=perf_counter(),
        )
        self._invalidate_active_load_state()
        return AddResult(
            request_id=command.request_id,
            accepted=True,
            engine_id=self.engine_id,
        )

    def stage_batch(
        self,
        commands: tuple[AddCommand, ...],
        sequences: tuple[Sequence, ...],
    ) -> tuple[AddResult, ...]:
        """Validate and enqueue a batch before the scheduler timer starts.

        The centralized profiler constructs its waiting queue before timing
        ``Scheduler.schedule()``.  This helper gives the hierarchical
        profiler the same boundary while retaining the normal per-request
        validation and duplicate semantics.
        """
        if len(commands) != len(sequences):
            raise ValueError("admission command/Sequence count mismatch")
        return tuple(
            self.add(command, sequence)
            for command, sequence in zip(commands, sequences, strict=True)
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
        self._invalidate_active_load_state()

    def try_admit_batch(
        self,
        commands: tuple[AddCommand, ...],
        sequences: tuple[Sequence, ...],
    ) -> tuple[AddResult, ...]:
        """Compatibility path that plans a candidate batch locally.

        `Scheduler.admit()` invokes the same C++ SP placement planner used by
        the legacy centralized scheduler. Normal global-FIFO admission uses
        `commit_planned_batch()` so LocalEngine does not revise the LB's
        placement decision.
        """
        if len(commands) != len(sequences):
            raise ValueError("admission command/Sequence count mismatch")
        if not commands:
            return ()

        results = [
            self.add(command, sequence)
            for command, sequence in zip(commands, sequences, strict=True)
        ]
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

    def try_admit(
        self, command: AddCommand, sequence: Sequence
    ) -> AddResult:
        return self.try_admit_batch((command,), (sequence,))[0]

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
        batch_prefill_blocks: list[int],
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
            # The native bulk commit happens after this validation loop, so
            # account for blocks claimed by earlier requests in this same
            # batch explicitly.  Previously Python allocation happened
            # inline and the block manager reflected this automatically.
            free_blocks_after_batch = (
                block_manager.num_free_blocks - batch_prefill_blocks[sp_idx]
            )
            if (
                free_blocks_after_batch
                < prefill_blocks + reserved_blocks
                or free_blocks_after_batch
                < sequence.num_blocks(BlockContextSlot.ACTIVE, sp_idx)
            ):
                return False
        return self._state_manager.can_fit_lifetime(
            sequence, 1 + padded_completion_len
        )

    def commit_planned_batch(
        self,
        commands: tuple[AddCommand, ...],
        reservations: tuple[AdmissionReservation, ...],
        sequences: tuple[Sequence, ...],
        *,
        pre_staged: bool = False,
    ) -> tuple[AddResult, ...]:
        """Validate and commit LB-selected placements without replanning."""
        if not len(commands) == len(reservations) == len(sequences):
            raise ValueError(
                "planned admission command/reservation/Sequence count mismatch"
            )
        active_load = self._active_load_state()
        if pre_staged:
            results = []
            for command, sequence in zip(commands, sequences, strict=True):
                record = self._records.get(command.request_id)
                if (
                    record is None
                    or record.sequence is not sequence
                    or record.state != RequestState.WAITING_ADMISSION
                ):
                    results.append(
                        AddResult(
                            request_id=command.request_id,
                            accepted=False,
                            engine_id=self.engine_id,
                            reason="staged_request_missing",
                        )
                    )
                else:
                    results.append(
                        AddResult(
                            request_id=command.request_id,
                            accepted=True,
                            engine_id=self.engine_id,
                        )
                    )
        else:
            results = [
                self.add(command, sequence)
                for command, sequence in zip(commands, sequences, strict=True)
            ]
        batch_master_counts = [0] * self.topology.attention_sp
        batch_receiver_counts = [0] * self.topology.attention_sp
        batch_tokens = [0] * self.topology.attention_sp
        batch_prefill_blocks = [0] * self.topology.attention_sp
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
                    batch_prefill_blocks=batch_prefill_blocks,
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
            batch_master_counts[reservation.master_sp_idx] += 1
            for sp_idx, token_count in enumerate(
                reservation.dispatched_tokens
            ):
                if (
                    token_count > 0
                    and sp_idx != reservation.master_sp_idx
                ):
                    batch_receiver_counts[sp_idx] += 1
                batch_prefill_blocks[sp_idx] += (
                    sequence.num_blocks(BlockContextSlot.ACTIVE, sp_idx)
                )
            batch_tokens[reservation.master_sp_idx] += sequence.num_tokens
            admitted.append(sequence)

        # Placement checks above intentionally remain in Python for contract
        # diagnostics.  The state transition itself is a native bulk call so
        # queue mutation and KV allocation do not pay one Python round-trip per
        # request.  Keep a compatibility fallback for an extension that was
        # built before the bulk entry point was added; a rebuilt extension
        # always takes the native branch.
        native_commit = getattr(
            self._scheduler, "commit_planned_sequences", None
        )
        if native_commit is None:
            for sequence in admitted:
                self._state_manager.allocate(sequence)
                sequence.status = SequenceStatus.RUNNING
                self._state_manager.running.append(sequence)
        else:
            native_commit(admitted)
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
        admitted_sequences = tuple(
            self._records[request_id].sequence
            for request_id in admitted_ids
        )
        self._active_load_cache = self._active_load_with_added_running(
            active_load,
            admitted_sequences,
        )
        return tuple(results)

    def _defer_admission(self, sequence: Sequence) -> None:
        if sequence not in self._state_manager.running:
            raise RuntimeError("cannot defer a sequence that is not running")
        self._state_manager.running.remove(sequence)
        self._scheduler.preempt(0, sequence)
        self._invalidate_active_load_state()

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
            self._invalidate_active_load_state()
            admitted_ids.append(sequence.seq_id)
        return tuple(admitted_ids)

    def admit(self) -> tuple[int, ...]:
        admitted = list(self._scheduler.admit()[0])
        self._invalidate_active_load_state()
        return self._finalize_admitted(admitted)

    def _reconcile_preemptions(self) -> tuple[int, int]:
        waiting_ids = {
            sequence.seq_id for sequence in self._scheduler.waiting_migration
        }
        running_ids = {
            sequence.seq_id for sequence in self._state_manager.running
        }
        waiting = 0
        running = 0
        for request_id, record in self._records.items():
            if not record.state.is_terminal and record.state != (
                RequestState.ABORT_PENDING
            ):
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
            if record.state == RequestState.WAITING_ADMISSION:
                waiting += 1
            elif record.state in {
                RequestState.RUNNING_DECODE,
                RequestState.ABORT_PENDING,
            }:
                running += 1
            else:
                raise RuntimeError(
                    "LocalScheduler live record has invalid state: "
                    f"request_id={record.sequence.seq_id}, "
                    f"state={record.state.value}"
                )
        return waiting, running

    def plan_decode(self, *, wave_id: int, quantum_id: int) -> LocalDecodeBatch:
        """Plan one decode batch through the native C++ scheduler.

        The native scheduler call is kept separate from the Python contract
        bookkeeping so callers that need to measure the scheduler boundary can
        use the exact same C++ ``Scheduler.schedule()`` call as the
        centralized path and build the batch afterwards.
        """
        return self.build_decode_batch_from_schedule_result(
            self._scheduler.plan_decode(),
            wave_id=wave_id,
            quantum_id=quantum_id,
        )

    def build_decode_batch_from_schedule_result(
        self,
        schedule_result: object,
        *,
        wave_id: int,
        quantum_id: int,
    ) -> LocalDecodeBatch:
        """Build contract metadata from an already-computed native result.

        ``schedule_result`` must be the result of this LocalScheduler's
        underlying native scheduler.  Keeping this conversion outside the
        native timing interval avoids charging Python-only batch bookkeeping
        to the scheduler algorithm metric.
        """
        if self._inflight_ids:
            raise RuntimeError("cannot freeze a second batch while one is in flight")

        self._invalidate_active_load_state()
        # ``Scheduler.schedule()`` returns a ``ScheduleResult`` while the
        # legacy ``plan_decode()`` binding returns the raw DP-sequence vector.
        # Accept both representations so the public helper remains backward
        # compatible with existing contract callers.
        dp_seqs = getattr(schedule_result, "dp_seqs", schedule_result)
        if len(dp_seqs) != 1:
            raise RuntimeError(
                "LocalScheduler native result must contain exactly one DP batch"
            )
        sequences = list(dp_seqs[0])
        waiting_count, running_count = self._reconcile_preemptions()
        attention_sp = self.topology.attention_sp
        master_counts = [0] * attention_sp
        receiver_counts = [0] * attention_sp
        dispatched_tokens = [0] * attention_sp
        mastered_sequence_lists = {
            global_rank: [] for global_rank in self.topology.global_ranks
        }
        real_row_index_lists = {
            global_rank: [] for global_rank in self.topology.global_ranks
        }
        request_order_lists = {
            global_rank: [] for global_rank in self.topology.global_ranks
        }
        control_dummy_ids: set[int] = set()
        control_dummy_object_ids: set[int] = set()
        real_sequences: list[Sequence] = []
        request_master_global_rank: dict[int, int] = {}
        for sequence in sequences:
            block_ctx = sequence.block_ctx(BlockContextSlot.ACTIVE)
            master_sp_idx = block_ctx.master_sp_idx
            global_rank = self.topology.global_rank(master_sp_idx)
            mastered_sequences = mastered_sequence_lists[global_rank]
            row_index = len(mastered_sequences)
            mastered_sequences.append(sequence)
            if self._state_manager.is_control_dummy(sequence):
                control_dummy_ids.add(sequence.seq_id)
                control_dummy_object_ids.add(id(sequence))
                continue
            real_sequences.append(sequence)
            master_counts[master_sp_idx] += 1
            for sp_idx, token_count in enumerate(
                block_ctx.num_dispatched_tokens
            ):
                dispatched_tokens[sp_idx] += token_count
                if token_count > 0 and sp_idx != master_sp_idx:
                    receiver_counts[sp_idx] += 1
            request_master_global_rank[sequence.seq_id] = global_rank
            request_order_lists[global_rank].append(sequence.seq_id)
            real_row_index_lists[global_rank].append(row_index)

        frozen_mastered_sequences = {
            global_rank: tuple(mastered_sequence_lists[global_rank])
            for global_rank in self.topology.global_ranks
        }
        frozen_real_row_indices = {
            global_rank: tuple(real_row_index_lists[global_rank])
            for global_rank in self.topology.global_ranks
        }
        frozen_request_order = {
            global_rank: tuple(request_order_lists[global_rank])
            for global_rank in self.topology.global_ranks
        }
        self._inflight_ids = {sequence.seq_id for sequence in real_sequences}
        self._last_master_batch_sizes = [
            len(frozen_request_order[global_rank])
            for global_rank in self.topology.global_ranks
        ]
        if running_count != len(real_sequences):
            raise RuntimeError(
                "planned decode batch does not cover all running requests: "
                f"running={running_count}, planned={len(real_sequences)}"
            )
        active_load = _ActiveLoadState(
            waiting=waiting_count,
            running=running_count,
            master_counts=tuple(master_counts),
            receiver_counts=tuple(receiver_counts),
            dispatched_tokens=tuple(dispatched_tokens),
        )
        self._active_load_cache = active_load
        frozen_load_snapshot = self._load_snapshot_from_active_state(
            wave_id=wave_id,
            quantum_id=quantum_id,
            active_load=active_load,
        )
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
            frozen_mastered_sequences=frozen_mastered_sequences,
            frozen_real_row_indices=frozen_real_row_indices,
            request_master_global_rank=request_master_global_rank,
            frozen_request_order=frozen_request_order,
            control_dummy_ids=frozenset(control_dummy_ids),
            frozen_load_snapshot=frozen_load_snapshot,
            _all_sequences=sequences,
            _control_dummy_object_ids=frozenset(
                control_dummy_object_ids
            ),
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
            self._invalidate_active_load_state()
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
        self._invalidate_active_load_state()

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
        surviving_ids = tuple(
            sorted(self._inflight_ids.difference(aborted_ids))
        )
        completed_before = {
            request_id: self._records[request_id].sequence.num_completed_tokens
            for request_id in surviving_ids
        }
        self._last_itl_token_slots = 0
        for request_id in sorted(aborted_ids):
            self._finish_aborted(request_id)

        dp_sp_sequences: list[list[Sequence]] = []
        dp_sp_token_ids: list[list[list[int]]] = []
        for sp_idx in range(self.topology.attention_sp):
            global_rank = self.topology.global_rank(sp_idx)
            result = results_by_rank[global_rank]
            mastered_sequences = batch.frozen_mastered_sequences[
                global_rank
            ]
            real_row_indices = batch.frozen_real_row_indices[global_rank]
            if len(real_row_indices) != len(result.sampled_token_ids):
                raise RuntimeError(
                    f"rank {global_rank} frozen real-row layout mismatch"
                )
            rank_sequences: list[Sequence] = []
            rank_token_ids: list[list[int]] = []
            real_result_index = 0
            next_real_row = (
                real_row_indices[0] if real_row_indices else None
            )
            for row_index, sequence in enumerate(mastered_sequences):
                if row_index == next_real_row:
                    tokens = result.sampled_token_ids[real_result_index]
                    real_result_index += 1
                    next_real_row = (
                        real_row_indices[real_result_index]
                        if real_result_index < len(real_row_indices)
                        else None
                    )
                    if sequence.seq_id in aborted_ids:
                        continue
                    rank_sequences.append(sequence)
                    rank_token_ids.append(list(tokens))
                    continue
                rank_sequences.append(sequence)
                rank_token_ids.append([0] * HIERARCHICAL_LOOP_COUNT)
            if real_result_index != len(result.sampled_token_ids):
                raise RuntimeError(
                    f"rank {global_rank} frozen real-row layout mismatch"
                )
            dp_sp_sequences.append(rank_sequences)
            dp_sp_token_ids.append(rank_token_ids)

        self._scheduler.postprocess(
            dp_sp_sequences,
            dp_sp_token_ids,
            metrics_manager=None,
            loop_count=HIERARCHICAL_LOOP_COUNT,
        )
        self._invalidate_active_load_state()
        for request_id in surviving_ids:
            previous_tokens = completed_before[request_id]
            completed_tokens = self._records[
                request_id
            ].sequence.num_completed_tokens
            generated_tokens = completed_tokens - previous_tokens
            if generated_tokens < 0:
                raise RuntimeError(
                    "hierarchical completed-token counter moved backwards: "
                    f"request_id={request_id}"
                )
            master_global_rank = batch.request_master_global_rank[request_id]
            master_sp_idx = self._sp_idx_by_global_rank.get(
                master_global_rank
            )
            if master_sp_idx is None:
                raise RuntimeError(
                    "decoded request has invalid master SP rank: "
                    f"request_id={request_id}, rank={master_global_rank}"
                )
            self._useful_decode_tokens += generated_tokens
            self._mastered_decode_tokens[master_sp_idx] += generated_tokens
            self._last_itl_token_slots += (
                generated_tokens
                if previous_tokens > 0
                else max(0, generated_tokens - 1)
            )
        for request_id in surviving_ids:
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

    def _load_snapshot_from_active_state(
        self,
        *,
        wave_id: int,
        quantum_id: int,
        active_load: _ActiveLoadState,
    ) -> LoadSnapshot:
        free_blocks = [
            self._state_manager.block_manager[sp_idx].num_free_blocks
            for sp_idx in range(self.topology.attention_sp)
        ]
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

    def load_snapshot(self, *, wave_id: int, quantum_id: int) -> LoadSnapshot:
        return self._load_snapshot_from_active_state(
            wave_id=wave_id,
            quantum_id=quantum_id,
            active_load=self._active_load_state(),
        )
