from __future__ import annotations

from collections import deque
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
    ResourceReleaseEvent,
    TokenCommitEvent,
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
    admission_wave_id: int
    first_forward_started_at: float | None = None
    first_token_emitted: bool = False
    terminal_emitted: bool = False
    generation_epoch: int = 0
    committed_output_count: int = 0
    terminal_reason: str | None = None
    terminal_state: RequestState | None = None
    outstanding_output_placeholders: int = 0
    last_scheduled_quantum: tuple[int, int] | None = None
    last_completed_quantum: tuple[int, int] | None = None
    drain_fence: tuple[int, int] | None = None
    resources_allocated: bool = False


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
        if config.eos < 0:
            eos_token_id = getattr(config.hf_config, "eos_token_id", None)
            if not isinstance(eos_token_id, int) or isinstance(
                eos_token_id, bool
            ):
                raise ValueError(
                    "hierarchical scheduler requires one integer EOS token id"
                )
            config.eos = eos_token_id
        if not 0 <= config.eos < config.hf_config.vocab_size:
            raise ValueError("EOS token id is outside the model vocabulary")
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
        self._token_commit_events: list[TokenCommitEvent] = []
        self._first_token_events: list[FirstTokenEvent] = []
        self._resource_release_events: list[ResourceReleaseEvent] = []
        self._inflight_ids: set[int] = set()
        self._inflight_batches: deque[LocalDecodeBatch] = deque()
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
                RequestState.TERMINAL_PENDING_DRAIN,
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
                quantum_size=self.config.loop_count,
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
            admission_wave_id=command.wave_id,
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
        sequences: tuple[Sequence, ...],
    ) -> tuple[AddResult, ...]:
        """Validate and commit LB-selected placements without replanning."""
        if not len(commands) == len(reservations) == len(sequences):
            raise ValueError(
                "planned admission command/reservation/Sequence count mismatch"
            )
        active_load = self._active_load_state()
        results = [
            self.add(command, sequence)
            for command, sequence in zip(commands, sequences, strict=True)
        ]
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
            record.resources_allocated = True
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
            if record.state.is_terminal or record.state in {
                RequestState.ABORT_PENDING,
                RequestState.TERMINAL_PENDING_DRAIN,
            }:
                continue
            if request_id in waiting_ids:
                if record.sequence.num_bootstrap_tokens != 0:
                    raise RuntimeError(
                        "preempted request retained a bootstrap token"
                    )
                if record.state != RequestState.WAITING_ADMISSION:
                    if record.committed_output_count != 0:
                        raise RuntimeError(
                            "cannot reset a request after frontend-visible "
                            "tokens were committed: "
                            f"request_id={request_id}, "
                            f"committed={record.committed_output_count}"
                        )
                    self._preemption_count += 1
                    record.generation_epoch += 1
                record.state = RequestState.WAITING_ADMISSION
            elif request_id in running_ids:
                record.state = RequestState.RUNNING_DECODE

    def _freeze_batch(
        self,
        sequences: list[Sequence],
        *,
        wave_id: int,
        quantum_id: int,
        output_offsets: dict[int, int] | None = None,
    ) -> LocalDecodeBatch:
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
        batch = LocalDecodeBatch(
            wave_id=wave_id,
            quantum_id=quantum_id,
            engine_id=self.engine_id,
            engine_has_real=bool(real_sequences),
            per_rank_sequences=per_rank_sequences,
            request_master_global_rank=request_master_global_rank,
            frozen_request_order=frozen_request_order,
            control_dummy_ids=control_dummy_ids,
            per_request_epoch={
                sequence.seq_id: self._records[
                    sequence.seq_id
                ].generation_epoch
                for sequence in real_sequences
            },
            per_request_output_offset={
                sequence.seq_id: (
                    self._records[sequence.seq_id].committed_output_count
                    if output_offsets is None
                    else output_offsets[sequence.seq_id]
                )
                for sequence in real_sequences
            },
            _all_sequences=sequences,
            _control_dummy_object_ids=control_dummy_object_ids,
        )
        self._inflight_batches.append(batch)
        for sequence in real_sequences:
            record = self._records[sequence.seq_id]
            record.outstanding_output_placeholders += 1
            record.last_scheduled_quantum = (wave_id, quantum_id)
            self._inflight_ids.add(sequence.seq_id)
        return batch

    def _reserve_snapshot_output_block(
        self,
        snapshot: Sequence,
        canonical: Sequence,
    ) -> bool:
        before_tables = tuple(
            tuple(
                snapshot.block_table(BlockContextSlot.ACTIVE, sp_idx)
            )
            for sp_idx in range(self.topology.attention_sp)
        )
        if not self._state_manager.can_append(snapshot, 1):
            return False
        if not self._state_manager.may_append(snapshot, 1):
            raise RuntimeError(
                "could not reserve optimistic decode output block: "
                f"request_id={snapshot.seq_id}"
            )
        canonical_ctx = canonical.block_ctx(BlockContextSlot.ACTIVE)
        for sp_idx, before in enumerate(before_tables):
            snapshot_table = tuple(
                snapshot.block_table(BlockContextSlot.ACTIVE, sp_idx)
            )
            canonical_table = canonical.block_table(
                BlockContextSlot.ACTIVE, sp_idx
            )
            if tuple(canonical_table) != before:
                raise RuntimeError(
                    "canonical/snapshot KV block prefix diverged before "
                    f"reservation: request_id={snapshot.seq_id}, "
                    f"sp_idx={sp_idx}"
                )
            for block_id in snapshot_table[len(before) :]:
                canonical_table.append(block_id)
                canonical_ctx.block_location.append((sp_idx, block_id))
        return True

    def _plan_lookahead(
        self,
        predecessor: LocalDecodeBatch,
        *,
        wave_id: int,
        quantum_id: int,
    ) -> LocalDecodeBatch:
        if predecessor.wave_id != wave_id:
            raise RuntimeError("decode lookahead cannot cross a wave boundary")
        if predecessor.quantum_id + 1 != quantum_id:
            raise RuntimeError(
                "decode lookahead quantum is not consecutive: "
                f"predecessor={predecessor.quantum_id}, next={quantum_id}"
            )

        snapshots: list[Sequence] = []
        output_offsets: dict[int, int] = {}
        occupied_sp: set[int] = set()
        master_counts = [0] * self.topology.attention_sp
        predecessor_request_ids: set[int] = set()
        for source in predecessor._all_sequences:
            if predecessor.is_control_dummy(source):
                continue
            request_id = source.seq_id
            predecessor_request_ids.add(request_id)
            record = self._records[request_id]
            if record.state != RequestState.RUNNING_DECODE:
                continue
            predecessor_offset = predecessor.per_request_output_offset[
                request_id
            ]
            next_offset = predecessor_offset + HIERARCHICAL_LOOP_COUNT
            if (
                next_offset
                != record.committed_output_count
                + record.outstanding_output_placeholders
            ):
                raise RuntimeError(
                    "optimistic decode offset diverged from outstanding "
                    f"placeholders: request_id={request_id}"
                )
            if next_offset >= record.sequence.max_tokens:
                continue

            snapshot = source.clone_for_decode_dispatch()
            snapshot.num_tokens += HIERARCHICAL_LOOP_COUNT
            snapshot_ctx = snapshot.block_ctx(BlockContextSlot.ACTIVE)
            master_sp_idx = snapshot_ctx.master_sp_idx
            dispatched = list(snapshot_ctx.num_dispatched_tokens)
            dispatched[master_sp_idx] += HIERARCHICAL_LOOP_COUNT
            snapshot_ctx.num_dispatched_tokens = dispatched
            if not self._reserve_snapshot_output_block(
                snapshot, record.sequence
            ):
                continue
            snapshots.append(snapshot)
            output_offsets[request_id] = next_offset
            occupied_sp.add(master_sp_idx)
            master_counts[master_sp_idx] += 1

        # Admission remains live while an older decode is executing. A newly
        # admitted request has no predecessor dependency, so it can join the
        # optimistic batch directly instead of waiting for the rolling
        # depth-two pipeline to become empty.
        for canonical in self._state_manager.running:
            request_id = canonical.seq_id
            if request_id in predecessor_request_ids:
                continue
            record = self._records[request_id]
            if (
                record.state != RequestState.RUNNING_DECODE
                or record.outstanding_output_placeholders != 0
            ):
                continue
            master_sp_idx = canonical.block_ctx(
                BlockContextSlot.ACTIVE
            ).master_sp_idx
            if master_counts[master_sp_idx] >= self.config.max_num_seqs:
                continue
            if not self._state_manager.can_append(canonical, 1):
                continue
            if not self._state_manager.may_append(canonical, 1):
                raise RuntimeError(
                    "could not reserve first decode output block during "
                    f"lookahead: request_id={request_id}"
                )
            snapshots.append(canonical.clone_for_decode_dispatch())
            output_offsets[request_id] = record.committed_output_count
            occupied_sp.add(master_sp_idx)
            master_counts[master_sp_idx] += 1

        for sp_idx in range(self.topology.attention_sp):
            if sp_idx not in occupied_sp:
                snapshots.append(self._state_manager.dummy_seqs[sp_idx])
        return self._freeze_batch(
            snapshots,
            wave_id=wave_id,
            quantum_id=quantum_id,
            output_offsets=output_offsets,
        )

    def plan_decode(self, *, wave_id: int, quantum_id: int) -> LocalDecodeBatch:
        if len(self._inflight_batches) >= self.config.hierarchical_async_depth:
            raise RuntimeError("decode flight capacity is exhausted")
        if self._inflight_batches:
            return self._plan_lookahead(
                self._inflight_batches[-1],
                wave_id=wave_id,
                quantum_id=quantum_id,
            )

        sequences = list(self._scheduler.plan_decode()[0])
        self._reconcile_preemptions()
        return self._freeze_batch(
            sequences,
            wave_id=wave_id,
            quantum_id=quantum_id,
        )

    def cancel_all_dummy_lookahead(self, batch: LocalDecodeBatch) -> None:
        if batch.engine_has_real:
            raise RuntimeError("cannot cancel a real decode lookahead batch")
        if not self._inflight_batches or self._inflight_batches[-1] is not batch:
            raise RuntimeError("decode lookahead cancellation is not LIFO")
        self._inflight_batches.pop()

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
        if not any(pending is batch for pending in self._inflight_batches):
            raise RuntimeError("first-forward batch is not pending")
        if not batch_request_ids.issubset(self._inflight_ids):
            raise RuntimeError(
                "first-forward batch contains a non-inflight request"
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
        if record.terminal_emitted:
            return
        record.state = RequestState.ABORT_PENDING
        record.terminal_state = RequestState.ABORTED
        record.terminal_reason = "ABORTED"
        record.drain_fence = record.last_scheduled_quantum
        if record.sequence in self._state_manager.running:
            self._state_manager.running.remove(record.sequence)
        self._emit_terminal(record, "ABORTED")
        if record.outstanding_output_placeholders == 0:
            self._reclaim_terminal(record)

    def abort(self, request_id: int) -> AbortResult:
        record = self._records.get(request_id)
        if record is None:
            if request_id in self._terminal_states:
                return AbortResult(
                    request_id=request_id, status="already_terminal"
                )
            return AbortResult(request_id=request_id, status="not_found")
        if record.terminal_emitted:
            return AbortResult(
                request_id=request_id, status="already_terminal"
            )
        if record.outstanding_output_placeholders > 0:
            self._finish_aborted(request_id)
            return AbortResult(request_id=request_id, status="abort_pending")
        if record.state == RequestState.WAITING_ADMISSION:
            self._scheduler.waiting_migration.remove(record.sequence)
            self._finish_aborted(request_id)
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
        if record.terminal_state not in {
            RequestState.FINISHED,
            RequestState.ABORTED,
        }:
            raise RuntimeError(
                "cannot emit a terminal event without a final state: "
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
        if record.terminal_reason is None:
            raise RuntimeError(
                "terminal request has no finish reason: "
                f"request_id={request_id}, status={status}"
            )
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
                finish_reason=record.terminal_reason,
            )
        )

    def _reclaim_terminal(self, record: LocalRequestRecord) -> None:
        if not record.terminal_emitted or record.terminal_state is None:
            raise RuntimeError("cannot reclaim a non-terminal request")
        if record.outstanding_output_placeholders != 0:
            raise RuntimeError(
                "cannot reclaim request with outstanding decode placeholders"
            )
        request_id = record.sequence.seq_id
        if record.resources_allocated:
            if record.sequence in self._state_manager.running:
                self._state_manager.running.remove(record.sequence)
            self._state_manager.deallocate(
                record.sequence,
                BlockContextSlot.ACTIVE,
            )
            record.resources_allocated = False
        record.state = record.terminal_state
        self._records.pop(request_id)
        self._terminal_states[request_id] = record.terminal_state
        release_wave_id, release_quantum_id = (
            record.last_completed_quantum
            or record.drain_fence
            or (record.admission_wave_id, -1)
        )
        self._resource_release_events.append(
            ResourceReleaseEvent(
                request_id=request_id,
                engine_id=self.engine_id,
                generation_epoch=record.generation_epoch,
                wave_id=release_wave_id,
                quantum_id=release_quantum_id,
            )
        )

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
        return self._commit_decode_batch(
            batch,
            worker_results,
            execute_latency_ms=execute_latency_ms,
        )

    def _commit_decode_batch(
        self,
        batch: LocalDecodeBatch,
        worker_results: list[WorkerDecodeResult],
        *,
        execute_latency_ms: float | None,
    ) -> tuple[FinishEvent, ...]:
        if (
            not self._inflight_batches
            or self._inflight_batches[0] is not batch
        ):
            raise RuntimeError("decode batches must commit in FIFO order")
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

        token_by_request: dict[int, tuple[int, ...]] = {}
        for global_rank in self.topology.global_ranks:
            result = results_by_rank[global_rank]
            for request_id, token_ids in zip(
                result.mastered_request_ids,
                result.sampled_token_ids,
                strict=True,
            ):
                if request_id in token_by_request:
                    raise RuntimeError(
                        "worker results duplicated a mastered request: "
                        f"request_id={request_id}"
                    )
                token_by_request[request_id] = token_ids

        self._last_itl_token_slots = 0
        batch_request_ids = tuple(
            sequence.seq_id
            for sequence in batch._all_sequences
            if not batch.is_control_dummy(sequence)
        )
        for request_id in batch_request_ids:
            record = self._records[request_id]
            if (
                batch.per_request_epoch.get(request_id)
                != record.generation_epoch
            ):
                raise RuntimeError(
                    "decode batch generation epoch changed before commit: "
                    f"request_id={request_id}"
                )
            if record.outstanding_output_placeholders <= 0:
                raise RuntimeError(
                    "decode request has no outstanding output placeholder: "
                    f"request_id={request_id}"
                )

            stale_output = record.state in {
                RequestState.ABORT_PENDING,
                RequestState.TERMINAL_PENDING_DRAIN,
            }
            if stale_output:
                record.outstanding_output_placeholders -= 1
                record.last_completed_quantum = (
                    batch.wave_id,
                    batch.quantum_id,
                )
                if record.outstanding_output_placeholders == 0:
                    self._inflight_ids.discard(request_id)
                    self._reclaim_terminal(record)
                continue

            previous_tokens = record.sequence.num_completed_tokens
            if (
                batch.per_request_output_offset.get(request_id)
                != record.committed_output_count
            ):
                raise RuntimeError(
                    "decode batch output offset changed before commit: "
                    f"request_id={request_id}"
                )
            if record.committed_output_count != previous_tokens:
                raise RuntimeError(
                    "canonical/frontend output count mismatch before commit: "
                    f"request_id={request_id}, "
                    f"canonical={previous_tokens}, "
                    f"committed={record.committed_output_count}"
                )
            token_delta = token_by_request[request_id]
            if len(token_delta) != HIERARCHICAL_LOOP_COUNT:
                raise RuntimeError(
                    "loop-one request produced an invalid token delta: "
                    f"request_id={request_id}, tokens={len(token_delta)}"
                )
            invalid_token = next(
                (
                    token_id
                    for token_id in token_delta
                    if not 0 <= token_id < self.config.hf_config.vocab_size
                ),
                None,
            )
            if invalid_token is not None:
                raise RuntimeError(
                    "worker sampled token outside the model vocabulary: "
                    f"request_id={request_id}, token_id={invalid_token}"
                )

            master_sp_idx = record.sequence.block_ctx(
                BlockContextSlot.ACTIVE
            ).master_sp_idx
            if not 0 <= master_sp_idx < self.topology.attention_sp:
                raise RuntimeError(
                    "decoded request has invalid master SP rank: "
                    f"request_id={request_id}, sp_idx={master_sp_idx}"
                )
            for token_id in token_delta:
                record.sequence.append_token(
                    token_id,
                    BlockContextSlot.ACTIVE,
                    master_sp_idx,
                )
                self._state_manager.add_running_tokens(master_sp_idx, 1)
            completed_tokens = record.sequence.num_completed_tokens
            generated_tokens = completed_tokens - previous_tokens
            self._useful_decode_tokens += generated_tokens
            self._mastered_decode_tokens[master_sp_idx] += generated_tokens
            self._last_itl_token_slots += (
                generated_tokens
                if previous_tokens > 0
                else max(0, generated_tokens - 1)
            )
            materialized_delta = tuple(record.sequence.completion_token_ids)[
                previous_tokens:completed_tokens
            ]
            if materialized_delta != token_delta:
                raise RuntimeError(
                    "canonical Sequence did not materialize committed tokens: "
                    f"request_id={request_id}"
                )
            finish_reason = None
            if (
                not record.sequence.ignore_eos
                and token_delta[-1] == self.config.eos
            ):
                finish_reason = "EOS"
            elif completed_tokens >= record.sequence.max_tokens:
                finish_reason = "LENGTH"
            record.committed_output_count = completed_tokens
            self._token_commit_events.append(
                TokenCommitEvent(
                    request_id=request_id,
                    engine_id=self.engine_id,
                    generation_epoch=record.generation_epoch,
                    wave_id=batch.wave_id,
                    quantum_id=batch.quantum_id,
                    output_offset=completed_tokens,
                    token_ids=token_delta,
                    finish_reason=finish_reason,
                )
            )
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
            if finish_reason is not None:
                record.state = RequestState.TERMINAL_PENDING_DRAIN
                record.terminal_state = RequestState.FINISHED
                record.terminal_reason = finish_reason
                record.drain_fence = record.last_scheduled_quantum
                record.sequence.status = SequenceStatus.FINISHED
                if record.sequence in self._state_manager.running:
                    self._state_manager.running.remove(record.sequence)
                self._emit_terminal(
                    record,
                    "FINISHED",
                    final_quantum_execute_ms=execute_latency_ms,
                )
            else:
                record.state = RequestState.RUNNING_DECODE

            record.outstanding_output_placeholders -= 1
            record.last_completed_quantum = (
                batch.wave_id,
                batch.quantum_id,
            )
            if record.outstanding_output_placeholders == 0:
                self._inflight_ids.discard(request_id)
                if record.terminal_emitted:
                    self._reclaim_terminal(record)

        popped = self._inflight_batches.popleft()
        if popped is not batch:
            raise RuntimeError("decode batch FIFO changed during commit")
        return self.drain_terminal_events()

    def drain_terminal_events(self) -> tuple[FinishEvent, ...]:
        events = tuple(self._terminal_events)
        self._terminal_events.clear()
        return events

    def drain_token_commit_events(self) -> tuple[TokenCommitEvent, ...]:
        events = tuple(self._token_commit_events)
        self._token_commit_events.clear()
        return events

    def drain_first_token_events(self) -> tuple[FirstTokenEvent, ...]:
        events = tuple(self._first_token_events)
        self._first_token_events.clear()
        return events

    def drain_resource_release_events(
        self,
    ) -> tuple[ResourceReleaseEvent, ...]:
        events = tuple(self._resource_release_events)
        self._resource_release_events.clear()
        return events

    def is_finished(self) -> bool:
        return not self._records and not self._inflight_batches

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
