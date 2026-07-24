from __future__ import annotations

from dataclasses import dataclass

from nanodeploy._cpp import BlockContextSlot
from nanodeploy.config import Config
from nanodeploy.engine.hierarchical_contract import (
    AddCommand,
    AddResult,
    AbortResult,
    FinishEvent,
    HIERARCHICAL_LOOP_COUNT,
    LoadSnapshot,
    LocalDecodeBatch,
    RequestState,
    WorkerDecodeResult,
    round_up,
    validate_add_request,
)
from nanodeploy.engine.scheduler import Scheduler
from nanodeploy.engine.sequence import Sequence
from nanodeploy.engine.topology import EngineTopology
from nanodeploy.sampling_params import SamplingParams


@dataclass(slots=True)
class LocalRequestRecord:
    sequence: Sequence
    state: RequestState
    original_prompt_len: int
    padded_completion_len: int
    terminal_emitted: bool = False


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
        self._records: dict[int, LocalRequestRecord] = {}
        self._terminal_events: list[FinishEvent] = []
        self._inflight_ids: set[int] = set()
        self._all_dummy_engine_quantums = 0

    @property
    def cpp_scheduler(self) -> Scheduler:
        return self._scheduler

    @property
    def state_manager(self):
        return self._state_manager

    def _active_request_count(self) -> int:
        return sum(
            not record.state.is_terminal for record in self._records.values()
        )

    def _validate_service_capacity(
        self, total_capacity_len: int, padded_completion_len: int
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

    def add(self, command: AddCommand) -> AddResult:
        existing = self._records.get(command.request_id)
        if existing is not None:
            return AddResult(
                request_id=command.request_id,
                accepted=False,
                engine_id=self.engine_id,
                reason=f"duplicate request in state {existing.state.value}",
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
            self._validate_service_capacity(
                validation.total_capacity_len,
                validation.padded_completion_len,
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
        )
        return AddResult(
            request_id=command.request_id,
            accepted=True,
            engine_id=self.engine_id,
        )

    def admit(self) -> tuple[int, ...]:
        admitted = self._scheduler.admit()[0]
        admitted_ids: list[int] = []
        for sequence in admitted:
            record = self._records[sequence.seq_id]
            if not self._state_manager.can_fit_lifetime(
                sequence, 1 + record.padded_completion_len
            ):
                self._state_manager.running.remove(sequence)
                self._state_manager.deallocate(
                    sequence, BlockContextSlot.ACTIVE
                )
                raise RuntimeError(
                    "accepted request failed exact padded lifetime validation: "
                    f"request_id={sequence.seq_id}"
                )
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
            record.state = RequestState.RUNNING_DECODE
            admitted_ids.append(sequence.seq_id)
        return tuple(admitted_ids)

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
        real_sequences = [
            sequence
            for sequence in sequences
            if sequence.seq_id not in control_dummy_ids
        ]
        self._inflight_ids = {sequence.seq_id for sequence in real_sequences}
        if not real_sequences:
            self._all_dummy_engine_quantums += 1

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
        )

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
            return AbortResult(request_id=request_id, status="not_found")
        if record.state.is_terminal:
            return AbortResult(request_id=request_id, status="already_terminal")
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
        self, record: LocalRequestRecord, status: str
    ) -> None:
        if record.terminal_emitted:
            raise RuntimeError(
                f"duplicate terminal event for request {record.sequence.seq_id}"
            )
        record.terminal_emitted = True
        self._terminal_events.append(
            FinishEvent(
                request_id=record.sequence.seq_id,
                generated_count=record.sequence.num_completed_tokens,
                status=status,
                engine_id=self.engine_id,
            )
        )

    def postprocess(
        self,
        batch: LocalDecodeBatch,
        worker_results: list[WorkerDecodeResult],
    ) -> tuple[FinishEvent, ...]:
        if batch.engine_id != self.engine_id:
            raise ValueError("decode batch belongs to a different LocalScheduler")
        results_by_rank = batch.validate_worker_results(worker_results)

        aborted_ids = {
            request_id
            for request_id in self._inflight_ids
            if self._records[request_id].state == RequestState.ABORT_PENDING
        }
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
                if sequence.seq_id in aborted_ids:
                    continue
                rank_sequences.append(sequence)
                if sequence.seq_id in batch.control_dummy_ids:
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
        for request_id in sorted(self._inflight_ids.difference(aborted_ids)):
            record = self._records[request_id]
            if record.sequence.is_finished:
                record.state = RequestState.FINISHED
                self._emit_terminal(record, "FINISHED")
            else:
                record.state = RequestState.RUNNING_DECODE
        self._inflight_ids.clear()
        return self.drain_terminal_events()

    def drain_terminal_events(self) -> tuple[FinishEvent, ...]:
        events = tuple(self._terminal_events)
        self._terminal_events.clear()
        return events

    def is_finished(self) -> bool:
        return all(record.state.is_terminal for record in self._records.values())

    def load_snapshot(self, *, wave_id: int, quantum_id: int) -> LoadSnapshot:
        free_blocks = [
            block_manager.num_free_blocks
            for block_manager in self._state_manager.block_manager.values()
        ]
        return LoadSnapshot(
            engine_id=self.engine_id,
            ready=True,
            waiting=sum(
                record.state == RequestState.WAITING_ADMISSION
                for record in self._records.values()
            ),
            running=sum(
                record.state
                in {RequestState.RUNNING_DECODE, RequestState.ABORT_PENDING}
                for record in self._records.values()
            ),
            free_blocks_min=min(free_blocks, default=0),
            wave_id=wave_id,
            quantum_id=quantum_id,
            useful_real_batch_size=len(self._inflight_ids),
            control_dummy_count=len(self._state_manager.dummy_seqs),
            all_dummy_engine_quantums=self._all_dummy_engine_quantums,
        )
