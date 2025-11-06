import enum
from collections import deque
from typing import Literal

from nanodeploy.config import Config
from nanodeploy.engine.block_manager import BlockManager
from nanodeploy.engine.sequence import Sequence, SequenceStatus


class RoutingStrategy(enum.Enum):
    RoundRobin = enum.auto()
    LeastToken = enum.auto()
    LeastCache = enum.auto()


class SPBlockManager:
    def __init__(
        self,
        engine_id: str,
        attention_sp: str,
        num_kvcache_blocks: int,
        kvcache_block_size: int,
    ):
        self.attention_sp = attention_sp
        self.block_manager: dict[str, BlockManager] = {
            i: BlockManager(
                engine_id,
                i,
                num_kvcache_blocks,
                kvcache_block_size,
            )
            for i in range(attention_sp)
        }

    def can_append(self, seq: Sequence):
        return self.block_manager[0].can_append(seq)

    def may_append(self, seq: Sequence):
        return self.block_manager[0].may_append(seq)

    def can_allocate(self, seq: Sequence):
        return self.block_manager[0].can_allocate(seq)

    def allocate(self, seq: Sequence):
        return self.block_manager[0].allocate(seq)

    def deallocate(self, seq: Sequence):
        for sp_idx in range(self.attention_sp):
            return self.block_manager[sp_idx].deallocate(seq)


class SPWorkerState:
    def __init__(
        self,
        engine_id: str,
        attention_sp: int,
        num_kv_cache_blocks: int,
        kvcache_block_size: int,
    ):
        self.attention_sp = attention_sp

        self.running: deque[Sequence] = deque()
        self.sp_block_manager = SPBlockManager(
            engine_id, attention_sp, num_kv_cache_blocks, kvcache_block_size
        )

    @property
    def is_empty(self):
        return not self.running


class Scheduler:

    def __init__(self, config: Config):
        self.engine_id = config.engine_id
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos

        self.waiting_migration: deque[Sequence] = deque()
        self.waiting: deque[Sequence] = deque()

        self.attention_dp = config.attention_dp
        self.attention_sp = config.attention_sp
        self.rounting_strategy = RoutingStrategy.RoundRobin

        self.worker_state = [
            SPWorkerState(
                config.engine_id,
                self.attention_sp,
                config.num_kvcache_blocks,
                config.kvcache_block_size,
            )
            for _ in range(self.attention_dp)
        ]
        self.to_be_migrated: dict[str, tuple[Sequence, list[int]]] = {}

        self.mode: Literal["prefill", "decode", "hybrid"] = config.mode
        self.rr_generator = self.route_by_rr()

    def is_finished(self):
        waiting = self.waiting if self.mode != "decode" else self.waiting_migration
        return not waiting and all(w.is_empty for w in self.worker_state)

    def add(self, seq: Sequence):
        if self.mode == "decode":
            self.waiting_migration.append(seq)
        else:
            self.waiting.append(seq)

    def route_by_rr(self):
        if not hasattr(self, "rr_selected"):
            setattr(self, "rr_selected", 0)
        while True:
            yield self.rr_selected
            self.rr_selected = (self.rr_selected + 1) % self.attention_dp

    def running(self, selected_replica: int):
        return self.worker_state[selected_replica].running

    def to_be_migrated(self, selected_replica: int):
        return self.worker_state[selected_replica].to_be_migrated

    def block_manager(self, selected_replica: int):
        return self.worker_state[selected_replica].sp_block_manager

    def _schedule_prefill(self) -> list[list[Sequence]]:
        scheduled_seqs = [[] for _ in range(self.attention_dp)]
        num_seqs = {replica_id: 0 for replica_id in range(self.attention_dp)}
        num_batched_tokens = {replica_id: 0 for replica_id in range(self.attention_dp)}

        waiting = self.waiting if self.mode != "decode" else self.waiting_migration

        while waiting:
            seq = waiting[0]
            for _ in range(self.attention_dp):
                selected_replica = self.rr_generator.__next__()
                if num_seqs[selected_replica] >= self.max_num_seqs:
                    continue
                num_batched_tokens_satisfied = (
                    num_batched_tokens[selected_replica] + len(seq)
                    <= self.max_num_batched_tokens
                )
                can_allocate = self.block_manager(selected_replica).can_allocate(seq)
                if not num_batched_tokens_satisfied or not can_allocate:
                    continue

                num_seqs[selected_replica] += 1
                seq.block_ctx_map[self.engine_id].selected_dp_idx = selected_replica

                self.block_manager(selected_replica).allocate(seq)
                num_batched_tokens[selected_replica] += len(seq) - seq.num_cached_tokens
                seq.status = SequenceStatus.RUNNING
                waiting.popleft()
                self.running(selected_replica).append(seq)
                scheduled_seqs[selected_replica].append(seq)
                break
            else:
                break
        return scheduled_seqs

    def _schedule_decode(self) -> list[list[Sequence]]:
        scheduled_seqs = [[] for _ in range(self.attention_dp)]
        num_seqs = {replica_id: 0 for replica_id in range(self.attention_dp)}
        for selected_replica in range(self.attention_dp):
            while (
                self.running(selected_replica)
                and num_seqs[selected_replica] < self.max_num_seqs
            ):
                seq = self.running(selected_replica).popleft()
                while not self.block_manager(selected_replica).can_append(seq):
                    if self.running(selected_replica):
                        self.preempt(
                            selected_replica, self.running(selected_replica).pop()
                        )
                    else:
                        self.preempt(selected_replica, seq)
                        break
                else:
                    num_seqs[selected_replica] += 1
                    self.block_manager(selected_replica).may_append(seq)
                    scheduled_seqs[selected_replica].append(seq)
            self.running(selected_replica).extendleft(
                reversed(scheduled_seqs[selected_replica])
            )
        return scheduled_seqs

    def schedule(self) -> tuple[list[list[Sequence]], bool]:
        # prefill
        scheduled_seqs = self._schedule_prefill()

        if any(scheduled_seqs):
            return scheduled_seqs, True

        # decode
        scheduled_seqs = self._schedule_decode()

        assert any(scheduled_seqs)

        return scheduled_seqs, False

    def preempt(self, selected_replica: int, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager(selected_replica).deallocate(seq)
        seq.current_checkpointed_tokens = len(seq.token_ids)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[list[Sequence]], token_ids: list[list[int]]):
        for dp_idx in range(self.attention_dp):
            for idx, (seq) in enumerate(seqs[dp_idx]):
                token_id = token_ids[dp_idx][seq.block_ctx().master_sp_rank][idx]
                seq.append_token(token_id)
                if (
                    not seq.ignore_eos and token_id == self.eos
                ) or seq.num_generated_tokens_since_checkpoint == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager(dp_idx).deallocate(seq)
                    self.running(dp_idx).remove(seq)
                elif self.mode == "prefill":
                    seq.status = SequenceStatus.TO_BE_MIGRATED
                    seq.backup_engine_id = seq.active_engine_id
                    seq.active_engine_id = None
                    self.running(dp_idx).remove(seq)
                    self.to_be_migrated[seq.seq_id] = (seq, dp_idx)

    def free_to_be_migrated(self, seqs: Sequence | list[Sequence]):
        if isinstance(seqs, Sequence):
            seqs = [seqs]
        for seq in seqs:
            seq, selected_dp_idx = self.to_be_migrated[seq.seq_id]
            self.block_manager(selected_dp_idx).deallocate(seq)
            del self.to_be_migrated[seq.seq_id]
