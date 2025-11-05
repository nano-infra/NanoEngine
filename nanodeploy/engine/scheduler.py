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


class WorkerState:
    def __init__(
        self, engine_id: str, num_kv_cache_blocks: int, kvcache_block_size: int
    ):
        self.running: deque[Sequence] = deque()
        self.to_be_migrated: dict[str, Sequence] = dict()
        self.block_manager = BlockManager(
            engine_id, num_kv_cache_blocks, kvcache_block_size
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

        self.num_replica = config.attention_dp
        self.rounting_strategy = RoutingStrategy.RoundRobin

        self.worker_state = [
            WorkerState(
                config.engine_id, config.num_kvcache_blocks, config.kvcache_block_size
            )
            for _ in range(self.num_replica)
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
        if not hasattr(self, "selected_replica"):
            setattr(self, "selected_replica", 0)
        while True:
            yield self.selected_replica
            self.selected_replica = (self.selected_replica + 1) % self.num_replica

    def running(self, selected_replica: int):
        return self.worker_state[selected_replica].running

    def to_be_migrated(self, selected_replica: int):
        return self.worker_state[selected_replica].to_be_migrated

    def block_manager(self, selected_replica: int):
        return self.worker_state[selected_replica].block_manager

    def _schedule_prefill(self) -> list[list[Sequence]]:
        scheduled_seqs = [[] for _ in range(self.num_replica)]
        num_seqs = {replica_id: 0 for replica_id in range(self.num_replica)}
        num_batched_tokens = {replica_id: 0 for replica_id in range(self.num_replica)}

        waiting = self.waiting if self.mode != "decode" else self.waiting_migration

        while waiting:
            seq = waiting[0]
            for _ in range(self.num_replica):
                selected_replica = self.rr_generator.__next__()
                if num_seqs[selected_replica] >= self.max_num_seqs:
                    continue
                if num_batched_tokens[selected_replica] + len(
                    seq
                ) > self.max_num_batched_tokens or not self.block_manager(
                    selected_replica
                ).can_allocate(
                    seq
                ):
                    continue
                num_seqs[selected_replica] += 1
                seq.block_ctx_map[self.engine_id].selected_replica = selected_replica

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
        scheduled_seqs = [[] for _ in range(self.num_replica)]
        num_seqs = {replica_id: 0 for replica_id in range(self.num_replica)}
        for selected_replica in range(self.num_replica):
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
        for i in range(self.num_replica):
            for seq, token_id in zip(seqs[i], token_ids[i]):
                seq.append_token(token_id)
                if (
                    not seq.ignore_eos and token_id == self.eos
                ) or seq.num_generated_tokens_since_checkpoint == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager(i).deallocate(seq)
                    self.running(i).remove(seq)
                elif self.mode == "prefill":
                    seq.status = SequenceStatus.TO_BE_MIGRATED
                    seq.backup_engine_id = seq.active_engine_id
                    seq.active_engine_id = None
                    self.running(i).remove(seq)
                    self.to_be_migrated[seq.seq_id] = (seq, [i])

    def free_to_be_migrated(self, seqs: Sequence | list[Sequence]):
        if isinstance(seqs, Sequence):
            seqs = [seqs]
        for seq in seqs:
            seq, selected_replicas = self.to_be_migrated[seq.seq_id]
            for selected in selected_replicas:
                self.block_manager(selected).deallocate(seq)
            del self.to_be_migrated[seq.seq_id]
