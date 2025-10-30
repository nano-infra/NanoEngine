import dataclasses
import enum
from collections import deque
from typing import List

from nanovllm.engine.block_manager import BlockManager

from nanodeploy.config import Config
from nanodeploy.engine.sequence import Sequence, SequenceStatus


class RoutingStrategy(enum.Enum):
    RoundRobin = enum.auto()
    LeastToken = enum.auto()
    LeastCache = enum.auto()


class WorkerState:
    def __init__(self, num_kv_cache_blocks: int, kvcache_block_size: int):
        self.running: deque[Sequence] = deque()
        self.block_manager = BlockManager(num_kv_cache_blocks, kvcache_block_size)

    def is_finished(self):
        return not self.running


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.waiting: deque[Sequence] = deque()

        self.num_replica = config.data_parallel_size
        self.rounting_strategy = RoutingStrategy.RoundRobin
        self.worker_state = [
            WorkerState(config.num_kvcache_blocks, config.kvcache_block_size)
            for _ in range(self.num_replica)
        ]

        self.rr_generator = self.route_by_rr()

    def is_finished(self):
        return not self.waiting and all(w.is_finished() for w in self.worker_state)

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def route_by_rr(self):
        if not hasattr(self, "selected_replica"):
            setattr(self, "selected_replica", 0)
        while True:
            yield self.selected_replica
            self.selected_replica = (self.selected_replica + 1) % self.num_replica

    def running(self, selected_replica: int):
        return self.worker_state[selected_replica].running

    def block_manager(self, selected_replica: int):
        return self.worker_state[selected_replica].block_manager

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = [[] for _ in range(self.num_replica)]
        num_seqs = {replica_id: 0 for replica_id in range(self.num_replica)}
        num_batched_tokens = {replica_id: 0 for replica_id in range(self.num_replica)}
        while self.waiting :
            seq = self.waiting[0]
            for i in range(self.num_replica):
                selected_replica = self.rr_generator.__next__()
                if num_seqs[selected_replica] >= self.max_num_seqs:
                    continue
                if num_batched_tokens[selected_replica] + len(seq) > self.max_num_batched_tokens or not self.block_manager(selected_replica).can_allocate(seq):
                    continue
                num_seqs[selected_replica] += 1
                self.block_manager(selected_replica).allocate(seq)
                num_batched_tokens[selected_replica] += len(seq) - seq.num_cached_tokens
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running(selected_replica).append(seq)
                scheduled_seqs[selected_replica].append(seq)
                break
            else:
                break
        if any(scheduled_seqs):
            return scheduled_seqs, True

        # decode
        for selected_replica in range(self.num_replica):
            while self.running(selected_replica) and num_seqs[selected_replica] < self.max_num_seqs:
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
        assert any(scheduled_seqs)
        return scheduled_seqs, False

    def preempt(self, selected_replica: int, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager(selected_replica).deallocate(seq)
        self.waiting.appendleft(seq)
        seq.num_completion_tokens

    def postprocess(self, seqs: List[List[Sequence]], token_ids: List[List[int]]):
        for i in range(self.num_replica):
            for seq, token_id in zip(seqs[i], token_ids[i]):
                seq.append_token(token_id)
                if (
                    not seq.ignore_eos and token_id == self.eos
                ) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager(i).deallocate(seq)
                    self.running(i).remove(seq)
