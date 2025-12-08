import enum
from collections import deque
from itertools import count
from typing import List, Literal, postprocess_step, TYPE_CHECKING

import numpy as np

from nanodeploy.config import Config
from nanodeploy.engine.block_manager import BlockManager
from nanodeploy.engine.sequence import Sequence, SequenceStatus
from nanodeploy.logging import get_logger

if TYPE_CHECKING:
    from nanodeploy.metrics import MetricsManager


logger = get_logger()


class RoutingStrategy(enum.Enum):
    RoundRobin = enum.auto()
    LeastToken = enum.auto()
    LeastCache = enum.auto()


class SPStateManager:
    _segment_size = 1024

    def __init__(
        self,
        engine_id: str | None,
        attention_sp: int,
        num_kvcache_blocks: int,
        kvcache_block_size: int,
        max_num_seqs: int,
        max_num_batched_tokens: int,
    ):
        self.engine_id = engine_id
        self.attention_sp = attention_sp

        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens

        self.block_manager: dict[int, BlockManager] = {
            i: BlockManager(
                engine_id,
                i,
                num_kvcache_blocks,
                kvcache_block_size,
            )
            for i in range(attention_sp)
        }

        self.block_manager[0].free_block_ids

        self.running: deque[Sequence] = deque()

        self.routing_startegy = RoutingStrategy.RoundRobin
        self.sp_rr_counter = (idx % self.attention_sp for idx in count())

        self.dummy_seqs: list[Sequence] = []
        self._initialize_dummy_seqs()

    @property
    def is_empty(self):
        return not self.running

    def _initialize_dummy_seqs(self):
        for sp_idx in range(self.attention_sp):
            dummy_seq = Sequence(
                token_ids=[np.random.randint(8000)],
                sampling_params=None,
                engine_id=self.engine_id,
                master_sp_rank=sp_idx,
            )

            dummy_seq.append_token(
                np.random.randint(8000),
                self.engine_id,
                sp_idx,
            )

            self.block_manager[sp_idx].allocate(dummy_seq)
            self.dummy_seqs.append(dummy_seq)

    def can_append(self, seq: Sequence, num_tokens: int = 1):
        return self.block_manager[
            seq.block_ctx(self.engine_id).master_sp_idx
        ].can_append(seq, num_tokens)

    def may_append(self, seq: Sequence, num_tokens: int = 1):
        return self.block_manager[
            seq.block_ctx(self.engine_id).master_sp_idx
        ].may_append(seq, num_tokens)

    def can_allocate(
        self,
        seq: Sequence,
        num_seqs: dict[int, int],
        num_batched_tokens: dict[int, int],
    ):
        # Step 1: cal num_blocks and num_blocks_per_rank
        block_ctx = seq.block_ctx(self.engine_id)
        block_ctx.num_dispatched_tokens.clear()

        num_segments = (seq.num_tokens + self._segment_size - 1) // self._segment_size
        num_segments_per_rank = (
            num_segments + self.attention_sp - 1
        ) // self.attention_sp
        num_ranks = (num_segments + num_segments_per_rank - 1) // num_segments_per_rank

        master_rank = next(self.sp_rr_counter)

        if num_seqs[master_rank] >= self.max_num_seqs:
            return False

        if num_batched_tokens[master_rank] + len(seq) >= self.max_num_batched_tokens:
            return False

        rank_free_count = [
            (rank, len(block_manager.free_block_ids))
            for rank, block_manager in self.block_manager.items()
            if rank != master_rank
        ]
        rank_free_count_sorted = sorted(rank_free_count, key=lambda x: x[1])
        top_least_free_ranks = [
            item[0] for item in rank_free_count_sorted[: (num_ranks - 1)]
        ] + [master_rank]

        # step 2: allocation
        block_ctx.master_sp_idx = master_rank
        total_token_unalloc = seq.num_tokens
        for sp_idx in top_least_free_ranks:
            block_ctx.num_dispatched_tokens[sp_idx] = min(
                total_token_unalloc, num_segments_per_rank * self._segment_size
            )
            total_token_unalloc -= num_segments_per_rank * self._segment_size

        return all(
            self.block_manager[sp_idx].can_allocate(seq)
            for sp_idx in range(self.attention_sp)
        )

    def allocate(self, seq: Sequence):
        block_ctx = seq.block_ctx(self.engine_id)
        for sp_idx in range(self.attention_sp):
            if sp_idx != block_ctx.master_sp_idx:
                self.block_manager[sp_idx].allocate(seq)
        self.block_manager[block_ctx.master_sp_idx].allocate(seq)

    def deallocate(self, seq: Sequence):
        for sp_idx in range(self.attention_sp):
            return self.block_manager[sp_idx].deallocate(seq)
        seq.block_ctx(self.engine_id).sp_block_table.clear()
        seq.block_ctx(self.engine_id).block_location.clear()
        seq.block_ctx(self.engine_id).num_dispatched_tokens.clear()


class Scheduler:

    def __init__(self, config: Config):
        self.engine_id = config.engine_id
        self.loop_count = config.loop_count
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos

        self.waiting_migration: deque[Sequence] = deque()
        self.waiting: deque[Sequence] = deque()

        self.attention_dp = config.attention_dp
        self.attention_sp = config.attention_sp
        self.rounting_strategy = RoutingStrategy.RoundRobin

        self.worker_state = [
            SPStateManager(
                config.engine_id,
                self.attention_sp,
                config.num_kvcache_blocks,
                config.kvcache_block_size,
                config.max_num_seqs,
                config.max_num_batched_tokens,
            )
            for _ in range(self.attention_dp)
        ]
        self.to_be_migrated: dict[str, tuple[Sequence, int]] = {}

        self.mode: Literal["prefill", "decode", "hybrid"] = config.mode
        self.dp_rr_counter = (idx % self.attention_dp for idx in count())

    def is_finished(self):
        waiting = self.waiting if self.mode != "decode" else self.waiting_migration
        return not waiting and all(w.is_empty for w in self.worker_state)

    def add(self, seq: Sequence):
        if self.mode == "decode":
            self.waiting_migration.append(seq)
            seq.metric.record_arrival()
        else:
            self.waiting.append(seq)
            seq.metric.record_arrival()

    def running(self, dp_idx: int):
        return self.worker_state[dp_idx].running

    def block_manager(self, dp_idx: int):
        return self.worker_state[dp_idx].block_manager

    def _schedule_prefill(self) -> list[list[Sequence]]:
        scheduled_seqs = [[] for _ in range(self.attention_dp)]
        num_seqs: dict[int, dict[int, int]] = {
            dp_id: {sp_id: 0 for sp_id in range(self.attention_sp)}
            for dp_id in range(self.attention_dp)
        }
        num_batched_tokens: dict[int, dict[int, int]] = {
            dp_id: {sp_id: 0 for sp_id in range(self.attention_sp)}
            for dp_id in range(self.attention_dp)
        }

        waiting = self.waiting if self.mode != "decode" else self.waiting_migration

        while waiting:
            seq = waiting[0]
            if self.rounting_strategy == RoutingStrategy.RoundRobin:
                for _ in range(self.attention_dp):
                    selected_dp_idx = next(self.dp_rr_counter)

                    can_allocate = self.worker_state[selected_dp_idx].can_allocate(
                        seq, num_seqs[selected_dp_idx], num_seqs[selected_dp_idx]
                    )
                    if not can_allocate:
                        continue
                    block_ctx = seq.block_ctx(self.engine_id)
                    num_seqs[selected_dp_idx][block_ctx.master_sp_idx] += 1
                    seq.block_ctx_map[self.engine_id].dp_idx = selected_dp_idx

                    self.worker_state[selected_dp_idx].allocate(seq)
                    num_batched_tokens[selected_dp_idx][block_ctx.master_sp_idx] += (
                        len(seq) - seq.num_cached_tokens
                    )
                    seq.status = SequenceStatus.RUNNING
                    waiting.popleft()
                    self.running(selected_dp_idx).append(seq)
                    scheduled_seqs[selected_dp_idx].append(seq)
                    if seq.metric:
                        seq.metric.record_first_scheduled()
                    break
                else:
                    break
            elif self.rounting_strategy == RoutingStrategy.LeastToken:
                pass
            elif self.rounting_strategy == RoutingStrategy.LeastCache:
                pass
            else:
                raise AttributeError
        return scheduled_seqs

    def _schedule_decode(self) -> list[list[Sequence]]:
        scheduled_seqs = [[] for _ in range(self.attention_dp)]
        num_seqs = {replica_id: 0 for replica_id in range(self.attention_dp)}
        for selected_dp_idx in range(self.attention_dp):
            while self.running(selected_dp_idx):
                seq = self.running(selected_dp_idx).popleft()
                while not self.worker_state[selected_dp_idx].can_append(
                    seq, num_tokens=self.loop_count
                ):
                    if self.running(selected_dp_idx):
                        self.preempt(
                            selected_dp_idx, self.running(selected_dp_idx).pop()
                        )
                    else:
                        self.preempt(selected_dp_idx, seq)
                        break
                else:
                    num_seqs[selected_dp_idx] += 1
                    self.worker_state[selected_dp_idx].may_append(
                        seq, num_tokens=self.loop_count
                    )
                    scheduled_seqs[selected_dp_idx].append(seq)
            self.running(selected_dp_idx).extendleft(
                reversed(scheduled_seqs[selected_dp_idx])
            )

        for dp_idx, dp_seqs in enumerate(scheduled_seqs):
            sp_lens = [0 for _ in range(self.attention_sp)]
            for seq in dp_seqs:
                sp_lens[seq.block_ctx(self.engine_id).master_sp_idx] += len(seq)

            for sp_idx in range(self.attention_sp):
                if sp_lens[sp_idx] == 0:
                    scheduled_seqs[dp_idx].append(
                        self.worker_state[dp_idx].dummy_seqs[sp_idx]
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

    def preempt(self, dp_idx: int, seq: Sequence):
        logger.info("preemption happens")
        seq.status = SequenceStatus.WAITING
        self.worker_state[dp_idx].deallocate(seq)
        seq.num_checkpointed_tokens = len(seq.token_ids)
        self.waiting.appendleft(seq)

    def postprocess(
        self,
        dp_seqs: List[List[List[Sequence]]],
        dp_token_ids: List[List[List[List[int]]]],
        metrics_manager: "MetricsManager | None" = None,
    ):
        dp_dummy_seqs = [ws.dummy_seqs for ws in self.worker_state]

        postprocess_result = postprocess_step(
            dp_seqs,
            dp_token_ids,
            dp_dummy_seqs,
            self.engine_id,
            self.eos,
            self.mode == "prefill",
            metrics_manager,
        )

        finished_list = postprocess_result.finished
        migrated_list = postprocess_result.migrated

        for dp_idx, seq in finished_list:
            self.worker_state[dp_idx].deallocate(seq)
            try:
                self.running(dp_idx).remove(seq)
            except ValueError:
                pass

        for dp_idx, seq in migrated_list:
            # 已经在 C++ 设置了 seq.status, backup_engine_id 等
            try:
                self.running(dp_idx).remove(seq)
            except ValueError:
                pass
            self.to_be_migrated[seq.seq_id] = (seq, dp_idx)

    def free_to_be_migrated(self, seqs: Sequence | list[Sequence]):
        if isinstance(seqs, Sequence):
            seqs = [seqs]
        for seq in seqs:
            seq, selected_dp_idx = self.to_be_migrated[seq.seq_id]
            self.worker_state[selected_dp_idx].deallocate(seq)
            del self.to_be_migrated[seq.seq_id]
