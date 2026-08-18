from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

from nanodeploy.engine.hierarchical_contract import (
    AddCommand,
    AdmissionReservation,
    LoadSnapshot,
    round_up,
)


@dataclass(frozen=True, slots=True)
class AdmissionPlannerConfig:
    """Configuration needed to mirror one LocalEngine admission planner."""

    attention_sp: int
    kvcache_block_size: int
    max_num_seqs: int
    max_num_batched_tokens: int
    max_num_recv_seqs: int
    reserved_blocks_per_req: float
    segment_size: int
    queue_capacity: int
    use_new_decode_dynamic_sp_scheduler: bool = False
    dynamic_sp_size_strategy: str = "legacy"
    dynamic_sp_bucket_policy: str = ""
    enable_non_uniform_split: bool = False
    sp_master_selector: str = "LeastBatch"
    fixed_sp_size: int = 0

    @classmethod
    def from_config(cls, config: Any) -> AdmissionPlannerConfig:
        return cls(
            attention_sp=config.attention_sp,
            kvcache_block_size=config.kvcache_block_size,
            max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
            max_num_recv_seqs=config.max_num_recv_seqs,
            reserved_blocks_per_req=config.reserved_blocks_per_req,
            segment_size=config.segment_size,
            queue_capacity=config.hierarchical_queue_capacity,
            use_new_decode_dynamic_sp_scheduler=(
                config.use_new_decode_dynamic_sp_scheduler
            ),
            dynamic_sp_size_strategy=config.dynamic_sp_size_strategy,
            dynamic_sp_bucket_policy=config.dynamic_sp_bucket_policy,
            enable_non_uniform_split=config.enable_non_uniform_split,
            sp_master_selector=config.sp_master_selector,
            fixed_sp_size=config.fixed_sp_size,
        )

    def __post_init__(self) -> None:
        positive = {
            "attention_sp": self.attention_sp,
            "kvcache_block_size": self.kvcache_block_size,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_num_recv_seqs": self.max_num_recv_seqs,
            "segment_size": self.segment_size,
            "queue_capacity": self.queue_capacity,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(
                "admission planner values must be positive: "
                + ", ".join(invalid)
            )
        if self.reserved_blocks_per_req < 0:
            raise ValueError("reserved_blocks_per_req must be non-negative")


@dataclass(slots=True)
class AdmissionShadow:
    engine_id: int
    free_blocks: list[int]
    total_blocks: list[int]
    control_dummy_blocks: list[int]
    master_counts: list[int]
    receiver_counts: list[int]
    dispatched_tokens: list[int]
    batch_master_counts: list[int]
    batch_tokens: list[int]
    queue_slots: int
    rr_cursor: int

    def copy(self) -> AdmissionShadow:
        return AdmissionShadow(
            engine_id=self.engine_id,
            free_blocks=list(self.free_blocks),
            total_blocks=list(self.total_blocks),
            control_dummy_blocks=list(self.control_dummy_blocks),
            master_counts=list(self.master_counts),
            receiver_counts=list(self.receiver_counts),
            dispatched_tokens=list(self.dispatched_tokens),
            batch_master_counts=list(self.batch_master_counts),
            batch_tokens=list(self.batch_tokens),
            queue_slots=self.queue_slots,
            rr_cursor=self.rr_cursor,
        )


class AdmissionPlanner:
    """Pure frontend mirror of LocalEngine KV/SP admission feasibility."""

    def __init__(self, config: AdmissionPlannerConfig) -> None:
        self.config = config
        self._bucket_policy = self._parse_bucket_policy(
            config.dynamic_sp_bucket_policy
        )

    def shadow_from_snapshot(
        self, snapshot: LoadSnapshot
    ) -> AdmissionShadow | None:
        # A preempted local request remains ahead of any new global
        # admission command in Scheduler.waiting_migration. Let the
        # LocalEngine clear that queue before planning new work for this DP.
        if snapshot.waiting > 0:
            return None
        by_sp = {rank.sp_idx: rank for rank in snapshot.rank_loads}
        expected = set(range(self.config.attention_sp))
        if set(by_sp) != expected:
            return None
        ranks = [by_sp[sp_idx] for sp_idx in range(self.config.attention_sp)]
        return AdmissionShadow(
            engine_id=snapshot.engine_id,
            free_blocks=[rank.free_blocks for rank in ranks],
            total_blocks=[rank.total_blocks for rank in ranks],
            control_dummy_blocks=[
                rank.control_dummy_blocks for rank in ranks
            ],
            master_counts=[
                rank.active_master_requests for rank in ranks
            ],
            receiver_counts=[
                rank.active_receiver_requests for rank in ranks
            ],
            dispatched_tokens=[
                rank.active_dispatched_tokens for rank in ranks
            ],
            batch_master_counts=[0] * self.config.attention_sp,
            batch_tokens=[0] * self.config.attention_sp,
            queue_slots=snapshot.reserved_slots,
            rr_cursor=(
                sum(rank.master_assignments for rank in ranks)
                % self.config.attention_sp
            ),
        )

    def apply_reservation(
        self,
        shadow: AdmissionShadow,
        reservation: AdmissionReservation,
        *,
        current_batch: bool = False,
    ) -> None:
        block_size = self.config.kvcache_block_size
        master = reservation.master_sp_idx
        shadow.master_counts[master] += 1
        shadow.queue_slots += 1
        if self.config.sp_master_selector == "RoundRobin":
            shadow.rr_cursor = (master + 1) % self.config.attention_sp
        for sp_idx, token_count in enumerate(
            reservation.dispatched_tokens
        ):
            if token_count <= 0:
                continue
            shadow.free_blocks[sp_idx] -= self._ceil_div(
                token_count, block_size
            )
            shadow.dispatched_tokens[sp_idx] += token_count
            if sp_idx != master:
                shadow.receiver_counts[sp_idx] += 1
        if current_batch:
            shadow.batch_master_counts[master] += 1
            shadow.batch_tokens[master] += sum(
                reservation.dispatched_tokens
            )

    def plan(
        self,
        shadow: AdmissionShadow,
        command: AddCommand,
    ) -> AdmissionReservation | None:
        if shadow.queue_slots >= self.config.queue_capacity:
            return None
        if self.config.use_new_decode_dynamic_sp_scheduler:
            reservation = self._plan_balanced(shadow, command)
        else:
            reservation = self._plan_legacy(shadow, command)
        if reservation is None:
            return None
        if not self._fits_lifetime(reservation, command.max_tokens, shadow):
            return None
        self.apply_reservation(
            shadow, reservation, current_batch=True
        )
        return reservation

    def _plan_legacy(
        self,
        shadow: AdmissionShadow,
        command: AddCommand,
    ) -> AdmissionReservation | None:
        prompt_tokens = len(command.prompt_token_ids)
        master = self._select_master(shadow)
        if shadow.master_counts[master] + 1 > self.config.max_num_seqs:
            return None
        if (
            shadow.batch_tokens[master] + prompt_tokens
            >= self.config.max_num_batched_tokens
        ):
            return None

        num_segments = self._ceil_div(
            prompt_tokens, self.config.segment_size
        )
        segments_per_rank = self._ceil_div(
            num_segments, self.config.attention_sp
        )
        initial_ranks = self._ceil_div(num_segments, segments_per_rank)
        start_ranks = initial_ranks
        end_ranks = initial_ranks
        recompute_segments = False

        if self.config.fixed_sp_size > 0:
            forced = min(
                self.config.fixed_sp_size, max(1, prompt_tokens)
            )
            start_ranks = end_ranks = forced
            recompute_segments = True
        elif self.config.dynamic_sp_size_strategy == "bucket":
            forced = self._bucket_sp_size(prompt_tokens, initial_ranks)
            start_ranks = end_ranks = forced
            recompute_segments = True
        nonmasters = self._richest_nonmasters(shadow, master)
        for target_ranks in range(start_ranks, end_ranks + 1):
            target_segments = segments_per_rank
            if recompute_segments:
                target_segments = self._ceil_div(
                    num_segments, target_ranks
                )
            participants = nonmasters[: target_ranks - 1] + [master]
            if self.config.enable_non_uniform_split and (
                self.config.fixed_sp_size == 0
            ):
                dispatched = self._split_by_free_capacity(
                    shadow, participants, prompt_tokens
                )
            elif self.config.fixed_sp_size > 0:
                dispatched = self._split_uniform(
                    participants, prompt_tokens
                )
            else:
                dispatched = [0] * self.config.attention_sp
                remaining = prompt_tokens
                per_rank_tokens = (
                    target_segments * self.config.segment_size
                )
                for sp_idx in participants:
                    assigned = min(remaining, per_rank_tokens)
                    dispatched[sp_idx] = assigned
                    remaining -= assigned
            reservation = self._check_legacy_placement(
                shadow,
                command.request_id,
                master,
                dispatched,
            )
            if reservation is not None:
                return reservation
        return None

    def _check_legacy_placement(
        self,
        shadow: AdmissionShadow,
        request_id: int,
        master: int,
        dispatched: list[int],
    ) -> AdmissionReservation | None:
        block_size = self.config.kvcache_block_size
        # Earlier reservations in this frontend batch are already reflected
        # in master_counts by apply_reservation().
        projected_masters = [
            shadow.master_counts[sp_idx]
            + (1 if sp_idx == master else 0)
            for sp_idx in range(self.config.attention_sp)
        ]
        if sum(dispatched) <= 0:
            return None
        for sp_idx, token_count in enumerate(dispatched):
            if token_count <= 0 and sp_idx != master:
                continue
            if (
                self.config.fixed_sp_size == 0
                and sp_idx != master
                and token_count > 0
                and shadow.receiver_counts[sp_idx]
                >= self.config.max_num_recv_seqs
            ):
                return None
            prefill_blocks = self._ceil_div(token_count, block_size)
            reservation_blocks = ceil(
                projected_masters[sp_idx]
                * self.config.reserved_blocks_per_req
            )
            if (
                shadow.free_blocks[sp_idx]
                < prefill_blocks + reservation_blocks
            ):
                return None
        return AdmissionReservation(
            request_id=request_id,
            engine_id=shadow.engine_id,
            master_sp_idx=master,
            dispatched_tokens=tuple(dispatched),
        )

    def _plan_balanced(
        self,
        shadow: AdmissionShadow,
        command: AddCommand,
    ) -> AdmissionReservation | None:
        prompt_tokens = len(command.prompt_token_ids)
        allowed = [1]
        size = 2
        while size <= self.config.attention_sp:
            allowed.append(size)
            size *= 2
        if allowed[-1] != self.config.attention_sp:
            allowed.append(self.config.attention_sp)
        if self.config.fixed_sp_size > 0:
            allowed = [self.config.fixed_sp_size]

        master = self._select_master(shadow)
        if shadow.master_counts[master] + 1 > self.config.max_num_seqs:
            return None
        if (
            shadow.batch_tokens[master] + prompt_tokens
            >= self.config.max_num_batched_tokens
        ):
            return None
        others = sorted(
            (sp for sp in range(self.config.attention_sp) if sp != master),
            key=lambda sp: (
                shadow.dispatched_tokens[sp],
                -shadow.free_blocks[sp],
                sp,
            ),
        )
        for target_ranks in allowed:
            target_ranks = min(target_ranks, max(1, prompt_tokens))
            participants = [master] + others[: target_ranks - 1]
            dispatched = self._waterfill_tokens(
                shadow, participants, prompt_tokens
            )
            reservation = self._check_balanced_placement(
                shadow,
                command.request_id,
                master,
                dispatched,
            )
            if reservation is not None:
                return reservation
        return None

    def _check_balanced_placement(
        self,
        shadow: AdmissionShadow,
        request_id: int,
        master: int,
        dispatched: list[int],
    ) -> AdmissionReservation | None:
        block_size = self.config.kvcache_block_size
        for sp_idx, token_count in enumerate(dispatched):
            if token_count <= 0:
                continue
            if (
                self.config.fixed_sp_size == 0
                and sp_idx != master
                and shadow.receiver_counts[sp_idx] + 1
                > self.config.max_num_recv_seqs
            ):
                return None
            prefill_blocks = self._ceil_div(token_count, block_size)
            projected_masters = shadow.master_counts[sp_idx] + (
                1 if sp_idx == master else 0
            )
            reservation_blocks = ceil(
                projected_masters
                * self.config.reserved_blocks_per_req
            )
            if (
                shadow.free_blocks[sp_idx]
                < prefill_blocks + reservation_blocks
            ):
                return None
        return AdmissionReservation(
            request_id=request_id,
            engine_id=shadow.engine_id,
            master_sp_idx=master,
            dispatched_tokens=tuple(dispatched),
        )

    def _select_master(self, shadow: AdmissionShadow) -> int:
        selector = self.config.sp_master_selector
        if selector == "RoundRobin":
            return shadow.rr_cursor
        if selector == "LeastCache":
            return min(
                range(self.config.attention_sp),
                key=lambda sp: (-shadow.free_blocks[sp], sp),
            )
        return min(
            range(self.config.attention_sp),
            key=lambda sp: (shadow.master_counts[sp], sp),
        )

    def _richest_nonmasters(
        self, shadow: AdmissionShadow, master: int
    ) -> list[int]:
        return sorted(
            (sp for sp in range(self.config.attention_sp) if sp != master),
            key=lambda sp: (-shadow.free_blocks[sp], sp),
        )

    def _split_by_free_capacity(
        self,
        shadow: AdmissionShadow,
        participants: list[int],
        token_count: int,
    ) -> list[int]:
        block_size = self.config.kvcache_block_size
        ranked = sorted(
            participants,
            key=lambda sp: (-shadow.free_blocks[sp], sp),
        )
        target_free = 0
        contributing = len(ranked)
        for count in range(1, len(ranked) + 1):
            total_free = sum(
                shadow.free_blocks[sp] * block_size
                for sp in ranked[:count]
            )
            candidate = (total_free - token_count) // count
            if count == len(ranked) or (
                candidate >= shadow.free_blocks[ranked[count]] * block_size
            ):
                target_free = candidate
                contributing = count
                break
        dispatched = [0] * self.config.attention_sp
        for sp_idx in ranked:
            free_tokens = shadow.free_blocks[sp_idx] * block_size
            dispatched[sp_idx] = max(
                0, min(free_tokens, free_tokens - target_free)
            )
        remainder = token_count - sum(dispatched)
        index = 0
        while remainder > 0:
            dispatched[ranked[index]] += 1
            remainder -= 1
            index = (index + 1) % contributing
        while remainder < 0:
            sp_idx = ranked[index]
            if dispatched[sp_idx] > 0:
                dispatched[sp_idx] -= 1
                remainder += 1
            index = (index + 1) % contributing
        return dispatched

    def _split_uniform(
        self, participants: list[int], token_count: int
    ) -> list[int]:
        dispatched = [0] * self.config.attention_sp
        base, extra = divmod(token_count, len(participants))
        for index, sp_idx in enumerate(participants):
            dispatched[sp_idx] = base + (1 if index < extra else 0)
        return dispatched

    def _waterfill_tokens(
        self,
        shadow: AdmissionShadow,
        participants: list[int],
        token_count: int,
    ) -> list[int]:
        dispatched = [0] * self.config.attention_sp
        if token_count < len(participants):
            return dispatched
        loads = [shadow.dispatched_tokens[sp] for sp in participants]
        low = min(loads) + 1
        high = max(loads) + token_count
        best = low
        while low <= high:
            middle = (low + high) // 2
            needed = sum(max(1, middle - load) for load in loads)
            if needed <= token_count:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        allocations = [max(1, best - load) for load in loads]
        remaining = token_count - sum(allocations)
        order = sorted(
            range(len(participants)),
            key=lambda index: (
                loads[index] + allocations[index],
                participants[index],
            ),
        )
        index = 0
        while remaining > 0:
            allocations[order[index]] += 1
            remaining -= 1
            index = (index + 1) % len(order)
        for index, sp_idx in enumerate(participants):
            dispatched[sp_idx] = allocations[index]
        return dispatched

    def _fits_lifetime(
        self,
        reservation: AdmissionReservation,
        max_tokens: int,
        shadow: AdmissionShadow,
    ) -> bool:
        block_size = self.config.kvcache_block_size
        master = reservation.master_sp_idx
        padded_completion = round_up(max_tokens)
        for sp_idx, prompt_tokens in enumerate(
            reservation.dispatched_tokens
        ):
            required_tokens = prompt_tokens
            if sp_idx == master:
                required_tokens += 1 + padded_completion
            required_blocks = self._ceil_div(
                required_tokens, block_size
            )
            service_blocks = (
                shadow.total_blocks[sp_idx]
                - shadow.control_dummy_blocks[sp_idx]
            )
            if required_blocks > service_blocks:
                return False
        return True

    def _parse_bucket_policy(
        self, text: str
    ) -> tuple[tuple[int, int, int], ...]:
        intervals = []
        for item in text.split(";"):
            item = item.strip()
            if not item:
                continue
            size_text, limits = item.split(":", 1)
            low_text, high_text = limits.split("-", 1)
            intervals.append(
                (int(size_text), int(low_text), int(high_text))
            )
        return tuple(intervals)

    def _bucket_sp_size(
        self, token_count: int, fallback: int
    ) -> int:
        for size, low, high in self._bucket_policy:
            if low <= token_count <= high:
                return min(size, self.config.attention_sp)
        return fallback

    @staticmethod
    def _ceil_div(value: int, divisor: int) -> int:
        return (value + divisor - 1) // divisor
