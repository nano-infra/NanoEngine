from __future__ import annotations

import dataclasses
from collections import Counter, defaultdict
from typing import Any

from nanodeploy.worker.kv_p2p import (
    KVCacheP2PMove,
    KVCacheP2PResult,
)


class KVScaleDownRejected(RuntimeError):
    """The scheduler could not reserve a safe source-evacuation plan."""


class KVScaleDownPreDispatchAborted(KVScaleDownRejected):
    """A RESERVED plan failed before RPC dispatch and was safely aborted."""


class KVScaleDownAcknowledgementError(RuntimeError):
    """Worker acknowledgements cannot prove that the physical copy completed."""


@dataclasses.dataclass(frozen=True, slots=True)
class KVScaleDownResult:
    transaction_id: int
    group_id: int
    dp_idx: int
    source_rank: int
    retained_ranks: tuple[int, ...]
    num_tokens: int
    num_moves: int
    worker_results: tuple[KVCacheP2PResult, ...]


def execute_ls_kv_scale_down(
    scheduler: Any,
    executor: Any,
    *,
    group_id: int,
    source_rank: int,
    timeout: float | None = None,
) -> KVScaleDownResult:
    """Run one stop-the-world LS Decode source-rank evacuation.

    The scheduler reserves destination blocks and returns physical ranges while
    keeping ACTIVE metadata unchanged. Every worker then enters the P2P RPC.
    Metadata becomes visible and source blocks are released only after all
    workers report copy completion. Once the first RPC is dispatched, a worker
    exception is completion-ambiguous: the scheduler latches fatal state and
    retains the transaction guard instead of attempting rollback.

    The caller must invoke this at an engine iteration boundary, with no model
    forward or EP/SP collective in flight.
    """

    plan = scheduler.plan_ls_kv_scale_down(group_id, source_rank)
    return execute_planned_ls_kv_scale_down(
        scheduler,
        executor,
        plan,
        timeout=timeout,
    )


def execute_planned_ls_kv_scale_down(
    scheduler: Any,
    executor: Any,
    plan: Any,
    *,
    timeout: float | None = None,
    abort_on_copy_error: bool = True,
) -> KVScaleDownResult:
    """Execute one scheduler-owned, already-reserved scale-down plan.

    Automatic scheduling must call this coordinator directly. Re-planning by
    group/source would leak the original reservation and could select a plan
    that no longer matches the scheduler decision returned for this step.
    ``abort_on_copy_error`` is retained for source compatibility but no longer
    permits rollback after dispatch. From the first worker RPC onward copy
    completion is potentially ambiguous, so every failure is fatal and the
    scheduler keeps the DISPATCHED transaction guard.
    """

    if not plan.success:
        raise KVScaleDownRejected(plan.failure_reason)

    del abort_on_copy_error

    try:
        moves = [
            KVCacheP2PMove(
                dp_idx=move.dp_idx,
                src_sp_rank=move.src_sp_rank,
                dst_sp_rank=move.dst_sp_rank,
                src_block_id=move.src_block_id,
                src_token_offset=move.src_token_offset,
                dst_block_id=move.dst_block_id,
                dst_token_offset=move.dst_token_offset,
                num_tokens=move.num_tokens,
            )
            for move in plan.moves
        ]
        _validate_materialized_moves(plan, moves)
    except BaseException as error:
        # No worker RPC has been invoked, so the exact prepared reservation is
        # still safe to abort.
        _abort_pre_dispatch(scheduler, plan)
        raise KVScaleDownPreDispatchAborted(
            "LS KV scale-down move materialization failed before dispatch"
        ) from error

    if not scheduler.mark_ls_kv_scale_down_dispatched(plan):
        _abort_pre_dispatch(scheduler, plan)
        raise KVScaleDownPreDispatchAborted(
            "LS KV scale-down plan could not enter DISPATCHED state"
        )
    state_name = _plan_state_name(plan)
    if state_name != "DISPATCHED":
        if state_name == "RESERVED":
            _abort_pre_dispatch(scheduler, plan)
            raise KVScaleDownPreDispatchAborted(
                "LS KV scale-down dispatch marker did not publish DISPATCHED state"
            )
        raise RuntimeError(
            "LS KV scale-down dispatch marker returned an unexpected plan state: "
            f"{state_name}"
        )

    try:
        worker_results = executor.copy_kv_ranges_p2p(moves, timeout=timeout)
    except BaseException:
        _latch_dispatched_failure(scheduler)
        raise

    try:
        worker_results = _validate_worker_results(moves, worker_results, executor)
    except BaseException as error:
        _latch_dispatched_failure(scheduler)
        if isinstance(error, KVScaleDownAcknowledgementError):
            raise
        raise KVScaleDownAcknowledgementError(
            "could not validate LS KV scale-down worker acknowledgements"
        ) from error

    try:
        committed = scheduler.commit_ls_kv_scale_down(plan)
    except BaseException:
        # COMMIT may already have swapped ACTIVE metadata. Its exception is
        # engine-fatal; calling ABORT here could release reachable blocks.
        _latch_dispatched_failure(scheduler)
        raise
    if not committed:
        _latch_dispatched_failure(scheduler)
        raise RuntimeError("dispatched LS KV scale-down plan failed to commit")

    return KVScaleDownResult(
        transaction_id=plan.transaction_id,
        group_id=plan.group_id,
        dp_idx=plan.dp_idx,
        source_rank=plan.source_rank,
        retained_ranks=tuple(plan.retained_ranks),
        num_tokens=plan.num_tokens,
        num_moves=len(moves),
        worker_results=tuple(worker_results),
    )


def _validate_worker_results(
    moves: list[KVCacheP2PMove],
    worker_results: Any,
    executor: Any,
) -> tuple[KVCacheP2PResult, ...]:
    """Prove full worker participation and copy coverage before metadata commit."""

    if not moves:
        raise KVScaleDownAcknowledgementError(
            "a dispatched LS KV scale-down plan must contain at least one move"
        )

    target_dp = moves[0].dp_idx
    source_rank = moves[0].src_sp_rank
    destinations: set[int] = set()
    for move in moves:
        if (
            move.dp_idx != target_dp
            or move.src_sp_rank != source_rank
            or move.dst_sp_rank == source_rank
            or move.num_tokens <= 0
        ):
            raise KVScaleDownAcknowledgementError(
                "copy plan does not describe one non-empty source evacuation"
            )
        destinations.add(move.dst_sp_rank)

    chunk_tokens = _executor_chunk_tokens(executor)
    expected_by_destination = _expected_transport_coverage(moves, chunk_tokens)
    expected_source_moves = sum(item[0] for item in expected_by_destination.values())
    expected_source_chunks = sum(item[1] for item in expected_by_destination.values())
    expected_source_tokens = sum(item[2] for item in expected_by_destination.values())

    try:
        results = tuple(worker_results)
    except BaseException as error:
        raise KVScaleDownAcknowledgementError(
            "worker acknowledgement collection is not iterable"
        ) from error

    for result in results:
        if any(
            value < 0
            for value in (
                result.dp_idx,
                result.sp_rank,
                result.num_moves,
                result.num_chunks,
                result.sent_bytes,
                result.received_bytes,
            )
        ):
            raise KVScaleDownAcknowledgementError(
                "worker acknowledgement counters must be non-negative"
            )
        if result.role not in {"idle", "source", "destination"}:
            raise KVScaleDownAcknowledgementError(
                f"worker acknowledgement has invalid role {result.role!r}"
            )

    topology = _executor_topology(executor)
    actual_census = Counter((result.dp_idx, result.sp_rank) for result in results)
    attention_dp, attention_sp, attention_tp = topology
    expected_census = Counter(
        {
            (dp_idx, sp_rank): attention_tp
            for dp_idx in range(attention_dp)
            for sp_rank in range(attention_sp)
        }
    )
    if actual_census != expected_census:
        raise KVScaleDownAcknowledgementError(
            "worker acknowledgement census does not match the executor topology"
        )

    token_bytes_per_worker = []
    for result in results:
        key = (result.dp_idx, result.sp_rank)
        if result.dp_idx != target_dp or result.sp_rank not in {
            source_rank,
            *destinations,
        }:
            expected_role = "idle"
        elif result.sp_rank == source_rank:
            expected_role = "source"
        else:
            expected_role = "destination"

        if result.role != expected_role:
            raise KVScaleDownAcknowledgementError(
                f"worker {key} reported role {result.role!r}, expected {expected_role!r}"
            )

        if expected_role == "idle":
            if any(
                (
                    result.num_moves,
                    result.num_chunks,
                    result.sent_bytes,
                    result.received_bytes,
                )
            ):
                raise KVScaleDownAcknowledgementError(
                    f"idle worker {key} reported physical copy activity"
                )
            continue

        if expected_role == "source":
            expected_moves = expected_source_moves
            expected_chunks = expected_source_chunks
            expected_tokens = expected_source_tokens
            transferred_bytes = result.sent_bytes
            opposite_bytes = result.received_bytes
        else:
            expected_moves, expected_chunks, expected_tokens = (
                expected_by_destination[result.sp_rank]
            )
            transferred_bytes = result.received_bytes
            opposite_bytes = result.sent_bytes

        if (
            result.num_moves != expected_moves
            or result.num_chunks != expected_chunks
            or opposite_bytes != 0
            or transferred_bytes <= 0
            or transferred_bytes % expected_tokens != 0
        ):
            raise KVScaleDownAcknowledgementError(
                f"worker {key} acknowledgement does not cover its planned moves/chunks"
            )
        token_bytes_per_worker.append(transferred_bytes // expected_tokens)

    if not token_bytes_per_worker or len(set(token_bytes_per_worker)) != 1:
        raise KVScaleDownAcknowledgementError(
            "worker acknowledgements disagree on physical bytes per KV token"
        )
    sent_bytes = sum(result.sent_bytes for result in results)
    received_bytes = sum(result.received_bytes for result in results)
    if sent_bytes <= 0 or sent_bytes != received_bytes:
        raise KVScaleDownAcknowledgementError(
            "worker acknowledgement bytes are not conserved"
        )

    return results


def _validate_materialized_moves(plan: Any, moves: list[KVCacheP2PMove]) -> None:
    """Match the physical RPC payload exactly to immutable scheduler intent."""

    retained_ranks = tuple(plan.retained_ranks)
    if (
        not moves
        or not retained_ranks
        or any(rank < 0 for rank in retained_ranks)
        or plan.dp_idx < 0
        or plan.source_rank < 0
        or plan.source_rank in retained_ranks
        or len(retained_ranks) != len(set(retained_ranks))
        or plan.num_tokens <= 0
    ):
        raise ValueError("reserved LS KV scale-down plan is empty or inconsistent")
    if any(
        move.dp_idx != plan.dp_idx
        or move.src_sp_rank != plan.source_rank
        or move.dst_sp_rank == plan.source_rank
        or move.dst_sp_rank not in retained_ranks
        or move.src_block_id < 0
        or move.src_token_offset < 0
        or move.dst_block_id < 0
        or move.dst_token_offset < 0
        or move.num_tokens <= 0
        for move in moves
    ):
        raise ValueError("materialized KV moves differ from the reserved DP/rank plan")
    if sum(move.num_tokens for move in moves) != plan.num_tokens:
        raise ValueError("materialized KV move token coverage differs from the plan")
    _validate_physical_ranges_do_not_overlap(moves, source=True)
    _validate_physical_ranges_do_not_overlap(moves, source=False)


def _validate_physical_ranges_do_not_overlap(
    moves: list[KVCacheP2PMove], *, source: bool
) -> None:
    ranges: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for move in moves:
        if source:
            key = (move.src_sp_rank, move.src_block_id)
            begin = move.src_token_offset
        else:
            key = (move.dst_sp_rank, move.dst_block_id)
            begin = move.dst_token_offset
        ranges[key].append((begin, begin + move.num_tokens))

    side = "source" if source else "destination"
    for intervals in ranges.values():
        intervals.sort()
        if any(current[0] < previous[1] for previous, current in zip(intervals, intervals[1:])):
            raise ValueError(f"materialized KV moves contain overlapping {side} ranges")


def _executor_topology(executor: Any) -> tuple[int, int, int]:
    return (
        executor.config.attention_dp,
        executor.config.attention_sp,
        executor.config.attention_tp,
    )


def _executor_chunk_tokens(executor: Any) -> int:
    chunk_tokens = executor.config.ls_kv_consolidation_migration_chunk_tokens
    if chunk_tokens <= 0:
        raise KVScaleDownAcknowledgementError(
            "executor does not expose a positive KV P2P chunk size"
        )
    return chunk_tokens


def _expected_transport_coverage(
    moves: list[KVCacheP2PMove], chunk_tokens: int
) -> dict[int, tuple[int, int, int]]:
    split_lengths_by_destination: dict[int, list[int]] = defaultdict(list)
    normalized = sorted(
        moves,
        key=lambda move: (
            move.dp_idx,
            move.src_sp_rank,
            move.dst_sp_rank,
            move.src_block_id,
            move.src_token_offset,
            move.dst_block_id,
            move.dst_token_offset,
        ),
    )
    for move in normalized:
        remaining = move.num_tokens
        while remaining > 0:
            length = min(chunk_tokens, remaining)
            split_lengths_by_destination[move.dst_sp_rank].append(length)
            remaining -= length

    coverage = {}
    for destination, lengths in split_lengths_by_destination.items():
        num_chunks = 0
        current_chunk_tokens = 0
        for length in lengths:
            if current_chunk_tokens and current_chunk_tokens + length > chunk_tokens:
                num_chunks += 1
                current_chunk_tokens = 0
            current_chunk_tokens += length
        if current_chunk_tokens:
            num_chunks += 1
        coverage[destination] = (len(lengths), num_chunks, sum(lengths))
    return coverage


def _abort_pre_dispatch(scheduler: Any, plan: Any) -> None:
    """Abort RESERVED state and require positive proof of the safe terminal state."""

    scheduler.abort_ls_kv_scale_down(plan)
    if _plan_state_name(plan) != "ABORTED":
        raise RuntimeError(
            "LS KV scale-down abort did not confirm the ABORTED terminal state"
        )


def _plan_state_name(plan: Any) -> str:
    return plan.state.name


def _latch_dispatched_failure(scheduler: Any) -> None:
    """Latch fatal state without attempting an unsafe post-dispatch abort."""

    from nanodeploy._cpp import LSFatalCode

    scheduler.latch_ls_fatal(LSFatalCode.KV_CONSOLIDATION_FAILED)
