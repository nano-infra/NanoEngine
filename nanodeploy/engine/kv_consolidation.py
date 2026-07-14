from __future__ import annotations

import dataclasses
from typing import Any

from nanodeploy.worker.kv_p2p import (
    KVCacheP2PMove,
    KVCacheP2PResult,
)


class KVScaleDownRejected(RuntimeError):
    """The scheduler could not reserve a safe source-evacuation plan."""


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
    workers report copy completion. A worker-side exception aborts the
    reservation, leaving source placement valid.

    The caller must invoke this at an engine iteration boundary, with no model
    forward or EP/SP collective in flight.
    """

    plan = scheduler.plan_ls_kv_scale_down(group_id, source_rank)
    if not plan.success:
        raise KVScaleDownRejected(plan.failure_reason)

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

    try:
        worker_results = executor.copy_kv_ranges_p2p(moves, timeout=timeout)
    except BaseException:
        scheduler.abort_ls_kv_scale_down(plan)
        raise

    try:
        committed = scheduler.commit_ls_kv_scale_down(plan)
    except BaseException:
        # COMMIT may already have swapped ACTIVE metadata. Its exception is
        # engine-fatal; calling ABORT here could release reachable blocks.
        raise
    if not committed:
        scheduler.abort_ls_kv_scale_down(plan)
        raise RuntimeError("LS KV scale-down plan became stale before commit")

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
