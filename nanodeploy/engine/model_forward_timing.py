def summarize_model_forward_gpu_timings(
    timings_by_rank: list[list[float]],
) -> dict[str, float | int] | None:
    """Summarize worker CUDA-event timings on the distributed critical path."""
    if not timings_by_rank or not timings_by_rank[0]:
        return None

    loop_count = len(timings_by_rank[0])
    if any(len(rank_timings) != loop_count for rank_timings in timings_by_rank):
        return None

    # Every loop is a distributed forward, so its wall-critical duration is
    # represented by the slowest rank for that loop. The slowest rank can vary.
    critical_path_ms = sum(
        max(rank_timings[loop_idx] for rank_timings in timings_by_rank)
        for loop_idx in range(loop_count)
    )
    rank_total_ms = [sum(rank_timings) for rank_timings in timings_by_rank]
    return {
        "critical_path_ms": critical_path_ms,
        "per_loop_ms": critical_path_ms / loop_count,
        "rank_mean_total_ms": sum(rank_total_ms) / len(rank_total_ms),
        "rank_min_total_ms": min(rank_total_ms),
        "rank_max_total_ms": max(rank_total_ms),
        "loop_count": loop_count,
    }
