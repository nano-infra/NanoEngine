from nanodeploy.engine.model_forward_timing import (
    summarize_model_forward_gpu_timings,
)


def test_summarize_model_forward_gpu_timings_uses_per_loop_slowest_rank():
    summary = summarize_model_forward_gpu_timings(
        [
            [10.0, 13.0],
            [12.0, 11.0],
            [11.0, 12.0],
        ]
    )

    assert summary is not None
    assert summary["critical_path_ms"] == 25.0
    assert summary["per_loop_ms"] == 12.5
    assert summary["rank_mean_total_ms"] == 23.0
    assert summary["rank_min_total_ms"] == 23.0
    assert summary["rank_max_total_ms"] == 23.0
    assert summary["loop_count"] == 2


def test_summarize_model_forward_gpu_timings_rejects_inconsistent_loops():
    assert summarize_model_forward_gpu_timings([[10.0], [11.0, 12.0]]) is None


def test_summarize_model_forward_gpu_timings_rejects_empty_input():
    assert summarize_model_forward_gpu_timings([]) is None
    assert summarize_model_forward_gpu_timings([[]]) is None
