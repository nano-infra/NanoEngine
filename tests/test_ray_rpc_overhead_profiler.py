from __future__ import annotations

import pickle

import pytest

from scripts.ray_rpc_overhead import profile_ray_rpc_scalability as profiler
from scripts.ray_rpc_overhead.profile_ray_rpc_scalability import (
    _estimated_pickle_bytes,
    _parse_positive_ints,
    _percentile,
    _sample_token_rows,
    _stats,
)


def test_parse_positive_ints() -> None:
    assert _parse_positive_ints("32,64,128") == (32, 64, 128)
    with pytest.raises(Exception):
        _parse_positive_ints("32,0")
    with pytest.raises(Exception):
        _parse_positive_ints("32,32")


def test_percentile_and_stats() -> None:
    samples = [1.0, 2.0, 3.0, 4.0]
    assert _percentile(samples, 50.0) == 2.5
    stats = _stats(samples)
    assert stats.mean_ms == 2.5
    assert stats.min_ms == 1.0
    assert stats.max_ms == 4.0


def test_sample_token_rows_match_decode_shape() -> None:
    rows = _sample_token_rows(3, batch_size_per_gpu=4, loop_count=16)
    assert len(rows) == 4
    assert all(len(row) == 16 for row in rows)
    assert rows[0][0] == 3


def test_estimated_pickle_bytes_match_constructed_payloads() -> None:
    input_bytes, output_bytes = _estimated_pickle_bytes(4, 16)
    expected_input = len(
        pickle.dumps(([], False, True, 0.0), protocol=pickle.HIGHEST_PROTOCOL)
    )
    assert input_bytes == expected_input
    assert output_bytes > input_bytes


def test_profiler_disables_ray_uv_runtime_environment_detection() -> None:
    from ray._private import ray_constants

    assert profiler.os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] == "0"
    assert ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV is False


def test_main_skips_ray_runtime_environment_hook(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class ExpectedInit(Exception):
        pass

    init_kwargs = {}

    def capture_init(**kwargs):
        init_kwargs.update(kwargs)
        raise ExpectedInit

    monkeypatch.setattr(profiler.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(profiler.ray, "init", capture_init)

    with pytest.raises(ExpectedInit):
        profiler.main(
            [
                "--logical-workers",
                "2",
                "--batch-sizes",
                "2",
                "--warmup-iterations",
                "0",
                "--iterations",
                "1",
                "--output-dir",
                str(tmp_path / "results"),
            ]
        )

    assert init_kwargs["address"] == "local"
    assert init_kwargs["_skip_env_hook"] is True
