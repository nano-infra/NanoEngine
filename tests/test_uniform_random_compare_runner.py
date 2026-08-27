import os
from pathlib import Path
import subprocess


def test_uniform_random_compare_runner_dry_run(tmp_path):
    root = Path(__file__).resolve().parents[1]
    runner = root / "scripts" / "run_2node_r50_uniform_random_compare.sh"
    log_dir = tmp_path / "compare"
    result = subprocess.run(
        ["bash", str(runner)],
        cwd=root,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "COMPARE_LOG_DIR": str(log_dir),
            "RAY_ADDR": "127.0.0.1:6379",
            "MASTER_ADDR": "127.0.0.1:29500",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    progress = (log_dir / "compare.progress").read_text(encoding="utf-8")
    assert "RATE=50 DURATION_SECONDS=300 NUM_REQUESTS=15000" in progress
    assert "ROUNDS=2 EXPERIMENTS_PER_ROUND=4 TOTAL_EXPERIMENTS=8" in progress
    assert "MOE_ROUTING_SIMULATION_STRATEGY=uniform_random" in progress
    assert "MOE_ROUTING_SEED=0" in progress
    assert progress.count("--moe-routing-simulation-strategy uniform_random") == 8
    assert progress.count("--moe-routing-seed 0") == 8

    ordered_stages = []
    for round_index in (1, 2):
        assert f"ROUND_START round={round_index}/2" in progress
        ordered_stages.extend(
            [
                f"STAGE=round{round_index}_central_least_batch "
                "SCHEDULER_ARCH=legacy_global ROUTING_STRATEGY=LeastBatch "
                "ROUTER_POLICY=least_batch WORKER_TRANSPORT=ray",
                f"STAGE=round{round_index}_hierarchical_least_batch "
                "SCHEDULER_ARCH=hierarchical ROUTING_STRATEGY=LeastBatch "
                "ROUTER_POLICY=least_batch WORKER_TRANSPORT=zmq",
                f"STAGE=round{round_index}_central_least_projected_load "
                "SCHEDULER_ARCH=legacy_global "
                "ROUTING_STRATEGY=LeastProjectedLoad "
                "ROUTER_POLICY=least_projected_load WORKER_TRANSPORT=ray",
                f"STAGE=round{round_index}_hierarchical_least_projected_load "
                "SCHEDULER_ARCH=hierarchical ROUTING_STRATEGY=LeastBatch "
                "ROUTER_POLICY=least_projected_load WORKER_TRANSPORT=zmq",
            ]
        )
        assert f"ROUND_DONE round={round_index}/2" in progress

    stage_offsets = [progress.index(stage) for stage in ordered_stages]
    assert stage_offsets == sorted(stage_offsets)


def test_uniform_random_compare_runner_rejects_nonpositive_rounds(tmp_path):
    root = Path(__file__).resolve().parents[1]
    runner = root / "scripts" / "run_2node_r50_uniform_random_compare.sh"
    result = subprocess.run(
        ["bash", str(runner)],
        cwd=root,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "ROUNDS": "0",
            "COMPARE_LOG_DIR": str(tmp_path / "compare"),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "ROUNDS must be a positive integer" in result.stderr
