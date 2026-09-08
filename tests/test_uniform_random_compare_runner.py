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
    assert "QUANTUM_DIAGNOSTICS=1 RUN_PROJECTED_LOAD=1" in progress
    assert progress.count("--moe-routing-simulation-strategy uniform_random") == 8
    assert progress.count("--moe-routing-seed 0") == 8
    assert progress.count("--quantum-diagnostics") == 8

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


def test_uniform_random_compare_runner_can_disable_hot_path_logs(tmp_path):
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
            "QUANTUM_DIAGNOSTICS": "0",
            "NANODEPLOY_LOG_LEVEL": "WARNING",
            "NANODEPLOY_LOG_DECODE_STEP_DETAIL": "0",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    progress = (log_dir / "compare.progress").read_text(encoding="utf-8")
    assert (
        "QUANTUM_DIAGNOSTICS=0 RUN_PROJECTED_LOAD=1 "
        "NANODEPLOY_LOG_LEVEL=WARNING "
        "NANODEPLOY_LOG_DECODE_STEP_DETAIL=0"
    ) in progress
    assert "--quantum-diagnostics" not in progress
    assert progress.count("NANODEPLOY_LOG_LEVEL=WARNING") == 9
    assert progress.count("NANODEPLOY_LOG_DECODE_STEP_DETAIL=0") == 9


def test_uniform_random_compare_runner_can_run_least_batch_only(tmp_path):
    root = Path(__file__).resolve().parents[1]
    runner = root / "scripts" / "run_2node_r50_uniform_random_compare.sh"
    log_dir = tmp_path / "compare"
    result = subprocess.run(
        ["bash", str(runner)],
        cwd=root,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "ROUNDS": "1",
            "RUN_PROJECTED_LOAD": "0",
            "COMPARE_LOG_DIR": str(log_dir),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    progress = (log_dir / "compare.progress").read_text(encoding="utf-8")
    assert "ROUNDS=1 EXPERIMENTS_PER_ROUND=2 TOTAL_EXPERIMENTS=2" in progress
    assert "RUN_PROJECTED_LOAD=0" in progress
    assert progress.count("STAGE=") == 2
    assert "STAGE=round1_central_least_batch" in progress
    assert "STAGE=round1_hierarchical_least_batch" in progress
    assert "least_projected_load" not in progress
