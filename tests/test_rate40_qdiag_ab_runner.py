import os
from pathlib import Path
import subprocess


def test_rate40_qdiag_runner_smoke_dry_run(tmp_path):
    root = Path(__file__).resolve().parents[1]
    runner = root / "scripts" / "run_2node_rate40_qdiag_ab.sh"
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "SMOKE": "1",
        "CHAIN_LOG_DIR": str(tmp_path / "logs"),
        "RAY_ADDR": "127.0.0.1:6379",
        "MASTER_ADDR": "127.0.0.1:29500",
    }

    result = subprocess.run(
        ["bash", str(runner)],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    progress = (tmp_path / "logs" / "ab.progress").read_text(
        encoding="utf-8"
    )
    assert "RUN_ORDER=central hierarchical" in progress
    assert "DURATION_SECONDS=30 NUM_REQUESTS=1200" in progress
    assert progress.count("--quantum-diagnostics") == 2
    assert "--scheduler-arch legacy_global" in progress
    assert "--scheduler-arch hierarchical" in progress
    manifest = (tmp_path / "logs" / "run_manifest.tsv").read_text(
        encoding="utf-8"
    )
    assert "01_central" in manifest
    assert "02_hierarchical" in manifest


def test_rate40_qdiag_runner_rejects_unknown_stage(tmp_path):
    root = Path(__file__).resolve().parents[1]
    runner = root / "scripts" / "run_2node_rate40_qdiag_ab.sh"
    result = subprocess.run(
        ["bash", str(runner)],
        cwd=root,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "RUN_ORDER": "central unknown",
            "CHAIN_LOG_DIR": str(tmp_path / "logs"),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "unknown stage 'unknown'" in result.stderr
