from pathlib import Path
import subprocess


def test_sp_ablation_launcher_uses_python3():
    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "sp_ablation"
        / "start_bench.sh"
    ).read_text(encoding="utf-8")

    assert 'python3 -u "$PYTHON_SCRIPT"' in launcher
    assert 'python -u "$PYTHON_SCRIPT"' not in launcher


def test_sp_ablation_launcher_forwards_moe_routing_configuration():
    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "sp_ablation"
        / "start_bench.sh"
    ).read_text(encoding="utf-8")

    assert 'DEFAULT_MOE_ROUTING_SIMULATION_STRATEGY="perfect_eplb"' in launcher
    assert (
        '--moe-routing-simulation-strategy '
        '"$MOE_ROUTING_SIMULATION_STRATEGY"'
    ) in launcher
    assert '--moe-routing-seed "$MOE_ROUTING_SEED"' in launcher


def test_sp_ablation_launcher_rejects_invalid_moe_routing_options():
    root = Path(__file__).resolve().parents[1]
    launcher = root / "scripts" / "sp_ablation" / "start_bench.sh"
    invalid_cases = (
        (
            ["--moe-routing-simulation-strategy", "invalid", "1"],
            "Invalid MoE routing simulation strategy",
        ),
        (
            ["--moe-routing-seed", "not-an-int", "1"],
            "--moe-routing-seed must be an integer",
        ),
    )

    for arguments, expected_error in invalid_cases:
        result = subprocess.run(
            ["bash", str(launcher), *arguments],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert expected_error in result.stdout
