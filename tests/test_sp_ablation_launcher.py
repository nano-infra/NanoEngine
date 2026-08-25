from pathlib import Path


def test_sp_ablation_launcher_uses_python3():
    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "sp_ablation"
        / "start_bench.sh"
    ).read_text(encoding="utf-8")

    assert 'python3 -u "$PYTHON_SCRIPT"' in launcher
    assert 'python -u "$PYTHON_SCRIPT"' not in launcher
