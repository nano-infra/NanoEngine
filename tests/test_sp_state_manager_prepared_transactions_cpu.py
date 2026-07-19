import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
CPP_TEST = ROOT / "tests" / "cpp" / "test_sp_state_manager_prepared_transactions.cpp"
TEST_CASES = [
    "initial_abort_restores_exact_state",
    "initial_commit_and_prepared_release",
    "raii_move_and_stale_validation",
    "disjoint_initial_prepare_abort_then_commit",
]


@pytest.fixture(scope="session")
def sp_state_manager_prepared_transactions_binary(tmp_path_factory):
    compiler_command = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler_command or shutil.which(compiler_command[0]) is None:
        pytest.skip("a C++ compiler is required for the SPStateManager CPU harness")

    output = tmp_path_factory.mktemp("sp-state-manager-prepared-transactions") / "test"
    command = [
        *compiler_command,
        "-std=c++20",
        "-O0",
        "-g",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-I",
        str(ROOT / "csrc"),
        str(CPP_TEST),
        str(ROOT / "csrc" / "nanodeploy" / "scheduler" / "sp_state_manager.cpp"),
        str(ROOT / "csrc" / "nanodeploy" / "scheduler" / "block_manager.cpp"),
        str(ROOT / "csrc" / "nanodeploy" / "scheduler" / "block.cpp"),
        str(ROOT / "csrc" / "nanodeploy" / "sequence" / "sequence.cpp"),
        "-pthread",
        "-o",
        str(output),
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, (
        "failed to compile SPStateManager prepared-transaction CPU harness\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    return output


@pytest.mark.parametrize("case_name", TEST_CASES)
def test_sp_state_manager_prepared_transactions_cpu(
    sp_state_manager_prepared_transactions_binary, case_name
):
    result = subprocess.run(
        [str(sp_state_manager_prepared_transactions_binary), case_name],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"SPStateManager CPU case {case_name!r} failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
