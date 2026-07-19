import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
CPP_TEST = ROOT / "tests" / "cpp" / "test_sp_state_manager_prepared_iteration.cpp"
TEST_CASES = [
    "rank_local_same_rank_disjoint_transfer",
    "full_rank_swap_commits_atomically",
    "abort_restores_exact_context_allocator_and_counters",
    "combined_disjoint_groups_publish_once",
    "stale_validation_move_and_destructor_abort",
    "initial_iteration_composition_is_allocation_free_and_preserves_counters",
]


@pytest.fixture(scope="session")
def sp_state_manager_prepared_iteration_binary(tmp_path_factory):
    compiler_command = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler_command or shutil.which(compiler_command[0]) is None:
        pytest.skip("a C++ compiler is required for the prepared-iteration CPU harness")

    output = tmp_path_factory.mktemp("sp-state-manager-prepared-iteration") / "test"
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
        "failed to compile SPStateManager prepared-iteration CPU harness\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    return output


@pytest.mark.parametrize("case_name", TEST_CASES)
def test_sp_state_manager_prepared_iteration_cpu(
    sp_state_manager_prepared_iteration_binary, case_name
):
    result = subprocess.run(
        [str(sp_state_manager_prepared_iteration_binary), case_name],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"SPStateManager prepared-iteration CPU case {case_name!r} failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
