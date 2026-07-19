import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
CPP_TEST = ROOT / "tests" / "cpp" / "test_block_manager_prepared_mutation.cpp"
TEST_CASES = [
    "allocate_abort_and_destructor",
    "allocate_commit",
    "exact_allocate_snapshot_restore",
    "release_abort_and_destructor",
    "release_commit_and_shared_reference",
    "move_ownership_and_disjoint_preparations",
    "validation_and_prepared_guards",
]


@pytest.fixture(scope="session")
def block_manager_prepared_mutation_binary(tmp_path_factory):
    compiler_command = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler_command or shutil.which(compiler_command[0]) is None:
        pytest.skip("a C++ compiler is required for the BlockManager CPU harness")

    output = tmp_path_factory.mktemp("block-manager-prepared-mutation") / "test"
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
        str(ROOT / "csrc" / "nanodeploy" / "scheduler" / "block_manager.cpp"),
        str(ROOT / "csrc" / "nanodeploy" / "scheduler" / "block.cpp"),
        str(ROOT / "csrc" / "nanodeploy" / "sequence" / "sequence.cpp"),
        "-pthread",
        "-o",
        str(output),
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, (
        "failed to compile BlockManager prepared-mutation CPU harness\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    return output


@pytest.mark.parametrize("case_name", TEST_CASES)
def test_block_manager_prepared_mutation_cpu(block_manager_prepared_mutation_binary, case_name):
    result = subprocess.run(
        [str(block_manager_prepared_mutation_binary), case_name],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"BlockManager CPU case {case_name!r} failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
