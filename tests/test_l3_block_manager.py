"""L3 BlockManager logic lives in Rust unit tests.

The old Python test constructed and inspected ``Sequence`` internals directly.
That is no longer the public boundary. Keep this as a small pytest/standalone
bridge so the familiar command still verifies the Rust coverage.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_l3_block_manager_rust_unit_tests():
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(["cargo", "test", "l3::tests"], cwd=repo_root, check=True)


def main() -> int:
    test_l3_block_manager_rust_unit_tests()
    print("[ok] Rust L3 BlockManager tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
