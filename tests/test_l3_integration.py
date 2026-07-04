"""Manual L3/3FS integration placeholder for the Rust protocol boundary.

Pure BlockManager behavior is covered in Rust unit tests. The old Python
integration constructed Sequence objects directly, which is no longer the
public API. Restore this test once scheduler exposes explicit L3 load/offload
protocol hooks.
"""

from __future__ import annotations

import argparse
import os


def _skip(msg: str) -> int:
    print(f"[skip] {msg}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mountpoint", default=os.environ.get("HF3FS_MOUNTPOINT", "/3fs/mnt")
    )
    parser.parse_args()
    return _skip(
        "L3 integration now needs scheduler-protocol load/offload hooks; "
        "run `cargo test l3::tests` for pure BlockManager coverage"
    )


if __name__ == "__main__":
    raise SystemExit(main())
