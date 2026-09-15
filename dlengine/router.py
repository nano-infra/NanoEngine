"""Console entry point for the Rust DLEngine router."""

from __future__ import annotations

import sys

from dlengine._rust.wrapper import export

_run_router = export(("run_router",))["run_router"]


def main() -> None:
    try:
        _run_router(sys.argv[1:])
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
