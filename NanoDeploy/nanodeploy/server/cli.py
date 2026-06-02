"""``nanodeploy`` command-line entry point.

``serve`` uses the same argument style as ``engine_server.py`` (jsonargparse
+ :class:`~nanodeploy.config.Config`). ``--host`` / ``--port`` bind the HTTP
API (same fields as ZMQ ``engine_server``, but used for uvicorn here)::

    nanodeploy serve /path/to/model \\
        --host 0.0.0.0 --port 8100 \\
        --served-model-name Qwen3-4B \\
        --ctrl_address 127.0.0.1:4479 --ray_address 127.0.0.1:7078
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Sequence

# Parsed by jsonargparse on the serve sub-command only (not part of Config).
_SERVE_EXTRA_KEYS = frozenset({"config", "model", "served_model_name"})


def _serve_main(argv: Sequence[str]) -> None:
    from jsonargparse import ActionConfigFile, ArgumentParser

    from nanodeploy.config import Config
    from nanodeploy.server.openai_server import run_server

    parser = ArgumentParser(
        description="NanoDeploy OpenAI serve (in-process hybrid engine)"
    )
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument(
        "model",
        type=str,
        nargs="?",
        default=None,
        help="Path to the model (positional shorthand for --model)",
    )
    parser.add_argument(
        "--served-model-name",
        default=None,
        help="Model id for /v1/models and routing (default: basename of model path)",
    )
    parser.add_class_arguments(Config, fail_untyped=False)

    args = parser.parse_args(list(argv))

    init_args = {k: v for k, v in vars(args).items() if k not in _SERVE_EXTRA_KEYS}
    model_path = args.model or init_args.get("model")
    if not model_path:
        parser.error("model path is required (positional or --model)")

    init_args["model"] = model_path
    init_args["mode"] = "hybrid"

    try:
        config = Config(**init_args)
    except Exception as e:
        parser.error(f"invalid configuration: {e}")

    served_model_name = args.served_model_name or os.path.basename(
        model_path.rstrip("/")
    )

    run_server(
        config,
        served_model_name=served_model_name,
        ctrl_address=config.ctrl_address,
        ctrl_scope=config.ctrl_scope,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: nanodeploy serve [MODEL] [options]", file=sys.stderr)
        print("       nanodeploy serve --help", file=sys.stderr)
        sys.exit(2)

    if argv[0] in ("-h", "--help"):
        _serve_main(["--help"])
        return

    if argv[0] != "serve":
        print(
            f"nanodeploy: unknown command {argv[0]!r} (only 'serve' is supported)",
            file=sys.stderr,
        )
        sys.exit(2)

    _serve_main(argv[1:])


if __name__ == "__main__":
    main()
