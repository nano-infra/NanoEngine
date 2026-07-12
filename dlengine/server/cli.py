"""``dlengine`` command-line entry point.

``serve`` uses the same argument style as ``engine_server.py`` (jsonargparse
+ :class:`~dlengine.config.Config`). ``--host`` / ``--port`` bind the HTTP
API (same fields as ZMQ ``engine_server``, but used for uvicorn here)::

    dlengine serve /path/to/model \\
        --host 0.0.0.0 --port 8100 \\
        --served-model-name Qwen3-4B \\
        --ctrl_address 127.0.0.1:4479 --ray_address auto

``monitor`` writes a Prometheus/Grafana stack with Prometheus on 9090 and
Grafana on 3000::

    dlengine monitor --dlengine-target host.docker.internal:5000 --up
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Sequence

# Parsed by jsonargparse on the serve sub-command only (not part of Config).
_SERVE_EXTRA_KEYS = frozenset({"config", "model", "served_model_name"})


def _normalize_serve_argv(argv: Sequence[str]) -> list[str]:
    """Let ``--enable_monitor`` work as ``--enable_monitor true`` with jsonargparse."""
    normalized: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--enable_monitor":
            # Check if next arg is a boolean value
            if i + 1 < len(argv) and argv[i + 1].lower() in ("true", "false"):
                normalized.extend(["--enable_monitor", argv[i + 1].lower()])
                i += 2
            else:
                # Default to true if no value provided
                normalized.extend(["--enable_monitor", "true"])
                i += 1
        else:
            normalized.append(arg)
            i += 1
    return normalized


def _serve_main(argv: Sequence[str]) -> None:
    from jsonargparse import ActionConfigFile, ArgumentParser

    from dlengine.config import Config
    from dlengine.server.openai_server import run_server

    parser = ArgumentParser(
        description="DLEngine OpenAI serve (in-process hybrid engine)"
    )
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument(
        "model",
        type=str,
        nargs="?",
        help="Path to the model (positional shorthand for --model)",
    )
    parser.add_argument(
        "--served-model-name",
        default=None,
        help="Model id for /v1/models and routing (default: basename of model path)",
    )
    parser.add_class_arguments(Config, fail_untyped=False)

    args = parser.parse_args(_normalize_serve_argv(argv))

    init_args = {k: v for k, v in vars(args).items() if k not in _SERVE_EXTRA_KEYS}
    model_path = getattr(args, "model", None) or init_args.get("model")
    if not model_path:
        parser.error("model path is required (positional or --model)")

    init_args["model"] = model_path
    # ``mode`` comes from Config (--mode hybrid|prefill|decode, default hybrid).
    # hybrid runs prefill+decode in-process; prefill/decode enable PD
    # disaggregation where the decode engine RDMA-pulls KV from prefill.
    mode = init_args.get("mode", "hybrid")
    if mode not in ("hybrid", "prefill", "decode"):
        parser.error(f"invalid --mode {mode!r} (expected hybrid|prefill|decode)")

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
        print("usage: dlengine serve [MODEL] [options]", file=sys.stderr)
        print("       dlengine monitor [options]", file=sys.stderr)
        print("       dlengine serve --help", file=sys.stderr)
        sys.exit(2)

    if argv[0] in ("-h", "--help"):
        print("usage: dlengine <command> [options]")
        print()
        print("commands:")
        print("  serve     Start the OpenAI-compatible HTTP server")
        print("  monitor   Generate a Grafana/Prometheus monitoring stack")
        return

    if argv[0] == "monitor":
        from dlengine.monitor import main as _monitor_main

        _monitor_main(argv[1:])
        return

    if argv[0] != "serve":
        print(
            f"dlengine: unknown command {argv[0]!r} " "(supported: 'serve', 'monitor')",
            file=sys.stderr,
        )
        sys.exit(2)

    _serve_main(argv[1:])


if __name__ == "__main__":
    main()
