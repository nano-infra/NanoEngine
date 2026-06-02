"""``nanodeploy`` command-line entry point.

Currently exposes a single sub-command, ``serve``, which starts an
OpenAI-compatible HTTP server backed by an in-process hybrid engine::

    nanodeploy serve /path/to/model \
        --host 0.0.0.0 --port 8100 \
        --served-model-name Qwen3-4B \
        --ctrl-address 127.0.0.1:4479
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence


def _add_serve_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", help="Path to the model (or HF id)")

    # HTTP server
    parser.add_argument(
        "--host", default="0.0.0.0", help="HTTP bind host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="HTTP port (default: 8000)"
    )
    parser.add_argument(
        "--served-model-name",
        dest="served_model_name",
        default=None,
        help="Model id advertised via /v1/models and used for routing "
        "(default: basename of the model path)",
    )

    # dlslime-ctrl service discovery
    parser.add_argument(
        "--ctrl-address",
        dest="ctrl_address",
        default=None,
        help="dlslime-ctrl address (host:port). When set, the server registers "
        "its HTTP endpoint so a router can discover it.",
    )
    parser.add_argument(
        "--ctrl-scope",
        dest="ctrl_scope",
        default=None,
        help="dlslime-ctrl scope for multi-tenant isolation",
    )

    # Engine knobs (subset of nanodeploy.config.Config). Only forwarded when set.
    parser.add_argument("--max-model-len", dest="max_model_len", type=int, default=None)
    parser.add_argument("--max-num-seqs", dest="max_num_seqs", type=int, default=None)
    parser.add_argument(
        "--max-num-batched-tokens",
        dest="max_num_batched_tokens",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        dest="gpu_memory_utilization",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--kvcache-block-size", dest="kvcache_block_size", type=int, default=None
    )
    parser.add_argument("--attention-tp", dest="attention_tp", type=int, default=None)
    parser.add_argument("--attention-dp", dest="attention_dp", type=int, default=None)
    parser.add_argument("--attention-sp", dest="attention_sp", type=int, default=None)
    parser.add_argument("--ffn-tp", dest="ffn_tp", type=int, default=None)
    parser.add_argument("--ffn-ep", dest="ffn_ep", type=int, default=None)
    parser.add_argument("--ffn-dp", dest="ffn_dp", type=int, default=None)
    parser.add_argument(
        "--executor-backend",
        dest="executor_backend",
        choices=["ray", "dlslime"],
        default=None,
    )
    parser.add_argument("--ray-address", dest="ray_address", default=None)
    parser.add_argument("--master-address", dest="master_address", default=None)
    parser.add_argument(
        "--trust-remote-code", dest="trust_remote_code", action="store_true"
    )
    parser.add_argument("--enforce-eager", dest="enforce_eager", action="store_true")
    parser.add_argument("--dummy-weight", dest="dummy_weight", action="store_true")
    parser.add_argument("--log-level", dest="log_level", default=None)


# Args consumed by the server itself rather than forwarded to the engine Config.
_SERVER_ONLY = {
    "command",
    "model",
    "host",
    "port",
    "served_model_name",
    "ctrl_address",
    "ctrl_scope",
}


def _run_serve(args: argparse.Namespace) -> None:
    from nanodeploy.config import Config
    from nanodeploy.server.openai_server import run_server

    served_model_name = args.served_model_name or os.path.basename(
        args.model.rstrip("/")
    )

    config_kwargs: dict = {"model": args.model, "mode": "hybrid"}
    for key, value in vars(args).items():
        if key in _SERVER_ONLY:
            continue
        if value is None or value is False:
            # ``None`` -> not provided; ``store_true`` default False -> keep Config default.
            continue
        config_kwargs[key] = value

    config = Config(**config_kwargs)

    run_server(
        config,
        host=args.host,
        port=args.port,
        served_model_name=served_model_name,
        ctrl_address=args.ctrl_address,
        ctrl_scope=args.ctrl_scope,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="nanodeploy", description="NanoDeploy command-line interface"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve_parser = sub.add_parser(
        "serve",
        help="Start an OpenAI-compatible HTTP server (in-process hybrid engine)",
    )
    _add_serve_arguments(serve_parser)

    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    if args.command == "serve":
        _run_serve(args)
    else:  # pragma: no cover - argparse enforces a valid sub-command
        parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
