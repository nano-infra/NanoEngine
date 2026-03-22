"""Entry point: python -m NanoFold [options]"""

from __future__ import annotations

import argparse
import asyncio
import logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NanoFold structure prediction ZMQ server"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--port",
        type=int,
        default=8201,
        help="ZMQ ROUTER port (NanoRoute connects here)",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_name", default="protenix_base_default_v1.0.0")
    parser.add_argument("--checkpoint_dir", default="/models/fold/checkpoint")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--n_cycle", type=int, default=10)
    parser.add_argument("--trimul_kernel", default="cuequivariance")
    parser.add_argument("--triatt_kernel", default="cuequivariance")
    parser.add_argument("--shm_dir", default="/dev/shm/nanofold")
    parser.add_argument(
        "--nanoctrl_url",
        default=None,
        help="NanoCtrl base URL, e.g. http://10.x.x.x:3000",
    )
    parser.add_argument(
        "--nanoctrl_address",
        default=None,
        metavar="HOST:PORT",
        help="NanoCtrl address as host:port (alias for --nanoctrl_url)",
    )
    parser.add_argument("--nanoctrl_scope", default=None)
    parser.add_argument("--redis_url", default=None)
    parser.add_argument("--embed_ttl_s", type=int, default=3600)
    parser.add_argument("--log_level", default="info")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    # Resolve NanoCtrl URL: --nanoctrl_url wins, then --nanoctrl_address (host:port → URL)
    nanoctrl_url = args.nanoctrl_url
    if not nanoctrl_url and args.nanoctrl_address:
        addr = args.nanoctrl_address
        if not addr.startswith("http"):
            addr = f"http://{addr}"
        nanoctrl_url = addr
    if not nanoctrl_url:
        nanoctrl_url = "http://127.0.0.1:3000"

    from NanoFold.server.config import NanoFoldConfig
    from NanoFold.server.server import serve

    cfg = NanoFoldConfig(
        host=args.host,
        port=args.port,
        gpu=args.gpu,
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
        dtype=args.dtype,
        n_cycle=args.n_cycle,
        trimul_kernel=args.trimul_kernel,
        triatt_kernel=args.triatt_kernel,
        shm_dir=args.shm_dir,
        nanoctrl_url=nanoctrl_url,
        nanoctrl_scope=args.nanoctrl_scope,
        redis_url=args.redis_url,
        embed_ttl_s=args.embed_ttl_s,
    )
    asyncio.run(serve(cfg))


if __name__ == "__main__":
    main()
