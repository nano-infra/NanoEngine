"""VLEngineServer — Encoder-only server for EP-separated VL inference.

Manages a standalone ``EncoderEngine`` (ViT + EmbeddingPool + RDMA) and
registers it with NanoCtrl for service discovery.  Client-facing
``/v1/chat/completions`` is handled entirely by **NanoRoute**, which
discovers this encoder via NanoCtrl and communicates through ZMQ
(Action=5/6 encode protocol).

Architecture
------------
::

    Client (HTTP)
      │
      ▼
    NanoRoute (HTTP reverse proxy + ZMQ router)
      ├── tokenize text + detect images
      ├── ZMQ Action=5 → EncoderEngine (image encode)
      ├── ZMQ Action=6 ← vision_slots + input_ids
      ├── build Sequence (FlatBuffer) with vision_slots
      ├── ZMQ → LLM Engine (inference)
      └── SSE / JSON → Client

    VLEngineServer (this process)
      ├── EncoderEngine: ViT + EmbeddingPool + RDMA MR
      ├── ZMQ ROUTER: accept encode requests from NanoRoute
      ├── P2P ZMQ: accept slot-free from LLM engines
      └── NanoCtrl: register encoder + heartbeat
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from nanodeploy.logging import get_logger
from pydantic import BaseModel

from nanodeployvl.encoder.encoder_config import EncoderConfig
from nanodeployvl.encoder.encoder_engine import EncoderEngine

logger = get_logger("nanodeployvl")


# ── Server configuration ────────────────────────────────────────────


class VLServerConfig(BaseModel):
    """Configuration for VLEngineServer (encoder-only)."""

    model: str  # HF model directory

    # HTTP server (health check only)
    host: str = "0.0.0.0"
    port: int = 8000

    # Vision encoder
    encoder_device: str = "cuda:0"
    encoder_dtype: str = "bfloat16"
    num_slots: int = 64
    max_tokens_per_slot: int = 4096

    # NanoCtrl (for encoder RDMA registration + service discovery)
    nanoctrl_address: str | None = None
    nanoctrl_scope: str | None = None

    # P2P (encoder free listener)
    p2p_port: int = 0


# ── VLEngineServer ───────────────────────────────────────────────────


class VLEngineServer:
    """Encoder-only server for EP-separated VL inference.

    Manages an ``EncoderEngine`` (vision encoding + RDMA EmbeddingPool)
    and registers with NanoCtrl.  Client requests are routed by
    NanoRoute, which connects to the encoder's ZMQ encode service.
    """

    def __init__(self, config: VLServerConfig) -> None:
        self.config = config
        self.model_name = config.model.rstrip("/").split("/")[-1]
        self.encoder_engine: EncoderEngine | None = None

    async def start(self) -> None:
        """Initialise EncoderEngine."""
        cfg = self.config

        encoder_cfg = EncoderConfig(
            model=cfg.model,
            vision_device=cfg.encoder_device,
            vision_dtype=cfg.encoder_dtype,
            num_slots=cfg.num_slots,
            max_tokens_per_slot=cfg.max_tokens_per_slot,
            nanoctrl_address=cfg.nanoctrl_address,
            nanoctrl_scope=cfg.nanoctrl_scope,
            p2p_port=cfg.p2p_port,
        )
        logger.info("Starting EncoderEngine on %s", cfg.encoder_device)
        self.encoder_engine = EncoderEngine(encoder_cfg)
        logger.info("EncoderEngine ready (model=%s)", self.model_name)

    async def stop(self) -> None:
        if self.encoder_engine is not None:
            self.encoder_engine.shutdown()


# ── FastAPI application factory ──────────────────────────────────────


def create_app(config: VLServerConfig) -> FastAPI:
    """Create a FastAPI application with health-check endpoint."""
    server = VLEngineServer(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await server.start()
        yield
        await server.stop()

    app = FastAPI(title="NanoDeployVL", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


# ── CLI entry point ──────────────────────────────────────────────────


def main():
    """Launch VLEngineServer (encoder-only) from command line."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(
        description="NanoDeployVL Encoder Server "
        "(client requests go through NanoRoute)"
    )
    parser.add_argument("--model", required=True, help="HF model directory")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--encoder-device", default="cuda:0")
    parser.add_argument("--encoder-dtype", default="bfloat16")
    parser.add_argument("--num-slots", type=int, default=64)
    parser.add_argument("--max-tokens-per-slot", type=int, default=4096)
    parser.add_argument("--nanoctrl-address", default=None)
    parser.add_argument("--nanoctrl-scope", default=None)
    parser.add_argument("--p2p-port", type=int, default=0)

    args = parser.parse_args()

    config = VLServerConfig(
        model=args.model,
        host=args.host,
        port=args.port,
        encoder_device=args.encoder_device,
        encoder_dtype=args.encoder_dtype,
        num_slots=args.num_slots,
        max_tokens_per_slot=args.max_tokens_per_slot,
        nanoctrl_address=args.nanoctrl_address,
        nanoctrl_scope=args.nanoctrl_scope,
        p2p_port=args.p2p_port,
    )

    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
