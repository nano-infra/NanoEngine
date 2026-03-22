from __future__ import annotations

import os

from pydantic_settings import BaseSettings


class NanoFoldConfig(BaseSettings):
    # Server
    host: str = "0.0.0.0"
    port: int = 8201
    gpu: int = 0

    # Model
    model_name: str = "protenix_base_default_v1.0.0"
    checkpoint_dir: str = "/models/fold/checkpoint"
    dtype: str = "bf16"
    n_cycle: int = 10
    trimul_kernel: str = "cuequivariance"
    triatt_kernel: str = "cuequivariance"
    enable_cache: bool = True
    enable_fusion: bool = True
    enable_tf32: bool = True

    # Storage
    shm_dir: str = "/dev/shm/nanofold"
    embed_ttl_s: int = 3600  # seconds before embed cache expires
    job_ttl_s: int = 86400  # seconds before completed job expires

    # NanoCtrl / Redis
    nanoctrl_url: str = "http://127.0.0.1:3000"
    nanoctrl_scope: str | None = None
    # Redis URL resolved from NanoCtrl at startup; can also be set directly
    redis_url: str | None = None

    model_config = {"env_prefix": "NANOFOLD_"}
