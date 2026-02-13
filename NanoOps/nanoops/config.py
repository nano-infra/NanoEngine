"""Configuration management for NanoOps."""

import os
from typing import Optional

from pydantic import BaseModel, Field


class NanoOpsConfig(BaseModel):
    """NanoOps configuration."""

    # Connection settings
    redis_url: str = Field("redis://localhost:6379", description="Redis connection URL")
    ray_address: str = Field(
        "http://localhost:8265", description="Ray dashboard HTTP address"
    )
    nanoctrl_address: str = Field(
        "http://localhost:3000", description="NanoCtrl HTTP address"
    )

    # NanoCtrl management
    nanoctrl_auto_start: bool = Field(
        True, description="Auto-start NanoCtrl if not running"
    )
    nanoctrl_binary_path: Optional[str] = Field(
        None, description="Path to NanoCtrl binary"
    )
    nanoctrl_config_path: Optional[str] = Field(
        None, description="Path to NanoCtrl config.toml"
    )

    # Ray job settings
    code_mode: str = Field(
        "preinstalled", description="Code distribution: preinstalled | upload"
    )
    working_dir: Optional[str] = Field(
        None, description="Working directory for Ray jobs (upload mode)"
    )

    # Defaults for engine spawning
    default_kvcache_blocks: int = 15000
    default_kvcache_block_size: int = 256
    default_max_model_len: int = 16384
    default_max_num_batched_tokens: int = 16384

    @classmethod
    def from_file(cls, path: str) -> "NanoOpsConfig":
        """Load config from TOML file."""
        import toml

        with open(path) as f:
            data = toml.load(f)
        return cls(**data)

    @classmethod
    def from_env(cls) -> "NanoOpsConfig":
        """Load config from environment variables.

        Only includes fields whose env vars are actually set, so that
        model_dump(exclude_unset=True) correctly skips unset fields.
        """
        kwargs = {}
        if os.getenv("NANOCTRL_REDIS_URL"):
            kwargs["redis_url"] = os.getenv("NANOCTRL_REDIS_URL")
        if os.getenv("RAY_ADDRESS"):
            kwargs["ray_address"] = os.getenv("RAY_ADDRESS")
        if os.getenv("NANOCTRL_ADDRESS"):
            kwargs["nanoctrl_address"] = os.getenv("NANOCTRL_ADDRESS")
        return cls(**kwargs)


def load_config(config_file: Optional[str] = None) -> NanoOpsConfig:
    """Load configuration with precedence: CLI > Env > File > Defaults."""
    # Start with defaults
    config = NanoOpsConfig()

    # Override from file if exists
    if config_file and os.path.exists(config_file):
        config = NanoOpsConfig.from_file(config_file)
    elif os.path.exists("nanoops.toml"):
        config = NanoOpsConfig.from_file("nanoops.toml")

    # Override from environment variables
    env_config = NanoOpsConfig.from_env()
    config = config.model_copy(update=env_config.model_dump(exclude_unset=True))

    return config
