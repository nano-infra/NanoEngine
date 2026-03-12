"""EncoderConfig – configuration for standalone vision encoder engine.

Separate from VLConfig: the encoder is a single-responsibility engine
that only does vision encoding + EmbeddingPool management.
"""

from __future__ import annotations

from typing import Any

from nanodeploy.logging import get_logger
from pydantic import BaseModel, Field, model_validator
from transformers import AutoConfig

logger = get_logger("encoder")


class EncoderConfig(BaseModel):
    """Configuration for a standalone vision encoder engine.

    Parameters
    ----------
    model : str
        Path to the HF model directory (same ckpt as the LLM, contains
        ``visual.*`` weights).
    vision_device : str
        CUDA device for the encoder.
    vision_dtype : str
        Torch dtype name for encoder weights.
    num_slots : int
        Number of concurrent embedding slots in EmbeddingPool.
    max_tokens_per_slot : int
        Max merged vision tokens per slot.
    nanoctrl_address : str | None
        NanoCtrl server URL for service registration.
    nanoctrl_scope : str | None
        NanoCtrl scope for multi-tenant isolation.
    host : str
        Bind address for ZMQ sockets.
    p2p_port : int
        Port for P2P ZMQ free-slot notifications.
    zmq_port : int
        Port for the ZMQ encode service (NanoRoute connects here).
    """

    model: str
    vision_device: str = "cuda:0"
    vision_dtype: str = "bfloat16"
    num_slots: int = 64
    max_tokens_per_slot: int = 4096
    nanoctrl_address: str | None = None
    nanoctrl_scope: str | None = None
    host: str = "0.0.0.0"
    p2p_port: int = 0  # 0 = auto-bind
    zmq_port: int = 0  # 0 = auto-bind; NanoRoute connects to this port

    # Populated during validation
    vision_config: Any = Field(default=None, exclude=True)
    hidden_size: int = Field(default=0, exclude=True)

    @model_validator(mode="after")
    def _load_hf_vision_config(self) -> "EncoderConfig":
        hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        if hasattr(hf_config, "vision_config"):
            self.vision_config = hf_config.vision_config
            self.hidden_size = getattr(
                hf_config.vision_config,
                "out_hidden_size",
                getattr(hf_config, "hidden_size", 3584),
            )
        else:
            raise ValueError(
                f"Model {self.model} does not have vision_config; "
                "cannot use as an encoder."
            )
        return self
