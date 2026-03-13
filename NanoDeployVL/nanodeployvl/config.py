"""VLConfig – configuration for Vision-Language inference.

Extends NanoDeploy's ``Config`` with vision-specific parameters while
keeping backward compatibility with the LLM engine.
"""

from __future__ import annotations

from typing import Any, Optional

from nanodeploy.config import Config
from nanodeploy.logging import get_logger

from pydantic import Field, model_validator
from transformers import AutoConfig

logger = get_logger("nanodeployvl")


class VLConfig(Config):
    """Configuration for Vision-Language engine.

    Inherits all LLM parameters from ``Config`` and adds vision-related
    fields.  The vision config is automatically extracted from the HF
    model config during validation.

    New fields
    ----------
    vision_device : str
        Device for the vision encoder (default: ``"cuda:0"``).  Future
        optimisation: allow a separate GPU or data-parallel sharding.
    vision_dtype : str
        Torch dtype name for vision encoder weights (default: ``"bfloat16"``).
    vision_batch_size : int
        Max images encoded in one forward pass (memory bound).
    """

    # --- vision encoder placement ---
    vision_device: str = "cuda:0"
    vision_dtype: str = "bfloat16"
    vision_batch_size: int = 8

    # Populated during validation
    vision_config: Any = Field(default=None, exclude=True)
    image_token_id: int = Field(default=-1, exclude=True)
    video_token_id: int = Field(default=-1, exclude=True)
    vision_start_token_id: int = Field(default=-1, exclude=True)
    vision_end_token_id: int = Field(default=-1, exclude=True)

    @model_validator(mode="after")
    def validate_vl_config(self) -> "VLConfig":
        """Extract vision config from HF model config after parent validation."""
        hf_config = self.hf_config
        if hf_config is None:
            return self

        # Extract vision_config
        if hasattr(hf_config, "vision_config"):
            self.vision_config = hf_config.vision_config
        else:
            logger.warning(
                "No vision_config found in HF config; "
                "this model may not support multimodal input."
            )

        # Extract special token IDs
        for attr in (
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
        ):
            val = getattr(hf_config, attr, None)
            if val is not None:
                setattr(self, attr, val)

        return self
