"""Image/video processor wrapper for Qwen3.5-MoE VLM.

Wraps the HuggingFace ``Qwen3VLProcessor`` to provide a simple interface
for preprocessing images and videos into the format expected by our
``VisionEncoder``.

Design notes
------------
- We delegate all heavy lifting (resize, normalize, patch grid calculation)
  to the HF processor.
- The processor also handles tokenization, expanding ``<|image_pad|>``
  placeholders to the correct number of tokens.
- Future: support custom preprocessors or batch-optimised pipelines.
"""

from __future__ import annotations

from typing import Union

import torch

from nanodeploy.logging import get_logger
from PIL import Image
from transformers import AutoProcessor

logger = get_logger("nanodeployvl")


class ImageProcessor:
    """Wraps HuggingFace Qwen3VLProcessor for image/video preprocessing.

    Parameters
    ----------
    model_path : str
        Path to the model checkpoint (used to load processor configs).
    """

    def __init__(self, model_path: str) -> None:
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        logger.info("Loaded HF processor: %s", type(self.processor).__name__)

    def apply_chat_template(
        self,
        messages: list[dict],
        add_generation_prompt: bool = True,
    ) -> str:
        """Apply chat template with vision token placeholders.

        This uses the HF processor's chat template which correctly
        inserts ``<|vision_start|><|image_pad|><|vision_end|>`` for
        images and similar tokens for videos.

        Args:
            messages: Chat messages in OpenAI format, e.g.::

                [{"role": "user", "content": [
                    {"type": "image", "image": <PIL.Image>},
                    {"type": "text", "text": "Describe this image."}
                ]}]

            add_generation_prompt: Whether to append assistant turn start.

        Returns:
            Formatted prompt string with vision placeholders.
        """
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def process(
        self,
        text: str | list[str],
        images: list[Image.Image] | None = None,
        videos: list | None = None,
    ) -> dict[str, torch.Tensor]:
        """Preprocess text + images/videos for the model.

        Calls the HF processor which:
        1. Tokenizes text (expanding vision placeholders to correct length)
        2. Preprocesses images (resize, normalize, extract patches)
        3. Computes ``image_grid_thw`` / ``video_grid_thw``

        Args:
            text: Prompt string(s) with vision placeholders already inserted.
            images: List of PIL images (one per ``<|image_pad|>`` in text).
            videos: List of videos (optional).

        Returns:
            Dict with keys like ``input_ids``, ``attention_mask``,
            ``pixel_values``, ``image_grid_thw``, etc.
        """
        kwargs = {"text": text, "return_tensors": "pt", "padding": True}
        if images is not None:
            kwargs["images"] = images
        if videos is not None:
            kwargs["videos"] = videos

        outputs = self.processor(**kwargs)
        return dict(outputs)

    def get_token_ids(self, text: str) -> list[int]:
        """Tokenize text without image processing."""
        return self.tokenizer.encode(text)

    def decode(self, token_ids: list[int] | torch.Tensor) -> str:
        """Decode token IDs to string."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)
