"""VLEngine – Vision-Language engine wrapping NanoDeploy's LLM engine.

Orchestrates:
1. Image/video preprocessing via HF Qwen3VLProcessor
2. Vision encoding via standalone VisionEncoder on the driver
3. LLM inference via NanoDeploy's LLMEngine (scheduler + Ray workers)

The vision embeddings are passed to the LLM model by replacing placeholder
tokens in the input embedding layer during prefill.

Design notes for future optimisation
-------------------------------------
- **MRoPE**: Currently uses 1D positions.  To support MRoPE (3D position
  IDs), the ``get_rope_index()`` method is provided but not yet wired to
  the model runner.  The interface is ready: ``run_model`` would accept
  an optional ``position_ids_3d`` parameter.
- **Encoder placement**: Vision encoder runs on the driver.  To use a
  dedicated GPU or data-parallel sharding, wrap the encoder in a Ray
  actor and call ``encode()`` remotely.
- **Streaming**: For now, ``generate()`` is synchronous.  The engine
  can be extended to support streaming by yielding tokens as they are
  generated.
"""

from __future__ import annotations

from typing import Optional

import torch

from nanodeploy.engine.llm_engine import LLMEngine
from nanodeploy.engine.sequence import Sequence
from nanodeploy.logging import get_logger
from nanodeploy.sampling_params import SamplingParams
from PIL import Image

from nanodeployvl.config import VLConfig
from nanodeployvl.vision.encoder import VisionEncoder
from nanodeployvl.vision.processor import ImageProcessor

logger = get_logger("nanodeployvl")


class VLEngine:
    """Vision-Language inference engine.

    This engine manages a ``VisionEncoder`` + ``LLMEngine`` pipeline.
    Vision encoding happens once per request (during prefill), and the
    resulting embeddings are injected into the LLM's input.

    Parameters
    ----------
    config : VLConfig
        VL-specific configuration (inherits all LLM params plus vision).

    Example
    -------
    >>> from nanodeployvl.config import VLConfig
    >>> from nanodeployvl.engine.vl_engine import VLEngine
    >>> config = VLConfig(model="/models/Qwen3.5-35B-A3B", attention_dp=1)
    >>> engine = VLEngine(config)
    >>> result = engine.generate_vl(
    ...     messages=[{"role": "user", "content": [
    ...         {"type": "image", "image": some_pil_image},
    ...         {"type": "text", "text": "What is in this image?"}
    ...     ]}],
    ... )
    """

    def __init__(self, config: VLConfig) -> None:
        self.config = config

        # Build image processor (HF-based)
        logger.info("Initializing ImageProcessor …")
        self.processor = ImageProcessor(config.model)

        # Build vision encoder (standalone ViT on driver)
        self._vision_encoder: Optional[VisionEncoder] = None
        if config.vision_config is not None:
            logger.info("Initializing VisionEncoder …")
            dtype = getattr(torch, config.vision_dtype, torch.bfloat16)
            self._vision_encoder = VisionEncoder(
                vision_config=config.vision_config,
                model_path=config.model,
                device=config.vision_device,
                dtype=dtype,
            )

        # Build LLM engine (NanoDeploy)
        logger.info("Initializing LLMEngine …")
        self.llm_engine = LLMEngine(config)

        # Cache special token IDs
        self._image_token_id = config.image_token_id
        self._video_token_id = config.video_token_id

    @property
    def tokenizer(self):
        return self.llm_engine.tokenizer

    # ------------------------------------------------------------------
    # Vision encoding
    # ------------------------------------------------------------------

    def _encode_images(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Run vision encoder on images.

        Returns a list of embedding tensors, one per image.
        """
        if self._vision_encoder is None:
            raise RuntimeError("No vision encoder available for this model")
        return self._vision_encoder.encode(pixel_values, image_grid_thw)

    def _encode_videos(
        self,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Run vision encoder on videos."""
        if self._vision_encoder is None:
            raise RuntimeError("No vision encoder available for this model")
        return self._vision_encoder.encode_video(pixel_values_videos, video_grid_thw)

    # ------------------------------------------------------------------
    # Embedding merging
    # ------------------------------------------------------------------

    def _merge_vision_embeddings(
        self,
        input_ids: torch.Tensor,
        text_embeddings: torch.Tensor,
        image_embeds: list[torch.Tensor] | None = None,
        video_embeds: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Replace placeholder token embeddings with vision embeddings.

        Mirrors the logic in HF's ``Qwen3_5MoeModel.forward()``::

            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        Args:
            input_ids: 1D token IDs (for mask computation).
            text_embeddings: LLM text embeddings from ``embed_tokens(input_ids)``.
            image_embeds: Per-image embedding tensors from vision encoder.
            video_embeds: Per-video embedding tensors from vision encoder.

        Returns:
            Combined embeddings with vision features inserted.
        """
        result = text_embeddings.clone()

        if image_embeds:
            all_image_embeds = torch.cat(image_embeds, dim=0).to(
                device=result.device, dtype=result.dtype
            )
            image_mask = input_ids == self._image_token_id
            n_image_tokens = image_mask.sum().item()
            if n_image_tokens != all_image_embeds.shape[0]:
                raise ValueError(
                    f"Image token count ({n_image_tokens}) != image embedding count "
                    f"({all_image_embeds.shape[0]}). Check preprocessing."
                )
            image_mask_expanded = image_mask.unsqueeze(-1).expand_as(result)
            result = result.masked_scatter(image_mask_expanded, all_image_embeds)

        if video_embeds:
            all_video_embeds = torch.cat(video_embeds, dim=0).to(
                device=result.device, dtype=result.dtype
            )
            video_mask = input_ids == self._video_token_id
            n_video_tokens = video_mask.sum().item()
            if n_video_tokens != all_video_embeds.shape[0]:
                raise ValueError(
                    f"Video token count ({n_video_tokens}) != video embedding count "
                    f"({all_video_embeds.shape[0]}). Check preprocessing."
                )
            video_mask_expanded = video_mask.unsqueeze(-1).expand_as(result)
            result = result.masked_scatter(video_mask_expanded, all_video_embeds)

        return result

    # ------------------------------------------------------------------
    # High-level generation API
    # ------------------------------------------------------------------

    def preprocess(
        self,
        messages: list[dict],
        images: list[Image.Image] | None = None,
        videos: list | None = None,
    ) -> dict:
        """Preprocess a single request with text + images/videos.

        1. Apply chat template (expands vision placeholders)
        2. Tokenize + preprocess images via HF processor
        3. Encode images/videos through vision encoder
        4. Return dict with token_ids, vision embeddings, etc.

        Args:
            messages: Chat messages in OpenAI format.
            images: PIL images corresponding to ``<image>`` entries in messages.
            videos: Video data corresponding to ``<video>`` entries.

        Returns:
            Dict with keys:
            - ``input_ids``: Token IDs with vision placeholders expanded.
            - ``image_embeds``: List of per-image embedding tensors (or None).
            - ``video_embeds``: List of per-video embedding tensors (or None).
        """
        # Step 1: Apply chat template to get prompt with vision tokens
        prompt = self.processor.apply_chat_template(messages)

        # Step 2: Process through HF processor (tokenize + image preprocessing)
        processed = self.processor.process(
            text=prompt,
            images=images,
            videos=videos,
        )

        input_ids = processed["input_ids"].squeeze(0)  # [seq_len]
        image_embeds = None
        video_embeds = None

        # Step 3: Encode images
        if "pixel_values" in processed and processed["pixel_values"] is not None:
            image_grid_thw = processed["image_grid_thw"]
            image_embeds = self._encode_images(
                processed["pixel_values"], image_grid_thw
            )

        # Step 4: Encode videos
        if (
            "pixel_values_videos" in processed
            and processed["pixel_values_videos"] is not None
        ):
            video_grid_thw = processed["video_grid_thw"]
            video_embeds = self._encode_videos(
                processed["pixel_values_videos"], video_grid_thw
            )

        return {
            "input_ids": input_ids,
            "image_embeds": image_embeds,
            "video_embeds": video_embeds,
        }

    def create_vl_sequence(
        self,
        preprocessed: dict,
        sampling_params: SamplingParams | None = None,
    ) -> Sequence:
        """Create a Sequence for the LLM engine from preprocessed VL data.

        The token_ids include expanded vision placeholders.  The vision
        embeddings will be injected during model forward via the
        ``inputs_embeds`` mechanism.

        Args:
            preprocessed: Output of ``preprocess()``.
            sampling_params: Generation parameters.

        Returns:
            A ``Sequence`` object ready for ``add_request()``.
        """
        if sampling_params is None:
            sampling_params = SamplingParams(max_tokens=256)

        input_ids = preprocessed["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()

        seq = Sequence(input_ids, sampling_params=sampling_params)
        return seq

    def set_worker_vision_embeds(
        self,
        image_embeds: list[torch.Tensor] | None = None,
        video_embeds: list[torch.Tensor] | None = None,
    ) -> None:
        """Push vision embeddings to all Ray model workers.

        This sets a side-channel on each ``ModelRunner`` so that during
        prefill the model can access vision embeddings to replace
        placeholder tokens.

        Note: This is the initial integration approach.  Future optimisation
        may use a more efficient transfer mechanism or compute embeddings
        directly on the worker GPU.
        """
        all_embeds = {}
        if image_embeds:
            cat = torch.cat(image_embeds, dim=0)
            all_embeds["image"] = cat.cpu()  # Send CPU tensors via Ray
        if video_embeds:
            cat = torch.cat(video_embeds, dim=0)
            all_embeds["video"] = cat.cpu()

        if all_embeds:
            self.llm_engine.executor.set_vision_embeds(all_embeds)

    def clear_worker_vision_embeds(self) -> None:
        """Clear vision embeddings from all workers after prefill."""
        self.llm_engine.executor.clear_vision_embeds()

    def generate_vl(
        self,
        messages: list[dict],
        images: list[Image.Image] | None = None,
        videos: list | None = None,
        sampling_params: SamplingParams | None = None,
        use_tqdm: bool = True,
    ) -> str:
        """End-to-end VL generation.

        1. Preprocess (tokenize + encode images)
        2. Push vision embeddings to workers
        3. Create sequence and run through LLM engine
        4. Decode output tokens

        Args:
            messages: Chat messages (OpenAI format with image/video entries).
            images: List of PIL images.
            videos: List of videos (optional).
            sampling_params: Generation parameters.
            use_tqdm: Show progress bar.

        Returns:
            Generated text string.
        """
        # Preprocess
        preprocessed = self.preprocess(messages, images=images, videos=videos)

        # Push vision embeds to workers
        self.set_worker_vision_embeds(
            image_embeds=preprocessed.get("image_embeds"),
            video_embeds=preprocessed.get("video_embeds"),
        )

        # Create sequence
        seq = self.create_vl_sequence(preprocessed, sampling_params)

        # Run through LLM engine
        self.llm_engine.add_request(seq)
        finished = self.llm_engine.generate(use_tqdm=use_tqdm)

        # Clear vision embeds from workers
        self.clear_worker_vision_embeds()

        # Decode
        if finished:
            output_seq = finished[0]
            completion_ids = output_seq.completion_token_ids
            return self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        return ""

    def generate_vl_batch(
        self,
        requests: list[dict],
        sampling_params: SamplingParams | None = None,
        use_tqdm: bool = True,
    ) -> list[str]:
        """Batch VL generation.

        Each request is a dict with keys:
        - ``messages``: Chat messages (OpenAI format).
        - ``images``: Optional list of PIL images.
        - ``videos``: Optional list of videos.

        Note: Currently processes requests sequentially for vision encoding,
        then batches them for LLM decoding.  Future: batch vision encoding.

        Args:
            requests: List of request dicts.
            sampling_params: Shared generation parameters.
            use_tqdm: Show progress bar.

        Returns:
            List of generated text strings.
        """
        # Preprocess all requests
        all_preprocessed = []
        all_image_embeds = []
        all_video_embeds = []

        for req in requests:
            preprocessed = self.preprocess(
                messages=req["messages"],
                images=req.get("images"),
                videos=req.get("videos"),
            )
            all_preprocessed.append(preprocessed)
            if preprocessed.get("image_embeds"):
                all_image_embeds.extend(preprocessed["image_embeds"])
            if preprocessed.get("video_embeds"):
                all_video_embeds.extend(preprocessed["video_embeds"])

        # Push all vision embeddings to workers
        self.set_worker_vision_embeds(
            image_embeds=all_image_embeds or None,
            video_embeds=all_video_embeds or None,
        )

        # Create sequences
        seqs = []
        for preprocessed in all_preprocessed:
            seq = self.create_vl_sequence(preprocessed, sampling_params)
            seqs.append(seq)

        # Run through LLM engine
        self.llm_engine.add_request(seqs)
        finished = self.llm_engine.generate(use_tqdm=use_tqdm)

        # Clear vision embeds
        self.clear_worker_vision_embeds()

        # Decode
        results = []
        for output_seq in finished:
            completion_ids = output_seq.completion_token_ids
            results.append(
                self.tokenizer.decode(completion_ids, skip_special_tokens=True)
            )
        return results

    def exit(self):
        """Clean up resources."""
        self.llm_engine.exit()
