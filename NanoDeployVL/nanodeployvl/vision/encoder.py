"""Vision encoder for Qwen3.5-MoE VLM.

Loads the ``visual.*`` weights from the model checkpoint and runs the
Qwen3VL-style ViT encoder on the driver process.  The encoder produces
image/video embeddings that are merged with text embeddings before the
LLM forward pass.

Design notes for future optimisation
-------------------------------------
- The encoder currently runs on a single device (``vision_device``).
  To add data-parallel sharding across GPUs, wrap ``encode()`` with a
  sharding routine similar to sglang's
  ``run_dp_sharded_mrope_vision_model()``.
- To move the encoder to a dedicated Ray actor, create a
  ``VisionEncoderActor`` that calls ``self.encoder.encode()`` remotely.
"""

from __future__ import annotations

import math
import os
from glob import glob
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanodeploy.logging import get_logger
from safetensors import safe_open
from tqdm import tqdm

logger = get_logger("nanodeployvl")


# ---------------------------------------------------------------------------
# Vision building blocks (following HuggingFace Qwen3VL / Qwen3.5-MoE)
# ---------------------------------------------------------------------------


class VisionRotaryEmbedding(nn.Module):
    """Rotary position embedding for the vision encoder."""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


class VisionMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        act_fn: str = "gelu_pytorch_tanh",
    ):
        super().__init__()
        self.linear_fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)
        if act_fn == "gelu_pytorch_tanh":
            self.act_fn = nn.GELU(approximate="tanh")
        else:
            self.act_fn = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class VisionPatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int,
        temporal_patch_size: int,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # kernel_size == stride, so each patch maps to exactly one output position.
        # Equivalent to Conv3d but expressed as a single matmul — much faster for
        # large patch counts (e.g. 4800 patches from a 1280×960 image).
        x = x.to(dtype=self.proj.weight.dtype)
        w = self.proj.weight.flatten(1)  # [embed_dim, in_ch*t*patch*patch]
        return F.linear(x, w, self.proj.bias)  # [N, embed_dim]


class VisionPatchMerger(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        spatial_merge_size: int,
        out_hidden_size: int,
        use_postshuffle_norm: bool = False,
    ):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        merged_dim = hidden_size * (spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(
            merged_dim if use_postshuffle_norm else hidden_size, eps=1e-6
        )
        self.linear_fc1 = nn.Linear(merged_dim, merged_dim)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(merged_dim, out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        merged_dim = self.linear_fc1.in_features
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, merged_dim)).view(-1, merged_dim)
        else:
            x = self.norm(x).view(-1, merged_dim)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


class VisionAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states).reshape(
            seq_length, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.permute(1, 0, 2, 3).unbind(0)

        cos, sin = position_embeddings
        q, k = _apply_rotary_pos_emb_vision(q, k, cos, sin)

        # Process each variable-length sequence separately
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        q_splits = torch.split(q, lengths, dim=0)
        k_splits = torch.split(k, lengths, dim=0)
        v_splits = torch.split(v, lengths, dim=0)

        outputs = []
        for qs, ks, vs in zip(q_splits, k_splits, v_splits):
            # [seq, heads, dim] -> [1, heads, seq, dim]
            qs = qs.transpose(0, 1).unsqueeze(0)
            ks = ks.transpose(0, 1).unsqueeze(0)
            vs = vs.transpose(0, 1).unsqueeze(0)
            out = F.scaled_dot_product_attention(
                qs, ks, vs, scale=self.scaling, is_causal=False
            )
            out = out.squeeze(0).transpose(0, 1)  # [seq, heads, dim]
            outputs.append(out)

        attn_output = torch.cat(outputs, dim=0).reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)


class VisionBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        act_fn: str = "gelu_pytorch_tanh",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = VisionAttention(hidden_size, num_heads)
        self.mlp = VisionMLP(hidden_size, intermediate_size, act_fn)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens, position_embeddings
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


# ---------------------------------------------------------------------------
# Full Vision Model
# ---------------------------------------------------------------------------


class VisionModel(nn.Module):
    """Qwen3VL-style vision transformer encoder.

    This mirrors the ``Qwen3_5MoeVisionModel`` from HuggingFace Transformers
    but is standalone (no HF base class dependency) so it can be loaded
    independently on the driver process.

    Parameters are named identically to the checkpoint so that weight
    loading is straightforward.
    """

    def __init__(self, vision_config) -> None:
        super().__init__()
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.patch_size = vision_config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size**2

        hidden_size = vision_config.hidden_size
        num_heads = vision_config.num_heads
        depth = vision_config.depth
        intermediate_size = vision_config.intermediate_size
        out_hidden_size = vision_config.out_hidden_size
        in_channels = vision_config.in_channels
        temporal_patch_size = vision_config.temporal_patch_size
        act_fn = getattr(vision_config, "hidden_act", "gelu_pytorch_tanh")

        self.patch_embed = VisionPatchEmbed(
            in_channels, hidden_size, self.patch_size, temporal_patch_size
        )
        self.pos_embed = nn.Embedding(
            getattr(vision_config, "num_position_embeddings", 2304), hidden_size
        )
        self.num_grid_per_side = int(
            getattr(vision_config, "num_position_embeddings", 2304) ** 0.5
        )

        head_dim = hidden_size // num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [
                VisionBlock(hidden_size, num_heads, intermediate_size, act_fn)
                for _ in range(depth)
            ]
        )
        self.merger = VisionPatchMerger(
            hidden_size,
            self.spatial_merge_size,
            out_hidden_size,
            use_postshuffle_norm=False,
        )

        # Deepstack mergers (for models that extract intermediate features)
        deepstack_indexes = getattr(vision_config, "deepstack_visual_indexes", [])
        self.deepstack_visual_indexes = (
            list(deepstack_indexes) if deepstack_indexes else []
        )
        self.deepstack_merger_list = nn.ModuleList(
            [
                VisionPatchMerger(
                    hidden_size,
                    self.spatial_merge_size,
                    out_hidden_size,
                    use_postshuffle_norm=True,
                )
                for _ in range(len(self.deepstack_visual_indexes))
            ]
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    # -- Position embedding helpers --

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        grid_thw_list = grid_thw.tolist()
        max_hw = max(max(h, w) for _, h, w in grid_thw_list)
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device

        total_tokens = sum(int(t * h * w) for t, h, w in grid_thw_list)
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw_list:
            num_frames, height, width = int(num_frames), int(height), int(width)
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            row_idx = (
                block_rows[:, None, None, None] * merge_size
                + intra_row[None, None, :, None]
            )
            col_idx = (
                block_cols[None, :, None, None] * merge_size
                + intra_col[None, None, None, :]
            )
            row_idx = row_idx.expand(
                merged_h, merged_w, merge_size, merge_size
            ).reshape(-1)
            col_idx = col_idx.expand(
                merged_h, merged_w, merge_size, merge_size
            ).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]
        return embeddings.flatten(1)

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Bilinear interpolation of position embeddings for variable sizes."""
        grid_thw_list = grid_thw.tolist()
        grid_ts = [int(row[0]) for row in grid_thw_list]
        grid_hs = [int(row[1]) for row in grid_thw_list]
        grid_ws = [int(row[2]) for row in grid_thw_list]
        device = self.pos_embed.weight.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in grid_thw_list:
            h, w = int(h), int(w)
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_floor = h_idxs.int()
            w_floor = w_idxs.int()
            h_ceil = (h_floor + 1).clamp(max=self.num_grid_per_side - 1)
            w_ceil = (w_floor + 1).clamp(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_floor.float()
            dw = w_idxs - w_floor.float()

            base_h = h_floor * self.num_grid_per_side
            base_h_ceil = h_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_floor[None]).flatten(),
                (base_h[None].T + w_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_floor[None]).flatten(),
                (base_h_ceil[None].T + w_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=device
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split(
            [h * w for h, w in zip(grid_hs, grid_ws)]
        )

        merge_size = self.spatial_merge_size
        result = []
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(
                    t, h // merge_size, merge_size, w // merge_size, merge_size, -1
                )
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            result.append(pos_embed)
        return torch.cat(result)

    # -- Main forward --

    @torch.inference_mode()
    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Encode pixel values through the ViT.

        Args:
            pixel_values: Preprocessed image patches, shape varies by
                input format (see HF Qwen2VLImageProcessorFast).
            grid_thw: ``(num_images, 3)`` tensor of (temporal, height, width)
                patch grid dimensions.

        Returns:
            Merged image embeddings of shape
            ``(total_tokens_after_merge, out_hidden_size)``.
        """
        hidden_states = self.patch_embed(pixel_values)
        hidden_states = hidden_states + self.fast_pos_embed_interpolate(grid_thw)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        seq_len = hidden_states.size(0)
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens, position_embeddings)

        return self.merger(hidden_states)


# ---------------------------------------------------------------------------
# High-level encoder wrapper
# ---------------------------------------------------------------------------


class VisionEncoder:
    """High-level wrapper that builds a ``VisionModel``, loads weights
    from a checkpoint, and provides a simple ``encode()`` API.

    The encoder is designed to run on the driver process (not inside Ray
    workers).  Future optimisation: move to a dedicated ``VisionEncoderActor``.

    Parameters
    ----------
    vision_config
        The ``vision_config`` object from the HF model config.
    model_path : str
        Path to the model checkpoint directory (contains safetensors).
    device : str
        Torch device string, e.g. ``"cuda:0"``.
    dtype : torch.dtype
        Weight / compute dtype for the vision encoder.
    """

    def __init__(
        self,
        vision_config,
        model_path: str,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.spatial_merge_size = vision_config.spatial_merge_size

        logger.info("Building VisionModel …")
        self.model = VisionModel(vision_config)
        self.model.to(device=self.device, dtype=self.dtype)

        self._load_weights(model_path)
        self.model.eval()
        self._warmup(vision_config)
        logger.info("VisionEncoder ready on %s (%s)", device, dtype)

    # -- Warmup --

    @torch.inference_mode()
    def _warmup(self, vision_config) -> None:
        """Run a dummy forward pass to trigger CUDA kernel JIT compilation."""
        logger.info("Warming up VisionEncoder …")
        in_ch = getattr(vision_config, "in_channels", 3)
        t_patch = getattr(vision_config, "temporal_patch_size", 2)
        patch = getattr(vision_config, "patch_size", 14)
        # Minimal input: 1 image, 1 temporal frame, 2×2 patch grid
        grid_thw = torch.tensor([[1, 2, 2]], device=self.device)
        n_patches = 1 * 2 * 2  # T * H * W
        patch_dim = in_ch * t_patch * patch * patch
        dummy_pv = torch.zeros(
            n_patches, patch_dim, device=self.device, dtype=self.dtype
        )
        self.model(dummy_pv, grid_thw)
        torch.cuda.synchronize(self.device)
        logger.info("VisionEncoder warmup done")

    # -- Weight loading --

    def _load_weights(self, path: str) -> None:
        """Load ``visual.*`` weights from safetensors checkpoint.

        The checkpoint stores vision weights under ``model.visual.*``.
        We strip that prefix to match our module names.
        """
        weight_files = sorted(glob(os.path.join(path, "*.safetensors")))
        loaded = 0
        skipped = 0

        for wf in tqdm(weight_files, desc="Loading vision weights"):
            with safe_open(wf, "pt", "cpu") as f:
                for raw_name in f.keys():
                    # Only load visual weights
                    if not raw_name.startswith("model.visual."):
                        continue
                    # Strip prefix: model.visual.blocks.0.attn.qkv.weight → blocks.0.attn.qkv.weight
                    param_name = raw_name[len("model.visual.") :]
                    try:
                        param = self.model.get_parameter(param_name)
                    except AttributeError:
                        skipped += 1
                        continue
                    tensor = f.get_tensor(raw_name).to(
                        device=self.device, dtype=self.dtype
                    )
                    param.data.copy_(tensor)
                    loaded += 1

        logger.info("Vision weights: %d loaded, %d skipped", loaded, skipped)

    # -- Encoding API --

    @torch.inference_mode()
    def encode(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Encode images into embeddings.

        Args:
            pixel_values: Preprocessed patches from the image processor.
            image_grid_thw: ``(num_images, 3)`` grid dimensions.

        Returns:
            List of tensors, one per image, each of shape
            ``(num_merged_tokens, hidden_size)``.
        """
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        image_grid_thw = image_grid_thw.to(device=self.device)

        image_embeds = self.model(pixel_values, image_grid_thw)

        # Split into per-image tensors
        split_sizes = (
            image_grid_thw.prod(dim=-1) // (self.spatial_merge_size**2)
        ).tolist()
        return list(torch.split(image_embeds, split_sizes))

    @torch.inference_mode()
    def encode_video(
        self,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Encode videos (same pipeline, different grid dimensions)."""
        return self.encode(pixel_values_videos, video_grid_thw)
