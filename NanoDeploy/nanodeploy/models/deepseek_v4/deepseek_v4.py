import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from nanodeploy.backends import get_backend
from nanodeploy.backends.gpu_generic.kernels.kv_store import store_kvcache
from nanodeploy.context.context import get_context
from nanodeploy.context.distributed import get_dist_context
from nanodeploy.layers.activation import SiluAndMul
from nanodeploy.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanodeploy.layers.layernorm import RMSNorm
from nanodeploy.layers.rotary_embedding import get_rope
from nanodeploy.models.deepseek_v2.deepseek_v2 import DeepseekV2MLP
from nanodeploy.models.quant_config import QuantizationConfig


def _getattr_any(config, *names, default=None):
    for name in names:
        if hasattr(config, name):
            value = getattr(config, name)
            if value is not None:
                return value
    return default


def _debug_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _debug_layer_enabled(layer_idx: int | None) -> bool:
    if layer_idx is None:
        return True
    layers = os.getenv("NANODEPLOY_DSV4_DEBUG_LAYERS", "0")
    if layers.strip().lower() in ("all", "*"):
        return True
    enabled = {int(x) for x in layers.replace(",", " ").split() if x.strip()}
    return layer_idx in enabled


def _debug_dump(name: str, tensor: torch.Tensor, layer_idx: int | None = None) -> None:
    out_dir = os.getenv("NANODEPLOY_DSV4_DEBUG_DIR")
    if not out_dir or not torch.is_tensor(tensor):
        return
    rank = _debug_rank()
    rank_filter = os.getenv("NANODEPLOY_DSV4_DEBUG_RANK", "0")
    if rank_filter.strip().lower() not in ("all", "*") and rank != int(rank_filter):
        return
    if not _debug_layer_enabled(layer_idx):
        return
    if os.getenv("NANODEPLOY_DSV4_DEBUG_PREFILL_ONLY", "1") != "0":
        try:
            context = get_context()
            if not context.is_prefill:
                return
            if (
                os.getenv("NANODEPLOY_DSV4_DEBUG_SKIP_DUMMY", "1") != "0"
                and context.is_dummy
            ):
                return
        except Exception:
            pass

    max_tokens = int(os.getenv("NANODEPLOY_DSV4_DEBUG_MAX_TOKENS", "8"))
    payload = tensor.detach()
    if payload.ndim > 0:
        payload = payload[:max_tokens]
    payload = payload.cpu().contiguous()
    layer = "global" if layer_idx is None else f"layer{layer_idx}"
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / f"nanodeploy_rank{rank}_{layer}_{name}.pt"
    if os.getenv("NANODEPLOY_DSV4_DEBUG_ONCE", "1") != "0" and file_path.exists():
        return
    torch.save(
        {
            "name": name,
            "rank": rank,
            "layer": layer_idx,
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "tensor": payload,
        },
        file_path,
    )


def _apply_rotary_interleaved(
    rotary_emb: nn.Module,
    positions: torch.Tensor,
    x: torch.Tensor,
    inverse: bool = False,
):
    cos_sin = rotary_emb.cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    x_pair = x.float().unflatten(-1, (-1, 2))
    x0 = x_pair[..., 0]
    x1 = x_pair[..., 1]
    if inverse:
        y0 = x0 * cos + x1 * sin
        y1 = x1 * cos - x0 * sin
    else:
        y0 = x0 * cos - x1 * sin
        y1 = x1 * cos + x0 * sin
    return torch.stack((y0, y1), dim=-1).flatten(-2).to(x.dtype)


def _fp8_quant_dequant_inplace(x: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """Simulate official DSV4 KV FP8 QAT quant-dequant in-place."""
    if x.numel() == 0:
        return x
    orig_shape = x.shape
    assert orig_shape[-1] % block_size == 0
    view = (
        x.reshape(-1, orig_shape[-1])
        .float()
        .view(-1, orig_shape[-1] // block_size, block_size)
    )
    amax = view.abs().amax(dim=-1).clamp(min=1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0))).unsqueeze(-1)
    quant = (view / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    dequant = (quant.float() * scale).view(orig_shape).to(x.dtype)
    x.copy_(dequant)
    return x


# ---------------------------------------------------------------------------
# FP8 packed KV cache helpers for flash_mla DSv4 sparse decode
# Layout per token inside a page buffer (uint8):
#   [0 .. 447]        : 448 FP8 E4M3FN  (nope dims)
#   [448 .. 575]       : 64 BF16 = 128 bytes  (rope dims)
#   NOTE: nope+rope are stored interleaved per-token in the FIRST part of the
#         page.  Scales are stored in a SECOND contiguous block AFTER all tokens'
#         nope+rope data.
# Page layout (uint8 flat):
#   bytes [0 .. page_size*576 - 1]             : nope+rope for all tokens
#   bytes [page_size*576 .. page_size*584 - 1] : scales (7+1 padded) per token
# Total bytes per page = page_size * 584
# ---------------------------------------------------------------------------

_DSV4_NOPE_DIM = 448
_DSV4_ROPE_DIM = 64
_DSV4_TILE_SIZE = 64
_DSV4_NUM_TILES = _DSV4_NOPE_DIM // _DSV4_TILE_SIZE  # 7
_DSV4_NOPE_ROPE_BYTES = _DSV4_NOPE_DIM + _DSV4_ROPE_DIM * 2  # 576
_DSV4_SCALE_PADDED = _DSV4_NUM_TILES + 1  # 8
_DSV4_BYTES_PER_TOKEN = _DSV4_NOPE_ROPE_BYTES + _DSV4_SCALE_PADDED  # 584


def _pack_kv_fp8(
    kv_bf16: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize KV [T, 512] BF16 → (nope_fp8[T,448], rope_bf16[T,64], scales_u8[T,7]).

    Uses power-of-2 (UE8M0) per-64-tile scales, matching official DSv4 / SGLang.
    """
    assert kv_bf16.dtype == torch.bfloat16 and kv_bf16.shape[-1] == 512
    nope, rope = kv_bf16.split([_DSV4_NOPE_DIM, _DSV4_ROPE_DIM], dim=-1)

    # Per-tile FP8 quantization with power-of-2 scales
    x = nope.contiguous().reshape(-1, _DSV4_NUM_TILES, _DSV4_TILE_SIZE).float()
    amax = x.abs().amax(dim=-1).clamp(min=1e-8)  # [T, 7]
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    scale_fp32 = torch.exp2(torch.ceil(torch.log2(amax / fp8_max)))  # [T, 7]
    nope_fp8 = (
        (x / scale_fp32.unsqueeze(-1)).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    )
    nope_fp8 = nope_fp8.reshape(-1, _DSV4_NOPE_DIM)

    # UE8M0 encoding: uint8 = exponent + 127
    exponent = torch.ceil(torch.log2(amax / fp8_max)).to(torch.int32)
    scales_u8 = (exponent + 127).to(torch.uint8)  # [T, 7]

    return nope_fp8, rope.contiguous(), scales_u8


def _store_dsv4_fp8_batched(
    kv_bf16: torch.Tensor,
    cache_buf: torch.Tensor,
    slot_mapping: torch.Tensor,
    page_size: int,
) -> None:
    """Pack KV and write into paged cache buffer using page-level split layout.

    flash_mla MODEL1 kernel reads data as:
      - nope+rope at: page_base + tok_in_page * 576
      - scales  at: page_base + page_size * 576 + tok_in_page * 8
    Total page = page_size * 584 bytes.

    The torch shape [num_pages, page_size, 1, 584] is for flash_mla stride
    validation only; the actual byte layout within a page is split.

    Args:
        kv_bf16: [T, 512] bfloat16
        cache_buf: [num_pages, page_size, 1, 584] uint8 — MUST include a dummy
                   last slot for graph-safe invalid writes.
        slot_mapping: [T] int32/int64. Slot = -1 redirects to the dummy last
                      slot (avoids data-dependent control flow for CUDAGraph).
    """
    nope_fp8, rope_bf16, scales_u8 = _pack_kv_fp8(kv_bf16)
    T = kv_bf16.shape[0]

    # Graph-safe: redirect negative slots to the dummy last slot.
    # Caller must ensure cache_buf has an extra dummy slot at the end.
    num_pages = cache_buf.shape[0]
    total_slots = num_pages * page_size
    dummy_slot = total_slots - 1
    # Use scalar third arg (not torch.tensor) — CUDAGraph-safe (no H2D).
    slots = torch.where(slot_mapping >= 0, slot_mapping.long(), dummy_slot)

    page_idx = slots // page_size
    tok_in_page = slots % page_size

    # Flatten to byte view
    bytes_per_page = page_size * _DSV4_BYTES_PER_TOKEN
    buf_flat = cache_buf.reshape(-1)  # [total_bytes]

    # Byte offsets for nope+rope (first block in page)
    nope_rope_base = page_idx * bytes_per_page + tok_in_page * _DSV4_NOPE_ROPE_BYTES
    # Byte offsets for scales (second block in page)
    s_page_offset = page_size * _DSV4_NOPE_ROPE_BYTES
    scale_base = (
        page_idx * bytes_per_page + s_page_offset + tok_in_page * _DSV4_SCALE_PADDED
    )

    # Build per-token nope||rope (576 bytes) and scatter
    nope_rope = torch.cat(
        [nope_fp8.view(torch.uint8), rope_bf16.view(torch.uint8)], dim=-1
    )  # [T, 576]

    # Scatter nope_rope bytes — fixed shape for graph capture
    nope_rope_offsets = nope_rope_base.unsqueeze(1) + torch.arange(
        _DSV4_NOPE_ROPE_BYTES, device=kv_bf16.device
    )  # [T, 576]
    buf_flat[nope_rope_offsets.reshape(-1)] = nope_rope.reshape(-1)

    # Scatter scale bytes (7 per token, padded to 8)
    scale_offsets = scale_base.unsqueeze(1) + torch.arange(
        _DSV4_NUM_TILES, device=kv_bf16.device
    )  # [T, 7]
    buf_flat[scale_offsets.reshape(-1)] = scales_u8.reshape(-1)


class DeepseekV4HCProjector(nn.Module):
    """Hyper-Connection mixing used by DeepSeek-V4.

    NanoDeploy keeps the sequence dimension flattened, so tensors are shaped
    [T, hc_mult, hidden] instead of the official demo's [B, S, hc_mult, hidden].
    """

    def __init__(self, hidden_size: int, hc_mult: int, sinkhorn_iters: int, eps: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * hidden_size
        self.fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape, dtype = x.shape, x.dtype
        x_flat = x.flatten(1).float()
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.eps)
        mixes = F.linear(x_flat, self.fn) * rsqrt

        hc = self.hc_mult
        pre = torch.sigmoid(mixes[:, :hc] * self.scale[0] + self.base[:hc]) + self.eps
        post = 2 * torch.sigmoid(
            mixes[:, hc : 2 * hc] * self.scale[1] + self.base[hc : 2 * hc]
        )
        comb = mixes[:, 2 * hc :].view(-1, hc, hc) * self.scale[2] + self.base[
            2 * hc :
        ].view(hc, hc)

        comb = comb.softmax(-1) + self.eps
        comb = comb / (comb.sum(-2, keepdim=True) + self.eps)
        for _ in range(max(0, self.sinkhorn_iters - 1)):
            comb = comb / (comb.sum(-1, keepdim=True) + self.eps)
            comb = comb / (comb.sum(-2, keepdim=True) + self.eps)

        y = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=1)
        return y.to(dtype), post.to(dtype), comb.to(dtype)


class DeepseekV4HCHead(nn.Module):
    def __init__(self, hidden_size: int, hc_mult: int, eps: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.eps = eps
        self.fn = nn.Parameter(
            torch.empty(hc_mult, hc_mult * hidden_size, dtype=torch.float32)
        )
        self.base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape, dtype = x.shape, x.dtype
        x_flat = x.flatten(1).float()
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.eps)
        mixes = F.linear(x_flat, self.fn) * rsqrt
        pre = torch.sigmoid(mixes * self.scale + self.base) + self.eps
        y = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=1)
        return y.to(dtype)


class _FloatLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x.float(), self.weight)


class DeepseekV4Compressor(nn.Module):
    """Reference-compatible prefill KV compressor for DSV4 compressed layers.

    State is tensorized for CUDAGraph compatibility:
    - _kv_states / _score_states: [max_slots, coeff*ratio, coeff*head_dim] fp32
    - Compressed output writes to external compressed_cache (FP8 packed) via
      _compressed_counts tracking.
    - Fallback: dict-based _states/_compressed_cache for backward compat when
      tensorized buffers are not allocated (e.g., during weight loading).
    """

    def __init__(self, config, compress_ratio: int, head_dim: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        coeff = 2 if self.overlap else 1

        self.ape = nn.Parameter(
            torch.empty(compress_ratio, coeff * head_dim, dtype=torch.float32)
        )
        self.wkv = _FloatLinear(self.hidden_size, coeff * head_dim)
        self.wgate = _FloatLinear(self.hidden_size, coeff * head_dim)
        self.norm = RMSNorm(head_dim, eps=config.rms_norm_eps)

        # Tensorized state (allocated by init_tensorized_state, None until then)
        self._kv_states: torch.Tensor | None = (
            None  # [max_slots, coeff*ratio, coeff*hd]
        )
        self._score_states: torch.Tensor | None = None  # same shape
        self._compressed_counts: torch.Tensor | None = None  # [max_slots] int32
        self._max_slots: int = 0

        # Dict-based fallback (used during prefill / before tensorization)
        self._states: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._compressed_cache: dict[int, torch.Tensor] = {}

    def init_tensorized_state(
        self,
        max_slots: int,
        device: torch.device,
        kv_view: torch.Tensor | None = None,
        score_view: torch.Tensor | None = None,
        counts_view: torch.Tensor | None = None,
    ):
        """Pre-allocate tensorized state buffers for CUDAGraph-safe decode.

        Buffer has ``max_slots + 1`` rows — the extra row at index ``max_slots``
        is a "dummy" slot. Batch positions with no valid state (e.g., during
        CUDAGraph warmup where slots are unassigned) are routed to the dummy
        so scatter writes never land in real state.

        S2.2: when called with kv_view / score_view / counts_view arguments
        (slices into the per-ratio flat buffers in CacheContext), we use those
        views directly so RDMA migration can target one MR per ratio.  When
        the views are None (no PD disagg / standalone test), allocate fresh.
        """
        coeff = 2 if self.overlap else 1
        ratio = self.compress_ratio
        self._max_slots = max_slots
        num_rows = max_slots + 1  # +1 dummy
        if kv_view is not None and score_view is not None and counts_view is not None:
            assert kv_view.shape == (num_rows, coeff * ratio, coeff * self.head_dim)
            assert score_view.shape == kv_view.shape
            assert counts_view.shape == (num_rows,)
            self._kv_states = kv_view
            self._score_states = score_view
            self._compressed_counts = counts_view
            # Initialize values explicitly (views may be reused across init calls).
            self._kv_states.zero_()
            self._score_states.fill_(float("-inf"))
            self._compressed_counts.zero_()
        else:
            self._kv_states = torch.zeros(
                num_rows,
                coeff * ratio,
                coeff * self.head_dim,
                dtype=torch.float32,
                device=device,
            )
            self._score_states = torch.full(
                (num_rows, coeff * ratio, coeff * self.head_dim),
                float("-inf"),
                dtype=torch.float32,
                device=device,
            )
            self._compressed_counts = torch.zeros(
                num_rows,
                dtype=torch.int32,
                device=device,
            )

    def _overlap_transform(self, tensor: torch.Tensor, value: float) -> torch.Tensor:
        # tensor: [num_blocks, ratio, 2 * head_dim]
        num_blocks = tensor.size(0)
        ratio, head_dim = self.compress_ratio, self.head_dim
        out = tensor.new_full((num_blocks, 2 * ratio, head_dim), value)
        out[:, ratio:] = tensor[:, :, head_dim:]
        out[1:, :ratio] = tensor[:-1, :, :head_dim]
        return out

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
        seq_key: int | None = None,
    ) -> torch.Tensor | None:
        seqlen = hidden_states.size(0)
        ratio = self.compress_ratio
        coeff = 2 if self.overlap else 1
        # Stage 4 — clear any stale state at this slot from a previously-evicted
        # owner. State slots are recycled by the scheduler's GDNStateManager;
        # without this reset, short prompts (seqlen < ratio) that don't fire
        # compression would inherit the previous seq's _compressed_counts and
        # then read garbage compressed pages from the pool.
        if (
            seq_key is not None
            and self._compressed_counts is not None
            and 0 <= seq_key < self._max_slots
        ):
            self._compressed_counts[seq_key] = 0
            self._kv_states[seq_key].zero_()
            self._score_states[seq_key].fill_(float("-inf"))
        # Dict-based fallback path: also clear any stale dict entries.
        if seq_key is not None:
            self._compressed_cache.pop(seq_key, None)
            self._states.pop(seq_key, None)
        kv_state = torch.zeros(
            coeff * ratio,
            coeff * self.head_dim,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        score_state = torch.full_like(kv_state, float("-inf"))

        dtype = hidden_states.dtype
        cutoff = seqlen - (seqlen % ratio)
        kv_all = self.wkv(hidden_states)
        score_all = self.wgate(hidden_states)
        offset = ratio if self.overlap else 0
        if self.overlap and cutoff >= ratio:
            kv_state[:ratio] = kv_all[cutoff - ratio : cutoff]
            score_state[:ratio] = score_all[cutoff - ratio : cutoff] + self.ape
        remainder = seqlen - cutoff
        if remainder > 0:
            kv_state[offset : offset + remainder] = kv_all[cutoff:]
            score_state[offset : offset + remainder] = (
                score_all[cutoff:] + self.ape[:remainder]
            )
        if seq_key is not None:
            if self._kv_states is not None and 0 <= seq_key < self._max_slots:
                self._kv_states[seq_key] = kv_state
                self._score_states[seq_key] = score_state
            else:
                self._states[seq_key] = (kv_state, score_state)
        if cutoff == 0:
            return None

        kv = kv_all[:cutoff].unflatten(0, (-1, ratio))
        score = score_all[:cutoff].unflatten(0, (-1, ratio)) + self.ape
        if self.overlap:
            kv = self._overlap_transform(kv, 0.0)
            score = self._overlap_transform(score, float("-inf"))
        kv = (kv * score.softmax(dim=1)).sum(dim=1)
        kv = self.norm(kv.to(dtype))

        rd = self.rope_head_dim
        compressed_positions = positions[:cutoff:ratio]
        kv[:, -rd:] = _apply_rotary_interleaved(
            rotary_emb,
            compressed_positions,
            kv[:, None, -rd:],
        ).squeeze(1)
        _fp8_quant_dequant_inplace(kv[:, :-rd], 64)
        if seq_key is not None:
            self._compressed_cache[seq_key] = kv
            # Also update compressed count for tensorized path
            if self._compressed_counts is not None and 0 <= seq_key < self._max_slots:
                self._compressed_counts[seq_key] = kv.shape[0]
        return kv

    def forward_decode(
        self,
        hidden_state: torch.Tensor,
        position: int,
        rotary_emb: nn.Module,
        seq_key: int,
    ) -> torch.Tensor | None:
        ratio = self.compress_ratio
        coeff = 2 if self.overlap else 1

        # Use tensorized state if available
        use_tensor = self._kv_states is not None and 0 <= seq_key < self._max_slots
        if use_tensor:
            kv_state = self._kv_states[seq_key]
            score_state = self._score_states[seq_key]
        else:
            state = self._states.get(seq_key)
            if state is None:
                kv_state = torch.zeros(
                    coeff * ratio,
                    coeff * self.head_dim,
                    dtype=torch.float32,
                    device=hidden_state.device,
                )
                score_state = torch.full_like(kv_state, float("-inf"))
                state = (kv_state, score_state)
                self._states[seq_key] = state
            kv_state, score_state = state

        pos_mod = position % ratio
        kv = self.wkv(hidden_state).squeeze(0)
        score = self.wgate(hidden_state).squeeze(0) + self.ape[pos_mod]
        should_compress = (position + 1) % ratio == 0
        compressed = None
        if self.overlap:
            kv_state[ratio + pos_mod] = kv
            score_state[ratio + pos_mod] = score
            if should_compress:
                kv_for_compress = torch.cat(
                    [
                        kv_state[:ratio, : self.head_dim],
                        kv_state[ratio:, self.head_dim :],
                    ],
                    dim=0,
                )
                score_for_compress = torch.cat(
                    [
                        score_state[:ratio, : self.head_dim],
                        score_state[ratio:, self.head_dim :],
                    ],
                    dim=0,
                )
                compressed = (kv_for_compress * score_for_compress.softmax(dim=0)).sum(
                    dim=0, keepdim=True
                )
                kv_state[:ratio] = kv_state[ratio:]
                score_state[:ratio] = score_state[ratio:]
        else:
            kv_state[pos_mod] = kv
            score_state[pos_mod] = score
            if should_compress:
                compressed = (kv_state * score_state.softmax(dim=0)).sum(
                    dim=0,
                    keepdim=True,
                )

        if compressed is None:
            return None
        dtype = hidden_state.dtype
        compressed = self.norm(compressed.to(dtype))
        rd = self.rope_head_dim
        compressed_pos = hidden_state.new_tensor(
            [position + 1 - ratio], dtype=torch.long
        )
        compressed[:, -rd:] = _apply_rotary_interleaved(
            rotary_emb,
            compressed_pos,
            compressed[:, None, -rd:],
        ).squeeze(1)
        _fp8_quant_dequant_inplace(compressed[:, :-rd], 64)

        # Update compressed cache
        existing = self._compressed_cache.get(seq_key)
        self._compressed_cache[seq_key] = (
            compressed if existing is None else torch.cat([existing, compressed], dim=0)
        )
        # Update tensorized compressed count
        if self._compressed_counts is not None and 0 <= seq_key < self._max_slots:
            self._compressed_counts[seq_key] += 1
        return compressed

    def forward_decode_batched(
        self,
        hidden_states: torch.Tensor,  # [bs, hidden_size]
        positions: torch.Tensor,  # [bs] int64
        rotary_emb: nn.Module,
        seq_slots: torch.Tensor,  # [bs] int64 - slot index in _kv_states
        compressed_cache: torch.Tensor | None,
        compressed_block_table: (
            torch.Tensor | None
        ) = None,  # [num_seqs_active, max_blocks] int32 (paged)
    ) -> None:
        """CUDAGraph-safe batched compressor decode.

        Always runs state update + compression compute for all bs sequences.
        Uses a dummy slot to absorb writes for sequences not triggering
        compression this step (avoids data-dependent control flow).
        """
        ratio = self.compress_ratio
        coeff = 2 if self.overlap else 1
        bs = hidden_states.shape[0]
        head_dim = self.head_dim
        dtype = hidden_states.dtype
        device = hidden_states.device

        # Projections (already batched)
        kv_all = self.wkv(hidden_states)  # [bs, coeff*head_dim]
        score_all = self.wgate(hidden_states)  # [bs, coeff*head_dim]

        pos_mod = (positions % ratio).long()  # [bs]
        pos_mod_i = pos_mod.to(torch.int64)

        # Batched state update: scatter at (seq_slot, slot_offset)
        ape_vals = self.ape[pos_mod_i]  # [bs, coeff*head_dim]
        if self.overlap:
            update_idx = ratio + pos_mod_i  # [bs]
        else:
            update_idx = pos_mod_i
        # Scatter into _kv_states and _score_states
        self._kv_states[seq_slots.long(), update_idx] = kv_all.float()
        self._score_states[seq_slots.long(), update_idx] = score_all.float() + ape_vals

        # Compute compression for all bs sequences (batched)
        kv_st = self._kv_states[seq_slots.long()]  # [bs, coeff*ratio, coeff*head_dim]
        score_st = self._score_states[seq_slots.long()]  # same

        if self.overlap:
            # Reconstruct [bs, 2*ratio, head_dim] via gather-cat
            kv_for_c = torch.cat(
                [kv_st[:, :ratio, :head_dim], kv_st[:, ratio:, head_dim:]], dim=1
            )  # [bs, 2*ratio, head_dim]
            score_for_c = torch.cat(
                [score_st[:, :ratio, :head_dim], score_st[:, ratio:, head_dim:]], dim=1
            )
            compressed = (kv_for_c * score_for_c.softmax(dim=1)).sum(
                dim=1
            )  # [bs, head_dim]
        else:
            # kv_st: [bs, ratio, head_dim], score_st same
            compressed = (kv_st * score_st.softmax(dim=1)).sum(dim=1)  # [bs, head_dim]

        # Apply norm + RoPE + FP8 QAT (batched)
        compressed = self.norm(compressed.to(dtype))  # [bs, head_dim]
        rd = self.rope_head_dim
        compressed_pos = (positions + 1 - ratio).clamp(min=0)  # [bs]
        # NOTE: cos_sin_cache has shape [max_pos, 1, rd] (singleton head dim),
        # so we must pass x with a head dim: [bs, 1, rd], not [bs, rd].
        compressed_rope = _apply_rotary_interleaved(
            rotary_emb, compressed_pos, compressed[:, None, -rd:]
        ).squeeze(
            1
        )  # [bs, rd]
        compressed = torch.cat([compressed[:, :-rd], compressed_rope], dim=-1)
        # In-place FP8 QAT on the nope portion (safe: compressed is a fresh tensor)
        _fp8_quant_dequant_inplace(compressed[:, :-rd], 64)

        # Which sequences actually compress this step?
        should_compress = (positions + 1) % ratio == 0  # [bs] bool

        # Post-shift for overlap case: kv_state[:ratio] = kv_state[ratio:]
        # Apply conditionally via torch.where (graph-safe)
        if self.overlap:
            sc_mask = should_compress.unsqueeze(-1).unsqueeze(-1)  # [bs, 1, 1]
            shifted_kv = self._kv_states[
                seq_slots.long(), ratio:
            ]  # [bs, ratio, coeff*head_dim]
            shifted_score = self._score_states[seq_slots.long(), ratio:]
            current_kv_head = self._kv_states[seq_slots.long(), :ratio]
            current_score_head = self._score_states[seq_slots.long(), :ratio]
            new_kv_head = torch.where(sc_mask, shifted_kv, current_kv_head)
            new_score_head = torch.where(sc_mask, shifted_score, current_score_head)
            self._kv_states[seq_slots.long(), :ratio] = new_kv_head
            self._score_states[seq_slots.long(), :ratio] = new_score_head

        # Write compressed output to FP8 cache (dummy slot absorbs invalid writes)
        if compressed_cache is not None and self._compressed_counts is not None:
            # compressed_cache shape: [num_pages+1, page_size, 1, 584]
            num_pages = compressed_cache.shape[0]
            page_size = compressed_cache.shape[1]
            total_slots = num_pages * page_size  # token-level slots
            cur_counts = self._compressed_counts[seq_slots.long()].long()
            if compressed_block_table is not None:
                # Paged addressing: page_id = block_table[state_slot, block_idx]
                # block_idx = count // page_size; tok_in_block = count % page_size
                block_idx = (cur_counts // page_size).long()  # [bs]
                tok_in_block = (cur_counts % page_size).long()  # [bs]
                # Clamp block_idx into [0, max_blocks_per_seq) for graph-safe gather.
                max_blocks = compressed_block_table.shape[1]
                block_idx_safe = block_idx.clamp(max=max_blocks - 1)
                # Gather page IDs: bt[seq_slot, block_idx_safe]
                page_ids = compressed_block_table[seq_slots.long(), block_idx_safe]
                physical_slots = page_ids.long() * page_size + tok_in_block
            else:
                # Backward-compat: contiguous-chunk addressing (legacy path).
                valid_slots = (num_pages - 1) * page_size
                max_compressed = valid_slots // self._max_slots
                physical_slots = seq_slots.long() * max_compressed + cur_counts
            # Redirect invalid (not compressing) writes to last slot (in dummy page).
            dummy_slot = total_slots - 1
            physical_slots = torch.where(should_compress, physical_slots, dummy_slot)
            _store_dsv4_fp8_batched(
                compressed,
                compressed_cache,
                physical_slots.to(torch.int32),
                page_size,
            )
            # Update counts only for compressing seqs
            inc = should_compress.to(torch.int32)
            self._compressed_counts.scatter_add_(0, seq_slots.long(), inc)

    def cached(self, seq_key: int) -> torch.Tensor | None:
        return self._compressed_cache.get(seq_key)

    def get_compressed_count(self, seq_key: int) -> int:
        """Return number of compressed tokens for a sequence."""
        if self._compressed_counts is not None and 0 <= seq_key < self._max_slots:
            return int(self._compressed_counts[seq_key].item())
        cached = self._compressed_cache.get(seq_key)
        return 0 if cached is None else cached.shape[0]

    def reset_slot(self, slot: int):
        """Clear state for a slot (called when sequence is deallocated)."""
        if self._kv_states is not None and 0 <= slot < self._max_slots:
            self._kv_states[slot].zero_()
            self._score_states[slot].fill_(float("-inf"))
            self._compressed_counts[slot] = 0
        self._states.pop(slot, None)
        self._compressed_cache.pop(slot, None)


class DeepseekV4Attention(nn.Module):
    """Initial H200-friendly DSV4 attention path.

    This deliberately uses NanoDeploy's existing paged GQA attention backend:
    the official FP4 sparse/compressed fast path can be added behind this module
    without changing the rest of the model.
    """

    def __init__(self, config, quantization_config: QuantizationConfig, layer_idx: int):
        super().__init__()
        if get_dist_context().attn_tp_world_size != 1:
            raise NotImplementedError("DeepseekV4 initial path requires attention_tp=1")

        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.o_groups = config.o_groups
        self.n_local_groups = self.o_groups
        self.window_size = getattr(config, "window_size", 128)
        self.rms_norm_eps = config.rms_norm_eps

        self.attn_sink = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))
        self.wq_a = get_backend().get_replicated_linear(
            self.hidden_size, self.q_lora_rank, bias=False
        )
        self.q_norm = RMSNorm(self.q_lora_rank, eps=self.rms_norm_eps)
        self.wq_b = get_backend().get_column_parallel_linear(
            self.q_lora_rank,
            self.num_heads * self.head_dim,
            bias=False,
            tp_group=get_dist_context().attn_tp_group,
        )
        self.wkv = get_backend().get_replicated_linear(
            self.hidden_size, self.head_dim, bias=False
        )
        self.kv_norm = RMSNorm(self.head_dim, eps=self.rms_norm_eps)
        self.wo_a = _WeightOnlyLinear(
            self.num_heads * self.head_dim // self.o_groups,
            self.o_groups * self.o_lora_rank,
        )
        self.wo_b = get_backend().get_row_parallel_linear(
            self.o_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            tp_group=get_dist_context().attn_tp_group,
        )

        rope_scaling = getattr(config, "rope_scaling", None)
        rope_theta = getattr(config, "rope_theta", 10000.0)
        compress_ratios = getattr(config, "compress_ratios", None) or []
        compress_ratio = (
            compress_ratios[layer_idx] if layer_idx < len(compress_ratios) else 0
        )
        self.compress_ratio = compress_ratio
        if self.compress_ratio:
            self.compressor = DeepseekV4Compressor(
                config,
                self.compress_ratio,
                self.head_dim,
            )
        if compress_ratio:
            rope_theta = getattr(config, "compress_rope_theta", rope_theta)
            if rope_scaling is not None:
                rope_scaling = dict(rope_scaling)
                # Official DSV4 uses YaRN frequency interpolation here, but
                # does not apply the extra mscale factor used by some HF paths.
                rope_scaling["mscale"] = 0.0
                rope_scaling["mscale_all_dim"] = 0.0
        else:
            # Official DSV4 disables YaRN for pure sliding-window layers.
            rope_scaling = None
        self.rotary_emb = get_rope(
            self.rope_head_dim,
            rotary_dim=self.rope_head_dim,
            max_position=config.max_position_embeddings,
            base=float(rope_theta),
            rope_scaling=rope_scaling,
        )
        self.softmax_scale = self.head_dim**-0.5
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])
        # DSv4 flash_mla caches (wired by model_runner after allocation)
        self.swa_cache: torch.Tensor | None = (
            None  # [num_pages, page_size, 1, 584] uint8
        )
        self.compressed_cache: torch.Tensor | None = (
            None  # [max_seqs, max_compressed, 1, 584] uint8
        )
        # Per-layer sched_meta cache: each layer's config differs by compress_ratio
        # so we cannot share a single FlashMLASchedMeta across layers. Keyed by
        # batch_size to reuse the same meta for repeated calls with same bs.
        self._dsv4_sched_metas: dict[int, object] = {}

    def _gather_seq_cache(
        self,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        seqlen: torch.Tensor | int,
    ) -> torch.Tensor:
        seqlen = int(seqlen.item()) if isinstance(seqlen, torch.Tensor) else int(seqlen)
        if seqlen == 0:
            return cache.new_empty(0, cache.shape[2], cache.shape[3])
        block_size = cache.shape[1]
        num_blocks = (seqlen + block_size - 1) // block_size
        blocks = block_table[:num_blocks].long()
        offsets = torch.arange(block_size, device=cache.device).repeat(num_blocks)
        slots = (blocks.repeat_interleave(block_size) * block_size + offsets)[:seqlen]
        return cache.reshape(-1, cache.shape[2], cache.shape[3])[slots]

    def _attend_one(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cached_len: int,
        causal: bool,
    ) -> torch.Tensor:
        k = k.squeeze(1)
        v = v.squeeze(1)
        scores = torch.einsum("thd,ld->thl", q.float(), k.float()) * self.softmax_scale
        if causal:
            q_len, k_len = q.shape[0], k.shape[0]
            q_pos = torch.arange(q_len, device=q.device).unsqueeze(-1) + cached_len
            k_pos = torch.arange(k_len, device=q.device).unsqueeze(0)
            scores = scores.masked_fill((k_pos > q_pos).unsqueeze(1), float("-inf"))
        scores_max = torch.maximum(
            scores.amax(dim=-1),
            self.attn_sink.float().view(1, -1),
        )
        probs = torch.exp(scores - scores_max.unsqueeze(-1))
        probs = probs / (
            probs.sum(dim=-1, keepdim=True)
            + torch.exp(
                self.attn_sink.float().view(1, -1, 1) - scores_max.unsqueeze(-1)
            )
        )
        probs = probs.to(q.dtype)
        return torch.einsum("thl,ld->thd", probs, v)

    def _window_topk_idxs(self, seqlen: int, device: torch.device) -> torch.Tensor:
        width = min(seqlen, self.window_size)
        base = torch.arange(seqlen, device=device).unsqueeze(1)
        idxs = (base - self.window_size + 1).clamp_min(0) + torch.arange(
            width,
            device=device,
        )
        return torch.where(idxs > base, -1, idxs)

    def _compress_topk_idxs(
        self,
        seqlen: int,
        num_compressed: int,
        offset: int,
        device: torch.device,
    ) -> torch.Tensor:
        if num_compressed == 0:
            return torch.empty(seqlen, 0, dtype=torch.long, device=device)
        block_ids = torch.arange(num_compressed, device=device)
        allowed_blocks = (
            torch.arange(1, seqlen + 1, device=device).unsqueeze(1)
            // self.compress_ratio
        )
        allowed = block_ids.unsqueeze(0) < allowed_blocks
        return torch.where(allowed, block_ids.unsqueeze(0) + offset, -1)

    def _seq_key(self, context, seq_idx: int) -> int:
        block_tables = getattr(context, "block_tables", None)
        if block_tables is not None and block_tables.numel() > 0:
            try:
                return int(block_tables[0, seq_idx, 0].item())
            except Exception:
                pass
        return seq_idx

    def _attend_sparse_one(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        topk_idxs: torch.Tensor,
    ) -> torch.Tensor:
        if kv.ndim == 3:
            assert kv.size(1) == 1
            kv = kv.squeeze(1)
        valid = topk_idxs >= 0
        safe_idxs = topk_idxs.clamp_min(0)
        selected = kv[safe_idxs]
        scores = (
            torch.einsum("thd,tkd->thk", q.float(), selected.float())
            * self.softmax_scale
        )
        scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
        scores_max = torch.maximum(
            scores.amax(dim=-1),
            self.attn_sink.float().view(1, -1),
        )
        probs = torch.exp(scores - scores_max.unsqueeze(-1))
        probs = probs / (
            probs.sum(dim=-1, keepdim=True)
            + torch.exp(
                self.attn_sink.float().view(1, -1, 1) - scores_max.unsqueeze(-1)
            )
        )
        return torch.einsum("thk,tkd->thd", probs.to(q.dtype), selected)

    def _prefill_attention(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        context = get_context()

        # Write KV to cache: either new FP8 SWA cache or legacy BF16 cache
        if self.swa_cache is not None and not context.is_dummy:
            kv_2d = kv.squeeze(1)  # [T, 512]
            page_size = self.swa_cache.shape[1]
            _store_dsv4_fp8_batched(
                kv_2d, self.swa_cache, context.slot_mapping, page_size
            )
        elif self.k_cache.numel() and self.v_cache.numel() and not context.is_dummy:
            store_kvcache(
                kv.contiguous(),
                kv.contiguous(),
                self.k_cache,
                self.v_cache,
                context.slot_mapping,
            )

        cu_seqlens = context.cu_seqlens_q
        outs = []
        debug_kv = None
        for seq_idx in range(cu_seqlens.shape[0] - 1):
            qs = int(cu_seqlens[seq_idx].item())
            qe = int(cu_seqlens[seq_idx + 1].item())
            seqlen = qe - qs
            kv_seq = kv[qs:qe].squeeze(1)
            compressed = None
            if self.compress_ratio:
                seq_key = self._seq_key(context, seq_idx)
                compressed = self.compressor.forward_prefill(
                    hidden_states[qs:qe],
                    positions[qs:qe],
                    self.rotary_emb,
                    seq_key=seq_key,
                )
                # Also write compressed KV to the FP8 compressed cache for flash_mla
                if compressed is not None and self.compressed_cache is not None:
                    n_compressed = compressed.shape[0]
                    page_size_c = self.compressed_cache.shape[1]
                    valid_slots = (self.compressed_cache.shape[0] - 1) * page_size_c
                    max_compressed = valid_slots // self.compressor._max_slots
                    if seq_key < self.compressor._max_slots:
                        base_slot = seq_key * max_compressed
                        slots = torch.arange(
                            base_slot,
                            base_slot + n_compressed,
                            dtype=torch.int32,
                            device=kv.device,
                        )
                        _store_dsv4_fp8_batched(
                            compressed, self.compressed_cache, slots, page_size_c
                        )
            if compressed is not None:
                kv_for_attn = torch.cat([kv_seq, compressed], dim=0)
                compress_topk = self._compress_topk_idxs(
                    seqlen,
                    compressed.size(0),
                    offset=seqlen,
                    device=q.device,
                )
                topk_idxs = torch.cat(
                    [self._window_topk_idxs(seqlen, q.device), compress_topk],
                    dim=-1,
                )
            else:
                kv_for_attn = kv_seq
                topk_idxs = self._window_topk_idxs(seqlen, q.device)
            if debug_kv is None:
                debug_kv = kv_for_attn
            outs.append(self._attend_sparse_one(q[qs:qe], kv_for_attn, topk_idxs))
        if debug_kv is not None:
            _debug_dump("attn_kv_after_rope", debug_kv, self.layer_idx)
        return torch.cat(outs, dim=0)

    def _decode_attention(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        context = get_context()
        if self.k_cache.numel() and self.v_cache.numel() and not context.is_dummy:
            store_kvcache(
                kv.contiguous(),
                kv.contiguous(),
                self.k_cache,
                self.v_cache,
                context.slot_mapping,
            )

        ntps = getattr(context, "num_tokens_per_seq", 1)
        total_tokens = q.shape[0]
        bs = total_tokens // ntps
        outs = []
        block_tables = context.block_tables[0, :bs]
        context_lens = context.context_lens[0, :bs]
        for seq_idx in range(bs):
            qs = seq_idx * ntps
            qe = qs + ntps
            if ntps != 1:
                k_len = int(context_lens[seq_idx].item())
                k_seq = self._gather_seq_cache(
                    self.k_cache, block_tables[seq_idx], k_len
                )
                outs.append(
                    self._attend_one(q[qs:qe], k_seq, k_seq, k_len - ntps, True)
                )
                continue

            seq_key = self._seq_key(context, seq_idx)
            position = int(positions[qs].item())
            if self.compress_ratio:
                self.compressor.forward_decode(
                    hidden_states[qs:qe],
                    position,
                    self.rotary_emb,
                    seq_key,
                )
            k_len = int(context_lens[seq_idx].item())
            k_seq = self._gather_seq_cache(self.k_cache, block_tables[seq_idx], k_len)
            if k_seq.size(0) > self.window_size:
                k_seq = k_seq[-self.window_size :]
            compressed = (
                self.compressor.cached(seq_key) if self.compress_ratio else None
            )
            kv_for_attn = (
                k_seq
                if compressed is None
                else torch.cat([k_seq.squeeze(1), compressed], dim=0)
            )
            topk_idxs = torch.arange(
                kv_for_attn.size(0),
                dtype=torch.long,
                device=q.device,
            ).view(1, -1)
            outs.append(self._attend_sparse_one(q[qs:qe], kv_for_attn, topk_idxs))
        return torch.cat(outs, dim=0)

    def _prefill_attention_flash_mla(
        self,
        q: torch.Tensor,  # [total_q, num_heads, head_dim]
        kv: torch.Tensor,  # [total_q, 1, head_dim]
        hidden_states: torch.Tensor,  # [total_q, hidden_size]
        positions: torch.Tensor,  # [total_q] absolute positions
    ) -> torch.Tensor:
        """Batched prefill using flash_mla — single kernel for all Q tokens.

        Each Q token is treated as its own batch entry with its own causal
        window SWA indices and compressed indices (same approach as SGLang).
        """
        import flash_mla

        context = get_context()

        # Fallback for warmup/dummy: no block_tables means we can't build
        # physical slot indices. Defer to the einsum path (correctness-only;
        # warmup output is discarded).
        if context.block_tables is None or context.is_dummy:
            return self._prefill_attention(q, kv, hidden_states, positions)

        total_q = q.shape[0]
        page_size = self.swa_cache.shape[1]

        # 1. Store current KV into SWA FP8 cache
        kv_2d = kv.squeeze(1)  # [total_q, 512]
        if not context.is_dummy:
            _store_dsv4_fp8_batched(
                kv_2d, self.swa_cache, context.slot_mapping, page_size
            )

        # 2. Run compressor prefill per sequence — produces compressed KV
        #    written directly into the FP8 compressed cache.
        cu_seqlens_q = context.cu_seqlens_q  # [num_seqs + 1]
        cu_seqlens_k = context.cu_seqlens_k  # [num_seqs + 1]
        num_seqs = cu_seqlens_q.shape[0] - 1
        block_tables = context.block_tables[0]  # [num_seqs, max_blocks]

        # Pull scheduler-assigned state slots into a Python list for per-seq lookup.
        # Falls back to batch position when unavailable (warmup / legacy path).
        if context.dsv4_state_slots is not None:
            state_slots_list = context.dsv4_state_slots[:num_seqs].tolist()
        else:
            state_slots_list = list(range(num_seqs))

        # Block table for this layer's compression ratio (paged path).  None
        # means use legacy contiguous-chunk addressing.
        cbts = getattr(context, "dsv4_compressed_block_tables", None) or {}
        comp_bt = cbts.get(self.compress_ratio) if self.compress_ratio else None

        if self.compress_ratio:
            for seq_idx in range(num_seqs):
                qs = int(cu_seqlens_q[seq_idx].item())
                qe = int(cu_seqlens_q[seq_idx + 1].item())
                seq_key = state_slots_list[seq_idx]
                compressed = self.compressor.forward_prefill(
                    hidden_states[qs:qe],
                    positions[qs:qe],
                    self.rotary_emb,
                    seq_key=seq_key,
                )
                if compressed is not None and self.compressed_cache is not None:
                    n_compressed = compressed.shape[0]
                    page_size_c = self.compressed_cache.shape[1]
                    if seq_key >= self.compressor._max_slots:
                        continue  # invalid slot — skip write
                    if comp_bt is not None:
                        # Paged: gather per-seq page IDs from the block table,
                        # convert (token_idx) → (page_id, tok_in_page).
                        max_blocks = comp_bt.shape[1]
                        page_ids_for_seq = comp_bt[seq_key]  # [max_blocks] int32
                        tok_idx = torch.arange(
                            n_compressed, dtype=torch.int64, device=kv.device
                        )
                        block_idx = (tok_idx // page_size_c).clamp(max=max_blocks - 1)
                        tok_in_block = tok_idx % page_size_c
                        slots = (
                            page_ids_for_seq[block_idx].long() * page_size_c
                            + tok_in_block
                        ).to(torch.int32)
                    else:
                        # Backward-compat: contiguous chunk per seq.
                        valid_slots = (self.compressed_cache.shape[0] - 1) * page_size_c
                        max_compressed = valid_slots // self.compressor._max_slots
                        base_slot = seq_key * max_compressed
                        slots = torch.arange(
                            base_slot,
                            base_slot + n_compressed,
                            dtype=torch.int32,
                            device=kv.device,
                        )
                    _store_dsv4_fp8_batched(
                        compressed, self.compressed_cache, slots, page_size_c
                    )

        # 3. Build per-Q-token SWA indices via vectorized tensor ops
        device = q.device
        # Per-seq metadata
        chunk_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).long()  # [num_seqs]
        ctx_lens = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).long()  # total KV len
        cached_lens = ctx_lens - chunk_lens  # pre-chunk KV

        # For each Q token, find its seq_idx and position within the sequence
        j_range = torch.arange(total_q, device=device, dtype=torch.int64)
        seq_idx_per_tok = torch.searchsorted(
            cu_seqlens_q[1:], j_range, right=True
        ).clamp(
            max=num_seqs - 1
        )  # [total_q]
        q_pos_in_chunk = j_range - cu_seqlens_q[seq_idx_per_tok].long()
        ctx_pos = cached_lens[seq_idx_per_tok] + q_pos_in_chunk  # [total_q]

        # SWA per-token window
        swa_topk = ((self.window_size + 63) // 64) * 64
        win_len = torch.clamp(ctx_pos + 1, max=self.window_size)  # [total_q]
        win_start = ctx_pos + 1 - win_len  # [total_q]
        tok_range = torch.arange(swa_topk, device=device, dtype=torch.int64)
        logical_pos = win_start.unsqueeze(1) + tok_range.unsqueeze(
            0
        )  # [total_q, swa_topk]
        valid = tok_range.unsqueeze(0) < win_len.unsqueeze(1)

        # Convert logical → physical via per-token block_table lookup
        per_q_block_tables = block_tables[seq_idx_per_tok]  # [total_q, max_blocks]
        page_indices = logical_pos // page_size
        tok_in_page = logical_pos % page_size
        page_indices_safe = page_indices.clamp(0, per_q_block_tables.shape[1] - 1)
        physical_blocks = per_q_block_tables.gather(1, page_indices_safe)
        physical_slots = physical_blocks * page_size + tok_in_page
        swa_indices = torch.where(valid, physical_slots, -1).to(torch.int32)
        swa_indices = swa_indices.unsqueeze(1)  # [total_q, 1, swa_topk]
        swa_topk_lengths = win_len.to(torch.int32)  # [total_q]

        # 4. Build per-Q-token compressed indices
        extra_k_cache = None
        extra_indices = None
        extra_topk_lengths = None
        if self.compress_ratio and self.compressed_cache is not None:
            page_size_c = self.compressed_cache.shape[1]

            cbts = getattr(context, "dsv4_compressed_block_tables", None) or {}
            comp_bt = cbts.get(self.compress_ratio)

            if context.dsv4_state_slots is not None:
                seq_slot_per_tok = context.dsv4_state_slots[:num_seqs][
                    seq_idx_per_tok
                ]  # [total_q]
            else:
                seq_slot_per_tok = seq_idx_per_tok
            # Visible compressed blocks (token count) for Q token at ctx_pos:
            visible_blocks = (ctx_pos + 1) // self.compress_ratio  # [total_q]

            if comp_bt is not None:
                # Paged addressing: gather block IDs from the per-seq table.
                max_blocks = comp_bt.shape[1]
                max_compressed = max_blocks * page_size_c
                extra_topk = ((max_compressed + 63) // 64) * 64
                tok_range_c = torch.arange(extra_topk, device=device, dtype=torch.int64)
                block_idx = tok_range_c // page_size_c
                tok_in_block = tok_range_c % page_size_c
                block_idx_safe = block_idx.clamp(max=max_blocks - 1)
                # comp_bt: [num_seqs_active, max_blocks]; we have per-tok seq_slot.
                page_ids = comp_bt[
                    seq_slot_per_tok.long().unsqueeze(1),
                    block_idx_safe.unsqueeze(0),
                ]
                physical = page_ids.long() * page_size_c + tok_in_block.unsqueeze(0)
                valid_c = tok_range_c.unsqueeze(0) < visible_blocks.unsqueeze(1)
                extra_indices = torch.where(valid_c, physical, -1).to(torch.int32)
                extra_indices = extra_indices.unsqueeze(1)  # [total_q, 1, extra_topk]
                extra_topk_lengths = visible_blocks.to(torch.int32)
            else:
                # Backward-compat: contiguous-chunk addressing.
                valid_slots = (self.compressed_cache.shape[0] - 1) * page_size_c
                max_compressed = valid_slots // self.compressor._max_slots
                extra_topk = ((max_compressed + 63) // 64) * 64
                base = seq_slot_per_tok * max_compressed
                tok_range_c = torch.arange(
                    extra_topk, device=device, dtype=torch.int64
                ).unsqueeze(0)
                idx = base.unsqueeze(1) + tok_range_c
                valid_c = tok_range_c < visible_blocks.unsqueeze(1)
                extra_indices = torch.where(valid_c, idx, -1).to(torch.int32)
                extra_indices = extra_indices.unsqueeze(1)
                extra_topk_lengths = visible_blocks.to(torch.int32)
            extra_k_cache = self.compressed_cache

        # 5. Single flash_mla call: bs=total_q, seq_len_q=1
        # Use per-layer sched_meta keyed by total_q (not shared across layers)
        tile_meta = self._dsv4_sched_metas.get(total_q)
        if tile_meta is None:
            tile_meta, _ = flash_mla.get_mla_metadata()
            self._dsv4_sched_metas[total_q] = tile_meta

        o, _lse = flash_mla.flash_mla_with_kvcache(
            q.reshape(total_q, 1, self.num_heads, self.head_dim),
            self.swa_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=self.head_dim,
            tile_scheduler_metadata=tile_meta,
            softmax_scale=self.softmax_scale,
            is_fp8_kvcache=True,
            indices=swa_indices,
            topk_length=swa_topk_lengths,
            attn_sink=self.attn_sink.detach(),
            extra_k_cache=extra_k_cache,
            extra_indices_in_kvcache=extra_indices,
            extra_topk_length=extra_topk_lengths,
        )
        return o.reshape(total_q, self.num_heads, self.head_dim)

    def _decode_attention_flash_mla(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Batched decode using flash_mla with FP8 packed KV cache.

        No per-sequence Python loop. Single kernel call for all sequences.
        """
        import flash_mla

        context = get_context()
        ntps = getattr(context, "num_tokens_per_seq", 1)
        total_tokens = q.shape[0]
        bs = total_tokens // ntps
        block_tables = context.block_tables[0, :bs]
        context_lens = context.context_lens[0, :bs]
        page_size = self.swa_cache.shape[1]

        # 1. Store KV to SWA paged FP8 cache
        kv_2d = kv.squeeze(1)  # [T, 512]
        _store_dsv4_fp8_batched(kv_2d, self.swa_cache, context.slot_mapping, page_size)

        # 2. Run compressor update — fully batched, CUDAGraph-safe.
        # Use scheduler-assigned state_slots (stable per-sequence identity
        # across decode steps). Falls back to arange(bs) when slots are not
        # provided (e.g., during warmup before scheduler populates them).
        if self.compress_ratio and ntps == 1:
            if context.dsv4_state_slots is not None:
                seq_slots = context.dsv4_state_slots[:bs]
            else:
                seq_slots = torch.arange(bs, dtype=torch.int64, device=q.device)
            positions_per_seq = positions[::ntps]  # [bs]
            # Block table for this layer's ratio (paged compressed cache).
            # None → forward_decode_batched falls back to legacy contiguous-chunk
            # addressing (kept for warmup / pre-Stage-3 compatibility).
            cbts = getattr(context, "dsv4_compressed_block_tables", None) or {}
            comp_bt = cbts.get(self.compress_ratio)
            self.compressor.forward_decode_batched(
                hidden_states[::ntps],
                positions_per_seq,
                self.rotary_emb,
                seq_slots,
                self.compressed_cache,
                compressed_block_table=comp_bt,
            )

        # 3. Build SWA indices [bs, ntps, swa_topk] — vectorized, no per-seq loop
        swa_topk = ((self.window_size + 63) // 64) * 64  # align to 64
        # Clamp to min(ctx_len, window_size) — each Q token attends to at most
        # window_size recent tokens. Also guard against exceeding swa_topk.
        swa_topk_lengths = context_lens.clamp(max=min(self.window_size, swa_topk))

        # Logical positions: for each seq, the last min(ctx_len, window) tokens
        # token_offsets[b, t] = ctx_len[b] - win_len[b] + t
        token_offsets = torch.arange(
            swa_topk, device=q.device, dtype=torch.int32
        ).unsqueeze(
            0
        )  # [1, swa_topk]
        start_pos = (context_lens - swa_topk_lengths).unsqueeze(1)  # [bs, 1]
        logical_pos = start_pos + token_offsets  # [bs, swa_topk]

        # Mark invalid positions
        valid_mask = token_offsets < swa_topk_lengths.unsqueeze(1)  # [bs, swa_topk]

        # Convert logical → physical via block_tables
        page_indices = logical_pos // page_size  # [bs, swa_topk]
        tok_in_page = logical_pos % page_size  # [bs, swa_topk]
        # Clamp page_indices to valid range for gather
        page_indices_safe = page_indices.clamp(0, block_tables.shape[1] - 1).long()
        physical_blocks = block_tables.gather(1, page_indices_safe)  # [bs, swa_topk]
        physical_slots = physical_blocks * page_size + tok_in_page

        # Set invalid positions to -1
        swa_indices = torch.where(valid_mask, physical_slots, -1).to(torch.int32)
        swa_indices = swa_indices.unsqueeze(1)  # [bs, 1, swa_topk]

        # 4. Build compressed indices [bs, 1, extra_topk] (if compressed layers)
        extra_k_cache = None
        extra_indices = None
        extra_topk_lengths = None
        if self.compress_ratio and self.compressed_cache is not None:
            # compressed_cache: [num_pages+1, page_size, 1, 584]; last page is dummy
            page_size_c = self.compressed_cache.shape[1]

            cbts = getattr(context, "dsv4_compressed_block_tables", None) or {}
            comp_bt = cbts.get(self.compress_ratio)

            # Use scheduler-assigned state slots (stable per-seq identity).
            if context.dsv4_state_slots is not None:
                seq_slots = context.dsv4_state_slots[:bs]
            else:
                seq_slots = torch.arange(bs, dtype=torch.int64, device=q.device)

            if comp_bt is not None:
                # Paged addressing: extra_topk = max_blocks_per_seq * page_size_c
                # capped at compressor's _compressed_counts (with kernel 64-align).
                max_blocks = comp_bt.shape[1]
                max_compressed = max_blocks * page_size_c
                extra_topk = ((max_compressed + 63) // 64) * 64
                # Clamp lengths to extra_topk (kernel reads first `length` indices)
                extra_topk_lengths = self.compressor._compressed_counts[
                    seq_slots
                ].clamp(max=extra_topk)
                # For each (b, t in [0, extra_topk)):
                #   block_idx = t // page_size_c
                #   tok_in_block = t % page_size_c
                #   page_id = comp_bt[seq_slots[b], block_idx]
                #   physical = page_id * page_size_c + tok_in_block
                tok_range = torch.arange(extra_topk, device=q.device, dtype=torch.int32)
                block_idx = (tok_range // page_size_c).long()  # [extra_topk]
                tok_in_block = (tok_range % page_size_c).long()  # [extra_topk]
                # Clamp block_idx for graph-safe gather (out-of-range entries
                # will be masked to -1 by valid_mask below).
                block_idx_safe = block_idx.clamp(max=max_blocks - 1)
                # Gather page IDs: [bs, extra_topk]
                page_ids = comp_bt[
                    seq_slots.long().unsqueeze(1), block_idx_safe.unsqueeze(0)
                ]
                physical = page_ids.long() * page_size_c + tok_in_block.unsqueeze(0)
                valid_mask = tok_range.unsqueeze(0) < extra_topk_lengths.unsqueeze(1)
                extra_indices = torch.where(valid_mask, physical, -1).to(torch.int32)
                extra_indices = extra_indices.unsqueeze(1)  # [bs, 1, extra_topk]
            else:
                # Backward-compat: contiguous-chunk addressing (legacy path).
                valid_slots = (self.compressed_cache.shape[0] - 1) * page_size_c
                max_compressed = valid_slots // self.compressor._max_slots
                extra_topk = ((max_compressed + 63) // 64) * 64
                extra_topk_lengths = self.compressor._compressed_counts[
                    seq_slots
                ].clamp(max=extra_topk)
                tok_range = torch.arange(
                    extra_topk, device=q.device, dtype=torch.int32
                ).unsqueeze(0)
                base_offsets = (seq_slots.to(torch.int32) * max_compressed).unsqueeze(1)
                extra_indices = base_offsets + tok_range
                valid_mask = tok_range < extra_topk_lengths.unsqueeze(1)
                extra_indices = torch.where(valid_mask, extra_indices, -1).to(
                    torch.int32
                )
                extra_indices = extra_indices.unsqueeze(1)
            extra_k_cache = self.compressed_cache

        # 5. Get or create per-layer FlashMLASchedMeta
        # Each layer's config differs (extra_page_block_size, extra_topk) so
        # we cannot share a single sched_meta across the 43 DSv4 layers.
        tile_meta = self._dsv4_sched_metas.get(bs)
        if tile_meta is None:
            tile_meta, _ = flash_mla.get_mla_metadata()
            self._dsv4_sched_metas[bs] = tile_meta

        # 6. Single flash_mla call
        o, lse = flash_mla.flash_mla_with_kvcache(
            q.reshape(bs, ntps, self.num_heads, self.head_dim),
            self.swa_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=self.head_dim,
            tile_scheduler_metadata=tile_meta,
            softmax_scale=self.softmax_scale,
            is_fp8_kvcache=True,
            indices=swa_indices,
            topk_length=swa_topk_lengths,
            attn_sink=self.attn_sink.detach(),
            extra_k_cache=extra_k_cache,
            extra_indices_in_kvcache=extra_indices,
            extra_topk_length=extra_topk_lengths,
        )
        # o: [bs, ntps, H, head_dim] → [total_tokens, H, head_dim]
        return o.reshape(total_tokens, self.num_heads, self.head_dim)

    def _paged_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        context = get_context()
        if self.k_cache.numel() and self.v_cache.numel() and not context.is_dummy:
            store_kvcache(
                k.contiguous(),
                v.contiguous(),
                self.k_cache,
                self.v_cache,
                context.slot_mapping,
            )

        if context.block_tables is None or context.is_dummy:
            if not context.is_prefill:
                return self._attend_one(q, k, v, cached_len=0, causal=False)
            outs = []
            for seq_idx in range(context.cu_seqlens_q.shape[0] - 1):
                qs = int(context.cu_seqlens_q[seq_idx].item())
                qe = int(context.cu_seqlens_q[seq_idx + 1].item())
                outs.append(self._attend_one(q[qs:qe], k[qs:qe], v[qs:qe], 0, True))
            return torch.cat(outs, dim=0)

        if context.is_prefill:
            outs = []
            block_tables = context.block_tables[0]
            for seq_idx in range(context.cu_seqlens_q.shape[0] - 1):
                qs = int(context.cu_seqlens_q[seq_idx].item())
                qe = int(context.cu_seqlens_q[seq_idx + 1].item())
                k_len = int(
                    context.cu_seqlens_k[seq_idx + 1].item()
                    - context.cu_seqlens_k[seq_idx].item()
                )
                cached_len = k_len - (qe - qs)
                k_seq = self._gather_seq_cache(
                    self.k_cache, block_tables[seq_idx], k_len
                )
                v_seq = self._gather_seq_cache(
                    self.v_cache, block_tables[seq_idx], k_len
                )
                outs.append(self._attend_one(q[qs:qe], k_seq, v_seq, cached_len, True))
            return torch.cat(outs, dim=0)

        ntps = context.num_tokens_per_seq
        total_tokens = q.shape[0]
        bs = total_tokens // ntps
        outs = []
        block_tables = context.block_tables[0, :bs]
        context_lens = context.context_lens[0, :bs]
        for seq_idx in range(bs):
            qs = seq_idx * ntps
            qe = qs + ntps
            k_len = int(context_lens[seq_idx].item())
            k_seq = self._gather_seq_cache(self.k_cache, block_tables[seq_idx], k_len)
            v_seq = self._gather_seq_cache(self.v_cache, block_tables[seq_idx], k_len)
            outs.append(
                self._attend_one(q[qs:qe], k_seq, v_seq, k_len - ntps, ntps > 1)
            )
        return torch.cat(outs, dim=0)

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        q_len = hidden_states.size(0)
        q_lora_pre = self.wq_a(hidden_states)
        _debug_dump("attn_q_lora_pre_norm", q_lora_pre, self.layer_idx)
        q = self.q_norm(q_lora_pre)
        _debug_dump("attn_q_lora", q, self.layer_idx)
        q_flat = self.wq_b(q)
        _debug_dump("attn_wq_b", q_flat, self.layer_idx)
        q = q_flat.view(q_len, self.num_heads, self.head_dim)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.rms_norm_eps)
        _debug_dump("attn_q_normed", q, self.layer_idx)

        kv_pre = self.wkv(hidden_states)
        _debug_dump("attn_kv_pre_norm", kv_pre, self.layer_idx)
        kv = self.kv_norm(kv_pre)
        _debug_dump("attn_kv", kv, self.layer_idx)
        q_rope = q[..., -self.rope_head_dim :]
        k_rope = kv[..., -self.rope_head_dim :].unsqueeze(1)
        q_rope = _apply_rotary_interleaved(self.rotary_emb, positions, q_rope)
        k_rope = _apply_rotary_interleaved(self.rotary_emb, positions, k_rope)
        q[..., -self.rope_head_dim :] = q_rope
        kv = kv.unsqueeze(1)
        kv[..., -self.rope_head_dim :] = k_rope
        _fp8_quant_dequant_inplace(kv[..., : -self.rope_head_dim], 64)
        _debug_dump("attn_q_after_rope", q, self.layer_idx)
        _debug_dump("attn_window_kv_after_rope", kv, self.layer_idx)

        if get_context().is_prefill:
            if self.swa_cache is not None:
                out = self._prefill_attention_flash_mla(q, kv, hidden_states, positions)
            else:
                out = self._prefill_attention(q, kv, hidden_states, positions)
        elif self.swa_cache is not None:
            # flash_mla batched decode path (no per-seq loop)
            out = self._decode_attention_flash_mla(q, kv, hidden_states, positions)
        else:
            _debug_dump("attn_kv_after_rope", kv, self.layer_idx)
            out = self._decode_attention(q, kv, hidden_states, positions)
        _debug_dump("attn_context", out, self.layer_idx)
        out[..., -self.rope_head_dim :] = _apply_rotary_interleaved(
            self.rotary_emb,
            positions,
            out[..., -self.rope_head_dim :],
            inverse=True,
        )
        _debug_dump("attn_context_inverse_rope", out, self.layer_idx)
        out = out.reshape(q_len, self.o_groups, -1)
        wo_a = self.wo_a.weight.view(self.o_groups, self.o_lora_rank, -1)
        out = torch.einsum("tgd,grd->tgr", out, wo_a.to(out.dtype))
        _debug_dump("attn_wo_a", out, self.layer_idx)
        out = self.wo_b(out.flatten(1))
        _debug_dump("attn_out", out, self.layer_idx)
        return out


class _WeightOnlyLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=torch.bfloat16)
        )
        self.weight.weight_loader = self.weight_loader

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, weight_name: str = None
    ):
        param.data.copy_(loaded_weight.to(param.dtype))


class DeepseekV4MoE(nn.Module):
    def __init__(self, config, quantization_config: QuantizationConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.route_scale = getattr(config, "routed_scaling_factor", 1.0)
        self.score_func = getattr(config, "scoring_func", None) or getattr(
            config, "score_func", "sqrtsoftplus"
        )
        self.hash = layer_idx < getattr(config, "num_hash_layers", 0)
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        if self.hash:
            self.gate.tid2eid = nn.Parameter(
                torch.empty(config.vocab_size, self.top_k, dtype=torch.int32),
                requires_grad=False,
            )
            self.gate.e_score_correction_bias = None
        else:
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.zeros(self.num_experts, dtype=torch.float32, device="cuda"),
                requires_grad=False,
            )

        dist_ctx = get_dist_context()
        self.routed_experts = get_backend().get_distributed_routed_experts(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            num_experts=self.num_experts,
            top_k=self.top_k,
            ep_size=dist_ctx.ffn_ep_world_size,
            tp_size=dist_ctx.ffn_tp_world_size,
            ep_group=dist_ctx.ffn_ep_group,
            tp_group=dist_ctx.ffn_tp_group,
            scoring_func=self.score_func,
            routed_scaling_factor=self.route_scale,
            layer_idx=layer_idx,
        )
        assert config.n_shared_experts == 1
        self.shared_experts = DeepseekV2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            hidden_act=config.hidden_act,
            config=config,
            quantization_config=quantization_config,
        )

    def _scores(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        if self.score_func == "softmax":
            return logits.softmax(dim=-1)
        if self.score_func == "sigmoid":
            return logits.sigmoid()
        if self.score_func == "sqrtsoftplus":
            return F.softplus(logits).sqrt()
        raise ValueError(f"Unsupported DeepseekV4 score_func={self.score_func}")

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        residual = hidden_states
        scores = self._scores(self.gate(hidden_states))
        if self.hash:
            topk_ids = self.gate.tid2eid[input_ids].long()
        else:
            choice_scores = scores
            if self.gate.e_score_correction_bias is not None:
                choice_scores = (
                    choice_scores + self.gate.e_score_correction_bias.float()
                )
            topk_ids = torch.topk(choice_scores, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_ids)
        if self.score_func != "softmax":
            topk_weights = topk_weights / (
                topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            )
        topk_weights = topk_weights * self.route_scale
        _debug_dump("moe_scores", scores, self.layer_idx)
        _debug_dump("moe_topk_ids", topk_ids, self.layer_idx)
        _debug_dump("moe_topk_weights", topk_weights, self.layer_idx)
        out = self.routed_experts(
            hidden_states, topk_ids, topk_weights, is_prefill=get_context().is_prefill
        )
        _debug_dump("moe_routed_out", out, self.layer_idx)
        shared = self.shared_experts(residual)
        _debug_dump("moe_shared_out", shared, self.layer_idx)
        out = out + shared
        _debug_dump("moe_out", out, self.layer_idx)
        return out


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(self, config, quantization_config: QuantizationConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.self_attn = DeepseekV4Attention(config, quantization_config, layer_idx)
        self.mlp = DeepseekV4MoE(config, quantization_config, layer_idx)
        self.hc_attn = DeepseekV4HCProjector(
            config.hidden_size, config.hc_mult, config.hc_sinkhorn_iters, config.hc_eps
        )
        self.hc_ffn = DeepseekV4HCProjector(
            config.hidden_size, config.hc_mult, config.hc_sinkhorn_iters, config.hc_eps
        )

    @staticmethod
    def _hc_post(
        x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
    ):
        return post.unsqueeze(-1) * x.unsqueeze(1) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(1), dim=2
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
    ):
        _debug_dump("layer_input", hidden_states, self.layer_idx)
        residual = hidden_states
        x, post, comb = self.hc_attn(hidden_states)
        _debug_dump("hc_attn_y", x, self.layer_idx)
        _debug_dump("hc_attn_post", post, self.layer_idx)
        _debug_dump("hc_attn_comb", comb, self.layer_idx)
        x = self.input_layernorm(x)
        _debug_dump("attn_norm", x, self.layer_idx)
        x = self.self_attn(positions, x)
        _debug_dump("attn_block_out", x, self.layer_idx)
        hidden_states = self._hc_post(x, residual, post, comb)
        _debug_dump("after_attn_hc", hidden_states, self.layer_idx)

        residual = hidden_states
        x, post, comb = self.hc_ffn(hidden_states)
        _debug_dump("hc_ffn_y", x, self.layer_idx)
        x = self.post_attention_layernorm(x)
        _debug_dump("ffn_norm", x, self.layer_idx)
        x = self.mlp(x, input_ids)
        _debug_dump("ffn_block_out", x, self.layer_idx)
        out = self._hc_post(x, residual, post, comb)
        _debug_dump("layer_out", out, self.layer_idx)
        return out


class DeepseekV4Model(nn.Module):
    def __init__(self, config, quantization_config: QuantizationConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV4DecoderLayer(config, quantization_config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_head = DeepseekV4HCHead(
            config.hidden_size, config.hc_mult, config.hc_eps
        )

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor):
        hidden_states = self.embed_tokens(input_ids)
        _debug_dump("embed", hidden_states)
        hidden_states = hidden_states.unsqueeze(1).repeat(1, self.config.hc_mult, 1)
        _debug_dump("hc_expand", hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions, input_ids)
        hidden_states = self.hc_head(hidden_states)
        _debug_dump("hc_head", hidden_states)
        hidden_states = self.norm(hidden_states)
        _debug_dump("final_norm", hidden_states)
        return hidden_states


class DeepseekV4ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = _normalize_config(config)
        self.quantization_config = QuantizationConfig(
            **getattr(self.config, "quantization_config", {})
        )
        self.model = DeepseekV4Model(self.config, self.quantization_config)
        self.lm_head = ParallelLMHead(self.config.vocab_size, self.config.hidden_size)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor):
        return self.lm_head(hidden_states)

    def load_weights(self, weights):
        from .deepseek_v4_loader import load_weights

        load_weights(self, weights)


def _normalize_config(config):
    config.hidden_size = _getattr_any(config, "hidden_size", "dim")
    config.num_hidden_layers = _getattr_any(config, "num_hidden_layers", "n_layers")
    config.num_attention_heads = _getattr_any(config, "num_attention_heads", "n_heads")
    config.num_key_value_heads = _getattr_any(config, "num_key_value_heads", default=1)
    config.qk_rope_head_dim = _getattr_any(config, "qk_rope_head_dim", "rope_head_dim")
    config.moe_intermediate_size = _getattr_any(
        config, "moe_intermediate_size", "moe_inter_dim"
    )
    config.num_experts_per_tok = _getattr_any(
        config, "num_experts_per_tok", "n_activated_experts"
    )
    config.n_shared_experts = _getattr_any(config, "n_shared_experts", default=1)
    config.hidden_act = _getattr_any(config, "hidden_act", default="silu")
    config.rms_norm_eps = _getattr_any(config, "rms_norm_eps", "norm_eps", default=1e-6)
    config.hc_mult = _getattr_any(config, "hc_mult", default=4)
    config.hc_sinkhorn_iters = _getattr_any(config, "hc_sinkhorn_iters", default=20)
    config.hc_eps = _getattr_any(config, "hc_eps", default=1e-6)
    config.o_groups = _getattr_any(config, "o_groups", default=8)
    config.o_lora_rank = _getattr_any(config, "o_lora_rank", default=1024)
    config.window_size = _getattr_any(config, "window_size", default=128)
    config.index_topk = _getattr_any(config, "index_topk", default=512)
    config.num_hash_layers = _getattr_any(
        config, "num_hash_layers", "n_hash_layers", default=0
    )
    if not hasattr(config, "max_position_embeddings"):
        config.max_position_embeddings = 16384
    return config
