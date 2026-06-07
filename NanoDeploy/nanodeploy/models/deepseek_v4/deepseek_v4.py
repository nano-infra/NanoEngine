import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from nanodeploy._third_party.sglang_jit_kernel import (
    fused_kernels_enabled as _sglang_fused_kernels_enabled,
)
from nanodeploy.backends import get_backend
from nanodeploy.backends.gpu_generic.kernels.kv_store import store_kvcache
from nanodeploy.compile_utils import maybe_compile
from nanodeploy.context.context import get_context
from nanodeploy.context.distributed import get_dist_context
from nanodeploy.layers.activation import SiluAndMul
from nanodeploy.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanodeploy.layers.layernorm import RMSNorm
from nanodeploy.layers.rotary_embedding import get_rope
from nanodeploy.models.deepseek_v2.deepseek_v2 import DeepseekV2MLP
from nanodeploy.models.quant_config import QuantizationConfig


# --- Lazily-compiled helpers ----------------------------------------------
# Applying ``@torch.compile`` directly as a class-method decorator attaches
# ConfigModuleInstance references to the class, which break cloudpickle in
# Ray actors on torch >= 2.10. The class methods below are kept as thin
# trampolines; their bodies live in module-level functions that are
# compiled on first call.
def _routing_scores_with_bias_impl(
    logits: torch.Tensor, bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = logits.float()
    scores = F.softplus(scores).sqrt()
    choice_scores = scores + bias.float()
    return scores, choice_scores


def _normalize_topk_weights_impl(
    scores: torch.Tensor, topk_ids: torch.Tensor, route_scale: float
) -> torch.Tensor:
    topk_weights = scores.gather(1, topk_ids)
    topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weights = topk_weights * route_scale
    return topk_weights


def _fuse_routed_shared_impl(
    routed_out: torch.Tensor, shared_out: torch.Tensor
) -> torch.Tensor:
    return routed_out + shared_out


def _hc_post_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    return post.unsqueeze(-1) * x.unsqueeze(1) + torch.sum(
        comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=1
    )


_routing_scores_with_bias_fn = None
_normalize_topk_weights_fn = None
_fuse_routed_shared_fn = None
_hc_post_fn = None


def _routing_scores_with_bias_compiled(
    logits: torch.Tensor, bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    global _routing_scores_with_bias_fn
    if _routing_scores_with_bias_fn is None:
        _routing_scores_with_bias_fn = maybe_compile(
            _routing_scores_with_bias_impl, dynamic=False, fullgraph=True
        )
    return _routing_scores_with_bias_fn(logits, bias)


def _normalize_topk_weights_compiled(
    scores: torch.Tensor, topk_ids: torch.Tensor, route_scale: float
) -> torch.Tensor:
    global _normalize_topk_weights_fn
    if _normalize_topk_weights_fn is None:
        _normalize_topk_weights_fn = maybe_compile(
            _normalize_topk_weights_impl, dynamic=False, fullgraph=True
        )
    return _normalize_topk_weights_fn(scores, topk_ids, route_scale)


def _fuse_routed_shared_compiled(
    routed_out: torch.Tensor, shared_out: torch.Tensor
) -> torch.Tensor:
    global _fuse_routed_shared_fn
    if _fuse_routed_shared_fn is None:
        _fuse_routed_shared_fn = maybe_compile(
            _fuse_routed_shared_impl, dynamic=False, fullgraph=True
        )
    return _fuse_routed_shared_fn(routed_out, shared_out)


def _hc_post_compiled(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    global _hc_post_fn
    if _hc_post_fn is None:
        _hc_post_fn = maybe_compile(_hc_post_impl, dynamic=False, fullgraph=True)
    return _hc_post_fn(x, residual, post, comb)


# Gate every Hopper-only fused kernel (vendored sglang JIT + tilelang)
# behind a single GPU-arch check. On non-Hopper GPUs these stay ``None``
# and each call site uses its eager fallback (CUDAGraph still allowed).
_DSV4_FUSED_KERNELS = _sglang_fused_kernels_enabled()


# Optional vendored sglang DSV4 fused kernels. When present,
# _apply_rotary_interleaved replaces ~10 eager elementwise launches per
# call with a single CUDA kernel.
# Source (vendored under nanodeploy/_third_party/sglang_jit_kernel):
#   https://github.com/sgl-project/sglang
#   python/sglang/jit_kernel/deepseek_v4.py::fused_rope
# Runtime deps for the vendored slice: torch, triton, tvm-ffi.
if _DSV4_FUSED_KERNELS:
    try:
        from nanodeploy._third_party.sglang_jit_kernel.deepseek_v4 import (
            fused_norm_rope_inplace as _SGL_FUSED_NORM_ROPE,
            fused_rope as _SGL_FUSED_ROPE,
            rmsnorm_self as _SGL_RMSNORM_SELF,
        )
    except Exception:
        # ImportError if tvm-ffi isn't installed; any other Exception if
        # the vendored layout is broken on this checkout. Fall back to
        # eager either way.
        _SGL_FUSED_ROPE = None
        _SGL_FUSED_NORM_ROPE = None
        _SGL_RMSNORM_SELF = None
else:
    # Non-Hopper (or explicitly disabled): use the eager RoPE/RMSNorm path.
    _SGL_FUSED_ROPE = None
    _SGL_FUSED_NORM_ROPE = None
    _SGL_RMSNORM_SELF = None

# Triton kernels for fused UE8M0 FP8 quant. Replace the per-block
# amax/exp2/clamp/cast op chain (~10 launches per call) with one
# kernel each.
try:
    from nanodeploy.backends.gpu_generic.kernels.fp8_ue8m0_quant import (
        fp8_quant_dequant_inplace as _TRITON_FP8_QDQ,
        pack_kv_fp8 as _TRITON_FP8_PACK,
        store_dsv4_kv_fp8_fused as _TRITON_FP8_STORE,
    )
except Exception:
    _TRITON_FP8_QDQ = None
    _TRITON_FP8_PACK = None
    _TRITON_FP8_STORE = None

# Fused index-construction kernels for ``_decode_attention_flash_mla``.
# Replace the ~30 elementwise launches per call (arange/where/clamp/
# floor_divide/remainder/gather chain) with one triton kernel each.
try:
    from nanodeploy.models.deepseek_v4.index_kernels import (
        build_extra_indices_paged as _TRITON_BUILD_EXTRA_INDICES_PAGED,
        build_swa_indices as _TRITON_BUILD_SWA_INDICES,
        compress_counts_update as _TRITON_COMPRESS_COUNTS_UPDATE,
        compress_physical_slots_paged as _TRITON_COMPRESS_PHYSICAL_SLOTS,
        compress_post_shift_overlap as _TRITON_COMPRESS_POST_SHIFT,
        compress_scatter_update as _TRITON_COMPRESS_SCATTER_UPDATE,
        compute_compress_metadata as _TRITON_COMPUTE_COMPRESS_METADATA,
    )
except Exception:
    _TRITON_BUILD_SWA_INDICES = None
    _TRITON_BUILD_EXTRA_INDICES_PAGED = None
    _TRITON_COMPRESS_POST_SHIFT = None
    _TRITON_COMPRESS_PHYSICAL_SLOTS = None
    _TRITON_COMPRESS_SCATTER_UPDATE = None
    _TRITON_COMPUTE_COMPRESS_METADATA = None
    _TRITON_COMPRESS_COUNTS_UPDATE = None

# Compressor compute fused tilelang kernels: cat-rearrange + softmax +
# weighted-sum + bfloat16 cast in one launch each (overlap=True for
# ratio=4 layers, overlap=False for ratio=128 layers).
#
# History: an earlier Triton implementation (in the same file but later
# rewritten) caused `_SGL_MHC_PRE` (tilelang) to silently fail on first
# co-launch — the HC eager fallback fired, adding ~12k extra kernels per
# step and a 40% throughput regression. Rewriting in tilelang keeps both
# kernels on the same runtime and avoids the conflict.
if _DSV4_FUSED_KERNELS:
    try:
        from nanodeploy.models.deepseek_v4.compress_kernels import (
            compress_no_overlap_softmax_sum as _TILE_COMPRESS_NO_OVERLAP,
            compress_overlap_softmax_sum as _TILE_COMPRESS_OVERLAP,
        )
    except Exception:
        _TILE_COMPRESS_OVERLAP = None
        _TILE_COMPRESS_NO_OVERLAP = None
else:
    # Non-Hopper: eager cat-rearrange + softmax + weighted-sum fallback.
    _TILE_COMPRESS_OVERLAP = None
    _TILE_COMPRESS_NO_OVERLAP = None

# Vendored sglang DSV4 hyper-connection (HC) tilelang kernels. Fuses
# the eager F.linear + RMSNorm + sigmoid + sinkhorn + reduce chain in
# DeepseekV4HCProjector.forward into 2-3 tilelang kernels.
# Source: https://github.com/sgl-project/sglang
#   python/sglang/srt/layers/mhc.py
if _DSV4_FUSED_KERNELS:
    try:
        from nanodeploy._third_party.sglang_mhc import mhc_pre as _SGL_MHC_PRE
    except Exception:
        _SGL_MHC_PRE = None
else:
    # Non-Hopper: eager F.linear + RMSNorm + sigmoid + sinkhorn fallback.
    _SGL_MHC_PRE = None


def _maybe_fused_norm_rope(
    compressed: torch.Tensor,
    norm: nn.Module,
    rotary_emb: nn.Module,
    compressed_pos: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Try the single-kernel RMSNorm+RoPE; return None if unsupported.

    Replaces the pattern::

        out = norm(compressed)
        out_rope = _apply_rotary_interleaved(rotary_emb, pos, out[..., -rd:].unsqueeze(1)).squeeze(1)
        out = torch.cat([out[..., :-rd], out_rope], dim=-1)

    with one CUDA kernel that does both. The kernel mutates
    ``compressed`` in place and the caller continues to use the same
    tensor.

    Source: https://github.com/sgl-project/sglang
            python/sglang/jit_kernel/deepseek_v4.py::fused_norm_rope_inplace
    """
    if (
        _SGL_FUSED_NORM_ROPE is None
        or getattr(rotary_emb, "freqs_cis_cache", None) is None
        or getattr(norm, "add_unit_offset", False)
        or not compressed.is_cuda
        or compressed.dtype != torch.bfloat16
        or compressed.dim() != 2
    ):
        return None
    try:
        pos64 = (
            compressed_pos
            if compressed_pos.dtype == torch.int64
            else compressed_pos.long()
        )
        buf = compressed if compressed.is_contiguous() else compressed.contiguous()
        # The kernel is templated on (dtype, head_dim, rope_dim); JIT
        # builds on first call. weight dtype must equal kv dtype.
        weight = norm.weight
        if weight.dtype != buf.dtype:
            weight = weight.to(buf.dtype)
        _SGL_FUSED_NORM_ROPE(
            buf,
            weight,
            float(norm.eps),
            rotary_emb.freqs_cis_cache,
            pos64,
        )
        return buf
    except Exception:
        return None


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

    # Resolve context flags up front (these are Python bools/ints, not
    # tensors — no host↔device sync needed).
    is_prefill_run = True
    is_dummy = False
    try:
        context = get_context()
        is_prefill_run = bool(context.is_prefill)
        is_dummy = bool(getattr(context, "is_dummy", False))
    except Exception:
        pass

    # Skip warmup / graph-capture passes always: they don't carry real
    # data and would break CUDAGraph capture if we did any host sync.
    if is_dummy and os.getenv("NANODEPLOY_DSV4_DEBUG_SKIP_DUMMY", "1") != "0":
        return

    # Decode-step dumps require a host sync (.item() on context_lens) to
    # know which step we're on. That's only safe in eager mode — under
    # CUDAGraph replay the captured Python doesn't even run, and during
    # capture warmup we'd invalidate the stream. We disable decode dumps
    # entirely unless prefill-only is off AND we're not under graphs.
    decode_step: int | None = None
    if is_prefill_run:
        if os.getenv("NANODEPLOY_DSV4_DEBUG_PREFILL_ONLY", "1") != "0":
            pass  # prefill mode is the default and is graph-safe
    else:
        # Decode dumping disabled by default. Only enable with
        # NANODEPLOY_DSV4_DEBUG_PREFILL_ONLY=0 and require eager mode
        # (no CUDAGraph) — caller must run with --enforce_eager true.
        if os.getenv("NANODEPLOY_DSV4_DEBUG_PREFILL_ONLY", "1") != "0":
            return
        decode_steps_env = os.getenv("NANODEPLOY_DSV4_DEBUG_DECODE_STEPS", "")
        if not decode_steps_env.strip():
            return
        try:
            csl = getattr(get_context(), "context_lens", None)
            if csl is not None and csl.numel() > 0:
                decode_step = int(csl.flatten()[0].item())
        except Exception:
            return
        wanted = {
            int(x) for x in decode_steps_env.replace(",", " ").split() if x.strip()
        }
        if decode_step is None or decode_step not in wanted:
            return

    max_tokens = int(os.getenv("NANODEPLOY_DSV4_DEBUG_MAX_TOKENS", "8"))
    payload = tensor.detach()
    if payload.ndim > 0:
        payload = payload[:max_tokens]
    payload = payload.cpu().contiguous()
    layer = "global" if layer_idx is None else f"layer{layer_idx}"
    step_suffix = "" if decode_step is None else f"_step{decode_step}"
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / f"nanodeploy_rank{rank}_{layer}_{name}{step_suffix}.pt"
    if os.getenv("NANODEPLOY_DSV4_DEBUG_ONCE", "1") != "0" and file_path.exists():
        return
    torch.save(
        {
            "name": name,
            "rank": rank,
            "layer": layer_idx,
            "decode_step": decode_step,
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
    # Fast path: single-kernel RoPE via sglang's fused_rope. The kernel
    # is in-place and ``x`` is often a slice view of a bigger tensor
    # (e.g. ``kv[..., -rd:].unsqueeze(1)`` at the attention call site).
    # If we mutate that view in place and the caller then does
    # ``kv[..., -rd:] = result``, PyTorch detects the overlapping
    # memory and raises "some elements of the input tensor and the
    # written-to tensor refer to a single memory location". To match
    # the eager path's "return a fresh tensor" semantics we clone
    # first; the clone is one CUDA memcpy which is still cheaper than
    # the 7+ elementwise kernels in the eager fallback.
    # Adapted from https://github.com/sgl-project/sglang
    #   python/sglang/jit_kernel/deepseek_v4.py::fused_rope
    if (
        _SGL_FUSED_ROPE is not None
        and getattr(rotary_emb, "freqs_cis_cache", None) is not None
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and positions.dtype in (torch.int32, torch.int64)
        and positions.dim() == 1
        and x.dim() in (2, 3)
    ):
        squeeze_head = x.dim() == 2
        # Clone to a contiguous bf16 buffer so (a) the kernel's
        # last-dim-stride-1 contract is satisfied, and (b) we don't
        # mutate the caller's storage. ``.contiguous()`` is a no-op
        # when the slice is already C-contiguous, falling back to
        # ``.clone()`` so we always get a fresh tensor.
        x_buf = x.contiguous() if x.is_contiguous() else x.contiguous()
        if x_buf.data_ptr() == x.data_ptr():
            x_buf = x_buf.clone()
        x_view = x_buf.unsqueeze(1) if squeeze_head else x_buf
        # Compressor.forward_prefill passes ``positions[:cutoff:ratio]``
        # which is a strided view (stride=ratio). The kernel's
        # ``TensorMatcher({B})`` requires stride-1, so make it
        # contiguous here. Cheap when already stride-1.
        if not positions.is_contiguous():
            positions = positions.contiguous()
        try:
            _SGL_FUSED_ROPE(
                x_view, None, rotary_emb.freqs_cis_cache, positions, inverse
            )
            return x_view.squeeze(1) if squeeze_head else x_view
        except Exception as _exc:
            # Either the tensor-shape contract failed (unsupported
            # head_dim, non-contig stride) or the first-call JIT compile
            # failed. Fall through to the eager implementation.
            # One-time diagnostic: log why the fast path bailed so we
            # can fix it. Gated to fire once per (process, error class).
            global _FUSED_ROPE_WARNED
            if "_FUSED_ROPE_WARNED" not in globals():
                _FUSED_ROPE_WARNED = set()
            _key = type(_exc).__name__
            if _key not in _FUSED_ROPE_WARNED:
                _FUSED_ROPE_WARNED.add(_key)
                from nanodeploy.logging import get_logger

                get_logger().warning(
                    "fused_rope fast path bailed: %s. x.shape=%s dtype=%s "
                    "contig=%s positions.shape=%s dtype=%s inverse=%s. "
                    "Falling back to eager rope.",
                    _exc,
                    tuple(x.shape),
                    x.dtype,
                    x.is_contiguous(),
                    tuple(positions.shape),
                    positions.dtype,
                    inverse,
                )

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


def _apply_rotary_interleaved_inplace(
    rotary_emb: nn.Module,
    positions: torch.Tensor,
    x_view: torch.Tensor,
    inverse: bool = False,
) -> None:
    """In-place RoPE on ``x_view``. Caller's storage is mutated through
    the view; no return value, no write-back needed.

    The sglang ``fused_rope`` kernel accepts ``with_strides({-1, -1, 1})``
    — only the last dim must be stride-1, which is true for tail-dim
    slices like ``q[..., -rope_dim:]``. The non-inplace
    ``_apply_rotary_interleaved`` clones to a fresh contiguous buffer and
    the caller does ``q[..., -rope_dim:] = result`` (one ``index_put``
    write-back). Both are unnecessary when the kernel supports strided
    storage. This variant skips both, saving 2 kernels per call site.

    The eager fallback uses ``copy_`` instead of an assignment, achieving
    the same write-back-elision.
    """
    if (
        _SGL_FUSED_ROPE is not None
        and getattr(rotary_emb, "freqs_cis_cache", None) is not None
        and x_view.is_cuda
        and x_view.dtype == torch.bfloat16
        and positions.dtype in (torch.int32, torch.int64)
        and positions.dim() == 1
        and x_view.dim() in (2, 3)
        and x_view.stride(-1) == 1
    ):
        try:
            x3d = x_view.unsqueeze(1) if x_view.dim() == 2 else x_view
            pos = positions if positions.is_contiguous() else positions.contiguous()
            _SGL_FUSED_ROPE(x3d, None, rotary_emb.freqs_cis_cache, pos, inverse)
            return
        except Exception as _exc:
            global _FUSED_ROPE_INPLACE_WARNED
            if "_FUSED_ROPE_INPLACE_WARNED" not in globals():
                _FUSED_ROPE_INPLACE_WARNED = set()
            _key = type(_exc).__name__
            if _key not in _FUSED_ROPE_INPLACE_WARNED:
                _FUSED_ROPE_INPLACE_WARNED.add(_key)
                from nanodeploy.logging import get_logger

                get_logger().warning(
                    "fused_rope inplace fast path bailed: %s. Falling back "
                    "to eager rope + copy_.",
                    _exc,
                )
    # Eager fallback: compute then copy_back into x_view.
    cos_sin = rotary_emb.cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    x_pair = x_view.float().unflatten(-1, (-1, 2))
    x0 = x_pair[..., 0]
    x1 = x_pair[..., 1]
    if inverse:
        y0 = x0 * cos + x1 * sin
        y1 = x1 * cos - x0 * sin
    else:
        y0 = x0 * cos - x1 * sin
        y1 = x1 * cos + x0 * sin
    result = torch.stack((y0, y1), dim=-1).flatten(-2).to(x_view.dtype)
    x_view.copy_(result)


def _fp8_quant_dequant_inplace(x: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """Simulate official DSV4 KV FP8 QAT quant-dequant in-place.

    Fast path: a single triton kernel does the per-block amax + UE8M0
    scale + fp8 round-trip in one launch (~1 reduce + 1 elementwise
    instead of ~10 eager kernels). Falls back to the original eager
    chain on any failure.
    """
    if x.numel() == 0:
        return x
    if (
        _TRITON_FP8_QDQ is not None
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.shape[-1] % block_size == 0
    ):
        try:
            return _TRITON_FP8_QDQ(x, block_size)
        except Exception:
            pass

    # Eager fallback — full FP8 round-trip (NOT a no-op). See triton
    # kernel _ue8m0_quant_dequant_inplace_kernel for why we DON'T copy
    # reference's no-op semantics here: NanoDeploy's KV cache is FP8
    # (memory savings) so the local kv must be pre-quantized to keep
    # the eager-fallback attention path consistent with flash_mla.
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
    Fast path is a single triton kernel; falls back to the eager op chain
    if the kernel is unavailable.
    """
    assert kv_bf16.dtype == torch.bfloat16 and kv_bf16.shape[-1] == 512
    if _TRITON_FP8_PACK is not None and kv_bf16.is_cuda:
        try:
            return _TRITON_FP8_PACK(
                kv_bf16, _DSV4_NOPE_DIM, _DSV4_ROPE_DIM, _DSV4_TILE_SIZE
            )
        except Exception:
            pass

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
    # Fast path: single triton kernel does pack + scatter in one launch
    # (replaces 10+ elementwise/reduce launches that otherwise pile up
    # in the per-decode-step launch storm).
    if (
        _TRITON_FP8_STORE is not None
        and kv_bf16.is_cuda
        and cache_buf.is_cuda
        and cache_buf.dtype == torch.uint8
        and cache_buf.is_contiguous()
        and cache_buf.shape[-1] == _DSV4_BYTES_PER_TOKEN
    ):
        try:
            _TRITON_FP8_STORE(
                kv_bf16.contiguous(),
                cache_buf,
                slot_mapping,
                page_size,
                nope_dim=_DSV4_NOPE_DIM,
                rope_dim=_DSV4_ROPE_DIM,
                tile_size=_DSV4_TILE_SIZE,
            )
            return
        except Exception:
            pass

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
        # Fast path: vendored sglang ``mhc_pre`` collapses RMSNorm +
        # F.linear + sigmoid + sinkhorn + per-token reduction into a
        # 2-kernel pipeline (mhc_pre_gemm_sqrsum_splitk + mhc_pre_big_fuse).
        # Adapted from https://github.com/sgl-project/sglang
        #   python/sglang/srt/layers/mhc.py::mhc_pre
        shape, dtype = x.shape, x.dtype
        if (
            _SGL_MHC_PRE is not None
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and self.fn.dtype == torch.float32
        ):
            try:
                # mhc_pre expects residual shape [..., hc_mult, hidden];
                # nanodeploy's ``x`` is already that shape.
                post_mix, comb_mix, layer_input = _SGL_MHC_PRE(
                    x.contiguous(),
                    self.fn,
                    self.scale,
                    self.base,
                    rms_eps=self.eps,
                    hc_pre_eps=self.eps,
                    hc_sinkhorn_eps=self.eps,
                    hc_post_mult_value=2.0,  # nanodeploy: post = 2 * sigmoid(...)
                    sinkhorn_repeat=self.sinkhorn_iters,
                )
                # Outputs come out as fp32 (post_mix, comb_mix) and the
                # layer_input matches input dtype. Cast post/comb to
                # match nanodeploy's eager return contract.
                return (
                    layer_input,  # y
                    post_mix.squeeze(-1).to(dtype),  # post (drop trailing 1)
                    comb_mix.to(dtype),  # comb
                )
            except Exception:
                pass

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
        # Cached concat of wkv.weight + wgate.weight along the output
        # dim, lazily populated on first forward (after weight loading).
        # One FP32 Linear call replaces two cuBLAS gemvx calls per
        # compressed layer (~78/step → ~39/step). Stored as a plain
        # tensor (not nn.Parameter) so the loader still targets
        # ``wkv.weight`` and ``wgate.weight``.
        self._wkv_gate_weight: torch.Tensor | None = None
        self._coeff_head_dim_split = coeff * head_dim
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

    def _wkv_gate_forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One FP32 Linear call producing both kv_all and score_all.

        Replaces ``self.wkv(x), self.wgate(x)`` (two cuBLAS gemvx calls
        + two ``.float()`` casts) with one Linear call against the
        concatenated weight tensor. Lazily caches the concat on first
        call (after weight loading).
        """
        if self._wkv_gate_weight is None:
            with torch.no_grad():
                self._wkv_gate_weight = torch.cat(
                    [self.wkv.weight, self.wgate.weight], dim=0
                ).contiguous()
        out = F.linear(hidden_states.float(), self._wkv_gate_weight)
        d = self._coeff_head_dim_split
        return out[..., :d], out[..., d:]

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
        # Fused wkv + wgate: one cuBLAS call.
        kv_all, score_all = self._wkv_gate_forward(hidden_states)
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
        rd = self.rope_head_dim
        compressed_positions = positions[:cutoff:ratio]
        kv_dtype = kv.to(dtype).contiguous()
        fused = _maybe_fused_norm_rope(
            kv_dtype, self.norm, rotary_emb, compressed_positions
        )
        if fused is not None:
            kv = fused
        else:
            kv = self.norm(kv_dtype)
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
        # Dump prefill-compressor output (parity with reference's
        # Compressor.forward end-of-prefill dump). ratio is in the
        # filename so multiple compressor layers don't collide.
        _debug_dump(f"compressor_r{ratio}_compressed", kv, None)
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
        # Fused wkv + wgate: one cuBLAS call.
        kv_full, score_full = self._wkv_gate_forward(hidden_state)
        kv = kv_full.squeeze(0)
        score = score_full.squeeze(0) + self.ape[pos_mod]
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
        rd = self.rope_head_dim
        compressed_pos = hidden_state.new_tensor(
            [position + 1 - ratio], dtype=torch.long
        )
        compressed_dtype = compressed.to(dtype).contiguous()
        fused = _maybe_fused_norm_rope(
            compressed_dtype, self.norm, rotary_emb, compressed_pos
        )
        if fused is not None:
            compressed = fused
        else:
            compressed = self.norm(compressed_dtype)
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

        # Projections (already batched). Fused wkv + wgate Linear: one
        # cuBLAS gemvx call instead of two for the compressed-layer FP32
        # GEMVs.
        kv_all, score_all = self._wkv_gate_forward(hidden_states)

        # Triton fast path: fuses pos_mod + ape gather + update_idx +
        # 2 scatter writes into one kernel (6 kernels → 1).
        seq_slots_long = seq_slots.long()
        positions_long = positions.long()
        if _TRITON_COMPRESS_SCATTER_UPDATE is not None and self._kv_states.is_cuda:
            try:
                _TRITON_COMPRESS_SCATTER_UPDATE(
                    self._kv_states,
                    self._score_states,
                    kv_all,
                    score_all,
                    self.ape,
                    seq_slots_long,
                    positions_long,
                    ratio,
                    self.overlap,
                )
            except Exception:
                pos_mod = (positions % ratio).long()
                ape_vals = self.ape[pos_mod]
                update_idx = (ratio + pos_mod) if self.overlap else pos_mod
                self._kv_states[seq_slots_long, update_idx] = kv_all.float()
                self._score_states[seq_slots_long, update_idx] = (
                    score_all.float() + ape_vals
                )
        else:
            pos_mod = (positions % ratio).long()
            ape_vals = self.ape[pos_mod]
            update_idx = (ratio + pos_mod) if self.overlap else pos_mod
            self._kv_states[seq_slots_long, update_idx] = kv_all.float()
            self._score_states[seq_slots_long, update_idx] = (
                score_all.float() + ape_vals
            )

        # Compute compression for all bs sequences (batched)
        kv_st = self._kv_states[seq_slots.long()]  # [bs, coeff*ratio, coeff*head_dim]
        score_st = self._score_states[seq_slots.long()]  # same

        # Tilelang fast path: one kernel collapses cat-rearrange + softmax
        # + weighted-sum + bf16 cast (5-7 elementwise + 1 spatial softmax +
        # 1 reduce per call) into a single launch. Outputs bf16 directly so
        # the ``.to(dtype).contiguous()`` cast is also subsumed.
        compressed_dtype = None
        if self.overlap:
            if _TILE_COMPRESS_OVERLAP is not None and kv_st.is_cuda:
                try:
                    compressed_dtype = _TILE_COMPRESS_OVERLAP(kv_st, score_st, ratio)
                except Exception:
                    compressed_dtype = None
        else:
            if _TILE_COMPRESS_NO_OVERLAP is not None and kv_st.is_cuda:
                try:
                    compressed_dtype = _TILE_COMPRESS_NO_OVERLAP(kv_st, score_st)
                except Exception:
                    compressed_dtype = None

        if compressed_dtype is None:
            # Eager fallback (kept for non-CUDA / kernel JIT-failed paths).
            if self.overlap:
                # Reconstruct [bs, 2*ratio, head_dim] via gather-cat
                kv_for_c = torch.cat(
                    [kv_st[:, :ratio, :head_dim], kv_st[:, ratio:, head_dim:]], dim=1
                )
                score_for_c = torch.cat(
                    [score_st[:, :ratio, :head_dim], score_st[:, ratio:, head_dim:]],
                    dim=1,
                )
                compressed = (kv_for_c * score_for_c.softmax(dim=1)).sum(dim=1)
            else:
                compressed = (kv_st * score_st.softmax(dim=1)).sum(dim=1)
            compressed_dtype = compressed.to(dtype).contiguous()

        # Apply norm + RoPE + FP8 QAT (batched).
        # Fast path: one CUDA kernel for norm + interleaved RoPE.
        rd = self.rope_head_dim
        # Triton fast path: fuse the per-step compressor metadata math
        # (positions+1, sub-clamp, mod-eq) into one kernel that produces
        # both compressed_pos and should_compress (~5 small kernels → 1).
        if _TRITON_COMPUTE_COMPRESS_METADATA is not None and positions.is_cuda:
            try:
                compressed_pos, should_compress = _TRITON_COMPUTE_COMPRESS_METADATA(
                    positions, ratio
                )
            except Exception:
                compressed_pos = (positions + 1 - ratio).clamp(min=0)
                should_compress = (positions + 1) % ratio == 0
        else:
            compressed_pos = (positions + 1 - ratio).clamp(min=0)
            should_compress = (positions + 1) % ratio == 0
        fused = _maybe_fused_norm_rope(
            compressed_dtype, self.norm, rotary_emb, compressed_pos
        )
        if fused is not None:
            compressed = fused
        else:
            # Fallback: norm + slice rotate + cat (4–6 launches).
            compressed = self.norm(compressed_dtype)
            # cos_sin_cache has shape [max_pos, 1, rd] so x needs a head
            # axis: [bs, 1, rd], not [bs, rd].
            compressed_rope = _apply_rotary_interleaved(
                rotary_emb, compressed_pos, compressed[:, None, -rd:]
            ).squeeze(1)
            compressed = torch.cat([compressed[:, :-rd], compressed_rope], dim=-1)
        # In-place FP8 QAT on the nope portion (safe: compressed is a fresh tensor)
        _fp8_quant_dequant_inplace(compressed[:, :-rd], 64)

        # ``should_compress`` already computed above by
        # ``_TRITON_COMPUTE_COMPRESS_METADATA``.

        # Post-shift for overlap case: kv_state[:ratio] = kv_state[ratio:]
        # when should_compress[b]. Triton fast path collapses 8 kernels
        # (4 fancy gathers + 2 ``torch.where`` + 2 fancy index_puts)
        # into one in-place launch.
        if self.overlap:
            if _TRITON_COMPRESS_POST_SHIFT is not None and self._kv_states.is_cuda:
                try:
                    _TRITON_COMPRESS_POST_SHIFT(
                        self._kv_states,
                        self._score_states,
                        seq_slots.long(),
                        should_compress,
                        ratio,
                    )
                except Exception:
                    sc_mask = should_compress.unsqueeze(-1).unsqueeze(-1)
                    shifted_kv = self._kv_states[seq_slots.long(), ratio:]
                    shifted_score = self._score_states[seq_slots.long(), ratio:]
                    current_kv_head = self._kv_states[seq_slots.long(), :ratio]
                    current_score_head = self._score_states[seq_slots.long(), :ratio]
                    new_kv_head = torch.where(sc_mask, shifted_kv, current_kv_head)
                    new_score_head = torch.where(
                        sc_mask, shifted_score, current_score_head
                    )
                    self._kv_states[seq_slots.long(), :ratio] = new_kv_head
                    self._score_states[seq_slots.long(), :ratio] = new_score_head
            else:
                sc_mask = should_compress.unsqueeze(-1).unsqueeze(-1)
                shifted_kv = self._kv_states[seq_slots.long(), ratio:]
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
            dummy_slot = total_slots - 1
            if compressed_block_table is not None:
                # Triton fast path: collapses 6 kernels (count gather +
                # //, %, .clamp, table gather, *+, where) into 1 launch.
                if _TRITON_COMPRESS_PHYSICAL_SLOTS is not None:
                    try:
                        physical_slots = _TRITON_COMPRESS_PHYSICAL_SLOTS(
                            self._compressed_counts,
                            seq_slots.long(),
                            compressed_block_table,
                            should_compress,
                            page_size,
                            dummy_slot,
                        )
                    except Exception:
                        physical_slots = None
                else:
                    physical_slots = None

                if physical_slots is None:
                    cur_counts = self._compressed_counts[seq_slots.long()].long()
                    block_idx = (cur_counts // page_size).long()
                    tok_in_block = (cur_counts % page_size).long()
                    max_blocks = compressed_block_table.shape[1]
                    block_idx_safe = block_idx.clamp(max=max_blocks - 1)
                    page_ids = compressed_block_table[seq_slots.long(), block_idx_safe]
                    physical_slots = page_ids.long() * page_size + tok_in_block
                    physical_slots = torch.where(
                        should_compress, physical_slots, dummy_slot
                    )
            else:
                # Backward-compat: contiguous-chunk addressing (legacy path).
                cur_counts = self._compressed_counts[seq_slots.long()].long()
                valid_slots = (num_pages - 1) * page_size
                max_compressed = valid_slots // self._max_slots
                physical_slots = seq_slots.long() * max_compressed + cur_counts
                physical_slots = torch.where(
                    should_compress, physical_slots, dummy_slot
                )
            _store_dsv4_fp8_batched(
                compressed,
                compressed_cache,
                physical_slots.to(torch.int32),
                page_size,
            )
            # Update counts only for compressing seqs. Triton fast path
            # fuses the .to(int32) cast with the scatter_add atomic
            # increment (2 kernels → 1).
            if (
                _TRITON_COMPRESS_COUNTS_UPDATE is not None
                and self._compressed_counts.is_cuda
            ):
                try:
                    _TRITON_COMPRESS_COUNTS_UPDATE(
                        self._compressed_counts,
                        seq_slots.long(),
                        should_compress,
                    )
                except Exception:
                    inc = should_compress.to(torch.int32)
                    self._compressed_counts.scatter_add_(0, seq_slots.long(), inc)
            else:
                inc = should_compress.to(torch.int32)
                self._compressed_counts.scatter_add_(0, seq_slots.long(), inc)

        # Dump the compressed kv. We DO NOT gate on
        # ``bool(should_compress.any())`` because that would do a host
        # sync (.item()) and invalidate CUDAGraph capture. The dump
        # function itself early-returns when DEBUG_DIR is unset, so
        # this is a no-op in production. When debugging, the user is
        # expected to request a decode step where compression actually
        # fires (e.g. DECODE_STEPS=128 for prompt_len=8 ratio=128 — at
        # that step ratio=128, ratio=64, ratio=32, ratio=16, ratio=8,
        # ratio=4, ratio=2 all fire, so the dumped tensor is always
        # the real compressor output for the layers we care about).
        _debug_dump(
            f"compressor_r{ratio}_compressed",
            compressed,
            None,
        )

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

    # Class-level shared cache: layers with the same (compress_ratio, bs)
    # config share one FlashMLASchedMeta object. flash_mla repopulates the
    # meta on each invocation when topk/extra_topk values change (which they
    # do every step), but layers within the same step that share a meta
    # object trigger the populate kernel only once between them. With 3
    # unique compress_ratios in DSV4-Pro (0/4/128) and a fixed bs per step,
    # this cuts ``get_mla_metadata_kernel`` from 43/step to 3/step.
    _DSV4_SCHED_META_CACHE: dict = {}

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
        # NOTE: legacy per-instance cache; no longer used. The class-level
        # ``_DSV4_SCHED_META_CACHE`` (below) is shared across all
        # ``DeepseekV4Attention`` instances and keyed by
        # ``(compress_ratio, bs)`` so layers with the same config share one
        # meta object. flash_mla's ``get_mla_metadata_kernel`` then fires
        # 3x/step (one per unique compress_ratio) instead of 43x/step.
        self._dsv4_sched_metas: dict[int, object] = {}

        # Alt stream for compressor.forward_decode_batched. Lazily allocated on
        # first decode call (CUDA context must be initialized). Mirrors
        # sglang's _forward_prepare_multi_stream pattern: compressor runs on
        # this stream while KV store + SWA index construction run on the main
        # stream. The main stream waits before extra-index construction (which
        # reads compressor._compressed_counts). Compressor work (~2.5 ms)
        # dominates, so the gain is hiding KV store + SWA index (~100-200 µs).
        # CUDAGraph-safe: stream is created at warmup, sync points are CUDA
        # events captured inside the graph.
        self._compressor_stream: torch.cuda.Stream | None = None

        # Alt stream for the KV-side of attention prep (wkv → kv_norm →
        # k_rope → fp8_quant). Q-side (wq_a/b → q_norm → rmsnorm_self →
        # q_rope) runs on _q_stream concurrently. Sync before flash_mla.
        # The two paths are genuinely independent (only shared input is
        # hidden_states, which is read-only), so this is a real overlap.
        self._kv_stream: torch.cuda.Stream | None = None
        # Alt stream for the Q-side of attention prep. Mirrors KV-stream;
        # frees main stream to be a sync coordinator during prep.
        self._q_stream: torch.cuda.Stream | None = None

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
        # 2. Run compressor update — fully batched, CUDAGraph-safe.
        # Use scheduler-assigned state_slots (stable per-sequence identity
        # across decode steps). Falls back to arange(bs) when slots are not
        # provided (e.g., during warmup before scheduler populates them).
        # Compressor runs on an alt stream so it overlaps with the main
        # stream's KV store + SWA index construction. Main stream waits on
        # the alt stream before extra-index construction (which reads
        # compressor._compressed_counts).
        #
        # Capture main_stream and re-pin after the compressor's ``with`` block:
        # in profiler v18 the SWA-index kernel was observed to land on the
        # compressor alt stream for compressed layers (39 of 43). Pinning
        # explicitly to main_stream keeps everything except the compressor
        # work on the main stream. (v20 attempted this and regressed because
        # tilelang was not properly installed; with v24's tilelang in place,
        # retest.)
        main_stream = torch.cuda.current_stream()
        compressor_running = self.compress_ratio and ntps == 1
        if compressor_running:
            if self._compressor_stream is None:
                self._compressor_stream = torch.cuda.Stream()
            comp_stream = self._compressor_stream
            comp_stream.wait_stream(main_stream)

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
            with torch.cuda.stream(comp_stream):
                self.compressor.forward_decode_batched(
                    hidden_states[::ntps],
                    positions_per_seq,
                    self.rotary_emb,
                    seq_slots,
                    self.compressed_cache,
                    compressed_block_table=comp_bt,
                )

        # Re-pin the remaining work to the main stream (option 3): keeps
        # KV store, SWA index construction, extra index, and flash_mla on
        # main while compressor's ~2.5 ms runs on alt.
        torch.cuda.set_stream(main_stream)

        # KV store and SWA index construction nominally run on the main
        # stream, but in profiler v18 the SWA index kernel was observed to
        # land on the compressor alt stream for compressed layers (39 of the
        # 43 layers per step). Despite this leak the bench was 2x faster than
        # v20 (which forced everything onto main via ``set_stream``). The H200
        # is SM-saturated for this workload, so forcing more concurrency
        # caused contention and slowed every kernel ~2-3x. Leaving the leak
        # in place: the compressor stream then serializes its own follow-on
        # work, which empirically wins on this workload.
        _store_dsv4_fp8_batched(kv_2d, self.swa_cache, context.slot_mapping, page_size)

        # 3. Build SWA indices [bs, ntps, swa_topk] — vectorized, no per-seq loop
        swa_topk = ((self.window_size + 63) // 64) * 64  # align to 64
        # Clamp to min(ctx_len, window_size) — each Q token attends to at most
        # window_size recent tokens. Also guard against exceeding swa_topk.
        swa_topk_lengths = context_lens.clamp(max=min(self.window_size, swa_topk))

        # Fast path: one fused triton kernel for the entire SWA index
        # construction (~15 elementwise launches → 1). Falls back to the
        # eager torch chain on shape/dtype mismatch or kernel JIT failure.
        if (
            _TRITON_BUILD_SWA_INDICES is not None
            and context_lens.is_cuda
            and block_tables.is_cuda
            and context_lens.dtype == torch.int32
            and block_tables.dtype == torch.int32
        ):
            try:
                swa_indices = _TRITON_BUILD_SWA_INDICES(
                    context_lens,
                    block_tables,
                    swa_topk,
                    page_size,
                    min(self.window_size, swa_topk),
                ).unsqueeze(1)
            except Exception:
                swa_indices = None
        else:
            swa_indices = None

        if swa_indices is None:
            # Eager fallback (kept for non-CUDA / JIT-failed paths).
            token_offsets = torch.arange(
                swa_topk, device=q.device, dtype=torch.int32
            ).unsqueeze(
                0
            )  # [1, swa_topk]
            start_pos = (context_lens - swa_topk_lengths).unsqueeze(1)  # [bs, 1]
            logical_pos = start_pos + token_offsets  # [bs, swa_topk]
            valid_mask = token_offsets < swa_topk_lengths.unsqueeze(1)
            page_indices = logical_pos // page_size
            tok_in_page = logical_pos % page_size
            page_indices_safe = page_indices.clamp(0, block_tables.shape[1] - 1).long()
            physical_blocks = block_tables.gather(1, page_indices_safe)
            physical_slots = physical_blocks * page_size + tok_in_page
            swa_indices = torch.where(valid_mask, physical_slots, -1).to(torch.int32)
            swa_indices = swa_indices.unsqueeze(1)  # [bs, 1, swa_topk]

        # 4. Build compressed indices [bs, 1, extra_topk] (if compressed layers).
        # Wait on the compressor stream first — extra-index construction reads
        # compressor._compressed_counts which the compressor stream just wrote.
        if compressor_running:
            torch.cuda.current_stream().wait_stream(self._compressor_stream)

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

                # Fast path: one fused triton kernel for the entire paged
                # extra-indices construction (~15 elementwise launches → 1,
                # plus produces extra_topk_lengths). Eager fallback below.
                comp_counts = self.compressor._compressed_counts
                if (
                    _TRITON_BUILD_EXTRA_INDICES_PAGED is not None
                    and seq_slots.is_cuda
                    and comp_counts is not None
                    and comp_counts.is_cuda
                    and comp_bt.is_cuda
                    and comp_counts.dtype == torch.int32
                    and comp_bt.dtype == torch.int32
                ):
                    try:
                        extra_indices_2d, extra_topk_lengths = (
                            _TRITON_BUILD_EXTRA_INDICES_PAGED(
                                seq_slots,
                                comp_counts,
                                comp_bt,
                                extra_topk,
                                page_size_c,
                            )
                        )
                        extra_indices = extra_indices_2d.unsqueeze(1)
                    except Exception:
                        extra_indices = None
                else:
                    extra_indices = None

                if extra_indices is None:
                    # Eager fallback.
                    extra_topk_lengths = comp_counts[seq_slots].clamp(max=extra_topk)
                    tok_range = torch.arange(
                        extra_topk, device=q.device, dtype=torch.int32
                    )
                    block_idx = (tok_range // page_size_c).long()
                    tok_in_block = (tok_range % page_size_c).long()
                    block_idx_safe = block_idx.clamp(max=max_blocks - 1)
                    page_ids = comp_bt[
                        seq_slots.long().unsqueeze(1), block_idx_safe.unsqueeze(0)
                    ]
                    physical = page_ids.long() * page_size_c + tok_in_block.unsqueeze(0)
                    valid_mask = tok_range.unsqueeze(0) < extra_topk_lengths.unsqueeze(
                        1
                    )
                    extra_indices = torch.where(valid_mask, physical, -1).to(
                        torch.int32
                    )
                    extra_indices = extra_indices.unsqueeze(1)
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

        # Multi-stream attention prep: Q-side, KV-side, and (downstream)
        # compressor are all independent — shared input ``hidden_states``
        # is read-only. Q runs on ``_q_stream``, KV runs on ``_kv_stream``,
        # main is the sync coordinator. Mirrors sglang's
        # ``_forward_prepare_multi_stream`` pattern.
        if self._kv_stream is None:
            self._kv_stream = torch.cuda.Stream()
        if self._q_stream is None:
            self._q_stream = torch.cuda.Stream()
        kv_stream = self._kv_stream
        q_stream = self._q_stream
        main_stream = torch.cuda.current_stream()
        kv_stream.wait_stream(main_stream)
        q_stream.wait_stream(main_stream)

        # ─── KV-side on _kv_stream ───
        with torch.cuda.stream(kv_stream):
            kv_pre = self.wkv(hidden_states)
            _debug_dump("attn_kv_pre_norm", kv_pre, self.layer_idx)
            kv = self.kv_norm(kv_pre)
            _debug_dump("attn_kv", kv, self.layer_idx)
            kv = kv.unsqueeze(1)
            _apply_rotary_interleaved_inplace(
                self.rotary_emb, positions, kv[..., -self.rope_head_dim :]
            )
            _fp8_quant_dequant_inplace(kv[..., : -self.rope_head_dim], 64)

        # ─── Q-side on _q_stream (concurrent with KV-side) ───
        with torch.cuda.stream(q_stream):
            q_lora_pre = self.wq_a(hidden_states)
            _debug_dump("attn_q_lora_pre_norm", q_lora_pre, self.layer_idx)
            q = self.q_norm(q_lora_pre)
            _debug_dump("attn_q_lora", q, self.layer_idx)
            q_flat = self.wq_b(q)
            _debug_dump("attn_wq_b", q_flat, self.layer_idx)
            q = q_flat.view(q_len, self.num_heads, self.head_dim)
            # Fast path: vendored sglang per-head RMSNorm (in-place, no
            # weight) collapses square + mean + rsqrt + mul into one CUDA
            # kernel. Falls back to the eager 5-launch chain when the
            # vendor isn't built or shapes don't match.
            if (
                _SGL_RMSNORM_SELF is not None
                and q.is_cuda
                and q.dtype == torch.bfloat16
                and q.is_contiguous()
                and q.shape[-1] == self.head_dim
            ):
                try:
                    q = _SGL_RMSNORM_SELF(q, float(self.rms_norm_eps))
                except Exception as _exc:
                    global _RMSNORM_SELF_WARNED
                    if "_RMSNORM_SELF_WARNED" not in globals():
                        _RMSNORM_SELF_WARNED = set()
                    _key = type(_exc).__name__
                    if _key not in _RMSNORM_SELF_WARNED:
                        _RMSNORM_SELF_WARNED.add(_key)
                        from nanodeploy.logging import get_logger

                        get_logger().warning(
                            "rmsnorm_self fast path bailed: %s. q.shape=%s "
                            "dtype=%s contig=%s. Eager fallback.",
                            _exc,
                            tuple(q.shape),
                            q.dtype,
                            q.is_contiguous(),
                        )
                    q = q * torch.rsqrt(
                        q.square().mean(-1, keepdim=True) + self.rms_norm_eps
                    )
            else:
                q = q * torch.rsqrt(
                    q.square().mean(-1, keepdim=True) + self.rms_norm_eps
                )
            _debug_dump("attn_q_normed", q, self.layer_idx)
            # In-place RoPE on Q's rope tail.
            _apply_rotary_interleaved_inplace(
                self.rotary_emb, positions, q[..., -self.rope_head_dim :]
            )

        # Sync: main stream waits on both Q and KV streams before flash_mla.
        main_stream.wait_stream(kv_stream)
        main_stream.wait_stream(q_stream)
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
        # DSV4 ships ``swiglu_limit=10.0`` (clamp silu(gate)*up before
        # FP8 quant). Models without this attr leave it ``None`` and
        # the SwiGLU kernels run with ``+inf`` (no-op clamp).
        swiglu_limit = getattr(config, "swiglu_limit", None)
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
            swiglu_limit=swiglu_limit,
        )
        assert config.n_shared_experts == 1
        self.shared_experts = DeepseekV2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            hidden_act=config.hidden_act,
            config=config,
            quantization_config=quantization_config,
        )
        # Alt stream for shared_experts overlap with routed_experts (deep_ep
        # dispatch + experts + combine). Mirrors sglang's SBO pattern: while
        # routed_experts is mostly RDMA-bound (dispatch + combine ~2 ms), the
        # shared MLP (~500 µs of FP8 GEMMs) runs concurrently on the alt
        # stream. Lazily allocated on first forward.
        self._shared_stream: torch.cuda.Stream | None = None

    def _scores(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        if self.score_func == "softmax":
            return logits.softmax(dim=-1)
        if self.score_func == "sigmoid":
            return logits.sigmoid()
        if self.score_func == "sqrtsoftplus":
            return F.softplus(logits).sqrt()
        raise ValueError(f"Unsupported DeepseekV4 score_func={self.score_func}")

    @staticmethod
    def _routing_scores_with_bias(
        logits: torch.Tensor, bias: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused: gate-logits → sqrtsoftplus scores (untouched, used for
        weight gather later) and choice_scores (with e_score_correction_bias
        applied). inductor fuses softplus + sqrt + add (3 launches → 1).
        """
        return _routing_scores_with_bias_compiled(logits, bias)

    @staticmethod
    def _normalize_topk_weights(
        scores: torch.Tensor,
        topk_ids: torch.Tensor,
        route_scale: float,
    ) -> torch.Tensor:
        """Fused: gather → renorm by sum → mul by route_scale. Inductor
        fuses the post-gather chain (4-5 launches → 1)."""
        return _normalize_topk_weights_compiled(scores, topk_ids, route_scale)

    @staticmethod
    def _fuse_routed_shared(
        routed_out: torch.Tensor, shared_out: torch.Tensor
    ) -> torch.Tensor:
        """Compile-fused post-MoE combine: ``out = routed + shared``.

        Trivially small (one bf16 add on [T, 4096]) but each MoE layer
        otherwise emits a free-floating ``vectorized_elementwise_kernel``
        launch under graph replay. Putting this in a compile region
        lets inductor sometimes co-locate it with adjacent kernels.

        Note: ``route_scale`` is already baked into ``topk_weights`` by
        ``_normalize_topk_weights`` upstream, and the cross-expert
        topk-weighted sum is done by ``deep_ep::internode_ll::combine``
        inside ``routed_experts``. sglang's
        ``moe_sum_reduce_warp_per_token_vec_kernel`` is the TP-topology
        equivalent of those, so unnecessary under EP."""
        return _fuse_routed_shared_compiled(routed_out, shared_out)

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        residual = hidden_states
        logits = self.gate(hidden_states)
        if self.hash:
            scores = self._scores(logits)
            topk_ids = self.gate.tid2eid[input_ids].long()
            topk_weights = scores.gather(1, topk_ids)
            if self.score_func != "softmax":
                topk_weights = topk_weights / (
                    topk_weights.sum(dim=-1, keepdim=True) + 1e-20
                )
            topk_weights = topk_weights * self.route_scale
        elif (
            self.score_func == "sqrtsoftplus"
            and self.gate.e_score_correction_bias is not None
        ):
            # Fast path: torch.compile-fused scoring + topk-normalization.
            # Covers the production DSV4 config; falls through for softmax
            # / sigmoid score_funcs and bias-less gates.
            scores, choice_scores = self._routing_scores_with_bias(
                logits, self.gate.e_score_correction_bias
            )
            topk_ids = torch.topk(choice_scores, k=self.top_k, dim=-1, sorted=False)[1]
            topk_weights = self._normalize_topk_weights(
                scores, topk_ids, self.route_scale
            )
        else:
            scores = self._scores(logits)
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

        # SBO: launch shared_experts on alt stream BEFORE routed_experts so
        # the dense shared MLP overlaps with deep_ep dispatch + expert compute
        # + combine. Main stream waits before _fuse_routed_shared. Skip on
        # prefill (routed_experts behaviour differs and the alt stream's
        # benefit is small for batched prefill).
        is_prefill = get_context().is_prefill
        use_shared_stream = not is_prefill
        if use_shared_stream:
            if self._shared_stream is None:
                self._shared_stream = torch.cuda.Stream()
            shared_stream = self._shared_stream
            main_stream = torch.cuda.current_stream()
            shared_stream.wait_stream(main_stream)
            with torch.cuda.stream(shared_stream):
                shared = self.shared_experts(residual)
            # Re-pin to main_stream after the with block (option 3).
            torch.cuda.set_stream(main_stream)
            out = self.routed_experts(
                hidden_states, topk_ids, topk_weights, is_prefill=is_prefill
            )
            main_stream.wait_stream(shared_stream)
        else:
            out = self.routed_experts(
                hidden_states, topk_ids, topk_weights, is_prefill=is_prefill
            )
            shared = self.shared_experts(residual)
        _debug_dump("moe_routed_out", out, self.layer_idx)
        _debug_dump("moe_shared_out", shared, self.layer_idx)
        out = self._fuse_routed_shared(out, shared)
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
        # Alt stream for HC pre/post tilelang kernels (option 2). Sglang
        # records these on alt streams in their trace; mirroring that lets
        # the SM scheduler co-issue them with neighbouring main-stream
        # kernels. Lazily allocated on first forward.
        self._hc_stream: torch.cuda.Stream | None = None

    @staticmethod
    def _hc_post(
        x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
    ):
        # Reference (model.py:684-687):
        #     y = post[..., None] * x[..., None, :]
        #         + sum(comb[..., None] * residual[..., None, :], dim=-3)
        # i.e. y[..., k, d] = sum_j comb[..., j, k] * residual[..., j, d]
        # — that's ``comb.T @ residual`` over the hc axis. The earlier
        # NanoDeploy form ``residual.unsqueeze(1)`` + ``sum(dim=2)``
        # contracted on the wrong index (computed ``comb @ residual``),
        # which only matches the reference when ``comb`` is symmetric;
        # after sinkhorn it generally isn't, so every layer accumulated
        # token-level drift twice (post-attn and post-ffn). Verified
        # against the 4D reference at ``T=8, hc=4, d=8`` to give a
        # bit-identical result.
        #
        # ``@torch.compile`` lets inductor fuse the broadcast-multiply +
        # reduce-sum + add chain (4-5 launches per call) into a single
        # fused-reduce kernel. Called twice per layer × 43 layers per
        # decode step = 86 calls/step → ~350 launches/step collapsed.
        return _hc_post_compiled(x, residual, post, comb)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
    ):
        _debug_dump("layer_input", hidden_states, self.layer_idx)
        residual = hidden_states

        # Option 2: HC pre/post on alt stream. Mirrors sglang's per-stream
        # kernel attribution. Limited overlap window (HC outputs feed
        # directly into the next op), but the SM scheduler can still
        # co-issue these tilelang kernels with main-stream work.
        if self._hc_stream is None:
            self._hc_stream = torch.cuda.Stream()
        hc_stream = self._hc_stream
        main_stream = torch.cuda.current_stream()

        hc_stream.wait_stream(main_stream)
        with torch.cuda.stream(hc_stream):
            x, post, comb = self.hc_attn(hidden_states)
        torch.cuda.set_stream(main_stream)
        main_stream.wait_stream(hc_stream)
        _debug_dump("hc_attn_y", x, self.layer_idx)
        _debug_dump("hc_attn_post", post, self.layer_idx)
        _debug_dump("hc_attn_comb", comb, self.layer_idx)
        x = self.input_layernorm(x)
        _debug_dump("attn_norm", x, self.layer_idx)
        x = self.self_attn(positions, x)
        _debug_dump("attn_block_out", x, self.layer_idx)

        hc_stream.wait_stream(main_stream)
        with torch.cuda.stream(hc_stream):
            hidden_states = self._hc_post(x, residual, post, comb)
        torch.cuda.set_stream(main_stream)
        main_stream.wait_stream(hc_stream)
        _debug_dump("after_attn_hc", hidden_states, self.layer_idx)

        residual = hidden_states
        hc_stream.wait_stream(main_stream)
        with torch.cuda.stream(hc_stream):
            x, post, comb = self.hc_ffn(hidden_states)
        torch.cuda.set_stream(main_stream)
        main_stream.wait_stream(hc_stream)
        _debug_dump("hc_ffn_y", x, self.layer_idx)
        x = self.post_attention_layernorm(x)
        _debug_dump("ffn_norm", x, self.layer_idx)
        x = self.mlp(x, input_ids)
        _debug_dump("ffn_block_out", x, self.layer_idx)

        hc_stream.wait_stream(main_stream)
        with torch.cuda.stream(hc_stream):
            out = self._hc_post(x, residual, post, comb)
        torch.cuda.set_stream(main_stream)
        main_stream.wait_stream(hc_stream)
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
