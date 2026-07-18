"""NSA (Native Sparse Attention) Indexer for DeepSeek V3.2.

The Indexer is a lightweight MLP that runs per-layer during decode to select
which KV cache blocks each query token should attend to (sparse attention).

Architecture:
  - wq_b:          q_lora_rank -> n_heads * head_dim  (FP8 weight)
  - wk:            hidden_size -> head_dim             (FP8 weight)
  - k_norm:        LayerNorm(head_dim)
  - weights_proj:  hidden_size -> n_heads              (BF16 weight)
  - rotary_emb:    RoPE for first rope_head_dim dims
  - Hadamard rotation applied to query and key after RoPE

Forward flow (decode):
  1. Project q_lora -> query (n_heads, head_dim=128)
  2. Project hidden_states -> key (head_dim=128), apply k_norm
  3. Apply RoPE to first rope_head_dim=64 dims of query and key
  4. Apply Hadamard rotation to query and key
  5. Quantize query to FP8 (per-token, UE8M0 scale)
  6. Store key to indexer FP8 cache (quantize + pack)
  7. Compute gate weights from weights_proj
  8. Compute FP8 paged MQA logits: q_fp8 @ kv_cache_fp8 + gate
  9. TopK selection -> block indices for sparse attention

Weight names in HF checkpoint:
  model.layers.{i}.self_attn.indexer.wq_b.weight          (8192, 1536) FP8
  model.layers.{i}.self_attn.indexer.wq_b.weight_scale_inv (64, 12) FP32
  model.layers.{i}.self_attn.indexer.wk.weight             (128, 7168) FP8
  model.layers.{i}.self_attn.indexer.wk.weight_scale_inv   (1, 56) FP32
  model.layers.{i}.self_attn.indexer.k_norm.weight         (128,) FP32
  model.layers.{i}.self_attn.indexer.k_norm.bias           (128,) FP32
  model.layers.{i}.self_attn.indexer.weights_proj.weight   (64, 7168) BF16
"""

import math
import os

import deep_gemm
import torch
import torch.nn as nn
from fast_hadamard_transform import hadamard_transform

from dlengine.kernel.jit.sgl import fused_kernels_enabled
from dlengine.kernel.jit.sgl.deepseek_v4 import indexer_q_rope_hadamard_quant
from dlengine.kernel.triton.generic.fp8_ue8m0_quant import store_indexer_key_fp8_fused
from dlengine.kernel.triton.generic.indexer_transform import (
    indexer_k_rope_inplace,
    indexer_k_transform_store_fp8,
    indexer_layer_norm_bf16,
    indexer_qk_rope_inplace,
)
from dlengine.kernel.triton.hopper.block_gemm_fp8 import quant_fp8
from dlengine.layers import get_backend
from dlengine.layers.base_backend import ReplicatedLinearBase
from dlengine.layers.rotary_embedding import get_rope
from dlengine.logging import get_logger

logger = get_logger()


# FP8 quantization tile size (matches deep_gemm per_token_cast_to_fp8)
INDEXER_QUANT_BLOCK_SIZE = 128


def _per_token_cast_to_fp8_ue8m0(x: torch.Tensor):
    """Graph-safe per-token FP8 quantization with UE8M0 scales.

    Equivalent to deep_gemm.per_token_cast_to_fp8(x, use_ue8m0=True) but without
    .item() calls that would break CUDA graph capture.
    """
    assert x.dim() == 2
    m, n = x.shape
    # Pad to 128-byte alignment (same as deep_gemm)
    padded_n = (n + 127) // 128 * 128
    x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(min=1e-4)
    sf = x_amax / 448.0
    # ceil_to_ue8m0: round up to nearest power of 2 (no .item() call)
    sf = torch.exp2(torch.ceil(torch.log2(sf)))
    x_fp8 = (
        (x_view * (1.0 / sf.unsqueeze(2)))
        .to(torch.float8_e4m3fn)
        .view(m, padded_n)[:, :n]
        .contiguous()
    )
    return x_fp8, sf


def _hadamard_rotate(x: torch.Tensor) -> torch.Tensor:
    """Apply Hadamard rotation with normalization scaling."""
    hidden_size = x.size(-1)
    return hadamard_transform(x.contiguous(), scale=hidden_size**-0.5)


def _interleaved_to_half(x: torch.Tensor) -> torch.Tensor:
    """Convert RoPE dims from interleaved to half format."""
    *leading, d = x.shape
    return x.unflatten(-1, (-1, 2)).transpose(-1, -2).contiguous().flatten(-2)


def _weighted_relu_mqa_scores(
    query: torch.Tensor,
    weights: torch.Tensor,
    key: torch.Tensor,
    head_chunk: int = 4,
) -> torch.Tensor:
    """Compute exact Lightning-Indexer scores in bounded workspace.

    DeepGEMM defines the MQA score for query ``i`` and key ``j`` as::

        sum_h weights[i, h] * relu(dot(query[i, h], key[j]))

    ReLU is applied before the weighted head reduction, so gate weights cannot
    be folded into one query vector. Chunking the head dimension limits the
    largest temporary to ``[num_queries, head_chunk, num_keys]``.
    """
    if query.ndim != 3:
        raise RuntimeError(f"query must be [Q, H, D], got {query.shape}")
    if weights.shape != query.shape[:2]:
        raise RuntimeError(
            f"weights/query shape mismatch: {weights.shape} vs {query.shape[:2]}"
        )
    if key.ndim != 2 or key.shape[1] != query.shape[2]:
        raise RuntimeError(
            f"key/query shape mismatch: key={key.shape}, query={query.shape}"
        )
    if head_chunk <= 0:
        raise ValueError(f"head_chunk must be positive, got {head_chunk}")

    num_queries, num_heads, _ = query.shape
    num_keys = key.shape[0]
    if num_keys == 0:
        return torch.empty(
            num_queries, 0, dtype=torch.float32, device=query.device
        )

    query_f = query.float()
    weights_f = weights.float()
    key_t = key.float().T
    scores = torch.zeros(
        num_queries, num_keys, dtype=torch.float32, device=query.device
    )
    for h_start in range(0, num_heads, head_chunk):
        h_end = min(h_start + head_chunk, num_heads)
        head_scores = torch.matmul(query_f[:, h_start:h_end], key_t)
        head_scores.relu_()
        head_scores.mul_(weights_f[:, h_start:h_end, None])
        scores.add_(head_scores.sum(dim=1))
    return scores


def _expand_decode_context_lens(
    context_lens: torch.Tensor, next_n: int
) -> torch.Tensor:
    """Return DeepGEMM's ``[batch, next_n]`` context-length layout.

    ``context_lens`` contains the length after the last query token.  For
    multi-token decode (MTP, or an inactive DP rank's dummy batch), each query
    needs its own causal length.  DeepGEMM specializes both its metadata and
    logits kernel on ``next_n``, so passing ``[batch, 1]`` metadata with a
    ``[batch, next_n, ...]`` query is invalid.
    """
    if context_lens.dim() == 1:
        context_lens = context_lens[:, None]
    if context_lens.dim() != 2 or context_lens.shape[1] not in (1, next_n):
        raise ValueError(
            "Indexer context_lens must have shape [batch], [batch, 1], or "
            f"[batch, next_n]; got {tuple(context_lens.shape)} for next_n={next_n}"
        )
    if context_lens.shape[1] == next_n:
        return context_lens.to(torch.int32)

    offsets = torch.arange(
        next_n - 1,
        -1,
        -1,
        dtype=context_lens.dtype,
        device=context_lens.device,
    )
    return (context_lens - offsets).clamp_min(1).to(torch.int32)


class IndexerCache:
    """Per-layer FP8 cache for indexer keys.

    Layout per page (block_size=64 tokens):
        [64 * 128] bytes FP8 key data + [64 * 4] bytes FP32 per-token scale
        = 64 * 132 = 8448 bytes per page

    Storage: Single contiguous tensor (num_layers, num_pages, page_size * 132)
    as uint8. Per-layer views are accessible via ``buffers`` property or
    ``get_buffer(layer_id)``. The contiguous layout enables single-MR RDMA
    registration for PD disaggregation.
    """

    def __init__(
        self,
        num_layers: int,
        num_pages: int,
        page_size: int,
        head_dim: int,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_pages = num_pages
        self.page_size = page_size
        self.head_dim = head_dim
        self.quant_block_size = INDEXER_QUANT_BLOCK_SIZE
        self.bytes_per_token = head_dim + head_dim // self.quant_block_size * 4
        # Single contiguous buffer: (num_layers, num_pages, page_size * bytes_per_token)
        self.buffer = torch.zeros(
            (num_layers, num_pages, page_size * self.bytes_per_token),
            dtype=torch.uint8,
            device=device,
        )

    @property
    def buffers(self) -> list[torch.Tensor]:
        """Per-layer views into the contiguous buffer (backward compatible)."""
        return [self.buffer[i] for i in range(self.num_layers)]

    def get_buffer(self, layer_id: int) -> torch.Tensor:
        return self.buffer[layer_id]

    def store_key_fp8(
        self,
        layer_id: int,
        key_bf16: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        """Quantize key to FP8 + scale and write into paged buffer.

        Args:
            key_bf16: (num_tokens, head_dim) bfloat16
            slot_mapping: (num_tokens,) int — flat slot indices
        """
        buf = self.buffer[layer_id]
        num_tokens = key_bf16.shape[0]
        head_dim = self.head_dim
        page_size = self.page_size
        bpt = self.bytes_per_token

        if key_bf16.is_cuda:
            store_indexer_key_fp8_fused(
                key_bf16.contiguous(),
                buf,
                slot_mapping,
                page_size,
                group_size=self.quant_block_size,
                eps=1e-4,
            )
            return

        # Quantize: per-token FP8 with UE8M0 scale (graph-safe)
        key_fp8, key_scale = _per_token_cast_to_fp8_ue8m0(key_bf16)

        # Clamp invalid slots (-1) to 0 for graph-safe scatter (writes harmlessly to slot 0)
        safe_slots = torch.where(
            slot_mapping >= 0, slot_mapping, torch.zeros_like(slot_mapping)
        )

        # Compute page/offset for each token.
        #
        # IMPORTANT: deep_gemm's fp8_paged_mqa_logits (and sglang) expect each
        # page laid out as a contiguous FP8 block for ALL tokens, followed by a
        # contiguous scale block for all tokens:
        #     [tok0_fp8(128) .. tok63_fp8(128)] [tok0_scale(4) .. tok63_scale(4)]
        # i.e. SCALE_OFFSET = page_size * head_dim. (NOT per-token interleaved
        # [fp8|scale]; that mislayout makes the kernel read scale bytes from the
        # middle of the FP8 data, yielding ~1e29 garbage logits that only matter
        # once context exceeds index_topk and real top-k selection kicks in.)
        page_idx = safe_slots // page_size  # (N,)
        offset_in_page = safe_slots % page_size  # (N,)
        fp8_byte_offset = offset_in_page * head_dim  # (N,) within fp8 block
        scale_byte_offset = page_size * head_dim + offset_in_page * 4  # (N,)

        # Vectorised scatter into flat buffer view
        row_stride = page_size * bpt
        flat_base = page_idx.long() * row_stride  # (N,)

        # FP8 data indices: (N, head_dim)
        byte_range = torch.arange(head_dim, device=key_bf16.device)
        flat_fp8_idx = (flat_base + fp8_byte_offset.long()).unsqueeze(
            1
        ) + byte_range  # (N, head_dim)

        # Scale data indices: (N, 4)
        scale_range = torch.arange(4, device=key_bf16.device)
        flat_scale_idx = (flat_base + scale_byte_offset.long()).unsqueeze(
            1
        ) + scale_range  # (N, 4)

        fp8_bytes = key_fp8.view(torch.uint8)  # (N, head_dim)
        scale_bytes = (
            key_scale.view(torch.float32)
            .contiguous()
            .view(torch.uint8)
            .expand(num_tokens, 4)
        )  # (N, 4)

        buf_flat = buf.view(-1)
        buf_flat.scatter_(0, flat_fp8_idx.reshape(-1), fp8_bytes.reshape(-1))
        buf_flat.scatter_(0, flat_scale_idx.reshape(-1), scale_bytes.reshape(-1))


class Indexer(nn.Module):
    """NSA Indexer — selects which KV cache blocks to attend to.

    Args:
        hidden_size: Model hidden size (7168 for V3.2)
        index_n_heads: Number of indexer heads (64 for V3.2)
        index_head_dim: Indexer head dimension (128 for V3.2)
        qk_rope_head_dim: RoPE dimensions (64 for V3.2)
        q_lora_rank: Q LoRA rank from main attention (1536 for V3.2)
        index_topk: Number of top-k tokens to select (2048 for V3.2)
        max_position_embeddings: Max sequence length
        rope_theta: RoPE base frequency
        rope_scaling: RoPE scaling config dict
        layer_id: Layer index (for cache buffer selection)
    """

    _sel_dump_logged = False

    def __init__(
        self,
        hidden_size: int,
        index_n_heads: int,
        index_head_dim: int,
        qk_rope_head_dim: int,
        q_lora_rank: int,
        index_topk: int,
        max_position_embeddings: int,
        rope_theta: float,
        rope_scaling: dict | None,
        layer_id: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = index_n_heads
        self.head_dim = index_head_dim
        self.rope_head_dim = qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.index_topk = index_topk
        self.layer_id = layer_id
        self.softmax_scale = index_head_dim**-0.5

        # Linear projections
        self.wq_b: ReplicatedLinearBase = get_backend().get_replicated_linear(
            q_lora_rank,
            index_n_heads * index_head_dim,
            bias=False,
        )
        self.wk: ReplicatedLinearBase = get_backend().get_replicated_linear(
            hidden_size,
            index_head_dim,
            bias=False,
        )
        # weights_proj is BF16 in the checkpoint (not FP8-quantized),
        # use plain nn.Linear to avoid FP8 GEMM path.
        self.weights_proj = nn.Linear(
            hidden_size,
            index_n_heads,
            bias=False,
            dtype=torch.bfloat16,
        )

        # k_norm: LayerNorm with bias (FP32 weights in checkpoint)
        self.k_norm = nn.LayerNorm(index_head_dim, dtype=torch.float32)

        # RoPE for indexer (same config as main attention)
        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            rotary_dim=qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        self.sm_count = deep_gemm.get_num_sms()

        # Indexer cache reference (set externally after cache allocation)
        self.indexer_cache: IndexerCache | None = None

    def build_schedule_metadata(self, context_lens: torch.Tensor) -> torch.Tensor:
        """Build the per-step DeepGEMM schedule shared by all Indexer layers."""
        assert self.indexer_cache is not None
        context_lens = context_lens.to(torch.int32)
        if context_lens.dim() == 1:
            context_lens = context_lens[:, None]
        return deep_gemm.get_paged_mqa_logits_metadata(
            context_lens, self.indexer_cache.page_size, self.sm_count
        )

    def _compute_q_k(
        self,
        q_lora: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        defer_query_transform: bool = False,
        defer_key_transform: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project and transform Q and K for indexer scoring.

        Args:
            q_lora: (num_tokens, q_lora_rank) — intermediate Q from main attention
            hidden_states: (num_tokens, hidden_size) — input to this layer
            positions: (num_tokens,) — position indices

        Returns:
            query: (num_tokens, n_heads, head_dim) BF16
            key: (num_tokens, head_dim) BF16
        """
        num_tokens = q_lora.shape[0]

        # Q projection: q_lora -> (N, n_heads * head_dim) -> (N, n_heads, head_dim)
        query = self.wq_b(q_lora)
        query = query.view(num_tokens, self.n_heads, self.head_dim)

        # K projection + LayerNorm
        key = self.wk(hidden_states)
        if key.is_cuda and not defer_key_transform:
            key = indexer_layer_norm_bf16(
                key.contiguous(),
                self.k_norm.weight,
                self.k_norm.bias,
                self.k_norm.eps,
            )
            if defer_query_transform:
                indexer_k_rope_inplace(
                    key,
                    positions,
                    self.rotary_emb.cos_sin_cache,
                    self.rope_head_dim,
                )
            else:
                indexer_qk_rope_inplace(
                    query,
                    key,
                    positions,
                    self.rotary_emb.cos_sin_cache,
                    self.rope_head_dim,
                )
        elif not defer_key_transform:
            key = self.k_norm(key.float()).to(key.dtype)

            # Split rope / non-rope portions
            q_rope = query[..., : self.rope_head_dim]
            k_rope = key[..., : self.rope_head_dim]

            # Convert from interleaved to half format (consistent with main attention)
            q_rope = _interleaved_to_half(q_rope)
            k_rope_3d = _interleaved_to_half(k_rope.unsqueeze(1))

            # Apply RoPE
            q_rope, k_rope_3d = self.rotary_emb(positions, q_rope, k_rope_3d)

            # Write back rotated values
            query[..., : self.rope_head_dim] = q_rope
            key[..., : self.rope_head_dim] = k_rope_3d.squeeze(1)

        # Hadamard rotation
        if not defer_query_transform:
            query = _hadamard_rotate(query)
        if not defer_key_transform:
            key = _hadamard_rotate(key)

        return query, key

    def _compute_key(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute indexer key only (K-path of _compute_q_k).

        Used during prefill to store keys without running the full scoring pipeline.

        Args:
            hidden_states: (num_tokens, hidden_size)
            positions: (num_tokens,)

        Returns:
            key: (num_tokens, head_dim) BF16
        """
        key = self.wk(hidden_states)
        key = self.k_norm(key.float()).to(key.dtype)

        k_rope = key[..., : self.rope_head_dim]
        k_rope_3d = _interleaved_to_half(k_rope.unsqueeze(1))

        # RoPE needs a dummy q; pass k_rope_3d as both q and k, discard q output
        _, k_rope_3d = self.rotary_emb(positions, k_rope_3d, k_rope_3d)

        key[..., : self.rope_head_dim] = k_rope_3d.squeeze(1)
        key = _hadamard_rotate(key)
        return key

    def store_prefill_keys(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        """Compute and store indexer keys during prefill.

        This populates the indexer FP8 cache so that decode can score against
        all previously-seen tokens.

        Args:
            hidden_states: (num_tokens, hidden_size)
            positions: (num_tokens,)
            slot_mapping: (num_tokens,) int — flat slot indices
        """
        assert self.indexer_cache is not None, "IndexerCache not initialized"
        key = self._compute_key(hidden_states, positions)
        self.indexer_cache.store_key_fp8(self.layer_id, key, slot_mapping)

    def compute_prefill_topk(
        self,
        q_lora: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        query_chunk: int = 256,
    ) -> torch.Tensor:
        """Per-query top-k selection for the non-prefix (single-chunk) prefill.

        Mirrors the lightning-indexer scoring used at decode, but produces a
        causal top-k for *every* query token instead of just the last one.
        Computed in BF16/FP32 (no FP8 quantization) — the selection (a topk
        argmax) is robust to that, and this avoids the paged-FP8 MQA machinery
        which is decode-shaped.

        The indexer is MQA (a single key head shared by all ``n_heads`` query
        heads), but weighted-ReLU must preserve the head dimension until after
        activation::

            score[i, j] = sum_h w[i, h] * relu(dot(q[i, h], k[j]))

        Query and head chunking avoid materializing ``[L, H, L]``.

        Args:
            q_lora:        (num_tokens, q_lora_rank) — main-attn Q LoRA.
            hidden_states: (num_tokens, hidden_size)
            positions:     (num_tokens,) — absolute positions for RoPE.
            cu_seqlens:    (num_seqs + 1,) int — cumulative query lengths.

        Returns:
            indices: (num_tokens, index_topk) int32 — absolute key positions
                     into the (concatenated) KV, ``-1`` for invalid/padding.
        """
        query, key = self._compute_q_k(q_lora, hidden_states, positions)
        # query: (N, n_heads, head_dim), key: (N, head_dim)
        weights = self.weights_proj(hidden_states).float() * (self.n_heads**-0.5)
        key_f = key.float()

        num_tokens = query.shape[0]
        indices = torch.full(
            (num_tokens, self.index_topk),
            -1,
            dtype=torch.int32,
            device=query.device,
        )
        num_seqs = cu_seqlens.shape[0] - 1
        neg_inf = float("-inf")
        for s in range(num_seqs):
            start = int(cu_seqlens[s].item())
            end = int(cu_seqlens[s + 1].item())
            seq_len = end - start
            if seq_len <= 0:
                continue
            seq_key = key_f[start:end]  # (L, D)
            k = min(self.index_topk, seq_len)
            for a in range(0, seq_len, query_chunk):
                b = min(a + query_chunk, seq_len)
                seq_q = query[start + a : start + b]
                seq_weights = weights[start + a : start + b]
                score = _weighted_relu_mqa_scores(seq_q, seq_weights, seq_key)
                # Causal mask: query at local row (a + r) attends keys j <= a + r.
                rows = torch.arange(a, b, device=query.device).unsqueeze(1)
                cols = torch.arange(seq_len, device=query.device).unsqueeze(0)
                score.masked_fill_(cols > rows, neg_inf)
                top_val, top_idx = score.topk(k, dim=-1)
                top_idx = (top_idx + start).to(torch.int32)
                top_idx[top_val == neg_inf] = -1
                indices[start + a : start + b, :k] = top_idx
        return indices

    def _gather_cached_prefix_keys(
        self,
        block_table: torch.Tensor,
        cached_lens: torch.Tensor,
        cu_cached: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Gather and dequantize cached indexer keys for chunked prefill.

        This readable reference path materializes only previously cached prefix
        keys. A production implementation will replace it with paged-FP8
        score+TopK without a full-prefix dequantized tensor.
        """
        assert self.indexer_cache is not None, "IndexerCache not initialized"

        if block_table.shape[0] != cached_lens.numel():
            raise RuntimeError(
                "block_table/cached_lens batch mismatch: "
                f"{block_table.shape[0]} vs {cached_lens.numel()}"
            )

        total_cached = int(cu_cached[-1].item())
        if total_cached == 0:
            return torch.empty(
                0, self.head_dim, dtype=dtype, device=block_table.device
            )

        from dlengine.kernel.triton.generic.paged_gather import (
            build_paged_gather_indices,
        )

        cache = self.indexer_cache
        page_size = cache.page_size
        head_dim = cache.head_dim
        if head_dim != self.head_dim:
            raise RuntimeError(
                f"Indexer cache head_dim mismatch: cache={head_dim}, "
                f"indexer={self.head_dim}"
            )
        if head_dim != INDEXER_QUANT_BLOCK_SIZE:
            raise NotImplementedError(
                "Reference cache-aware prefill TopK currently assumes "
                f"indexer head_dim={INDEXER_QUANT_BLOCK_SIZE}, got {head_dim}"
            )

        physical_slots = build_paged_gather_indices(
            block_table,
            cu_cached,
            page_size,
            total_k=total_cached,
        )
        page_idx = physical_slots // page_size
        offset_in_page = physical_slots % page_size

        row_stride = page_size * cache.bytes_per_token
        flat_base = page_idx.long() * row_stride
        buf_flat = cache.get_buffer(self.layer_id).view(-1)

        byte_range = torch.arange(head_dim, device=block_table.device)
        fp8_byte_offset = offset_in_page.long() * head_dim
        fp8_indices = (
            flat_base.unsqueeze(1) + fp8_byte_offset.unsqueeze(1) + byte_range
        )
        key_fp8_bytes = buf_flat[fp8_indices.reshape(-1)].view(total_cached, head_dim)
        key_fp8 = key_fp8_bytes.contiguous().view(torch.float8_e4m3fn)

        scale_range = torch.arange(4, device=block_table.device)
        scale_byte_offset = page_size * head_dim + offset_in_page.long() * 4
        scale_indices = (
            flat_base.unsqueeze(1) + scale_byte_offset.unsqueeze(1) + scale_range
        )
        scale_bytes = buf_flat[scale_indices.reshape(-1)].view(total_cached, 4)
        scale = scale_bytes.contiguous().view(torch.float32).view(total_cached, 1)

        return (key_fp8.float() * scale).to(dtype)

    def compute_prefill_topk_cache_aware(
        self,
        q_lora: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        block_table: torch.Tensor,
        query_chunk: int = 256,
    ) -> torch.Tensor:
        """Compute exact TopK over cached prefix plus causal fresh keys.

        For a query at local row ``r`` with cached length ``C``, the visible
        key set is ``[0, C)`` plus fresh keys ``[C, C + r]``. Prefix and fresh
        candidates are selected separately and then merged; keeping up to K
        from each half is exactly equivalent to selecting K from their union.

        Returned indices address the concatenated ragged K layout described by
        ``cu_seqlens_k`` and use ``-1`` for invalid/padded candidates.
        """
        assert self.indexer_cache is not None, "IndexerCache not initialized"
        if cu_seqlens_q.shape != cu_seqlens_k.shape:
            raise RuntimeError(
                "cu_seqlens_q/cu_seqlens_k shape mismatch: "
                f"{cu_seqlens_q.shape} vs {cu_seqlens_k.shape}"
            )

        query, fresh_key = self._compute_q_k(q_lora, hidden_states, positions)
        weights = self.weights_proj(hidden_states).float() * (self.n_heads**-0.5)
        fresh_key_f = fresh_key.float()

        q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).long()
        k_lens = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).long()
        cached_lens = k_lens - q_lens
        if torch.any(cached_lens < 0):
            raise RuntimeError(
                "Invalid chunked-prefill lengths: cu_seqlens_k must be >= "
                "cu_seqlens_q for every sequence"
            )

        cu_cached = torch.zeros_like(cu_seqlens_k)
        cu_cached[1:] = cached_lens.cumsum(0).to(cu_cached.dtype)
        cached_key_f = self._gather_cached_prefix_keys(
            block_table,
            cached_lens,
            cu_cached,
            dtype=torch.float32,
        )

        num_tokens = query.shape[0]
        indices = torch.full(
            (num_tokens, self.index_topk),
            -1,
            dtype=torch.int32,
            device=query.device,
        )

        num_seqs = cu_seqlens_q.shape[0] - 1
        neg_inf = float("-inf")
        for seq_id in range(num_seqs):
            q_start = int(cu_seqlens_q[seq_id].item())
            q_end = int(cu_seqlens_q[seq_id + 1].item())
            k_start = int(cu_seqlens_k[seq_id].item())
            seq_q_len = q_end - q_start
            if seq_q_len <= 0:
                continue

            cached_start = int(cu_cached[seq_id].item())
            cached_end = int(cu_cached[seq_id + 1].item())
            cached_len = cached_end - cached_start

            seq_prefix_key = cached_key_f[cached_start:cached_end]
            seq_fresh_key = fresh_key_f[q_start:q_end]
            prefix_k = min(self.index_topk, cached_len)
            fresh_k = min(self.index_topk, seq_q_len)

            for a in range(0, seq_q_len, query_chunk):
                b = min(a + query_chunk, seq_q_len)
                seq_q = query[q_start + a : q_start + b]
                seq_weights = weights[q_start + a : q_start + b]

                candidate_values: list[torch.Tensor] = []
                candidate_indices: list[torch.Tensor] = []
                if prefix_k > 0:
                    prefix_scores = _weighted_relu_mqa_scores(
                        seq_q, seq_weights, seq_prefix_key
                    )
                    prefix_values, prefix_indices = prefix_scores.topk(
                        prefix_k, dim=-1
                    )
                    candidate_values.append(prefix_values)
                    candidate_indices.append(prefix_indices.to(torch.int64))

                if fresh_k > 0:
                    fresh_scores = _weighted_relu_mqa_scores(
                        seq_q, seq_weights, seq_fresh_key
                    )
                    rows = torch.arange(a, b, device=query.device).unsqueeze(1)
                    cols = torch.arange(seq_q_len, device=query.device).unsqueeze(0)
                    fresh_scores.masked_fill_(cols > rows, neg_inf)

                    fresh_values, fresh_indices = fresh_scores.topk(
                        fresh_k, dim=-1
                    )
                    fresh_indices = fresh_indices.to(torch.int64) + cached_len
                    fresh_indices = torch.where(
                        fresh_values == neg_inf,
                        torch.full_like(fresh_indices, -1),
                        fresh_indices,
                    )
                    candidate_values.append(fresh_values)
                    candidate_indices.append(fresh_indices)

                if not candidate_values:
                    continue

                merged_values = torch.cat(candidate_values, dim=-1)
                merged_indices = torch.cat(candidate_indices, dim=-1)
                final_k = min(self.index_topk, merged_values.shape[-1])
                final_values, final_positions = merged_values.topk(final_k, dim=-1)
                final_indices = torch.gather(
                    merged_indices, dim=-1, index=final_positions
                )
                final_indices = torch.where(
                    final_values == neg_inf,
                    torch.full_like(final_indices, -1),
                    final_indices + k_start,
                )
                indices[q_start + a : q_start + b, :final_k] = final_indices.to(
                    torch.int32
                )

        return indices

    def _compute_gate_weights(
        self,
        hidden_states: torch.Tensor,
        q_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gating weights for MQA logits.

        Args:
            hidden_states: (num_tokens, hidden_size)
            q_scale: (num_tokens, n_heads, 1) FP32 — FP8 quantization scale

        Returns:
            weights: (num_tokens, n_heads) FP32
        """
        # weights_proj: (N, hidden_size) -> (N, n_heads) BF16, then to FP32
        weights = self.weights_proj(hidden_states).float()
        # Scale: weights * (1/sqrt(n_heads)) * q_scale * softmax_scale
        weights = weights * (self.n_heads**-0.5)
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
        # Squeeze back: (N, n_heads, 1) -> (N, n_heads)
        weights = weights.squeeze(-1)
        return weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        context_lens: torch.Tensor,
        block_tables: torch.Tensor,
        slot_mapping: torch.Tensor,
        translate_topk: bool = False,
        topk_page_size: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run indexer to produce topk block indices for sparse attention.

        Args:
            hidden_states: (num_tokens, hidden_size)
            q_lora: (num_tokens, q_lora_rank) — from main attention's q_a_proj + layernorm
            positions: (num_tokens,) — position indices
            context_lens: (batch_size,) int32 — sequence lengths
            block_tables: (batch_size, max_num_blocks) int32 — page table
            slot_mapping: (num_tokens,) int — flat slot indices for cache write

        Returns:
            topk_indices: (num_tokens, index_topk) int32 selected logical token
                indices. When translate_topk is True, returns
                (logical_indices, physical_cache_slots).
        """
        assert self.indexer_cache is not None, "IndexerCache not initialized"
        num_tokens = hidden_states.shape[0]
        batch_size = context_lens.shape[0]

        # Step 1-4: Compute query and key (with RoPE + Hadamard)
        use_fused_query = fused_kernels_enabled()
        use_fused_key = use_fused_query and hidden_states.is_cuda
        query, key = self._compute_q_k(
            q_lora,
            hidden_states,
            positions,
            defer_query_transform=use_fused_query,
            defer_key_transform=use_fused_key,
        )

        # Step 5: RoPE + Hadamard + FP8 quantize query and scale gate weights.
        if use_fused_query:
            gate_weight = self.weights_proj(hidden_states)
            q_fp8, weights = indexer_q_rope_hadamard_quant(
                query,
                gate_weight,
                (self.n_heads**-0.5) * self.softmax_scale,
                self.rotary_emb.cos_sin_cache,
                positions,
            )
            weights = weights.squeeze(-1)
        else:
            q_flat = query.reshape(num_tokens * self.n_heads, self.head_dim)
            q_fp8, q_scale = quant_fp8(
                q_flat.contiguous(),
                self.head_dim,
                round_ue8m0=True,
                min_absmax=1e-4,
            )
            q_fp8 = q_fp8.view(num_tokens, self.n_heads, self.head_dim)
            q_scale_for_gate = q_scale.view(num_tokens, self.n_heads, 1)
            weights = self._compute_gate_weights(hidden_states, q_scale_for_gate)

        # Step 6: transform and store K. The fused path consumes the raw WK
        # output and performs LayerNorm, RoPE, Hadamard, quantization and the
        # paged-cache write in one launch.
        if use_fused_key:
            indexer_k_transform_store_fp8(
                key,
                self.k_norm.weight,
                self.k_norm.bias,
                self.k_norm.eps,
                positions,
                self.rotary_emb.cos_sin_cache,
                self.indexer_cache.get_buffer(self.layer_id),
                slot_mapping,
                self.indexer_cache.page_size,
            )
        else:
            self.indexer_cache.store_key_fp8(self.layer_id, key, slot_mapping)

        # Step 8: Compute FP8 paged MQA logits
        # q_fp8 needs shape (batch, next_n, n_heads, head_dim) for deep_gemm
        # For decode: next_n = num_tokens_per_seq (usually 1)
        ntps = num_tokens // batch_size
        q_fp8_4d = q_fp8.view(batch_size, ntps, self.n_heads, self.head_dim)

        # Get indexer KV cache buffer and reshape for deep_gemm
        kv_cache_buf = self.indexer_cache.get_buffer(self.layer_id)
        page_size = self.indexer_cache.page_size
        bpt = self.indexer_cache.bytes_per_token
        # Reshape: (num_pages, page_size * bpt) -> (num_pages, page_size, 1, bpt)
        kv_cache = kv_cache_buf.view(kv_cache_buf.shape[0], page_size, 1, bpt)

        # weights: (N, n_heads) -> deep_gemm expects (batch * ntps, n_heads) which is (N, n_heads)
        # No reshape needed — weights is already (num_tokens, n_heads)

        # Use block_table-derived max_context_len for CUDA-graph compatibility
        # (block_tables.shape[-1] * page_size is constant per captured graph).
        max_context_len = block_tables.shape[-1] * page_size
        context_lens_i32 = context_lens.to(torch.int32)
        context_lens_for_gemm = _expand_decode_context_lens(
            context_lens_i32, ntps
        )

        # All layers share this schedule. The model builds it once per forward;
        # retain the fallback for standalone Indexer calls and tests.
        from dlengine.context_v2.batch import get_batch_context

        schedule_meta = get_batch_context().indexer_schedule_meta
        if schedule_meta is None:
            schedule_meta = self.build_schedule_metadata(context_lens_for_gemm)

        # Compute logits: (batch * ntps, max_context_len) FP32
        logits = deep_gemm.fp8_paged_mqa_logits(
            q_fp8_4d,
            kv_cache,
            weights,
            context_lens_for_gemm,
            block_tables.to(torch.int32),
            schedule_meta,
            max_context_len,
            # DeepGEMM does not support clean_logits with the 2D context_lens
            # required by the paged MQA decode path.
            clean_logits=False,
        )

        # Step 9: TopK selection. On Hopper, fuse selection, invalid-index
        # handling, and logical-to-physical page translation into one kernel.
        if translate_topk:
            if topk_page_size is None:
                raise ValueError("topk_page_size is required when translate_topk=True")
            from dlengine.kernel.jit.sgl.deepseek_v4 import topk_transform

            seq_lens = context_lens_for_gemm.reshape(-1)
            page_tables = (
                block_tables
                if ntps == 1
                else block_tables.repeat_interleave(ntps, dim=0)
            )
            physical_indices = torch.empty(
                (batch_size * ntps, self.index_topk),
                dtype=torch.int32,
                device=logits.device,
            )
            logical_indices = torch.empty_like(physical_indices)
            topk_transform(
                logits,
                seq_lens,
                page_tables,
                physical_indices,
                topk_page_size,
                self.index_topk,
                logical_indices,
            )
            return logical_indices, physical_indices

        # Portable fallback: explicitly clean the logits because DeepGEMM cannot
        # enable clean_logits for 2D context_lens.
        ctx_expanded = context_lens_for_gemm.reshape(-1, 1)
        logit_positions = torch.arange(max_context_len, device=logits.device).unsqueeze(
            0
        )
        logits = logits.masked_fill(logit_positions >= ctx_expanded, float("-inf"))
        logits = torch.where(
            torch.isfinite(logits) & (logits.abs() < 1e30),
            logits,
            float("-inf"),
        )
        actual_topk = min(self.index_topk, max_context_len)
        _, topk_indices = torch.topk(logits, k=actual_topk, dim=-1)
        topk_indices = topk_indices.to(torch.int32)

        # Mark out-of-range indices as -1 (they had -inf logits but topk still returns them)
        topk_indices = torch.where(topk_indices < ctx_expanded, topk_indices, -1)

        if actual_topk < self.index_topk:
            topk_indices = torch.nn.functional.pad(
                topk_indices, (0, self.index_topk - actual_topk), value=-1
            )

        return topk_indices
