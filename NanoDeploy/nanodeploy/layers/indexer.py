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

import deep_gemm
import torch
import torch.nn as nn
from fast_hadamard_transform import hadamard_transform

from nanodeploy.backends import get_backend
from nanodeploy.backends.base_backend import ReplicatedLinearBase
from nanodeploy.layers.rotary_embedding import get_rope
from nanodeploy.logging import get_logger

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
        quant_block_size = INDEXER_QUANT_BLOCK_SIZE
        self.bytes_per_token = head_dim + head_dim // quant_block_size * 4
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

        # Quantize: per-token FP8 with UE8M0 scale (graph-safe)
        key_fp8, key_scale = _per_token_cast_to_fp8_ue8m0(key_bf16)

        # Clamp invalid slots (-1) to 0 for graph-safe scatter (writes harmlessly to slot 0)
        safe_slots = torch.where(
            slot_mapping >= 0, slot_mapping, torch.zeros_like(slot_mapping)
        )

        # Compute page/offset for each token
        page_idx = safe_slots // page_size  # (N,)
        offset_in_page = safe_slots % page_size  # (N,)
        fp8_byte_offset = offset_in_page * bpt  # (N,)
        scale_byte_offset = fp8_byte_offset + head_dim  # (N,)

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

    def _compute_q_k(
        self,
        q_lora: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
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
        query = _hadamard_rotate(query)
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
    ) -> torch.Tensor:
        """Run indexer to produce topk block indices for sparse attention.

        Args:
            hidden_states: (num_tokens, hidden_size)
            q_lora: (num_tokens, q_lora_rank) — from main attention's q_a_proj + layernorm
            positions: (num_tokens,) — position indices
            context_lens: (batch_size,) int32 — sequence lengths
            block_tables: (batch_size, max_num_blocks) int32 — page table
            slot_mapping: (num_tokens,) int — flat slot indices for cache write

        Returns:
            topk_indices: (num_tokens, index_topk) int32 — selected token indices
        """
        assert self.indexer_cache is not None, "IndexerCache not initialized"
        num_tokens = hidden_states.shape[0]
        batch_size = context_lens.shape[0]

        # Step 1-4: Compute query and key (with RoPE + Hadamard)
        query, key = self._compute_q_k(q_lora, hidden_states, positions)

        # Step 5: Quantize query to FP8
        # query: (N, n_heads, head_dim=128) -> reshape for per-token quantization
        # deep_gemm expects (N, D) for per_token_cast_to_fp8
        q_flat = query.reshape(num_tokens * self.n_heads, self.head_dim)
        q_fp8, q_scale = _per_token_cast_to_fp8_ue8m0(q_flat)
        # q_fp8: (N*H, 128) FP8, q_scale: (N*H, 1) FP32
        q_fp8 = q_fp8.view(num_tokens, self.n_heads, self.head_dim)
        q_scale_for_gate = q_scale.view(num_tokens, self.n_heads, 1)

        # Step 6: Store key to indexer cache
        self.indexer_cache.store_key_fp8(self.layer_id, key, slot_mapping)

        # Step 7: Compute gate weights
        weights = self._compute_gate_weights(hidden_states, q_scale_for_gate)

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

        # Schedule metadata for deep_gemm
        schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(
            context_lens_i32, page_size, self.sm_count
        )

        # Compute logits: (batch * ntps, max_context_len) FP32
        logits = deep_gemm.fp8_paged_mqa_logits(
            q_fp8_4d,
            kv_cache,
            weights,
            context_lens_i32,
            block_tables.to(torch.int32),
            schedule_meta,
            max_context_len,
            clean_logits=False,
        )

        # Step 9: TopK selection
        # logits: (batch * ntps, max_context_len) — mask invalid positions
        total_q = batch_size * ntps
        ctx_expanded = context_lens_i32.repeat_interleave(ntps).unsqueeze(
            1
        )  # (total_q, 1)

        # Mask invalid positions (beyond actual sequence length)
        arange = torch.arange(max_context_len, device=logits.device).unsqueeze(0)
        logits = logits.masked_fill(arange >= ctx_expanded, float("-inf"))

        # TopK: select top index_topk token positions
        # actual_topk is constant (block_table capacity >= index_topk in practice)
        actual_topk = min(self.index_topk, max_context_len)
        _, topk_indices = torch.topk(logits, k=actual_topk, dim=-1)
        topk_indices = topk_indices.to(torch.int32)

        # Mark out-of-range indices as -1 (they had -inf logits but topk still returns them)
        topk_indices = torch.where(topk_indices < ctx_expanded, topk_indices, -1)

        # Pad to index_topk if needed (with -1 for invalid)
        if actual_topk < self.index_topk:
            padding = torch.full(
                (total_q, self.index_topk - actual_topk),
                -1,
                dtype=torch.int32,
                device=topk_indices.device,
            )
            topk_indices = torch.cat([topk_indices, padding], dim=-1)

        return topk_indices
