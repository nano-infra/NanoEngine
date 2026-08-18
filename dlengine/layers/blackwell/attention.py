"""FA4 prefill and TRT-LLM paged decode for NVIDIA Blackwell GPUs."""

import os

import torch

from dlengine.context.batch import get_batch_context
from dlengine.kernel.triton.generic.kv_store import store_kvcache
from dlengine.layers.hopper.attention import _gather_kv_cached_concat, HopperAttention
from dlengine.logging import get_logger

logger = get_logger()
_DEBUG_SYNC = os.environ.get("DLENGINE_BLACKWELL_DEBUG_SYNC", "0") == "1"
_TRTLLM_WORKSPACE_BYTES = 512 * 1024 * 1024
_trtllm_workspace: torch.Tensor | None = None


def _debug_sync(stage: str, **fields) -> None:
    if not _DEBUG_SYNC:
        return
    logger.warning("[blackwell-debug] %s begin %s", stage, fields)
    torch.cuda.synchronize()
    logger.warning("[blackwell-debug] %s done", stage)


try:
    from flash_attn.cute import flash_attn_varlen_func as _fa4_varlen_func
except ImportError as error:
    _fa4_varlen_func = None
    _FA4_IMPORT_ERROR: ImportError | None = error
else:
    _FA4_IMPORT_ERROR = None

try:
    from flashinfer.decode import (
        trtllm_batch_decode_with_kv_cache as _trtllm_decode_func,
    )
except ImportError as error:
    _trtllm_decode_func = None
    _TRTLLM_IMPORT_ERROR: ImportError | None = error
else:
    _TRTLLM_IMPORT_ERROR = None

try:
    from flashinfer.mla import (
        trtllm_batch_decode_with_kv_cache_mla as _trtllm_mla_decode_func,
    )
except ImportError as error:
    _trtllm_mla_decode_func = None
    _TRTLLM_MLA_IMPORT_ERROR: ImportError | None = error
else:
    _TRTLLM_MLA_IMPORT_ERROR = None


def _require_blackwell_attention_kernels() -> None:
    if _fa4_varlen_func is None:
        message = (
            "Blackwell prefill requires FlashAttention-4 CuTeDSL. Install a "
            'CUDA 13 build with `pip install "flash-attn-4[cu13]"`. No naive '
            "or SDPA fallback is available."
        )
        if _FA4_IMPORT_ERROR is not None:
            raise RuntimeError(message) from _FA4_IMPORT_ERROR
        raise RuntimeError(message)
    if _trtllm_decode_func is None:
        message = (
            "Blackwell paged decode requires FlashInfer's TRT-LLM MHA kernel. "
            "Install a CUDA 13 FlashInfer build with TRTLLM-GEN artifacts. No "
            "naive or SDPA fallback is available."
        )
        if _TRTLLM_IMPORT_ERROR is not None:
            raise RuntimeError(message) from _TRTLLM_IMPORT_ERROR
        raise RuntimeError(message)


def _get_trtllm_workspace(device: torch.device) -> torch.Tensor:
    global _trtllm_workspace
    if _trtllm_workspace is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "TRT-LLM MHA workspace must be initialized before CUDA Graph capture."
            )
        _trtllm_workspace = torch.zeros(
            _TRTLLM_WORKSPACE_BYTES, dtype=torch.uint8, device=device
        )
    elif _trtllm_workspace.device != device:
        raise RuntimeError(
            "TRT-LLM MHA workspace was initialized on a different CUDA device."
        )
    return _trtllm_workspace


class BlackwellAttentionImpl:
    """FA4 prefill plus TRTLLM-GEN paged decode."""

    def __init__(self, num_heads, head_dim, scale, num_kv_heads, sliding_window=None):
        _require_blackwell_attention_kernels()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window

    @staticmethod
    def _unwrap(output):
        return output[0] if isinstance(output, tuple) else output

    def _decode(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        sequence_lengths: torch.Tensor,
        tokens_per_seq: int,
    ) -> torch.Tensor:
        page_size = k_cache.shape[1]
        if page_size not in (16, 32, 64):
            raise RuntimeError(
                "Blackwell TRT-LLM MHA decode requires KV page size 16, 32, or "
                f"64, got {page_size}. No fallback attention is available."
            )
        q_input = q.reshape(-1, self.num_heads, self.head_dim)
        # NanoDeploy NHD cache -> TRTLLM-GEN HND cache. The permute is a view;
        # TRTLLM-GEN permits arbitrary page/head strides and requires only D
        # to be contiguous.
        k_hnd = k_cache.permute(0, 2, 1, 3)
        v_hnd = v_cache.permute(0, 2, 1, 3)
        output = torch.empty_like(q_input)
        _debug_sync("trtllm-decode-input", q=tuple(q_input.shape))
        output = _trtllm_decode_func(
            query=q_input,
            kv_cache=(k_hnd, v_hnd),
            workspace_buffer=_get_trtllm_workspace(q.device),
            block_tables=page_table,
            seq_lens=sequence_lengths,
            max_seq_len=page_table.shape[1] * page_size,
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            window_left=(
                int(self.sliding_window) if self.sliding_window is not None else -1
            ),
            out=output,
            out_dtype=q.dtype,
            backend="auto",
            q_len_per_req=tokens_per_seq,
        )
        _debug_sync("trtllm-decode-output")
        return output.reshape_as(q)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        sparse_indices: torch.Tensor | None = None,
        write_kv_cache: bool = True,
    ) -> torch.Tensor:
        del sparse_indices
        context = get_batch_context()
        # FA4 CuTe assumes dense token-major Q/K/V. Fused QKV projections can
        # produce split views whose token stride still spans the full fused
        # row, which otherwise makes FA4 read subsequent tokens incorrectly.
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        _debug_sync(
            "forward-entry",
            prefill=context.is_prefill,
            q=tuple(q.shape),
            cache=tuple(k_cache.shape),
        )
        if write_kv_cache and k_cache.numel() and not context.is_dummy:
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            _debug_sync("store-kvcache")

        window_size = (
            (int(self.sliding_window) - 1, 0)
            if self.sliding_window is not None
            else (None, None)
        )
        common = {"softmax_scale": self.scale, "window_size": window_size}

        if context.block_tables is None:
            if not context.is_prefill:
                raise RuntimeError(
                    "Blackwell TRT-LLM decode requires a paged KV cache. No "
                    "fallback attention implementation is available."
                )
            return self._unwrap(
                _fa4_varlen_func(
                    q,
                    k,
                    v,
                    cu_seqlens_q=context.cu_seqlens_q,
                    cu_seqlens_k=context.cu_seqlens_k,
                    max_seqlen_q=context.max_seqlen_q,
                    max_seqlen_k=context.max_seqlen_k,
                    causal=True,
                    **common,
                )
            )

        if context.is_prefill:
            batch_size = context.cu_seqlens_q.shape[0] - 1
            page_table = context.block_tables[0, :batch_size]
            if self.head_dim == 256:
                k_input, v_input = _gather_kv_cached_concat(
                    k_cache,
                    v_cache,
                    k,
                    v,
                    page_table,
                    context.cu_seqlens_q,
                    context.cu_seqlens_k,
                    k_cache.shape[1],
                )
                output = _fa4_varlen_func(
                    q,
                    k_input,
                    v_input,
                    cu_seqlens_q=context.cu_seqlens_q,
                    cu_seqlens_k=context.cu_seqlens_k,
                    max_seqlen_q=context.max_seqlen_q,
                    max_seqlen_k=page_table.shape[1] * k_cache.shape[1],
                    causal=True,
                    **common,
                )
            else:
                sequence_lengths = context.cu_seqlens_k[1:] - context.cu_seqlens_k[:-1]
                output = _fa4_varlen_func(
                    q,
                    k_cache,
                    v_cache,
                    cu_seqlens_q=context.cu_seqlens_q,
                    seqused_k=sequence_lengths,
                    page_table=page_table,
                    max_seqlen_q=context.max_seqlen_q,
                    max_seqlen_k=page_table.shape[1] * k_cache.shape[1],
                    causal=True,
                    **common,
                )
            _debug_sync("fa4-prefill-output")
            return self._unwrap(output).reshape_as(q)

        tokens_per_seq = context.num_tokens_per_seq
        batch_size = q.shape[0] // tokens_per_seq
        page_table = context.block_tables[0, :batch_size]
        sequence_lengths = context.context_lens[0, :batch_size]
        if not page_table.is_contiguous():
            raise RuntimeError("Blackwell TRT-LLM requires a contiguous page table.")
        if (
            sequence_lengths.dtype != torch.int32
            or not sequence_lengths.is_contiguous()
        ):
            raise RuntimeError(
                "Blackwell TRT-LLM requires contiguous int32 KV lengths."
            )
        return self._decode(
            q,
            k_cache,
            v_cache,
            page_table,
            sequence_lengths,
            tokens_per_seq,
        )


class BlackwellMLAAttention(HopperAttention):
    """FlashInfer TRTLLM-GEN decode for compressed MLA caches on Blackwell."""

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        v_head_dim,
        attention_type: str = "MLA",
        nsa_index_topk: int = 0,
        mla_qk_nope_head_dim: int | None = None,
        **kwargs,
    ) -> None:
        del kwargs
        if attention_type != "MLA":
            raise ValueError(
                f"BlackwellMLAAttention requires MLA, got {attention_type}"
            )
        if _trtllm_mla_decode_func is None:
            message = (
                "Blackwell MLA decode requires FlashInfer's TRTLLM-GEN MLA "
                "kernel. No FlashMLA, torch, or naive fallback is available."
            )
            if _TRTLLM_MLA_IMPORT_ERROR is not None:
                raise RuntimeError(message) from _TRTLLM_MLA_IMPORT_ERROR
            raise RuntimeError(message)
        if num_kv_heads != 1:
            raise ValueError(f"MLA requires one compressed KV head, got {num_kv_heads}")
        if head_dim <= v_head_dim:
            raise ValueError(
                f"Invalid MLA dimensions: cache head_dim={head_dim}, "
                f"kv_lora_rank={v_head_dim}"
            )
        # Do not call HopperAttention.__init__: that constructs the legacy
        # FlashMLA implementation. Cache tensors are injected by ModelRunner.
        torch.nn.Module.__init__(self)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.v_head_dim = v_head_dim
        self.qk_rope_head_dim = head_dim - v_head_dim
        if mla_qk_nope_head_dim is None:
            raise ValueError("Blackwell MLA requires mla_qk_nope_head_dim")
        self.qk_nope_head_dim = mla_qk_nope_head_dim
        self.nsa_index_topk = nsa_index_topk
        self.k_cache = self.v_cache = torch.tensor([])
        self.hisparse_k_cache = self.hisparse_v_cache = torch.tensor([])

    def forward(self, q, k, v, sparse_indices=None, write_kv_cache=True):
        del v, sparse_indices
        context = get_batch_context()
        if context.is_prefill:
            raise RuntimeError(
                "BlackwellMLAAttention is decode-only; MLA prefill must use "
                "the non-absorbed FA4 path in DeepseekV2Attention."
            )
        if write_kv_cache and self.k_cache.numel() and not context.is_dummy:
            from dlengine.kernel.triton.generic.kv_store import store_kcache

            store_kcache(k, self.k_cache, context.slot_mapping)

        ntps = context.num_tokens_per_seq
        batch_size = q.shape[0] // ntps
        query = q.reshape(batch_size, ntps, self.num_heads, self.head_dim)
        block_tables = context.block_tables[0, :batch_size]
        # Attention-DP ranks without a local request execute a synthetic one-token
        # batch so that the shared EP collectives stay ordered. The generic dummy
        # metadata intentionally carries an empty page table, but TRTLLM-GEN MLA
        # requires its page-table batch dimension to match the query batch even
        # when no KV write is performed. Page zero is allocated and safe to read;
        # the dummy result is discarded by the scheduler.
        if context.is_dummy and block_tables.numel() == 0:
            block_tables = torch.zeros(
                (batch_size, 1), dtype=torch.int32, device=query.device
            )
        seq_lens = context.context_lens[0, :batch_size]
        kv_cache = self.k_cache
        if kv_cache.ndim == 4 and kv_cache.shape[2] == 1:
            kv_cache = kv_cache.squeeze(2)
        if not block_tables.is_contiguous():
            block_tables = block_tables.contiguous()
        # TRTLLM-GEN groups 128 tokens when constructing its paged schedule.
        # Therefore the page-table width must be a multiple of 128/page_size.
        # Scheduler metadata is intentionally trimmed to the active pages, so
        # pad only its unused tail; seq_lens remains the authoritative bound.
        page_group = 128 // kv_cache.shape[1]
        remainder = block_tables.shape[1] % page_group
        if remainder:
            padding = block_tables.new_zeros(
                (block_tables.shape[0], page_group - remainder)
            )
            block_tables = torch.cat((block_tables, padding), dim=1)
        if seq_lens.dtype != torch.int32 or not seq_lens.is_contiguous():
            seq_lens = seq_lens.to(dtype=torch.int32).contiguous()
        out = torch.empty(
            (*query.shape[:-1], self.v_head_dim),
            dtype=torch.bfloat16,
            device=query.device,
        )
        result = _trtllm_mla_decode_func(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=_get_trtllm_workspace(query.device),
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.v_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=block_tables.shape[1] * kv_cache.shape[1],
            sparse_mla_top_k=0,
            out=out,
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            backend="auto",
            is_var_seq=True,
            uses_shared_paged_kv_idx=True,
        )
        return result.reshape(-1, self.num_heads, self.v_head_dim)


class BlackwellAttention(HopperAttention):
    """Hopper cache plumbing with FA4 prefill and TRT-LLM decode."""

    def __init__(self, *args, attention_type: str = "MLA", **kwargs) -> None:
        if attention_type != "GQA":
            raise RuntimeError(
                f"Blackwell backend does not yet support {attention_type} attention; "
                "no Hopper, naive, or SDPA fallback will be selected."
            )
        super().__init__(*args, attention_type=attention_type, **kwargs)
        self.impl = BlackwellAttentionImpl(
            self.num_heads,
            self.head_dim,
            self.scale,
            self.num_kv_heads,
            sliding_window=kwargs.get("sliding_window"),
        )

    def forward(self, q, k, v, sparse_indices=None, write_kv_cache=True):
        return self.impl.forward(
            q,
            k,
            v,
            self.k_cache,
            self.v_cache,
            sparse_indices=sparse_indices,
            write_kv_cache=write_kv_cache,
        )
