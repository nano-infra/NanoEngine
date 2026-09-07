"""TRTLLM-GEN (Blackwell) dense MLA decode backend.

Wraps FlashInfer's ``trtllm_batch_decode_with_kv_cache_mla`` for compressed-KV
MLA decode on SM100+, with both the dense paged path and the FP8 sparse path
(shared with the DSA family). MLA prefill uses the non-absorbed FA4 path in
``DeepseekV2Attention.forward`` and must not reach this backend.
"""

import torch

from dlengine.runtime.context.batch import get_batch_context
from dlengine.runtime.layers.backends.mla.base import MlaAttentionBase

_TRTLLM_WORKSPACE_BYTES = 512 * 1024 * 1024
_trtllm_workspace: torch.Tensor | None = None

try:
    from flashinfer.mla import (
        trtllm_batch_decode_with_kv_cache_mla as _trtllm_mla_decode_func,
    )
except ImportError as error:
    _trtllm_mla_decode_func = None
    _TRTLLM_MLA_IMPORT_ERROR: ImportError | None = error
else:
    _TRTLLM_MLA_IMPORT_ERROR = None


def _get_trtllm_workspace(device: torch.device) -> torch.Tensor:
    global _trtllm_workspace
    if _trtllm_workspace is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "TRT-LLM MLA workspace must be initialized before CUDA Graph capture."
            )
        _trtllm_workspace = torch.zeros(
            _TRTLLM_WORKSPACE_BYTES, dtype=torch.uint8, device=device
        )
    elif _trtllm_workspace.device != device:
        raise RuntimeError(
            "TRT-LLM MLA workspace was initialized on a different CUDA device."
        )
    return _trtllm_workspace


class TrtllmMlaAttention(MlaAttentionBase):
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
        mla_kv_lora_rank: int | None = None,
        **kwargs,
    ) -> None:
        del kwargs
        if _trtllm_mla_decode_func is None:
            message = (
                "Blackwell MLA decode requires FlashInfer's TRTLLM-GEN MLA "
                "kernel. No FlashMLA, torch, or naive fallback is available."
            )
            if _TRTLLM_MLA_IMPORT_ERROR is not None:
                raise RuntimeError(message) from _TRTLLM_MLA_IMPORT_ERROR
            raise RuntimeError(message)
        kv_lora_rank = v_head_dim if mla_kv_lora_rank is None else mla_kv_lora_rank
        if head_dim < kv_lora_rank:
            raise ValueError(
                f"Invalid MLA dimensions: cache head_dim={head_dim}, "
                f"kv_lora_rank={kv_lora_rank}"
            )
        if mla_qk_nope_head_dim is None:
            raise ValueError("Blackwell MLA requires mla_qk_nope_head_dim")
        super().__init__(
            num_heads,
            head_dim,
            scale,
            num_kv_heads,
            v_head_dim,
            attention_type=attention_type,
            nsa_index_topk=nsa_index_topk,
        )
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = head_dim - kv_lora_rank
        self.qk_nope_head_dim = mla_qk_nope_head_dim

    def forward(self, q, k, v, sparse_indices=None, write_kv_cache=True):
        del v
        context = get_batch_context()
        if context.is_prefill:
            raise RuntimeError(
                "TrtllmMlaAttention is decode-only; MLA prefill must use "
                "the non-absorbed FA4 path in DeepseekV2Attention."
            )
        fp8_cache = self.k_cache.dtype == torch.float8_e4m3fn
        if fp8_cache:
            fp8_max = torch.finfo(torch.float8_e4m3fn).max
            q = q.clamp(min=-fp8_max, max=fp8_max).to(torch.float8_e4m3fn)
        if write_kv_cache and self.k_cache.numel() and not context.is_dummy:
            if fp8_cache:
                from dlengine.runtime.kernel.triton.hopper.fp8_utils import (
                    store_kcache_fp8,
                )

                store_kcache_fp8(k, self.k_cache, context.slot_mapping)
            else:
                from dlengine.runtime.kernel.triton.generic.kv_store import store_kcache

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
            # FlashInfer expects [pages, heads, page_size, dim]. The singleton
            # dimension is the KV-head axis, not the page-size axis.
            kv_cache = kv_cache.permute(0, 2, 1, 3)
        sparse_mla_top_k = 0
        if sparse_indices is not None and fp8_cache:
            expected_rows = batch_size * ntps
            if sparse_indices.shape[0] != expected_rows:
                raise RuntimeError(
                    "Sparse MLA index rows do not match query tokens: "
                    f"{sparse_indices.shape[0]} != {expected_rows}"
                )
            sparse_mla_top_k = sparse_indices.shape[-1]
            block_tables = sparse_indices.to(dtype=torch.int32).reshape(
                batch_size, ntps, sparse_mla_top_k
            )
        if not block_tables.is_contiguous():
            block_tables = block_tables.contiguous()
        # TRTLLM-GEN groups 128 tokens when constructing its paged schedule.
        # Therefore the page-table width must be a multiple of 128/page_size.
        # Scheduler metadata is intentionally trimmed to the active pages, so
        # pad only its unused tail; seq_lens remains the authoritative bound.
        if sparse_mla_top_k == 0:
            page_group = 128 // kv_cache.shape[2]
            remainder = block_tables.shape[1] % page_group
            if remainder:
                padding = block_tables.new_zeros(
                    (block_tables.shape[0], page_group - remainder)
                )
                block_tables = torch.cat((block_tables, padding), dim=1)
        if seq_lens.dtype != torch.int32 or not seq_lens.is_contiguous():
            seq_lens = seq_lens.to(dtype=torch.int32).contiguous()
        # Absorbed MLA returns compressed values. The model applies W_UV
        # afterwards to project kv_lora_rank to its configured v_head_dim.
        out = torch.empty(
            (*query.shape[:-1], self.kv_lora_rank),
            dtype=torch.bfloat16,
            device=query.device,
        )
        result = _trtllm_mla_decode_func(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=_get_trtllm_workspace(query.device),
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=(
                int(seq_lens.max().item())
                if sparse_mla_top_k and not torch.cuda.is_current_stream_capturing()
                # An idle attention-DP rank still reads the synthetic page
                # above; FlashInfer requires a positive scheduling bound.
                else max(1, context.block_tables.shape[-1]) * self.k_cache.shape[1]
            ),
            sparse_mla_top_k=sparse_mla_top_k,
            out=out,
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            backend="trtllm-gen" if fp8_cache else "auto",
            is_var_seq=True,
            uses_shared_paged_kv_idx=True,
        )
        return result.reshape(-1, self.num_heads, self.kv_lora_rank)


__all__ = ["TrtllmMlaAttention", "_get_trtllm_workspace"]
