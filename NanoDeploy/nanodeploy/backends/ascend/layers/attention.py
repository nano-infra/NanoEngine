"""Ascend NPU GQA attention implementation.

Uses torch_npu npu_fused_infer_attention_score (TND layout, sparse_mode=3):
  - Prefill: TND, block_table=None, actual_seq_lengths = cumulative q seqlens
  - Decode:  TND, k/v_cache reshaped to [blocks, block_size, nkv*dim],
             actual_seq_lengths = cumulative decode tokens (1 per seq)

This matches the calling convention used by vllm-ascend.
Sequence Parallelism (sp > 1) is not supported in this version.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from nanodeploy.backends.base_backend import AttentionBase
from nanodeploy.backends.ascend.ops.kv_ops import store_kvcache_npu

# Fused KV cache store (single NPU op, much faster than multi-op PyTorch path).
# Available via torch_npu ATB; matches vllm-ascend's reshape_and_cache.
_USE_FUSED_KV_STORE = True

# Graph mode flag — set by model_runner during capture/replay.
#   "eager"   → default, uses _forward_decode_paged (with .tolist())
#   "capture" → graph capture, uses _forward_decode_graph_capture (graph_task_group)
#   "replay"  → after capture; decode is driven by graph.replay(), not forward()
_NPU_GRAPH_MODE: str = "eager"

from nanodeploy.context.context import get_context
from nanodeploy.logging import get_logger

logger = get_logger()

# Pre-computed 2048×2048 causal mask (upper triangle = 1, lower = 0).
# sparse_mode=3 uses this as an additive bias; positions where mask=1 are
# effectively masked out. Shape matches vllm-ascend convention.
_CAUSAL_MASK_CACHE: torch.Tensor | None = None


def _get_causal_mask(device: torch.device) -> torch.Tensor:
    global _CAUSAL_MASK_CACHE
    if _CAUSAL_MASK_CACHE is None or _CAUSAL_MASK_CACHE.device != device:
        _CAUSAL_MASK_CACHE = (
            torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1).to(device)
        )
    return _CAUSAL_MASK_CACHE


# ---------------------------------------------------------------------------
# Graph-capture metadata for direct paged attention via graph_task_group
# ---------------------------------------------------------------------------

@dataclass
class AscendGraphParams:
    """Per-layer graph task group handles and params for replay."""
    handles: dict[int, list] = field(default_factory=dict)      # attn_bs -> [handle per layer]
    workspaces: dict[int, Any] = field(default_factory=dict)    # attn_bs -> workspace tensor
    attn_params: dict[int, list] = field(default_factory=dict)  # attn_bs -> [param tuple per layer]


_graph_params: AscendGraphParams | None = None


def init_graph_params(all_attn_bs: list[int]) -> None:
    """Initialize graph params with empty lists for each capture batch size."""
    global _graph_params
    _graph_params = AscendGraphParams()
    for bs in all_attn_bs:
        _graph_params.handles[bs] = []
        _graph_params.attn_params[bs] = []


def update_graph_attention_params(update_stream, context, compute_bs: int) -> None:
    """Update captured attention graph task groups with fresh seq lengths.

    Called by model_runner BEFORE graph.replay() to re-bind the FIA kernel's
    actual_seq_lengths_kv parameter (a Python list, host-side) without
    re-capturing the graph.

    All updates run on update_stream.  The caller is responsible for syncing
    update_stream → main stream (one event) before graph.replay().
    """
    import torch_npu

    assert _graph_params is not None, "init_graph_params() was not called"

    # Use the CPU Python list directly to avoid a D2H .tolist() sync.
    # context_lens_for_attn_cpu is populated from meta.context_lens_for_attn
    # in prepare_decode_bytes — same data, never leaves the host.
    if context.context_lens_for_attn_cpu is not None:
        actual_seq_kv = context.context_lens_for_attn_cpu[:compute_bs]
    else:
        actual_seq_kv = context.context_lens_for_attn[:compute_bs].tolist()
    workspace = _graph_params.workspaces.get("shared")

    with torch.npu.stream(update_stream):
        for param, handle in zip(
            _graph_params.attn_params[compute_bs],
            _graph_params.handles[compute_bs],
        ):
            (query, key, value, block_table, block_size,
             _old_seq_kv, actual_seq_q,
             num_kv_heads, num_heads, scale,
             output, softmax_lse) = param

            torch.npu.graph_task_update_begin(update_stream, handle)
            torch_npu.npu_fused_infer_attention_score.out(
                query=query,
                key=key, value=value,
                block_table=block_table,
                input_layout="TND", block_size=block_size,
                actual_seq_lengths=actual_seq_q,
                actual_seq_lengths_kv=actual_seq_kv,
                num_key_value_heads=num_kv_heads,
                num_heads=num_heads,
                scale=scale, sparse_mode=0,
                workspace=workspace,
                out=[output, softmax_lse],
            )
            torch.npu.graph_task_update_end(update_stream)


class AscendAttention(AttentionBase):
    """Ascend NPU GQA attention using npu_fused_infer_attention_score (TND)."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        v_head_dim: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.v_head_dim = v_head_dim
        # Placeholders: assigned by model_runner.allocate_kvcache
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        context = get_context()

        # Store new tokens into paged KV cache (skip for dummy warmup)
        if self.k_cache.numel() and self.v_cache.numel() and not context.is_dummy:
            if _USE_FUSED_KV_STORE:
                import torch_npu
                torch_npu._npu_reshape_and_cache(
                    key=k,
                    value=v,
                    key_cache=self.k_cache,
                    value_cache=self.v_cache,
                    slot_indices=context.slot_mapping.to(torch.int32),
                )
            else:
                store_kvcache_npu(k, v, self.k_cache, self.v_cache, context.slot_mapping)

        if context.is_prefill:
            return self._forward_prefill(q, k, v, context)
        else:
            return self._forward_decode(q, context)

    def _forward_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context,
    ) -> torch.Tensor:
        """Prefill via npu_fused_infer_attention_score, TND layout, sparse_mode=3.

        Args:
            q: [total_tokens, num_heads, head_dim]   (TND — T=total, N=heads, D=dim)
            k: [total_tokens, num_kv_heads, head_dim]
            v: [total_tokens, num_kv_heads, head_dim]
        """
        import torch_npu

        # cu_seqlens_q: [num_seqs+1] cumulative token offsets.
        # actual_seq_lengths for TND must be CUMULATIVE (not per-seq lengths).
        cu_seqlens_q = context.cu_seqlens_q  # [num_seqs+1]
        actual_seq_lengths_q = cu_seqlens_q[1:].tolist()

        if context.cu_seqlens_k is not None:
            actual_seq_lengths_kv = context.cu_seqlens_k[1:].tolist()
        else:
            actual_seq_lengths_kv = actual_seq_lengths_q

        total_tokens = int(cu_seqlens_q[-1].item())
        atten_mask = _get_causal_mask(q.device)

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=q[:total_tokens].contiguous(),
            key=k[:total_tokens].contiguous(),
            value=v[:total_tokens].contiguous(),
            atten_mask=atten_mask,
            block_table=None,
            input_layout="TND",
            block_size=128,
            actual_seq_lengths=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
        )
        # attn_output: [total_tokens * num_heads * head_dim] or [total_tokens, num_heads, head_dim]
        return attn_output.view(total_tokens, self.num_heads, self.head_dim)

    def _forward_decode(
        self,
        q: torch.Tensor,
        context,
    ) -> torch.Tensor:
        if _NPU_GRAPH_MODE == "warmup":
            return self._forward_decode_graph_warmup(q, context)
        if _NPU_GRAPH_MODE == "capture":
            return self._forward_decode_graph_capture(q, context)
        # Both eager and replay use direct paged attention.
        # (During replay, this method is never called — graph.replay() drives execution.
        #  The update_graph_attention_params() call refreshes FIA params before replay.)
        return self._forward_decode_paged(q, context)

    def _forward_decode_paged(
        self,
        q: torch.Tensor,
        context,
    ) -> torch.Tensor:
        """Decode via npu_fused_infer_attention_score with block_table (no pre-gather).

        Cache is reshaped (view, no copy) to [num_blocks, block_size, nkv*dim]
        and the kernel indexes into blocks via block_table.  Matches vllm-ascend.

        Uses .tolist() for actual_seq_lengths_kv — cannot be used inside graph capture.
        """
        import torch_npu

        bs = q.shape[0]
        compute_bs = getattr(context, "attention_compute_bs", bs) or bs
        q_compute = q[:compute_bs]

        context_lens = context.context_lens_for_attn[:compute_bs]
        block_tables = context.block_tables[:compute_bs]

        n_blks_total, block_size, nkv, kdim = self.k_cache.shape

        key = self.k_cache.view(n_blks_total, block_size, -1)
        value = self.v_cache.view(n_blks_total, block_size, nkv * self.v_head_dim)

        if not hasattr(self, "_cum_q_lens") or len(self._cum_q_lens) < compute_bs:
            self._cum_q_lens = list(range(1, compute_bs + 1))
        actual_seq_q = self._cum_q_lens[:compute_bs]
        # Use CPU list when available to avoid D2H .tolist() sync per layer.
        if context.context_lens_for_attn_cpu is not None:
            actual_seq_kv = context.context_lens_for_attn_cpu[:compute_bs]
        else:
            actual_seq_kv = context_lens.tolist()

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=q_compute.contiguous(),
            key=key,
            value=value,
            block_table=block_tables.to(torch.int32),
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=actual_seq_q,
            actual_seq_lengths_kv=actual_seq_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=0,
        )
        out = attn_output.view(compute_bs, self.num_heads, self.head_dim)

        if compute_bs < bs:
            pad = torch.zeros(
                bs - compute_bs, self.num_heads, self.head_dim,
                dtype=out.dtype, device=out.device,
            )
            out = torch.cat([out, pad], dim=0)

        return out

    def _forward_decode_graph_warmup(
        self,
        q: torch.Tensor,
        context,
    ) -> torch.Tensor:
        """Warmup path: pre-compute FIA workspace outside graph capture context.

        Runs normal paged attention but also caches the workspace tensor that
        will be needed during the subsequent graph capture pass.  Workspace
        computation allocates device memory and may sync, so it MUST happen
        outside ``torch.npu.graph()``.
        """
        import torch_npu

        bs = q.shape[0]
        compute_bs = getattr(context, "attention_compute_bs", bs) or bs
        q_compute = q[:compute_bs]

        n_blks, block_size, nkv, kdim = self.k_cache.shape
        key = self.k_cache.view(n_blks, block_size, -1)
        value = self.v_cache.view(n_blks, block_size, nkv * self.v_head_dim)

        block_tables = context.block_tables[:compute_bs]
        context_lens = context.context_lens_for_attn[:compute_bs]
        bt_int32 = block_tables.to(torch.int32)

        actual_seq_q = list(range(1, compute_bs + 1))
        actual_seq_kv = context_lens.tolist()   # OK here — not inside graph capture

        # Compute & cache a single shared workspace (max-sized, valid for all bs).
        # _get_max_workspace returns the maximum workspace needed, so the workspace
        # computed for the largest compute_bs works for all smaller sizes too.
        if not _graph_params.workspaces:
            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                query=q_compute.contiguous(),
                key=key, value=value,
                block_table=bt_int32,
                input_layout="TND", block_size=block_size,
                actual_seq_lengths=actual_seq_q,
                actual_seq_lengths_kv=actual_seq_kv,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                sparse_mode=0, scale=self.scale,
            )
            _graph_params.workspaces["shared"] = workspace

        # Run normal paged attention for warmup output
        return self._forward_decode_paged(q, context)

    def _forward_decode_graph_capture(
        self,
        q: torch.Tensor,
        context,
    ) -> torch.Tensor:
        """Graph-capture path: wrap FIA in graph_task_group for later param update.

        Must run INSIDE ``torch.npu.graph()`` context.  Workspace must already
        be cached by a prior ``_forward_decode_graph_warmup`` call.
        """
        import torch_npu

        bs = q.shape[0]
        compute_bs = getattr(context, "attention_compute_bs", bs) or bs
        q_compute = q[:compute_bs]

        n_blks, block_size, nkv, kdim = self.k_cache.shape
        key = self.k_cache.view(n_blks, block_size, -1)
        value = self.v_cache.view(n_blks, block_size, nkv * self.v_head_dim)

        block_tables = context.block_tables[:compute_bs]
        bt_int32 = block_tables.to(torch.int32)

        # Dummy seq lengths — real values set via update_graph_attention_params
        actual_seq_q = list(range(1, compute_bs + 1))
        actual_seq_kv = [1] * compute_bs

        # Pre-allocate output + softmax_lse for .out() variant
        output = torch.empty(compute_bs, self.num_heads, self.head_dim,
                             dtype=q.dtype, device=q.device)
        softmax_lse = torch.empty(1, dtype=q.dtype, device=q.device)

        # Workspace MUST already be cached from warmup phase (single shared workspace)
        workspace = _graph_params.workspaces.get("shared")
        assert workspace is not None, (
            "Workspace not cached. "
            "Run warmup phase (_NPU_GRAPH_MODE='warmup') before capture."
        )

        # Store params for replay updates
        _graph_params.attn_params[compute_bs].append((
            q_compute,
            key, value,
            bt_int32,
            block_size,
            actual_seq_kv,     # Python list — updated fresh on each replay
            actual_seq_q,      # Python list — static
            self.num_kv_heads, self.num_heads, self.scale,
            output, softmax_lse,
        ))

        # Capture attention in a graph task group
        stream = torch_npu.npu.current_stream()
        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score.out(
            query=q_compute.contiguous(),
            key=key, value=value,
            block_table=bt_int32,
            input_layout="TND", block_size=block_size,
            actual_seq_lengths=actual_seq_q,
            actual_seq_lengths_kv=actual_seq_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale, sparse_mode=0,
            workspace=workspace,
            out=[output, softmax_lse],
        )
        handle = torch.npu.graph_task_group_end(stream)
        _graph_params.handles[compute_bs].append(handle)

        out = output.view(compute_bs, self.num_heads, self.head_dim)
        if compute_bs < bs:
            pad = torch.zeros(bs - compute_bs, self.num_heads, self.head_dim,
                              dtype=out.dtype, device=out.device)
            out = torch.cat([out, pad], dim=0)
        return out
