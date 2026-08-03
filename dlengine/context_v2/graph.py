"""Persistent graph/runtime context.

This module owns state that survives per-step runtime context resets: CUDA
graph helpers, graph-safe backend wrappers, and long-lived planning buffers.
Per-step batch metadata should stay in ``BatchContext``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch

from dlengine.context_v2 import BaseContext


class PagedAttentionStrategy(str, Enum):
    AUTO = "auto"
    FLASHINFER = "flashinfer"
    FLASH_ATTN = "flash_attn"
    FLASH_MLA = "flash_mla"


@dataclass(slots=True)
class FlashInferDecodeGraphConfig:
    enabled: bool = False
    max_num_blocks: int = 0
    block_size: int = 0
    num_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    softmax_scale: float = 1.0
    dtype: torch.dtype | None = None
    use_tensor_cores: bool = False
    disable_split_kv: bool = True
    fixed_split_size: int = 0
    reuse_page_plan: bool = False


@dataclass
class FlashInferDecodeGraphState:
    config: FlashInferDecodeGraphConfig
    flashinfer: object | None = None
    wrappers: dict[int, object] = field(default_factory=dict)
    workspaces: dict[int, torch.Tensor] = field(default_factory=dict)
    indptr: dict[int, torch.Tensor] = field(default_factory=dict)
    indices: dict[int, torch.Tensor] = field(default_factory=dict)
    last_page_len: dict[int, torch.Tensor] = field(default_factory=dict)
    plan_keys: dict[int, tuple] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.config.enabled:
            import flashinfer

            self.flashinfer = flashinfer

    def ensure_wrapper(self, master_bs: int) -> object | None:
        if not self.config.enabled or self.flashinfer is None:
            return None
        wrapper = self.wrappers.get(master_bs)
        if wrapper is not None:
            return wrapper

        max_indices = master_bs * self.config.max_num_blocks
        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        indptr = torch.empty(master_bs + 1, dtype=torch.int32, device="cuda")
        indices = torch.empty(max_indices, dtype=torch.int32, device="cuda")
        last_page_len = torch.empty(master_bs, dtype=torch.int32, device="cuda")
        wrapper = self.flashinfer.CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
            workspace,
            indptr,
            indices,
            last_page_len,
            kv_layout="NHD",
            use_tensor_cores=self.config.use_tensor_cores,
        )
        self.workspaces[master_bs] = workspace
        self.indptr[master_bs] = indptr
        self.indices[master_bs] = indices
        self.last_page_len[master_bs] = last_page_len
        self.wrappers[master_bs] = wrapper
        return wrapper

    def plan(
        self,
        master_bs: int,
        bs: int,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        page_plan_key: tuple[int, ...] | None = None,
    ) -> object | None:
        wrapper = self.ensure_wrapper(master_bs)
        if wrapper is None:
            return None

        indptr, indices, last_page_len = paged_decode_metadata(
            block_tables, context_lens, bs, self.config.block_size
        )
        plan_indptr = indptr
        plan_indices = indices
        plan_last_page_len = last_page_len
        if self.config.reuse_page_plan:
            self.indptr[master_bs].copy_(indptr, non_blocking=True)
            self.last_page_len[master_bs].copy_(last_page_len, non_blocking=True)
            self.indices[master_bs][: len(indices)].copy_(indices, non_blocking=True)
            plan_indptr = self.indptr[master_bs]
            plan_indices = self.indices[master_bs][: len(indices)]
            plan_last_page_len = self.last_page_len[master_bs]
            plan_key = (
                (bs, page_plan_key[:bs])
                if page_plan_key is not None
                else (
                    bs,
                    int(indptr[-1].item()),
                    tuple((indptr[1:] - indptr[:-1]).tolist()),
                )
            )
            # Equal page counts do not imply equal physical KV pages after a
            # request slot is reused. The persistent buffers above must always
            # be refreshed before reusing the existing wrapper plan.
            if self.plan_keys.get(master_bs) == plan_key:
                return wrapper
            self.plan_keys[master_bs] = plan_key

        plan_kwargs = {}
        if self.config.fixed_split_size > 0:
            plan_kwargs["fixed_split_size"] = self.config.fixed_split_size
        if self.config.disable_split_kv:
            plan_kwargs["disable_split_kv"] = True
        wrapper.plan(
            plan_indptr,
            plan_indices,
            plan_last_page_len,
            num_qo_heads=self.config.num_heads,
            num_kv_heads=self.config.num_kv_heads,
            head_dim=self.config.head_dim,
            page_size=self.config.block_size,
            pos_encoding_mode="NONE",
            q_data_type=self.config.dtype,
            kv_data_type=self.config.dtype,
            o_data_type=self.config.dtype,
            sm_scale=self.config.softmax_scale,
            **plan_kwargs,
        )
        return wrapper


@dataclass
class DecodeGraphContext:
    max_num_seqs: int
    block_size: int
    max_num_blocks: int
    is_mla: bool
    is_dsv4: bool
    has_indexer: bool
    input_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    context_lens: torch.Tensor
    block_tables: torch.Tensor
    outputs: torch.Tensor
    returns_logits: bool
    logits: torch.Tensor | None
    hisparse_slots: torch.Tensor
    hisparse_slot_mapping: torch.Tensor
    per_layer_token_part: torch.Tensor | None = None
    gdn_state_slots: torch.Tensor | None = None
    dummy_gdn_slot: int | None = None
    dsv4_state_slots: torch.Tensor | None = None
    dummy_dsv4_slot: int | None = None
    dsv4_compressed_block_tables: dict[int, torch.Tensor] = field(default_factory=dict)
    dsv4_compressed_dummy_pages: dict[int, int] = field(default_factory=dict)
    sched_metas: dict[int, object] = field(default_factory=dict)
    sparse_sched_metas: dict[int, object] = field(default_factory=dict)
    graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = field(default_factory=dict)
    graph_map: dict[int, list[int]] = field(default_factory=dict)
    graph_pool: object | None = None
    flashinfer_decode: FlashInferDecodeGraphState | None = None


@dataclass
class GraphContext(BaseContext):
    decode: DecodeGraphContext | None = None
    active_flashinfer_decode_wrapper: object | None = None

    @classmethod
    def get_context_type(cls) -> str:
        return "graph"

    @classmethod
    def get_context_name(cls) -> str:
        return "GraphContext"

    @property
    def flashinfer_decode(self) -> FlashInferDecodeGraphState | None:
        return self.decode.flashinfer_decode if self.decode is not None else None

    def set_decode(
        self, decode: DecodeGraphContext | None
    ) -> DecodeGraphContext | None:
        self.decode = decode
        return self.decode

    def set_flashinfer_decode(
        self, config: FlashInferDecodeGraphConfig | None
    ) -> FlashInferDecodeGraphState | None:
        if self.decode is None:
            return None
        self.decode.flashinfer_decode = (
            FlashInferDecodeGraphState(config) if config is not None else None
        )
        return self.decode.flashinfer_decode

    def set_active_flashinfer_decode_wrapper(self, wrapper: object | None) -> None:
        self.active_flashinfer_decode_wrapper = wrapper

    def clear_step_context(self) -> None:
        self.active_flashinfer_decode_wrapper = None

    def clear_context(self) -> None:
        self.clear_step_context()

    def reset_context(self) -> None:
        self.clear_step_context()
        self.decode = None


def paged_decode_metadata(
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    bs: int,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_lens = context_lens[:bs].to(torch.int32)
    pages_per_seq = torch.div(
        seq_lens + page_size - 1, page_size, rounding_mode="floor"
    )
    indptr = torch.empty(bs + 1, device=seq_lens.device, dtype=torch.int32)
    indptr[0] = 0
    indptr[1:] = torch.cumsum(pages_per_seq, dim=0)

    max_pages = block_tables.shape[1]
    page_offsets = torch.arange(
        max_pages, device=block_tables.device, dtype=torch.int32
    )
    mask = page_offsets.unsqueeze(0) < pages_per_seq.unsqueeze(1)
    indices = block_tables[:bs, :max_pages][mask].contiguous()

    last_page_len = seq_lens % page_size
    last_page_len = torch.where(
        last_page_len == 0,
        torch.full_like(last_page_len, page_size),
        last_page_len,
    )
    return indptr, indices, last_page_len


def update_decode_last_page_len(
    dst: torch.Tensor,
    context_lens: torch.Tensor,
    bs: int,
    page_size: int,
) -> None:
    active = dst[:bs]
    active.copy_(context_lens[:bs], non_blocking=True)
    active.remainder_(page_size)
    active.masked_fill_(active == 0, page_size)


_CONTEXT = GraphContext()


def get_graph_context() -> GraphContext:
    return _CONTEXT


def reset_graph_context() -> None:
    global _CONTEXT
    _CONTEXT = GraphContext()


__all__ = [
    "FlashInferDecodeGraphConfig",
    "FlashInferDecodeGraphState",
    "DecodeGraphContext",
    "GraphContext",
    "PagedAttentionStrategy",
    "get_graph_context",
    "paged_decode_metadata",
    "reset_graph_context",
]
