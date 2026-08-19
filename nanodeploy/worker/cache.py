import dataclasses
import os
from collections import defaultdict
from typing import Literal

import dlslime
import torch
import torch.distributed as dist
from nanodeploy._cpp import BlockContextSlot
from nanodeploy.engine.sequence import Sequence
from nanodeploy.worker.distributed import get_dist_context


@dataclasses.dataclass
class _KVTokenRange:
    remote_sp_rank: int
    remote_block_id: int
    remote_token_offset: int
    local_block_id: int
    local_token_offset: int
    num_tokens: int


def _cached_token_slots(block_ctx, block_size: int) -> list[tuple[int, int, int]]:
    """Return physical slots whose KV has already been produced.

    A sequence is migrated immediately after prefill samples its first token.
    That token is already reflected in ``num_dispatched_tokens``, but its KV is
    produced by the first decode forward.  It is always the last local token on
    the master rank, so omit that one slot from both layouts.
    """
    dispatched = list(block_ctx.num_dispatched_tokens)
    master_sp_rank = block_ctx.master_sp_idx
    if not 0 <= master_sp_rank < len(dispatched):
        raise RuntimeError(
            f"Invalid migration master SP rank {master_sp_rank} for "
            f"{len(dispatched)} dispatched-token entries"
        )
    if dispatched[master_sp_rank] <= 0:
        raise RuntimeError(
            "The migration master rank has no slot for the pending decode token"
        )

    slots: list[tuple[int, int, int]] = []
    for sp_rank, num_tokens in enumerate(dispatched):
        num_cached_tokens = num_tokens - int(sp_rank == master_sp_rank)
        block_table = block_ctx.sp_block_table[sp_rank]
        if num_cached_tokens > len(block_table) * block_size:
            raise RuntimeError(
                "KV migration metadata exceeds the allocated block table: "
                f"sp_rank={sp_rank}, cached_tokens={num_cached_tokens}, "
                f"blocks={len(block_table)}, block_size={block_size}"
            )
        for local_token_idx in range(num_cached_tokens):
            slots.append(
                (
                    sp_rank,
                    block_table[local_token_idx // block_size],
                    local_token_idx % block_size,
                )
            )
    return slots


def _plan_kv_migration_ranges(
    remote_ctx,
    local_ctx,
    block_size: int,
    local_sp_rank: int,
) -> list[_KVTokenRange]:
    """Map a possibly differently-sharded remote KV layout to this SP rank."""
    remote_slots = _cached_token_slots(remote_ctx, block_size)
    local_slots = _cached_token_slots(local_ctx, block_size)
    if len(remote_slots) != len(local_slots):
        raise RuntimeError(
            "P/D KV migration layouts contain different cached-token counts: "
            f"remote={len(remote_slots)}, local={len(local_slots)}"
        )

    ranges: list[_KVTokenRange] = []
    for remote_slot, local_slot in zip(remote_slots, local_slots):
        remote_sp_rank, remote_block_id, remote_token_offset = remote_slot
        dst_sp_rank, local_block_id, local_token_offset = local_slot
        if dst_sp_rank != local_sp_rank:
            continue

        if ranges:
            previous = ranges[-1]
            can_extend = (
                previous.remote_sp_rank == remote_sp_rank
                and previous.remote_block_id == remote_block_id
                and previous.remote_token_offset + previous.num_tokens
                == remote_token_offset
                and previous.local_block_id == local_block_id
                and previous.local_token_offset + previous.num_tokens
                == local_token_offset
            )
            if can_extend:
                previous.num_tokens += 1
                continue

        ranges.append(
            _KVTokenRange(
                remote_sp_rank=remote_sp_rank,
                remote_block_id=remote_block_id,
                remote_token_offset=remote_token_offset,
                local_block_id=local_block_id,
                local_token_offset=local_token_offset,
                num_tokens=1,
            )
        )
    return ranges


def _get_slime_qp_num() -> int:
    raw = os.environ.get("SLIME_QP_NUM", "1")
    try:
        num_qp = int(raw)
    except ValueError:
        return 1
    return max(num_qp, 1)


@dataclasses.dataclass
class CacheContext:
    num_kv_heads: int
    head_dim: int
    block_size: int
    num_hidden_layers: int
    attention_tp: int
    gpu_memory_utilization: float
    gpu_memory_limit_gb: float | None = None
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    mode: Literal["gqa", "mla"] = "gqa"
    num_local_kvcache_blocks = -1
    num_remote_kvcache_blocks: dict[str, int] = None
    kv_cache: torch.Tensor = None
    selected_nic: str | None = None
    endpoints: dict[str, dict[int, dlslime.RDMAEndpoint]] = None

    # used for MLA mode
    kv_lora_rank: int = 0
    qk_rope_head_dim: int = 0

    @property
    def num_local_kv_heads(self):
        return self.num_kv_heads // self.attention_tp

    def __post_init__(self):

        free, total = torch.cuda.mem_get_info()
        if self.gpu_memory_limit_gb is not None:
             total = min(total, self.gpu_memory_limit_gb * 1024**3)
        used = torch.cuda.mem_get_info()[1] - free # real used
        memory_stats = torch.cuda.memory_stats()
        peak = memory_stats["allocated_bytes.all.peak"]
        current = memory_stats["allocated_bytes.all.current"]

        if self.mode == "gqa":
            assert self.attention_tp <= self.num_kv_heads
        elif self.mode == "mla":
            assert self.attention_tp == 1
            assert self.block_size == 64, "MLA mode only support block_size=64"
            self.num_kv_heads = 1
            self.head_dim = self.kv_lora_rank + self.qk_rope_head_dim
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        block_bytes = (
            self.num_hidden_layers
            * self.block_size
            * self.num_local_kv_heads
            * self.head_dim
            * self.dtype.itemsize
        )
        if self.mode == "gqa":
            block_bytes *= 2

        self.num_local_kvcache_blocks = (
            int(total * self.gpu_memory_utilization - used - peak + current)
            // block_bytes
        )

        print(
            f"Rank{dist.get_rank()} num_local_kvcache_blocks: {self.num_local_kvcache_blocks}"
        )

        assert self.num_local_kvcache_blocks > 0

        available_nics = dlslime.available_nic()
        selected_nic_idx = dist.get_rank() % len(available_nics)
        self.selected_nic = available_nics[selected_nic_idx]
        assert self.selected_nic

        self.endpoints = {}
        self.num_remote_kvcache_blocks = {}

    def block_stride(self, block_idx: int):
        return (
            block_idx
            * self.block_size
            * self.num_local_kv_heads
            * self.head_dim
            * self.dtype.itemsize
        )

    def local_layer_stride(self, layer_idx: int, block_idx: int):
        return (
            self.block_stride(self.num_local_kvcache_blocks)
        ) * layer_idx + self.block_stride(block_idx)

    def remote_layer_stride(
        self, layer_idx: int, block_idx: int, remote_engine_id: str
    ):
        return (
            self.block_stride(self.num_remote_kvcache_blocks[remote_engine_id])
        ) * layer_idx + self.block_stride(block_idx)

    def local_kv_stride(self, kv_idx: int, layer_idx: int, block_idx: int):
        return self.local_layer_stride(
            self.num_hidden_layers, 0
        ) * kv_idx + self.local_layer_stride(layer_idx, block_idx)

    def remote_kv_stride(
        self, kv_idx: int, layer_idx: int, block_idx: int, remote_engine_id: str
    ):
        return self.remote_layer_stride(
            self.num_hidden_layers, 0, remote_engine_id
        ) * kv_idx + self.remote_layer_stride(layer_idx, block_idx, remote_engine_id)

    def allocate_kvcache(self, num_kvcache_blocks):
        self.num_local_kvcache_blocks = num_kvcache_blocks
        
        kv_count = 2 if self.mode == "gqa" else 1
        
        self.kv_cache = torch.empty(
            kv_count,
            self.num_hidden_layers,
            self.num_local_kvcache_blocks,
            self.block_size,
            self.num_local_kv_heads,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )

    def p2p_init(
        self, remote_engine_name: str, num_kv_blocks: int, remote_world_size: int
    ) -> dict[int, dict]:
        # init endpoint
        # register memory region
        endpoints = self.endpoints[remote_engine_name] = {}
        endpoints_info = {}
        self.num_remote_kvcache_blocks[remote_engine_name] = num_kv_blocks
        num_qp = _get_slime_qp_num()
        for i in range(remote_world_size):
            endpoint = dlslime.RDMAEndpoint(
                device_name=self.selected_nic, num_qp=num_qp
            )
            if i == 0:
                endpoint.register_memory_region(
                    get_dist_context().rank,
                    self.kv_cache.data_ptr() + self.kv_cache.storage_offset(),
                    self.kv_cache.numel() * self.kv_cache.itemsize,
                )
            endpoint_info = endpoint.endpoint_info()
            endpoints[i] = endpoint
            endpoints_info[i] = endpoint_info
        return endpoints_info

    def p2p_connect(
        self, remote_engine_id: str, endpoints_info_list: list[dict[int, dict]]
    ):
        for i, endpoints_info in enumerate(endpoints_info_list):
            endpoint_info = endpoints_info[dist.get_rank()]
            self.endpoints[remote_engine_id][i].connect(endpoint_info)

    def migrate(self, seqs: list[Sequence]):
        assigns = defaultdict(lambda: defaultdict(list))
        sp_idx = get_dist_context().attn_sp_rank
        for seq in seqs:
            remote_ctx = seq.block_ctx(BlockContextSlot.MIGRATE)
            local_ctx = seq.block_ctx(BlockContextSlot.ACTIVE)
            ranges = _plan_kv_migration_ranges(
                remote_ctx, local_ctx, self.block_size, sp_idx
            )
            token_bytes = self.block_stride(1) // self.block_size
            for token_range in ranges:
                remote_rank = (
                    seq.dp_idx(BlockContextSlot.MIGRATE)
                    * remote_ctx.attention_sp
                    + token_range.remote_sp_rank
                )
                for kv_idx in range(self.kv_cache.size(0)):
                    for layer_idx in range(self.num_hidden_layers):
                        assignment = (
                            get_dist_context().rank,
                            remote_rank,
                            self.remote_kv_stride(
                                kv_idx,
                                layer_idx,
                                token_range.remote_block_id,
                                remote_ctx.engine_id,
                            )
                            + token_range.remote_token_offset * token_bytes,
                            self.local_kv_stride(
                                kv_idx, layer_idx, token_range.local_block_id
                            )
                            + token_range.local_token_offset * token_bytes,
                            token_range.num_tokens * token_bytes,
                        )
                        assigns[remote_ctx.engine_id][remote_rank].append(assignment)

        futures = []
        for endpoint_key, endpoint_assign_batch in assigns.items():
            for replica_key, assign_batch in endpoint_assign_batch.items():
                futures.append(
                    self.endpoints[endpoint_key][replica_key].read(assign_batch)
                )

        [future.wait() for future in futures]


_CACHE_CONTEXT: CacheContext


def get_cache_context():
    return _CACHE_CONTEXT


def set_cache_context(
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    num_hidden_layers: int,
    attention_tp: int,
    gpu_memory_utilization: float,
    gpu_memory_limit_gb: float | None = None,
    kv_lora_rank: int = 0,
    qk_rope_head_dim: int = 0,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    mode: Literal["gqa", "mla"] = "gqa",
):
    global _CACHE_CONTEXT
    _CACHE_CONTEXT = CacheContext(
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_size=block_size,
        num_hidden_layers=num_hidden_layers,
        attention_tp=attention_tp,
        gpu_memory_utilization=gpu_memory_utilization,
        gpu_memory_limit_gb=gpu_memory_limit_gb,
        device=device,
        dtype=dtype,
        mode=mode,
    )
    return _CACHE_CONTEXT
