import dataclasses
import os
from collections import defaultdict
from typing import Any, Literal

import dlslime
import torch
import torch.distributed as dist
from nanodeploy._cpp import BlockContextSlot
from nanodeploy.engine.sequence import Sequence
from nanodeploy.worker.distributed import get_dist_context


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
            for remote_block_idx, source_block_idx in zip(
                seq.block_ctx(BlockContextSlot.MIGRATE).block_location,
                seq.block_ctx(BlockContextSlot.ACTIVE).block_location,
            ):
                for kv_idx in range(self.kv_cache.size(0)):
                    for layer_idx in range(self.num_hidden_layers):
                        if source_block_idx[0] == sp_idx:
                            remote_rank = (
                                seq.dp_idx(BlockContextSlot.MIGRATE)
                                * seq.block_ctx(BlockContextSlot.MIGRATE).attention_sp
                                + remote_block_idx[0]
                            )
                            assignment = (
                                get_dist_context().rank,
                                remote_rank,
                                self.remote_kv_stride(
                                    kv_idx,
                                    layer_idx,
                                    remote_block_idx[1],
                                    seq.block_ctx(BlockContextSlot.MIGRATE).engine_id,
                                ),
                                self.local_kv_stride(
                                    kv_idx, layer_idx, source_block_idx[1]
                                ),
                                self.block_stride(1),
                            )
                            assigns[seq.block_ctx(BlockContextSlot.MIGRATE).engine_id][
                                remote_rank
                            ].append(assignment)

            futures = []
            for endpoint_key, endpoint_assign_batch in assigns.items():
                for replica_key, assign_batch in endpoint_assign_batch.items():
                    futures.append(
                        self.endpoints[endpoint_key][replica_key].read(assign_batch)
                    )

            [future.wait() for future in futures]

    def migrate_decode_kv_blocks(self, migration_plans: list[dict[str, Any]]):
        """Copy decode KV cache blocks between SP ranks within the same DP group."""
        if not migration_plans:
            return

        dist_context = get_dist_context()
        dp_rank = dist_context.attn_dp_rank
        sp_rank = dist_context.attn_sp_rank
        tp_rank = dist_context.attn_tp_rank
        sp_size = dist_context.attn_sp_world_size
        tp_size = dist_context.attn_tp_world_size

        send_tensors: list[torch.Tensor] = []
        recv_assignments: list[tuple[torch.Tensor, torch.Tensor]] = []
        ops: list[dist.P2POp] = []

        def global_rank_for(dp_idx: int, sp_idx: int) -> int:
            return dp_idx * sp_size * tp_size + sp_idx * tp_size + tp_rank

        for plan in migration_plans:
            plan_dp_idx = int(plan.get("dp_idx", -1))
            if plan_dp_idx != dp_rank:
                continue

            for item in plan.get("items", []):
                src_sp = int(item["src_sp"])
                dst_sp = int(item["dst_sp"])
                src_blocks = list(item.get("src_block_ids", []))
                dst_blocks = list(item.get("dst_block_ids", []))
                if len(src_blocks) != len(dst_blocks):
                    raise ValueError("src_block_ids and dst_block_ids length mismatch")

                if sp_rank == src_sp:
                    dst_rank = global_rank_for(plan_dp_idx, dst_sp)
                    for src_block in src_blocks:
                        payload = self.kv_cache[:, :, src_block : src_block + 1].contiguous()
                        send_tensors.append(payload)
                        ops.append(dist.P2POp(dist.isend, payload, dst_rank))

                if sp_rank == dst_sp:
                    src_rank = global_rank_for(plan_dp_idx, src_sp)
                    for dst_block in dst_blocks:
                        dst_view = self.kv_cache[:, :, dst_block : dst_block + 1]
                        recv_buffer = torch.empty_like(
                            dst_view, memory_format=torch.contiguous_format
                        )
                        recv_assignments.append((recv_buffer, dst_view))
                        ops.append(dist.P2POp(dist.irecv, recv_buffer, src_rank))

        if not ops:
            return

        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
        for recv_buffer, dst_view in recv_assignments:
            dst_view.copy_(recv_buffer)


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
