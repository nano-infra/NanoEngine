import concurrent.futures
import dataclasses
from collections import defaultdict
from typing import Any, Literal

import dlslime
import torch
import torch.distributed as dist
from nanodeploy._cpp import BlockContextSlot
from nanodeploy.context.distributed import get_dist_context
from nanodeploy.engine.sequence import Sequence
from nanodeploy.logging import get_logger

logger = get_logger("nanodeploy")

# Broker path: RDMALazyPeer uses (local_mr_key, remote_mr_key, target_off, source_off, length)
_KV_CACHE_BUFFER_ID = "kv_cache"


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
    endpoints: dict[str, dict[int, Any]] = None  # RDMAEndpoint or RDMALazyPeer

    # used for MLA mode
    kv_lora_rank: int = 0
    qk_rope_head_dim: int = 0

    # Broker path: optional, set when ensure_p2p_connected is used
    broker_host: str | None = None
    broker_base_port: int | None = None

    @property
    def num_local_kv_heads(self):
        return self.num_kv_heads // self.attention_tp

    def __post_init__(self):

        free, total = torch.cuda.mem_get_info()
        if self.gpu_memory_limit_gb is not None:
            total = min(total, self.gpu_memory_limit_gb * 1024**3)
        used = torch.cuda.mem_get_info()[1] - free  # real used
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

        logger.info(
            f"Rank{dist.get_rank()} num_local_kvcache_blocks: {self.num_local_kvcache_blocks}"
        )

        assert self.num_local_kvcache_blocks > 0

        available_nics = dlslime.available_nic()
        selected_nic_idx = dist.get_rank() % len(available_nics)
        self.selected_nic = available_nics[selected_nic_idx]
        assert self.selected_nic

        self.endpoints = {}
        self.num_remote_kvcache_blocks = {}
        self.peer_addrs: dict[str, list[str]] = {}  # peer_id -> addrs for Broker path
        self._broker = None
        self._accept_channel = None  # created at allocate_kvcache for accepting
        start_broker_fn = getattr(dlslime, "start_broker", None)
        if (
            self.broker_base_port is not None
            and self.broker_host is not None
            and callable(start_broker_fn)
        ):
            rank = dist.get_rank()
            bind_addr = f"{self.broker_host}:{self.broker_base_port + rank}"
            self._broker = start_broker_fn(bind_addr)
            logger.info(f"Broker started on {bind_addr}")

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
        # Broker path: create one peer for accepting connections (responder side)
        if self._broker is not None:
            self._accept_channel = self._broker.peer(device_name=self.selected_nic)
            kv_size = self.kv_cache.numel() * self.kv_cache.itemsize
            self._accept_channel.register_buffer(
                _KV_CACHE_BUFFER_ID,
                self.kv_cache.data_ptr(),
                kv_size,
            )
            logger.info("Accept channel created for Broker (responder)")

    def p2p_init(
        self, remote_engine_name: str, num_kv_blocks: int, remote_world_size: int
    ) -> list[list[dict]]:
        # init endpoint
        # register memory region
        endpoints = self.endpoints[remote_engine_name] = {}
        self.num_remote_kvcache_blocks[remote_engine_name] = num_kv_blocks

        def create_endpoint(i):
            endpoint = dlslime.RDMAEndpoint(device_name=self.selected_nic, num_qp=1)
            endpoint.register_memory_region(
                get_dist_context().rank,
                self.kv_cache.data_ptr(),
                self.kv_cache.storage_offset(),
                self.kv_cache.numel() * self.kv_cache.itemsize,
            )
            endpoint_info = endpoint.endpoint_info()
            return i, endpoint, endpoint_info

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(create_endpoint, range(remote_world_size))

        endpoints_info = []
        for i, endpoint, endpoint_info in results:
            endpoints[i] = endpoint
            endpoints_info.append(endpoint_info)

        return endpoints_info

    def p2p_connect(self, remote_engine_id: str, endpoints_info_list: list[list[dict]]):
        def connect_endpoint(args):
            i, endpoints_info = args
            endpoint_info = endpoints_info[dist.get_rank()]
            self.endpoints[remote_engine_id][i].connect(endpoint_info)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(executor.map(connect_endpoint, enumerate(endpoints_info_list)))

    def ensure_p2p_connected(
        self, peer_id: str, addrs: list[str], num_blocks: int
    ) -> None:
        """Ensure P2P link to peer via broker addrs (lazy connect). Idempotent."""
        if peer_id in self.endpoints:
            return
        if self._broker is None:
            raise RuntimeError(
                "ensure_p2p_connected requires Broker; set broker_host and broker_base_port in set_cache_context and build dlslime with BUILD_RDMA_RENDEZVOUS_ZMQ"
            )
        self.num_remote_kvcache_blocks[peer_id] = num_blocks
        self.peer_addrs[peer_id] = addrs
        endpoints = self.endpoints[peer_id] = {}
        kv_size = (
            self.kv_cache.numel() * self.kv_cache.itemsize
            if self.kv_cache is not None
            else 0
        )
        if kv_size == 0:
            raise RuntimeError("ensure_p2p_connected called before allocate_kvcache")
        for r, addr in enumerate(addrs):
            channel = self._broker.peer(device_name=self.selected_nic)
            channel.connect(addr)
            channel.register_buffer(
                _KV_CACHE_BUFFER_ID,
                self.kv_cache.data_ptr(),
                kv_size,
            )
            endpoints[r] = channel
        logger.info(f"P2P link ensured to {peer_id} ({len(addrs)} ranks)")

    def p2p_disconnect(self, remote_engine_id: str):
        if remote_engine_id in self.endpoints:
            del self.endpoints[remote_engine_id]
            if remote_engine_id in self.num_remote_kvcache_blocks:
                del self.num_remote_kvcache_blocks[remote_engine_id]
            if remote_engine_id in self.peer_addrs:
                del self.peer_addrs[remote_engine_id]
            logger.info(f"P2P Link to {remote_engine_id} disconnected and cleared.")

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
                    channel = self.endpoints[endpoint_key][replica_key]
                    if hasattr(channel, "get_remote_mr_key"):
                        # Broker path: RDMALazyPeer uses (local_mr_key, remote_mr_key, target_off, source_off, length)
                        addr = self.peer_addrs[endpoint_key][replica_key]
                        local_mr = channel.get_local_mr_key(addr, _KV_CACHE_BUFFER_ID)
                        remote_mr = channel.get_remote_mr_key(addr, _KV_CACHE_BUFFER_ID)
                        converted = [
                            (local_mr, remote_mr, a[3], a[2], a[4])
                            for a in assign_batch
                        ]
                        futures.append(channel.read(addr, converted))
                    else:
                        futures.append(channel.read(assign_batch))

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
    broker_host: str | None = None,
    broker_base_port: int | None = None,
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
        broker_host=broker_host,
        broker_base_port=broker_base_port,
    )
    return _CACHE_CONTEXT
