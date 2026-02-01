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

# PeerAgent path: buffer ID for kv_cache registration
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

    # PeerAgent: host and base port for RDMA peer agent
    peer_agent_host: str | None = None
    peer_agent_base_port: int | None = None

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
        self._peer_agent = None
        self._peer_agent_addr: str | None = None
        self._connected_peers: set[str] = set()  # track connected peer addresses

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
        # PeerAgent path: initialize RdmaPeerAgent and register kv_cache buffer
        if self.peer_agent_host is not None and self.peer_agent_base_port is not None:
            start_peer_agent_fn = getattr(dlslime, "start_peer_agent", None)
            if callable(start_peer_agent_fn):
                rank = dist.get_rank()
                bind_addr = f"{self.peer_agent_host}:{self.peer_agent_base_port + rank}"
                self._peer_agent = start_peer_agent_fn(bind_addr)
                self._peer_agent_addr = self._peer_agent.client_addr
                kv_size = self.kv_cache.numel() * self.kv_cache.itemsize
                self._peer_agent.register_buffer(
                    _KV_CACHE_BUFFER_ID,
                    self.kv_cache.data_ptr(),
                    kv_size,
                )
                logger.info(
                    f"PeerAgent started on {bind_addr}, registered kv_cache buffer"
                )

    def get_peer_agent_addr(self) -> str | None:
        """Return the local peer agent address for this rank."""
        return self._peer_agent_addr

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
            logger.info(f"P2P Link to {remote_engine_id} disconnected and cleared.")

    def migrate(
        self, seqs: list[Sequence], peer_endpoints: dict[str, list[str]] = None
    ):
        """Migrate KV cache blocks from local to remote engine using PeerAgent.

        This method uses lazy connection: it will connect to remote peer
        on-demand based on peer_endpoints dict (engine_id -> list of peer_agent addresses).

        Args:
            seqs: List of sequences to migrate
            peer_endpoints: dict mapping engine_id to list of peer addresses per rank
        """
        if self._peer_agent is None:
            logger.error("migrate called but PeerAgent not initialized")
            return

        if peer_endpoints is None:
            peer_endpoints = {}

        assigns = defaultdict(lambda: defaultdict(list))
        sp_idx = get_dist_context().attn_sp_rank

        # Collect all unique remote endpoints from peer_endpoints dict and ensure connections
        remote_endpoints_to_connect: dict[str, str] = {}  # remote_addr -> engine_id
        for seq in seqs:
            migrate_ctx = seq.block_ctx(BlockContextSlot.MIGRATE)
            engine_id = migrate_ctx.engine_id
            endpoints = peer_endpoints.get(engine_id, [])
            if endpoints:
                for addr in endpoints:
                    if addr and addr not in self._connected_peers:
                        self.num_remote_kvcache_blocks[engine_id] = (
                            migrate_ctx.num_kvcache_blocks
                        )
                        remote_endpoints_to_connect[addr] = engine_id

        # Lazy connect to all required remote peers
        for remote_addr, engine_id in remote_endpoints_to_connect.items():
            try:
                logger.info(
                    f"Lazy connecting to peer {remote_addr} for engine {engine_id}"
                )
                self._peer_agent.connect(remote_addr, self.selected_nic)
                self._connected_peers.add(remote_addr)
                logger.info(f"Connected to peer {remote_addr}")
            except Exception as e:
                logger.error(f"Failed to connect to peer {remote_addr}: {e}")
                raise

        # Build assignment list for each remote endpoint
        for seq in seqs:
            migrate_ctx = seq.block_ctx(BlockContextSlot.MIGRATE)
            active_ctx = seq.block_ctx(BlockContextSlot.ACTIVE)
            engine_id = migrate_ctx.engine_id
            endpoints = peer_endpoints.get(engine_id, [])
            if not endpoints:
                logger.warning(
                    f"Sequence {seq.seq_id} has no endpoints for engine {engine_id}"
                )
                continue

            for remote_block_idx, source_block_idx in zip(
                migrate_ctx.block_location,
                active_ctx.block_location,
            ):
                for kv_idx in range(self.kv_cache.size(0)):
                    for layer_idx in range(self.num_hidden_layers):
                        if source_block_idx[0] == sp_idx:
                            remote_rank = (
                                seq.dp_idx(BlockContextSlot.MIGRATE)
                                * migrate_ctx.attention_sp
                                + remote_block_idx[0]
                            )
                            # Get remote peer address for this rank
                            if remote_rank < len(endpoints):
                                remote_addr = endpoints[remote_rank]
                            else:
                                logger.error(
                                    f"remote_rank {remote_rank} >= len(endpoints) {len(endpoints)}"
                                )
                                continue

                            assignment = (
                                remote_addr,
                                kv_idx,
                                layer_idx,
                                remote_block_idx[1],
                                source_block_idx[1],
                            )
                            assigns[engine_id][remote_addr].append(assignment)

        # Execute RDMA reads using PeerAgent
        futures = []
        stream = torch.cuda.current_stream()
        for engine_id, addr_assigns in assigns.items():
            for remote_addr, assign_batch in addr_assigns.items():
                # PeerAgent uses (local_mr_key, remote_mr_key, local_off, remote_off, length)
                local_mr = self._peer_agent.get_local_mr_key(
                    remote_addr, _KV_CACHE_BUFFER_ID
                )
                remote_mr = self._peer_agent.get_remote_mr_key(
                    remote_addr, _KV_CACHE_BUFFER_ID
                )

                rdma_assigns = []
                for (
                    addr,
                    kv_idx,
                    layer_idx,
                    remote_block_idx,
                    source_block_idx,
                ) in assign_batch:
                    local_off = self.local_kv_stride(
                        kv_idx, layer_idx, source_block_idx
                    )
                    remote_off = self.remote_kv_stride(
                        kv_idx, layer_idx, remote_block_idx, engine_id
                    )
                    print(
                        f"{kv_idx=}, {layer_idx=}, {remote_block_idx=}, {source_block_idx=}, {local_off=}, {remote_off=}, {self.block_stride(1)=}"
                    )
                    length = self.block_stride(1)
                    rdma_assigns.append(
                        (local_mr, remote_mr, remote_off, local_off, length)
                    )

                print(f"{self.num_remote_kvcache_blocks[engine_id]=}")
                future = self._peer_agent.read(remote_addr, rdma_assigns)
                futures.append(future)

        # Wait for all RDMA operations to complete
        for future in futures:
            future.wait()


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
    peer_agent_host: str | None = None,
    peer_agent_base_port: int | None = None,
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
        peer_agent_host=peer_agent_host,
        peer_agent_base_port=peer_agent_base_port,
    )
    return _CACHE_CONTEXT
