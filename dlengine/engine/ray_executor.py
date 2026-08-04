import copyreg
import importlib
import os
import threading
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import ray
from ray.util.placement_group import placement_group, remove_placement_group

from dlengine._rust.proto import RunnerOut
from dlengine.config import Config
from dlengine.logging import get_logger
from dlengine.worker.model_runner import ModelRunner

logger = get_logger()


def _register_config_module_pickler() -> None:
    """Make ``torch._dynamo.config_utils.ConfigModuleInstance`` picklable.

    torch >= 2.10 ships several module-typed config singletons (e.g.
    ``torch.distributed.config``, ``torch.cuda.config``) whose class is
    ``ConfigModuleInstance`` — a ``ModuleType`` subclass with a
    ``__reduce__`` that refuses pickling. When Ray cloudpickles an actor
    class, it walks transitively reachable modules and trips over these,
    failing with ``cannot pickle 'ConfigModuleInstance' object``.

    The configs are never actually needed on the remote side at unpickle
    time, but Ray's serialize path still has to *get past* them. Register
    a reducer that just re-imports the module by name on the other side.
    """
    try:
        import torch.distributed.config as _probe
    except Exception:
        return

    cls = type(_probe)
    if cls.__name__ != "ConfigModuleInstance":
        return

    def _reduce_config_module(mod):
        return (importlib.import_module, (mod.__name__,))

    copyreg.pickle(cls, _reduce_config_module)


_register_config_module_pickler()


def _collect_dsv4_debug_env() -> dict[str, str] | None:
    """Forward opt-in debug env vars to Ray actors.

    Ray actors do not inherit the driver's environment, so any debugging env
    var must be explicitly forwarded here (applied in ``ModelRunner.__init__``
    via ``os.environ.update`` before CUDA/dist init). Forwards the DeepSeek-V4
    debug knobs plus general debugging vars (synchronous CUDA launches, our
    per-op synchronize probe, and NCCL/MCCL debug logging).
    """
    import os

    _PASSTHROUGH = (
        "CUDA_LAUNCH_BLOCKING",
        "DLENGINE_DEBUG_SYNC",
        "DLENGINE_DEBUG_OPSYNC",
        "DLENGINE_BLACKWELL_DEBUG_SYNC",
        "DLENGINE_CHUNKED_ALLREDUCE",
        "NCCL_DEBUG",
        "NCCL_DEBUG_SUBSYS",
        "MCCL_DEBUG",
        "MCCL_DEBUG_SUBSYS",
        "TORCH_NCCL_BLOCKING_WAIT",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    )
    debug_env = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("DLENGINE_DSV4_DEBUG_") or key in _PASSTHROUGH
    }
    return debug_env or None


from dlengine.engine.ray_utils import get_available_nodes_with_master_first


class RayExecutor:
    """Ray executor. Only support DP+EP+SP Mode"""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.lock = threading.Lock()

        # 1. 初始化 Ray 连接
        # Disable Ray's per-worker log deduplication so that identical log
        # lines from all DP/EP ranks are forwarded separately to the driver.
        # Without this, a hang on a single rank is invisible because Ray
        # collapses the other ranks' identical messages into one line.
        os.environ.setdefault("RAY_DEDUP_LOGS", "0")
        with self.lock:
            ray.init(address=config.ray_address, ignore_reinit_error=True)

        self.workers = []
        self.placement_groups = []
        architecture = (getattr(config.hf_config, "architectures", None) or [""])[0]
        self.worker_runtime_env = None
        if architecture == "KimiK3ForConditionalGeneration":
            # DeepGEMM MegaMoE uses PyTorch symmetric memory, whose ranks must
            # share one CUDA ordinal namespace. Ray's normal one-visible-GPU
            # isolation makes every rank's distinct physical device cuda:0.
            self.worker_runtime_env = {
                "env_vars": {
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                }
            }
        self.worker_debug_env = _collect_dsv4_debug_env()
        if self.worker_debug_env:
            logger.info(
                "Forwarding DeepSeek-V4 debug env to Ray workers: "
                f"{sorted(self.worker_debug_env)}"
            )
        assert config.attn_world_size == config.ffn_world_size

        # Check if running under NanoOps orchestration
        nanoops_pg_id = os.getenv("NANOOPS_PLACEMENT_GROUP_ID")

        if nanoops_pg_id:
            # NanoOps orchestration mode: use pre-created placement group
            logger.info(
                f"Using pre-created placement group from NanoOps: {nanoops_pg_id}"
            )
            self._init_with_existing_pg(nanoops_pg_id)
            self.external_pg = True  # Mark as externally managed
        else:
            # Manual launch mode: create placement groups as before
            logger.info("Creating placement groups (manual launch mode)")
            self._init_with_new_pg()
            self.external_pg = False

        logger.info("All workers scheduled successfully.")

    def _init_distributed_workers(self, master_address: str | None = None) -> None:
        """Discover rank 0's address and initialize every worker together."""
        if master_address is None:
            worker_ip, free_port = ray.get(self.workers[0].get_node_info.remote())
            master_address = f"{worker_ip}:{free_port}"
            logger.info("Discovered distributed master at %s", master_address)
        else:
            logger.info("Using configured distributed master at %s", master_address)

        self.config.master_address = master_address
        # dist.init_process_group is collective: submit all actor calls before
        # waiting for any one of them.
        ray.get([w.init_dist.remote(master_address) for w in self.workers])
        logger.info("All workers completed distributed init.")

    def _init_with_existing_pg(self, pg_id_hex: str):
        """Initialize workers using pre-created placement group from NanoOps.

        When running inside a Ray job with placement_group_id in runtime_env,
        all tasks are automatically scheduled to that placement group.
        We don't need to explicitly pass the PG to workers.

        Args:
            pg_id_hex: Placement group ID in hex format
        """
        logger.info(
            f"Scheduling {self.config.world_size} workers using Ray job's placement group"
        )

        # Create every worker first with distributed initialization deferred;
        # rank 0's actual placement determines the rendezvous address.
        for rank in range(self.config.world_size):
            worker = ModelRunner.options(runtime_env=self.worker_runtime_env).remote(
                self.config,
                rank,
                defer_dist_init=True,
                debug_env=self.worker_debug_env,
            )
            self.workers.append(worker)

        # The external placement group decides where rank 0 lives, so always
        # derive the rendezvous endpoint from that worker.
        self._init_distributed_workers()

        # No placement group object to store - managed by Ray job runtime
        self.placement_groups = []

    def _init_with_new_pg(self):
        """Initialize workers by creating new placement groups (existing logic)."""
        # Define the maximum number of workers packed into one node-sized PG.
        workers_per_node = 8

        # A configured master is a legacy node-placement override. With the
        # default None, PGs carry no node hint and Ray chooses available nodes.
        node_ids = None
        world_size = self.config.world_size
        if self.config.master_address is not None:
            required_on_master = min(world_size, workers_per_node)
            nodes = get_available_nodes_with_master_first(
                self.config.master_address, required_gpus=required_on_master
            )
            node_ids = [node["NodeID"] for node in nodes]

        # When world size exceeds a single node, it must be a multiple of
        # workers_per_node so each node is fully packed.
        if world_size > workers_per_node and world_size % workers_per_node != 0:
            raise ValueError(
                f"world_size ({world_size}) must be a "
                f"multiple of {workers_per_node} when larger than {workers_per_node}"
            )

        # 4. 计算需要多少个节点
        num_nodes_needed = (world_size + workers_per_node - 1) // workers_per_node
        if node_ids is not None and num_nodes_needed > len(node_ids):
            raise ValueError(
                f"insufficient resources, {num_nodes_needed} on demand，but only find {len(node_ids)} nodes"
            )

        # 5. 为每个目标节点创建 Placement Group，并调度相应的 workers
        for node_idx in range(num_nodes_needed):
            target_node_id = node_ids[node_idx] if node_ids is not None else None
            logger.info("--- scheduling worker group %s ---", node_idx)

            start_rank = node_idx * workers_per_node
            end_rank = min(start_rank + workers_per_node, world_size)
            num_workers_on_node = end_rank - start_rank

            # PG names must be globally unique. Include the engine id so several
            # engines (e.g. prefill + decode for PD disaggregation) can co-locate
            # on the same node without colliding on a fixed ``pg-node-<id>`` name.
            # Fall back to the executor's object id if engine_id is unset.
            engine_tag = getattr(self.config, "engine_id", None) or id(self)
            pg_options = dict(
                bundles=[{"CPU": 0.1, "GPU": 1.0} for _ in range(num_workers_on_node)],
                strategy="STRICT_PACK",
                name=f"pg-workers-{node_idx}-{engine_tag}",
            )
            if target_node_id is not None:
                pg_options["_soft_target_node_id"] = target_node_id
            pg = placement_group(**pg_options)

            ray.get(pg.ready())

            self.placement_groups.append(pg)

            for rank in range(start_rank, end_rank):
                worker = ModelRunner.options(
                    placement_group=pg,
                    runtime_env=self.worker_runtime_env,
                ).remote(
                    self.config,
                    rank,
                    defer_dist_init=True,
                    debug_env=self.worker_debug_env,
                )
                self.workers.append(worker)

        self._init_distributed_workers(self.config.master_address)

    def __del__(self):
        if hasattr(self, "workers") and self.workers:
            logger.info(f"Terminating {len(self.workers)} workers...")
            for worker in self.workers:
                try:
                    ray.kill(worker)
                    logger.debug(f"Worker {worker} terminated successfully.")
                except Exception as e:
                    logger.warning(f"Failed to terminate worker {worker}: {e}")
            del self.workers

        # Only remove placement groups if we created them (not externally managed)
        if hasattr(self, "placement_groups") and self.placement_groups:
            if hasattr(self, "external_pg") and self.external_pg:
                logger.info(
                    "Skipping placement group removal (externally managed by NanoOps)"
                )
            else:
                for pg in self.placement_groups:
                    try:
                        remove_placement_group(pg)
                    except Exception as e:
                        logger.error(f"Warning: Failed to remove Placement Group: {e}")

        logger.debug("Ray Executor deconstructed")

    def collective_rpc(
        self,
        method: str,
        args: tuple | None = None,
        kwargs: dict | None = None,
        timeout: float | None = None,
    ):
        """Collective rpc."""
        if args is None:
            args = tuple()
        if kwargs is None:
            kwargs = dict()

        return ray.get(
            [
                getattr(worker, method).remote(*args, **kwargs)
                for worker in self.workers
            ],
            timeout=timeout,
        )

    def migrate_batch_bytes(
        self,
        batch_bytes: List[bytes],
        timeout: float | None = None,
    ) -> list[int]:
        return ray.get(
            [
                getattr(worker, "migrate_from_bytes").remote(b)
                for b, worker in zip(batch_bytes, self.workers)
            ],
            timeout=timeout,
        )

    def l3_load(
        self,
        per_worker_pairs: List[List[tuple]],
        timeout: float | None = None,
    ) -> list[int]:
        """Load (hash, block_id) blocks from 3FS into each worker's GPU.

        ``per_worker_pairs[i]`` is dispatched to ``self.workers[i]`` (the L3
        PoC requires attention_sp == attention_tp == 1, so worker index == the
        scheduler's worker_state/dp index).
        """
        return ray.get(
            [
                getattr(w, "l3_load_blocks").remote(pairs)
                for pairs, w in zip(per_worker_pairs, self.workers)
            ],
            timeout=timeout,
        )

    def l3_store(
        self,
        per_worker_pairs: List[List[tuple]],
        timeout: float | None = None,
    ) -> list[int]:
        """Persist (hash, block_id) GPU blocks to 3FS, per worker."""
        return ray.get(
            [
                getattr(w, "l3_store_blocks").remote(pairs)
                for pairs, w in zip(per_worker_pairs, self.workers)
            ],
            timeout=timeout,
        )

    def swap_out_blocks_to_host(
        self,
        per_worker_tasks: List[List[tuple]],
        timeout: float | None = None,
    ) -> list[int]:
        return ray.get(
            [
                getattr(w, "swap_out_blocks_to_host").remote(tasks)
                for tasks, w in zip(per_worker_tasks, self.workers)
            ],
            timeout=timeout,
        )

    def swap_in_blocks_from_host(
        self,
        per_worker_tasks: List[List[tuple]],
        timeout: float | None = None,
    ) -> list[int]:
        return ray.get(
            [
                getattr(w, "swap_in_blocks_from_host").remote(tasks)
                for tasks, w in zip(per_worker_tasks, self.workers)
            ],
            timeout=timeout,
        )

    def l3_stats(self) -> list[dict]:
        return self.collective_rpc("l3_stats")

    def run_batch_bytes_async(self, batch_bytes: List[bytes], is_prefill: bool) -> dict:
        """Submit serialized RunnerIn bytes without waiting."""
        # Bytes sent to the runners this forward (serialized RunnerIn input).
        self.last_run_request_bytes = sum(len(b) for b in batch_bytes)
        ray_futures = [
            getattr(worker, "run_from_bytes").remote(b, is_prefill)
            for b, worker in zip(batch_bytes, self.workers)
        ]
        return {"futures": ray_futures}

    def run_wait(self, handle: dict) -> list[list[list[int]]]:
        return ray.get(handle["futures"])

    def run_wait_runner_outs(self, handle: dict) -> list[RunnerOut]:
        return [self._to_runner_out(result) for result in ray.get(handle["futures"])]

    def run(
        self,
        batch_bytes: List[bytes],
        is_prefill: bool,
        timeout: float | None = None,
    ) -> list[list[list[int]]]:
        handle = self.run_batch_bytes_async(batch_bytes, is_prefill)
        return ray.get(
            handle["futures"],
            timeout=timeout,
        )

    @staticmethod
    def _to_runner_out(result) -> RunnerOut:
        if isinstance(result, tuple):
            token_ids, logprobs = result
        else:
            token_ids, logprobs = result, None
        return RunnerOut(token_ids, logprobs, 0)

    def update_kvcache_blocks(self):
        num_cache_blocks = min(self.collective_rpc("num_kvcache_blocks"))
        num_host_cache_blocks = min(self.collective_rpc("num_host_kvcache_blocks"))
        logger.info(f"Set {num_cache_blocks=}")
        if num_host_cache_blocks > 0:
            logger.info(f"Set {num_host_cache_blocks=}")
        self.config.num_host_kvcache_blocks = num_host_cache_blocks
        self.collective_rpc("allocate_kvcache", (num_cache_blocks,))
        return num_cache_blocks

    def get_peer_agent_addrs(self) -> list[str]:
        """Get peer agent addresses from all workers."""
        return self.collective_rpc("get_peer_agent_addr")

    def p2p_disconnect(self, remote_name: str):
        return self.collective_rpc("p2p_disconnect", (remote_name,))

    def gather_free_mem(self):
        """Get free memory."""
        return self.collective_rpc("get_free_mem")

    def get_cache_block_size(self, block_size, world_size):
        """Get cache block size."""
        return self.collective_rpc("get_cache_block_size", (block_size, world_size))

    def allocate_kvcache(self, num_block_per_rank):
        """Allocate kv cache."""
        return self.collective_rpc("allocate_gpu_cache", args=(num_block_per_rank,))

    def init_cudagraph_buffer(self):
        """Initialize cuda graph buffer."""
        return self.collective_rpc("init_cudagraph_buffer")

    def capture_cudagraph(self):
        """Capture cuda graph."""
        return self.collective_rpc("capture_cudagraph")
