import os

# Set KV cache export/verify environment variables for LLMComponent process
# These need to be set here because ModelRunner is a different process
os.environ.setdefault("KVCACHE_EXPORT_ENABLED", "1")
os.environ.setdefault(
    "KVCACHE_EXPORT_DIR", "/mnt/nvme1n1/ml_research/majinming/src/NanoInfra/NanoDeploy/"
)
os.environ.setdefault("KVCACHE_VERIFY_ENABLED", "1")

import atexit
import json
import threading
import time
import uuid
from dataclasses import fields
from time import perf_counter
from typing import Any, Dict, List, Literal, Optional, Set

# Cache TTL for peer_endpoints from NanoCtrl (seconds)
# Engine registration rarely changes, use longer TTL to reduce list_engines calls
_PEER_ENDPOINTS_CACHE_TTL = 60.0

import flatbuffers
import httpx
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanodeploy._cpp import BlockContextSlot
from nanodeploy.config import Config
from nanodeploy.engine.scheduler import Scheduler
from nanodeploy.engine.sequence import Sequence, SequenceStatus
from nanodeploy.fbs import EngineInfo as EngineInfoModule
from nanodeploy.fbs.EngineInfo import EngineInfo
from nanodeploy.logging import get_logger, set_log_level
from nanodeploy.metrics import MetricsManager

logger = get_logger()


class LLMEngine:
    def __init__(self, config: Config):
        self.engine_id = str(uuid.uuid4())

        self.config = config
        self.config.engine_id = self.engine_id

        # Set log level globally first
        if self.config.log_level:
            set_log_level(self.config.log_level)

        self.ps = []
        self.events = []

        from nanodeploy.engine.ray_executor import RayExecutor

        self.executor = RayExecutor(config=config)
        self.update_num_kvcache_blocks()

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id

        self.scheduler = Scheduler(config)
        logger.info(
            f"Initialized Scheduler with RoutingStrategy: {self.scheduler.routing_strategy}"
        )
        self.metrics_manager = MetricsManager()

        # Register engine with NanoCtrl if configured
        # Delay registration until executor is fully initialized
        self._nanoctrl_registered = False
        self._heartbeat_stop_event = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._peer_endpoints_cache: Optional[tuple[float, Dict[str, List[str]]]] = None
        if config.nanoctrl_address:
            # Register after executor is ready (peer_addrs will be available)
            self._register_with_nanoctrl()

        atexit.register(self.exit)

    def exit(self):
        # Stop heartbeat thread
        if hasattr(self, "_heartbeat_stop_event"):
            self._heartbeat_stop_event.set()
        if hasattr(self, "_heartbeat_thread") and self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2.0)

        # Unregister from NanoCtrl before exiting
        if hasattr(self, "_nanoctrl_registered") and self._nanoctrl_registered:
            self._unregister_from_nanoctrl()
        del self.executor

    def _register_with_nanoctrl(self):
        """Register engine information with NanoCtrl control plane."""
        if not self.config.nanoctrl_address:
            return

        try:
            # Get peer agent addresses (may be empty initially, but that's ok)
            peer_addrs = self.get_peer_agent_addrs()
            logger.info(
                f"Registering engine {self.engine_id} with {len(peer_addrs)} peer addresses"
            )

            # Get engine info
            engine_info = {
                "id": self.engine_id,
                "role": self.config.mode,
                "rank": 0,
                "world_size": self.config.attn_world_size,
                "num_blocks": self.config.num_kvcache_blocks,
                "host": self.config.host,
                "port": self.config.port,
                "status": "ready",
                "peer_addrs": peer_addrs,
            }

            # Prepare registration payload
            payload = {
                "engine_id": engine_info["id"],
                "role": engine_info["role"],
                "world_size": engine_info["world_size"],
                "num_blocks": engine_info["num_blocks"],
                "host": engine_info["host"],
                "port": engine_info["port"],
                "peer_addrs": engine_info["peer_addrs"],
            }

            url = f"http://{self.config.nanoctrl_address}/register_engine"
            logger.info(
                f"Registering engine with NanoCtrl at {url}, payload: {payload}"
            )

            # Use sync client since this is called during init
            # Disable proxy to avoid SOCKS proxy issues
            with httpx.Client(timeout=10.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                if data.get("status") == "ok":
                    self._nanoctrl_registered = True
                    logger.info(
                        f"Successfully registered engine {payload['engine_id']} with NanoCtrl"
                    )
                    # Start heartbeat thread after successful registration
                    self._start_heartbeat()
                else:
                    logger.error(
                        f"Failed to register engine: {data.get('message', 'Unknown error')}"
                    )
                    logger.error(f"Response data: {data}")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error registering engine with NanoCtrl: {e}")
            logger.error(
                f"Response: {e.response.text if e.response else 'No response'}"
            )
        except Exception as e:
            logger.error(f"Error registering engine with NanoCtrl: {e}", exc_info=True)
            # Don't raise - allow engine to continue even if registration fails

    def _unregister_from_nanoctrl(self):
        """Unregister engine from NanoCtrl control plane."""
        if not self.config.nanoctrl_address:
            return

        try:
            payload = {"engine_id": self.engine_id}
            url = f"http://{self.config.nanoctrl_address}/unregister_engine"
            logger.info(f"Unregistering engine {self.engine_id} from NanoCtrl")

            # Disable proxy to avoid SOCKS proxy issues
            with httpx.Client(timeout=5.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                if data.get("status") == "ok":
                    logger.info(
                        f"Successfully unregistered engine {self.engine_id} from NanoCtrl"
                    )
                else:
                    logger.warning(
                        f"Failed to unregister engine: {data.get('message', 'Unknown error')}"
                    )
        except Exception as e:
            logger.error(f"Error unregistering engine from NanoCtrl: {e}")
            # Don't raise - exit should continue even if unregistration fails

    def _start_heartbeat(self):
        """Start heartbeat thread to keep engine registration alive."""
        if not self.config.nanoctrl_address or not self._nanoctrl_registered:
            return

        # Stop existing heartbeat thread if any
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return

        self._heartbeat_stop_event.clear()

        def heartbeat_loop():
            """Heartbeat loop: send heartbeat every 15 seconds."""
            while not self._heartbeat_stop_event.wait(15.0):
                try:
                    self._heartbeat_to_nanoctrl()
                except Exception as e:
                    logger.error(f"Error in heartbeat loop: {e}", exc_info=True)

        self._heartbeat_thread = threading.Thread(
            target=heartbeat_loop, name=f"heartbeat-{self.engine_id}", daemon=True
        )
        self._heartbeat_thread.start()
        logger.info(
            f"Started heartbeat thread for engine {self.engine_id} (interval: 15s)"
        )

    def _heartbeat_to_nanoctrl(self):
        """Send heartbeat to NanoCtrl to refresh TTL."""
        if not self.config.nanoctrl_address:
            return

        try:
            payload = {"engine_id": self.engine_id}
            url = f"http://{self.config.nanoctrl_address}/heartbeat_engine"

            # Use sync client with short timeout
            with httpx.Client(timeout=5.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()

                if data.get("status") == "ok":
                    logger.debug(f"Heartbeat successful for engine {self.engine_id}")
                elif data.get("status") == "not_found":
                    # Engine not found, re-register
                    logger.warning(
                        f"Engine {self.engine_id} not found in NanoCtrl, re-registering..."
                    )
                    self._nanoctrl_registered = False
                    self._register_with_nanoctrl()
                else:
                    logger.warning(
                        f"Heartbeat failed for engine {self.engine_id}: {data.get('message', 'Unknown error')}"
                    )
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error in heartbeat: {e}")
        except Exception as e:
            logger.error(f"Error sending heartbeat: {e}", exc_info=True)

    def _fetch_peer_endpoints_from_nanoctrl(self) -> Dict[str, List[str]]:
        """Query peer_endpoints (engine_id -> peer_addrs) from NanoCtrl list_engines, with caching."""
        now = time.time()
        if self._peer_endpoints_cache is not None:
            cached_at, cached = self._peer_endpoints_cache
            if now - cached_at < _PEER_ENDPOINTS_CACHE_TTL:
                logger.debug(
                    f"Using cached peer_endpoints (age={now - cached_at:.1f}s)"
                )
                return cached

        if not self.config.nanoctrl_address:
            logger.warning(
                "nanoctrl_address not configured, peer_endpoints will be empty"
            )
            return {}

        try:
            url = f"http://{self.config.nanoctrl_address}/list_engines"
            with httpx.Client(timeout=5.0) as client:
                response = client.post(url, json={})
                response.raise_for_status()
                data = response.json()
                if data.get("status") != "ok":
                    logger.warning(
                        f"list_engines returned status: {data.get('status')}"
                    )
                    return {}

                engines = data.get("engines", [])
                peer_endpoints: Dict[str, List[str]] = {}
                for eng in engines:
                    engine_id = eng.get("id")
                    peer_addrs = eng.get("peer_addrs", [])
                    if engine_id and peer_addrs:
                        peer_endpoints[engine_id] = peer_addrs

                self._peer_endpoints_cache = (now, peer_endpoints)
                logger.info(
                    f"Fetched peer_endpoints from NanoCtrl: {list(peer_endpoints.keys())}"
                )
                return peer_endpoints
        except Exception as e:
            logger.error(f"Error fetching peer_endpoints from NanoCtrl: {e}")
            if self._peer_endpoints_cache is not None:
                return self._peer_endpoints_cache[1]
            return {}

    def update_num_kvcache_blocks(self):
        self.config.num_kvcache_blocks = self.executor.update_kvcache_blocks()
        self.executor.init_rpc_endpoint()

    def get_engine_id(self):
        return self.engine_id

    def get_num_kv_blocks(self):
        return self.config.num_kvcache_blocks

    def get_attn_world_size(self):
        return self.config.attn_world_size

    def get_peer_agent_addrs(self) -> list[str]:
        """Get peer agent addresses from all workers."""
        return self.executor.get_peer_agent_addrs()

    def add_request(self, seqs: Sequence | list[Sequence]):
        if isinstance(seqs, Sequence):
            seqs = [seqs]
        for seq in seqs:
            # Debug: log received sequence info
            if self.config.mode == "decode":
                logger.info(
                    f"[DEBUG] Decode engine received seq {seq.seq_id}: last_token={seq.last_token}, num_tokens={seq.num_tokens}, token_ids_len={len(seq.token_ids)}, token_ids_last10={seq.token_ids[-10:] if seq.token_ids else []}"
                )
            seq.metric = self.metrics_manager.create_sequence_metric(
                seq.seq_id, seq.num_prompt_tokens
            )
            self.scheduler.add(seq)

    def free_to_be_migrated(self, seqs: Sequence | list[Sequence]):
        self.scheduler.free_to_be_migrated(seqs)

    def step(self):
        dp_size = self.config.attention_dp
        sp_size = self.config.attention_sp
        tp_size = self.config.attention_tp
        sch_begin = time.time()
        sch_res = self.scheduler.schedule()
        dp_seqs = sch_res.dp_seqs
        is_prefill = sch_res.is_prefill
        dp_sp_seqs = sch_res.dp_sp_seqs
        filtered_dp_sp_seqs = sch_res.filtered_dp_sp_seqs
        total_running = sum(len(seqs) for seqs in dp_seqs)
        total_waiting = len(self.scheduler.waiting)
        total_waiting_migration = len(self.scheduler.waiting_migration)
        self.metrics_manager.server_metric.update_running_requests(total_running)
        self.metrics_manager.server_metric.update_waiting_requests(total_waiting)
        self.metrics_manager.server_metric.update_waiting_migration_requests(
            total_waiting_migration
        )

        if self.scheduler.waiting_migration:
            logger.info(f"{self.scheduler.waiting_migration[0].num_tokens=}")

        dp_sp_tp_seqs = [seqs for seqs in dp_sp_seqs for _ in range(tp_size)]

        dp_sp_tp_seqs = [seqs for seqs in dp_sp_seqs for _ in range(tp_size)]
        # dp_batch_sizes = [len(seqs) for seqs in dp_seqs]
        sp_batch_sizes = [
            [
                len(filtered_dp_sp_seqs[dp_idx * sp_size + sp_idx])
                for sp_idx in range(sp_size)
            ]
            for dp_idx in range(dp_size)
        ]

        sp_send_counts = sch_res.sp_send_counts
        sp_recv_counts = sch_res.sp_recv_counts
        # sp_comm_matrix = sch_res.sp_comm_matrix
        sp_q_matrix = sch_res.sp_q_matrix
        # sp_res_matrix = sch_res.sp_res_matrix

        # Update metrics with raw counts
        self.metrics_manager.server_metric.update_sp_stats(
            sp_send_counts, sp_recv_counts
        )

        waiting_head_blocks = sch_res.waiting_head_blocks
        waiting_total_blocks = sch_res.waiting_total_blocks
        self.metrics_manager.server_metric.update_waiting_blocks(
            waiting_head_blocks, waiting_total_blocks
        )

        logger.debug(
            {
                "mode": "prefill" if is_prefill else "decode",
                # "dp_batch_sizes": dp_batch_sizes,
                "sp_batch_sizes": sp_batch_sizes,
                "sp_send_counts": sp_send_counts,
                "sp_recv_counts": sp_recv_counts,
                "waiting_head_blocks": waiting_head_blocks,
                "waiting_total_blocks": waiting_total_blocks,
                # "sp_comm_matrix": sp_comm_matrix,
                "sp_q_matrix": sp_q_matrix,
                # "sp_res_matrix": sp_res_matrix,
                "free_blocks": [
                    [
                        len(worker_state.block_manager[i].free_block_ids)
                        for i in range(self.scheduler.attention_sp)
                    ]
                    for worker_state in self.scheduler.worker_state
                ],
            }
        )

        sch_end = time.time()
        post_sch_begin = 0
        post_sch_end = 0

        # Run prefill to populate KV cache (or skip for decode engine receiving prefill request)
        if not (is_prefill and self.config.mode == "decode"):
            # Normal execution: prefill engine runs prefill, or decode engine runs decode
            token_ids = self.executor.run(dp_sp_tp_seqs, is_prefill)[::tp_size]
            post_sch_begin = time.time()
            self.scheduler.postprocess(
                filtered_dp_sp_seqs, token_ids, self.metrics_manager
            )
            post_sch_end = time.time()

        else:
            # PD disaggregation: decode engine receives prefill request
            # DO NOT run prefill on decode engine - KV cache will be migrated from prefill engine
            logger.info(
                f"Decode engine receiving prefill request, skipping local prefill execution"
            )
            post_sch_begin = time.time()
            post_sch_end = time.time()

            # Perform migration from prefill engine
            logger.info(f"Performing migration from prefill engine")
            target_engine_ids = set()
            for seqs in dp_sp_seqs:
                for seq in seqs:
                    if getattr(seq, "is_to_be_migrated", False):
                        ctx = seq.block_ctx(BlockContextSlot.MIGRATE)
                        eid = getattr(ctx, "engine_id", None)
                        if eid:
                            target_engine_ids.add(eid)
            logger.info(f"Target engine IDs for migration: {target_engine_ids}")
            ensure_p2p = getattr(self, "ensure_p2p_connected", None)
            if callable(ensure_p2p):
                for eid in target_engine_ids:
                    ensure_p2p(eid)
            # Query peer_endpoints from NanoCtrl (Redis) and cache
            peer_endpoints = self._fetch_peer_endpoints_from_nanoctrl()
            logger.info(
                f"Calling executor.migrate with peer_endpoints: {peer_endpoints}"
            )
            self.executor.migrate(dp_sp_seqs, peer_endpoints=peer_endpoints)
        outputs = []
        num_tokens = 0

        for dp_idx, seqs in enumerate(dp_seqs):
            num_tokens_in_dp = sum(len(seq) for seq in seqs)
            self.metrics_manager.server_metric.update_token_usage(
                dp_idx, num_tokens_in_dp
            )

        for seqs in dp_seqs:
            num_tokens += (
                sum(len(seq) for seq in seqs)
                if is_prefill
                else -len(seqs) * self.config.loop_count
            )
            for seq in seqs:
                if seq.is_finished or seq.is_to_be_migrated:
                    if seq.is_finished or seq.is_to_be_migrated:
                        self.metrics_manager.complete_sequence(seq.seq_id)
                    outputs.append(seq)
        return (
            dp_seqs,
            outputs,
            num_tokens,
            sum(len(seqs) for seqs in dp_seqs),
            (sch_end - sch_begin) * 1000,
            (post_sch_end - post_sch_begin) * 1000,
        )

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        use_tqdm: bool = True,
        log_metrics_interval: int = 10,
    ) -> list[Sequence]:
        num_reqs = len(self.scheduler.waiting)
        if use_tqdm:
            pbar = tqdm(total=num_reqs, desc="Generating", dynamic_ncols=True)

        finished_seqs = []
        prefill_throughput = decode_throughput = 0.0
        step_count = 0

        while not self.is_finished():
            t = perf_counter()
            dp_seqs, output, num_tokens, bs, sch_latency, post_sch_latency = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                    self.metrics_manager.server_metric.record_prefill_throughput(
                        num_tokens, (perf_counter() - t)
                    )
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                    self.metrics_manager.server_metric.record_decode_throughput(
                        -num_tokens, (perf_counter() - t)
                    )
                itl = (perf_counter() - t) * 1000 / self.config.loop_count
                pbar.set_postfix(
                    {
                        "bs": f"{bs}",
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                        "itl": f"{itl:.2f}ms",
                        "sch_ovhd": f"{sch_latency:.2f}ms",
                        "post_sch_ovhd": f"{post_sch_latency:.2f}ms",
                    }
                )
            for seq in output:
                finished_seqs.append(seq)
                if use_tqdm:
                    pbar.update(1)
        if use_tqdm:
            pbar.close()

        logger.info("=" * 60)
        logger.info("Final Server Metrics Summary")
        logger.info("=" * 60)
        self.metrics_manager.log_server_metrics(include_detailed=True)
        summary = self.metrics_manager.get_server_summary()
        for key, value in summary.items():
            if value is not None:
                logger.info(f"  {key}: {value}")
        logger.info("=" * 60)

        return finished_seqs
