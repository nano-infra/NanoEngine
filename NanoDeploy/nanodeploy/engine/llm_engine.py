import atexit
import json
import threading
import time
import uuid
from dataclasses import fields
from time import perf_counter
from typing import Any, Dict, List, Literal, Optional, Set

import etcd3
import flatbuffers
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

        # Etcd Discovery & Mesh
        etcd_host, etcd_port = config.etcd_address.split(":")
        self.etcd = etcd3.client(host=etcd_host, port=int(etcd_port))
        self.cluster_id = config.cluster_id
        self.lease = None
        self.active_p2p_links: Set[str] = set()
        self.handshake_watches: Dict[str, Any] = {}

        self._setup_etcd()
        self.discovery_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self.discovery_thread.start()

        atexit.register(self.exit)

    def _setup_etcd(self):
        """Register node in etcd and start keep-alive."""
        logger.info(f"Registering node {self.engine_id} in cluster {self.cluster_id}")
        self.lease = self.etcd.lease(ttl=10)

        builder = flatbuffers.Builder(1024)
        id_off = builder.CreateString(self.engine_id)
        role_off = builder.CreateString(self.config.mode)
        host_off = builder.CreateString(self.config.host)
        status_off = builder.CreateString("initializing")

        EngineInfoModule.Start(builder)
        EngineInfoModule.AddId(builder, id_off)
        EngineInfoModule.AddRole(builder, role_off)
        EngineInfoModule.AddRank(builder, 0)
        EngineInfoModule.AddWorldSize(builder, self.config.attn_world_size)
        EngineInfoModule.AddNumBlocks(builder, self.config.num_kvcache_blocks)
        EngineInfoModule.AddHost(builder, host_off)
        EngineInfoModule.AddPort(builder, self.config.port)
        EngineInfoModule.AddStatus(builder, status_off)
        info_off = EngineInfoModule.End(builder)
        builder.Finish(info_off)

        val = bytes(builder.Output())
        key = f"/nanodeploy/mesh/{self.cluster_id}/nodes/{self.engine_id}"
        self.etcd.put(key, val, lease=self.lease)

        # Start keep-alive thread
        def keep_alive():
            logger.info("Starting etcd lease keep-alive loop")
            while hasattr(self, "etcd"):
                try:
                    self.lease.refresh()
                    time.sleep(3)  # Refresh every 3s (TTL is 10s)
                except Exception as e:
                    logger.error(f"Lease refresh failed: {e}")
                    time.sleep(1)

        self.ka_thread = threading.Thread(target=keep_alive, daemon=True)
        self.ka_thread.start()

    def _update_status(self, status: str):
        """Update node status in etcd."""
        logger.info(f"Updating node status to {status}")
        builder = flatbuffers.Builder(1024)
        id_off = builder.CreateString(self.engine_id)
        role_off = builder.CreateString(self.config.mode)
        host_off = builder.CreateString(self.config.host)
        status_off = builder.CreateString(status)

        EngineInfoModule.Start(builder)
        EngineInfoModule.AddId(builder, id_off)
        EngineInfoModule.AddRole(builder, role_off)
        EngineInfoModule.AddRank(builder, 0)
        EngineInfoModule.AddWorldSize(builder, self.config.attn_world_size)
        EngineInfoModule.AddNumBlocks(builder, self.config.num_kvcache_blocks)
        EngineInfoModule.AddHost(builder, host_off)
        EngineInfoModule.AddPort(builder, self.config.port)
        EngineInfoModule.AddStatus(builder, status_off)
        info_off = EngineInfoModule.End(builder)
        builder.Finish(info_off)

        val = bytes(builder.Output())
        key = f"/nanodeploy/mesh/{self.cluster_id}/nodes/{self.engine_id}"
        self.etcd.put(key, val, lease=self.lease)

    def _watch_loop(self):
        """Watch for new nodes and handshake events."""
        node_prefix = f"/nanodeploy/mesh/{self.cluster_id}/nodes/"
        logger.info(f"Starting discovery scan on prefix: {node_prefix}")

        # 1. Scan for existing nodes
        try:
            existing_nodes = self.etcd.get_prefix(node_prefix)
            count = 0
            for val, meta in existing_nodes:
                count += 1
                key = meta.key.decode("utf-8")
                peer_id = key.split("/")[-1]
                logger.info(f"Scan found node: {peer_id} (Key: {key})")

                if peer_id == self.engine_id:
                    logger.info("Skipping self in scan.")
                    continue

                logger.info(f"Discovered existing peer: {peer_id}")
                peer_info = EngineInfo.GetRootAsEngineInfo(val, 0)
                self._on_peer_online(peer_id, peer_info)
            logger.info(f"Scan completed. Found {count} nodes.")
            self._update_status("ready")
        except Exception as e:
            logger.error(f"Discovery scan failed: {e}")
            import traceback

            traceback.print_exc()

        # 2. Watch for future events
        logger.info("Entering event watch loop...")
        events_iterator, cancel = self.etcd.watch_prefix(node_prefix)
        for event in events_iterator:
            key = event.key.decode("utf-8")
            peer_id = key.split("/")[-1]
            if peer_id == self.engine_id:
                continue
            if isinstance(event, etcd3.events.PutEvent):
                peer_info = EngineInfo.GetRootAsEngineInfo(event.value, 0)
                self._on_peer_online(peer_id, peer_info)
            elif isinstance(event, etcd3.events.DeleteEvent):
                self._on_peer_offline(peer_id)

    def _on_peer_online(self, peer_id: str, peer_info):
        peer_role = peer_info.Role().decode("utf-8")
        my_role = self.config.mode
        if (my_role == "prefill" and peer_role == "decode") or (
            my_role == "decode" and peer_role == "prefill"
        ):
            low_id = min(self.engine_id, peer_id)
            high_id = max(self.engine_id, peer_id)
            h_root = f"/nanodeploy/mesh/{self.cluster_id}/handshake/{low_id}/{high_id}"
            if self.engine_id == low_id:
                logger.info(f"Initiating handshake with peer {peer_id}")
                my_meta = self.p2p_init(
                    peer_id, peer_info.NumBlocks(), peer_info.WorldSize()
                )
                self.etcd.put(f"{h_root}/meta_low", json.dumps(my_meta))
                self._start_handshake_watch(peer_id, f"{h_root}/meta_high", "meta_high")
            else:
                logger.info(f"Waiting for handshake from peer {peer_id}")
                self._start_handshake_watch(peer_id, f"{h_root}/meta_low", "meta_low")

    def _start_handshake_watch(self, peer_id: str, key: str, expected: str):
        if peer_id in self.handshake_watches:
            return

        # Use a polling thread instead of relying solely on etcd watches (which seem flaky)
        logger.info(f"Starting polling for handshake key: {key}")

        def poll_loop():
            while peer_id not in self.active_p2p_links:
                # Safety check for shutdown
                if not hasattr(self, "executor"):
                    return
                try:
                    val, _ = self.etcd.get(key)
                    if val:
                        logger.info(f"Polled handshake key {key} found!")
                        self._handle_handshake_step(
                            peer_id, expected, val.decode("utf-8")
                        )
                        return
                    time.sleep(0.5)
                except Exception as e:
                    # Suppress errors if we are shutting down
                    if not hasattr(self, "executor"):
                        return
                    logger.error(f"Error polling {key}: {e}")
                    time.sleep(1.0)

        t = threading.Thread(target=poll_loop, daemon=True)
        t.start()
        self.handshake_watches[peer_id] = t

    def _convert_keys_to_int(self, obj):
        if isinstance(obj, dict):
            return {
                (
                    int(k) if isinstance(k, str) and k.isdigit() else k
                ): self._convert_keys_to_int(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [self._convert_keys_to_int(x) for x in obj]
        return obj

    def _handle_handshake_step(self, peer_id: str, step: str, meta_json: str):
        # Safety check
        if not hasattr(self, "executor"):
            return

        # Idempotency Check:
        if peer_id in self.active_p2p_links:
            logger.debug(f"Ignoring handshake step for {peer_id} (already connected)")
            return

        logger.info(f"Processing handshake step '{step}' from {peer_id}")
        # ... logic as before ...

        meta = json.loads(meta_json)
        # Fix JSON integer keys
        meta = self._convert_keys_to_int(meta)

        if step == "meta_low":
            node_key = f"/nanodeploy/mesh/{self.cluster_id}/nodes/{peer_id}"
            val, _ = self.etcd.get(node_key)
            if not val:
                logger.error(f"Cannot find node info for {peer_id}")
                return
            peer_info = EngineInfo.GetRootAsEngineInfo(val, 0)
            my_meta = self.p2p_init(
                peer_id, peer_info.NumBlocks(), peer_info.WorldSize()
            )
            low_id, high_id = min(self.engine_id, peer_id), max(self.engine_id, peer_id)
            self.etcd.put(
                f"/nanodeploy/mesh/{self.cluster_id}/handshake/{low_id}/{high_id}/meta_high",
                json.dumps(my_meta),
            )
            self.p2p_connect(peer_id, meta)
            self.active_p2p_links.add(peer_id)
        elif step == "meta_high":
            self.p2p_connect(peer_id, meta)
            self.active_p2p_links.add(peer_id)

        if peer_id in self.handshake_watches:
            self.etcd.cancel_watch(self.handshake_watches[peer_id])
            del self.handshake_watches[peer_id]

    def _on_peer_offline(self, peer_id: str):
        if peer_id in self.active_p2p_links:
            logger.info(f"Peer {peer_id} offline. Cleaning up DLSlime link.")
            try:
                self.executor.p2p_disconnect(peer_id)
            except:
                pass
            self.active_p2p_links.remove(peer_id)

    def exit(self):
        del self.executor

    def update_num_kvcache_blocks(self):
        self.config.num_kvcache_blocks = self.executor.update_kvcache_blocks()
        self.executor.init_rpc_endpoint()

    def get_engine_id(self):
        return self.engine_id

    def get_engine_info(self):
        return {
            "id": self.engine_id,
            "mode": self.config.mode,
            "world_size": self.config.attn_world_size,
            "num_blocks": self.config.num_kvcache_blocks,
        }

    def get_num_kv_blocks(self):
        return self.config.num_kvcache_blocks

    def get_attn_world_size(self):
        return self.config.attn_world_size

    def add_request(self, seqs: Sequence | list[Sequence]):
        if isinstance(seqs, Sequence):
            seqs = [seqs]
        for seq in seqs:
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
        if is_prefill and self.config.mode == "decode":
            if not self.config.dummy_prefill:
                logger.debug("perform migration")
                for seqs in filtered_dp_sp_seqs:
                    for seq in seqs:
                        logger.debug(
                            f"{seq.block_ctx().block_location}, "
                            f"{seq.block_ctx(BlockContextSlot.MIGRATE).block_location}"
                        )
                self.executor.migrate(dp_sp_seqs)
            else:
                for dp_idx, seqs in enumerate(dp_seqs):
                    for seq in seqs:
                        if not self.scheduler.worker_state[dp_idx].may_append(seq, 1):
                            logger.error(
                                "Failed to allocate block for sequence %s during dummy prefill; skipping token append.",
                                getattr(seq, "seq_id", "<unknown>"),
                            )
                            continue
                        seq.append_token(0, BlockContextSlot.ACTIVE)
                for seqs in dp_seqs:
                    for seq in seqs:
                        if seq.metric and seq.metric.num_generated_tokens == 0:
                            seq.metric.record_first_token()
                            seq.metric.num_generated_tokens = 1
        else:
            token_ids = self.executor.run(dp_sp_tp_seqs, is_prefill)[::tp_size]
            post_sch_begin = time.time()
            self.scheduler.postprocess(
                filtered_dp_sp_seqs, token_ids, self.metrics_manager
            )
            post_sch_end = time.time()
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

    def p2p_init(
        self, remote_engine_name: str, num_kv_blocks: int, remote_world_size: int
    ):
        return self.executor.p2p_init(
            remote_engine_name, num_kv_blocks, remote_world_size
        )

    def p2p_connect(
        self, remote_engine_name: str, remote_endpoints_info: list[list[dict]]
    ):
        return self.executor.p2p_connect(remote_engine_name, remote_endpoints_info)

    def wait_for_mesh(self, expected_peers: int, timeout: float = 60.0):
        """Wait until the expected number of P2P links are established."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check number of connected peers in cache context on workers
            # (assuming all workers have the same view of connected peers)
            num_connected = self.executor.collective_rpc("get_num_connected_peers")[0]
            if num_connected >= expected_peers:
                logger.info(f"Mesh ready! Connected peers: {num_connected}")
                return True
            time.sleep(1.0)
        raise TimeoutError(
            f"Mesh not ready after {timeout}s. Connected: {num_connected}/{expected_peers}"
        )

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
