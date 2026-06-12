import os
import uuid

import ray

from dlengine._cpp import Sequence, serialize_migrate_batch, serialize_run_batch
from dlengine.config import Config
from dlengine.engine.dlslime_protocol import (
    decode_run_result,
    encode_run_request,
    ModelRunnerRpcService,
    unpack_reply_header,
)
from dlengine.engine.ray_executor import RayExecutor
from dlengine.logging import get_logger

logger = get_logger()


class DLSLimeExecutor(RayExecutor):
    """Ray-managed workers with DLSLime for hot-path run/migrate calls."""

    def __init__(self, config: Config) -> None:
        try:
            import dlslime
            from dlslime.rpc import proxy, wait_all
        except ImportError as exc:
            raise ImportError(
                "executor_backend='dlslime' requires the optional 'dlslime' "
                "dependency. Install DLEngine with the 'dlslime' extra."
            ) from exc

        if not config.ctrl_address:
            raise ValueError(
                "executor_backend='dlslime' requires ctrl_address so "
                "PeerAgents can register with NanoCtrl"
            )

        self._dlslime = dlslime
        self._proxy_factory = proxy
        self._wait_all = wait_all
        self._driver_agent = None
        self._driver_alias = None
        self._worker_aliases = []
        self._proxies = []

        super().__init__(config=config)

    def update_kvcache_blocks(self):
        num_cache_blocks = super().update_kvcache_blocks()
        if self._driver_agent is None:
            self._bootstrap_dlslime()
        return num_cache_blocks

    def _bootstrap_dlslime(self) -> None:
        available_nics = self._dlslime.available_nic()
        if not available_nics:
            raise RuntimeError("No available NICs found for DLSLime driver agent")

        self._driver_alias = f"{self.config.engine_id}:driver:{uuid.uuid4().hex[:8]}"
        self._driver_agent = self._dlslime.start_peer_agent(
            ctrl_url=self.config.ctrl_address,
            alias=self._driver_alias,
            device=available_nics[0],
            scope=self.config.ctrl_scope,
        )
        driver_qp_num = int(os.environ.get("SLIME_QP_NUM", 1))
        worker_aliases = ray.get(
            [
                worker.start_dlslime_server.remote(self._driver_alias)
                for worker in self.workers
            ]
        )
        self._worker_aliases = worker_aliases
        pending_conns = [
            self._driver_agent.connect_to(alias, ib_port=1, qp_num=driver_qp_num)
            for alias in worker_aliases
        ]
        for conn in pending_conns:
            conn.wait(timeout=60)
        self._proxies = [
            self._proxy_factory(self._driver_agent, alias, ModelRunnerRpcService)
            for alias in worker_aliases
        ]
        logger.info(f"DLSLime transport ready for {len(self._worker_aliases)} workers")

    def _probe_totals(self) -> tuple[int, int, int, int]:
        """Sum the C++ RpcSession timing probes across all DP-shard proxies.

        Returns cumulative (write_with_imm_ns, write_with_imm_count,
        imm_recv_ns, imm_recv_count). Missing/older sessions (no probe
        support) contribute 0 so this degrades gracefully.
        """
        wwi_ns = wwi_cnt = imm_ns = imm_cnt = 0
        for p in self._proxies:
            session = getattr(getattr(p, "_runtime", None), "session", None)
            if session is None:
                continue
            wwi_ns += getattr(session, "write_with_imm_ns_total", 0)
            wwi_cnt += getattr(session, "write_with_imm_count", 0)
            imm_ns += getattr(session, "imm_recv_ns_total", 0)
            imm_cnt += getattr(session, "imm_recv_count", 0)
        return wwi_ns, wwi_cnt, imm_ns, imm_cnt

    def run(
        self,
        dp_seqs: list[list[Sequence]],
        is_prefill: bool,
        timeout: float | None = None,
    ) -> list[list[list[int]]]:
        import time as _time

        _t0 = _time.perf_counter()
        batch_bytes = [serialize_run_batch(seqs, is_prefill) for seqs in dp_seqs]
        _t1 = _time.perf_counter()
        # Bytes sent to the runners this forward (serialized RunBatch input).
        self.last_run_request_bytes = sum(len(b) for b in batch_bytes)
        # Snapshot the C++ RPC timing probes before issuing the forward so we
        # can attribute the writeWithImm (send) / immRecv (recv) cost to this
        # step. On the no-pump fast path both verbs complete synchronously
        # inside run_batch, so the snapshot must straddle submit + wait_all.
        wwi_ns0, _, imm_ns0, _ = self._probe_totals()
        futures = [
            proxy.run_batch(encode_run_request(data, is_prefill))
            for proxy, data in zip(self._proxies, batch_bytes)
        ]
        _t2 = _time.perf_counter()
        replies = self._wait_all(futures)
        _t3 = _time.perf_counter()
        wwi_ns1, _, imm_ns1, _ = self._probe_totals()
        # Summed across DP shards; reported per-step in the heartbeat.
        self.last_run_wwi_ms = max(wwi_ns1 - wwi_ns0, 0) / 1e6
        self.last_run_immrecv_ms = max(imm_ns1 - imm_ns0, 0) / 1e6
        self.last_run_reply_bytes = sum(len(d) for d in replies)
        result = [decode_run_result(data) for data in replies]
        _t4 = _time.perf_counter()
        # Pure network latency = client round trip - remote handler time. Each
        # reply carries the server-side handler duration (decode + forward) in
        # its 8-byte header; shards run in parallel so the slowest one bounds
        # the compute overlapped with this step.
        server_compute_ms = 0.0
        if replies:
            server_compute_ms = max(unpack_reply_header(d) for d in replies) / 1e6
        self.last_run_server_compute_ms = server_compute_ms
        # transfer (round-trip wall clock) minus remote compute ≈ wire +
        # queueing + dispatch, with no GPU compute or pump idle pollution.
        self.last_run_net_ms = max((_t3 - _t1) * 1000.0 - server_compute_ms, 0.0)
        # Per-forward DLSlime timing breakdown (surfaced in the engine heartbeat).
        self.last_run_serialize_ms = (_t1 - _t0) * 1000.0
        self.last_run_submit_ms = (_t2 - _t1) * 1000.0
        self.last_run_wait_ms = (_t3 - _t2) * 1000.0
        self.last_run_decode_ms = (_t4 - _t3) * 1000.0
        # Transfer = send (submit) + wait-for-reply round trip.
        self.last_run_transfer_ms = (_t3 - _t1) * 1000.0
        if not is_prefill:
            logger.debug(
                f"[dlslime run] serialize={self.last_run_serialize_ms:.2f}ms "
                f"submit={self.last_run_submit_ms:.2f}ms "
                f"wait_all={self.last_run_wait_ms:.2f}ms "
                f"decode={self.last_run_decode_ms:.2f}ms "
                f"total={(_t4-_t0)*1000:.2f}ms"
            )
        return result

    def migrate(
        self,
        dp_seqs: list[list[Sequence]],
        timeout: float | None = None,
    ) -> list[int]:
        batch_bytes = [serialize_migrate_batch(seqs) for seqs in dp_seqs]
        futures = [
            proxy.migrate_batch(data) for proxy, data in zip(self._proxies, batch_bytes)
        ]
        self._wait_all(futures)
        return [0 for _ in dp_seqs]

    def __del__(self):
        # Guard against partial initialization (e.g., __init__ raised before
        # _driver_agent was assigned).
        agent = getattr(self, "_driver_agent", None)
        try:
            if agent is not None:
                agent.shutdown()
        except Exception as e:
            logger.warning(f"Failed to shutdown DLSLime driver agent: {e}")
        try:
            super().__del__()
        except Exception:
            pass
