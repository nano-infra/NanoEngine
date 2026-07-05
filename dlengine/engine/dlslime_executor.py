import os
import uuid

import ray

from dlengine.config import Config
from dlengine.engine.dlslime_protocol import (
    decode_run_result,
    decode_runner_out,
    encode_run_request,
    ModelRunnerRpcService,
    server_handler_ns,
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
        """Sum the transport RpcSession timing probes across all DP-shard proxies.

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

    def _probe_per_proxy(self) -> tuple[list[int], list[int]]:
        """Per-proxy cumulative (write_with_imm_ns, imm_recv_ns) snapshots.

        The DP shards issue and complete their RPCs concurrently, so the
        per-step wall-clock cost of these verbs is the *slowest* shard (max of
        the per-proxy deltas), not the sum across shards. Returning per-proxy
        values lets :meth:`run_wait` take that max instead of an ~Nx-inflated
        sum. Missing/older sessions contribute 0 (graceful degradation).
        """
        wwi: list[int] = []
        imm: list[int] = []
        for p in self._proxies:
            session = getattr(getattr(p, "_runtime", None), "session", None)
            if session is None:
                wwi.append(0)
                imm.append(0)
                continue
            wwi.append(getattr(session, "write_with_imm_ns_total", 0))
            imm.append(getattr(session, "imm_recv_ns_total", 0))
        return wwi, imm

    def run_batch_bytes_async(self, batch_bytes: list[bytes], is_prefill: bool) -> dict:
        """Submit serialized RunnerIn bytes without waiting.

        Returns an opaque handle for :meth:`run_wait`. Splitting submit from
        wait lets the driver overlap its own bookkeeping (stepout emission,
        metrics, request drain) with the GPU forward of the next step.
        """
        import time as _time

        _t0 = _time.perf_counter()
        _t1 = _time.perf_counter()
        # Snapshot the RPC timing probes before issuing the forward so we
        # can attribute the writeWithImm (send) / immRecv (recv) cost to this
        # step. On the no-pump fast path both verbs complete synchronously
        # inside run_batch, so the snapshot must straddle submit + wait_all.
        wwi_ns0_list, imm_ns0_list = self._probe_per_proxy()
        futures = [
            proxy.run_batch(encode_run_request(data))
            for proxy, data in zip(self._proxies, batch_bytes)
        ]
        _t2 = _time.perf_counter()
        return {
            "futures": futures,
            "is_prefill": is_prefill,
            "request_bytes": sum(len(b) for b in batch_bytes),
            "t0": _t0,
            "t1": _t1,
            "t2": _t2,
            "wwi_ns0_list": wwi_ns0_list,
            "imm_ns0_list": imm_ns0_list,
        }

    def prepare_batch_bytes_async(
        self, batch_bytes: list[bytes], is_prefill: bool
    ) -> dict:
        """Submit the prepare half of a forward to all workers."""
        import time as _time

        _t0 = _time.perf_counter()
        futures = [
            proxy.prepare_batch(encode_run_request(data))
            for proxy, data in zip(self._proxies, batch_bytes)
        ]
        _t1 = _time.perf_counter()
        return {
            "futures": futures,
            "is_prefill": is_prefill,
            "request_bytes": sum(len(b) for b in batch_bytes),
            "t0": _t0,
            "t1": _t1,
        }

    def prepare_wait(self, handle: dict) -> list[bytes]:
        """Wait for prepare_batch_bytes_async and return worker-local handles."""
        return self._wait_all(handle["futures"])

    def run_prepared_bytes_async(
        self,
        prepared_handles: list[bytes],
        is_prefill: bool,
        request_bytes: int = 0,
    ) -> dict:
        """Submit the run half for worker-local prepare handles."""
        import time as _time

        _t0 = _time.perf_counter()
        _t1 = _time.perf_counter()
        wwi_ns0_list, imm_ns0_list = self._probe_per_proxy()
        futures = [
            proxy.run_prepared(handle)
            for proxy, handle in zip(self._proxies, prepared_handles)
        ]
        _t2 = _time.perf_counter()
        return {
            "futures": futures,
            "is_prefill": is_prefill,
            "request_bytes": request_bytes,
            "t0": _t0,
            "t1": _t1,
            "t2": _t2,
            "wwi_ns0_list": wwi_ns0_list,
            "imm_ns0_list": imm_ns0_list,
        }

    def run_wait(self, handle: dict) -> list[list[list[int]]]:
        """Wait for a forward submitted via :meth:`run_async` and decode it."""
        runner_outs = self.run_wait_runner_outs(handle)
        return [out.result for out in runner_outs]

    def run_wait_runner_outs(self, handle: dict):
        """Wait for a forward submitted via :meth:`run_async` and return RunnerOuts."""
        import time as _time

        is_prefill = handle["is_prefill"]
        # Bytes sent to the runners this forward (serialized RunnerIn input).
        self.last_run_request_bytes = handle["request_bytes"]
        replies = self._wait_all(handle["futures"])
        _t3 = _time.perf_counter()
        wwi_ns1_list, imm_ns1_list = self._probe_per_proxy()
        # DP shards run concurrently, so the per-step wall-clock cost of each
        # verb is the slowest shard (max of per-proxy deltas), NOT the sum —
        # summing inflates the metric ~Nx (N = #DP shards).
        wwi_delta = max(
            (a - b for a, b in zip(wwi_ns1_list, handle["wwi_ns0_list"])),
            default=0,
        )
        imm_delta = max(
            (a - b for a, b in zip(imm_ns1_list, handle["imm_ns0_list"])),
            default=0,
        )
        self.last_run_wwi_ms = max(wwi_delta, 0) / 1e6
        self.last_run_immrecv_ms = max(imm_delta, 0) / 1e6
        self.last_run_reply_bytes = sum(len(d) for d in replies)
        result = [decode_runner_out(data) for data in replies]
        _t4 = _time.perf_counter()
        # Pure network latency = client round trip - remote handler time. Each
        # RunnerOut reply carries the server-side handler duration (decode +
        # forward); shards run in parallel so the slowest one bounds the
        # compute overlapped with this step.
        server_compute_ms = 0.0
        if replies:
            server_compute_ms = max(server_handler_ns(d) for d in replies) / 1e6
        self.last_run_server_compute_ms = server_compute_ms
        # transfer (round-trip wall clock) minus remote compute ≈ wire +
        # queueing + dispatch, with no GPU compute or pump idle pollution.
        self.last_run_net_ms = max(
            (_t3 - handle["t1"]) * 1000.0 - server_compute_ms, 0.0
        )
        # Per-forward DLSlime timing breakdown (surfaced in the engine heartbeat).
        self.last_run_serialize_ms = (handle["t1"] - handle["t0"]) * 1000.0
        self.last_run_submit_ms = (handle["t2"] - handle["t1"]) * 1000.0
        self.last_run_wait_ms = (_t3 - handle["t2"]) * 1000.0
        self.last_run_decode_ms = (_t4 - _t3) * 1000.0
        # Transfer = send (submit) + wait-for-reply round trip.
        self.last_run_transfer_ms = (_t3 - handle["t1"]) * 1000.0
        if not is_prefill:
            logger.debug(
                f"[dlslime run] serialize={self.last_run_serialize_ms:.2f}ms "
                f"submit={self.last_run_submit_ms:.2f}ms "
                f"wait_all={self.last_run_wait_ms:.2f}ms "
                f"decode={self.last_run_decode_ms:.2f}ms "
                f"total={(_t4-handle['t0'])*1000:.2f}ms"
            )
        return result

    def run(
        self,
        batch_bytes: list[bytes],
        is_prefill: bool,
        timeout: float | None = None,
    ) -> list[list[list[int]]]:
        return self.run_wait(self.run_batch_bytes_async(batch_bytes, is_prefill))

    def migrate_batch_bytes(
        self,
        batch_bytes: list[bytes],
        timeout: float | None = None,
    ) -> list[int]:
        futures = [
            proxy.migrate_batch(data) for proxy, data in zip(self._proxies, batch_bytes)
        ]
        self._wait_all(futures)
        return [0 for _ in batch_bytes]

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
