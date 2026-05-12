import os
import uuid

import ray

from nanodeploy._cpp import Sequence, serialize_migrate_batch, serialize_run_batch
from nanodeploy.config import Config
from nanodeploy.engine.dlslime_protocol import (
    decode_run_result,
    encode_run_request,
    ModelRunnerRpcService,
)
from nanodeploy.engine.ray_executor import RayExecutor
from nanodeploy.logging import get_logger

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
                "dependency. Install NanoDeploy with the 'dlslime' extra."
            ) from exc

        if not config.nanoctrl_address:
            raise ValueError(
                "executor_backend='dlslime' requires nanoctrl_address so "
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
            nanoctrl_url=self.config.nanoctrl_address,
            alias=self._driver_alias,
            device=available_nics[0],
            scope=self.config.nanoctrl_scope,
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
        futures = [
            proxy.run_batch(encode_run_request(data, is_prefill))
            for proxy, data in zip(self._proxies, batch_bytes)
        ]
        _t2 = _time.perf_counter()
        replies = self._wait_all(futures)
        _t3 = _time.perf_counter()
        result = [decode_run_result(data) for data in replies]
        _t4 = _time.perf_counter()
        if not is_prefill:
            logger.debug(
                f"[dlslime run] serialize={(_t1-_t0)*1000:.2f}ms "
                f"submit={(_t2-_t1)*1000:.2f}ms "
                f"wait_all={(_t3-_t2)*1000:.2f}ms "
                f"unpickle={(_t4-_t3)*1000:.2f}ms "
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
