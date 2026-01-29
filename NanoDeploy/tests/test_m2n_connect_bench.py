"""
8x8 M2N (many-to-many) connection establishment time benchmark.

1. Mock: simulates ensure_p2p_connected with configurable sleep (no RDMA).
2. DLSlime Broker: real Broker + RDMALazyPeer, measures actual handshake time.
   Requires RDMA device and dlslime built with BUILD_RDMA_RENDEZVOUS_ZMQ=ON.

Run:
  python tests/test_m2n_connect_bench.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from typing import List, Tuple

DLSLIME_BENCH_TIMEOUT_S = 25

# Allow importing nanodeploy / dlslime from repo root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class MockExecutor:
    """Simulates executor.ensure_p2p_connected(peer_id, addrs, num_blocks) with sleep."""

    def __init__(self, connect_latency_ms: float = 2.0):
        self.connect_latency_ms = connect_latency_ms
        self.ensure_calls: List[Tuple[str, List[str], int]] = []

    def ensure_p2p_connected(
        self, peer_id: str, addrs: List[str], num_blocks: int
    ) -> None:
        self.ensure_calls.append((peer_id, addrs, num_blocks))
        time.sleep(self.connect_latency_ms / 1000.0)


# --- Mock node: has _peer_info and ensure_p2p_connected like LLMComponent ---


class MockNode:
    """Minimal node that only does ensure_p2p_connected (no etcd, no real executor)."""

    def __init__(self, engine_id: str, executor: MockExecutor):
        self.engine_id = engine_id
        self.executor = executor
        self.active_p2p_links: set[str] = set()
        self._peer_info: dict[str, Tuple[List[str], int]] = {}

    def ensure_p2p_connected(self, peer_id: str) -> None:
        if peer_id in self.active_p2p_links:
            return
        if peer_id not in self._peer_info:
            return
        addrs, num_blocks = self._peer_info[peer_id]
        self.executor.ensure_p2p_connected(peer_id, addrs, num_blocks)
        self.active_p2p_links.add(peer_id)

    def add_peer(self, peer_id: str, addrs: List[str], num_blocks: int) -> None:
        self._peer_info[peer_id] = (addrs, num_blocks)


# --- Build 8 nodes with full M2N peer info ---

BASE_PORT = 50051
N_NODES = 8


def build_m2n_nodes(
    n: int = N_NODES,
    base_port: int = BASE_PORT,
    connect_latency_ms: float = 2.0,
    shared_executor: bool = False,
) -> List[MockNode]:
    """Build n nodes; each node has peer_info for the other n-1 nodes (one addr per peer).
    If shared_executor=False (default), each node has its own executor so parallel bench is realistic.
    """
    if shared_executor:
        executor = MockExecutor(connect_latency_ms=connect_latency_ms)
        nodes = [MockNode(engine_id=f"node-{i}", executor=executor) for i in range(n)]
    else:
        nodes = [
            MockNode(
                engine_id=f"node-{i}",
                executor=MockExecutor(connect_latency_ms=connect_latency_ms),
            )
            for i in range(n)
        ]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            peer_id = f"node-{j}"
            addrs = [f"127.0.0.1:{base_port + j}"]
            nodes[i].add_peer(peer_id, addrs, num_blocks=1000)
    return nodes


# --- Benchmarks ---


def run_sequential(nodes: List[MockNode]) -> float:
    """One node connects to all its peers sequentially. Returns elapsed seconds."""
    node = nodes[0]
    peers = list(node._peer_info.keys())
    t0 = time.perf_counter()
    for peer_id in peers:
        node.ensure_p2p_connected(peer_id)
    return time.perf_counter() - t0


def run_parallel(nodes: List[MockNode]) -> float:
    """All nodes connect to their peers in parallel (8 threads). Returns elapsed seconds."""

    def connect_all(node: MockNode) -> None:
        for peer_id in list(node._peer_info.keys()):
            node.ensure_p2p_connected(peer_id)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=connect_all, args=(n,)) for n in nodes]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.perf_counter() - t0


# --- DLSlime Broker: real 8x8 M2N connect time ---


def run_dlslime_m2n_bench(
    n: int = 4,
    base_port: int = BASE_PORT,
    n_runs: int = 2,
    timeout_s: int | None = None,
    parallel_pairs: bool = True,
) -> Tuple[float | None, int]:
    """
    Use real DLSlime Broker + RDMALazyPeer to establish n*n M2N links.
    Peer connect is symmetric and idempotent: no initiator/target; either side can connect first.
    Uses n=4 by default to avoid RDMA memory registration limits (8*7=56 links can OOM).
    If timeout_s is set, each run's connect phase is limited to that many seconds (avoids hang).
    parallel_pairs: if True, establish all pairs in parallel (one thread per pair, both directions).
    Returns (average elapsed seconds over n_runs, total_links) or (None, 0) if unavailable.
    """
    try:
        import dlslime._slime_c as _slime_c
    except Exception:
        return (None, 0)
    if not getattr(_slime_c, "_BUILD_RDMA_RENDEZVOUS_ZMQ", False):
        return (None, 0)
    try:
        from dlslime import available_nic, start_broker
    except Exception:
        return (None, 0)
    devices = available_nic()
    if not devices:
        return (None, 0)
    device = devices[0]

    total_links = n * (n - 1)
    times: List[float] = []
    pairs = [(i, j) for i in range(n) for j in range(n) if i < j]

    for run in range(n_runs):
        brokers = []
        for i in range(n):
            brokers.append(start_broker(f"0.0.0.0:{base_port + i}"))
        time.sleep(0.2)
        addrs = [brokers[i].client_addr for i in range(n)]

        err = [None]

        def do_connect(i: int, j: int) -> None:
            """Single connect: broker i -> broker j. Pair (i,j) needs both do_connect(i,j) and do_connect(j,i) to complete."""
            try:
                brokers[i].connect(addrs[j], device)
            except Exception as e:
                err[0] = e

        def do_one_pair(i: int, j: int) -> None:
            """Sequential: (j,i) then (i,j) so j registers on broker i before i blocks in GetPeerInfo."""
            try:
                brokers[j].connect(addrs[i], device)
                brokers[i].connect(addrs[j], device)
            except Exception as e:
                err[0] = e

        t0 = time.perf_counter()
        join_timeout = timeout_s or 10
        if parallel_pairs:
            # Two threads per pair: (i,j).connect and (j,i).connect must run in parallel or we deadlock (each blocks until the other registers).
            threads = []
            for i, j in pairs:
                threads.append(
                    threading.Thread(target=do_connect, args=(i, j), daemon=True)
                )
                threads.append(
                    threading.Thread(target=do_connect, args=(j, i), daemon=True)
                )
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=join_timeout)
            if any(t.is_alive() for t in threads):
                err[0] = TimeoutError("DLSlime connect (parallel) timed out")
        else:
            for i, j in pairs:
                if err[0]:
                    break
                do_one_pair(i, j)

        elapsed = time.perf_counter() - t0
        if err[0]:
            if isinstance(err[0], TimeoutError):
                if timeout_s:
                    sys.stderr.write(
                        f"DLSlime M2N connect timed out: {err[0]!s} "
                        f"(timeout {timeout_s}s per pair)\n"
                    )
                    sys.stderr.flush()
                return (None, total_links)
            raise err[0]
        times.append(elapsed)

        for b in brokers:
            try:
                b.stop()
            except Exception:
                pass

    return (sum(times) / len(times) if times else None, total_links)


def _run_dlslime_in_subprocess(
    n: int, base_port: int, n_runs: int, timeout_s: int
) -> Tuple[float | None, int]:
    """Run DLSlime bench in subprocess with timeout; return (avg_sec, total_links) or (None, 0)."""
    script_path = os.path.join(_REPO_ROOT, "tests", "test_m2n_connect_bench.py")
    try:
        proc = subprocess.run(
            [
                sys.executable,
                script_path,
                "--dlslime-only",
                str(n),
                str(base_port),
                str(n_runs),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=_REPO_ROOT,
        )
        if proc.returncode != 0:
            return (None, 0)
        for line in (proc.stdout or "").strip().splitlines():
            if line.startswith("RESULT:"):
                rest = line[7:].strip()
                part = rest.split(":")
                if len(part) == 2:
                    try:
                        t_val = float(part[0]) if part[0] != "None" else None
                        links_val = int(part[1])
                        return (t_val, links_val)
                    except ValueError:
                        pass
        return (None, 0)
    except subprocess.TimeoutExpired:
        return (None, 0)
    except Exception:
        return (None, 0)


def main() -> int:
    connect_latency_ms = 2.0
    n = N_NODES
    total_links = n * (n - 1)  # 8*7 = 56
    n_runs = 3

    print("=" * 60)
    print("8x8 M2N connection establishment 对比 (mock vs real)")
    print(
        f"  nodes={n}, total_links={total_links}, "
        f"latency_per_connect={connect_latency_ms}ms, runs={n_runs}"
    )
    print("=" * 60)

    print("  [模拟 mock]")
    seq_times = []
    for _ in range(n_runs):
        nodes_seq = build_m2n_nodes(
            n=n, connect_latency_ms=connect_latency_ms, shared_executor=True
        )
        seq_times.append(run_sequential(nodes_seq))
    t_seq = sum(seq_times) / n_runs
    print(
        f"  Sequential (1 node -> {n-1} peers): {t_seq*1000:.2f} ms  "
        f"({t_seq*1000/(n-1):.2f} ms/link)"
    )

    par_times = []
    for _ in range(n_runs):
        nodes_par = build_m2n_nodes(
            n=n, connect_latency_ms=connect_latency_ms, shared_executor=False
        )
        par_times.append(run_parallel(nodes_par))
    t_par = sum(par_times) / n_runs
    print(
        f"  Parallel   (8 nodes -> {n-1} peers each, {total_links} links): "
        f"{t_par*1000:.2f} ms  ({t_par*1000/total_links:.2f} ms/link)"
    )

    print("  [实际 real] DLSlime Broker")
    n_dlslime = 4
    t_dlslime, links_dlslime = _run_dlslime_in_subprocess(
        n=n_dlslime,
        base_port=BASE_PORT + 100,
        n_runs=2,
        timeout_s=DLSLIME_BENCH_TIMEOUT_S,
    )
    if t_dlslime is not None and links_dlslime > 0:
        print(
            f"  {n_dlslime}x{n_dlslime} M2N, {links_dlslime} links: "
            f"{t_dlslime*1000:.2f} ms  ({t_dlslime*1000/links_dlslime:.2f} ms/link)"
        )
    else:
        print("  skipped (no RDMA, no ZMQ, timeout, or resource error)")

    print("=" * 60)
    print("[ok] M2N connect benchmark done")
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--dlslime-only":
        # Run only DLSlime bench (no mock). Used by subprocess or direct: python ... --dlslime-only [n] [base_port] [n_runs]
        _n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
        _base = int(sys.argv[3]) if len(sys.argv) > 3 else BASE_PORT + 100
        _runs = int(sys.argv[4]) if len(sys.argv) > 4 else 2
        _timeout = 40  # per-pair join timeout when run directly (6 pairs * 40s max)
        try:
            _t, _links = run_dlslime_m2n_bench(
                n=_n, base_port=_base, n_runs=_runs, timeout_s=_timeout
            )
        except Exception as e:
            _t, _links = None, 0
            sys.stderr.write(f"DLSlime M2N connect error: {e}\n")
            sys.stderr.flush()
        if _t is None and _links > 0:
            sys.stderr.write(
                "DLSlime returned None (timeout or handshake failure). "
                "Check RDMA devices and that brokers/peers are ready.\n"
            )
        sys.stderr.flush()
        print(f"RESULT:{_t}:{_links}")
        raise SystemExit(0 if (_t is not None and _links > 0) else 1)
    raise SystemExit(main())
