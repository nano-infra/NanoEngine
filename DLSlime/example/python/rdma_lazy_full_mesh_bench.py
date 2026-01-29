#!/usr/bin/env python3
"""
Benchmark: N 个端点做 full mesh（N(N-1)/2 条链路）的建链时间，使用惰性建链。

每个节点 i 有 (N-1) 个 endpoint，分别连到其他节点 j；共 N*(N-1) 个 endpoint，N*(N-1)/2 条链路。
协议：对每对 (i,j) 且 i<j，节点 i 为发起方、节点 j 为对端；经中心 Broker 完成惰性建链。

若出现 ibv_reg_mr failed errno=12：
  - 先提高 ulimit -l（脚本会打印当前 memlock）。
  - 可减小 N：python example/python/rdma_lazy_full_mesh_bench.py 4
"""

import argparse
import resource
import threading
import time

import dlslime._slime_c as _slime_c

if not getattr(_slime_c, "_BUILD_RDMA_RENDEZVOUS_ZMQ", False):
    raise RuntimeError(
        "dlslime was built without BUILD_RDMA_RENDEZVOUS_ZMQ; rebuild with -DBUILD_RDMA_RENDEZVOUS_ZMQ=ON"
    )

from dlslime import (
    available_nic,
    RDMAEndpoint,
    RdmaRendezvousBackend,
    ZmqRendezvousServer,
    ZmqRendezvousStub,
)

BROKER_PORT = 50051
LAZY_TIMEOUT_SEC = 60.0


def run_node(i, N, broker_addr, barrier, devices, n_dev, timing):
    """Node i: create (N-1) endpoints, then do lazy handshake phases."""
    stub = ZmqRendezvousStub(broker_addr)
    # endpoints[i][j] = endpoint at node i for connection to node j (j != i)
    endpoints = {}
    for j in range(N):
        if j == i:
            continue
        ep = RDMAEndpoint(
            device_name=devices[(i * N + j) % n_dev],
            ib_port=1,
            link_type="RoCE",
        )
        endpoints[j] = ep

    barrier.wait()  # all nodes created endpoints
    if i == 0:
        timing[0] = time.perf_counter()

    # Phase 1: initiators request (i < j: i initiates to j)
    for j in range(i + 1, N):
        stub.request_lazy_handshake(str(i), str(j), endpoints[j].endpoint_info())
    barrier.wait()

    # Phase 2: peers get_pending, connect, respond (j < i: i is peer for j)
    pending = stub.get_pending_lazy_handshakes(str(i))
    for item in pending:
        initiator_id = item["initiator_id"]
        initiator_info = item["endpoint_info"]
        j = int(initiator_id)
        endpoints[j].connect(initiator_info)
        stub.respond_lazy_handshake(str(j), str(i), endpoints[j].endpoint_info())
    barrier.wait()

    # Phase 3: initiators get_response and connect (i < j)
    for j in range(i + 1, N):
        peer_info = stub.get_lazy_handshake_response(
            str(i), str(j), timeout_sec=LAZY_TIMEOUT_SEC
        )
        if not peer_info:
            raise RuntimeError(f"get_lazy_handshake_response({i}, {j}) timed out")
        endpoints[j].connect(peer_info)
    barrier.wait()

    if i == 0:
        timing[1] = time.perf_counter()

    stub.close()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark N endpoints full mesh (N(N-1)/2 links) with lazy handshake"
    )
    parser.add_argument(
        "N",
        type=int,
        nargs="?",
        default=4,
        help="Number of nodes in full mesh (default: 4)",
    )
    args = parser.parse_args()
    N = args.N
    if N < 2:
        parser.error("N must be >= 2 for full mesh")

    try:
        memlock_kb = resource.getrlimit(resource.RLIMIT_MEMLOCK)[1]
        if memlock_kb == -1:
            print("ulimit -l (memlock): unlimited")
        else:
            print(
                "ulimit -l (memlock):",
                memlock_kb,
                "KB (~%.1f GB)" % (memlock_kb / 1024 / 1024),
            )
    except Exception:
        pass

    devices = available_nic()
    assert devices, "No RDMA devices."
    n_dev = len(devices)

    broker_backend = RdmaRendezvousBackend(None)
    broker_server = ZmqRendezvousServer(broker_backend, "0.0.0.0:%d" % BROKER_PORT)
    broker_server.start()
    broker_addr = "127.0.0.1:%d" % BROKER_PORT

    barrier = threading.Barrier(N)
    timing = [None, None]
    errors = []
    lock = threading.Lock()

    def worker(i):
        try:
            run_node(i, N, broker_addr, barrier, devices, n_dev, timing)
        except Exception as e:
            with lock:
                errors.append((i, e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t1 = time.perf_counter()

    broker_server.stop()

    if errors:
        print("Errors: %d/%d" % (len(errors), N))
        for i, e in errors[:5]:
            print("  node[%d] %s" % (i, e))
        if len(errors) > 5:
            print("  ... and %d more" % (len(errors) - 5))
        return

    num_links = N * (N - 1) // 2
    num_endpoints = N * (N - 1)
    elapsed = (
        timing[1] - timing[0]
        if timing[0] is not None and timing[1] is not None
        else (t1 - t0)
    )
    print(
        "%d endpoints, %d nodes, %d links (full mesh, lazy handshake): %.3f s"
        % (num_endpoints, N, num_links, elapsed)
    )
    print("  Per link: %.2f ms" % (elapsed / num_links * 1000))


if __name__ == "__main__":
    main()
