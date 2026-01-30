#!/usr/bin/env python3
"""
Benchmark: N 对点对点连接的建链时间（2*N 个 endpoint）。
dlslime 仅支持点对点：每对 2 个 endpoint，一端 bind、一端 handshake。

若出现 ibv_reg_mr failed errno=12 (Cannot allocate memory)：
  - 先提高进程 locked memory：ulimit -l unlimited（脚本会打印当前 ulimit -l）。
  - 若已 unlimited 仍失败，可能是内核或软 RoCE (rxe) 对总 pin 内存有限制（128 端约 2GB+），
    可减小对数：python example/python/rdma_zmq_connect_64_bench.py 32
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

BASE_PORT = 50051


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark N point-to-point ZMQ rendezvous connections (2*N endpoints)"
    )
    parser.add_argument(
        "pairs", type=int, nargs="?", default=64, help="Number of pairs (default: 64)"
    )
    args = parser.parse_args()
    num_pairs = args.pairs
    if num_pairs < 1:
        parser.error("pairs must be >= 1")
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

    # num_pairs 对：每对 1 个 bind 端 + 1 个 handshake 端 = 2*num_pairs endpoints
    server_eps = []
    server_backends = []
    servers = []
    for i in range(num_pairs):
        ep = RDMAEndpoint(device_name=devices[i % n_dev], ib_port=1, link_type="RoCE")
        backend = RdmaRendezvousBackend(ep)
        srv = ZmqRendezvousServer(backend, f"0.0.0.0:{BASE_PORT + i}")
        server_eps.append(ep)
        server_backends.append(backend)
        servers.append(srv)

    for srv in servers:
        srv.start()
    time.sleep(0.3)  # 等所有 bind 完成

    errors = []
    lock = threading.Lock()

    def do_handshake(i):
        try:
            ep = RDMAEndpoint(
                device_name=devices[(i + 1) % n_dev], ib_port=1, link_type="RoCE"
            )
            stub = ZmqRendezvousStub(f"127.0.0.1:{BASE_PORT + i}")
            my_info = ep.endpoint_info()
            remote_info = stub.handshake(my_info)
            ep.connect(remote_info)
            stub.close()
        except Exception as e:
            with lock:
                errors.append((i, e))

    # num_pairs 对并行建链
    t0 = time.perf_counter()
    threads = [
        threading.Thread(target=do_handshake, args=(i,)) for i in range(num_pairs)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t1 = time.perf_counter()

    for srv in servers:
        srv.stop()

    if errors:
        print(f"Errors: {len(errors)}/{num_pairs}")
        for i, e in errors[:5]:
            print(f"  pair[{i}] {e}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
    else:
        elapsed = t1 - t0
        print(
            f"{2 * num_pairs} endpoints, {num_pairs} point-to-point connections (C++ ZMQ rendezvous): {elapsed:.3f} s"
        )
        print(f"  Per connection: {elapsed / num_pairs * 1000:.2f} ms")


if __name__ == "__main__":
    main()
