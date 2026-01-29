"""
P2P RDMA RC Write 封装示例：Broker 先创建，peer 惰性（无 broker.peer() 申请）。
建链只靠 broker.connect(对端 broker.client_addr)；两 broker 间永远只有一条链路。connect 对称且幂等。
"""

import ctypes
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dlslime._slime_c as _slime_c

if not getattr(_slime_c, "_BUILD_RDMA_RENDEZVOUS_ZMQ", False):
    raise RuntimeError(
        "dlslime was built without BUILD_RDMA_RENDEZVOUS_ZMQ; rebuild with -DBUILD_RDMA_RENDEZVOUS_ZMQ=ON"
    )

from dlslime import available_nic, start_broker

BROKER_A_ADDR = "0.0.0.0:50051"
BROKER_B_ADDR = "0.0.0.0:50052"


def main():
    devices = available_nic()
    assert devices, "No RDMA devices."
    print("RDMA devices:", devices, flush=True)

    broker_a = start_broker(BROKER_A_ADDR)
    broker_b = start_broker(BROKER_B_ADDR)
    time.sleep(0.1)

    # 只做 connect，对称且幂等；peer 惰性，两 broker 间一条链路
    # Try single-sided connection
    print("Initiating connection from A to B...", flush=True)
    broker_a.connect(broker_b.client_addr, devices[0])
    print("Connection from A to B done (if successful).", flush=True)

    # A 写 B：分配 buffer、取 MR、write（均走 broker）
    ptr_a, _ = broker_a.alloc_and_register_buffer("buf_a", 16)
    ctypes.memset(ctypes.c_void_p(ptr_a), 0, 8)
    ctypes.memset(ctypes.c_void_p(ptr_a + 8), 1, 8)

    ptr_b, _ = broker_b.alloc_and_register_buffer("buf_t", 16)
    ctypes.memset(ctypes.c_void_p(ptr_b), 1, 16)

    remote = broker_b.client_addr
    local_mr = broker_a.get_local_mr_key(remote, "buf_a")
    remote_mr = broker_a.get_remote_mr_key(remote, "buf_t")
    broker_a.write(remote, [(local_mr, remote_mr, 0, 0, 8)], None).wait()

    first_8 = ctypes.string_at(ptr_b, 8)
    last_8 = ctypes.string_at(ptr_b + 8, 8)
    assert all(b == 0 for b in first_8), "First 8 bytes should be zeros"
    assert all(b == 1 for b in last_8), "Last 8 bytes should remain ones"

    broker_a.close()
    broker_b.close()
    broker_a.stop()
    broker_b.stop()
    print("OK: rdma rc write (encapsulated, lazy peer)")


if __name__ == "__main__":
    main()
