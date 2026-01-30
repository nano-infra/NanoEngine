"""
P2P RDMA RC Write 封装示例：PeerAgent 先创建，peer 惰性（无 agent.peer() 申请）。
建链只靠 agent.connect(对端 agent.client_addr)；两 agent 间永远只有一条链路。connect 对称且幂等。
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

from dlslime import available_nic, start_peer_agent

AGENT_A_ADDR = "0.0.0.0:50051"
AGENT_B_ADDR = "0.0.0.0:50052"


def main():
    devices = available_nic()
    assert devices, "No RDMA devices."
    print("RDMA devices:", devices, flush=True)

    agent_a = start_peer_agent(AGENT_A_ADDR)
    agent_b = start_peer_agent(AGENT_B_ADDR)
    time.sleep(0.1)

    # 只做 connect，对称且幂等；peer 惰性，两 agent 间一条链路
    # Try single-sided connection
    print("Initiating connection from A to B...", flush=True)
    agent_a.connect(agent_b.client_addr, devices[0])
    print("Connection from A to B done (if successful).", flush=True)

    # A 写 B：分配 buffer、取 MR、write（均走 agent）
    ptr_a, _ = agent_a.alloc_and_register_buffer("buf_a", 16)
    ctypes.memset(ctypes.c_void_p(ptr_a), 0, 8)
    ctypes.memset(ctypes.c_void_p(ptr_a + 8), 1, 8)

    ptr_b, _ = agent_b.alloc_and_register_buffer("buf_t", 16)
    ctypes.memset(ctypes.c_void_p(ptr_b), 1, 16)

    remote = agent_b.client_addr
    local_mr = agent_a.get_local_mr_key(remote, "buf_a")
    remote_mr = agent_a.get_remote_mr_key(remote, "buf_t")
    agent_a.write(remote, [(local_mr, remote_mr, 0, 0, 8)], None).wait()

    first_8 = ctypes.string_at(ptr_b, 8)
    last_8 = ctypes.string_at(ptr_b + 8, 8)
    assert all(b == 0 for b in first_8), "First 8 bytes should be zeros"
    assert all(b == 1 for b in last_8), "Last 8 bytes should remain ones"

    agent_a.close()
    agent_b.close()
    agent_a.stop()
    agent_b.stop()
    print("OK: rdma rc write (encapsulated, lazy peer)")


if __name__ == "__main__":
    main()
