#!/usr/bin/env python3
"""
惰性建链示例：通过中心 Broker 按需建立 RDMA 连接。
发起方先登记「我要连对端」+ 自己的 endpoint_info；
对端从 Broker 拉取待处理请求，创建 endpoint 并连到发起方，再回写自己的 endpoint_info；
发起方取回对端 endpoint_info 后完成连接。

运行前确保有 RDMA 设备，且已安装 dlslime（BUILD_RDMA_RENDEZVOUS_ZMQ=ON）：
  python example/python/rdma_lazy_handshake_usage.py
"""

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


def main():
    devices = available_nic()
    assert devices, "No RDMA devices."

    # 中心 Broker：无 endpoint，仅做惰性建链路由
    broker_backend = RdmaRendezvousBackend(None)
    broker_server = ZmqRendezvousServer(broker_backend, f"0.0.0.0:{BROKER_PORT}")
    broker_server.start()

    stub = ZmqRendezvousStub(f"127.0.0.1:{BROKER_PORT}")

    # 发起方 A、对端 B 各创建一个 endpoint（惰性：仅在需要时建链）
    initiator_ep = RDMAEndpoint(device_name=devices[0], ib_port=1, link_type="RoCE")
    peer_ep = RDMAEndpoint(device_name=devices[-1], ib_port=1, link_type="RoCE")

    # 1) 发起方 A：登记「我要连 B」，并提交自己的 endpoint_info
    stub.request_lazy_handshake("A", "B", initiator_ep.endpoint_info())

    # 2) 对端 B：拉取「谁要连我」的待处理请求（消费式）
    pending = stub.get_pending_lazy_handshakes("B")
    assert len(pending) == 1, f"Expected 1 pending handshake, got {len(pending)}"
    item = pending[0]
    initiator_id = item["initiator_id"]
    initiator_info = item["endpoint_info"]

    # 3) 对端 B：先连到发起方 A
    peer_ep.connect(initiator_info)

    # 4) 对端 B：把 B 的 endpoint_info 回写给 Broker，供 A 取用
    stub.respond_lazy_handshake(initiator_id, "B", peer_ep.endpoint_info())

    # 5) 发起方 A：从 Broker 取回 B 的 endpoint_info（阻塞直到 B 已 respond 或超时）
    peer_info = stub.get_lazy_handshake_response("A", "B", timeout_sec=30.0)
    assert peer_info, "get_lazy_handshake_response timed out or returned empty"

    # 6) 发起方 A：连到对端 B，完成双向建链
    initiator_ep.connect(peer_info)

    stub.close()
    broker_server.stop()

    print("Lazy handshake done: A <-> B connected.")


if __name__ == "__main__":
    main()
