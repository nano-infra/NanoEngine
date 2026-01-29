#!/usr/bin/env python3
"""
使用方式：bind / handshake / alloc_shared_buffer 示例脚本。
直接使用 C++ ZMQ 建链（RdmaRendezvousBackend + ZmqRendezvousServer + ZmqRendezvousStub）。

运行前确保有 RDMA 设备，且已安装 dlslime（BUILD_RDMA_RENDEZVOUS_ZMQ=ON）：
  python example/python/rdma_bind_handshake_alloc_usage.py
"""

import ctypes
import os

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


def _alloc_page_aligned(size):
    try:
        pagesize = os.sysconf("SC_PAGESIZE")
    except Exception:
        pagesize = 4096
    aligned_size = ((size + pagesize - 1) // pagesize) * pagesize
    libc = ctypes.CDLL("libc.so.6")
    libc.posix_memalign.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    libc.posix_memalign.restype = ctypes.c_int
    ptr_ref = ctypes.c_void_p()
    err = libc.posix_memalign(
        ctypes.byref(ptr_ref), ctypes.c_size_t(pagesize), ctypes.c_size_t(aligned_size)
    )
    if err != 0:
        raise RuntimeError(f"posix_memalign failed: {err}")
    return ptr_ref.value, aligned_size


def _get_mr_info_for_ptr(endpoint, ptr):
    info = endpoint.endpoint_info()
    mr_info = info.get("mr_info", {})
    return mr_info.get(str(ptr))


def _alloc_shared_buffer(endpoint, buffer_id, size, allocated_buffers):
    ptr, aligned_size = _alloc_page_aligned(size)
    endpoint.register_memory_region(ptr, ptr, 0, aligned_size)
    mr = _get_mr_info_for_ptr(endpoint, ptr)
    if not mr:
        raise RuntimeError(f"mr_info not found for ptr {ptr}")
    mr_key = int(mr["mr_key"])
    addr = int(mr["addr"])
    rkey = int(mr["rkey"])
    length = int(mr["length"])
    allocated_buffers[buffer_id] = (mr_key, addr, rkey, length)
    return buffer_id, ptr, mr_key


devices = available_nic()
assert devices, "No RDMA devices."

# 创建两个 endpoint（例如本机双网卡）
initiator_ep = RDMAEndpoint(device_name=devices[0], ib_port=1, link_type="RoCE")
target_ep = RDMAEndpoint(device_name=devices[-1], ib_port=1, link_type="RoCE")

initiator_backend = RdmaRendezvousBackend(initiator_ep)
target_backend = RdmaRendezvousBackend(target_ep)

initiator_server = ZmqRendezvousServer(initiator_backend, "0.0.0.0:50051")
target_server = ZmqRendezvousServer(target_backend, "0.0.0.0:50052")
initiator_server.start()
target_server.start()

# 对称建链：两端都 handshake
initiator_stub = ZmqRendezvousStub("127.0.0.1:50052")
target_stub = ZmqRendezvousStub("127.0.0.1:50051")
my_info = initiator_ep.endpoint_info()
remote_info = initiator_stub.handshake(my_info)
initiator_ep.connect(remote_info)
my_info_t = target_ep.endpoint_info()
remote_info_t = target_stub.handshake(my_info_t)
target_ep.connect(remote_info_t)

# 共享内存：先申请 alloc_shared_buffer，再绑定 bind_shared_buffer 推送给对端；对端 get_shared_buffer
initiator_buffers = {}
buf_id, ptr, mr_key = _alloc_shared_buffer(
    initiator_ep, "buf_1", 1024, initiator_buffers
)
initiator_stub.register_shared_buffer(
    {
        "buffer_id": buf_id,
        "mr_key": initiator_buffers[buf_id][0],
        "addr": initiator_buffers[buf_id][1],
        "rkey": initiator_buffers[buf_id][2],
        "length": initiator_buffers[buf_id][3],
    }
)

bid, mr_info = target_backend.get_pending_shared_buffer(None, 30.0)
assert bid is not None and mr_info is not None, "get_shared_buffer timed out"
remote_mr_key = mr_info["mr_key"]
addr = mr_info["addr"]
rkey = mr_info["rkey"]
length = mr_info["length"]
target_ep.register_remote_memory_region(
    remote_mr_key, {"addr": addr, "rkey": rkey, "length": length}
)

print(
    f"alloc_shared_buffer -> buffer_id={buf_id!r}, ptr={ptr}, mr_key={mr_key}; "
    f"bind_shared_buffer({buf_id!r}); get_shared_buffer -> remote_mr_key={remote_mr_key}, length={length}"
)
print("Usage script finished successfully.")

initiator_stub.close()
target_stub.close()
initiator_server.stop()
target_server.stop()
