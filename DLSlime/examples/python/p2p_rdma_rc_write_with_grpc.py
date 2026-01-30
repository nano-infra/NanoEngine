"""
P2P RDMA RC Write example using C++ ZMQ 建链（bind / handshake / alloc_shared_buffer）。

单端建链：只有发起方调用 handshake()，对端只 bind() 即可。
  - 发起方：handshake(remote_addr) 建链，alloc + bind_shared_buffer 推送，get_remote_shared_buffer 拉取对端 buffer
  - 对端：只 bind()，get_shared_buffer 接收，alloc + publish_shared_buffer 暴露给发起方拉取
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

# Two endpoints (e.g. same machine, two NICs)
initiator_ep = RDMAEndpoint(device_name=devices[0], ib_port=1, link_type="RoCE")
target_ep = RDMAEndpoint(device_name=devices[-1], ib_port=1, link_type="RoCE")

initiator_backend = RdmaRendezvousBackend(initiator_ep)
target_backend = RdmaRendezvousBackend(target_ep)

initiator_server = ZmqRendezvousServer(initiator_backend, "0.0.0.0:50051")
target_server = ZmqRendezvousServer(target_backend, "0.0.0.0:50052")
initiator_server.start()
target_server.start()

# 只有发起方 handshake，对端不调用 handshake
initiator_stub = ZmqRendezvousStub("127.0.0.1:50052")
my_info = initiator_ep.endpoint_info()
remote_info = initiator_stub.handshake(my_info)
initiator_ep.connect(remote_info)

# 发起方：申请并推送给对端
initiator_buffers = {}
buf_id_a, ptr_a, key_a = _alloc_shared_buffer(
    initiator_ep, "buf_a", 16, initiator_buffers
)
initiator_stub.register_shared_buffer(
    {
        "buffer_id": buf_id_a,
        "mr_key": initiator_buffers[buf_id_a][0],
        "addr": initiator_buffers[buf_id_a][1],
        "rkey": initiator_buffers[buf_id_a][2],
        "length": initiator_buffers[buf_id_a][3],
    }
)
ctypes.memset(ctypes.c_void_p(ptr_a), 0, 8)
ctypes.memset(ctypes.c_void_p(ptr_a + 8), 1, 8)

# 对端：接收发起方 buffer，申请本地 buffer 并 publish 供发起方拉取
target_buffers = {}
_, mr_info_a = target_backend.get_pending_shared_buffer(None, 30.0)
assert mr_info_a is not None, "get_shared_buffer timed out"
target_ep.register_remote_memory_region(
    mr_info_a["mr_key"],
    {
        "addr": mr_info_a["addr"],
        "rkey": mr_info_a["rkey"],
        "length": mr_info_a["length"],
    },
)
buf_id_t, ptr_t, key_t = _alloc_shared_buffer(target_ep, "buf_t", 16, target_buffers)
target_backend.register_local_buffer(
    buf_id_t,
    {
        "mr_key": target_buffers[buf_id_t][0],
        "addr": target_buffers[buf_id_t][1],
        "rkey": target_buffers[buf_id_t][2],
        "length": target_buffers[buf_id_t][3],
    },
)
ctypes.memset(ctypes.c_void_p(ptr_t), 1, 16)

# 发起方：从对端拉取已 publish 的 buffer
mr_info_t = initiator_stub.get_local_buffer(buf_id_t)
remote_key_t = mr_info_t["mr_key"]
initiator_ep.register_remote_memory_region(
    remote_key_t,
    {
        "addr": mr_info_t["addr"],
        "rkey": mr_info_t["rkey"],
        "length": mr_info_t["length"],
    },
)

# RDMA write: initiator writes 8 bytes from its buffer to target's buffer
slot = initiator_ep.write([(key_a, remote_key_t, 0, 0, 8)], None)
slot.wait()

# Target's buffer: first 8 bytes should now be zeros (written by initiator)
first_8 = ctypes.string_at(ptr_t, 8)
assert all(b == 0 for b in first_8), "First 8 bytes should be zeros"
last_8 = ctypes.string_at(ptr_t + 8, 8)
assert all(b == 1 for b in last_8), "Last 8 bytes should remain ones"
print("Remote buffer after RDMA write: first 8 bytes = 0, last 8 bytes = 1")

initiator_stub.close()
initiator_server.stop()
target_server.stop()
print("run rdma rc write (C++ ZMQ rendezvous) example successful")
