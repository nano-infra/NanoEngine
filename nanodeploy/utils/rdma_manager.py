import pickle
import struct

import torch


def get_available_nics():
    """
    延迟加载获取网卡列表，避免 Ray 在序列化函数时因为包含 PyCapsule 而报错。
    """
    try:
        from dlslime import available_nic

        return available_nic()
    except ImportError:
        print("Warning: 'dlslime' not found. Returning empty NIC list.")
        return []


class RDMAManager:
    def __init__(self, nic_name: str, buffer_size: int = 64 * 1024 * 1024):
        """
        Args:
            nic_name: 指定使用的物理网卡名称，例如 'mlx5_0'
            buffer_size: 缓冲区大小
        """
        self.buffer_size = buffer_size

        # [修复核心] 在 __init__ 内部导入 C++ 模块
        # 这样只有在 Worker 进程真正实例化对象时才会加载 C++ 扩展，
        # 而不是在 Ray 传输 Class 定义时。
        try:
            from dlslime import _slime_c
        except ImportError:
            raise ImportError(
                "Could not import 'dlslime'. Please ensure it is installed."
            )

        print(f"[RDMA] Initializing on NIC: {nic_name} (Buffer: CPU Pinned Memory)")

        # 创建 endpoint
        self.endpoint = _slime_c.rdma_endpoint(nic_name, 1, "RoCE", 1)

        # 使用 CPU Pinned Memory
        self.tensor_pool = torch.zeros(
            buffer_size, dtype=torch.uint8, device="cpu"
        ).pin_memory()

        # 注册 RDMA Buffer
        # 注意：_slime_c 在这里是局部变量，很安全
        self.buffer = _slime_c.rdma_buffer(
            self.endpoint, self.tensor_pool.data_ptr(), 0, self.tensor_pool.numel()
        )

        # 绑定 recv_buffer 到 endpoint (如果库需要显式绑定，参考你的原始逻辑)
        # 假设 _slime_c.rdma_buffer 构造时已经关联了 endpoint

    def get_context(self):
        return (
            self.endpoint.get_data_context_info(),
            self.endpoint.get_meta_context_info(),
        )

    def connect(self, remote_ctx):
        remote_data_ctx, remote_meta_ctx = remote_ctx
        self.endpoint.context_connect(remote_data_ctx, remote_meta_ctx)

    def post_send(self, data_bytes: bytes):
        """
        异步发送的第一步：拷贝数据到 Pinned Memory 并提交 RDMA Send 请求。
        函数会立即返回，不等待传输完成。
        """
        size = len(data_bytes)
        if size > self.buffer_size:
            raise ValueError(f"Data size {size} exceeds buffer size {self.buffer_size}")

        # 1. CPU Copy to Pinned Memory
        # src_tensor = torch.frombuffer(data_bytes, dtype=torch.uint8)
        # self.tensor_pool[:size].copy_(src_tensor)
        dst_view = memoryview(self.tensor_pool[:size].numpy())
        dst_view[:] = data_bytes

        # 2. Post Send Request (Non-blocking)
        self.buffer.send(None)

    def wait_send(self):
        """
        异步发送的第二步：等待之前的 Send 请求完成。
        """
        self.buffer.wait_send()

    def send_data(self, data_bytes: bytes):
        """
        同步发送模式
        """
        size = len(data_bytes)
        if size > self.buffer_size:
            raise ValueError(f"Data size {size} exceeds buffer size {self.buffer_size}")

        # CPU Copy to Pinned Memory
        src_tensor = torch.frombuffer(data_bytes, dtype=torch.uint8)
        self.tensor_pool[:size].copy_(src_tensor)

        self.buffer.send(None)
        self.buffer.wait_send()

    def recv_data(self, size: int) -> bytes:
        self.buffer.recv(None)
        self.buffer.wait_recv()
        # 直接返回 CPU 内存的 bytes
        return self.tensor_pool[:size].numpy().tobytes()

    def post_recv(self):
        """
        [异步] 提交接收请求
        """
        self.buffer.recv(None)
