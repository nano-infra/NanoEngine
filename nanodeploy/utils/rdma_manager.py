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

    def send_int64_tensor(self, tensor: torch.Tensor):
        """
        [极速版] 仅发送 torch.int64 类型的 Tensor。
        无 Pickle，无元数据，只有 4 字节长度头 + 裸数据。
        """
        # 1. 检查类型 (可选，确保安全)
        # if tensor.dtype != torch.int64:
        #     raise ValueError("Only torch.int64 is supported")

        # 2. 计算字节大小
        # int64 占 8 字节
        num_elements = tensor.numel()
        data_len = num_elements * 8

        total_len = 4 + data_len
        if total_len > self.buffer_size:
            raise ValueError(
                f"Tensor bytes {total_len} exceeds buffer size {self.buffer_size}"
            )

        # 3. 写入 Header (4 bytes, Big-endian)
        # 这里表示有效载荷的字节长度
        header_bytes = struct.pack(">I", data_len)
        header_tensor = torch.frombuffer(header_bytes, dtype=torch.uint8)
        self.tensor_pool[0:4].copy_(header_tensor)

        # 4. 写入 Body (Zero-copy view copy)
        # 将 int64 tensor 视为 uint8 视图，直接 copy 到 uint8 的 pinned buffer
        # 这种方式比 numpy 中转快且稳定
        src_view = tensor.view(torch.uint8).reshape(-1)  # 展平以匹配 buffer 维度
        self.tensor_pool[4 : 4 + data_len].copy_(src_view)

        # 5. Send
        try:
            self.buffer.send(total_len)
        except TypeError:
            self.buffer.send(None)

        self.buffer.wait_send()

    def wait_int64_tensor(self) -> torch.Tensor:
        """
        [极速版] 接收裸数据并转回 int64 Tensor
        """
        # 1. Wait RDMA
        self.buffer.wait_recv()

        # 2. Read Header
        # 安全读取 4 字节长度
        header_bytes = self.tensor_pool[0:4].cpu().numpy().tobytes()
        data_len = struct.unpack(">I", header_bytes)[0]

        if data_len == 0:
            # 返回空 Tensor
            return torch.empty(0, dtype=torch.int64)

        # 3. Read Body & Cast directly
        # 取出有效数据的视图 (In Pinned Memory)
        raw_bytes_view = self.tensor_pool[4 : 4 + data_len]

        # 将 uint8 视图转为 int64 视图
        # 注意：这里得到的 tensor 内存仍在 pinned buffer 上
        # 如果后续要 .tolist() 或者 .to(device)，会自动发生拷贝，这里是安全的
        tensor_out = raw_bytes_view.view(torch.int64)

        # 如果你需要拷贝出来（防止 buffer 被下一轮覆盖），可以加 .clone()
        # 但考虑到你后续是 tensor_out.tolist()，直接返回 view 即可，省一次拷贝
        return tensor_out
