import socket
from dataclasses import dataclass

import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception as e:
        return f"Fail to get local ip：{str(e)}"


@dataclass
class DistContext:
    rank: int = 0
    world_size: int = 1

    attention_dp: int = 1
    attention_tp: int = 1
    attention_sp: int = 1

    ffn_dp: int = 1
    ffn_ep: int = 1
    ffn_tp: int = 1

    pp: int = 1

    @property
    def attn_dp_rank(self):
        """Get the current process rank in attention data parallel group"""
        return dist.get_rank(self.attn_dp_group)

    @property
    def attn_dp_world_size(self):
        """Get the total number of processes in attention data parallel group"""
        return dist.get_world_size(self.attn_dp_group)

    @property
    def attn_dp_group(self):
        """Get the communication group for attention data parallelism"""
        return self.attn_device_mesh.get_group("attn_dp")

    @property
    def attn_sp_rank(self):
        """Get the current process rank in attention expert parallel group"""
        return dist.get_rank(self.attn_sp_group)

    @property
    def attn_sp_world_size(self):
        """Get the total number of processes in attention expert parallel group"""
        return dist.get_world_size(self.attn_sp_group)

    @property
    def attn_sp_group(self):
        """Get the communication group for attention expert parallelism"""
        return self.attn_device_mesh.get_group("attn_sp")

    @property
    def attn_tp_rank(self):
        """Get the current process rank in attention tensor parallel group"""
        return dist.get_rank(self.attn_tp_group)

    @property
    def attn_tp_world_size(self):
        """Get the total number of processes in attention tensor parallel group"""
        return dist.get_world_size(self.attn_tp_group)

    @property
    def attn_tp_group(self):
        """Get the communication group for attention tensor parallelism"""
        return self.attn_device_mesh.get_group("attn_tp")

    @property
    def ffn_dp_rank(self):
        """Get the current process rank in FFN data parallel group"""
        return dist.get_rank(self.ffn_dp_group)

    @property
    def ffn_dp_world_size(self):
        """Get the total number of processes in FFN data parallel group"""
        return dist.get_world_size(self.ffn_dp_group)

    @property
    def ffn_dp_group(self):
        """Get the communication group for FFN data parallelism"""
        return self.ffn_device_mesh.get_group("ffn_dp")

    @property
    def ffn_ep_rank(self):
        """Get the current process rank in FFN expert parallel group"""
        return dist.get_rank(self.ffn_ep_group)

    @property
    def ffn_ep_world_size(self):
        """Get the total number of processes in FFN expert parallel group"""
        return dist.get_world_size(self.ffn_ep_group)

    @property
    def ffn_ep_group(self):
        """Get the communication group for FFN expert parallelism"""
        return self.ffn_device_mesh.get_group("ffn_ep")

    @property
    def ffn_tp_rank(self):
        """Get the current process rank in FFN tensor parallel group"""
        return dist.get_rank(self.ffn_tp_group)

    @property
    def ffn_tp_world_size(self):
        """Get the total number of processes in FFN tensor parallel group"""
        return dist.get_world_size(self.ffn_tp_group)

    @property
    def ffn_tp_group(self):
        """Get the communication group for FFN tensor parallelism"""
        return self.ffn_device_mesh.get_group("ffn_tp")

    @property
    def local_rank(self):
        """Get the local rank of the current process"""
        return dist.get_node_local_rank(self.rank % 8)

    @property
    def cpu_world_rank(self):
        """Get the current process rank in CPU global communication group"""
        return dist.get_rank(self.cpu_world_group)

    @property
    def cpu_world_size(self):
        """Get the total number of processes in CPU global communication group"""
        return dist.get_world_size(self.cpu_world_group)

    @property
    def cpu_world_group(self):
        """Get the global communication group for CPU world"""
        return self.cpu_world_mesh.get_group("world")

    @property
    def cuda_world_rank(self):
        """Get the current process rank in CUDA global communication group"""
        return dist.get_rank(self.cuda_world_group)

    @property
    def cuda_world_size(self):
        """Get the total number of processes in CUDA global communication group"""
        return dist.get_world_size(self.cuda_world_group)

    @property
    def cuda_world_group(self):
        """Get the total number of processes in CUDA global communication group"""
        return self.cuda_world_mesh.get_group("world")

    def __post_init__(self):
        self.cpu_world_mesh = init_device_mesh(
            "cpu", (self.world_size,), mesh_dim_names=("world",)
        )

        self.cuda_world_mesh = init_device_mesh(
            "cuda", (self.world_size,), mesh_dim_names=("world",)
        )

        self.attn_device_mesh = init_device_mesh(
            "cuda",
            (self.attention_dp, self.attention_sp, self.attention_tp),
            mesh_dim_names=("attn_dp", "attn_sp", "attn_tp"),
        )

        self.ffn_device_mesh = init_device_mesh(
            "cuda",
            (self.ffn_dp, self.ffn_ep, self.ffn_tp),
            mesh_dim_names=("ffn_dp", "ffn_ep", "ffn_tp"),
        )


_DIST_CONTEXT: DistContext


def get_dist_context():
    return _DIST_CONTEXT


def set_dist_context(
    rank,
    world_size,
    attention_dp=1,
    attention_sp=1,
    attention_tp=1,
    ffn_dp=1,
    ffn_ep=1,
    ffn_tp=1,
    pp=1,
):
    global _DIST_CONTEXT
    _DIST_CONTEXT = DistContext(
        rank=rank,
        world_size=world_size,
        attention_dp=attention_dp,
        attention_sp=attention_sp,
        attention_tp=attention_tp,
        ffn_dp=ffn_dp,
        ffn_ep=ffn_ep,
        ffn_tp=ffn_tp,
        pp=pp,
    )


def reset_dist_context():
    global _CONTEXT
    _CONTEXT = DistContext()
