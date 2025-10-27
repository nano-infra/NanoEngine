from dataclasses import dataclass

from torch.distributed.device_mesh import init_device_mesh


@dataclass
class ParallelContext:
    rank: int = 0
    world_size: int = 1

    dp: int = 1
    ep: int = 1
    pp: int = 1
    tp: int = 1
    sp: int = 1

    @property
    def attn_dp_group(self):
        return self.attn_device_mesh.get_group("dp")

    @property
    def attn_sp_group(self):
        return self.attn_device_mesh.get_group("sp")

    @property
    def attn_tp_group(self):
        return self.attn_device_mesh.get_group("tp")

    @property
    def ffn_ep_group(self):
        return self.ffn_device_mesh.get_group("ep")

    @property
    def cpu_world_group(self):
        return self.cpu_world_mesh.get_group("world")

    def __post_init__(self):
        if self.world_size == 1:
            return
        # initialize attn and ffn parallel group

        self.cpu_world_mesh = init_device_mesh(
            "cpu",
            [self.world_size],
            mesh_dim_names=["world"]
        )

        self.attn_device_mesh = init_device_mesh(
            "cuda",
            [self.dp, self.sp, self.tp],
            mesh_dim_names=["dp", "sp", "tp"]
        )

        self.ffn_device_mesh = init_device_mesh(
            "cuda",
            [self.ep, self.tp],
            mesh_dim_names=["ep", "tp"]
        )

_PARALLEL_CONTEXT = ParallelContext()

def get_parallel_context():
    return _PARALLEL_CONTEXT

def set_parallel_context(
    rank, world_size, dp=1, ep=1, pp=1, tp=1, sp=1):
    global _PARALLEL_CONTEXT
    _PARALLEL_CONTEXT = ParallelContext(rank=rank, world_size=world_size, dp=dp, ep=ep, pp=pp, tp=tp, sp=sp)

def reset_parallel_context():
    global _CONTEXT
    _CONTEXT = ParallelContext()

