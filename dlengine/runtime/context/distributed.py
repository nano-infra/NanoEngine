from dataclasses import dataclass, field
from typing import Any, Optional

import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from dlengine.runtime.context import BaseContext, ContextManagerMixin
from dlengine.utils.network import get_local_ip


@dataclass
class DistContext(ContextManagerMixin, BaseContext):
    rank: int = 0
    world_size: int = 1

    attention_dp: int = 1
    attention_tp: int = 1
    attention_sp: int = 1

    ffn_dp: int = 1
    ffn_ep: int = 1
    ffn_tp: int = 1

    pp: int = 1
    initialize_meshes: bool = True

    cpu_world_mesh: Optional[Any] = field(default=None, init=False)
    cuda_world_mesh: Optional[Any] = field(default=None, init=False)
    attn_cpu_device_mesh: Optional[Any] = field(default=None, init=False)
    ffn_cpu_device_mesh: Optional[Any] = field(default=None, init=False)
    attn_device_mesh: Optional[Any] = field(default=None, init=False)
    ffn_device_mesh: Optional[Any] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._validate_parallelism()
        if self.initialize_meshes:
            self.reset_context()

    @classmethod
    def get_context_type(cls) -> str:
        return "distributed"

    @classmethod
    def get_context_name(cls) -> str:
        return "DistContext"

    def clear_context(self) -> None:
        self.cpu_world_mesh = None
        self.cuda_world_mesh = None
        self.attn_cpu_device_mesh = None
        self.ffn_cpu_device_mesh = None
        self.attn_device_mesh = None
        self.ffn_device_mesh = None

    def reset_context(self) -> None:
        self.clear_context()
        pp = max(1, self.pp)
        self.cpu_world_mesh = init_device_mesh(
            "cpu", (self.world_size,), mesh_dim_names=("world",)
        )
        self.cuda_world_mesh = init_device_mesh(
            "cuda", (self.world_size,), mesh_dim_names=("world",)
        )
        # The pipeline dimension is the outermost mesh axis, so global rank
        # layout is pp-major: rank = pp_idx * inner + inner_rank. This keeps the
        # attn/ffn (dp/sp/tp/ep) sub-groups confined to a single pipeline stage
        # while ``get_group("pp")`` spans stages for point-to-point transfer.
        self.attn_cpu_device_mesh = init_device_mesh(
            "cpu",
            (pp, self.attention_dp, self.attention_sp, self.attention_tp),
            mesh_dim_names=("pp", "attn_dp", "attn_sp", "attn_tp"),
        )
        self.ffn_cpu_device_mesh = init_device_mesh(
            "cpu",
            (pp, self.ffn_dp, self.ffn_ep, self.ffn_tp),
            mesh_dim_names=("pp", "ffn_dp", "ffn_ep", "ffn_tp"),
        )
        self.attn_device_mesh = init_device_mesh(
            "cuda",
            (pp, self.attention_dp, self.attention_sp, self.attention_tp),
            mesh_dim_names=("pp", "attn_dp", "attn_sp", "attn_tp"),
        )
        self.ffn_device_mesh = init_device_mesh(
            "cuda",
            (pp, self.ffn_dp, self.ffn_ep, self.ffn_tp),
            mesh_dim_names=("pp", "ffn_dp", "ffn_ep", "ffn_tp"),
        )

    def _validate_parallelism(self) -> None:
        pp = max(1, self.pp)
        attn_world_size = pp * self.attention_dp * self.attention_sp * self.attention_tp
        ffn_world_size = pp * self.ffn_dp * self.ffn_ep * self.ffn_tp
        if attn_world_size != self.world_size:
            raise ValueError(
                "attention parallelism (incl. pp) must match world_size: "
                f"{attn_world_size} != {self.world_size}"
            )
        if ffn_world_size != self.world_size:
            raise ValueError(
                "ffn parallelism (incl. pp) must match world_size: "
                f"{ffn_world_size} != {self.world_size}"
            )

    @property
    def attn_dp_rank(self) -> int:
        return dist.get_rank(self.attn_dp_group)

    @property
    def attn_dp_world_size(self) -> int:
        return dist.get_world_size(self.attn_dp_group)

    @property
    def attn_dp_group(self):
        return self.attn_device_mesh.get_group("attn_dp")

    @property
    def attn_sp_rank(self) -> int:
        return dist.get_rank(self.attn_sp_group)

    @property
    def attn_sp_world_size(self) -> int:
        return dist.get_world_size(self.attn_sp_group)

    @property
    def attn_sp_group(self):
        return self.attn_device_mesh.get_group("attn_sp")

    @property
    def attn_tp_rank(self) -> int:
        return dist.get_rank(self.attn_tp_group)

    @property
    def attn_tp_world_size(self) -> int:
        return dist.get_world_size(self.attn_tp_group)

    @property
    def attn_tp_group(self):
        return self.attn_device_mesh.get_group("attn_tp")

    @property
    def attn_cpu_dp_rank(self) -> int:
        return dist.get_rank(self.attn_cpu_dp_group)

    @property
    def attn_cpu_dp_world_size(self) -> int:
        return dist.get_world_size(self.attn_cpu_dp_group)

    @property
    def attn_cpu_dp_group(self):
        return self.attn_cpu_device_mesh.get_group("attn_dp")

    @property
    def attn_cpu_sp_rank(self) -> int:
        return dist.get_rank(self.attn_cpu_sp_group)

    @property
    def attn_cpu_sp_world_size(self) -> int:
        return dist.get_world_size(self.attn_cpu_sp_group)

    @property
    def attn_cpu_sp_group(self):
        return self.attn_cpu_device_mesh.get_group("attn_sp")

    @property
    def attn_cpu_tp_rank(self) -> int:
        return dist.get_rank(self.attn_cpu_tp_group)

    @property
    def attn_cpu_tp_world_size(self) -> int:
        return dist.get_world_size(self.attn_cpu_tp_group)

    @property
    def attn_cpu_tp_group(self):
        return self.attn_cpu_device_mesh.get_group("attn_tp")

    @property
    def ffn_dp_rank(self) -> int:
        return dist.get_rank(self.ffn_dp_group)

    @property
    def ffn_dp_world_size(self) -> int:
        return dist.get_world_size(self.ffn_dp_group)

    @property
    def ffn_dp_group(self):
        return self.ffn_device_mesh.get_group("ffn_dp")

    @property
    def ffn_ep_rank(self) -> int:
        return dist.get_rank(self.ffn_ep_group)

    @property
    def ffn_ep_world_size(self) -> int:
        return dist.get_world_size(self.ffn_ep_group)

    @property
    def ffn_ep_group(self):
        return self.ffn_device_mesh.get_group("ffn_ep")

    @property
    def ffn_tp_rank(self) -> int:
        return dist.get_rank(self.ffn_tp_group)

    @property
    def ffn_tp_world_size(self) -> int:
        return dist.get_world_size(self.ffn_tp_group)

    @property
    def ffn_tp_group(self):
        return self.ffn_device_mesh.get_group("ffn_tp")

    @property
    def ffn_cpu_dp_rank(self) -> int:
        return dist.get_rank(self.ffn_cpu_dp_group)

    @property
    def ffn_cpu_dp_world_size(self) -> int:
        return dist.get_world_size(self.ffn_cpu_dp_group)

    @property
    def ffn_cpu_dp_group(self):
        return self.ffn_cpu_device_mesh.get_group("ffn_dp")

    @property
    def ffn_cpu_ep_rank(self) -> int:
        return dist.get_rank(self.ffn_cpu_ep_group)

    @property
    def ffn_cpu_ep_world_size(self) -> int:
        return dist.get_world_size(self.ffn_cpu_ep_group)

    @property
    def ffn_cpu_ep_group(self):
        return self.ffn_cpu_device_mesh.get_group("ffn_ep")

    @property
    def ffn_cpu_tp_rank(self) -> int:
        return dist.get_rank(self.ffn_cpu_tp_group)

    @property
    def ffn_cpu_tp_world_size(self) -> int:
        return dist.get_world_size(self.ffn_cpu_tp_group)

    @property
    def ffn_cpu_tp_group(self):
        return self.ffn_cpu_device_mesh.get_group("ffn_tp")

    @property
    def local_rank(self) -> int:
        import os

        return int(os.environ.get("LOCAL_RANK", str(self.rank % 8)))

    @property
    def cpu_world_rank(self) -> int:
        return dist.get_rank(self.cpu_world_group)

    @property
    def cpu_world_size(self) -> int:
        return dist.get_world_size(self.cpu_world_group)

    @property
    def cpu_world_group(self):
        return self.cpu_world_mesh.get_group("world")

    @property
    def cuda_world_rank(self) -> int:
        return dist.get_rank(self.cuda_world_group)

    @property
    def cuda_world_size(self) -> int:
        return dist.get_world_size(self.cuda_world_group)

    @property
    def cuda_world_group(self):
        return self.cuda_world_mesh.get_group("world")

    # ------------------------------------------------------------------ #
    # Pipeline parallelism helpers                                         #
    # ------------------------------------------------------------------ #
    @property
    def pp_world_size(self) -> int:
        return max(1, self.pp)

    @property
    def pp_inner_world_size(self) -> int:
        """Number of ranks within a single pipeline stage."""
        return self.world_size // self.pp_world_size

    @property
    def pp_rank(self) -> int:
        """Index of this rank's pipeline stage (pp-major global layout)."""
        return self.rank // self.pp_inner_world_size

    @property
    def is_first_pp_stage(self) -> bool:
        return self.pp_rank == 0

    @property
    def is_last_pp_stage(self) -> bool:
        return self.pp_rank == self.pp_world_size - 1

    @property
    def pp_prev_global_rank(self) -> int:
        """Global rank of the same inner position in the previous stage."""
        return self.rank - self.pp_inner_world_size

    @property
    def pp_next_global_rank(self) -> int:
        """Global rank of the same inner position in the next stage."""
        return self.rank + self.pp_inner_world_size

    @property
    def pp_group(self):
        return self.attn_device_mesh.get_group("pp")


DistributedContext = DistContext

_DIST_CONTEXT: Optional[DistContext] = None


def get_dist_context() -> DistContext:
    if _DIST_CONTEXT is None:
        raise RuntimeError("DistContext has not been initialized")
    return _DIST_CONTEXT


def set_dist_context(
    rank: int,
    world_size: int,
    attention_dp: int = 1,
    attention_sp: int = 1,
    attention_tp: int = 1,
    ffn_dp: int = 1,
    ffn_ep: int = 1,
    ffn_tp: int = 1,
    pp: int = 1,
    initialize_meshes: bool = True,
) -> DistContext:
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
        initialize_meshes=initialize_meshes,
    )
    return _DIST_CONTEXT


def clear_dist_context() -> None:
    global _DIST_CONTEXT
    if _DIST_CONTEXT is not None:
        _DIST_CONTEXT.clear_context()
    _DIST_CONTEXT = None


def reset_dist_context() -> DistContext:
    context = get_dist_context()
    context.reset_context()
    return context


def get_distributed_context() -> DistContext:
    return get_dist_context()


def set_distributed_context(*args, **kwargs) -> DistContext:
    return set_dist_context(*args, **kwargs)


def clear_distributed_context() -> None:
    clear_dist_context()


def reset_distributed_context() -> DistContext:
    return reset_dist_context()


__all__ = [
    "DistContext",
    "DistributedContext",
    "clear_dist_context",
    "clear_distributed_context",
    "get_dist_context",
    "get_distributed_context",
    "get_local_ip",
    "reset_dist_context",
    "reset_distributed_context",
    "set_dist_context",
    "set_distributed_context",
]
