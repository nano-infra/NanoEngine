"""Minimal NVLS communicator storage for K3 SP collectives."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from .utils import cache_once, load_jit


@cache_once
def _communicator_class():
    import tvm_ffi

    module = load_jit(
        "communicator",
        cuda_files=["distributed/communicator.cuh"],
        cuda_wrappers=[("register_once", "register_communicator")],
    )
    module.register_once()

    @tvm_ffi.register_object("sgl.Communicator")
    class Communicator(tvm_ffi.Object):
        def __init__(
            self,
            rank: int,
            world_size: int,
            push_workspaces: list[torch.Tensor],
            pull_workspaces: list[torch.Tensor],
            pull_semaphores: list[torch.Tensor],
            push_counter: torch.Tensor,
            pull_mc_workspace: int | None,
        ) -> None:
            self.__ffi_init__(
                rank,
                world_size,
                push_workspaces,
                pull_workspaces,
                pull_semaphores,
                push_counter,
                pull_mc_workspace,
            )

    return Communicator


class K3SPCommunicator:
    """Own symmetric storage shared by the K3 TP collective kernels."""

    def __init__(self, group, *, max_rows: int = 512, hidden_size: int = 7168):
        from torch._C._distributed_c10d import _SymmetricMemory

        self.group = group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.device = torch.device("cuda", torch.cuda.current_device())
        push_bytes = max_rows * hidden_size * 2
        push_slots = 2 * self.world_size
        pull_bytes = 1024
        num_blocks = 256
        sem_bytes = num_blocks * 128
        push_region_bytes = push_slots * push_bytes
        total_bytes = push_region_bytes + pull_bytes + sem_bytes
        self._region = _SymmetricMemory.empty_strided_p2p(
            (total_bytes,), [1], torch.uint8, self.device, group.group_name
        )
        handle = _SymmetricMemory.rendezvous(self._region)
        peers = [
            handle.get_buffer(i, [total_bytes], torch.uint8)
            for i in range(self.world_size)
        ]
        peers[self.rank].zero_()
        torch.cuda.synchronize()
        dist.barrier(group=group)
        push = [p[:push_region_bytes].view(push_slots, push_bytes) for p in peers]
        pull = [p[push_region_bytes : push_region_bytes + pull_bytes] for p in peers]
        sem_offset = push_region_bytes + pull_bytes
        sem = [p[sem_offset:].view(num_blocks, 128) for p in peers]
        self._counter = torch.zeros(
            (num_blocks, 4), dtype=torch.uint8, device=self.device
        )
        mc_base = int(handle.multicast_ptr)
        if mc_base == 0:
            raise RuntimeError("K3 SP collectives require an NVLS multicast mapping")
        self.mc_base_ptr = mc_base
        self.pull_sem_mc_ptr = mc_base + sem_offset
        Communicator = _communicator_class()
        self.obj = Communicator(
            self.rank, self.world_size, push, pull, sem, self._counter, None
        )
        self._buffers: dict[str, tuple[torch.Tensor, int]] = {}

    def symmetric_buffer(
        self, name: str, rows: int, width: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, int]:
        from torch._C._distributed_c10d import _SymmetricMemory

        cached = self._buffers.get(name)
        if cached is None:
            region = _SymmetricMemory.empty_strided_p2p(
                (512, width), (width, 1), dtype, self.device, self.group.group_name
            )
            handle = _SymmetricMemory.rendezvous(region)
            mc_ptr = int(handle.multicast_ptr)
            if mc_ptr == 0:
                raise RuntimeError("K3 SP output buffer has no multicast mapping")
            cached = (region, mc_ptr)
            self._buffers[name] = cached
        region, mc_ptr = cached
        if rows > region.shape[0]:
            raise ValueError(f"K3 SP buffer rows={rows} exceed {region.shape[0]}")
        return region[:rows], mc_ptr


_STATE: dict[tuple[int, int], K3SPCommunicator] = {}


def get_k3_sp_communicator() -> K3SPCommunicator | None:
    from dlengine.context.distributed import get_dist_context

    ctx = get_dist_context()
    if ctx.attn_tp_world_size not in (4, 8):
        return None
    if torch.cuda.get_device_capability()[0:2] != (10, 3):
        return None
    key = (id(ctx.attn_tp_group), torch.cuda.current_device())
    state = _STATE.get(key)
    if state is None:
        state = K3SPCommunicator(ctx.attn_tp_group)
        _STATE[key] = state
    return state
