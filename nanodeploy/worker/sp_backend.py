from __future__ import annotations

from functools import lru_cache
from typing import Literal, Protocol, runtime_checkable

import torch
import torch.distributed as dist


SPBackend = Literal["hao_basic", "nccl"]


@runtime_checkable
class MLAAllToAllBufferProtocol(Protocol):
    @property
    def local_buffer(self) -> torch.Tensor: ...

    def connect_full_mesh(self, group: dist.ProcessGroup) -> None: ...

    def all_to_all_ll(
        self,
        x: torch.Tensor,
        is_transpose: bool = False,
        mask: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
        dst_row_indices: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


class MLAAllToAllBackendFactoryProtocol(Protocol):
    @staticmethod
    def get_buffer_size_hint(
        max_dispatch_per_msg: int,
        max_bs: int,
        max_msg_size: int,
        itemsize: int,
    ) -> int: ...

    def create_buffer(
        self,
        *,
        max_dispatch_per_msg: int,
        max_bs: int,
        rank: int,
        world_size: int,
        buffer_size_bytes: int,
    ) -> MLAAllToAllBufferProtocol: ...


@lru_cache(maxsize=1)
def _resolve_hao_symbols():
    import dlslime

    return getattr(dlslime, "AllToAllBuffer", None), getattr(dlslime, "KernelImpl", None)


@lru_cache(maxsize=1)
def _resolve_hao_dst_row_indices_version() -> int:
    import dlslime

    return int(getattr(dlslime, "ALLTOALL_DST_ROW_INDICES_VERSION", 0))


def _group_size(group: dist.ProcessGroup) -> int:
    size = getattr(group, "size", None)
    if callable(size):
        return size()
    return dist.get_world_size(group)


def _maybe_get_local_buffer(buffer) -> torch.Tensor | None:
    getter = getattr(buffer, "get_local_buffer", None)
    if callable(getter):
        candidate = getter()
        if isinstance(candidate, torch.Tensor):
            return candidate

    candidate = getattr(buffer, "local_buffer", None)
    if isinstance(candidate, torch.Tensor):
        return candidate

    return None


class HaoAllToAllBufferAdapter:
    def __init__(
        self,
        *,
        max_dispatch_per_msg: int,
        max_bs: int,
        rank: int,
        world_size: int,
        buffer_size_bytes: int,
    ):
        buffer_cls, kernel_impl = _resolve_hao_symbols()
        if buffer_cls is None or kernel_impl is None:
            raise RuntimeError(
                "sp_backend='hao_basic' requires a DLSlime build that exports "
                "AllToAllBuffer and KernelImpl."
            )

        self.rank = rank
        self.world_size = world_size
        self.max_bs = max_bs
        self.buffer_size_bytes = buffer_size_bytes
        self._buffer = buffer_cls(rank, world_size, max_bs, buffer_size_bytes)
        self._kernel_impl = kernel_impl.Basic
        self._dst_row_indices_version = _resolve_hao_dst_row_indices_version()
        self._native_local_buffer = _maybe_get_local_buffer(self._buffer)
        self._compat_mode = self._native_local_buffer is None
        self._non_transpose_input_scratch: torch.Tensor | None = None

        if self._native_local_buffer is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._staging_local_buffer = torch.empty(
                buffer_size_bytes, dtype=torch.uint8, device=device
            )
        else:
            self._staging_local_buffer = None

    @property
    def local_buffer(self) -> torch.Tensor:
        if self._native_local_buffer is not None:
            return self._native_local_buffer
        assert self._staging_local_buffer is not None
        return self._staging_local_buffer

    def connect_full_mesh(self, group: dist.ProcessGroup) -> None:
        my_handle_info = self._buffer.get_ipc_handle_info()
        all_handle_infos = [None for _ in range(_group_size(group))]
        dist.all_gather_object(all_handle_infos, my_handle_info, group=group)
        self._buffer.connect_full_mesh(all_handle_infos)

    def all_to_all_ll(
        self,
        x: torch.Tensor,
        is_transpose: bool = False,
        mask: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
        dst_row_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if dst_row_indices is not None:
            if self._compat_mode:
                raise NotImplementedError(
                    "hao_basic dst_row_indices require a native DLSlime build; "
                    "compat mode is unsupported."
                )
            if self._dst_row_indices_version < 1:
                raise RuntimeError(
                    "hao_basic destination-aware Q routing requires a DLSlime "
                    "build with ALLTOALL_DST_ROW_INDICES_VERSION >= 1."
                )
            if is_transpose:
                raise NotImplementedError(
                    "hao_basic dst_row_indices only support non-transpose all-to-all."
                )
            if mask is not None or offsets is not None:
                raise ValueError(
                    "hao_basic dst_row_indices cannot be combined with mask or offsets."
                )
        if offsets is not None and self._compat_mode:
            raise NotImplementedError(
                "hao_basic offsets require a native DLSlime build; compat mode is unsupported."
            )
        if offsets is not None and is_transpose:
            raise NotImplementedError(
                "hao_basic offsets only support non-transpose all-to-all."
            )

        backend_x = x
        backend_mask = mask
        backend_offsets = offsets
        backend_is_transpose = is_transpose

        # NanoDeploy's MLA path keeps `mask` at [world_size, max_bs], while the
        # `Q` payload only passes the active master-rank rows (`bs <= max_bs`).
        # Native hao_basic currently expects masked non-transpose inputs to have
        # exactly `max_bs` rows, so pad inactive slots here to preserve the
        # existing NanoDeploy call contract.
        if offsets is None and mask is not None and not is_transpose:
            backend_x = self._pad_masked_non_transpose_input(x, mask)

        if self._compat_mode and mask is not None:
            backend_mask = mask.transpose(0, 1).contiguous()
            if is_transpose:
                backend_x = self._collapse_transposed_input(x, mask)
                backend_is_transpose = False

        if dst_row_indices is None:
            output = self._buffer.all_to_all(
                backend_x,
                impl=self._kernel_impl,
                is_transpose=backend_is_transpose,
                mask=backend_mask,
                offsets=backend_offsets,
            )
        else:
            output = self._buffer.all_to_all(
                backend_x,
                impl=self._kernel_impl,
                is_transpose=backend_is_transpose,
                mask=None,
                offsets=None,
                dst_row_indices=dst_row_indices,
            )

        if self._compat_mode and mask is not None:
            self._patch_self_slice(output)

        return output

    def _collapse_transposed_input(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(
                f"hao_basic transpose compatibility expects x to be 2D, got {x.ndim}D"
            )
        if mask.ndim != 2:
            raise ValueError(
                f"hao_basic transpose compatibility expects mask to be 2D, got {mask.ndim}D"
            )
        if mask.size(0) != self.world_size:
            raise ValueError(
                "hao_basic transpose compatibility expects mask shape "
                f"[world_size, batch], got {tuple(mask.shape)}"
            )

        batch_size = mask.size(1)
        expected_rows = self.world_size * batch_size
        if x.size(0) != expected_rows:
            raise ValueError(
                "hao_basic transpose compatibility expects x shape "
                f"[world_size * batch, msg], got {tuple(x.shape)}"
            )

        active_targets = mask.sum(dim=0)
        if torch.any(active_targets > 1):
            raise NotImplementedError(
                "Compat hao_basic adapter only supports a single target per slot on "
                "masked transpose paths."
            )

        x_3d = x.view(self.world_size, batch_size, x.size(1))
        target_index = mask.argmax(dim=0)
        slot_index = torch.arange(batch_size, device=x.device)
        return x_3d[target_index, slot_index].contiguous()

    def _pad_masked_non_transpose_input(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(
                f"hao_basic masked non-transpose path expects x to be 2D, got {x.ndim}D"
            )
        if mask.ndim != 2:
            raise ValueError(
                f"hao_basic masked non-transpose path expects mask to be 2D, got {mask.ndim}D"
            )
        if mask.size(0) != self.world_size or mask.size(1) != self.max_bs:
            raise ValueError(
                "hao_basic masked non-transpose path expects mask shape "
                f"[world_size, max_bs], got {tuple(mask.shape)}"
            )
        if x.size(0) > self.max_bs:
            raise ValueError(
                "hao_basic masked non-transpose path expects x rows to be <= max_bs, "
                f"got rows={x.size(0)} max_bs={self.max_bs}"
            )
        if x.size(0) == self.max_bs:
            return x

        scratch = self._non_transpose_input_scratch
        if (
            scratch is None
            or scratch.shape != (self.max_bs, x.size(1))
            or scratch.dtype != x.dtype
            or scratch.device != x.device
        ):
            scratch = torch.empty(
                (self.max_bs, x.size(1)),
                dtype=x.dtype,
                device=x.device,
            )
            self._non_transpose_input_scratch = scratch

        scratch.zero_()
        scratch[: x.size(0)].copy_(x)
        return scratch

    def _patch_self_slice(self, output: torch.Tensor) -> None:
        staging = self.local_buffer.view(dtype=output.dtype)[: output.numel()].view_as(output)
        output[self.rank].copy_(staging[self.rank])


class NcclStaticAllToAllBufferAdapter:
    def __init__(
        self,
        *,
        max_dispatch_per_msg: int,
        max_bs: int,
        rank: int,
        world_size: int,
        buffer_size_bytes: int,
    ):
        self.max_dispatch_per_msg = max_dispatch_per_msg
        self.max_bs = max_bs
        self.rank = rank
        self.world_size = world_size
        self.buffer_size_bytes = buffer_size_bytes
        self._group: dist.ProcessGroup | None = None
        self._scratch: dict[
            tuple[str, tuple[int, ...], torch.dtype, torch.device], torch.Tensor
        ] = {}

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._local_buffer = torch.zeros(buffer_size_bytes, dtype=torch.uint8, device=device)

    @property
    def local_buffer(self) -> torch.Tensor:
        return self._local_buffer

    def connect_full_mesh(self, group: dist.ProcessGroup) -> None:
        self._group = group

    def all_to_all_ll(
        self,
        x: torch.Tensor,
        is_transpose: bool = False,
        mask: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
        dst_row_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if dst_row_indices is not None:
            raise NotImplementedError(
                "NCCL static all-to-all does not use DLSlime dst_row_indices."
            )
        if self._group is None:
            raise RuntimeError("NCCL static all-to-all buffer is not connected to a group.")
        if x.ndim != 2:
            raise ValueError(f"NCCL static all-to-all expects 2D input, got {x.ndim}D")
        comm_bs = self._get_comm_bs(mask, x, is_transpose=is_transpose)
        if offsets is not None and offsets.numel() != self.world_size + 1:
            raise ValueError(
                "NCCL static all-to-all expects offsets shape [world_size + 1], "
                f"got {tuple(offsets.shape)}"
            )
        if offsets is not None and is_transpose:
            raise NotImplementedError("NCCL static offsets only support Q all-to-all.")

        msg_dim = x.size(1)
        if is_transpose:
            send = self._build_transpose_send_buffer(x, mask, msg_dim, comm_bs)
            recv = self._all_to_all_equal(send, "transpose_recv")
            output = recv.view(self.world_size * comm_bs, msg_dim)
            output.add_(self._local_patch(output))
            return output

        send = self._build_non_transpose_send_buffer(x, mask, msg_dim, comm_bs)
        recv = self._all_to_all_equal(send, "q_recv")
        if offsets is None:
            output = recv.view(self.world_size * comm_bs, msg_dim)
            output.add_(self._local_patch(output))
            return output

        if mask is None:
            raise ValueError("NCCL static Q all-to-all with offsets requires a mask.")
        recv_mask = self._all_to_all_mask(mask, comm_bs)
        return self._pack_q_output(recv, recv_mask, offsets, msg_dim, comm_bs)

    def _get_comm_bs(
        self,
        mask: torch.Tensor | None,
        x: torch.Tensor,
        *,
        is_transpose: bool,
    ) -> int:
        if mask is not None:
            if mask.ndim != 2 or mask.size(0) != self.world_size:
                raise ValueError(
                    "NCCL static all-to-all expects mask shape "
                    f"[world_size, comm_bs], got {tuple(mask.shape)}"
                )
            if mask.size(1) > self.max_bs:
                raise ValueError(
                    "NCCL static all-to-all mask comm_bs must be <= max_bs, "
                    f"got comm_bs={mask.size(1)} max_bs={self.max_bs}"
                )
            return int(mask.size(1))

        if is_transpose:
            if x.size(0) % self.world_size != 0:
                raise ValueError(
                    "NCCL static transpose input rows must be divisible by world_size, "
                    f"got rows={x.size(0)} world_size={self.world_size}"
                )
            comm_bs = x.size(0) // self.world_size
        else:
            comm_bs = x.size(0)

        if comm_bs > self.max_bs:
            raise ValueError(
                "NCCL static all-to-all comm_bs must be <= max_bs, "
                f"got comm_bs={comm_bs} max_bs={self.max_bs}"
            )
        return int(comm_bs)

    def _scratch_tensor(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        storage_shape = self._scratch_storage_shape(name, shape)
        key = (name, storage_shape, dtype, device)
        tensor = self._scratch.get(key)
        if tensor is None:
            tensor = torch.empty(storage_shape, dtype=dtype, device=device)
            self._scratch[key] = tensor
        return self._scratch_view(name, tensor, shape)

    def _scratch_storage_shape(
        self,
        name: str,
        shape: tuple[int, ...],
    ) -> tuple[int, ...]:
        if name in {"q_send", "q_recv", "transpose_send", "transpose_recv"}:
            if len(shape) != 3 or shape[0] != self.world_size:
                raise ValueError(f"Invalid NCCL scratch shape for {name}: {shape}")
            if shape[1] > self.max_bs:
                raise ValueError(
                    f"Invalid NCCL scratch comm_bs for {name}: "
                    f"comm_bs={shape[1]} max_bs={self.max_bs}"
                )
            return (self.world_size * self.max_bs, shape[2])

        if name == "recv_mask":
            if len(shape) != 2 or shape[0] != self.world_size:
                raise ValueError(f"Invalid NCCL scratch shape for {name}: {shape}")
            if shape[1] > self.max_bs:
                raise ValueError(
                    f"Invalid NCCL scratch comm_bs for {name}: "
                    f"comm_bs={shape[1]} max_bs={self.max_bs}"
                )
            return (self.world_size * self.max_bs,)

        if name == "q_padded":
            if len(shape) != 2:
                raise ValueError(f"Invalid NCCL scratch shape for {name}: {shape}")
            if shape[0] > self.max_bs:
                raise ValueError(
                    f"Invalid NCCL scratch comm_bs for {name}: "
                    f"comm_bs={shape[0]} max_bs={self.max_bs}"
                )
            return (self.max_bs, shape[1])

        if name == "q_packed_output":
            if len(shape) != 2:
                raise ValueError(f"Invalid NCCL scratch shape for {name}: {shape}")
            if shape[0] > self.world_size * self.max_bs:
                raise ValueError(
                    f"Invalid NCCL scratch rows for {name}: "
                    f"rows={shape[0]} max_rows={self.world_size * self.max_bs}"
                )
            return (self.world_size * self.max_bs, shape[1])

        return shape

    def _scratch_view(
        self,
        name: str,
        tensor: torch.Tensor,
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        if name in {"q_send", "q_recv", "transpose_send", "transpose_recv"}:
            rows = shape[0] * shape[1]
            return tensor[:rows].view(shape)

        if name == "recv_mask":
            rows = shape[0] * shape[1]
            return tensor[:rows].view(shape)

        if name == "q_padded":
            return tensor[: shape[0], : shape[1]]

        if name == "q_packed_output":
            return tensor[: shape[0], : shape[1]]

        return tensor

    def _build_non_transpose_send_buffer(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        msg_dim: int,
        comm_bs: int,
    ) -> torch.Tensor:
        if x.size(0) > comm_bs:
            raise ValueError(
                "NCCL static non-transpose input rows must be <= comm_bs, "
                f"got rows={x.size(0)} comm_bs={comm_bs}"
            )

        padded = self._scratch_tensor(
            "q_padded",
            (comm_bs, msg_dim),
            x.dtype,
            x.device,
        )
        padded.zero_()
        padded[: x.size(0)].copy_(x)

        send = self._scratch_tensor(
            "q_send",
            (self.world_size, comm_bs, msg_dim),
            x.dtype,
            x.device,
        )
        send.copy_(padded.unsqueeze(0).expand(self.world_size, -1, -1))
        if mask is not None:
            send.masked_fill_(mask.to(device=x.device).unsqueeze(-1) == 0, 0)
        return send

    def _build_transpose_send_buffer(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        msg_dim: int,
        comm_bs: int,
    ) -> torch.Tensor:
        expected_rows = self.world_size * comm_bs
        if x.size(0) != expected_rows:
            raise ValueError(
                "NCCL static transpose input rows must equal world_size * comm_bs, "
                f"got rows={x.size(0)} expected={expected_rows}"
            )

        send = self._scratch_tensor(
            "transpose_send",
            (self.world_size, comm_bs, msg_dim),
            x.dtype,
            x.device,
        )
        send.copy_(x.view(self.world_size, comm_bs, msg_dim))
        if mask is not None:
            send.masked_fill_(mask.to(device=x.device).unsqueeze(-1) == 0, 0)
        return send

    def _all_to_all_equal(self, send: torch.Tensor, scratch_name: str) -> torch.Tensor:
        recv = self._scratch_tensor(scratch_name, tuple(send.shape), send.dtype, send.device)
        assert self._group is not None
        dist.all_to_all_single(recv, send.contiguous(), group=self._group)
        return recv

    def _all_to_all_mask(self, mask: torch.Tensor, comm_bs: int) -> torch.Tensor:
        send_mask = mask.to(device=self.local_buffer.device, dtype=torch.int32).contiguous()
        recv_mask = self._scratch_tensor(
            "recv_mask",
            (self.world_size, comm_bs),
            torch.int32,
            send_mask.device,
        )
        assert self._group is not None
        dist.all_to_all_single(recv_mask, send_mask, group=self._group)
        return recv_mask

    def _pack_q_output(
        self,
        recv: torch.Tensor,
        recv_mask: torch.Tensor,
        offsets: torch.Tensor,
        msg_dim: int,
        comm_bs: int,
    ) -> torch.Tensor:
        output = self._scratch_tensor(
            "q_packed_output",
            (self.world_size * comm_bs, msg_dim),
            recv.dtype,
            recv.device,
        )
        output.zero_()

        valid = recv_mask.to(device=recv.device) != 0
        prefix = torch.cumsum(valid.to(torch.int32), dim=1) - 1
        offsets = offsets.to(device=recv.device, dtype=torch.int32)
        destinations = offsets[:-1].view(self.world_size, 1) + prefix
        destinations = torch.where(valid, destinations, torch.zeros_like(destinations))

        values = recv * valid.unsqueeze(-1).to(dtype=recv.dtype)
        output.index_add_(0, destinations.reshape(-1).to(torch.long), values.view(-1, msg_dim))
        output.add_(self._local_patch(output))
        return output

    def _local_patch(self, output: torch.Tensor) -> torch.Tensor:
        return self.local_buffer.view(dtype=output.dtype)[: output.numel()].view_as(output)


class HaoBasicBackendFactory:
    @staticmethod
    def get_buffer_size_hint(
        max_dispatch_per_msg: int,
        max_bs: int,
        max_msg_size: int,
        itemsize: int,
    ) -> int:
        return max_dispatch_per_msg * max_bs * max_msg_size * itemsize

    def create_buffer(
        self,
        *,
        max_dispatch_per_msg: int,
        max_bs: int,
        rank: int,
        world_size: int,
        buffer_size_bytes: int,
    ) -> MLAAllToAllBufferProtocol:
        return HaoAllToAllBufferAdapter(
            max_dispatch_per_msg=max_dispatch_per_msg,
            max_bs=max_bs,
            rank=rank,
            world_size=world_size,
            buffer_size_bytes=buffer_size_bytes,
        )


class NcclStaticBackendFactory:
    @staticmethod
    def get_buffer_size_hint(
        max_dispatch_per_msg: int,
        max_bs: int,
        max_msg_size: int,
        itemsize: int,
    ) -> int:
        return max_dispatch_per_msg * max_bs * max_msg_size * itemsize

    def create_buffer(
        self,
        *,
        max_dispatch_per_msg: int,
        max_bs: int,
        rank: int,
        world_size: int,
        buffer_size_bytes: int,
    ) -> MLAAllToAllBufferProtocol:
        return NcclStaticAllToAllBufferAdapter(
            max_dispatch_per_msg=max_dispatch_per_msg,
            max_bs=max_bs,
            rank=rank,
            world_size=world_size,
            buffer_size_bytes=buffer_size_bytes,
        )


def create_sp_backend_factory(
    backend: SPBackend,
) -> MLAAllToAllBackendFactoryProtocol:
    if backend == "hao_basic":
        return HaoBasicBackendFactory()
    if backend == "nccl":
        return NcclStaticBackendFactory()
    raise ValueError(f"Unsupported SP backend: {backend}")
