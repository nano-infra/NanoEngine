from __future__ import annotations

from functools import lru_cache
from typing import Literal, Protocol, runtime_checkable

import torch
import torch.distributed as dist


SPBackend = Literal["legacy_ll", "hao_basic"]


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
def _resolve_legacy_buffer_cls():
    from dlslime.buffer.intra.all_to_all_intra_ll_buffer import AllToAllIntraLLBuffer

    return AllToAllIntraLLBuffer


@lru_cache(maxsize=1)
def _resolve_hao_symbols():
    import dlslime

    return getattr(dlslime, "AllToAllBuffer", None), getattr(dlslime, "KernelImpl", None)


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


class LegacyIntraLLBufferAdapter:
    def __init__(
        self,
        *,
        max_dispatch_per_msg: int,
        max_bs: int,
        rank: int,
        world_size: int,
        buffer_size_bytes: int,
    ):
        buffer_cls = _resolve_legacy_buffer_cls()
        self._buffer = buffer_cls(
            max_dispatch_per_msg,
            max_bs,
            rank,
            world_size,
            buffer_size_bytes,
        )

    @property
    def local_buffer(self) -> torch.Tensor:
        return self._buffer.local_buffer

    def connect_full_mesh(self, group: dist.ProcessGroup) -> None:
        self._buffer.connect_full_mesh(group)

    def all_to_all_ll(
        self,
        x: torch.Tensor,
        is_transpose: bool = False,
        mask: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._buffer.all_to_all_ll(
            x,
            is_transpose=is_transpose,
            mask=mask,
            offsets=offsets,
        )


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
    ) -> torch.Tensor:
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

        output = self._buffer.all_to_all(
            backend_x,
            impl=self._kernel_impl,
            is_transpose=backend_is_transpose,
            mask=backend_mask,
            offsets=backend_offsets,
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


class LegacyIntraLLBackendFactory:
    @staticmethod
    def get_buffer_size_hint(
        max_dispatch_per_msg: int,
        max_bs: int,
        max_msg_size: int,
        itemsize: int,
    ) -> int:
        buffer_cls = _resolve_legacy_buffer_cls()
        return buffer_cls.get_buffer_size_hint(
            max_dispatch_per_msg,
            max_bs,
            max_msg_size,
            itemsize,
        )

    def create_buffer(
        self,
        *,
        max_dispatch_per_msg: int,
        max_bs: int,
        rank: int,
        world_size: int,
        buffer_size_bytes: int,
    ) -> MLAAllToAllBufferProtocol:
        return LegacyIntraLLBufferAdapter(
            max_dispatch_per_msg=max_dispatch_per_msg,
            max_bs=max_bs,
            rank=rank,
            world_size=world_size,
            buffer_size_bytes=buffer_size_bytes,
        )


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


def create_sp_backend_factory(
    backend: SPBackend,
) -> MLAAllToAllBackendFactoryProtocol:
    if backend == "legacy_ll":
        return LegacyIntraLLBackendFactory()
    if backend == "hao_basic":
        return HaoBasicBackendFactory()
    raise ValueError(f"Unsupported SP backend: {backend}")
