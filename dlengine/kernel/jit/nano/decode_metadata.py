from __future__ import annotations

from typing import TYPE_CHECKING

from .utils import cache_once, load_jit

if TYPE_CHECKING:
    import torch
    from tvm_ffi.module import Module


@cache_once
def _jit_decode_metadata_module() -> Module:
    return load_jit(
        "decode_metadata_unpack",
        cuda_files=["decode_metadata/unpack.cuh"],
        cuda_wrappers=[
            ("register_mapped", "DecodeMetadataUnpack::register_mapped"),
            ("unregister_mapped", "DecodeMetadataUnpack::unregister_mapped"),
            ("launch", "DecodeMetadataUnpack::launch"),
            ("launch_graph", "DecodeMetadataUnpack::launch_graph"),
        ],
        extra_cuda_cflags=["--relocatable-device-code=false"],
    )


def register_mapped(payload: torch.Tensor) -> int:
    return int(_jit_decode_metadata_module().register_mapped(payload))


def unregister_mapped(payload: torch.Tensor) -> None:
    _jit_decode_metadata_module().unregister_mapped(payload)


def launch(
    mapped_device_pointer: int,
    dummy_state_slot: int,
    outputs: tuple[torch.Tensor, ...],
) -> None:
    _jit_decode_metadata_module().launch(
        mapped_device_pointer,
        dummy_state_slot,
        *outputs,
    )


def launch_graph(
    mapped_device_pointer: int,
    dummy_state_slot: int,
    outputs: tuple[torch.Tensor, ...],
    plan_buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    _jit_decode_metadata_module().launch_graph(
        mapped_device_pointer,
        dummy_state_slot,
        *outputs,
        *plan_buffers,
    )


__all__ = ["launch", "launch_graph", "register_mapped", "unregister_mapped"]
