from __future__ import annotations

from typing import TYPE_CHECKING

from .utils import cache_once, load_jit

if TYPE_CHECKING:
    import torch
    from tvm_ffi.module import Module


@cache_once
def _jit_decode_metadata_benchmark_module() -> Module:
    return load_jit(
        "decode_metadata_benchmark",
        cuda_files=["decode_metadata/benchmark.cuh"],
        cuda_wrappers=[
            ("register_mapped", "DecodeMetadataBenchmark::register_mapped"),
            ("unregister_mapped", "DecodeMetadataBenchmark::unregister_mapped"),
            ("launch_mapped", "DecodeMetadataBenchmark::launch_mapped"),
            ("launch_device", "DecodeMetadataBenchmark::launch_device"),
        ],
        # This module has no cross-translation-unit device symbols. Disabling
        # RDC avoids pulling cudadevrt into the runtime-loaded shared object.
        extra_cuda_cflags=["--relocatable-device-code=false"],
    )


def register_mapped(payload: torch.Tensor) -> None:
    _jit_decode_metadata_benchmark_module().register_mapped(payload)


def unregister_mapped(payload: torch.Tensor) -> None:
    _jit_decode_metadata_benchmark_module().unregister_mapped(payload)


def launch_mapped(payload: torch.Tensor, outputs: tuple[torch.Tensor, ...]) -> None:
    _jit_decode_metadata_benchmark_module().launch_mapped(payload, *outputs)


def launch_device(payload: torch.Tensor, outputs: tuple[torch.Tensor, ...]) -> None:
    _jit_decode_metadata_benchmark_module().launch_device(payload, *outputs)


__all__ = [
    "launch_device",
    "launch_mapped",
    "register_mapped",
    "unregister_mapped",
]
