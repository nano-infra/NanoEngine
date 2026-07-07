from __future__ import annotations

import functools
import os
import pathlib
from functools import lru_cache
from typing import Any, Callable, List, Tuple, TYPE_CHECKING, TypeAlias, TypeVar, Union

import torch

if TYPE_CHECKING:
    from tvm_ffi import Module


def _make_wrapper(tup: Tuple[str, str]) -> str:
    export_name, kernel_name = tup
    return f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({export_name}, ({kernel_name}));"


@lru_cache()
def _resolve_kernel_path() -> pathlib.Path:
    return pathlib.Path(__file__).parent.resolve()


KERNEL_PATH = _resolve_kernel_path()
SGL_INCLUDE = (KERNEL_PATH.parent / "sgl" / "include").resolve()
DEFAULT_CFLAGS = ["-std=c++20", "-O3"]
DEFAULT_CUDA_CFLAGS = [
    "-std=c++20",
    "-O3",
    "--expt-relaxed-constexpr",
    "-rdc=true",
]
DEFAULT_INCLUDE = [str(SGL_INCLUDE)]
DEFAULT_LDFLAGS: list[str] = []
CPP_TEMPLATE_TYPE: TypeAlias = Union[int, float, str, bool, torch.dtype]


class CPPArgList(list):
    def __str__(self) -> str:
        return ", ".join(self)


CPP_DTYPE_MAP = {
    torch.float: "fp32_t",
    torch.float16: "fp16_t",
    torch.bfloat16: "bf16_t",
    torch.int32: "int32_t",
    torch.int64: "int64_t",
}


def make_cpp_args(*args: CPP_TEMPLATE_TYPE) -> CPPArgList:
    def _convert(arg: CPP_TEMPLATE_TYPE) -> str:
        if isinstance(arg, bool):
            return "true" if arg else "false"
        if isinstance(arg, (int, str, float)):
            return str(arg)
        if isinstance(arg, torch.dtype):
            return CPP_DTYPE_MAP[arg]
        raise TypeError(f"Unsupported argument type for cpp template: {type(arg)}")

    return CPPArgList(_convert(arg) for arg in args)


def _env_paths(name: str) -> list[str]:
    value = os.environ.get(name, "")
    return [part for part in value.split(os.pathsep) if part]


def load_jit(
    *args: str,
    cuda_files: List[str] | None = None,
    cuda_wrappers: List[Tuple[str, str]] | None = None,
    extra_cflags: List[str] | None = None,
    extra_cuda_cflags: List[str] | None = None,
    extra_ldflags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    build_directory: str | None = None,
) -> "Module":
    from tvm_ffi.cpp import load_inline

    cuda_files = cuda_files or []
    cuda_wrappers = cuda_wrappers or []
    extra_cflags = extra_cflags or []
    extra_cuda_cflags = extra_cuda_cflags or []
    extra_ldflags = extra_ldflags or []
    extra_include_paths = extra_include_paths or []

    cuda_paths = [(KERNEL_PATH / f).resolve() for f in cuda_files]
    cuda_sources = [f'#include "{path}"' for path in cuda_paths]
    cuda_sources += [_make_wrapper(tup) for tup in cuda_wrappers]

    return load_inline(
        "dlengine_nano_jit_" + "_".join(str(arg) for arg in args),
        cpp_sources=[],
        cuda_sources=cuda_sources,
        extra_cflags=DEFAULT_CFLAGS + extra_cflags,
        extra_cuda_cflags=DEFAULT_CUDA_CFLAGS + extra_cuda_cflags,
        extra_ldflags=DEFAULT_LDFLAGS + _env_paths("DLENGINE_NANO_JIT_LDFLAGS") + extra_ldflags,
        extra_include_paths=DEFAULT_INCLUDE
        + _env_paths("DLENGINE_NANO_JIT_INCLUDE_PATHS")
        + extra_include_paths,
        build_directory=build_directory,
    )


F = TypeVar("F", bound=Callable[..., Any])


def cache_once(fn: F) -> F:
    result_map: dict = {}

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items(), key=lambda x: x[0])))
        if key not in result_map:
            result_map[key] = fn(*args, **kwargs)
        return result_map[key]

    return wrapper  # type: ignore[return-value]
