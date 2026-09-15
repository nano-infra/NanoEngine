import os

import pytest
import torch

from dlengine.runtime.kernel.jit.sgl import attn_res


@pytest.mark.parametrize(
    "capability,target,arch_define",
    [
        ((10, 0), "compute_100a,code=sm_100a", "-DSGL_CUDA_ARCH=1000"),
        ((10, 3), "compute_103a,code=sm_103a", "-DSGL_CUDA_ARCH=1030"),
    ],
)
@pytest.mark.parametrize("previous", [None, "9.0a 10.3a"])
@pytest.mark.parametrize("build_fails", [False, True])
def test_attn_res_jit_targets_current_device(
    monkeypatch, capability, target, arch_define, previous, build_fails
):
    cpp = pytest.importorskip("tvm_ffi.cpp")
    key = "TVM_FFI_CUDA_ARCH_LIST"
    if previous is None:
        monkeypatch.delenv(key, raising=False)
    else:
        monkeypatch.setenv(key, previous)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)

    def get_capability(device):
        assert device == 2
        return capability

    monkeypatch.setattr(torch.cuda, "get_device_capability", get_capability)
    module = object()

    def load_inline(name, **kwargs):
        assert cpp.extension._get_cuda_target() == f"-gencode=arch={target}"
        assert arch_define in kwargs["extra_cuda_cflags"]
        if build_fails:
            raise RuntimeError("test build failure")
        return module

    monkeypatch.setattr(cpp, "load_inline", load_inline)
    # Bypass the module cache so each simulated GPU exercises compilation.
    if build_fails:
        with pytest.raises(RuntimeError, match="test build failure"):
            attn_res._jit_module.__wrapped__(4, 1, 200)
    else:
        assert attn_res._jit_module.__wrapped__(4, 1, 200) is module
    assert os.environ.get(key) == previous
