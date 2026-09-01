import importlib
import sys

import torch


def test_import_does_not_initialize_cuda(monkeypatch):
    def fail_cuda_probe(*_args, **_kwargs):
        raise AssertionError("CUDA must not be queried while importing kernel utilities")

    monkeypatch.setattr(torch.cuda, "current_device", fail_cuda_probe)
    monkeypatch.setattr(torch.cuda, "get_device_capability", fail_cuda_probe)
    monkeypatch.setattr(torch.cuda, "get_device_properties", fail_cuda_probe)

    module_name = "dlengine.runtime.kernel.triton.generic.utils"
    sys.modules.pop(module_name, None)
    importlib.import_module(module_name)
