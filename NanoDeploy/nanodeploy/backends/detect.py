"""Hardware backend auto-detection.

Priority:
  1. Ascend NPU  — torch_npu importable + NPU device available
  2. Hopper GPU  — CUDA compute capability >= 9.0
  3. GPU Generic — CUDA available but not Hopper
"""


def detect_backend() -> str:
    """Probe installed hardware and return explicit backend string.

    Never returns None; raises RuntimeError if nothing is available.
    """
    try:
        import torch_npu

        if torch_npu.npu.is_available():
            return "ascend"
    except ImportError:
        pass

    import torch

    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        return "hopper" if cap[0] >= 9 else "gpu_generic"

    raise RuntimeError(
        "No supported accelerator found. "
        "Install torch_npu for Ascend or a CUDA GPU for Hopper/GPU-generic."
    )
