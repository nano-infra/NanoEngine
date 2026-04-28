import torch


class QuantizationConfig:
    def __init__(self, **kwargs):
        self.quant_method = kwargs.get("quant_method", None)
        self.fmt = kwargs.get("fmt", None)

        # configuration for block-wise quantization
        self.block_size = kwargs.get("weight_block_size", list())

        # Scale format: "ue8m0" rounds activation/weight scales to the
        # nearest power of two (matching DSV4's QAT regime); ``None``
        # leaves them as plain fp32. DSV4 ships ``scale_fmt: "ue8m0"``
        # in the HF config — when this is set, online activation quant
        # MUST round scales to power-of-two or the FP8 GEMM will see
        # subtly different inputs vs the trained-on regime, which
        # accumulates drift across layers and flips greedy top-1.
        self.scale_fmt = kwargs.get("scale_fmt", None)

    @property
    def dtype(self):
        if not self.quant_method:
            return torch.get_default_dtype()
        elif self.quant_method == "fp8":
            # Support both explicit fmt="e4m3" and implicit fp8 (default to e4m3fn)
            if self.fmt is None or self.fmt == "e4m3":
                return torch.float8_e4m3fn
        raise AttributeError(f"Unsupported dtype: {self.quant_method=}, {self.fmt=}")
