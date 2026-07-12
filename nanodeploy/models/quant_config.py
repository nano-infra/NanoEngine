import torch


class QuantizationConfig:
    def __init__(self, **kwargs):
        self.quant_method = kwargs.get("quant_method", None)
        self.fmt = kwargs.get("fmt", None)

        # configuration for block-wise quantization
        self.block_size = kwargs.get("weight_block_size", list())

    @property
    def dtype(self):
        if not self.quant_method:
            return torch.get_default_dtype()
        elif self.quant_method == "fp8" and self.fmt == "e4m3":
            return torch.float8_e4m3fn
        raise AttributeError(f"Unsupported dtype: {self.quant_method=}, {self.fmt=}")
