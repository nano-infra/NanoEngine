import torch


class QuantizationConfig:
    def __init__(self, **kwargs):
        self.quant_method = kwargs.get("quant_method", None)
        self.quant_algo = str(kwargs.get("quant_algo", "")).upper()
        self.fmt = kwargs.get("fmt", None)
        self.compression_format = kwargs.get("format", None)
        self.is_mxfp4 = self.compression_format in {
            "mxfp4-pack-quantized",
            "mxfp4-quantized",
        } or self.quant_method in {"mxfp4", "nvfp4"}
        # ModelOpt NVFP4 is group-16, unlike K3 MXFP4/group-32.
        self.is_modelopt_nvfp4 = (
            str(self.quant_method).lower() == "modelopt" and self.quant_algo == "NVFP4"
        )
        if self.is_mxfp4:
            # Compressed-tensors describes the checkpoint container, not the
            # dense GEMM datatype. K3 ignores attention/shared/dense linears;
            # those remain BF16 while the experts backend consumes MXFP4.
            if self.quant_method == "compressed-tensors":
                self.quant_method = None
            self.weight_group_size = self._get_mxfp4_group_size(kwargs)
            if self.weight_group_size != 32:
                raise ValueError(
                    "MXFP4 requires a weight group size of 32; "
                    f"checkpoint declares {self.weight_group_size}"
                )

        # Kimi-K3 MXFP4 stores two E2M1 values per byte and one E8M0
        # scale per group. Native backends must opt into it explicitly.

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
        if not self.quant_method or self.is_modelopt_nvfp4:
            return torch.get_default_dtype()
        elif self.quant_method == "fp8":
            # Support both explicit fmt="e4m3" and implicit fp8 (default to e4m3fn)
            if self.fmt is None or self.fmt == "e4m3":
                return torch.float8_e4m3fn
        raise AttributeError(f"Unsupported dtype: {self.quant_method=}, {self.fmt=}")

    @staticmethod
    def _get_mxfp4_group_size(config: dict) -> int:
        groups = config.get("config_groups") or {}
        for group in groups.values():
            weights = group.get("weights") or {}
            if "group_size" in weights:
                return int(weights["group_size"])
        return 32

    @property
    def linear_quant_method(self):
        """Quantization method for ordinary linear layers, excluding MoE."""
        return None if self.is_modelopt_nvfp4 else self.quant_method
