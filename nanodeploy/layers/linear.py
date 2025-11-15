from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from nanodeploy.kernels.block_gemm_fp8 import deep_gemm_fp8, quant_fp8_tma
from nanodeploy.models.quant_config import QuantizationConfig
from nanodeploy.worker.context import get_context
from nanodeploy.worker.distributed import get_dist_context
from torch import nn


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
        meta: bool = False,
        weight_tensor: torch.Tensor | None = None,
        bias_tensor: torch.Tensor | None = None,
        scale_tensor: torch.Tensor | None = None,
        quantization_config: QuantizationConfig = None,
    ):
        super().__init__()

        self.quantization_config = quantization_config or QuantizationConfig

        self.tp_dim = tp_dim
        self.tp_rank = get_dist_context().attn_tp_rank
        self.tp_size = get_dist_context().attn_tp_world_size

        device = torch.get_default_device() if not meta else torch.device("meta")

        weight_dtype = self.quantization_config.dtype

        self.weight = nn.Parameter(
            weight_tensor
            if weight_tensor is not None
            else torch.empty(output_size, input_size, dtype=weight_dtype, device=device)
        )

        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(
                bias_tensor
                if bias_tensor is not None
                else torch.empty(output_size, device=device)
            )
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

        if scale_tensor is not None:
            self.weight_scale_inv = nn.Parameter(scale_tensor)
        elif quantization_config.quant_method == "fp8":
            self.weight_scale_inv = nn.Parameter(
                torch.empty(
                    output_size // quantization_config.block_size[0],
                    input_size // quantization_config.block_size[1],
                    dtype=torch.float32,
                    device=device,
                )
            )
        else:
            self.weight_scale_inv = nn.Parameter(
                torch.empty(output_size, input_size, dtype=torch.float32, device="meta")
            )
        self.weight_scale_inv.weight_loader = self.weight_loader

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantization_config.quant_method:
            return F.linear(x, self.weight, self.bias)
        elif self.quantization_config.quant_method == "fp8":
            input_quant, input_scale = quant_fp8_tma(
                x, self.quantization_config.block_size[0], dtype=self.weight.dtype
            )

            out = deep_gemm_fp8(
                input_quant,
                input_scale,
                self.weight,
                self.weight_scale_inv,
                out_dtype=x.dtype,
            )
            out = out[: x.size(0)]
            if self.bias is not None:
                out += self.bias

            return out
        else:
            raise AttributeError(f"Unsupported Quant Method")


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        meta: bool = False,
        weight_tensor: torch.Tensor | None = None,
        bias_tensor: torch.Tensor | None = None,
        scale_tensor: torch.Tensor | None = None,
        quantization_config: QuantizationConfig = None,
    ):
        super().__init__(
            input_size,
            output_size,
            bias,
            meta=meta,
            weight_tensor=weight_tensor,
            bias_tensor=bias_tensor,
            scale_tensor=scale_tensor,
            quantization_config=quantization_config,
        )

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, weight_name: str = None
    ):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantization_config.quant_method:
            return F.linear(x, self.weight, self.bias)
        elif self.quantization_config.quant_method == "fp8":
            input_quant, input_scale = quant_fp8_tma(
                x, self.quantization_config.block_size[0], dtype=self.weight.dtype
            )

            out = deep_gemm_fp8(
                input_quant,
                input_scale,
                self.weight,
                self.weight_scale_inv,
                out_dtype=x.dtype,
            )
            out = out[: x.size(0)]
            if self.bias is not None:
                out += self.bias

            return out
        else:
            raise AttributeError(f"Unsupported Quant Method")


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        meta: bool = False,
        weight_tensor: torch.Tensor | None = None,
        bias_tensor: torch.Tensor | None = None,
        scale_tensor: torch.Tensor | None = None,
        quantization_config: QuantizationConfig = None,
    ):
        tp_size = get_dist_context().attn_tp_world_size
        super().__init__(
            input_size,
            divide(output_size, tp_size),
            bias,
            0,
            meta=meta,
            weight_tensor=weight_tensor,
            bias_tensor=bias_tensor,
            scale_tensor=scale_tensor,
            quantization_config=quantization_config,
        )

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, weight_name: str = None
    ):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(
        self, x: torch.Tensor, enable_zero_copy: Optional[bool] = False
    ) -> torch.Tensor:
        if not self.quantization_config.quant_method:
            return F.linear(x, self.weight, self.bias)
        elif self.quantization_config.quant_method == "fp8":
            input_quant, input_scale = quant_fp8_tma(
                x,
                self.quantization_config.block_size[0],
                dtype=self.weight.dtype,
                enable_zero_copy=enable_zero_copy,
            )

            out = deep_gemm_fp8(
                input_quant,
                input_scale,
                self.weight,
                self.weight_scale_inv,
                out_dtype=x.dtype,
                enable_zero_copy=enable_zero_copy,
            )
            out = out[: x.size(0)]
            if self.bias is not None:
                out += self.bias
            return out
        else:
            raise AttributeError(f"Unsupported Quant Method")


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        meta: bool = False,
        weight_tensor: torch.Tensor | None = None,
        bias_tensor: torch.Tensor | None = None,
        scale_tensor: torch.Tensor | None = None,
        quantization_config: QuantizationConfig = None,
    ):
        super().__init__(
            input_size,
            sum(output_sizes),
            bias,
            meta=meta,
            weight_tensor=weight_tensor,
            bias_tensor=bias_tensor,
            scale_tensor=scale_tensor,
            quantization_config=quantization_config,
        )
        self.output_sizes = output_sizes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x, False)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int,
        weight_name,
    ):
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        if "inv" in weight_name:
            output_sizes = [
                out // self.quantization_config.block_size[0]
                for out in self.output_sizes
            ]
            loaded_weight = loaded_weight.to(torch.float32)
        else:
            output_sizes = self.output_sizes
        param_data = param.data
        shard_offset = sum(output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        meta: bool = False,
        weight_tensor: torch.Tensor | None = None,
        bias_tensor: torch.Tensor | None = None,
        scale_tensor: torch.Tensor | None = None,
        quantization_config: QuantizationConfig = QuantizationConfig,
    ):
        tp_size = get_dist_context().attn_tp_world_size
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(
            hidden_size,
            output_size,
            bias,
            meta=meta,
            weight_tensor=weight_tensor,
            bias_tensor=bias_tensor,
            scale_tensor=scale_tensor,
            quantization_config=quantization_config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_prefill = get_context().is_prefill
        sp_size = get_dist_context().attn_sp_world_size
        if not is_prefill and get_context().enable_zero_copy and sp_size > 1:
            enable_zero_copy = True
        else:
            enable_zero_copy = False
        return super().forward(x, enable_zero_copy)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str,
        weight_name: str = None,
    ):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = (
                self.num_heads * self.head_size + self.num_kv_heads * self.head_size
            )
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        if "inv" in weight_name:
            shard_offset = shard_offset // self.quantization_config.block_size[0]
            shard_size = shard_size // self.quantization_config.block_size[0]
            loaded_weight = loaded_weight.to(torch.float32)
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)

        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        meta: bool = False,
        weight_tensor: torch.Tensor | None = None,
        bias_tensor: torch.Tensor | None = None,
        scale_tensor: torch.Tensor | None = None,
        quantization_config: QuantizationConfig = None,
    ):
        tp_size = get_dist_context().attn_tp_world_size
        super().__init__(
            divide(input_size, tp_size),
            output_size,
            bias,
            1,
            meta=meta,
            weight_tensor=weight_tensor,
            bias_tensor=bias_tensor,
            scale_tensor=scale_tensor,
            quantization_config=quantization_config,
        )

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, weight_name: str = None
    ):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantization_config.quant_method:
            y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
            if self.tp_size > 1:
                dist.all_reduce(y, group=get_dist_context().attn_tp_group)
            return y
        elif self.quantization_config.quant_method == "fp8":
            x_shape = x.shape
            x = x.flatten(0, -2)

            input_quant, input_scale = quant_fp8_tma(
                x, self.quantization_config.block_size[0], dtype=self.weight.dtype
            )

            out = deep_gemm_fp8(
                input_quant,
                input_scale,
                self.weight,
                self.weight_scale_inv,
                out_dtype=x.dtype,
            )
            out = out[: x.size(0)]
            if self.bias is not None:
                out += self.bias

            out = out.unflatten(0, x_shape[:-1])
            if self.tp_size > 1:
                dist.all_reduce(out, group=get_dist_context().attn_tp_group)
            return out
