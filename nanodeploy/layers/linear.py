from typing import List

import torch
import torch.distributed as dist
import torch.nn.functional as F
from nanodeploy.kernels.block_gemm_fp8 import deep_gemm_fp8, quant_fp8_tma

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
        weight_tensor: torch.Tensor | None = None,
        bias_tensor: torch.Tensor | None = None,
        scale_tensor: torch.Tensor | None = None,
        block_size: int | None = 128,
    ):
        super().__init__()
        self.block_size = block_size
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank(group=get_dist_context().attn_tp_group)
        self.tp_size = dist.get_world_size(group=get_dist_context().attn_tp_group)

        self.weight = nn.Parameter(
            weight_tensor
            if weight_tensor is not None
            else torch.empty(output_size, input_size)
        )

        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(
                bias_tensor if bias_tensor is not None else torch.empty(output_size)
            )
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

        if scale_tensor is not None:
            self.weight_scale_inv = nn.Parameter(scale_tensor)
            self.weight_scale_inv.weight_loader = self.weight_loader

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        weight_tensor: torch.Tensor | None = None,
        bias_tensor: torch.Tensor | None = None,
        scale_tensor: torch.Tensor | None = None,
        block_size: int | None = 128,
    ):
        super().__init__(
            input_size,
            output_size,
            bias,
            weight_tensor=weight_tensor,
            bias_tensor=bias_tensor,
            scale_tensor=scale_tensor,
            block_size=block_size,
        )

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, weight_name: str = None
    ):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        weight_tensor: torch.Tensor | None = None,
        bias_tensor: torch.Tensor | None = None,
        scale_tensor: torch.Tensor | None = None,
        block_size: int | None = 128,
    ):
        tp_size = dist.get_world_size(group=get_dist_context().attn_tp_group)
        super().__init__(
            input_size,
            divide(output_size, tp_size),
            bias,
            0,
            weight_tensor=weight_tensor,
            bias_tensor=bias_tensor,
            scale_tensor=scale_tensor,
            block_size=128,
        )
        self.block_size = block_size

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, weight_name: str = None
    ):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        weight_tensor: torch.Tensor | None = None,
        bias_tensor: torch.Tensor | None = None,
        scale_tensor: torch.Tensor | None = None,
        block_size: int | None = 128,
    ):
        self.output_sizes = output_sizes
        self.block_size = block_size
        super().__init__(
            input_size,
            sum(output_sizes),
            bias,
            weight_tensor=weight_tensor,
            bias_tensor=bias_tensor,
            scale_tensor=scale_tensor,
            block_size=block_size,
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int,
        weight_name,
    ):
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        if "inv" in weight_name:
            output_sizes = [out // self.block_size for out in self.output_sizes]
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
        weight_tensor: torch.Tensor | None = None,
        bias_tensor: torch.Tensor | None = None,
        scale_tensor: torch.Tensor | None = None,
        block_size: int | None = 128,
    ):
        self.block_size = block_size
        tp_size = dist.get_world_size(group=get_dist_context().attn_tp_group)
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(
            hidden_size,
            output_size,
            bias,
            weight_tensor=weight_tensor,
            bias_tensor=bias_tensor,
            scale_tensor=scale_tensor,
        )
        self.block_size = 128

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
            shard_offset = shard_offset // self.block_size
            shard_size = shard_size // self.block_size
            loaded_weight = loaded_weight.to(torch.float32)
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)

        param_data.copy_(loaded_weight)

    def forward(self, x):
        """forward."""
        x_shape = x.shape
        x = x.flatten(0, -2)

        input_quant, input_scale = quant_fp8_tma(
            x, self.block_size, dtype=self.weight.dtype
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
        return out


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        weight_tensor: torch.Tensor | None = None,
        bias_tensor: torch.Tensor | None = None,
        scale_tensor: torch.Tensor | None = None,
        block_size: int | None = None,
    ):
        tp_size = dist.get_world_size(group=get_dist_context().attn_tp_group)
        super().__init__(
            divide(input_size, tp_size),
            output_size,
            bias,
            1,
            weight_tensor=weight_tensor,
            bias_tensor=bias_tensor,
            scale_tensor=scale_tensor,
            block_size=block_size,
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
        if hasattr(self, "weight_scale_inv"):
            x_shape = x.shape
            x = x.flatten(0, -2)

            input_quant, input_scale = quant_fp8_tma(
                x, self.block_size, dtype=self.weight.dtype
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
            return out

        else:
            y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
            if self.tp_size > 1:
                dist.all_reduce(y, group=get_dist_context().attn_tp_group)
            return y
