"""Hopper (FP8 + DeepGEMM) linear layer implementations.

All classes extend the abstract base types from ``backends.base_backend``
and support both FP8 (Hopper-native) and BF16 (fallback) computation paths.
"""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from nanodeploy.backends.base_backend import (
    ColumnParallelLinearBase,
    MergedColumnParallelLinearBase,
    QKVParallelLinearBase,
    ReplicatedLinearBase,
    RowParallelLinearBase,
)
from nanodeploy.backends.hopper.kernels.block_gemm_fp8 import (
    deep_gemm_fp8,
    quant_fp8_tma,
)
from nanodeploy.context.distributed import get_dist_context
from nanodeploy.models.quant_config import QuantizationConfig


def _divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


# ---------------------------------------------------------------------------
# Internal mixin: shared weight allocation + FP8 forward
# ---------------------------------------------------------------------------


class _HopperLinearMixin:
    """Mixin that allocates weights and implements the FP8/BF16 forward.

    Subclasses must call ``_init_weights(input_size, output_size, bias,
    tp_dim, meta, weight_tensor, bias_tensor, scale_tensor, quant_config)``
    from their own ``__init__``.
    """

    def _init_weights(
        self,
        input_size: int,
        output_size: int,
        bias: bool,
        tp_dim: Optional[int],
        meta: bool,
        weight_tensor: Optional[torch.Tensor],
        bias_tensor: Optional[torch.Tensor],
        scale_tensor: Optional[torch.Tensor],
        quantization_config: QuantizationConfig,
        tp_group: Optional[dist.ProcessGroup] = None,
    ):
        self.quantization_config = quantization_config or QuantizationConfig()
        self.tp_dim = tp_dim
        if tp_group is None:
            tp_group = get_dist_context().attn_tp_group

        self.tp_rank = dist.get_rank(tp_group)
        self.tp_size = dist.get_world_size(tp_group)
        self._tp_group = tp_group

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
        elif self.quantization_config.quant_method == "fp8":
            n_blk, k_blk = self.quantization_config.block_size
            self.weight_scale_inv = nn.Parameter(
                torch.empty(
                    (output_size + n_blk - 1) // n_blk,
                    (input_size + k_blk - 1) // k_blk,
                    dtype=torch.float32,
                    device=device,
                )
            )
        else:
            self.weight_scale_inv = nn.Parameter(
                torch.empty(output_size, input_size, dtype=torch.float32, device="meta")
            )
        self.weight_scale_inv.weight_loader = self.weight_loader

    def _fp8_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Shared FP8 GEMM path (used by column/row/replicated variants)."""
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
            out = out + self.bias
        return out


# ---------------------------------------------------------------------------
# HopperReplicatedLinear
# ---------------------------------------------------------------------------


class HopperReplicatedLinear(_HopperLinearMixin, ReplicatedLinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        meta: bool = False,
        weight_tensor: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
        scale_tensor: Optional[torch.Tensor] = None,
        quantization_config: Optional[QuantizationConfig] = None,
    ):
        nn.Module.__init__(self)
        self._init_weights(
            input_size,
            output_size,
            bias,
            None,
            meta,
            weight_tensor,
            bias_tensor,
            scale_tensor,
            quantization_config,
        )

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, weight_name: str = None
    ):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantization_config.quant_method:
            return F.linear(x, self.weight, self.bias)
        elif self.quantization_config.quant_method == "fp8":
            return self._fp8_forward(x)
        else:
            raise AttributeError(
                f"Unsupported quant method: {self.quantization_config.quant_method}"
            )


# ---------------------------------------------------------------------------
# HopperColumnParallelLinear
# ---------------------------------------------------------------------------


class HopperColumnParallelLinear(_HopperLinearMixin, ColumnParallelLinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        meta: bool = False,
        weight_tensor: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
        scale_tensor: Optional[torch.Tensor] = None,
        quantization_config: Optional[QuantizationConfig] = None,
        tp_group: Optional[dist.ProcessGroup] = None,
    ):
        nn.Module.__init__(self)
        if tp_group is None:
            tp_group = get_dist_context().attn_tp_group
        tp_size = dist.get_world_size(tp_group)
        self._init_weights(
            input_size,
            _divide(output_size, tp_size),
            bias,
            0,
            meta,
            weight_tensor,
            bias_tensor,
            scale_tensor,
            quantization_config,
            tp_group=tp_group,
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
            return F.linear(x, self.weight, self.bias)
        elif self.quantization_config.quant_method == "fp8":
            return self._fp8_forward(x)
        else:
            raise AttributeError(
                f"Unsupported quant method: {self.quantization_config.quant_method}"
            )


# ---------------------------------------------------------------------------
# HopperMergedColumnParallelLinear
# ---------------------------------------------------------------------------


class HopperMergedColumnParallelLinear(
    _HopperLinearMixin, MergedColumnParallelLinearBase
):

    def __init__(
        self,
        input_size: int,
        output_sizes: list,
        bias: bool = False,
        meta: bool = False,
        weight_tensor: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
        scale_tensor: Optional[torch.Tensor] = None,
        quantization_config: Optional[QuantizationConfig] = None,
        tp_group: Optional[dist.ProcessGroup] = None,
    ):
        nn.Module.__init__(self)
        if tp_group is None:
            tp_group = get_dist_context().attn_tp_group
        tp_size = dist.get_world_size(tp_group)
        self.output_sizes = output_sizes
        self._init_weights(
            input_size,
            _divide(sum(output_sizes), tp_size),
            bias,
            0,
            meta,
            weight_tensor,
            bias_tensor,
            scale_tensor,
            quantization_config,
            tp_group=tp_group,
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int,
        weight_name: str,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantization_config.quant_method:
            return F.linear(x, self.weight, self.bias)
        elif self.quantization_config.quant_method == "fp8":
            return self._fp8_forward(x)
        else:
            raise AttributeError(
                f"Unsupported quant method: {self.quantization_config.quant_method}"
            )


# ---------------------------------------------------------------------------
# HopperQKVParallelLinear
# ---------------------------------------------------------------------------


class HopperQKVParallelLinear(_HopperLinearMixin, QKVParallelLinearBase):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: Optional[int] = None,
        bias: bool = False,
        meta: bool = False,
        weight_tensor: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
        scale_tensor: Optional[torch.Tensor] = None,
        quantization_config: Optional[QuantizationConfig] = None,
        tp_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        if tp_group is None:
            tp_group = get_dist_context().attn_tp_group
        tp_size = dist.get_world_size(tp_group)
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = _divide(total_num_heads, tp_size)
        self.num_kv_heads = _divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size
        self._init_weights(
            hidden_size,
            _divide(output_size, tp_size),
            bias,
            0,
            meta,
            weight_tensor,
            bias_tensor,
            scale_tensor,
            quantization_config,
        )

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
        if weight_name is not None and "inv" in weight_name:
            shard_offset = shard_offset // self.quantization_config.block_size[0]
            shard_size = shard_size // self.quantization_config.block_size[0]
            loaded_weight = loaded_weight.to(torch.float32)
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantization_config.quant_method:
            return F.linear(x, self.weight, self.bias)
        elif self.quantization_config.quant_method == "fp8":
            return self._fp8_forward(x)
        else:
            raise AttributeError(
                f"Unsupported quant method: {self.quantization_config.quant_method}"
            )


# ---------------------------------------------------------------------------
# HopperRowParallelLinear
# ---------------------------------------------------------------------------


class HopperRowParallelLinear(_HopperLinearMixin, RowParallelLinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        meta: bool = False,
        weight_tensor: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
        scale_tensor: Optional[torch.Tensor] = None,
        quantization_config: Optional[QuantizationConfig] = None,
        tp_group: Optional[dist.ProcessGroup] = None,
    ):
        nn.Module.__init__(self)
        if tp_group is None:
            tp_group = get_dist_context().attn_tp_group
        tp_size = dist.get_world_size(tp_group)
        self._init_weights(
            _divide(input_size, tp_size),
            output_size,
            bias,
            1,
            meta,
            weight_tensor,
            bias_tensor,
            scale_tensor,
            quantization_config,
            tp_group=tp_group,
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
                dist.all_reduce(y, group=self._tp_group)
            return y
        elif self.quantization_config.quant_method == "fp8":
            x_shape = x.shape
            x = x.flatten(0, -2)
            out = self._fp8_forward(x)
            out = out.unflatten(0, x_shape[:-1])
            if self.tp_size > 1:
                dist.all_reduce(out, group=self._tp_group)
            return out
        else:
            raise AttributeError(
                f"Unsupported quant method: {self.quantization_config.quant_method}"
            )
