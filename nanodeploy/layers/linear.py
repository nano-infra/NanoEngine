from typing import List

import torch
import torch.distributed as dist
import torch.nn.functional as F

from nanodeploy.kernels.block_gemm_fp8 import deep_gemm_fp8, quant_fp8_tma
from nanodeploy.models.quant_config import QuantizationConfig
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
        tp_size: int | None = None,
        tp_rank: int | None = None,
        tp_group: dist.ProcessGroup | None = None,
    ):
        super().__init__()

        self.quantization_config = quantization_config or QuantizationConfig

        self.tp_dim = tp_dim
        self.tp_rank = tp_rank if tp_rank is not None else get_dist_context().attn_tp_rank
        self.tp_size = tp_size if tp_size is not None else get_dist_context().attn_tp_world_size
        self.tp_group = tp_group if tp_group is not None else get_dist_context().attn_tp_group

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
        
        n_blk_size, k_blk_size = quantization_config.block_size
        if scale_tensor is not None:
            self.weight_scale_inv = nn.Parameter(scale_tensor)
        elif quantization_config.quant_method == "fp8":
            self.weight_scale_inv = nn.Parameter(
                torch.empty(
                    (output_size + n_blk_size - 1) // n_blk_size,
                    (input_size + k_blk_size - 1) // k_blk_size,
                    dtype=torch.float32,
                    device=device,
                )
            )
        else:
            self.weight_scale_inv = nn.Parameter(
                torch.empty(output_size, input_size, dtype=torch.float32, device="meta")
            )
        self.weight_scale_inv.weight_loader = self.weight_loader

    def load_weight_tp(self, param, loaded_weight, ckpt_shard_id=None, ckpt_num_shards=None):
        param_data = param.data
        tp_dim = self.tp_dim
        shard_size = param_data.size(tp_dim)
        loaded_dim_size = loaded_weight.size(tp_dim)
        
        # Check if we are loading a TP-shard
        if loaded_dim_size == shard_size:
             # Case: Loaded weight matches local shard size.
             # This happens if:
             # 1. Target TP=1, Ckpt TP=1 (Full matching)
             # 2. Target TP=N, Ckpt TP=N (Sharded matching)
             
             if ckpt_shard_id is not None and self.tp_size > 1:
                  # If we have shard info and we are sharded.
                  # Verify rank matches.
                  if ckpt_shard_id == self.tp_rank:
                       param_data.copy_(loaded_weight)
                  # else: skip (not our shard)
             else:
                  # Fallback: sizes match, just copy.
                  param_data.copy_(loaded_weight)

        elif loaded_dim_size > shard_size:
             # Case: Loaded weight is larger (Full tensor, or larger shard).
             # Slice it.
             start_idx = self.tp_rank * shard_size
             if start_idx + shard_size <= loaded_dim_size:
                 loaded_weight = loaded_weight.narrow(tp_dim, start_idx, shard_size)
                 param_data.copy_(loaded_weight)
        
        elif loaded_dim_size < shard_size:
             # Case: Loaded weight is smaller.
             # Target TP < Ckpt TP (Merging shards).
             if ckpt_shard_id is not None:
                  offset = ckpt_shard_id * loaded_dim_size
                  if offset + loaded_dim_size <= shard_size:
                        param_data.narrow(tp_dim, offset, loaded_dim_size).copy_(loaded_weight)

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
        tp_size: int | None = None,
        tp_rank: int | None = None,
        tp_group: dist.ProcessGroup | None = None,
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
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=tp_group,
        )

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, weight_name: str = None, **kwargs
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
        tp_size: int | None = None,
        tp_rank: int | None = None,
        tp_group: dist.ProcessGroup | None = None,
    ):
        if tp_size is None:
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
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=tp_group,
        )

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, weight_name: str = None, ckpt_shard_id=None, ckpt_num_shards=None, **kwargs
    ):
        self.load_weight_tp(param, loaded_weight, ckpt_shard_id, ckpt_num_shards)

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
        tp_size: int | None = None,
        tp_rank: int | None = None,
        tp_group: dist.ProcessGroup | None = None,
    ):
        if tp_size is None:
            tp_size = get_dist_context().attn_tp_world_size

        super(ColumnParallelLinear, self).__init__(
            input_size,
            sum(output_sizes),
            bias,
            0, # tp_dim
            meta=meta,
            weight_tensor=weight_tensor,
            bias_tensor=bias_tensor,
            scale_tensor=scale_tensor,
            quantization_config=quantization_config,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=tp_group,
        )
        self.output_sizes = output_sizes

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int,
        weight_name,
        ckpt_shard_id=None,
        ckpt_num_shards=None,
        **kwargs
    ):
        if "inv" in weight_name:
            output_sizes = [
                out // self.quantization_config.block_size[0]
                for out in self.output_sizes
            ]
            loaded_weight = loaded_weight.to(torch.float32)
        else:
            output_sizes = self.output_sizes
        
        # Determine the slice of the parameter corresponding to this component (e.g. gate or up)
        # Note: output_sizes are full sizes.
        # But the param might be sharded if tp_size > 1.
        # shard_offset/size here refers to the split between gate/up within the param.
        
        current_offset = sum(output_sizes[:loaded_shard_id]) // self.tp_size
        current_size = output_sizes[loaded_shard_id] // self.tp_size
        
        param_subset = param.data.narrow(self.tp_dim, current_offset, current_size)
        
        # Now use standard TP loader to load (potentially sharded) weight into this subset
        # We treat param_subset as a parameter that needs to be filled.
        # But load_weight_tp expects a Parameter (to access .data), or just Tensor?
        # My load_weight_tp takes 'param' and uses param.data.
        # I can pass a dummy object or modify load_weight_tp to take tensor.
        # Or just inline the logic / use a helper that takes tensor.
        
        # Refactoring load_weight_tp to take tensor is cleaner but requires modifying LinearBase.
        # For now, I'll wrap param_subset in a dummy object.
        class DummyParam:
            def __init__(self, data):
                self.data = data
        
        self.load_weight_tp(DummyParam(param_subset), loaded_weight, ckpt_shard_id, ckpt_num_shards)


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
        tp_size: int | None = None,
        tp_rank: int | None = None,
        tp_group: dist.ProcessGroup | None = None,
    ):
        if tp_size is None:
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
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=tp_group,
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str,
        weight_name: str = None,
        **kwargs
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
        tp_size: int | None = None,
        tp_rank: int | None = None,
        tp_group: dist.ProcessGroup | None = None,
    ):
        if tp_size is None:
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
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=tp_group,
        )

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, weight_name: str = None, ckpt_shard_id=None, ckpt_num_shards=None, **kwargs
    ):
        self.load_weight_tp(param, loaded_weight, ckpt_shard_id, ckpt_num_shards)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantization_config.quant_method:
            y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
            if self.tp_size > 1:
                dist.all_reduce(y, group=self.tp_group)
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
                dist.all_reduce(out, group=self.tp_group)
            return out
