"""Generic GPU (BF16) linear layer implementations.

These layers use standard ``torch.nn.functional.linear`` with all-reduce for
tensor parallelism.  No FP8 kernels are used — weights are stored in BF16.
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
from nanodeploy.context.distributed import get_dist_context


def _divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


# ---------------------------------------------------------------------------
# Internal mixin: BF16 weight allocation (no scale tensors)
# ---------------------------------------------------------------------------


class _GenericLinearMixin:
    """Allocates BF16 weights and sets weight_loader on parameters."""

    def _init_weights(
        self,
        input_size: int,
        output_size: int,
        bias: bool,
        tp_dim: Optional[int],
        meta: bool,
        weight_tensor: Optional[torch.Tensor],
        bias_tensor: Optional[torch.Tensor],
        tp_group: Optional[dist.ProcessGroup] = None,
    ):
        self.tp_dim = tp_dim
        if tp_group is None:
            tp_group = get_dist_context().attn_tp_group

        self.tp_rank = dist.get_rank(tp_group)
        self.tp_size = dist.get_world_size(tp_group)
        self._tp_group = tp_group

        device = torch.get_default_device() if not meta else torch.device("meta")

        self.weight = nn.Parameter(
            weight_tensor
            if weight_tensor is not None
            else torch.empty(output_size, input_size, device=device)
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


# ---------------------------------------------------------------------------
# GenericReplicatedLinear
# ---------------------------------------------------------------------------


class GenericReplicatedLinear(_GenericLinearMixin, ReplicatedLinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        meta: bool = False,
        weight_tensor: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
    ):
        nn.Module.__init__(self)
        self._init_weights(
            input_size, output_size, bias, None, meta, weight_tensor, bias_tensor
        )

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, weight_name: str = None
    ):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


# ---------------------------------------------------------------------------
# GenericColumnParallelLinear
# ---------------------------------------------------------------------------


class GenericColumnParallelLinear(_GenericLinearMixin, ColumnParallelLinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        meta: bool = False,
        weight_tensor: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
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
        return F.linear(x, self.weight, self.bias)


# ---------------------------------------------------------------------------
# GenericMergedColumnParallelLinear
# ---------------------------------------------------------------------------


class GenericMergedColumnParallelLinear(
    _GenericLinearMixin, MergedColumnParallelLinearBase
):

    def __init__(
        self,
        input_size: int,
        output_sizes: list,
        bias: bool = False,
        meta: bool = False,
        weight_tensor: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
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
        output_sizes = self.output_sizes
        param_data = param.data
        shard_offset = sum(output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


# ---------------------------------------------------------------------------
# GenericQKVParallelLinear
# ---------------------------------------------------------------------------


class GenericQKVParallelLinear(_GenericLinearMixin, QKVParallelLinearBase):

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
        tp_group: Optional[dist.ProcessGroup] = None,
    ):
        nn.Module.__init__(self)
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
            tp_group=tp_group,
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
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


# ---------------------------------------------------------------------------
# GenericRowParallelLinear
# ---------------------------------------------------------------------------


class GenericRowParallelLinear(_GenericLinearMixin, RowParallelLinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        meta: bool = False,
        weight_tensor: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
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
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y, group=self._tp_group)
        return y
