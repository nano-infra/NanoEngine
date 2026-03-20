"""Ascend NPU BF16 linear layer implementations.

These layers use standard ``torch.nn.functional.linear`` with HCCL all-reduce
for tensor parallelism.  Weights are stored in BF16 — identical logic to the
gpu_generic backend, just under the Ascend namespace.
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

ACL_FORMAT_FRACTAL_NZ = 29


def _divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


def _maybe_cast_nz(weight: torch.Tensor) -> torch.Tensor:
    """Convert BF16/FP16 weight to FRACTAL_NZ format for faster matmul on Ascend NPU."""
    if weight.dtype not in (torch.bfloat16, torch.float16):
        return weight
    try:
        import torch_npu
        return torch_npu.npu_format_cast(weight.contiguous(), ACL_FORMAT_FRACTAL_NZ)
    except (ImportError, AttributeError, RuntimeError):
        return weight


# ---------------------------------------------------------------------------
# Internal mixin: BF16 weight allocation
# ---------------------------------------------------------------------------


class _AscendLinearMixin:
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
# AscendReplicatedLinear
# ---------------------------------------------------------------------------


class AscendReplicatedLinear(_AscendLinearMixin, ReplicatedLinearBase):

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
# AscendColumnParallelLinear
# ---------------------------------------------------------------------------


class AscendColumnParallelLinear(_AscendLinearMixin, ColumnParallelLinearBase):

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
# AscendMergedColumnParallelLinear
# ---------------------------------------------------------------------------


class AscendMergedColumnParallelLinear(
    _AscendLinearMixin, MergedColumnParallelLinearBase
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
# AscendQKVParallelLinear
# ---------------------------------------------------------------------------


class AscendQKVParallelLinear(_AscendLinearMixin, QKVParallelLinearBase):

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
# AscendRowParallelLinear
# ---------------------------------------------------------------------------


class AscendRowParallelLinear(_AscendLinearMixin, RowParallelLinearBase):

    _hcomm_cache: dict[int, str] = {}  # group-id → hccl comm name

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

        # Try to get HCCL comm name for fused matmul+allreduce
        self._hcomm_info: str | None = None
        if self.tp_size > 1:
            self._hcomm_info = self._get_hcomm_info(self._tp_group)

    @classmethod
    def _get_hcomm_info(cls, group: dist.ProcessGroup) -> str | None:
        """Get HCCL comm name for npu_mm_all_reduce_base."""
        gid = id(group)
        if gid in cls._hcomm_cache:
            return cls._hcomm_cache[gid]
        try:
            rank = dist.get_rank(group)
            global_rank = dist.get_global_rank(group, rank)
            backend = group._get_backend(torch.device("npu"))
            hcomm = backend.get_hccl_comm_name(global_rank)
            cls._hcomm_cache[gid] = hcomm
            return hcomm
        except (AttributeError, RuntimeError):
            cls._hcomm_cache[gid] = None
            return None

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, weight_name: str = None
    ):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias if self.tp_rank == 0 else None
        if self._hcomm_info is not None:
            # Fused matmul + HCCL all-reduce (single kernel launch)
            import torch_npu
            return torch_npu.npu_mm_all_reduce_base(
                x, self.weight.t(), self._hcomm_info, bias=bias
            )
        y = F.linear(x, self.weight, bias)
        if self.tp_size > 1:
            dist.all_reduce(y, group=self._tp_group)
        return y
