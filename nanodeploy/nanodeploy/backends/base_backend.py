"""Abstract base classes for the NanoDeploy hardware abstraction layer.

These classes serve as type hints in model topology files and define the
interface that all backend implementations must satisfy.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn


class LinearBase(nn.Module, ABC):
    """Abstract base for all linear layer variants."""

    @abstractmethod
    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, **kwargs
    ) -> None: ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


class RowParallelLinearBase(LinearBase):
    """Row-parallel linear: splits input along the input dimension."""


class ColumnParallelLinearBase(LinearBase):
    """Column-parallel linear: splits output along the output dimension."""


class MergedColumnParallelLinearBase(ColumnParallelLinearBase):
    """Merged column-parallel linear for fused gate+up projections."""


class QKVParallelLinearBase(ColumnParallelLinearBase):
    """QKV parallel linear for attention projections."""


class ReplicatedLinearBase(LinearBase):
    """Replicated (non-sharded) linear layer."""


class DistributedRoutedExpertsBase(nn.Module, ABC):
    """Abstract base for MoE distributed routed experts."""

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        is_prefill: bool = True,
    ) -> torch.Tensor: ...


class GatedDeltaNetBase(nn.Module, ABC):
    """Abstract base class for GatedDeltaNet linear attention layer."""


class AttentionBase(nn.Module, ABC):
    """Abstract base for Attention layer."""

    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor: ...


class BackendFactory(ABC):
    """Abstract factory for creating hardware-specific layer instances.

    Each backend (Hopper, GPU-generic, etc.) provides a concrete subclass.
    The factory owns the quantization config so model topology files do not
    need to reference hardware-specific quantization details.
    """

    @abstractmethod
    def get_row_parallel_linear(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        meta: bool = False,
        weight_tensor: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
        scale_tensor: Optional[torch.Tensor] = None,
        tp_group: Optional["dist.ProcessGroup"] = None,
        **kwargs,
    ) -> RowParallelLinearBase: ...

    @abstractmethod
    def get_column_parallel_linear(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        meta: bool = False,
        weight_tensor: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
        scale_tensor: Optional[torch.Tensor] = None,
        tp_group: Optional["dist.ProcessGroup"] = None,
        **kwargs,
    ) -> ColumnParallelLinearBase: ...

    @abstractmethod
    def get_merged_column_parallel_linear(
        self,
        input_size: int,
        output_sizes: list,
        bias: bool = False,
        meta: bool = False,
        weight_tensor: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
        scale_tensor: Optional[torch.Tensor] = None,
        tp_group: Optional["dist.ProcessGroup"] = None,
        **kwargs,
    ) -> MergedColumnParallelLinearBase: ...

    @abstractmethod
    def get_qkv_parallel_linear(
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
        tp_group: Optional["dist.ProcessGroup"] = None,
        **kwargs,
    ) -> QKVParallelLinearBase: ...

    @abstractmethod
    def get_replicated_linear(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        meta: bool = False,
        weight_tensor: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
        scale_tensor: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> ReplicatedLinearBase: ...

    @abstractmethod
    def get_gated_delta_net(
        self,
        layer_idx: int,
        config,
        quantization_config: Optional["QuantizationConfig"] = None,
        **kwargs,
    ) -> GatedDeltaNetBase: ...

    @abstractmethod
    def get_distributed_routed_experts(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        ep_size: int,
        tp_size: int,
        **kwargs,
    ) -> DistributedRoutedExpertsBase: ...

    @abstractmethod
    def get_attention(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        v_head_dim: int,
        attention_type: str = "MLA",
        nsa_index_topk: int = 0,
    ) -> AttentionBase: ...
