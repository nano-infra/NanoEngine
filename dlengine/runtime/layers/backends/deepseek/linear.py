"""DeepGEMM-backed quantized linear layers."""

from dlengine.runtime.layers.hopper.linear import (
    HopperColumnParallelLinear,
    HopperColumnParallelLinear as ColumnParallelLinear,
    HopperMergedColumnParallelLinear,
    HopperMergedColumnParallelLinear as MergedColumnParallelLinear,
    HopperQKVParallelLinear,
    HopperQKVParallelLinear as QKVParallelLinear,
    HopperReplicatedLinear,
    HopperReplicatedLinear as ReplicatedLinear,
    HopperRowParallelLinear,
    HopperRowParallelLinear as RowParallelLinear,
)

__all__ = [
    "ColumnParallelLinear",
    "MergedColumnParallelLinear",
    "QKVParallelLinear",
    "ReplicatedLinear",
    "RowParallelLinear",
    "HopperColumnParallelLinear",
    "HopperMergedColumnParallelLinear",
    "HopperQKVParallelLinear",
    "HopperReplicatedLinear",
    "HopperRowParallelLinear",
]
