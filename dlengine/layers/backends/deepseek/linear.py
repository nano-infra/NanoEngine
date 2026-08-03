"""DeepGEMM-backed quantized linear layers."""

from dlengine.layers.hopper.linear import (
    HopperColumnParallelLinear,
    HopperMergedColumnParallelLinear,
    HopperQKVParallelLinear,
    HopperReplicatedLinear,
    HopperRowParallelLinear,
    HopperColumnParallelLinear as ColumnParallelLinear,
    HopperMergedColumnParallelLinear as MergedColumnParallelLinear,
    HopperQKVParallelLinear as QKVParallelLinear,
    HopperReplicatedLinear as ReplicatedLinear,
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
