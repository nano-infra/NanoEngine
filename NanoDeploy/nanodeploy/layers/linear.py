"""Backward-compatibility shim for nanodeploy.layers.linear.

All concrete layer classes are now created via the backend factory.
Imports from this module continue to work for code that has not yet
been migrated to use ``get_backend()`` directly.

New code should use:
    from nanodeploy.backends import get_backend
    self.proj = get_backend().get_row_parallel_linear(...)
"""

from nanodeploy.backends import get_backend


def RowParallelLinear(
    input_size,
    output_size,
    bias=False,
    meta=False,
    weight_tensor=None,
    bias_tensor=None,
    scale_tensor=None,
    quantization_config=None,
    **kwargs
):
    return get_backend().get_row_parallel_linear(
        input_size,
        output_size,
        bias=bias,
        meta=meta,
        weight_tensor=weight_tensor,
        bias_tensor=bias_tensor,
        scale_tensor=scale_tensor,
    )


def ColumnParallelLinear(
    input_size,
    output_size,
    bias=False,
    meta=False,
    weight_tensor=None,
    bias_tensor=None,
    scale_tensor=None,
    quantization_config=None,
    **kwargs
):
    return get_backend().get_column_parallel_linear(
        input_size,
        output_size,
        bias=bias,
        meta=meta,
        weight_tensor=weight_tensor,
        bias_tensor=bias_tensor,
        scale_tensor=scale_tensor,
    )


def MergedColumnParallelLinear(
    input_size,
    output_sizes,
    bias=False,
    meta=False,
    weight_tensor=None,
    bias_tensor=None,
    scale_tensor=None,
    quantization_config=None,
    **kwargs
):
    return get_backend().get_merged_column_parallel_linear(
        input_size,
        output_sizes,
        bias=bias,
        meta=meta,
        weight_tensor=weight_tensor,
        bias_tensor=bias_tensor,
        scale_tensor=scale_tensor,
    )


def QKVParallelLinear(
    hidden_size,
    head_size,
    total_num_heads,
    total_num_kv_heads=None,
    bias=False,
    meta=False,
    weight_tensor=None,
    bias_tensor=None,
    scale_tensor=None,
    quantization_config=None,
    **kwargs
):
    return get_backend().get_qkv_parallel_linear(
        hidden_size,
        head_size,
        total_num_heads,
        total_num_kv_heads=total_num_kv_heads,
        bias=bias,
        meta=meta,
        weight_tensor=weight_tensor,
        bias_tensor=bias_tensor,
        scale_tensor=scale_tensor,
    )


def ReplicatedLinear(
    input_size,
    output_size,
    bias=False,
    meta=False,
    weight_tensor=None,
    bias_tensor=None,
    scale_tensor=None,
    quantization_config=None,
    **kwargs
):
    return get_backend().get_replicated_linear(
        input_size,
        output_size,
        bias=bias,
        meta=meta,
        weight_tensor=weight_tensor,
        bias_tensor=bias_tensor,
        scale_tensor=scale_tensor,
    )


# Re-export LinearBase for any code that imports it from this module
from nanodeploy.backends.base_backend import LinearBase  # noqa: E402, F401
