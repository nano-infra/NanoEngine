import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    output_ptr,
    input_ptr,
    weight_ptr,
    n_elements,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    # Load input row
    row_start_ptr = input_ptr + row_idx * n_elements
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(row_start_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Calculate variance: mean(x^2)
    x2 = x * x
    var = tl.sum(x2, axis=0) / n_elements
    rsqrt_var = tl.math.rsqrt(var + eps)
    
    # Normalize
    x_norm = x * rsqrt_var
    
    # Apply weight
    w = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = x_norm * w
    
    # Store result
    out_start_ptr = output_ptr + row_idx * n_elements
    tl.store(out_start_ptr + offsets, y.to(output_ptr.dtype.element_ty), mask=mask)


@triton.jit
def fused_add_rms_norm_kernel(
    input_ptr,
    residual_ptr,
    weight_ptr,
    n_elements,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    # Load input and residual row
    row_offset = row_idx * n_elements
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + row_offset + offsets, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(residual_ptr + row_offset + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # x = x + res
    x = x + res
    
    # Store updated residual (sum)
    tl.store(residual_ptr + row_offset + offsets, x.to(residual_ptr.dtype.element_ty), mask=mask)
    
    # Calculate variance: mean(x^2)
    x2 = x * x
    var = tl.sum(x2, axis=0) / n_elements
    rsqrt_var = tl.math.rsqrt(var + eps)
    
    # Normalize
    x_norm = x * rsqrt_var
    
    # Apply weight
    w = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = x_norm * w
    
    # Store normalized result in input tensor
    tl.store(input_ptr + row_offset + offsets, y.to(input_ptr.dtype.element_ty), mask=mask)


def rms_norm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float):
    # Flatten if needed, assuming [..., hidden_size]
    orig_shape = x.shape
    x = x.reshape(-1, orig_shape[-1])
    
    n_rows, n_elements = x.shape
    out = torch.empty_like(x)
    
    BLOCK_SIZE = triton.next_power_of_2(n_elements)
    
    grid = (n_rows,)
    rms_norm_kernel[grid](
        out, x, weight, n_elements, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out.reshape(orig_shape)


def fused_add_rms_norm_triton(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float):
    # Flatten if needed, assuming [..., hidden_size]
    orig_shape = x.shape
    x = x.reshape(-1, orig_shape[-1])
    residual = residual.reshape(-1, orig_shape[-1])
    
    n_rows, n_elements = x.shape
    
    BLOCK_SIZE = triton.next_power_of_2(n_elements)
    
    grid = (n_rows,)
    fused_add_rms_norm_kernel[grid](
        x, residual, weight, n_elements, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return x.reshape(orig_shape), residual.reshape(orig_shape)
