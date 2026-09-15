"""Allocation-free LSE merge for the CP experiment; FP32 communication/accumulation."""
import torch
import torch.distributed as dist
import triton
import triton.language as tl


@triton.jit
def _pack(O, L, M, P, Q: tl.constexpr, H: tl.constexpr, V: tl.constexpr,
          LS0: tl.constexpr, LS1: tl.constexpr, R: tl.constexpr):
    row = tl.program_id(0) * R + tl.arange(0, R)
    col = tl.arange(0, triton.next_power_of_2(V))
    q, h = row // H, row % H
    lse = tl.load(L + h * LS0 + q * LS1, row < Q * H, other=-float('inf'))
    maximum = tl.load(M + h * Q + q, row < Q * H, other=0)
    weight = tl.exp(lse - maximum)
    weight = tl.where(maximum == -float('inf'), 0., weight)
    value = tl.load(O + row[:, None] * V + col[None, :],
                    (row[:, None] < Q * H) & (col[None, :] < V), other=0).to(tl.float32)
    tl.store(P + row[:, None] * (V + 1) + col[None, :], value * weight[:, None],
             (row[:, None] < Q * H) & (col[None, :] < V))
    tl.store(P + row * (V + 1) + V, weight, row < Q * H)


@triton.jit
def _finish(P, O, N: tl.constexpr, V: tl.constexpr, R: tl.constexpr):
    row = tl.program_id(0) * R + tl.arange(0, R)
    col = tl.arange(0, triton.next_power_of_2(V))
    denom = tl.load(P + row * (V + 1) + V, row < N, other=1)
    value = tl.load(P + row[:, None] * (V + 1) + col[None, :],
                    (row[:, None] < N) & (col[None, :] < V), other=0)
    value = tl.where(denom[:, None] > 0, value / denom[:, None], 0.)
    tl.store(O + row[:, None] * V + col[None, :], value,
             (row[:, None] < N) & (col[None, :] < V))


class ContextMerger:
    def __init__(self, queries, heads, value, group):
        self.q, self.h, self.v, self.group = queries, heads, value, group
        self.maximum = torch.empty(heads, queries, device='cuda', dtype=torch.float32)
        self.packed = torch.empty(queries, heads, value + 1, device='cuda', dtype=torch.float32)
        self.output = torch.empty(queries, heads, value, device='cuda', dtype=torch.bfloat16)

    def __call__(self, output, lse):
        self.maximum.copy_(lse)
        dist.all_reduce(self.maximum, op=dist.ReduceOp.MAX, group=self.group)
        grid = (triton.cdiv(self.q * self.h, 16),)
        _pack[grid](output, lse, self.maximum, self.packed, self.q, self.h, self.v,
                    lse.stride(0), lse.stride(1), 16)
        dist.all_reduce(self.packed, group=self.group)
        _finish[grid](self.packed, self.output, self.q * self.h, self.v, 16)
        return self.output
