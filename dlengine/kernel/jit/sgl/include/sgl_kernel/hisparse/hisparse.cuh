#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace sgl_kernel::hisparse {

// Gemma/SWA uses a per-request circular hot buffer. Keep this primitive
// graph-safe: all runtime-varying values are tensors at stable addresses and
// the caller owns the output allocation.
__global__ void build_ring_slot_mapping_kernel(const int64_t* __restrict__ slots,
                                               const int64_t* __restrict__ positions,
                                               int32_t* __restrict__ output,
                                               const int32_t* __restrict__ num_real_reqs,
                                               int64_t num_tokens,
                                               int64_t max_num_seqs,
                                               int64_t tokens_per_seq)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_tokens) {
        return;
    }

    const int64_t real = num_real_reqs == nullptr ? num_tokens : *num_real_reqs;
    const int64_t slot = slots[idx];
    if (idx >= real || slot < 0 || slot >= max_num_seqs || tokens_per_seq <= 0) {
        output[idx] = -1;
        return;
    }

    int64_t offset = positions[idx] % tokens_per_seq;
    if (offset < 0) {
        offset += tokens_per_seq;
    }
    output[idx] = static_cast<int32_t>(slot * tokens_per_seq + offset);
}

inline void launch_build_ring_slot_mapping(const int64_t* slots,
                                           const int64_t* positions,
                                           int32_t*       output,
                                           const int32_t* num_real_reqs,
                                           int64_t        num_tokens,
                                           int64_t        max_num_seqs,
                                           int64_t        tokens_per_seq,
                                           cudaStream_t   stream)
{
    if (num_tokens <= 0) {
        return;
    }
    constexpr int kThreads = 256;
    const int     blocks   = static_cast<int>((num_tokens + kThreads - 1) / kThreads);
    build_ring_slot_mapping_kernel<<<blocks, kThreads, 0, stream>>>(
        slots, positions, output, num_real_reqs, num_tokens, max_num_seqs, tokens_per_seq);
}

}  // namespace sgl_kernel::hisparse
