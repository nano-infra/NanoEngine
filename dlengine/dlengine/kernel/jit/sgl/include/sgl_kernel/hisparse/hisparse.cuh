#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace sgl_kernel::hisparse {

// Phase 1 keeps KV/indexer cache resident and uses deterministic dummy-prefill.
// The kernel interface is kept separate so later HiSparse cache migration can
// replace the mapping implementation without changing CUDA graph inputs.
__global__ void identity_remap_slots_kernel(const int32_t* __restrict__ in_slots,
                                            int32_t* __restrict__ out_slots,
                                            const int32_t* __restrict__ num_real_reqs,
                                            int max_slots)
{
    int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int real = num_real_reqs == nullptr ? max_slots : *num_real_reqs;
    int n    = real < max_slots ? real : max_slots;
    if (idx < n) {
        out_slots[idx] = in_slots[idx];
    }
}

inline void launch_identity_remap_slots(const int32_t* in_slots,
                                        int32_t* out_slots,
                                        const int32_t* num_real_reqs,
                                        int max_slots,
                                        cudaStream_t stream)
{
    if (max_slots <= 0) {
        return;
    }
    constexpr int kThreads = 256;
    int blocks = (max_slots + kThreads - 1) / kThreads;
    identity_remap_slots_kernel<<<blocks, kThreads, 0, stream>>>(
        in_slots, out_slots, num_real_reqs, max_slots);
}

}  // namespace sgl_kernel::hisparse
