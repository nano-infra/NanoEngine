#pragma once

#include <cstdlib>
#include <string>

namespace nanodeploy {

// Set NVSHMEM environment variables required for DeepEP low-latency mode
// Must be called BEFORE creating the Buffer (which initializes NVSHMEM)
//
// @param num_qps_per_rank: Number of queue pairs per rank, should be >= number of local experts
// @param single_card: If true, use minimal configuration for single-card mode
inline void setup_nvshmem_env(int num_qps_per_rank = 32, bool single_card = false)
{
    if (single_card) {
        // Single-card: Minimal NVSHMEM setup
        setenv("NVSHMEM_DISABLE_P2P", "0", 0);
        setenv("NVSHMEM_DISABLE_NVLS", "1", 0);
        setenv("NVSHMEM_DISABLE_MNNVL", "1", 0);
        setenv("NVSHMEM_CUMEM_GRANULARITY", "536870912", 0);  // 512 MiB
    }
    else {
        // Multi-card: Full NVSHMEM configuration
        // Enable IBGDA (InfiniBand GPUDirect Async)
        setenv("NVSHMEM_IB_ENABLE_IBGDA", "1", 0);

        // Number of QPs per rank - should be >= number of local experts
        setenv("NVSHMEM_IBGDA_NUM_RC_PER_PE", std::to_string(num_qps_per_rank).c_str(), 0);

        // Allow P2P (NVLink) traffic
        setenv("NVSHMEM_DISABLE_P2P", "0", 0);

        // QP depth - must be larger than on-flight WRs
        setenv("NVSHMEM_QP_DEPTH", "1024", 0);

        // Reduce GPU memory usage
        setenv("NVSHMEM_MAX_TEAMS", "7", 0);

        // Disable NVLink SHArP
        setenv("NVSHMEM_DISABLE_NVLS", "1", 0);

        // NVSHMEM initialization requires at least 256 MiB granularity
        setenv("NVSHMEM_CUMEM_GRANULARITY", "536870912", 0);  // 2^29 = 512 MiB

        // Disable multi-node NVLink detection (for single node testing)
        setenv("NVSHMEM_DISABLE_MNNVL", "1", 0);
    }
}

}  // namespace nanodeploy
