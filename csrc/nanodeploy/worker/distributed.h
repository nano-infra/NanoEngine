#pragma once

#include <memory>
#include <mutex>
#include <vector>

namespace nanodeploy {

struct DistributedConfig {
    int global_rank = 0;
    int world_size  = 1;

    int tp_degree = 1;  // Tensor Parallelism
    int pp_degree = 1;  // Pipeline Parallelism
    int dp_degree = 1;  // Data Parallelism
    int ep_degree = 1;  // Expert Parallelism (MoE)
};

class DistContext {
public:
    static DistContext& instance()
    {
        static DistContext instance;
        return instance;
    }

    void init(const DistributedConfig& config);
    bool initialized() const
    {
        return initialized_;
    }

    int global_rank() const
    {
        return config_.global_rank;
    }
    int world_size() const
    {
        return config_.world_size;
    }

    int tp_rank() const;
    int tp_world_size() const
    {
        return config_.tp_degree;
    }

    int pp_rank() const;
    int pp_world_size() const
    {
        return config_.pp_degree;
    }

    int dp_rank() const;
    int dp_world_size() const
    {
        return config_.dp_degree;
    }

    // Helper to get group information (placeholder for DLSlime/ProcessGroup integration)
    // For now, we calculate ranks based on assumed topology ordering:
    // [PP, DP, TP] -> Inner to Outer: TP is innermost dimension (contiguous ranks)
    // Rank = (pp_rank * dp_size * tp_size) + (dp_rank * tp_size) + tp_rank

private:
    DistContext()                              = default;
    ~DistContext()                             = default;
    DistContext(const DistContext&)            = delete;
    DistContext& operator=(const DistContext&) = delete;

    bool              initialized_ = false;
    DistributedConfig config_;
    std::mutex        mtx_;
};

// Global accessor
DistContext& get_dist_context();

}  // namespace nanodeploy
