#pragma once

#include <memory>
#include <mutex>
#include <vector>

namespace nanodeploy {

struct DistributedConfig {
    int global_rank = 0;
    int world_size  = 1;

    // Legacy / Global
    int tp_degree = 1;
    int pp_degree = 1;
    int dp_degree = 1;
    int ep_degree = 1;

    // Detailed Attention Parallelism
    int attention_tp = 1;
    int attention_dp = 1;
    int attention_sp = 1;

    // Detailed FFN Parallelism
    int ffn_tp = 1;
    int ffn_dp = 1;
    int ffn_ep = 1;
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

    // Accessors for detailed config
    const DistributedConfig& config() const
    {
        return config_;
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

    int ep_rank() const;
    int ep_world_size() const
    {
        return config_.ep_degree;
    }

    // Attention
    int attention_tp_world_size() const
    {
        return config_.attention_tp;
    }
    int attention_dp_world_size() const
    {
        return config_.attention_dp;
    }
    int attention_sp_world_size() const
    {
        return config_.attention_sp;
    }

    // FFN
    int ffn_tp_world_size() const
    {
        return config_.ffn_tp;
    }
    int ffn_dp_world_size() const
    {
        return config_.ffn_dp;
    }
    int ffn_ep_world_size() const
    {
        return config_.ffn_ep;
    }

    // Rank helpers (assuming standard layout or relying on global_rank mapping)
    // For now, reuse ep_rank() logic for ffn_ep_rank if topology matches
    // But ideally we should calculate ranks based on group membership.

    // Alias for Qwen3 MoE compatibility
    int ffn_ep_rank() const
    {
        // If we assume [DP, EP] layout where EP ranks are contiguous or strided?
        // Usually EP = WorldSize / TP.
        // If Attention DP=8, Expert EP=8. Rank 0..7.
        // rank 0 is EP rank 0.
        // rank 1 is EP rank 1.
        return global_rank() % ffn_ep_world_size();
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
