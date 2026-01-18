#pragma once

#include <memory>
#include <mutex>

namespace nanodeploy {

// Simplified distributed configuration focusing on attention and FFN parallelism
struct DistributedConfig {
    int global_rank = 0;
    int world_size  = 1;

    // Pipeline parallelism (if needed)
    int pp_degree = 1;

    // Attention Parallelism
    int attention_tp = 1;  // Tensor parallelism for attention
    int attention_dp = 1;  // Data parallelism for attention
    int attention_sp = 1;  // Sequence parallelism for attention

    // FFN/MoE Parallelism
    int ffn_tp = 1;  // Tensor parallelism for FFN
    int ffn_dp = 1;  // Data parallelism for FFN
    int ffn_ep = 1;  // Expert parallelism for MoE

    bool enable_cuda_graph = false;
};

class DistributedContext {
public:
    // Singleton accessor
    static DistributedContext& instance()
    {
        static DistributedContext instance;
        return instance;
    }

    void init(const DistributedConfig& config);

    bool initialized() const
    {
        return initialized_;
    }

    // Global accessors
    int global_rank() const
    {
        return config_.global_rank;
    }
    int world_size() const
    {
        return config_.world_size;
    }

    // Pipeline parallelism
    int pp_degree() const
    {
        return config_.pp_degree;
    }
    int pp_rank() const;

    // Attention parallelism
    int attention_tp() const
    {
        return config_.attention_tp;
    }
    int attention_dp() const
    {
        return config_.attention_dp;
    }
    int attention_sp() const
    {
        return config_.attention_sp;
    }
    int attention_tp_rank() const;
    int attention_dp_rank() const;

    // FFN/MoE parallelism
    int ffn_tp() const
    {
        return config_.ffn_tp;
    }
    int ffn_dp() const
    {
        return config_.ffn_dp;
    }
    int ffn_ep() const
    {
        return config_.ffn_ep;
    }
    int ffn_tp_rank() const;
    int ffn_dp_rank() const;
    int ffn_ep_rank() const;

    bool enable_cuda_graph() const
    {
        return config_.enable_cuda_graph;
    }

    // Compatibility aliases (matching old DistContext API)
    int ffn_ep_world_size() const
    {
        return ffn_ep();
    }

    // Access full config
    const DistributedConfig& config() const
    {
        return config_;
    }

private:
    // Private constructor for singleton
    DistributedContext()                                     = default;
    ~DistributedContext()                                    = default;
    DistributedContext(const DistributedContext&)            = delete;
    DistributedContext& operator=(const DistributedContext&) = delete;

    bool              initialized_ = false;
    DistributedConfig config_;
    std::mutex        mtx_;
};

// Global accessor function (for compatibility)
inline DistributedContext& get_dist_context()
{
    return DistributedContext::instance();
}

}  // namespace nanodeploy
