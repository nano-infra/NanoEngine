#include "nanodeploy/csrc/context/distributed_context.h"
#include "nanodeploy/csrc/logging.h"

namespace nanodeploy {

void DistributedContext::init(const DistributedConfig& config)
{
    if (initialized_) {
        NANODEPLOY_LOG_WARN("DistributedContext already initialized. Overwriting config.");
    }

    // Validation
    int total_attn          = config.attention_tp * config.attention_sp * config.attention_dp;
    int total_ffn           = config.ffn_tp * config.ffn_ep * config.ffn_dp;
    int world_size_pp_group = config.world_size / config.pp_degree;

    if (total_attn != total_ffn) {
        NANODEPLOY_ABORT("DistributedContext Init Failed: Attention world size (",
                         total_attn,
                         ") != FFN world size (",
                         total_ffn,
                         ").");
    }
    if (total_attn != world_size_pp_group) {
        NANODEPLOY_ABORT("DistributedContext Init Failed: Group world size (",
                         world_size_pp_group,
                         ") != Attention world size (",
                         total_attn,
                         "). check PP degree.");
    }

    config_      = config;
    initialized_ = true;

    NANODEPLOY_LOG_INFO("DistributedContext Initialized: Rank=", config.global_rank, " WorldSize=", config.world_size);
    NANODEPLOY_LOG_INFO("  Attention Mesh (DP->SP->TP): TP=",
                        config.attention_tp,
                        " SP=",
                        config.attention_sp,
                        " DP=",
                        config.attention_dp);
    NANODEPLOY_LOG_INFO("  FFN Mesh (DP->EP->TP): TP=", config.ffn_tp, " EP=", config.ffn_ep, " DP=", config.ffn_dp);
}

int DistributedContext::pp_rank() const
{
    // PP is outermost dimension
    // Group size for one pipeline stage (DP * SP * TP)
    int stage_group_size = config_.world_size / config_.pp_degree;
    return config_.global_rank / stage_group_size;
}

// Helper for local rank within a PP stage
inline int get_local_rank(const DistributedConfig& c)
{
    int stage_group_size = c.world_size / c.pp_degree;
    return c.global_rank % stage_group_size;
}

// Attention Mesh: DP -> SP -> TP (Innermost)
// global_rank mapped to (dp, sp, tp)
// index = dp * (sp * tp) + sp * (tp) + tp
// So:
// tp = index % tp_size
// sp = (index / tp_size) % sp_size
// dp = (index / (tp_size * sp_size)) % dp_size

int DistributedContext::attention_tp_rank() const
{
    int local = get_local_rank(config_);
    return local % config_.attention_tp;
}

int DistributedContext::attention_sp_rank() const
{
    int local = get_local_rank(config_);
    return (local / config_.attention_tp) % config_.attention_sp;
}

int DistributedContext::attention_dp_rank() const
{
    int local  = get_local_rank(config_);
    int stride = config_.attention_tp * config_.attention_sp;
    return (local / stride) % config_.attention_dp;
}

// FFN Mesh: DP -> EP -> TP (Innermost)
// index = dp * (ep * tp) + ep * tp + tp
// tp = index % tp_size
// ep = (index / tp_size) % ep_size
// dp = (index / (tp_size * ep_size)) % dp_size

int DistributedContext::ffn_tp_rank() const
{
    int local = get_local_rank(config_);
    return local % config_.ffn_tp;
}

int DistributedContext::ffn_ep_rank() const
{
    int local = get_local_rank(config_);
    return (local / config_.ffn_tp) % config_.ffn_ep;
}

int DistributedContext::ffn_dp_rank() const
{
    int local  = get_local_rank(config_);
    int stride = config_.ffn_tp * config_.ffn_ep;
    return (local / stride) % config_.ffn_dp;
}

}  // namespace nanodeploy
