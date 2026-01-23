#include "nanodeploy/csrc/context/attention_context.h"
#include "nanodeploy/csrc/logging.h"

namespace nanodeploy {

AttentionContext::AttentionContext(torch::Device device): device_(device) {}

AttentionContext::~AttentionContext() = default;

void AttentionContext::init(const core::ModelConfig& config)
{
    int head_dim   = config.head_dim > 0 ? config.head_dim : (config.hidden_size / config.num_attention_heads);
    int block_size = Sequence::block_size;

    // Store for delayed KV init
    num_layers_   = config.num_hidden_layers;
    num_kv_heads_ = config.num_key_value_heads;
    head_dim_     = head_dim;

    NANODEPLOY_LOG_INFO("Initializing FlashInferOps: layers=",
                        config.num_hidden_layers,
                        " heads=",
                        config.num_attention_heads,
                        " kv_heads=",
                        config.num_key_value_heads,
                        " head_dim=",
                        head_dim);

    handler_ = std::make_unique<ops::FlashInferOps>(config.num_hidden_layers,
                                                    config.num_attention_heads,
                                                    config.num_key_value_heads,
                                                    head_dim,
                                                    block_size,
                                                    device_);
}

bool AttentionContext::init_kv_cache(int max_batch_size, int num_blocks, int block_size)
{
    NANODEPLOY_LOG_INFO(
        "Initializing KvCache: block_size=", block_size, " num_blocks=", num_blocks, " max_bs=", max_batch_size);

    // 1. Allocate KvCache
    kv_cache_ = std::make_unique<KvCache>(num_layers_, num_kv_heads_, head_dim_, num_blocks, block_size, device_);

    // 2. Initialize FlashInfer Workspace (Static for CUDAGraphs)
    if (handler_) {
        // max_total_blocks is effectively num_blocks in the system
        // But indices_ buffer only needs to hold blocks for *active* requests.
        // In theory one request can use all blocks? Yes.
        // So max_total_blocks = num_blocks.
        handler_->init_workspace(max_batch_size, num_blocks);
    }

    return true;
}

}  // namespace nanodeploy
