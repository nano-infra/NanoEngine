#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <torch/torch.h>
#include <tuple>
#include <vector>

#include "nanodeploy/csrc/core/config.h"
#include "nanodeploy/csrc/logging.h"
#include "nanodeploy/csrc/ops/flashinfer_ops.h"
#include "nanodeploy/csrc/sequence/sequence.h"

namespace nanodeploy {

class KvCache {
public:
    KvCache(int num_layers, int num_kv_heads, int head_dim, int num_blocks, int block_size, torch::Device device)
    {
        // Use BFloat16 to match FlashInfer handler and Model weights
        auto options = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
        // Layout: [num_blocks, num_kv_heads, block_size, head_dim] (NHD-like but block-based)

        // Calculate and log tensor shape before allocation
        size_t per_tensor_bytes = (size_t)num_blocks * num_kv_heads * block_size * head_dim * 2;  // BF16 = 2 bytes
        size_t total_bytes      = per_tensor_bytes * num_layers * 2;  // K + V for each layer
        NANODEPLOY_LOG_INFO(
            "[KvCache] Shape: [", num_blocks, ", ", num_kv_heads, ", ", block_size, ", ", head_dim, "]");
        NANODEPLOY_LOG_INFO("[KvCache] Layers: ", num_layers, ", Per-tensor: ", per_tensor_bytes / 1024 / 1024, " MB");
        NANODEPLOY_LOG_INFO(
            "[KvCache] Total: ", total_bytes / 1024 / 1024 / 1024, " GB (", num_layers * 2, " tensors)");

        for (int i = 0; i < num_layers; ++i) {
            // Use zeros instead of empty to avoid uninitialized memory issues
            k_caches.push_back(torch::zeros({num_blocks, num_kv_heads, block_size, head_dim}, options));
            v_caches.push_back(torch::zeros({num_blocks, num_kv_heads, block_size, head_dim}, options));
        }
    }

    std::vector<torch::Tensor> k_caches;
    std::vector<torch::Tensor> v_caches;

    // Helper to write to cache using flat slot mapping
    void set_kv(int layer_idx, torch::Tensor slot_mapping, torch::Tensor k, torch::Tensor v)
    {
        // Ensure inputs are on the same device/dtype as cache
        auto device = k_caches[layer_idx].device();
        auto dtype  = k_caches[layer_idx].scalar_type();

        if (k.scalar_type() != dtype)
            k = k.to(dtype);
        if (v.scalar_type() != dtype)
            v = v.to(dtype);
        if (k.device() != device)
            k = k.to(device);
        if (v.device() != device)
            v = v.to(device);
        if (slot_mapping.device() != device)
            slot_mapping = slot_mapping.to(device);

        int64_t block_size = k_caches[layer_idx].size(2);
        int64_t head_dim   = k_caches[layer_idx].size(3);

        // Checker: [num_blocks, num_kv_heads, block_size, head_dim]
        // k, v shape: [batch, num_kv_heads, head_dim] (flattened?)
        // Actually `AttentionContext` calls `view({-1, num_kv_heads, head_dim})`
        // So k, v are [total_tokens, num_kv_heads, head_dim]

        // Let's verify K/V last dim matches head_dim
        if (k.size(-1) != head_dim || v.size(-1) != head_dim) {
            NANODEPLOY_LOG_WARN("KvCache::set_kv: Dimension mismatch! Expected HeadDim=",
                                head_dim,
                                " Got K=",
                                k.size(-1),
                                " V=",
                                v.size(-1));
        }

        // Verify num_kv_heads
        int64_t num_kv_heads = k_caches[layer_idx].size(1);
        if (k.size(-2) != num_kv_heads || v.size(-2) != num_kv_heads) {
            NANODEPLOY_LOG_WARN("KvCache::set_kv: Head count mismatch! Expected KVHeads=",
                                num_kv_heads,
                                " Got K=",
                                k.size(-2),
                                " V=",
                                v.size(-2));
        }

        // Vectorized indexing on GPU
        auto block_indices  = slot_mapping.div(block_size, "trunc");
        auto offset_indices = slot_mapping.remainder(block_size);

        using namespace torch::indexing;

        // k, v shape: [batch_size, num_kv_heads, head_dim]
        // Indices shapes: [batch_size]
        // Target index: [block_indices, :, offset_indices, :]

        // We use index_put_ with advanced indexing.
        // Note: We need to ensure indices are LongTensor
        if (block_indices.scalar_type() != torch::kLong)
            block_indices = block_indices.to(torch::kLong);
        if (offset_indices.scalar_type() != torch::kLong)
            offset_indices = offset_indices.to(torch::kLong);

        k_caches[layer_idx].index_put_({block_indices, Slice(), offset_indices, Slice()}, k);
        v_caches[layer_idx].index_put_({block_indices, Slice(), offset_indices, Slice()}, v);
    }
};

class AttentionContext {
public:
    AttentionContext(torch::Device device);
    ~AttentionContext();

    void init(const core::ModelConfig& config);
    bool init_kv_cache(int max_batch_size, int num_blocks, int block_size);

    ops::FlashInferOps* get_handler() const
    {
        return handler_.get();
    }
    KvCache* get_kv_cache() const
    {
        return kv_cache_.get();
    }

    // Getters for KV cache calculation
    int num_layers() const
    {
        return num_layers_;
    }
    int num_kv_heads() const
    {
        return num_kv_heads_;
    }
    int head_dim() const
    {
        return head_dim_;
    }

    // Future extension: support for other attention backends

private:
    torch::Device                       device_;
    std::unique_ptr<ops::FlashInferOps> handler_;
    std::unique_ptr<KvCache>            kv_cache_;

    // Stored config for delayed init
    int num_layers_   = 0;
    int num_kv_heads_ = 0;
    int head_dim_     = 0;
};

}  // namespace nanodeploy
