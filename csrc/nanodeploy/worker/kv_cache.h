#pragma once

#include <cuda_runtime.h>  // Added for cudaDeviceSynchronize
#include <torch/torch.h>
#include <vector>

namespace nanodeploy {

class KvCache {
public:
    KvCache(int num_layers, int num_kv_heads, int head_dim, int num_blocks, int block_size, torch::Device device)
    {
        // Use BFloat16 to match FlashInfer handler and Model weights
        auto options = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
        // Layout: [num_blocks, num_kv_heads, block_size, head_dim] (NHD-like but block-based)
        // This matches FlashInfer typical expectation (NHD or HND supported).
        // We choose [num_blocks, num_kv_heads, block_size, head_dim].
        for (int i = 0; i < num_layers; ++i) {
            // k_cache: [num_blocks, num_kv_heads, block_size, head_dim]
            k_caches.push_back(torch::empty({num_blocks, num_kv_heads, block_size, head_dim}, options));
            // v_cache: [num_blocks, num_kv_heads, block_size, head_dim]
            v_caches.push_back(torch::empty({num_blocks, num_kv_heads, block_size, head_dim}, options));
        }
    }

    std::vector<torch::Tensor> k_caches;
    std::vector<torch::Tensor> v_caches;

    // Helper to write to cache using flat slot mapping
    // slot_mapping: [batch_size] (for decode) or [total_tokens] (for prefill)
    // k, v: [batch_size, num_kv_heads, head_dim]
    void set_kv(int layer_idx, torch::Tensor slot_mapping, torch::Tensor k, torch::Tensor v)
    {
        // Loop-based method with correct indexing
        // Cache layout: [NumBlocks, NumKV, BlockSize, HeadDim]

        // Ensure k, v are on same device/dtype as cache
        auto device = k_caches[layer_idx].device();
        auto dtype  = k_caches[layer_idx].scalar_type();

        if (k.scalar_type() != dtype)
            k = k.to(dtype);
        if (v.scalar_type() != dtype)
            v = v.to(dtype);
        // Keep k, v on GPU
        if (k.device() != device)
            k = k.to(device);
        if (v.device() != device)
            v = v.to(device);

        int64_t block_size = k_caches[layer_idx].size(2);
        int     batch_size = slot_mapping.size(0);

        // Sync slot_mapping to CPU for indexing
        auto slots_cpu = slot_mapping.to(torch::kCPU, torch::kLong);
        auto slots_acc = slots_cpu.accessor<int64_t, 1>();

        using namespace torch::indexing;

        for (int i = 0; i < batch_size; ++i) {
            int64_t slot      = slots_acc[i];
            int64_t block_idx = slot / block_size;
            int64_t offset    = slot % block_size;

            // k[i] is [NumKV, HeadDim]
            // Target: cache[block_idx, :, offset, :] which is [NumKV, HeadDim]
            // Use index_put_ with scalar indices
            k_caches[layer_idx].index_put_({block_idx, Slice(), offset, Slice()}, k[i]);
            v_caches[layer_idx].index_put_({block_idx, Slice(), offset, Slice()}, v[i]);
        }
    }
    // Gather KV from cache for SDPA (Slow Path)
    // block_tables: [Batch, MaxBlocks]
    // seq_lens: [Batch]
    // Returns {K, V} as [Batch, NumKV, TotalSeq, Dim] (Compatible with SDPA broadcasting)
    std::tuple<torch::Tensor, torch::Tensor>
    gather_kv(int layer_idx, torch::Tensor block_tables, torch::Tensor seq_lens)
    {
        int batch_size  = block_tables.size(0);
        int max_seq_len = seq_lens.max().item<int>();
        int num_kv      = k_caches[layer_idx].size(1);
        int block_size  = k_caches[layer_idx].size(2);
        int head_dim    = k_caches[layer_idx].size(3);

        auto options = k_caches[layer_idx].options();
        auto k_out   = torch::zeros({batch_size, num_kv, max_seq_len, head_dim}, options);
        auto v_out   = torch::zeros({batch_size, num_kv, max_seq_len, head_dim}, options);

        auto block_tables_cpu_t = block_tables.to(torch::kCPU);
        auto block_tables_cpu   = block_tables_cpu_t.accessor<int, 2>();
        auto seq_lens_cpu_t     = seq_lens.to(torch::kCPU);
        auto seq_lens_cpu       = seq_lens_cpu_t.accessor<int, 1>();

        using namespace torch::indexing;

        for (int i = 0; i < batch_size; ++i) {
            int seq_len    = seq_lens_cpu[i];
            int num_blocks = (seq_len + block_size - 1) / block_size;

            for (int b = 0; b < num_blocks; ++b) {
                int block_idx = block_tables_cpu[i][b];
                // Determine length of this block valid for this seq
                // If it's the last block, it might be partial?
                // Actually seq_len determines how many tokens we need.
                // block b covers range [b*BS, (b+1)*BS).
                // Intersection with [0, seq_len).

                int start = b * block_size;
                int end   = std::min(start + block_size, seq_len);
                int len   = end - start;

                if (len <= 0)
                    break;

                // Copy
                // Src: cache[block_idx, :, 0:len, :]
                // Dst: out[i, :, start:end, :]

                k_out.index_put_({i, Slice(), Slice(start, end), Slice()},
                                 k_caches[layer_idx].index({block_idx, Slice(), Slice(0, len), Slice()}));
                v_out.index_put_({i, Slice(), Slice(start, end), Slice()},
                                 v_caches[layer_idx].index({block_idx, Slice(), Slice(0, len), Slice()}));
            }
        }

        // Transpose to [Batch, NumKV, Seq, Dim] -> [Batch, Seq, NumKV, Dim]?
        // SDPA expects [Batch, Heads, Seq, Dim] (or NumKV broadcasted).
        // My output is [B, KV, S, D].
        // For SDPA we usually want [B, H, S, D]. Since this is K/V, [B, KV, S, D] is fine if H=KV.
        // If GQA, we repeat later.
        // But SDPA usually takes [B, H, S, D]. PyTorch SDPA supports GQA natively in newer versions/or via repeat.
        // Let's keep it [B, KV, S, D] and let caller handle repeat.
        // Actually, let's enable native broadcasting by keeping it as is.

        return {k_out, v_out};
    }
};

}  // namespace nanodeploy
