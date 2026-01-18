#pragma once

#include <memory>
#include <torch/torch.h>
#include <vector>

// Forward declaration to hide implementation details
namespace flashinfer {
struct DecodePlanInfo;
}

namespace nanodeploy {
namespace ops {

class FlashInferOps {
public:
    FlashInferOps(int num_layers, int num_heads, int num_kv_heads, int head_dim, int page_size, torch::Device device);
    ~FlashInferOps();  // Destructor needed for pimpl

    // Prepare metadata for a batch
    // NOTE: block_tables_host and seq_lens_host MUST point to CPU (Host) memory!
    void begin_forward(int* block_tables_host,
                       int* seq_lens_host,
                       int  batch_size,
                       int  max_num_blocks,
                       int  num_qo_heads,
                       int  num_kv_heads,
                       int  head_dim,
                       int  page_size,
                       int  window_left = -1);

    // Initialize static workspace for CUDA Graph support
    void init_workspace(int max_batch_size, int max_total_blocks);

    // Internal impl mostly for pimpl pattern if needed
    void begin_forward_impl(int* block_tables_host,
                            int* seq_lens_host,
                            int  batch_size,
                            int  max_num_blocks,
                            int  num_qo_heads,
                            int  num_kv_heads,
                            int  head_dim,
                            int  page_size,
                            int  window_left);

    // Run Attention
    torch::Tensor attention(void* q_ptr,
                            void* k_cache_ptr,
                            void* v_cache_ptr,
                            int   batch,
                            int   seq,
                            int   heads,
                            int   head_dim,
                            int   layer_idx = 0);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ops
}  // namespace nanodeploy
