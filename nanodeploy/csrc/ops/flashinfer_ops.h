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
    void begin_forward(int32_t* block_tables_host,
                       int32_t* seq_lens_host,
                       int      batch_size,
                       int      max_num_blocks,
                       int      num_qo_heads,
                       int      num_kv_heads,
                       int      head_dim,
                       int      page_size,
                       int      window_left = -1);

    // Initialize static workspace for CUDA Graph support
    void init_workspace(int max_batch_size, int max_total_blocks);

    // Internal impl mostly for pimpl pattern if needed
    void begin_forward_impl(int32_t* block_tables_host,
                            int32_t* seq_lens_host,
                            int      batch_size,
                            int      max_num_blocks,
                            int      num_qo_heads,
                            int      num_kv_heads,
                            int      head_dim,
                            int      page_size,
                            int      window_left);

    void begin_forward_prefill(int32_t* q_indptr,
                               int32_t* block_tables,
                               int32_t* last_page_len_host,
                               int      batch_size,
                               int      max_num_blocks,  // Added
                               int      num_qo_heads,
                               int      num_kv_heads,
                               int      head_dim,
                               int      page_size);

    torch::Tensor prefill(void* q_ptr, void* k_cache_ptr, void* v_cache_ptr, int total_tokens, int layer_idx = 0);

    // Ragged prefill: Directly uses Q/K/V tensors without PagedKV cache
    // Similar to flash_attn_varlen_func
    torch::Tensor prefill_ragged(void*    q_ptr,
                                 void*    k_ptr,
                                 void*    v_ptr,
                                 int      total_q_tokens,
                                 int      total_kv_tokens,
                                 int32_t* q_indptr,   // [batch_size + 1], cumulative q lengths
                                 int32_t* kv_indptr,  // [batch_size + 1], cumulative kv lengths
                                 int      batch_size);

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
