#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace sgl_kernel::hisparse {

// Gemma/SWA uses a per-request circular hot buffer. Keep this primitive
// graph-safe: all runtime-varying values are tensors at stable addresses and
// the caller owns the output allocation.
__global__ void build_ring_slot_mapping_kernel(const int64_t* __restrict__ slots,
                                               const int64_t* __restrict__ positions,
                                               int32_t* __restrict__ output,
                                               const int32_t* __restrict__ num_real_reqs,
                                               int64_t num_tokens,
                                               int64_t max_num_seqs,
                                               int64_t tokens_per_seq)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_tokens) {
        return;
    }

    const int64_t real = num_real_reqs == nullptr ? num_tokens : *num_real_reqs;
    const int64_t slot = slots[idx];
    if (idx >= real || slot < 0 || slot >= max_num_seqs || tokens_per_seq <= 0) {
        output[idx] = -1;
        return;
    }

    int64_t offset = positions[idx] % tokens_per_seq;
    if (offset < 0) {
        offset += tokens_per_seq;
    }
    output[idx] = static_cast<int32_t>(slot * tokens_per_seq + offset);
}

inline void launch_build_ring_slot_mapping(const int64_t* slots,
                                           const int64_t* positions,
                                           int32_t*       output,
                                           const int32_t* num_real_reqs,
                                           int64_t        num_tokens,
                                           int64_t        max_num_seqs,
                                           int64_t        tokens_per_seq,
                                           cudaStream_t   stream)
{
    if (num_tokens <= 0) {
        return;
    }
    constexpr int kThreads = 256;
    const int     blocks   = static_cast<int>((num_tokens + kThreads - 1) / kThreads);
    build_ring_slot_mapping_kernel<<<blocks, kThreads, 0, stream>>>(
        slots, positions, output, num_real_reqs, num_tokens, max_num_seqs, tokens_per_seq);
}

__global__ void load_mla_slot_kernel(const int32_t* __restrict__ logical_indices,
                                     const int32_t* __restrict__ indices,
                                     const int64_t* __restrict__ request_slots,
                                     const int32_t* __restrict__ seq_lens,
                                     uint8_t* __restrict__ resident_tokens,
                                     const char* __restrict__ cold,
                                     char* __restrict__ hot,
                                     int32_t* __restrict__ output,
                                     int32_t* __restrict__ hot_output_slots,
                                     const int32_t* __restrict__ num_real_reqs,
                                     int64_t num_rows,
                                     int64_t topk,
                                     int64_t max_num_seqs,
                                     int64_t hot_capacity,
                                     int64_t slot_stride_tokens,
                                     int64_t block_size,
                                     int64_t cold_block_stride,
                                     int64_t cold_token_stride,
                                     int64_t hot_block_stride,
                                     int64_t hot_token_stride,
                                     int64_t item_size_bytes)
{
    const int64_t row  = blockIdx.x;
    const int64_t i    = blockIdx.y;
    const int64_t real = num_real_reqs == nullptr ? num_rows : *num_real_reqs;
    if (row >= real)
        return;
    const int64_t slot = request_slots[row];
    if (slot < 0 || slot >= max_num_seqs) {
        if (threadIdx.x == 0) {
            output[row * topk + i] = -1;
            if (i == 0)
                hot_output_slots[row] = -1;
        }
        return;
    }
    const int64_t  hot_base       = slot * slot_stride_tokens;
    const int32_t  seq_len        = seq_lens[row];
    const bool     short_sequence = seq_len > 0 && seq_len <= hot_capacity;
    uint8_t* const slot_resident  = resident_tokens + slot * hot_capacity;

    const int32_t src     = indices[row * topk + i];
    const int32_t logical = logical_indices[row * topk + i];
    const bool    valid   = src >= 0 && logical >= 0 && (!short_sequence || logical < seq_len);
    const bool    current = logical == seq_len - 1;
    if (threadIdx.x == 0) {
        output[row * topk + i] =
            src < 0 || logical < 0 || (short_sequence && logical >= seq_len) ?
                -1 :
                static_cast<int32_t>(hot_base + (short_sequence ? logical : (current ? hot_capacity : i)));
        if (i == 0) {
            hot_output_slots[row] = static_cast<int32_t>(hot_base + (short_sequence ? seq_len - 1 : hot_capacity));
            if (short_sequence) {
                // store_kcache writes this token later on the same CUDA stream.
                slot_resident[seq_len - 1] = 1;
            }
        }
    }

    if (!valid || current || (short_sequence && slot_resident[logical]))
        return;
    const int64_t src_block  = src / block_size;
    const int64_t src_offset = src % block_size;
    const int64_t dst        = hot_base + (short_sequence ? logical : i);
    const int64_t dst_block  = dst / block_size;
    const int64_t dst_offset = dst % block_size;
    const char*   src_ptr    = cold + src_block * cold_block_stride + src_offset * cold_token_stride;
    char*         dst_ptr    = hot + dst_block * hot_block_stride + dst_offset * hot_token_stride;
    for (int64_t byte = threadIdx.x; byte < item_size_bytes; byte += blockDim.x) {
        dst_ptr[byte] = src_ptr[byte];
    }
    if (short_sequence && threadIdx.x == 0)
        slot_resident[logical] = 1;
}

inline void launch_load_mla_slot(const int32_t* logical_indices,
                                 const int32_t* indices,
                                 const int64_t* request_slots,
                                 const int32_t* seq_lens,
                                 uint8_t*       resident_tokens,
                                 const void*    cold,
                                 void*          hot,
                                 int32_t*       output,
                                 int32_t*       hot_output_slots,
                                 const int32_t* num_real_reqs,
                                 int64_t        num_rows,
                                 int64_t        topk,
                                 int64_t        max_num_seqs,
                                 int64_t        hot_capacity,
                                 int64_t        slot_stride_tokens,
                                 int64_t        block_size,
                                 int64_t        cold_block_stride,
                                 int64_t        cold_token_stride,
                                 int64_t        hot_block_stride,
                                 int64_t        hot_token_stride,
                                 int64_t        item_size_bytes,
                                 cudaStream_t   stream)
{
    if (num_rows <= 0)
        return;
    load_mla_slot_kernel<<<dim3(num_rows, topk), 256, 0, stream>>>(logical_indices,
                                                                   indices,
                                                                   request_slots,
                                                                   seq_lens,
                                                                   resident_tokens,
                                                                   static_cast<const char*>(cold),
                                                                   static_cast<char*>(hot),
                                                                   output,
                                                                   hot_output_slots,
                                                                   num_real_reqs,
                                                                   num_rows,
                                                                   topk,
                                                                   max_num_seqs,
                                                                   hot_capacity,
                                                                   slot_stride_tokens,
                                                                   block_size,
                                                                   cold_block_stride,
                                                                   cold_token_stride,
                                                                   hot_block_stride,
                                                                   hot_token_stride,
                                                                   item_size_bytes);
}

__global__ void writeback_mla_slot_kernel(const int32_t* logical_slots,
                                          const int32_t* hot_slots,
                                          const char*    hot,
                                          char*          cold,
                                          const int32_t* num_real_reqs,
                                          int64_t        num_rows,
                                          int64_t        block_size,
                                          int64_t        hot_block_stride,
                                          int64_t        hot_token_stride,
                                          int64_t        cold_block_stride,
                                          int64_t        cold_token_stride,
                                          int64_t        item_size_bytes)
{
    const int64_t row = blockIdx.x;
    if (row >= *num_real_reqs)
        return;
    const int32_t logical  = logical_slots[row];
    const int32_t hot_slot = hot_slots[row];
    if (logical < 0 || hot_slot < 0)
        return;
    const char* src = hot + (hot_slot / block_size) * hot_block_stride + (hot_slot % block_size) * hot_token_stride;
    char*       dst = cold + (logical / block_size) * cold_block_stride + (logical % block_size) * cold_token_stride;
    for (int64_t byte = threadIdx.x; byte < item_size_bytes; byte += blockDim.x) {
        dst[byte] = src[byte];
    }
}

inline void launch_writeback_mla_slot(const int32_t* logical_slots,
                                      const int32_t* hot_slots,
                                      const void*    hot,
                                      void*          cold,
                                      const int32_t* num_real_reqs,
                                      int64_t        num_rows,
                                      int64_t        block_size,
                                      int64_t        hot_block_stride,
                                      int64_t        hot_token_stride,
                                      int64_t        cold_block_stride,
                                      int64_t        cold_token_stride,
                                      int64_t        item_size_bytes,
                                      cudaStream_t   stream)
{
    if (num_rows <= 0)
        return;
    writeback_mla_slot_kernel<<<num_rows, 256, 0, stream>>>(logical_slots,
                                                            hot_slots,
                                                            static_cast<const char*>(hot),
                                                            static_cast<char*>(cold),
                                                            num_real_reqs,
                                                            num_rows,
                                                            block_size,
                                                            hot_block_stride,
                                                            hot_token_stride,
                                                            cold_block_stride,
                                                            cold_token_stride,
                                                            item_size_bytes);
}

}  // namespace sgl_kernel::hisparse
