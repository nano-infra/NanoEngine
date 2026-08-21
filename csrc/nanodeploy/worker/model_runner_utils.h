#pragma once

#include <string>
#include <vector>

#include "nanodeploy/sequence/sequence.h"

namespace nanodeploy {

struct PrefillMetadata {
    std::vector<int64_t> input_ids;
    std::vector<int64_t> positions;
    std::vector<int>     cu_seqlens_q;
    std::vector<int>     cu_seqlens_k;
    int                  max_seqlen_q = 0;
    int                  max_seqlen_k = 0;
    std::vector<int>     slot_mapping;

    // Flattened block tables: [sp_size * max_num_seqs * max_num_blocks]
    std::vector<int> block_tables_flat;
    int              max_num_blocks   = 0;
    bool             use_block_tables = false;
};

struct DecodeMetadata {
    std::vector<int64_t> input_ids;
    std::vector<int64_t> positions;
    std::vector<int>     slot_mapping;

    // Flattened [sp_size * max_num_seqs]
    std::vector<int> context_lens_flat;
    // Flattened [sp_size * max_num_seqs]
    std::vector<int> global_context_lens_flat;

    // Flattened block tables
    std::vector<int> block_tables_flat;
    int              max_num_blocks = 0;

    std::vector<int> context_lens_for_attn;
    bool             use_sp_a2a = false;

    std::vector<int> q_slice_get;
    std::vector<int> q_slice_fill;
    std::vector<int> q_copy_mask;

    std::vector<int> res_slice_get_to_buffer_output;
    std::vector<int> res_slice_fill_to_buffer_output;
    std::vector<int> res_to_buffer_output_mask;

    std::vector<int> res_slice_get_to_buffer_input;
    std::vector<int> res_slice_fill_to_buffer_input;
    std::vector<int> res_to_buffer_input_mask;

    std::vector<int> q_output_stride;
    std::vector<int> q_offsets;
    // Flattened [sp_size * max_num_seqs]. For each destination and local
    // sender row, stores the absolute row in that destination's packed Q
    // receive buffer. A negative value means that no remote write is needed.
    std::vector<int> q_dst_row_indices_flat;
};

PrefillMetadata
prepare_prefill_cpp(const std::vector<Sequence*>& seqs, int sp_rank, int sp_size, int block_size, int max_num_seqs);

DecodeMetadata prepare_decode_cpp(const std::vector<Sequence*>& dp_seqs,
                                  int                           sp_rank,
                                  int                           sp_size,
                                  int                           block_size,
                                  int                           max_num_seqs);

void update_seqs_inner_loop(const std::vector<Sequence*>& sp_seqs, int sp_rank);

}  // namespace nanodeploy
