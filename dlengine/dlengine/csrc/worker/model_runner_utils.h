#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "dlengine/csrc/sequence/sequence.h"

namespace dlengine {

struct PrefillMetadata {
    std::vector<int64_t> input_ids;
    std::vector<int64_t> positions;
    std::vector<int>     cu_seqlens_q;
    std::vector<int>     cu_seqlens_k;
    int                  max_seqlen_q = 0;
    int                  max_seqlen_k = 0;
    std::vector<int>     slot_mapping;

    // Flattened block tables: [group_size * max_num_seqs * max_num_blocks]
    std::vector<int> block_tables_flat;
    int              max_num_blocks   = 0;
    bool             use_block_tables = false;

    // Per-sequence sampling info for chunked prefill:
    // sampling_token_indices[i] = index into the Q (hidden_states) tensor for
    //   the last token of the i-th final-chunk sequence.
    // sampling_seq_indices[i]   = which sequence (0-based, among master-sp seqs)
    //   that token belongs to.
    // Both vectors are empty when every sequence in the batch is a non-final chunk
    // (skip lm_head entirely).
    std::vector<int> sampling_token_indices;
    std::vector<int> sampling_seq_indices;
};

struct DecodeMetadata {
    std::vector<int64_t> input_ids;
    std::vector<int64_t> positions;
    std::vector<int>     slot_mapping;

    // Flattened [group_size * max_num_seqs]
    std::vector<int> context_lens_flat;
    // Flattened [group_size * max_num_seqs]
    std::vector<int> global_context_lens_flat;

    // Flattened block tables
    std::vector<int> block_tables_flat;
    int              max_num_blocks = 0;

    std::vector<int> context_lens_for_attn;

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
};

// ========== Auxiliary data extracted from RunBatchInput ==========

struct BatchAuxData {
    std::vector<double> temperatures;          // per master-sp seq
    std::vector<int>    state_slots;           // per master-sp seq
    std::vector<int>    master_group_indices;  // per all seqs
    int                 num_group_seqs = 0;    // count where master_group == group_rank
    // Per master-sp seq: whether to compute per-token logprobs at sample
    // time. Mirrors SamplingParams.return_completion_logprobs. Vector size
    // matches ``temperatures`` so the sampler can index in lockstep.
    std::vector<bool> return_completion_logprobs;
    // Set when ANY seq in the batch wants logprobs — sampler can fast-path
    // back to the original (logprob-free) forward when this is false.
    bool any_return_completion_logprobs = false;

    // DSv4 compressed-cache page IDs per ratio.
    // Outer map key = ratio. Inner outer index = master-group seq index.
    // Inner inner vector = list of physical page IDs owned by that seq.
    // Empty when the model has no compressed layers.
    std::unordered_map<int, std::vector<std::vector<int>>> compressed_block_tables;
};

// ========== Vision slot refs extracted from RunBatchInput ==========

struct VisionSlotView {
    std::string encoder_engine_id;
    int         slot_idx;
    int         num_tokens;
    int         hidden_size;
    int         max_tokens_per_slot;
    int         seq_index;  // index in the batch (which SequenceInput)
};

std::vector<VisionSlotView> extract_vision_slots_from_bytes(const uint8_t* data, size_t data_len);

// ========== New bytes-based API (Sequence-free on runner side) ==========

// Combined deserialize + prepare: zero Sequence objects created
PrefillMetadata prepare_prefill_from_bytes(const uint8_t* data,
                                           size_t         data_len,
                                           int            group_rank,
                                           int            group_size,
                                           int            block_size,
                                           int            max_num_seqs,
                                           int            num_gpu_blocks);

DecodeMetadata prepare_decode_from_bytes(const uint8_t* data,
                                         size_t         data_len,
                                         int            group_rank,
                                         int            group_size,
                                         int            block_size,
                                         int            max_num_seqs,
                                         int            num_gpu_blocks);

// Extract temperatures, state_slots, master_group_indices from serialized RunBatchInput
BatchAuxData extract_aux_from_bytes(const uint8_t* data, size_t data_len, int group_rank);

// ========== Migrate data structures ==========

struct MigrateSequenceView {
    uint64_t                         seq_id;
    std::string                      migrate_engine_id;
    int                              migrate_num_kvcache_blocks;
    int                              migrate_group_size;
    int                              migrate_dp_idx;
    std::vector<std::pair<int, int>> migrate_block_location;  // (group_id, block_idx)
    int                              migrate_state_slot;
    std::vector<std::pair<int, int>> active_block_location;  // (group_id, block_idx)
    int                              active_state_slot;

    // DSv4 (S2.6): per-ratio compressed page IDs the migrating seq owns on
    // the prefill engine (migrate_*) and the freshly-allocated local pages
    // on the decode engine (active_*).  Each map: ratio -> list of page IDs.
    std::unordered_map<int, std::vector<int>> migrate_compressed_block_tables;
    std::unordered_map<int, std::vector<int>> active_compressed_block_tables;
};

std::vector<MigrateSequenceView> parse_migrate_batch(const uint8_t* data, size_t data_len);

}  // namespace dlengine
