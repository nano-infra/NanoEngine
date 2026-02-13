#include <algorithm>
#include <iostream>

#include "nanosequence/csrc/sequence/sequence.h"
#include "sequence_generated.h"

#include "model_runner_utils.h"

namespace nanodeploy {

static void build_block_tables_packed(const std::vector<Sequence*>& dp_seqs,
                                      int                           sp_rank,
                                      int                           sp_size,
                                      std::vector<int>&             block_tables_flat,
                                      int&                          max_num_blocks)
{
    // 1. Group sequences
    std::vector<std::vector<Sequence*>> dp_sp_seqs(sp_size);
    for (auto* seq : dp_seqs) {
        int m_sp = seq->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx;
        if (m_sp >= 0 && m_sp < sp_size) {
            dp_sp_seqs[m_sp].push_back(seq);
        }
    }

    // 2. Collect valid block tables and calculate max_num_blocks
    max_num_blocks = 0;
    std::vector<std::vector<int>> valid_tables;

    for (int sp_idx = 0; sp_idx < sp_size; ++sp_idx) {
        for (auto* seq : dp_sp_seqs[sp_idx]) {
            const auto& bt = seq->block_table(BlockContextSlot::ACTIVE, sp_rank);
            if (!bt.empty()) {
                valid_tables.push_back(bt);
                if ((int)bt.size() > max_num_blocks) {
                    max_num_blocks = (int)bt.size();
                }
            }
        }
    }

    // 3. Flatten and pad
    // If there are no valid block tables, block_tables_flat is empty and max_num_blocks is 0
    if (valid_tables.empty()) {
        block_tables_flat.clear();
        return;
    }

    block_tables_flat.reserve(valid_tables.size() * max_num_blocks);
    for (const auto& bt : valid_tables) {
        // Copy data
        block_tables_flat.insert(block_tables_flat.end(), bt.begin(), bt.end());
        // Pad with -1
        int padding = max_num_blocks - (int)bt.size();
        for (int k = 0; k < padding; ++k) {
            block_tables_flat.push_back(-1);
        }
    }
}

// Helper for dense block tables (keep for prefill if needed, or implement inline)
static void build_block_tables_dense(const std::vector<Sequence*>& dp_seqs,
                                     int                           sp_rank,
                                     int                           sp_size,
                                     int                           max_num_seqs,
                                     std::vector<int>&             block_tables_flat,
                                     int&                          max_num_blocks)
{
    // 1. Group sequences by master_sp_idx
    std::vector<std::vector<Sequence*>> dp_sp_seqs(sp_size);
    for (auto* seq : dp_seqs) {
        int m_sp = seq->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx;
        if (m_sp >= 0 && m_sp < sp_size) {
            dp_sp_seqs[m_sp].push_back(seq);
        }
    }

    // 2. Calculate max_num_blocks
    max_num_blocks = 0;
    for (int sp_idx = 0; sp_idx < sp_size; ++sp_idx) {
        for (auto* seq : dp_sp_seqs[sp_idx]) {
            int size = (int)seq->block_table(BlockContextSlot::ACTIVE, sp_rank).size();
            if (size > max_num_blocks)
                max_num_blocks = size;
        }
    }

    // 3. Fill table: [sp_size, max_num_seqs, max_num_blocks]
    size_t total_size = (size_t)sp_size * max_num_seqs * max_num_blocks;
    block_tables_flat.assign(total_size, -1);

    for (int sp_idx = 0; sp_idx < sp_size; ++sp_idx) {
        const auto& seqs = dp_sp_seqs[sp_idx];
        for (int seq_id = 0; seq_id < max_num_seqs; ++seq_id) {
            size_t base_offset = ((size_t)sp_idx * max_num_seqs + seq_id) * max_num_blocks;
            if (seq_id < (int)seqs.size()) {
                Sequence*   seq = seqs[seq_id];
                const auto& bt  = seq->block_table(BlockContextSlot::ACTIVE, sp_rank);
                for (size_t i = 0; i < bt.size(); ++i) {
                    block_tables_flat[base_offset + i] = bt[i];
                }
            }
        }
    }
}

PrefillMetadata
prepare_prefill_cpp(const std::vector<Sequence*>& seqs, int sp_rank, int sp_size, int block_size, int max_num_seqs)
{
    PrefillMetadata meta;
    meta.cu_seqlens_q.push_back(0);
    meta.cu_seqlens_k.push_back(0);

    size_t est_tokens = seqs.size() * 256;
    meta.input_ids.reserve(est_tokens);
    meta.positions.reserve(est_tokens);
    meta.slot_mapping.reserve(est_tokens);

    for (auto* seq : seqs) {
        if (seq->block_ctx().master_sp_idx != sp_rank) {
            continue;
        }

        int seqlen     = seq->num_tokens();
        int num_cached = seq->num_cached_tokens();
        int seqlen_q   = seqlen - num_cached;
        int seqlen_k   = seqlen;

        const auto& full_tokens = seq->token_ids();
        for (int i = num_cached; i < seqlen; ++i) {
            meta.input_ids.push_back(full_tokens[i]);
            meta.positions.push_back(i);
        }

        meta.cu_seqlens_q.push_back(meta.cu_seqlens_q.back() + seqlen_q);
        meta.cu_seqlens_k.push_back(meta.cu_seqlens_k.back() + seqlen_k);
        meta.max_seqlen_q = std::max(meta.max_seqlen_q, seqlen_q);
        meta.max_seqlen_k = std::max(meta.max_seqlen_k, seqlen_k);

        const auto& bt = seq->block_table(BlockContextSlot::ACTIVE, sp_rank);
        if (bt.empty()) {
            continue;
        }

        int num_blocks        = seq->num_blocks(BlockContextSlot::ACTIVE, sp_rank);
        int num_cached_blocks = seq->num_cached_blocks();

        for (int i = num_cached_blocks; i < num_blocks; ++i) {
            int block_id = bt[i];
            int start    = block_id * block_size;
            int end      = (i != num_blocks - 1) ? start + block_size :
                                                   start + seq->last_block_num_tokens(BlockContextSlot::ACTIVE, sp_rank);

            for (int k = start; k < end; ++k) {
                meta.slot_mapping.push_back(k);
            }
        }
    }

    if (meta.cu_seqlens_k.back() > meta.cu_seqlens_q.back()) {
        meta.use_block_tables = true;
        build_block_tables_dense(seqs, sp_rank, sp_size, max_num_seqs, meta.block_tables_flat, meta.max_num_blocks);
    }

    return meta;
}

DecodeMetadata
prepare_decode_cpp(const std::vector<Sequence*>& dp_seqs, int sp_rank, int sp_size, int block_size, int max_num_seqs)
{
    DecodeMetadata meta;

    // 1. Prepare input_ids, positions, slot_mapping
    for (auto* seq : dp_seqs) {
        if (seq->block_ctx().master_sp_idx == sp_rank) {
            meta.input_ids.push_back(seq->last_token());
            meta.positions.push_back(seq->num_tokens() - 1);

            int page_id = seq->last_block_page_id(BlockContextSlot::ACTIVE, sp_rank);
            int offset  = seq->last_block_num_tokens(BlockContextSlot::ACTIVE, sp_rank);
            meta.slot_mapping.push_back(page_id * block_size + offset - 1);
        }
    }

    // Group sequences
    std::vector<std::vector<Sequence*>> sp_seqs(sp_size);
    for (auto* seq : dp_seqs) {
        int m_sp = seq->block_ctx().master_sp_idx;
        if (m_sp >= 0 && m_sp < sp_size) {
            sp_seqs[m_sp].push_back(seq);
        }
    }

    // 2. Prepare context_lens, global_context_lens
    meta.context_lens_flat.assign(sp_size * max_num_seqs, 0);
    meta.global_context_lens_flat.assign(sp_size * max_num_seqs, 0);

    // Used internally to calculate q_slice_fill and q_offsets
    std::vector<int> sp_valid_request_counts(sp_size, 0);

    // context_lens
    for (int sp_idx = 0; sp_idx < sp_size; ++sp_idx) {
        const auto& batch_seqs  = sp_seqs[sp_idx];
        int         valid_count = 0;
        for (int seq_id = 0; seq_id < max_num_seqs; ++seq_id) {
            if (seq_id < (int)batch_seqs.size()) {
                Sequence* seq     = batch_seqs[seq_id];
                int       ctx_len = seq->context_len(BlockContextSlot::ACTIVE, sp_rank);
                meta.context_lens_flat[sp_idx * max_num_seqs + seq_id] = ctx_len;

                if (ctx_len > 0) {
                    valid_count++;
                }
            }
        }
        sp_valid_request_counts[sp_idx] = valid_count;
    }

    // Fill global_context_lens
    const auto& my_master_seqs = sp_seqs[sp_rank];
    for (int sp_idx = 0; sp_idx < sp_size; ++sp_idx) {
        for (int seq_id = 0; seq_id < max_num_seqs; ++seq_id) {
            if (seq_id < (int)my_master_seqs.size()) {
                Sequence* seq = my_master_seqs[seq_id];
                meta.global_context_lens_flat[sp_idx * max_num_seqs + seq_id] =
                    seq->context_len(BlockContextSlot::ACTIVE, sp_idx);
            }
        }
    }

    // 3. Block tables
    build_block_tables_packed(dp_seqs, sp_rank, sp_size, meta.block_tables_flat, meta.max_num_blocks);

    // 4. context_lens_for_attn
    for (int sp_idx = 0; sp_idx < sp_size; ++sp_idx) {
        const auto& batch_seqs = sp_seqs[sp_idx];
        for (int seq_id = 0; seq_id < (int)batch_seqs.size(); ++seq_id) {
            int ctx_len = batch_seqs[seq_id]->context_len(BlockContextSlot::ACTIVE, sp_rank);
            if (ctx_len > 0) {
                meta.context_lens_for_attn.push_back(ctx_len);
            }
        }
    }

    // --- Migration Logic ---

    // q_slice_get: seq_id for my rank where ctx_len > 0
    const auto& my_sp_seqs = sp_seqs[sp_rank];
    for (int seq_id = 0; seq_id < (int)my_sp_seqs.size(); ++seq_id) {
        int ctx_len = my_sp_seqs[seq_id]->context_len(BlockContextSlot::ACTIVE, sp_rank);
        if (ctx_len > 0) {
            meta.q_slice_get.push_back(seq_id);
        }
    }

    // q_slice_fill
    int current_pos = 0;
    for (int sp_idx = 0; sp_idx < sp_size; ++sp_idx) {
        if (sp_idx == sp_rank) {
            for (size_t k = 0; k < meta.q_slice_get.size(); ++k) {
                meta.q_slice_fill.push_back(current_pos + k);
            }
        }
        current_pos += sp_valid_request_counts[sp_idx];
    }

    // q_copy_mask
    meta.q_copy_mask.assign(meta.q_slice_get.size(), 1);

    // res_slice_get_to_buffer_output (same logic as q_slice_fill logic for sp_rank in python code)
    // In python: res_slice_get_to_buffer_output = q_slice_fill.copy()
    meta.res_slice_get_to_buffer_output = meta.q_slice_fill;

    // res_slice_fill_to_buffer_output
    for (int seq_index : meta.q_slice_get) {
        meta.res_slice_fill_to_buffer_output.push_back(sp_rank * max_num_seqs + seq_index);
    }

    // res_to_buffer_output_mask
    meta.res_to_buffer_output_mask.assign(meta.res_slice_get_to_buffer_output.size(), 1);

    // res_slice_get_to_buffer_input & res_slice_fill_to_buffer_input
    int current_attention_pos = 0;
    for (int sp_idx = 0; sp_idx < sp_size; ++sp_idx) {
        if (sp_idx == sp_rank) {
            current_attention_pos += sp_valid_request_counts[sp_idx];
            continue;
        }

        const auto& batch_seqs = sp_seqs[sp_idx];
        for (int seq_id = 0; seq_id < (int)batch_seqs.size(); ++seq_id) {
            int ctx_len = batch_seqs[seq_id]->context_len(BlockContextSlot::ACTIVE, sp_rank);
            if (ctx_len > 0) {
                meta.res_slice_get_to_buffer_input.push_back(current_attention_pos);
                meta.res_slice_fill_to_buffer_input.push_back(sp_idx * max_num_seqs + seq_id);
                current_attention_pos++;
            }
        }
    }

    // res_to_buffer_input_mask
    meta.res_to_buffer_input_mask.assign(meta.res_slice_get_to_buffer_input.size(), 1);

    // q_offsets (cumsum of sp_valid_request_counts)
    meta.q_offsets.resize(sp_size + 1);
    meta.q_offsets[0] = 0;
    for (int i = 0; i < sp_size; ++i) {
        meta.q_offsets[i + 1] = meta.q_offsets[i] + sp_valid_request_counts[i];
    }

    return meta;
}

void update_seqs_inner_loop(const std::vector<Sequence*>& sp_seqs, int sp_rank)
{
    for (auto seq : sp_seqs) {
        seq->set_num_tokens(seq->num_tokens() + 1);
        seq->block_ctx().num_dispatched_tokens[sp_rank] += 1;
    }
}

}  // namespace nanodeploy
