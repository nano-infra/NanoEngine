#include "model_runner_utils.h"
#include "nanodeploy/engine/sequence.h"
#include <algorithm>
#include <iostream>

namespace nanodeploy {

// Helper to mimic Python's prepare_block_tables
// Helper to mimic Python's updated prepare_block_tables logic (Packed format)
// Helper to mimic Python's updated prepare_block_tables logic (Packed format)
static void build_block_tables_packed(
    const std::vector<Sequence*>& dp_seqs,
    int sp_rank,
    int sp_size,
    std::vector<int>& block_tables_flat,
    int& max_num_blocks
) {
    // 1. Group sequences by master_sp_idx
    std::vector<std::vector<Sequence*>> dp_sp_seqs(sp_size);
    for (auto* seq : dp_seqs) {
        int m_sp = seq->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_;
        if (m_sp >= 0 && m_sp < sp_size) {
            dp_sp_seqs[m_sp].push_back(seq);
        }
    }

    // 2. Collect valid block tables and calculate max_num_blocks
    struct ValidBlockTable {
        const std::vector<int>* bt_ptr;
    };
    std::vector<ValidBlockTable> valid_tables;

    max_num_blocks = 0;
    
    // Iterate in sp_idx order as per Python implementation
    for (int sp_idx = 0; sp_idx < sp_size; ++sp_idx) {
        for (auto* seq : dp_sp_seqs[sp_idx]) {
            const auto& bt = seq->block_table(BlockContextSlot::ACTIVE, sp_rank);
            if (!bt.empty()) {
                valid_tables.push_back({&bt});
                if ((int)bt.size() > max_num_blocks) {
                    max_num_blocks = (int)bt.size();
                }
            }
        }
    }

    // 3. Fill flattened table: [num_valid_seqs, max_num_blocks]
    if (valid_tables.empty()) {
        block_tables_flat.clear();
        max_num_blocks = 0;
        return;
    }

    size_t total_size = valid_tables.size() * max_num_blocks;
    block_tables_flat.assign(total_size, -1);

    for (size_t i = 0; i < valid_tables.size(); ++i) {
        const auto& bt = *valid_tables[i].bt_ptr;
        size_t base_offset = i * max_num_blocks;
        for (size_t k = 0; k < bt.size(); ++k) {
            block_tables_flat[base_offset + k] = bt[k];
        }
    }
}

static void build_block_tables(
    const std::vector<Sequence*>& dp_seqs,
    int sp_rank,
    int sp_size,
    int max_num_seqs,
    std::vector<int>& block_tables_flat,
    int& max_num_blocks
) {
     // 1. Group sequences by master_sp_idx
    std::vector<std::vector<Sequence*>> dp_sp_seqs(sp_size);
    for (auto* seq : dp_seqs) {
        int m_sp = seq->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_;
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
        const auto& seqs           = dp_sp_seqs[sp_idx];
        size_t      num_seqs_in_sp = seqs.size();

        for (int seq_id = 0; seq_id < max_num_seqs; ++seq_id) {
            size_t base_offset = ((size_t)sp_idx * max_num_seqs + seq_id) * max_num_blocks;

            if (seq_id < (int)num_seqs_in_sp) {
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
        // Filter by master_sp_rank
        if (seq->block_ctx().master_sp_idx_ != sp_rank) {
            continue;
        }

        int seqlen     = seq->num_tokens;
        int num_cached = seq->num_cached_tokens;
        int seqlen_q   = seqlen - num_cached;
        int seqlen_k   = seqlen;

        const auto& full_tokens = seq->token_ids;
        for (int i = num_cached; i < seqlen; ++i) {
            meta.input_ids.push_back(full_tokens[i]);
            meta.positions.push_back(i);
        }

        meta.cu_seqlens_q.push_back(meta.cu_seqlens_q.back() + seqlen_q);
        meta.cu_seqlens_k.push_back(meta.cu_seqlens_k.back() + seqlen_k);
        meta.max_seqlen_q = std::max(meta.max_seqlen_q, seqlen_q);
        meta.max_seqlen_k = std::max(meta.max_seqlen_k, seqlen_k);

        // [FIXED] Check for empty block table (Warmup case) AFTER processing tokens.
        // If empty, we just skip slot mapping calculation.
        const auto& bt = seq->block_table(BlockContextSlot::ACTIVE, sp_rank);
        if (bt.empty()) {
            continue;
        }

        // Calculate slot mapping
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
        build_block_tables(seqs, sp_rank, sp_size, max_num_seqs, meta.block_tables_flat, meta.max_num_blocks);
    }

    return meta;
}

DecodeMetadata
prepare_decode_cpp(const std::vector<Sequence*>& dp_seqs, int sp_rank, int sp_size, int block_size, int max_num_seqs)
{
    DecodeMetadata meta;

    // 1. Prepare input_ids, positions, slot_mapping
    for (auto* seq : dp_seqs) {
        if (seq->block_ctx().master_sp_idx_ == sp_rank) {
            meta.input_ids.push_back(seq->last_token);
            meta.positions.push_back(seq->num_tokens - 1);

            int page_id = seq->last_block_page_id(BlockContextSlot::ACTIVE, sp_rank);
            int offset  = seq->last_block_num_tokens(BlockContextSlot::ACTIVE, sp_rank);
            meta.slot_mapping.push_back(page_id * block_size + offset - 1);
        }
    }

    // 3. Prepare context_lens and global_context_lens
    meta.context_lens_flat.assign(sp_size * max_num_seqs, 0);
    meta.global_context_lens_flat.assign(sp_size * max_num_seqs, 0);

    // Group sequences
    std::vector<std::vector<Sequence*>> sp_seqs(sp_size);
    for (auto* seq : dp_seqs) {
        int m_sp = seq->block_ctx().master_sp_idx_;
        if (m_sp >= 0 && m_sp < sp_size) {
            sp_seqs[m_sp].push_back(seq);
        }
    }

    // Fill context_lens
    for (int sp_idx = 0; sp_idx < sp_size; ++sp_idx) {
        const auto& batch_seqs = sp_seqs[sp_idx];
        for (int seq_id = 0; seq_id < max_num_seqs; ++seq_id) {
            if (seq_id < (int)batch_seqs.size()) {
                Sequence* seq = batch_seqs[seq_id];
                meta.context_lens_flat[sp_idx * max_num_seqs + seq_id] =
                    seq->context_len(BlockContextSlot::ACTIVE, sp_rank);
            }
        }
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
    // 3. Block tables
    build_block_tables_packed(dp_seqs, sp_rank, sp_size, meta.block_tables_flat, meta.max_num_blocks);

    meta.attention_compute_bs = (int)meta.input_ids.size();
    meta.q_output_stride.assign(sp_size, 0);

    return meta;
}

void update_seqs_inner_loop(const std::vector<Sequence*>& sp_seqs, int sp_rank)
{
    for (auto seq : sp_seqs) {
        seq->num_tokens += 1;
        seq->block_ctx().num_dispatched_tokens[sp_rank] += 1;
    }
}

}  // namespace nanodeploy
