#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "interface_generated.h"
#include "nanosequence/csrc/sequence/sequence.h"
#include "sequence_generated.h"

#include "model_runner_utils.h"

namespace nanodeploy {

// ========================================================================
// Existing Sequence*-based helpers (unchanged)
// ========================================================================

static void build_block_tables_packed(const std::vector<Sequence*>& dp_seqs,
                                      int                           sp_rank,
                                      int                           sp_size,
                                      std::vector<int>&             block_tables_flat,
                                      int&                          max_num_blocks)
{
    std::vector<std::vector<Sequence*>> dp_sp_seqs(sp_size);
    for (auto* seq : dp_seqs) {
        int m_sp = seq->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx;
        if (m_sp >= 0 && m_sp < sp_size) {
            dp_sp_seqs[m_sp].push_back(seq);
        }
    }

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

    if (valid_tables.empty()) {
        block_tables_flat.clear();
        return;
    }

    block_tables_flat.reserve(valid_tables.size() * max_num_blocks);
    for (const auto& bt : valid_tables) {
        block_tables_flat.insert(block_tables_flat.end(), bt.begin(), bt.end());
        int padding = max_num_blocks - (int)bt.size();
        for (int k = 0; k < padding; ++k) {
            block_tables_flat.push_back(-1);
        }
    }
}

static void build_block_tables_dense(const std::vector<Sequence*>& dp_seqs,
                                     int                           sp_rank,
                                     int                           sp_size,
                                     int                           max_num_seqs,
                                     std::vector<int>&             block_tables_flat,
                                     int&                          max_num_blocks)
{
    std::vector<std::vector<Sequence*>> dp_sp_seqs(sp_size);
    for (auto* seq : dp_seqs) {
        int m_sp = seq->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx;
        if (m_sp >= 0 && m_sp < sp_size) {
            dp_sp_seqs[m_sp].push_back(seq);
        }
    }

    max_num_blocks = 0;
    for (int sp_idx = 0; sp_idx < sp_size; ++sp_idx) {
        for (auto* seq : dp_sp_seqs[sp_idx]) {
            int size = (int)seq->block_table(BlockContextSlot::ACTIVE, sp_rank).size();
            if (size > max_num_blocks)
                max_num_blocks = size;
        }
    }

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

    for (auto* seq : dp_seqs) {
        if (seq->block_ctx().master_sp_idx == sp_rank) {
            meta.input_ids.push_back(seq->last_token());
            meta.positions.push_back(seq->num_tokens() - 1);

            int page_id = seq->last_block_page_id(BlockContextSlot::ACTIVE, sp_rank);
            int offset  = seq->last_block_num_tokens(BlockContextSlot::ACTIVE, sp_rank);
            meta.slot_mapping.push_back(page_id * block_size + offset - 1);
        }
    }

    std::vector<std::vector<Sequence*>> sp_seqs(sp_size);
    for (auto* seq : dp_seqs) {
        int m_sp = seq->block_ctx().master_sp_idx;
        if (m_sp >= 0 && m_sp < sp_size) {
            sp_seqs[m_sp].push_back(seq);
        }
    }

    meta.context_lens_flat.assign(sp_size * max_num_seqs, 0);
    meta.global_context_lens_flat.assign(sp_size * max_num_seqs, 0);

    std::vector<int> sp_valid_request_counts(sp_size, 0);

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

    build_block_tables_packed(dp_seqs, sp_rank, sp_size, meta.block_tables_flat, meta.max_num_blocks);

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
    const auto& my_sp_seqs = sp_seqs[sp_rank];
    for (int seq_id = 0; seq_id < (int)my_sp_seqs.size(); ++seq_id) {
        int ctx_len = my_sp_seqs[seq_id]->context_len(BlockContextSlot::ACTIVE, sp_rank);
        if (ctx_len > 0) {
            meta.q_slice_get.push_back(seq_id);
        }
    }

    int current_pos = 0;
    for (int sp_idx = 0; sp_idx < sp_size; ++sp_idx) {
        if (sp_idx == sp_rank) {
            for (size_t k = 0; k < meta.q_slice_get.size(); ++k) {
                meta.q_slice_fill.push_back(current_pos + k);
            }
        }
        current_pos += sp_valid_request_counts[sp_idx];
    }

    meta.q_copy_mask.assign(meta.q_slice_get.size(), 1);

    meta.res_slice_get_to_buffer_output = meta.q_slice_fill;

    for (int seq_index : meta.q_slice_get) {
        meta.res_slice_fill_to_buffer_output.push_back(sp_rank * max_num_seqs + seq_index);
    }

    meta.res_to_buffer_output_mask.assign(meta.res_slice_get_to_buffer_output.size(), 1);

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

    meta.res_to_buffer_input_mask.assign(meta.res_slice_get_to_buffer_input.size(), 1);

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

// ========================================================================
// Helper: get context_len from SequenceInput (equivalent to Sequence::context_len)
// context_len(slot, sp_idx) = num_dispatched_tokens[sp_idx]
// context_len(slot, nullopt) = num_dispatched_tokens[master_sp_idx]
// ========================================================================

static inline int seq_input_context_len(const fbs::SequenceInput* si, int sp_idx)
{
    auto* ndt = si->num_dispatched_tokens();
    if (!ndt || sp_idx >= (int)ndt->size())
        return 0;
    return ndt->Get(sp_idx);
}

static inline int seq_input_num_blocks(const fbs::SequenceInput* si, int sp_idx)
{
    auto* sbt = si->sp_block_table();
    if (!sbt || sp_idx >= (int)sbt->size())
        return 0;
    auto* list = sbt->Get(sp_idx);
    if (!list || !list->values())
        return 0;
    return (int)list->values()->size();
}

static inline const flatbuffers::Vector<int32_t>* seq_input_block_table(const fbs::SequenceInput* si, int sp_idx)
{
    auto* sbt = si->sp_block_table();
    if (!sbt || sp_idx >= (int)sbt->size())
        return nullptr;
    auto* list = sbt->Get(sp_idx);
    if (!list)
        return nullptr;
    return list->values();
}

static inline int seq_input_last_block_page_id(const fbs::SequenceInput* si, int sp_idx)
{
    auto* bt = seq_input_block_table(si, sp_idx);
    if (!bt || bt->size() == 0)
        return 0;
    return bt->Get(bt->size() - 1);
}

static inline int seq_input_last_block_num_tokens(const fbs::SequenceInput* si, int sp_idx, int block_size)
{
    int ctx = seq_input_context_len(si, sp_idx);
    if (ctx == 0)
        return 0;
    int rem = ctx % block_size;
    return rem == 0 ? block_size : rem;
}

// ========================================================================
// New bytes-based API: prepare_prefill_from_bytes
// ========================================================================

// Helper for dense block tables from SequenceInput array
static void build_block_tables_dense_from_si(const flatbuffers::Vector<flatbuffers::Offset<fbs::SequenceInput>>* si_vec,
                                             int               sp_rank,
                                             int               sp_size,
                                             int               max_num_seqs,
                                             std::vector<int>& block_tables_flat,
                                             int&              max_num_blocks)
{
    // Group sequences by master_sp_idx
    std::vector<std::vector<int>> dp_sp_indices(sp_size);
    for (int i = 0; i < (int)si_vec->size(); ++i) {
        int m_sp = si_vec->Get(i)->master_sp_idx();
        if (m_sp >= 0 && m_sp < sp_size) {
            dp_sp_indices[m_sp].push_back(i);
        }
    }

    max_num_blocks = 0;
    for (int sp_idx = 0; sp_idx < sp_size; ++sp_idx) {
        for (int idx : dp_sp_indices[sp_idx]) {
            auto* bt   = seq_input_block_table(si_vec->Get(idx), sp_rank);
            int   size = bt ? (int)bt->size() : 0;
            if (size > max_num_blocks)
                max_num_blocks = size;
        }
    }

    size_t total_size = (size_t)sp_size * max_num_seqs * max_num_blocks;
    block_tables_flat.assign(total_size, -1);

    for (int sp_idx = 0; sp_idx < sp_size; ++sp_idx) {
        const auto& indices = dp_sp_indices[sp_idx];
        for (int seq_id = 0; seq_id < max_num_seqs; ++seq_id) {
            size_t base = ((size_t)sp_idx * max_num_seqs + seq_id) * max_num_blocks;
            if (seq_id < (int)indices.size()) {
                auto* bt = seq_input_block_table(si_vec->Get(indices[seq_id]), sp_rank);
                if (bt) {
                    for (size_t i = 0; i < bt->size(); ++i) {
                        block_tables_flat[base + i] = bt->Get(i);
                    }
                }
            }
        }
    }
}

// Helper for packed block tables from SequenceInput array
static void
build_block_tables_packed_from_si(const flatbuffers::Vector<flatbuffers::Offset<fbs::SequenceInput>>* si_vec,
                                  int                                                                 sp_rank,
                                  int                                                                 sp_size,
                                  std::vector<int>&                                                   block_tables_flat,
                                  int&                                                                max_num_blocks)
{
    std::vector<std::vector<int>> dp_sp_indices(sp_size);
    for (int i = 0; i < (int)si_vec->size(); ++i) {
        int m_sp = si_vec->Get(i)->master_sp_idx();
        if (m_sp >= 0 && m_sp < sp_size) {
            dp_sp_indices[m_sp].push_back(i);
        }
    }

    max_num_blocks = 0;
    std::vector<const flatbuffers::Vector<int32_t>*> valid_tables;

    for (int sp_idx = 0; sp_idx < sp_size; ++sp_idx) {
        for (int idx : dp_sp_indices[sp_idx]) {
            auto* bt = seq_input_block_table(si_vec->Get(idx), sp_rank);
            if (bt && bt->size() > 0) {
                valid_tables.push_back(bt);
                if ((int)bt->size() > max_num_blocks) {
                    max_num_blocks = (int)bt->size();
                }
            }
        }
    }

    if (valid_tables.empty()) {
        block_tables_flat.clear();
        return;
    }

    block_tables_flat.reserve(valid_tables.size() * max_num_blocks);
    for (auto* bt : valid_tables) {
        for (size_t i = 0; i < bt->size(); ++i) {
            block_tables_flat.push_back(bt->Get(i));
        }
        int padding = max_num_blocks - (int)bt->size();
        for (int k = 0; k < padding; ++k) {
            block_tables_flat.push_back(-1);
        }
    }
}

PrefillMetadata prepare_prefill_from_bytes(
    const uint8_t* data, size_t data_len, int sp_rank, int sp_size, int block_size, int max_num_seqs)
{
    flatbuffers::Verifier verifier(data, data_len);
    (void)verifier;  // available for debug verification
    auto* batch  = flatbuffers::GetRoot<fbs::RunBatchInput>(data);
    auto* si_vec = batch->sequences();

    PrefillMetadata meta;
    meta.cu_seqlens_q.push_back(0);
    meta.cu_seqlens_k.push_back(0);

    if (!si_vec || si_vec->size() == 0)
        return meta;

    size_t est_tokens = si_vec->size() * 256;
    meta.input_ids.reserve(est_tokens);
    meta.positions.reserve(est_tokens);
    meta.slot_mapping.reserve(est_tokens);

    for (size_t i = 0; i < si_vec->size(); ++i) {
        auto* si = si_vec->Get(i);
        if (si->master_sp_idx() != sp_rank)
            continue;

        int seqlen     = si->num_tokens();
        int num_cached = si->num_cached_tokens();
        int seqlen_q   = seqlen - num_cached;
        int seqlen_k   = seqlen;

        // token_ids in the serialized form are already the uncached portion
        auto* token_ids = si->token_ids();
        if (token_ids) {
            for (size_t t = 0; t < token_ids->size(); ++t) {
                meta.input_ids.push_back(token_ids->Get(t));
                meta.positions.push_back(num_cached + t);
            }
        }

        meta.cu_seqlens_q.push_back(meta.cu_seqlens_q.back() + seqlen_q);
        meta.cu_seqlens_k.push_back(meta.cu_seqlens_k.back() + seqlen_k);
        meta.max_seqlen_q = std::max(meta.max_seqlen_q, seqlen_q);
        meta.max_seqlen_k = std::max(meta.max_seqlen_k, seqlen_k);

        auto* bt = seq_input_block_table(si, sp_rank);
        if (!bt || bt->size() == 0)
            continue;

        int num_blocks        = (int)bt->size();
        int num_cached_blocks = num_cached / block_size;
        int last_block_tokens = seq_input_last_block_num_tokens(si, sp_rank, block_size);

        for (int b = num_cached_blocks; b < num_blocks; ++b) {
            int block_id = bt->Get(b);
            int start    = block_id * block_size;
            int end      = (b != num_blocks - 1) ? start + block_size : start + last_block_tokens;
            for (int k = start; k < end; ++k) {
                meta.slot_mapping.push_back(k);
            }
        }
    }

    if (meta.cu_seqlens_k.back() > meta.cu_seqlens_q.back()) {
        meta.use_block_tables = true;
        build_block_tables_dense_from_si(
            si_vec, sp_rank, sp_size, max_num_seqs, meta.block_tables_flat, meta.max_num_blocks);
    }

    return meta;
}

// ========================================================================
// New bytes-based API: prepare_decode_from_bytes
// ========================================================================

DecodeMetadata prepare_decode_from_bytes(
    const uint8_t* data, size_t data_len, int sp_rank, int sp_size, int block_size, int max_num_seqs)
{
    flatbuffers::Verifier verifier(data, data_len);
    (void)verifier;  // available for debug verification
    auto* batch  = flatbuffers::GetRoot<fbs::RunBatchInput>(data);
    auto* si_vec = batch->sequences();

    DecodeMetadata meta;
    if (!si_vec || si_vec->size() == 0)
        return meta;

    // Group by master_sp_idx
    std::vector<std::vector<int>> sp_indices(sp_size);
    for (int i = 0; i < (int)si_vec->size(); ++i) {
        int m_sp = si_vec->Get(i)->master_sp_idx();
        if (m_sp >= 0 && m_sp < sp_size) {
            sp_indices[m_sp].push_back(i);
        }
    }

    // 1. input_ids, positions, slot_mapping
    for (int idx : sp_indices[sp_rank]) {
        auto* si = si_vec->Get(idx);
        meta.input_ids.push_back(si->last_token());
        meta.positions.push_back(si->num_tokens() - 1);

        int page_id = seq_input_last_block_page_id(si, sp_rank);
        int offset  = seq_input_last_block_num_tokens(si, sp_rank, block_size);
        meta.slot_mapping.push_back(page_id * block_size + offset - 1);
    }

    // 2. context_lens, global_context_lens
    meta.context_lens_flat.assign(sp_size * max_num_seqs, 0);
    meta.global_context_lens_flat.assign(sp_size * max_num_seqs, 0);

    std::vector<int> sp_valid_request_counts(sp_size, 0);

    for (int sp_idx = 0; sp_idx < sp_size; ++sp_idx) {
        const auto& indices     = sp_indices[sp_idx];
        int         valid_count = 0;
        for (int seq_id = 0; seq_id < max_num_seqs; ++seq_id) {
            if (seq_id < (int)indices.size()) {
                auto* si                                               = si_vec->Get(indices[seq_id]);
                int   ctx_len                                          = seq_input_context_len(si, sp_rank);
                meta.context_lens_flat[sp_idx * max_num_seqs + seq_id] = ctx_len;
                if (ctx_len > 0)
                    valid_count++;
            }
        }
        sp_valid_request_counts[sp_idx] = valid_count;
    }

    // global_context_lens
    const auto& my_master_indices = sp_indices[sp_rank];
    for (int sp_idx = 0; sp_idx < sp_size; ++sp_idx) {
        for (int seq_id = 0; seq_id < max_num_seqs; ++seq_id) {
            if (seq_id < (int)my_master_indices.size()) {
                auto* si                                                      = si_vec->Get(my_master_indices[seq_id]);
                meta.global_context_lens_flat[sp_idx * max_num_seqs + seq_id] = seq_input_context_len(si, sp_idx);
            }
        }
    }

    // 3. Block tables
    build_block_tables_packed_from_si(si_vec, sp_rank, sp_size, meta.block_tables_flat, meta.max_num_blocks);

    // 4. context_lens_for_attn
    for (int sp_idx = 0; sp_idx < sp_size; ++sp_idx) {
        const auto& indices = sp_indices[sp_idx];
        for (int seq_id = 0; seq_id < (int)indices.size(); ++seq_id) {
            int ctx_len = seq_input_context_len(si_vec->Get(indices[seq_id]), sp_rank);
            if (ctx_len > 0) {
                meta.context_lens_for_attn.push_back(ctx_len);
            }
        }
    }

    // --- Migration Logic ---
    const auto& my_indices = sp_indices[sp_rank];
    for (int seq_id = 0; seq_id < (int)my_indices.size(); ++seq_id) {
        int ctx_len = seq_input_context_len(si_vec->Get(my_indices[seq_id]), sp_rank);
        if (ctx_len > 0) {
            meta.q_slice_get.push_back(seq_id);
        }
    }

    int current_pos = 0;
    for (int sp_idx = 0; sp_idx < sp_size; ++sp_idx) {
        if (sp_idx == sp_rank) {
            for (size_t k = 0; k < meta.q_slice_get.size(); ++k) {
                meta.q_slice_fill.push_back(current_pos + k);
            }
        }
        current_pos += sp_valid_request_counts[sp_idx];
    }

    meta.q_copy_mask.assign(meta.q_slice_get.size(), 1);
    meta.res_slice_get_to_buffer_output = meta.q_slice_fill;

    for (int seq_index : meta.q_slice_get) {
        meta.res_slice_fill_to_buffer_output.push_back(sp_rank * max_num_seqs + seq_index);
    }

    meta.res_to_buffer_output_mask.assign(meta.res_slice_get_to_buffer_output.size(), 1);

    int current_attention_pos = 0;
    for (int sp_idx = 0; sp_idx < sp_size; ++sp_idx) {
        if (sp_idx == sp_rank) {
            current_attention_pos += sp_valid_request_counts[sp_idx];
            continue;
        }

        const auto& indices = sp_indices[sp_idx];
        for (int seq_id = 0; seq_id < (int)indices.size(); ++seq_id) {
            int ctx_len = seq_input_context_len(si_vec->Get(indices[seq_id]), sp_rank);
            if (ctx_len > 0) {
                meta.res_slice_get_to_buffer_input.push_back(current_attention_pos);
                meta.res_slice_fill_to_buffer_input.push_back(sp_idx * max_num_seqs + seq_id);
                current_attention_pos++;
            }
        }
    }

    meta.res_to_buffer_input_mask.assign(meta.res_slice_get_to_buffer_input.size(), 1);

    meta.q_offsets.resize(sp_size + 1);
    meta.q_offsets[0] = 0;
    for (int i = 0; i < sp_size; ++i) {
        meta.q_offsets[i + 1] = meta.q_offsets[i] + sp_valid_request_counts[i];
    }

    return meta;
}

// ========================================================================
// Extract auxiliary data from RunBatchInput bytes
// ========================================================================

BatchAuxData extract_aux_from_bytes(const uint8_t* data, size_t data_len, int sp_rank)
{
    flatbuffers::Verifier verifier(data, data_len);
    (void)verifier;  // available for debug verification
    auto* batch  = flatbuffers::GetRoot<fbs::RunBatchInput>(data);
    auto* si_vec = batch->sequences();

    BatchAuxData aux;
    if (!si_vec)
        return aux;

    aux.master_sp_indices.reserve(si_vec->size());
    int sp_count = 0;
    for (size_t i = 0; i < si_vec->size(); ++i) {
        auto* si   = si_vec->Get(i);
        int   m_sp = si->master_sp_idx();
        aux.master_sp_indices.push_back(m_sp);

        if (m_sp == sp_rank) {
            aux.temperatures.push_back(si->temperature());
            aux.state_slots.push_back(si->state_slot());
            sp_count++;
        }
    }
    aux.num_sp_seqs = sp_count;
    return aux;
}

// ========================================================================
// Parse MigrateBatchInput bytes
// ========================================================================

std::vector<MigrateSequenceView> parse_migrate_batch(const uint8_t* data, size_t data_len)
{
    flatbuffers::Verifier verifier(data, data_len);
    (void)verifier;  // available for debug verification
    auto* batch   = flatbuffers::GetRoot<fbs::MigrateBatchInput>(data);
    auto* msi_vec = batch->sequences();

    std::vector<MigrateSequenceView> views;
    if (!msi_vec)
        return views;

    views.reserve(msi_vec->size());
    for (size_t i = 0; i < msi_vec->size(); ++i) {
        auto*               msi = msi_vec->Get(i);
        MigrateSequenceView v;
        v.seq_id                     = msi->seq_id();
        v.migrate_engine_id          = msi->migrate_engine_id() ? msi->migrate_engine_id()->str() : "";
        v.migrate_num_kvcache_blocks = msi->migrate_num_kvcache_blocks();
        v.migrate_attention_sp       = msi->migrate_attention_sp();
        v.migrate_dp_idx             = msi->migrate_dp_idx();
        v.migrate_state_slot         = msi->migrate_state_slot();
        v.active_state_slot          = msi->active_state_slot();

        if (msi->migrate_block_location()) {
            for (size_t j = 0; j < msi->migrate_block_location()->size(); ++j) {
                auto* bl = msi->migrate_block_location()->Get(j);
                v.migrate_block_location.emplace_back(bl->sp_idx(), bl->block_idx());
            }
        }

        if (msi->active_block_location()) {
            for (size_t j = 0; j < msi->active_block_location()->size(); ++j) {
                auto* bl = msi->active_block_location()->Get(j);
                v.active_block_location.emplace_back(bl->sp_idx(), bl->block_idx());
            }
        }

        views.push_back(std::move(v));
    }
    return views;
}

}  // namespace nanodeploy
