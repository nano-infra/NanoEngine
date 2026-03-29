#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "nanodeploy/csrc/sequence/sequence.h"

#include "interface_generated.h"
#include "sequence_generated.h"

#include "model_runner_utils.h"

namespace nanodeploy {

// ========================================================================
// Helper: get context_len from SequenceInput (equivalent to Sequence::context_len)
// context_len(slot, group_id) = num_dispatched_tokens[group_id]
// context_len(slot, nullopt) = num_dispatched_tokens[master_group_id]
// ========================================================================

static inline int seq_input_context_len(const fbs::SequenceInput* si, int group_id)
{
    auto* ndt = si->num_dispatched_tokens();
    if (!ndt || group_id >= (int)ndt->size())
        return 0;
    return ndt->Get(group_id);
}

static inline int seq_input_num_blocks(const fbs::SequenceInput* si, int group_id)
{
    auto* sbt = si->group_block_table();
    if (!sbt || group_id >= (int)sbt->size())
        return 0;
    auto* list = sbt->Get(group_id);
    if (!list || !list->values())
        return 0;
    return (int)list->values()->size();
}

static inline const flatbuffers::Vector<int32_t>* seq_input_block_table(const fbs::SequenceInput* si, int group_id)
{
    auto* sbt = si->group_block_table();
    if (!sbt || group_id >= (int)sbt->size())
        return nullptr;
    auto* list = sbt->Get(group_id);
    if (!list)
        return nullptr;
    return list->values();
}

static inline int seq_input_last_block_page_id(const fbs::SequenceInput* si, int group_id)
{
    auto* bt = seq_input_block_table(si, group_id);
    if (!bt || bt->size() == 0)
        return 0;
    return bt->Get(bt->size() - 1);
}

static inline int seq_input_last_block_num_tokens(const fbs::SequenceInput* si, int group_id, int block_size)
{
    int ctx = seq_input_context_len(si, group_id);
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
                                             int               group_rank,
                                             int               group_size,
                                             int               max_num_seqs,
                                             std::vector<int>& block_tables_flat,
                                             int&              max_num_blocks)
{
    // Group sequences by master_group_id
    std::vector<std::vector<int>> dp_group_indices(group_size);
    for (int i = 0; i < (int)si_vec->size(); ++i) {
        int m_group = si_vec->Get(i)->master_group_id();
        if (m_group >= 0 && m_group < group_size) {
            dp_group_indices[m_group].push_back(i);
        }
    }

    max_num_blocks = 0;
    for (int group_id = 0; group_id < group_size; ++group_id) {
        for (int idx : dp_group_indices[group_id]) {
            auto* bt   = seq_input_block_table(si_vec->Get(idx), group_rank);
            int   size = bt ? (int)bt->size() : 0;
            if (size > max_num_blocks)
                max_num_blocks = size;
        }
    }

    size_t total_size = (size_t)group_size * max_num_seqs * max_num_blocks;
    block_tables_flat.assign(total_size, -1);

    for (int group_id = 0; group_id < group_size; ++group_id) {
        const auto& indices = dp_group_indices[group_id];
        for (int seq_id = 0; seq_id < max_num_seqs; ++seq_id) {
            size_t base = ((size_t)group_id * max_num_seqs + seq_id) * max_num_blocks;
            if (seq_id < (int)indices.size()) {
                auto* bt = seq_input_block_table(si_vec->Get(indices[seq_id]), group_rank);
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
                                  int                                                                 group_rank,
                                  int                                                                 group_size,
                                  std::vector<int>&                                                   block_tables_flat,
                                  int&                                                                max_num_blocks)
{
    std::vector<std::vector<int>> dp_group_indices(group_size);
    for (int i = 0; i < (int)si_vec->size(); ++i) {
        int m_group = si_vec->Get(i)->master_group_id();
        if (m_group >= 0 && m_group < group_size) {
            dp_group_indices[m_group].push_back(i);
        }
    }

    max_num_blocks = 0;
    std::vector<const flatbuffers::Vector<int32_t>*> valid_tables;

    for (int group_id = 0; group_id < group_size; ++group_id) {
        for (int idx : dp_group_indices[group_id]) {
            auto* bt = seq_input_block_table(si_vec->Get(idx), group_rank);
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

PrefillMetadata prepare_prefill_from_bytes(const uint8_t* data,
                                           size_t         data_len,
                                           int            group_rank,
                                           int            group_size,
                                           int            block_size,
                                           int            max_num_seqs,
                                           int            num_gpu_blocks)
{
    flatbuffers::Verifier verifier(data, data_len);
    if (!verifier.VerifyBuffer<fbs::RunBatchInput>(nullptr)) {
        throw std::runtime_error("prepare_prefill_from_bytes: invalid FlatBuffers buffer");
    }
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
        if (si->master_group_id() != group_rank)
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

        auto* bt = seq_input_block_table(si, group_rank);
        if (!bt || bt->size() == 0)
            continue;

        int num_blocks             = (int)bt->size();
        int num_cached_blocks      = num_cached / block_size;
        int cached_offset_in_block = num_cached % block_size;
        int last_block_tokens      = seq_input_last_block_num_tokens(si, group_rank, block_size);

        for (int b = num_cached_blocks; b < num_blocks; ++b) {
            int block_id = bt->Get(b);
            if (block_id < 0 || block_id >= num_gpu_blocks) {
                throw std::runtime_error("prepare_prefill_from_bytes: block_id " + std::to_string(block_id)
                                         + " out of range [0, " + std::to_string(num_gpu_blocks) + ")");
            }
            int64_t start = (int64_t)block_id * block_size;
            int64_t end   = (b != num_blocks - 1) ? start + block_size : start + last_block_tokens;
            if (b == num_cached_blocks) {
                start += cached_offset_in_block;
            }
            for (int64_t k = start; k < end; ++k) {
                meta.slot_mapping.push_back((int)k);
            }
        }
    }

    if (meta.cu_seqlens_k.back() > meta.cu_seqlens_q.back()) {
        meta.use_block_tables = true;
        build_block_tables_dense_from_si(
            si_vec, group_rank, group_size, max_num_seqs, meta.block_tables_flat, meta.max_num_blocks);
    }

    // Populate sampling indices: for each master-sp sequence that has reached
    // the end of its prompt (num_tokens == num_prompt_tokens), record the index
    // of its last Q token so the Python side can run lm_head only on those.
    {
        int seq_pos = 0;  // index among master-sp sequences processed above
        int q_start = 0;  // cumulative Q token offset before this seq
        for (size_t i = 0; i < si_vec->size(); ++i) {
            auto* si = si_vec->Get(i);
            if (si->master_group_id() != group_rank)
                continue;

            int seqlen_q = si->num_tokens() - si->num_cached_tokens();
            int q_end    = q_start + seqlen_q;

            int num_prompt = si->num_prompt_tokens();
            if (num_prompt == 0) {
                throw std::runtime_error("prepare_prefill_from_bytes: num_prompt_tokens is 0 for sequence "
                                         + std::to_string(i)
                                         + ". This field is required and must be set in SequenceInput.");
            }
            bool is_final = (si->num_tokens() >= num_prompt);
            if (is_final) {
                meta.sampling_token_indices.push_back(q_end - 1);
                meta.sampling_seq_indices.push_back(seq_pos);
            }

            q_start = q_end;
            seq_pos++;
        }
    }

    return meta;
}

// ========================================================================
// New bytes-based API: prepare_decode_from_bytes
// ========================================================================

DecodeMetadata prepare_decode_from_bytes(const uint8_t* data,
                                         size_t         data_len,
                                         int            group_rank,
                                         int            group_size,
                                         int            block_size,
                                         int            max_num_seqs,
                                         int            num_gpu_blocks)
{
    flatbuffers::Verifier verifier(data, data_len);
    if (!verifier.VerifyBuffer<fbs::RunBatchInput>(nullptr)) {
        throw std::runtime_error("prepare_decode_from_bytes: invalid FlatBuffers buffer");
    }
    auto* batch  = flatbuffers::GetRoot<fbs::RunBatchInput>(data);
    auto* si_vec = batch->sequences();

    DecodeMetadata meta;
    if (!si_vec || si_vec->size() == 0)
        return meta;

    // Group by master_group_id
    std::vector<std::vector<int>> group_indices(group_size);
    for (int i = 0; i < (int)si_vec->size(); ++i) {
        int m_group = si_vec->Get(i)->master_group_id();
        if (m_group >= 0 && m_group < group_size) {
            group_indices[m_group].push_back(i);
        }
    }

    // 1. input_ids, positions, slot_mapping
    for (int idx : group_indices[group_rank]) {
        auto* si = si_vec->Get(idx);
        meta.input_ids.push_back(si->last_token());
        meta.positions.push_back(si->num_tokens() - 1);

        int page_id = seq_input_last_block_page_id(si, group_rank);
        int offset  = seq_input_last_block_num_tokens(si, group_rank, block_size);
        if (page_id < 0 || page_id >= num_gpu_blocks) {
            throw std::runtime_error("prepare_decode_from_bytes: page_id " + std::to_string(page_id)
                                     + " out of range [0, " + std::to_string(num_gpu_blocks) + ")");
        }
        meta.slot_mapping.push_back((int)((int64_t)page_id * block_size + offset - 1));
    }

    // 2. context_lens, global_context_lens
    meta.context_lens_flat.assign(group_size * max_num_seqs, 0);
    meta.global_context_lens_flat.assign(group_size * max_num_seqs, 0);

    std::vector<int> group_valid_request_counts(group_size, 0);

    for (int group_id = 0; group_id < group_size; ++group_id) {
        const auto& indices     = group_indices[group_id];
        int         valid_count = 0;
        for (int seq_id = 0; seq_id < max_num_seqs; ++seq_id) {
            if (seq_id < (int)indices.size()) {
                auto* si                                                 = si_vec->Get(indices[seq_id]);
                int   ctx_len                                            = seq_input_context_len(si, group_rank);
                meta.context_lens_flat[group_id * max_num_seqs + seq_id] = ctx_len;
                if (ctx_len > 0)
                    valid_count++;
            }
        }
        group_valid_request_counts[group_id] = valid_count;
    }

    // global_context_lens
    const auto& my_master_indices = group_indices[group_rank];
    for (int group_id = 0; group_id < group_size; ++group_id) {
        for (int seq_id = 0; seq_id < max_num_seqs; ++seq_id) {
            if (seq_id < (int)my_master_indices.size()) {
                auto* si = si_vec->Get(my_master_indices[seq_id]);
                meta.global_context_lens_flat[group_id * max_num_seqs + seq_id] = seq_input_context_len(si, group_id);
            }
        }
    }

    // 3. Block tables
    build_block_tables_packed_from_si(si_vec, group_rank, group_size, meta.block_tables_flat, meta.max_num_blocks);

    // 4. context_lens_for_attn
    for (int group_id = 0; group_id < group_size; ++group_id) {
        const auto& indices = group_indices[group_id];
        for (int seq_id = 0; seq_id < (int)indices.size(); ++seq_id) {
            int ctx_len = seq_input_context_len(si_vec->Get(indices[seq_id]), group_rank);
            if (ctx_len > 0) {
                meta.context_lens_for_attn.push_back(ctx_len);
            }
        }
    }

    // --- Migration Logic ---
    const auto& my_indices = group_indices[group_rank];
    for (int seq_id = 0; seq_id < (int)my_indices.size(); ++seq_id) {
        int ctx_len = seq_input_context_len(si_vec->Get(my_indices[seq_id]), group_rank);
        if (ctx_len > 0) {
            meta.q_slice_get.push_back(seq_id);
        }
    }

    int current_pos = 0;
    for (int group_id = 0; group_id < group_size; ++group_id) {
        if (group_id == group_rank) {
            for (size_t k = 0; k < meta.q_slice_get.size(); ++k) {
                meta.q_slice_fill.push_back(current_pos + k);
            }
        }
        current_pos += group_valid_request_counts[group_id];
    }

    meta.q_copy_mask.assign(meta.q_slice_get.size(), 1);
    meta.res_slice_get_to_buffer_output = meta.q_slice_fill;

    for (int seq_index : meta.q_slice_get) {
        meta.res_slice_fill_to_buffer_output.push_back(group_rank * max_num_seqs + seq_index);
    }

    meta.res_to_buffer_output_mask.assign(meta.res_slice_get_to_buffer_output.size(), 1);

    int current_attention_pos = 0;
    for (int group_id = 0; group_id < group_size; ++group_id) {
        if (group_id == group_rank) {
            current_attention_pos += group_valid_request_counts[group_id];
            continue;
        }

        const auto& indices = group_indices[group_id];
        for (int seq_id = 0; seq_id < (int)indices.size(); ++seq_id) {
            int ctx_len = seq_input_context_len(si_vec->Get(indices[seq_id]), group_rank);
            if (ctx_len > 0) {
                meta.res_slice_get_to_buffer_input.push_back(current_attention_pos);
                meta.res_slice_fill_to_buffer_input.push_back(group_id * max_num_seqs + seq_id);
                current_attention_pos++;
            }
        }
    }

    meta.res_to_buffer_input_mask.assign(meta.res_slice_get_to_buffer_input.size(), 1);

    meta.q_offsets.resize(group_size + 1);
    meta.q_offsets[0] = 0;
    for (int i = 0; i < group_size; ++i) {
        meta.q_offsets[i + 1] = meta.q_offsets[i] + group_valid_request_counts[i];
    }

    return meta;
}

// ========================================================================
// Extract vision slot refs from RunBatchInput bytes
// ========================================================================

std::vector<VisionSlotView> extract_vision_slots_from_bytes(const uint8_t* data, size_t data_len)
{
    flatbuffers::Verifier verifier(data, data_len);
    if (!verifier.VerifyBuffer<fbs::RunBatchInput>(nullptr)) {
        throw std::runtime_error("extract_vision_slots_from_bytes: invalid FlatBuffers buffer");
    }
    auto* batch  = flatbuffers::GetRoot<fbs::RunBatchInput>(data);
    auto* si_vec = batch->sequences();

    std::vector<VisionSlotView> views;
    if (!si_vec)
        return views;

    for (size_t i = 0; i < si_vec->size(); ++i) {
        auto* si = si_vec->Get(i);
        auto* vs = si->vision_slots();
        if (!vs)
            continue;
        for (size_t j = 0; j < vs->size(); ++j) {
            auto*          ref = vs->Get(j);
            VisionSlotView v;
            v.encoder_engine_id   = ref->encoder_engine_id() ? ref->encoder_engine_id()->str() : "";
            v.slot_idx            = ref->slot_idx();
            v.num_tokens          = ref->num_tokens();
            v.hidden_size         = ref->hidden_size();
            v.max_tokens_per_slot = ref->max_tokens_per_slot();
            v.seq_index           = static_cast<int>(i);
            views.push_back(std::move(v));
        }
    }
    return views;
}

// ========================================================================
// Extract auxiliary data from RunBatchInput bytes
// ========================================================================

BatchAuxData extract_aux_from_bytes(const uint8_t* data, size_t data_len, int group_rank)
{
    flatbuffers::Verifier verifier(data, data_len);
    if (!verifier.VerifyBuffer<fbs::RunBatchInput>(nullptr)) {
        throw std::runtime_error("extract_aux_from_bytes: invalid FlatBuffers buffer");
    }
    auto* batch  = flatbuffers::GetRoot<fbs::RunBatchInput>(data);
    auto* si_vec = batch->sequences();

    BatchAuxData aux;
    if (!si_vec)
        return aux;

    aux.master_group_indices.reserve(si_vec->size());
    int group_count = 0;
    for (size_t i = 0; i < si_vec->size(); ++i) {
        auto* si      = si_vec->Get(i);
        int   m_group = si->master_group_id();
        aux.master_group_indices.push_back(m_group);

        if (m_group == group_rank) {
            aux.temperatures.push_back(si->temperature());
            aux.state_slots.push_back(si->state_slot());
            group_count++;
        }
    }
    aux.num_group_seqs = group_count;
    return aux;
}

// ========================================================================
// Parse MigrateBatchInput bytes
// ========================================================================

std::vector<MigrateSequenceView> parse_migrate_batch(const uint8_t* data, size_t data_len)
{
    flatbuffers::Verifier verifier(data, data_len);
    if (!verifier.VerifyBuffer<fbs::MigrateBatchInput>(nullptr)) {
        throw std::runtime_error("parse_migrate_batch: invalid FlatBuffers buffer");
    }
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
        v.migrate_group_size         = msi->migrate_group_size();
        v.migrate_dp_idx             = msi->migrate_dp_idx();
        v.migrate_state_slot         = msi->migrate_state_slot();
        v.active_state_slot          = msi->active_state_slot();

        if (msi->migrate_block_location()) {
            for (size_t j = 0; j < msi->migrate_block_location()->size(); ++j) {
                auto* bl = msi->migrate_block_location()->Get(j);
                v.migrate_block_location.emplace_back(bl->group_id(), bl->block_idx());
            }
        }

        if (msi->active_block_location()) {
            for (size_t j = 0; j < msi->active_block_location()->size(); ++j) {
                auto* bl = msi->active_block_location()->Get(j);
                v.active_block_location.emplace_back(bl->group_id(), bl->block_idx());
            }
        }

        views.push_back(std::move(v));
    }
    return views;
}

}  // namespace nanodeploy
