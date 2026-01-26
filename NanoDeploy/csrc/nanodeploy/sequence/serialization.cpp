#include "nanocommon/logging.h"

#include "sequence_generated.h"
#include "serialization.h"
#include <flatbuffers/flatbuffers.h>

namespace nanodeploy {

namespace {

flatbuffers::Offset<nanodeploy::fbs::BlockContext> pack_block_context(flatbuffers::FlatBufferBuilder& builder,
                                                                      const BlockContext&             ctx)
{
    auto engine_id_off = builder.CreateString(ctx.engine_id_);

    std::vector<nanodeploy::fbs::BlockLocation> locs;
    auto                                        loc_vec_off = builder.CreateVectorOfStructs(locs);

    auto disp_vec_off = builder.CreateVector(ctx.num_dispatched_tokens);

    // Vector<IntList> (nested vectors for sp_block_table)
    std::vector<flatbuffers::Offset<nanodeploy::fbs::IntList>> int_list_offs;
    int_list_offs.reserve(ctx.sp_block_table.size());
    for (const auto& inner : ctx.sp_block_table) {
        auto inner_vec = builder.CreateVector(inner);
        int_list_offs.push_back(nanodeploy::fbs::CreateIntList(builder, inner_vec));
    }
    auto sp_block_table_off = builder.CreateVector(int_list_offs);

    return nanodeploy::fbs::CreateBlockContext(builder,
                                               engine_id_off,
                                               ctx.dp_idx_,
                                               ctx.master_sp_idx_,
                                               ctx.attention_sp_,
                                               ctx.attention_dp_,
                                               loc_vec_off,
                                               disp_vec_off,
                                               sp_block_table_off);
}

void unpack_block_context(const nanodeploy::fbs::BlockContext* fb_ctx, BlockContext& ctx)
{
    if (!fb_ctx)
        return;

    if (fb_ctx->engine_id()) {
        ctx.engine_id_ = fb_ctx->engine_id()->str();
    }
    ctx.dp_idx_        = fb_ctx->dp_idx();
    ctx.master_sp_idx_ = fb_ctx->master_sp_idx();
    ctx.attention_sp_  = fb_ctx->attention_sp();
    ctx.attention_dp_  = fb_ctx->attention_dp();

    // block_location
    if (auto locs = fb_ctx->block_location()) {
        ctx.block_location.clear();
        ctx.block_location.reserve(locs->size());
        for (const auto* loc : *locs) {
            ctx.block_location.emplace_back(loc->first(), loc->second());
        }
    }

    // num_dispatched_tokens
    if (auto disps = fb_ctx->num_dispatched_tokens()) {
        ctx.num_dispatched_tokens.assign(disps->begin(), disps->end());
    }

    // sp_block_table
    if (auto table = fb_ctx->sp_block_table()) {
        ctx.sp_block_table.clear();
        ctx.sp_block_table.resize(table->size());
        for (size_t i = 0; i < table->size(); ++i) {
            const auto* int_list = table->Get(i);
            if (int_list && int_list->values()) {
                ctx.sp_block_table[i].assign(int_list->values()->begin(), int_list->values()->end());
            }
        }
    }
}

}  // namespace

size_t serialize_sequences(uintptr_t                                     data_ptr,
                           size_t                                        buffer_size,
                           const std::vector<std::shared_ptr<Sequence>>& seqs,
                           bool                                          is_prefill)
{
    flatbuffers::FlatBufferBuilder builder(buffer_size);
    // NANOCOMMON_LOG_INFO("serializing seqs, is_prefill=", is_prefill);

    std::vector<flatbuffers::Offset<nanodeploy::fbs::Sequence>> seq_offsets;
    seq_offsets.reserve(seqs.size());

    size_t total_token_bytes       = 0;
    size_t total_block_loc_bytes   = 0;
    size_t total_block_table_bytes = 0;
    size_t total_engine_id_bytes   = 0;
    size_t total_disp_token_bytes  = 0;
    size_t total_inner_lists       = 0;

    for (const auto& seq_ptr : seqs) {
        if (!seq_ptr)
            continue;
        const auto& seq = *seq_ptr;

        flatbuffers::Offset<flatbuffers::Vector<int>> token_ids_off;
        // Always serialize token_ids to ensure downstream consumers (Engine/Workers) have full context
        // This fixes ZeroDivisionError in quantization kernels which likely derived shapes from token count
        token_ids_off = builder.CreateVector(seq.token_ids);
        total_token_bytes += seq.token_ids.size() * sizeof(int);

        const auto& ctx = seq.slots_[(size_t)BlockContextSlot::ACTIVE];

        total_disp_token_bytes += ctx.num_dispatched_tokens.size() * sizeof(int);
        total_inner_lists += ctx.sp_block_table.size();

        for (const auto& inner : ctx.sp_block_table) {
            total_block_table_bytes += inner.size() * sizeof(int);
        }

        std::vector<flatbuffers::Offset<nanodeploy::fbs::BlockContext>> slot_offsets;
        slot_offsets.push_back(pack_block_context(builder, seq.slots_[(size_t)BlockContextSlot::ACTIVE]));
        auto slots_vec_off = builder.CreateVector(slot_offsets);

        auto sampling_params_off = nanodeploy::fbs::CreateSamplingParams(
            builder, seq.sampling_params.temperature, seq.sampling_params.max_tokens, seq.sampling_params.ignore_eos);

        auto seq_off = nanodeploy::fbs::CreateSequence(builder,
                                                       seq.seq_id,
                                                       static_cast<nanodeploy::fbs::SequenceStatus>(seq.status),
                                                       sampling_params_off,
                                                       seq.last_token,
                                                       seq.num_tokens,
                                                       seq.num_prompt_tokens,
                                                       seq.num_checkpointed_tokens,
                                                       seq.num_cached_tokens,
                                                       token_ids_off,
                                                       slots_vec_off);
        seq_offsets.push_back(seq_off);
    }

    // NANOCOMMON_LOG_INFO("Serialization Breakdown: Tokens=" + std::to_string(total_token_bytes)
    //                     + " B, BlockLocs=" + std::to_string(total_block_loc_bytes) + " B, BlockTables(Content)="
    //                     + std::to_string(total_block_table_bytes) + " B, BlockTables(Count)="
    //                     + std::to_string(total_inner_lists) + " B, EngineID=" + std::to_string(total_engine_id_bytes)
    //                     + " B, DispTokens=" + std::to_string(total_disp_token_bytes) + " B");

    auto seq_list_off = nanodeploy::fbs::CreateSequenceList(builder, builder.CreateVector(seq_offsets));

    builder.Finish(seq_list_off);

    // Copy to output
    size_t size = builder.GetSize();
    if (size > buffer_size) {
        NANOCOMMON_ABORT("Buffer Overflow: Serialized size " + std::to_string(size) + " > buffer size "
                         + std::to_string(buffer_size));
    }

    std::memcpy(reinterpret_cast<void*>(data_ptr), builder.GetBufferPointer(), size);
    return size;
}

std::vector<std::shared_ptr<Sequence>> deserialize_sequences(uintptr_t data_ptr, size_t data_len)
{
    const uint8_t* buffer = reinterpret_cast<const uint8_t*>(data_ptr);

    // Verify
    flatbuffers::Verifier verifier(buffer, data_len);
    if (!nanodeploy::fbs::VerifySequenceListBuffer(verifier)) {
        throw std::runtime_error("Invalid FlatBuffer: SequenceList verification failed");
    }

    const auto* seq_list  = nanodeploy::fbs::GetSequenceList(buffer);
    const auto* sequences = seq_list->sequences();

    std::vector<std::shared_ptr<Sequence>> result;
    if (!sequences)
        return result;

    result.reserve(sequences->size());

    for (const auto* fb_seq : *sequences) {
        // Core fields
        std::vector<int> token_ids;
        if (auto tids = fb_seq->token_ids()) {
            token_ids.assign(tids->begin(), tids->end());
        }

        SamplingParams sampling_params;
        if (auto sp = fb_seq->sampling_params()) {
            sampling_params.temperature = sp->temperature();
            sampling_params.max_tokens  = sp->max_tokens();
            sampling_params.ignore_eos  = sp->ignore_eos();
        }

        auto seq = std::make_shared<Sequence>(token_ids, sampling_params);

        seq->seq_id                  = fb_seq->seq_id();
        seq->status                  = static_cast<SequenceStatus>(fb_seq->status());
        seq->last_token              = fb_seq->last_token();
        seq->num_tokens              = fb_seq->num_tokens();
        seq->num_prompt_tokens       = fb_seq->num_prompt_tokens();
        seq->num_checkpointed_tokens = fb_seq->num_checkpointed_tokens();
        seq->num_cached_tokens       = fb_seq->num_cached_tokens();

        // Ensure last_token is consistent if token_ids is not empty
        if (!seq->token_ids.empty()) {
            seq->last_token = seq->token_ids.back();
        }

        NANOCOMMON_LOG_DEBUG(
            "Deserialized Sequence: ID=" + std::to_string(seq->seq_id) + " Status=" + std::to_string((int)seq->status)
            + " LastToken=" + std::to_string(seq->last_token) + " NumTokens=" + std::to_string(seq->num_tokens)
            + " NumPrompt=" + std::to_string(seq->num_prompt_tokens) + " NumChpt="
            + std::to_string(seq->num_checkpointed_tokens) + " NumCached=" + std::to_string(seq->num_cached_tokens));

        // Slots
        if (auto slots = fb_seq->slots()) {
            for (size_t i = 0; i < slots->size() && i < seq->slots_.size(); ++i) {
                unpack_block_context(slots->Get(i), seq->slots_[i]);
            }
        }

        result.push_back(seq);
    }

    return result;
}

}  // namespace nanodeploy
