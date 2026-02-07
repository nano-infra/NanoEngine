#include "nanosequence/csrc/ffi/nanosequence_c.h"
#include "nanosequence/csrc/sequence/sequence.h"
#include "nanosequence/csrc/sequence/serialization.h"
#include <cstring>
#include <memory>
#include <vector>

extern "C" {

// SamplingParams API
NanosequenceSamplingParams nanosequence_sampling_params_new(double temperature, int max_tokens, bool ignore_eos)
{
    auto* params        = new nanodeploy::SamplingParams();
    params->temperature = temperature;
    params->max_tokens  = max_tokens;
    params->ignore_eos  = ignore_eos;
    return reinterpret_cast<NanosequenceSamplingParams>(params);
}

void nanosequence_sampling_params_free(NanosequenceSamplingParams params)
{
    if (params) {
        delete reinterpret_cast<nanodeploy::SamplingParams*>(params);
    }
}

// Sequence API
NanosequenceSequence
nanosequence_sequence_new(const int32_t* token_ids, size_t token_count, NanosequenceSamplingParams params)
{
    std::vector<int>            tokens(token_ids, token_ids + token_count);
    nanodeploy::SamplingParams* sp = params ? reinterpret_cast<nanodeploy::SamplingParams*>(params) : nullptr;
    nanodeploy::SamplingParams  default_params;
    nanodeploy::SamplingParams& sampling_params = sp ? *sp : default_params;

    auto* seq = new nanodeploy::Sequence(tokens, sampling_params);
    return reinterpret_cast<NanosequenceSequence>(seq);
}

void nanosequence_sequence_free(NanosequenceSequence seq)
{
    if (seq) {
        delete reinterpret_cast<nanodeploy::Sequence*>(seq);
    }
}

// Sequence properties
uint64_t nanosequence_sequence_get_seq_id(NanosequenceSequence seq)
{
    if (!seq)
        return 0;
    return reinterpret_cast<nanodeploy::Sequence*>(seq)->seq_id;
}

void nanosequence_sequence_set_seq_id(NanosequenceSequence seq, uint64_t seq_id)
{
    if (seq) {
        reinterpret_cast<nanodeploy::Sequence*>(seq)->seq_id = seq_id;
    }
}

NanosequenceStatus nanosequence_sequence_get_status(NanosequenceSequence seq)
{
    if (!seq)
        return NANOSEQUENCE_STATUS_WAITING;
    auto status = reinterpret_cast<nanodeploy::Sequence*>(seq)->status;
    switch (status) {
        case nanodeploy::SequenceStatus::WAITING:
            return NANOSEQUENCE_STATUS_WAITING;
        case nanodeploy::SequenceStatus::RUNNING:
            return NANOSEQUENCE_STATUS_RUNNING;
        case nanodeploy::SequenceStatus::FINISHED:
            return NANOSEQUENCE_STATUS_FINISHED;
        case nanodeploy::SequenceStatus::TO_BE_MIGRATED:
            return NANOSEQUENCE_STATUS_TO_BE_MIGRATED;
        default:
            return NANOSEQUENCE_STATUS_WAITING;
    }
}

void nanosequence_sequence_set_status(NanosequenceSequence seq, NanosequenceStatus status)
{
    if (!seq)
        return;
    auto* s = reinterpret_cast<nanodeploy::Sequence*>(seq);
    switch (status) {
        case NANOSEQUENCE_STATUS_WAITING:
            s->status = nanodeploy::SequenceStatus::WAITING;
            break;
        case NANOSEQUENCE_STATUS_RUNNING:
            s->status = nanodeploy::SequenceStatus::RUNNING;
            break;
        case NANOSEQUENCE_STATUS_FINISHED:
            s->status = nanodeploy::SequenceStatus::FINISHED;
            break;
        case NANOSEQUENCE_STATUS_TO_BE_MIGRATED:
            s->status = nanodeploy::SequenceStatus::TO_BE_MIGRATED;
            break;
    }
}

int nanosequence_sequence_get_num_tokens(NanosequenceSequence seq)
{
    if (!seq)
        return 0;
    return reinterpret_cast<nanodeploy::Sequence*>(seq)->num_tokens;
}

int nanosequence_sequence_get_num_prompt_tokens(NanosequenceSequence seq)
{
    if (!seq)
        return 0;
    return reinterpret_cast<nanodeploy::Sequence*>(seq)->num_prompt_tokens;
}

int nanosequence_sequence_get_last_token(NanosequenceSequence seq)
{
    if (!seq)
        return 0;
    return reinterpret_cast<nanodeploy::Sequence*>(seq)->last_token;
}

void nanosequence_sequence_get_token_ids(NanosequenceSequence seq, int32_t* out_buffer, size_t buffer_size)
{
    if (!seq || !out_buffer)
        return;
    auto*  s         = reinterpret_cast<nanodeploy::Sequence*>(seq);
    size_t copy_size = std::min(buffer_size, s->token_ids.size());
    for (size_t i = 0; i < copy_size; ++i) {
        out_buffer[i] = static_cast<int32_t>(s->token_ids[i]);
    }
}

size_t nanosequence_sequence_get_token_count(NanosequenceSequence seq)
{
    if (!seq)
        return 0;
    return reinterpret_cast<nanodeploy::Sequence*>(seq)->token_ids.size();
}

void nanosequence_sequence_append_token(NanosequenceSequence         seq,
                                        int32_t                      token_id,
                                        NanosequenceBlockContextSlot slot,
                                        int                          sp_idx)
{
    if (!seq)
        return;
    auto*                        s = reinterpret_cast<nanodeploy::Sequence*>(seq);
    nanodeploy::BlockContextSlot cpp_slot;
    switch (slot) {
        case NANOSEQUENCE_SLOT_ACTIVE:
            cpp_slot = nanodeploy::BlockContextSlot::ACTIVE;
            break;
        case NANOSEQUENCE_SLOT_MIGRATE:
            cpp_slot = nanodeploy::BlockContextSlot::MIGRATE;
            break;
        case NANOSEQUENCE_SLOT_SWAP:
            cpp_slot = nanodeploy::BlockContextSlot::SWAP;
            break;
        default:
            cpp_slot = nanodeploy::BlockContextSlot::ACTIVE;
            break;
    }
    s->append_token(token_id, cpp_slot, sp_idx >= 0 ? std::make_optional(sp_idx) : std::nullopt);
}

int32_t nanosequence_sequence_active(
    NanosequenceSequence seq, const char* engine_id, int attention_sp, int attention_dp, int num_kvcache_blocks)
{
    if (!seq)
        return -1;
    auto* s = reinterpret_cast<nanodeploy::Sequence*>(seq);
    return s->active(std::string(engine_id), attention_sp, attention_dp, num_kvcache_blocks);
}

int32_t nanosequence_sequence_migrate(NanosequenceSequence seq)
{
    if (!seq)
        return -1;
    auto* s = reinterpret_cast<nanodeploy::Sequence*>(seq);
    return s->migrate();
}

NanosequenceBlockContext nanosequence_sequence_get_block_ctx(NanosequenceSequence         seq,
                                                             NanosequenceBlockContextSlot slot)
{
    if (!seq)
        return nullptr;
    auto*                        s = reinterpret_cast<nanodeploy::Sequence*>(seq);
    nanodeploy::BlockContextSlot cpp_slot;
    switch (slot) {
        case NANOSEQUENCE_SLOT_ACTIVE:
            cpp_slot = nanodeploy::BlockContextSlot::ACTIVE;
            break;
        case NANOSEQUENCE_SLOT_MIGRATE:
            cpp_slot = nanodeploy::BlockContextSlot::MIGRATE;
            break;
        case NANOSEQUENCE_SLOT_SWAP:
            cpp_slot = nanodeploy::BlockContextSlot::SWAP;
            break;
        default:
            cpp_slot = nanodeploy::BlockContextSlot::ACTIVE;
            break;
    }
    auto& ctx = s->block_ctx(cpp_slot);
    return reinterpret_cast<NanosequenceBlockContext>(&ctx);
}

// BlockContext API
void nanosequence_block_context_free(NanosequenceBlockContext ctx)
{
    // BlockContext is owned by Sequence, so we don't free it here
    // This is a no-op, but kept for API consistency
    (void)ctx;
}

const char* nanosequence_block_context_get_engine_id(NanosequenceBlockContext ctx)
{
    if (!ctx)
        return nullptr;
    auto* c = reinterpret_cast<nanodeploy::BlockContext*>(ctx);
    // Note: This returns a pointer to internal string data
    // Caller should copy the string if they need to keep it
    return c->engine_id_.c_str();
}

int nanosequence_block_context_get_dp_idx(NanosequenceBlockContext ctx)
{
    if (!ctx)
        return -1;
    return reinterpret_cast<nanodeploy::BlockContext*>(ctx)->dp_idx_;
}

int nanosequence_block_context_get_master_sp_idx(NanosequenceBlockContext ctx)
{
    if (!ctx)
        return 0;
    return reinterpret_cast<nanodeploy::BlockContext*>(ctx)->master_sp_idx_;
}

int nanosequence_block_context_get_attention_sp(NanosequenceBlockContext ctx)
{
    if (!ctx)
        return 1;
    return reinterpret_cast<nanodeploy::BlockContext*>(ctx)->attention_sp_;
}

int nanosequence_block_context_get_attention_dp(NanosequenceBlockContext ctx)
{
    if (!ctx)
        return 1;
    return reinterpret_cast<nanodeploy::BlockContext*>(ctx)->attention_dp_;
}

int nanosequence_block_context_get_num_kvcache_blocks(NanosequenceBlockContext ctx)
{
    if (!ctx)
        return -1;
    return reinterpret_cast<nanodeploy::BlockContext*>(ctx)->num_kvcache_blocks_;
}

// Serialization API
size_t nanosequence_serialize_sequences(uintptr_t                   data_ptr,
                                        size_t                      buffer_size,
                                        const NanosequenceSequence* sequences,
                                        size_t                      sequence_count,
                                        bool                        is_prefill)
{
    if (!sequences || sequence_count == 0)
        return 0;

    std::vector<std::shared_ptr<nanodeploy::Sequence>> seqs;
    seqs.reserve(sequence_count);

    for (size_t i = 0; i < sequence_count; ++i) {
        if (sequences[i]) {
            auto* seq = reinterpret_cast<nanodeploy::Sequence*>(sequences[i]);
            // Create a shared_ptr that doesn't own (we'll manage lifetime separately)
            // In practice, the caller should ensure sequences remain valid during serialization
            seqs.push_back(std::shared_ptr<nanodeploy::Sequence>(seq, [](nanodeploy::Sequence*) {}));
        }
    }

    return nanodeploy::serialize_sequences(data_ptr, buffer_size, seqs, is_prefill);
}

// Deserialization API
NanosequenceSequence*
nanosequence_deserialize_sequences(uintptr_t data_ptr, size_t data_len, size_t* out_sequence_count)
{
    if (!out_sequence_count)
        return nullptr;

    auto seqs           = nanodeploy::deserialize_sequences(data_ptr, data_len);
    *out_sequence_count = seqs.size();

    if (seqs.empty()) {
        return nullptr;
    }

    // Allocate array of opaque pointers
    auto* result = static_cast<NanosequenceSequence*>(std::malloc(sizeof(NanosequenceSequence) * seqs.size()));
    if (!result) {
        *out_sequence_count = 0;
        return nullptr;
    }

    // Convert shared_ptr to raw pointers
    // We extract the raw pointer from shared_ptr and transfer ownership
    // Note: This is safe because we're creating new objects that the caller will own
    for (size_t i = 0; i < seqs.size(); ++i) {
        // Get the raw pointer - the shared_ptr will be destroyed but we create a new one
        // Actually, we need to keep the shared_ptr alive, so we'll create a copy
        auto& seq = seqs[i];
        // Create a new Sequence with the same token_ids
        auto* new_seq = new nanodeploy::Sequence(seq->token_ids, seq->sampling_params);
        // Copy all public members
        new_seq->seq_id                  = seq->seq_id;
        new_seq->status                  = seq->status;
        new_seq->num_tokens              = seq->num_tokens;
        new_seq->num_prompt_tokens       = seq->num_prompt_tokens;
        new_seq->num_checkpointed_tokens = seq->num_checkpointed_tokens;
        new_seq->num_cached_tokens       = seq->num_cached_tokens;
        new_seq->last_token              = seq->last_token;
        // Copy block contexts using public API
        for (size_t j = 0; j < static_cast<size_t>(nanodeploy::BlockContextSlot::_COUNT); ++j) {
            auto  slot    = static_cast<nanodeploy::BlockContextSlot>(j);
            auto& src_ctx = seq->block_ctx(slot);
            auto& dst_ctx = new_seq->block_ctx(slot);
            dst_ctx       = src_ctx;  // Use copy assignment
        }
        result[i] = reinterpret_cast<NanosequenceSequence>(new_seq);
    }

    return result;
}

void nanosequence_sequences_free(NanosequenceSequence* sequences, size_t count)
{
    if (!sequences)
        return;

    for (size_t i = 0; i < count; ++i) {
        if (sequences[i]) {
            delete reinterpret_cast<nanodeploy::Sequence*>(sequences[i]);
        }
    }

    std::free(sequences);
}

}  // extern "C"
