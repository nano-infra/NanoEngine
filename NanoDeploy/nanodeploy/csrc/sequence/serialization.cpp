#include "nanocommon/logging.h"

#include "sequence_generated.h"
#include "serialization.h"
#include <flatbuffers/flatbuffers.h>

namespace nanodeploy {

// Helper factory function
std::shared_ptr<Sequence> sequence_from_data(std::unique_ptr<SequenceT> data)
{
    auto seq   = std::make_shared<Sequence>(std::vector<int>{}, SamplingParams());
    seq->data_ = std::move(data);
    return seq;
}

// Helper: Validate and fix BlockContext before serialization
static void validate_block_context(fbs::BlockContextT& ctx)
{
    // Ensure engine_id is valid (initialize if empty)
    if (ctx.engine_id.empty()) {
        ctx.engine_id = "";
    }

    // Ensure vectors are initialized (not null)
    // block_location should be a valid vector (can be empty)
    // num_dispatched_tokens should match group_size size
    if (ctx.num_dispatched_tokens.size() != static_cast<size_t>(ctx.group_size)) {
        ctx.num_dispatched_tokens.resize(ctx.group_size, 0);
    }

    // Ensure group_block_table matches group_size size and has no null pointers
    if (ctx.group_block_table.size() != static_cast<size_t>(ctx.group_size)) {
        ctx.group_block_table.resize(ctx.group_size);
    }
    for (size_t i = 0; i < ctx.group_block_table.size(); ++i) {
        if (!ctx.group_block_table[i]) {
            ctx.group_block_table[i] = std::make_unique<fbs::IntListT>();
        }
    }

    // Ensure endpoints vector is initialized (can be empty)
    // No action needed for endpoints, it's a std::vector<std::string>
}

// Simplified: Use FlatBuffers Pack() directly
std::vector<uint8_t> serialize_sequence(const Sequence& seq)
{
    // Validate all BlockContexts in slots before serialization (same as serialize_sequences)
    for (size_t slot_idx = 0; slot_idx < seq.data_->slots.size(); ++slot_idx) {
        auto& slot = seq.data_->slots[slot_idx];
        if (slot) {
            validate_block_context(*slot);
        }
        else {
            slot                     = std::make_unique<fbs::BlockContextT>();
            slot->engine_id          = "";
            slot->dp_idx             = 0;
            slot->master_group_id    = 0;
            slot->group_size         = 0;
            slot->attention_dp       = 0;
            slot->num_kvcache_blocks = 0;
        }
    }

    flatbuffers::FlatBufferBuilder builder(64 * 1024);  // 64KB initial
    auto                           offset = fbs::Sequence::Pack(builder, seq.data_.get());
    builder.Finish(offset);

    const uint8_t* buf  = builder.GetBufferPointer();
    size_t         size = builder.GetSize();
    return std::vector<uint8_t>(buf, buf + size);
}

// Simplified: Use FlatBuffers UnPack() directly
std::shared_ptr<Sequence> deserialize_sequence(const uint8_t* buffer, size_t size)
{
    flatbuffers::Verifier verifier(buffer, size);
    if (!verifier.VerifyBuffer<fbs::Sequence>()) {
        throw std::runtime_error("Invalid FlatBuffer: Sequence verification failed");
    }

    auto fb_seq = flatbuffers::GetRoot<fbs::Sequence>(buffer);
    return sequence_from_data(std::unique_ptr<fbs::SequenceT>(fb_seq->UnPack()));
}

size_t serialize_sequences(uintptr_t                                     data_ptr,
                           size_t                                        buffer_size,
                           const std::vector<std::shared_ptr<Sequence>>& seqs,
                           bool                                          is_prefill [[maybe_unused]])
{
    flatbuffers::FlatBufferBuilder builder(buffer_size);

    std::vector<flatbuffers::Offset<fbs::Sequence>> seq_offsets;
    seq_offsets.reserve(seqs.size());

    // Simplified: Use Pack() directly on each sequence
    for (const auto& seq_ptr : seqs) {
        if (!seq_ptr)
            continue;

        try {
            // Validate all BlockContexts in slots before serialization
            for (size_t slot_idx = 0; slot_idx < seq_ptr->data_->slots.size(); ++slot_idx) {
                auto& slot = seq_ptr->data_->slots[slot_idx];
                if (slot) {
                    validate_block_context(*slot);
                }
                else {
                    // Initialize null slot with safe defaults
                    slot                     = std::make_unique<fbs::BlockContextT>();
                    slot->engine_id          = "";
                    slot->dp_idx             = 0;
                    slot->master_group_id    = 0;
                    slot->group_size         = 0;
                    slot->attention_dp       = 0;
                    slot->num_kvcache_blocks = 0;
                }
            }

            seq_offsets.push_back(fbs::Sequence::Pack(builder, seq_ptr->data_.get()));
        }
        catch (const std::exception& e) {
            NANOCOMMON_ABORT("Failed to serialize sequence " + std::to_string(seq_ptr->seq_id()) + ": "
                             + std::string(e.what()));
        }
    }

    auto seq_list_off = fbs::CreateSequenceList(builder, builder.CreateVector(seq_offsets));
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

    flatbuffers::Verifier verifier(buffer, data_len);
    if (!fbs::VerifySequenceListBuffer(verifier)) {
        throw std::runtime_error("Invalid FlatBuffer: SequenceList verification failed");
    }

    const auto* seq_list  = fbs::GetSequenceList(buffer);
    const auto* sequences = seq_list->sequences();

    std::vector<std::shared_ptr<Sequence>> result;
    if (!sequences)
        return result;

    result.reserve(sequences->size());

    // Simplified: Use UnPack() directly
    for (const auto* fb_seq : *sequences) {
        auto seq = sequence_from_data(std::unique_ptr<fbs::SequenceT>(fb_seq->UnPack()));

        // Validate all BlockContexts after deserialization
        for (auto& slot : seq->data_->slots) {
            if (slot) {
                validate_block_context(*slot);
            }
        }

        NANOCOMMON_LOG_DEBUG("Deserialized Sequence: ID=" + std::to_string(seq->seq_id()) + " Status="
                             + std::to_string((int)seq->status()) + " LastToken=" + std::to_string(seq->last_token())
                             + " NumTokens=" + std::to_string(seq->num_tokens()));

        result.push_back(seq);
    }

    return result;
}

}  // namespace nanodeploy
