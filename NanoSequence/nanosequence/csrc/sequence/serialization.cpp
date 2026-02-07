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

// Simplified: Use FlatBuffers Pack() directly
std::vector<uint8_t> serialize_sequence(const Sequence& seq)
{
    flatbuffers::FlatBufferBuilder builder(1024);
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
        seq_offsets.push_back(fbs::Sequence::Pack(builder, seq_ptr->data_.get()));
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

        NANOCOMMON_LOG_DEBUG("Deserialized Sequence: ID=" + std::to_string(seq->seq_id()) + " Status="
                             + std::to_string((int)seq->status()) + " LastToken=" + std::to_string(seq->last_token())
                             + " NumTokens=" + std::to_string(seq->num_tokens()));

        result.push_back(seq);
    }

    return result;
}

}  // namespace nanodeploy
