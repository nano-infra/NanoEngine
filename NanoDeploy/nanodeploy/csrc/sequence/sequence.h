#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

// Include flatbuffer generated header (will be available at build time)
// We need the full definition here because we use std::make_unique<BlockContext>
// in inline methods. The header will be generated before this file is compiled.
#include "sequence_generated.h"

namespace nanodeploy {

// Forward declaration
class SequenceMetric;

// Use flatbuffer generated SequenceStatus enum directly
using SequenceStatus = fbs::SequenceStatus;

enum class BlockContextSlot : int {
    ACTIVE,
    MIGRATE,
    SWAP,

    _COUNT
};

// Use flatbuffer generated BlockContextT type
using BlockContext = fbs::BlockContextT;

// Free functions for BlockContext operations
std::unique_ptr<BlockContext>
     make_block_context(const std::string& engine_id, int group_size, int attention_dp, int num_kvcache_blocks);
void reset_block_context(
    BlockContext& ctx, const std::string& engine_id, int group_size, int attention_dp, int num_kvcache_blocks);

// Use flatbuffer generated types
using SequenceT       = fbs::SequenceT;
using SamplingParamsT = fbs::SamplingParamsT;
using SamplingParams  = SamplingParamsT;

class Sequence {
public:
    static int  block_size;
    static void set_block_size(int bs)
    {
        block_size = bs;
    }

    Sequence(const std::vector<int>& token_ids, const SamplingParams& sampling_params = {});

    // Factory method for deserialization - creates Sequence directly from unpacked FlatBuffer data
    static std::shared_ptr<Sequence> from_data(std::unique_ptr<SequenceT> data);

    // Jumping
    int32_t active(const std::string& engine_id, int group_size, int attention_dp, int num_kvcache_blocks)
    {
        ensure_slot(BlockContextSlot::ACTIVE);
        reset_block_context(
            *get_slot(BlockContextSlot::ACTIVE), engine_id, group_size, attention_dp, num_kvcache_blocks);
        return 0;
    }

    int32_t migrate()
    {

        set_num_checkpointed_tokens(num_tokens());

        ensure_slot(BlockContextSlot::MIGRATE);
        ensure_slot(BlockContextSlot::ACTIVE);  // Ensure ACTIVE exists before move

        // Move ACTIVE to MIGRATE
        get_slot(BlockContextSlot::MIGRATE) = std::move(get_slot(BlockContextSlot::ACTIVE));

        // Don't null out ACTIVE - create a new empty BlockContext instead
        // to avoid segfault during FlatBuffers serialization
        get_slot(BlockContextSlot::ACTIVE) = std::make_unique<BlockContext>();

        return 0;
    }

    BlockContext& active_ctx()
    {
        ensure_slot(BlockContextSlot::ACTIVE);
        return *get_slot(BlockContextSlot::ACTIVE);
    }

    BlockContext& migrate_ctx()
    {
        ensure_slot(BlockContextSlot::MIGRATE);
        return *get_slot(BlockContextSlot::MIGRATE);
    }

    int context_len(BlockContextSlot slot = BlockContextSlot::ACTIVE, std::optional<int> group_id = std::nullopt);

    void append_token(int                token_id,
                      BlockContextSlot   slot     = BlockContextSlot::ACTIVE,
                      std::optional<int> group_id = std::nullopt);

    // Block related methods
    int num_blocks(BlockContextSlot slot, int group_id);
    int last_block_page_id(BlockContextSlot slot, int group_id);
    int last_block_num_tokens(BlockContextSlot slot, int group_id);
    // Returns a pointer/size view into the internal token storage for block `i`.
    // The returned pointer is valid only as long as the underlying storage is not
    // modified in a way that can reallocate or invalidate the buffer (e.g., appending
    // tokens to the same sequence). Callers MUST NOT store this pointer beyond the
    // duration in which they can guarantee no such modifications occur.
    std::pair<const int*, size_t> block_view(int i, BlockContextSlot slot, int group_id) const;
    std::vector<int>              block(int i, BlockContextSlot slot, int group_id);

    // Accessors
    BlockContext&       block_ctx(BlockContextSlot slot = BlockContextSlot::ACTIVE);
    const BlockContext& block_ctx(BlockContextSlot slot = BlockContextSlot::ACTIVE) const;
    std::vector<int>&   block_table(BlockContextSlot slot = BlockContextSlot::ACTIVE, int group_id = 0);

    int dp_idx(BlockContextSlot slot);

    // State management
    int  state_slot(BlockContextSlot slot = BlockContextSlot::ACTIVE) const;
    void set_state_slot(BlockContextSlot slot, int state_slot);

    // Properties
    bool is_finished() const
    {
        return data_->status == SequenceStatus::FINISHED;
    }

    // Properties
    bool is_to_be_migrated() const
    {
        return data_->status == SequenceStatus::TO_BE_MIGRATED;
    }

    int num_completed_tokens() const
    {
        return data_->num_tokens - data_->num_prompt_tokens;
    }
    int num_generated_tokens_since_checkpoint() const
    {
        return data_->num_tokens - data_->num_checkpointed_tokens;
    }
    std::vector<int> prompt_token_ids() const;
    std::vector<int> completion_token_ids() const;
    int              num_cached_blocks() const
    {
        return data_->num_cached_tokens / block_size;
    }

    // Accessors for flatbuffers data
    uint64_t seq_id() const
    {
        return data_->seq_id;
    }
    void set_seq_id(uint64_t id)
    {
        data_->seq_id = id;
    }

    SequenceStatus status() const
    {
        return data_->status;
    }
    void set_status(SequenceStatus s)
    {
        data_->status = s;
    }

    std::vector<int>& token_ids()
    {
        return data_->token_ids;
    }
    const std::vector<int>& token_ids() const
    {
        return data_->token_ids;
    }

    int last_token() const
    {
        return data_->last_token;
    }
    void set_last_token(int token)
    {
        data_->last_token = token;
    }

    int num_tokens() const
    {
        return data_->num_tokens;
    }
    void set_num_tokens(int n)
    {
        data_->num_tokens = n;
    }

    int num_prompt_tokens() const
    {
        return data_->num_prompt_tokens;
    }
    void set_num_prompt_tokens(int n)
    {
        data_->num_prompt_tokens = n;
    }

    int num_checkpointed_tokens() const
    {
        return data_->num_checkpointed_tokens;
    }
    void set_num_checkpointed_tokens(int n)
    {
        data_->num_checkpointed_tokens = n;
    }

    int num_cached_tokens() const
    {
        return data_->num_cached_tokens;
    }
    void set_num_cached_tokens(int n)
    {
        data_->num_cached_tokens = n;
    }

    SamplingParams sampling_params() const
    {
        if (data_->sampling_params) {
            return *data_->sampling_params;
        }
        return SamplingParams();
    }
    void set_sampling_params(const SamplingParams& params)
    {
        if (!data_->sampling_params) {
            data_->sampling_params = std::make_unique<SamplingParamsT>();
        }
        *data_->sampling_params = params;
    }

    // Access to flatbuffers data
    std::unique_ptr<SequenceT> data_;

    // Vision slot management (EP separated mode)
    void add_vision_slot(
        const std::string& encoder_engine_id, int slot_idx, int num_tokens, int hidden_size, int max_tokens_per_slot)
    {
        auto vs                 = std::make_unique<fbs::VisionSlotT>();
        vs->encoder_engine_id   = encoder_engine_id;
        vs->slot_idx            = slot_idx;
        vs->num_tokens          = num_tokens;
        vs->hidden_size         = hidden_size;
        vs->max_tokens_per_slot = max_tokens_per_slot;
        data_->vision_slots.push_back(std::move(vs));
    }

    void clear_vision_slots()
    {
        data_->vision_slots.clear();
    }

    const auto& vision_slots() const
    {
        return data_->vision_slots;
    }

    std::shared_ptr<SequenceMetric> metric;

private:
    // Helper methods for slot management
    std::unique_ptr<BlockContext>& get_slot(BlockContextSlot slot)
    {
        size_t idx = static_cast<size_t>(slot);
        if (data_->slots.size() <= idx) {
            data_->slots.resize((size_t)BlockContextSlot::_COUNT);
        }
        return data_->slots[idx];
    }

    const std::unique_ptr<BlockContext>& get_slot(BlockContextSlot slot) const
    {
        size_t idx = static_cast<size_t>(slot);
        if (data_->slots.size() <= idx) {
            const_cast<Sequence*>(this)->data_->slots.resize((size_t)BlockContextSlot::_COUNT);
        }
        return data_->slots[idx];
    }

    void ensure_slot(BlockContextSlot slot) const
    {
        auto& slot_ref = const_cast<Sequence*>(this)->get_slot(slot);
        if (!slot_ref) {
            slot_ref = std::make_unique<BlockContext>();
        }
    }

    // Make get_slot accessible to serialization code
    friend void unpack_block_context(const nanodeploy::fbs::BlockContext* fb_ctx, BlockContext& ctx);

    static std::atomic<uint64_t> next_seq_id_;
};

}  // namespace nanodeploy
