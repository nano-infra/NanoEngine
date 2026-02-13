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

enum class SequenceStatus {
    WAITING,
    RUNNING,
    FINISHED,
    TO_BE_MIGRATED,

    _COUNT
};

enum class BlockContextSlot : int {
    ACTIVE,
    MIGRATE,
    SWAP,

    _COUNT
};

struct BlockLocationList: public std::vector<std::pair<int, int>> {
    using std::vector<std::pair<int, int>>::vector;
};
struct BlockIdList: public std::vector<int> {
    using std::vector<int>::vector;
};
struct SpBlockTable: public std::vector<BlockIdList> {
    using std::vector<BlockIdList>::vector;
};

// Use flatbuffer generated BlockContextT type
using BlockContext = fbs::BlockContextT;

// Free functions for BlockContext operations
std::unique_ptr<BlockContext>
     make_block_context(const std::string& engine_id, int attention_sp, int attention_dp, int num_kvcache_blocks);
void reset_block_context(
    BlockContext& ctx, const std::string& engine_id, int attention_sp, int attention_dp, int num_kvcache_blocks);

struct OptionalStringHash {
    std::size_t operator()(const std::optional<std::string>& s) const
    {
        if (!s.has_value()) {
            return 0;
        }
        return std::hash<std::string>{}(*s);
    }
};

// Use flatbuffer generated types
using SequenceT       = fbs::SequenceT;
using SamplingParamsT = fbs::SamplingParamsT;

// Legacy SamplingParams for backward compatibility (maps to SamplingParamsT)
struct SamplingParams {
    double temperature = 1.0;
    int    max_tokens  = 256;
    bool   ignore_eos  = false;

    // Conversion to/from flatbuffers type
    SamplingParamsT to_flatbuffers() const
    {
        auto params         = std::make_unique<SamplingParamsT>();
        params->temperature = temperature;
        params->max_tokens  = max_tokens;
        params->ignore_eos  = ignore_eos;
        return *params;
    }

    static SamplingParams from_flatbuffers(const SamplingParamsT& params)
    {
        SamplingParams result;
        result.temperature = params.temperature;
        result.max_tokens  = params.max_tokens;
        result.ignore_eos  = params.ignore_eos;
        return result;
    }
};

class Sequence {
public:
    static constexpr int block_size = 256;

    Sequence(const std::vector<int>& token_ids, const SamplingParams& sampling_params = {});

    // Factory method for deserialization - creates Sequence directly from unpacked FlatBuffer data
    static std::shared_ptr<Sequence> from_data(std::unique_ptr<SequenceT> data);

    // Jumping
    int32_t active(const std::string& engine_id, int attention_sp, int attention_dp, int num_kvcache_blocks)
    {
        ensure_slot(BlockContextSlot::ACTIVE);
        reset_block_context(
            *get_slot(BlockContextSlot::ACTIVE), engine_id, attention_sp, attention_dp, num_kvcache_blocks);
        return 0;
    }

    int32_t migrate()
    {
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

    int context_len(BlockContextSlot slot = BlockContextSlot::ACTIVE, std::optional<int> sp_idx = std::nullopt);

    void append_token(int                token_id,
                      BlockContextSlot   slot   = BlockContextSlot::ACTIVE,
                      std::optional<int> sp_idx = std::nullopt);

    // Block related methods
    int num_blocks(BlockContextSlot slot, int sp_idx);
    int last_block_page_id(BlockContextSlot slot, int sp_idx);
    int last_block_num_tokens(BlockContextSlot slot, int sp_idx);
    // Returns a pointer/size view into the internal token storage for block `i`.
    // The returned pointer is valid only as long as the underlying storage is not
    // modified in a way that can reallocate or invalidate the buffer (e.g., appending
    // tokens to the same sequence). Callers MUST NOT store this pointer beyond the
    // duration in which they can guarantee no such modifications occur.
    std::pair<const int*, size_t> block_view(int i, BlockContextSlot slot, int sp_idx) const;
    std::vector<int>              block(int i, BlockContextSlot slot, int sp_idx);

    // Accessors
    BlockContext&       block_ctx(BlockContextSlot slot = BlockContextSlot::ACTIVE);
    const BlockContext& block_ctx(BlockContextSlot slot = BlockContextSlot::ACTIVE) const;
    std::vector<int>&   block_table(BlockContextSlot slot = BlockContextSlot::ACTIVE, int sp_idx = 0);

    int dp_idx(BlockContextSlot slot);

    // Properties
    bool is_finished() const
    {
        return static_cast<SequenceStatus>(data_->status) == SequenceStatus::FINISHED;
    }

    // Properties
    bool is_to_be_migrated() const
    {
        return static_cast<SequenceStatus>(data_->status) == SequenceStatus::TO_BE_MIGRATED;
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
        return static_cast<SequenceStatus>(data_->status);
    }
    void set_status(SequenceStatus s)
    {
        data_->status = static_cast<fbs::SequenceStatus>(s);
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
            return SamplingParams::from_flatbuffers(*data_->sampling_params);
        }
        return SamplingParams();
    }
    void set_sampling_params(const SamplingParams& params)
    {
        if (!data_->sampling_params) {
            data_->sampling_params = std::make_unique<SamplingParamsT>();
        }
        *data_->sampling_params = params.to_flatbuffers();
    }

    using StateTuple = std::tuple<int,
                                  int,
                                  int,
                                  std::optional<std::string>,
                                  std::optional<std::string>,
                                  std::array<std::unique_ptr<BlockContext>, (size_t)BlockContextSlot::_COUNT>,
                                  double,
                                  std::vector<int>,
                                  int,
                                  int,
                                  bool>;

    // Access to flatbuffers data
    std::unique_ptr<SequenceT> data_;

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
    friend std::vector<std::shared_ptr<Sequence>> deserialize_sequences(uintptr_t data_ptr, size_t data_len);

    static std::atomic<uint64_t> next_seq_id_;
};

}  // namespace nanodeploy
