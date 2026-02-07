#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

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

struct BlockContext {
    std::string engine_id_;

    int dp_idx_{-1};
    int master_sp_idx_{0};
    int attention_sp_{1};
    int attention_dp_{1};
    int num_kvcache_blocks_{-1};

    // Wrapper container types.
    //
    // Motivation: pybind11 converts STL containers to Python list/dict copies by
    // default. These wrappers are distinct C++ types, allowing us to bind them
    // with pybind11 (stl_bind) as *mutable proxy objects* without affecting other
    // std::vector<int> usages (e.g., Sequence::token_ids).
    struct BlockLocationList: public std::vector<std::pair<int, int>> {
        using std::vector<std::pair<int, int>>::vector;
    };
    struct BlockIdList: public std::vector<int> {
        using std::vector<int>::vector;
    };
    struct SpBlockTable: public std::vector<BlockIdList> {
        using std::vector<BlockIdList>::vector;
    };

    // block_location: vector of (sp_idx, block_id) pairs
    BlockLocationList block_location;

    // sp_block_table: sp_idx -> list of block_ids
    SpBlockTable sp_block_table;

    // num_dispatched_tokens: sp_idx -> count
    std::vector<int> num_dispatched_tokens;

    BlockContext() = default;
    BlockContext(const std::string& engine_id, int attention_sp, int attention_dp, int num_kvcache_blocks);

    // For pickle
    std::tuple<std::string,
               int,
               int,
               int,
               int,
               int,
               std::vector<std::pair<int, int>>,
               std::vector<std::vector<int>>,
               std::vector<int>>
    getstate() const;

    static BlockContext setstate(const std::tuple<std::string,
                                                  int,
                                                  int,
                                                  int,
                                                  int,
                                                  int,
                                                  std::vector<std::pair<int, int>>,
                                                  std::vector<std::vector<int>>,
                                                  std::vector<int>>& state);

    void reset(const std::string& engine_id, int attention_sp, int attention_dp, int num_kvcache_blocks);
};

struct OptionalStringHash {
    std::size_t operator()(const std::optional<std::string>& s) const
    {
        if (!s.has_value()) {
            return 0;
        }
        return std::hash<std::string>{}(*s);
    }
};

struct SamplingParams {
    double temperature = 1.0;
    int    max_tokens  = 256;
    bool   ignore_eos  = false;
};

class Sequence {
public:
    static constexpr int block_size = 256;

    Sequence(const std::vector<int>& token_ids, const SamplingParams& sampling_params = {});

    // Jumping
    int32_t active(const std::string& engine_id, int attention_sp, int attention_dp, int num_kvcache_blocks)
    {
        slots_[(size_t)BlockContextSlot::ACTIVE].reset(engine_id, attention_sp, attention_dp, num_kvcache_blocks);
        return 0;
    }

    int32_t migrate()
    {
        slots_[(size_t)BlockContextSlot::MIGRATE] = std::move(slots_[(size_t)BlockContextSlot::ACTIVE]);
        return 0;
    }

    BlockContext& active_ctx()
    {
        return slots_[(size_t)BlockContextSlot::ACTIVE];
    }

    BlockContext& migrate_ctx()
    {
        return slots_[(size_t)BlockContextSlot::MIGRATE];
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
    BlockContext&              block_ctx(BlockContextSlot slot = BlockContextSlot::ACTIVE);
    const BlockContext&        block_ctx(BlockContextSlot slot = BlockContextSlot::ACTIVE) const;
    BlockContext::BlockIdList& block_table(BlockContextSlot slot = BlockContextSlot::ACTIVE, int sp_idx = 0);

    int dp_idx(BlockContextSlot slot);

    // Properties
    bool is_finished() const
    {
        return status == SequenceStatus::FINISHED;
    }

    // Properties
    bool is_to_be_migrated() const
    {
        return status == SequenceStatus::TO_BE_MIGRATED;
    }

    int num_completed_tokens() const
    {
        return num_tokens - num_prompt_tokens;
    }
    int num_generated_tokens_since_checkpoint() const
    {
        return num_tokens - num_checkpointed_tokens;
    }
    std::vector<int> prompt_token_ids() const;
    std::vector<int> completion_token_ids() const;
    int              num_cached_blocks() const
    {
        return num_cached_tokens / block_size;
    }

    using StateTuple = std::tuple<int,
                                  int,
                                  int,
                                  std::optional<std::string>,
                                  std::optional<std::string>,
                                  std::array<BlockContext, (size_t)BlockContextSlot::_COUNT>,
                                  double,
                                  std::vector<int>,
                                  int,
                                  int,
                                  bool>;

    // Public members
    uint64_t         seq_id;
    SequenceStatus   status = SequenceStatus::WAITING;
    std::vector<int> token_ids;
    int              last_token;
    int              num_tokens;
    int              num_prompt_tokens;
    int              num_checkpointed_tokens;
    int              num_cached_tokens = 0;

    std::shared_ptr<SequenceMetric> metric;

    SamplingParams sampling_params;

    std::array<BlockContext, (size_t)BlockContextSlot::_COUNT> slots_;

private:
    static std::atomic<uint64_t> next_seq_id_;
};

}  // namespace nanodeploy
