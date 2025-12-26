#pragma once

#include <atomic>
#include <iomanip>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
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
    int         dp_idx_        = -1;
    int         master_sp_idx_ = 0;
    int         attention_sp_  = 1;
    int         attention_dp_  = 1;

    std::vector<std::pair<int, int>> block_location;

    std::vector<std::vector<int>> sp_block_table;

    std::vector<int> num_dispatched_tokens;

    BlockContext() = default;
    BlockContext(const std::string& engine_id, int attention_sp, int attention_dp);

    // For pickle
    std::tuple<std::string,
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
                                                  std::vector<std::pair<int, int>>,
                                                  std::vector<std::vector<int>>,
                                                  std::vector<int>>& state);

    void reset(const std::string& engine_id, int attention_sp, int attention_dp);
};

// Custom hash for optional string to be used in unordered_map
struct OptionalStringHash {
    std::size_t operator()(const std::optional<std::string>& s) const
    {
        if (!s.has_value()) {
            return 0;
        }
        return std::hash<std::string>{}(*s);
    }
};

class Sequence {
public:
    static constexpr int block_size = 256;

    Sequence(const std::vector<int>& token_ids,
             double                  temperature = 1.0,
             int                     max_tokens  = 256,
             bool                    ignore_eos  = false);

    int32_t active(const std::string& engine_id, int attention_sp, int attention_dp)
    {
        slots_[(size_t)BlockContextSlot::ACTIVE].reset(engine_id, attention_sp, attention_dp);
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

    int num_blocks(BlockContextSlot slot, int sp_idx);
    int last_block_page_id(BlockContextSlot slot, int sp_idx);
    int last_block_num_tokens(BlockContextSlot slot, int sp_idx);

    std::pair<const int*, size_t> block_view(int i, BlockContextSlot slot, int sp_idx) const;
    std::vector<int>              block(int i, BlockContextSlot slot, int sp_idx);

    BlockContext&       block_ctx(BlockContextSlot slot = BlockContextSlot::ACTIVE);
    const BlockContext& block_ctx(BlockContextSlot slot = BlockContextSlot::ACTIVE) const;
    std::vector<int>&   block_table(BlockContextSlot slot = BlockContextSlot::ACTIVE, int sp_idx = 0);

    int dp_idx(BlockContextSlot slot);

    // Properties
    bool is_finished() const
    {
        return status == SequenceStatus::FINISHED;
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

    uint64_t         seq_id;
    SequenceStatus   status = SequenceStatus::WAITING;
    std::vector<int> token_ids;
    int              last_token;
    int              num_tokens;
    int              num_prompt_tokens;
    int              num_checkpointed_tokens;
    int              num_cached_tokens = 0;

    std::shared_ptr<SequenceMetric> metric;

    double temperature;
    int    max_tokens;
    bool   ignore_eos;

    std::array<BlockContext, (size_t)BlockContextSlot::_COUNT> slots_;

private:
    static std::atomic<uint64_t> next_seq_id_;
};

}  // namespace nanodeploy
