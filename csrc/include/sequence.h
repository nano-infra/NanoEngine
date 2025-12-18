#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <memory>
#include <tuple>
#include <random>
#include <sstream>
#include <iomanip>

namespace nanodeploy {

// Forward declaration
class SequenceMetric;

struct BlockContext {
    std::optional<std::string> engine_id;
    int dp_idx = -1;
    int master_sp_idx = 0;
    int attention_sp = 1;
    int attention_dp = 1;
    
    // block_location: vector of (sp_idx, block_id) pairs
    std::vector<std::pair<int, int>> block_location;
    
    // sp_block_table: sp_idx -> list of block_ids
    std::unordered_map<int, std::vector<int>> sp_block_table;
    
    // num_dispatched_tokens: sp_idx -> count
    std::unordered_map<int, int> num_dispatched_tokens;
    
    BlockContext() = default;
    BlockContext(const std::optional<std::string>& engine_id,
                 int dp_idx, int master_sp_idx,
                 int attention_sp, int attention_dp);
                 
    // For pickle
    std::tuple<std::optional<std::string>, int, int, int, int, 
               std::vector<std::pair<int, int>>,
               std::unordered_map<int, std::vector<int>>,
               std::unordered_map<int, int>> getstate() const;
               
    static BlockContext setstate(const std::tuple<std::optional<std::string>, int, int, int, int, 
               std::vector<std::pair<int, int>>,
               std::unordered_map<int, std::vector<int>>,
               std::unordered_map<int, int>>& state);
};

enum class SequenceStatus {
    WAITING,
    RUNNING,
    FINISHED,
    TO_BE_MIGRATED
};

// Custom hash for optional string to be used in unordered_map
struct OptionalStringHash {
    std::size_t operator()(const std::optional<std::string>& s) const {
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
             double temperature = 1.0,
             int max_tokens = 256,
             bool ignore_eos = false,
             const std::optional<std::string>& engine_id = std::nullopt,
             int master_sp_rank = 0);
    
    // Core methods
    void set_engine_id(const std::string& engine_id, 
                       int attention_dp = 1, 
                       int attention_sp = 1);
    
    int context_len(const std::optional<std::string>& engine_id = std::nullopt,
                    std::optional<int> sp_idx = std::nullopt);
    
    void append_token(int token_id,
                      const std::optional<std::string>& engine_id = std::nullopt,
                      std::optional<int> sp_idx = std::nullopt);

    // Helpers for Python-side mutation of internal containers.
    // pybind11 converts STL containers to Python copies by default; these methods
    // ensure mutations update the underlying C++ state.
    void block_table_append(int block_id,
                            const std::optional<std::string>& engine_id = std::nullopt,
                            int sp_idx = 0);
    void block_table_clear(const std::optional<std::string>& engine_id = std::nullopt,
                           int sp_idx = 0);
    void block_table_set(const std::vector<int>& table,
                         const std::optional<std::string>& engine_id = std::nullopt,
                         int sp_idx = 0);

    void block_location_append(int sp_idx,
                               int block_id,
                               const std::optional<std::string>& engine_id = std::nullopt);
    void block_location_clear(const std::optional<std::string>& engine_id = std::nullopt);

    void sp_block_table_clear(const std::optional<std::string>& engine_id = std::nullopt);
    void num_dispatched_tokens_clear(const std::optional<std::string>& engine_id = std::nullopt);
    
    // Block related methods
    int num_blocks(const std::optional<std::string>& engine_id, int sp_idx);
    int last_block_page_id(const std::optional<std::string>& engine_id, int sp_idx);
    int last_block_num_tokens(const std::optional<std::string>& engine_id, int sp_idx);
    std::vector<int> block(int i, const std::optional<std::string>& engine_id, int sp_idx);
    
    // Accessors
    BlockContext& block_ctx(const std::optional<std::string>& engine_id = std::nullopt);
    const BlockContext& block_ctx(const std::optional<std::string>& engine_id = std::nullopt) const;
    std::vector<int>& block_table(const std::optional<std::string>& engine_id = std::nullopt, 
                                   int sp_idx = 0);
    
    int dp_idx(const std::optional<std::string>& engine_id);

    // Properties
    bool is_finished() const { return status == SequenceStatus::FINISHED; }
    int num_completed_tokens() const { return num_tokens - num_prompt_tokens; }
    int num_generated_tokens_since_checkpoint() const { 
        return num_tokens - num_checkpointed_tokens; 
    }
    std::vector<int> prompt_token_ids() const;
    std::vector<int> completion_token_ids() const;
    int num_cached_blocks() const { return num_cached_tokens / block_size; }
    
    // Pickle support
    // (num_tokens, num_checkpointed_tokens, num_cached_tokens, backup_engine_id, active_engine_id, block_ctx_map, temperature, token_ids/last_token)
    // Note: token_ids/last_token logic is handled in getstate implementation
    using StateTuple = std::tuple<
        int, int, int, 
        std::optional<std::string>, std::optional<std::string>,
        std::unordered_map<std::optional<std::string>, BlockContext, OptionalStringHash>,
        double,
        std::vector<int>, // We will always return full token_ids for simplicity in C++ or handle the logic
        int // last_token, used if we don't return full token_ids? 
            // Python logic: if num_generated_tokens_since_checkpoint == 0: token_ids else: last_token
            // We can use a variant or just return both and ignore one.
            // Let's stick to Python's tuple structure. It returns a tuple where the last element varies.
            // In C++, we can't easily return a tuple with varying types.
            // We will return a custom struct or handle it in binding.
            // Let's define a specific getstate for binding.
    >;
    
    // Public members
    std::string seq_id;
    SequenceStatus status = SequenceStatus::WAITING;
    std::vector<int> token_ids;
    int last_token;
    int num_tokens;
    int num_prompt_tokens;
    int num_checkpointed_tokens;
    int num_cached_tokens = 0;
    
    std::optional<std::string> backup_engine_id;
    std::optional<std::string> active_engine_id;
    std::unordered_map<std::optional<std::string>, BlockContext, OptionalStringHash> block_ctx_map;
    
    std::shared_ptr<SequenceMetric> metric;
    
    double temperature;
    int max_tokens;
    bool ignore_eos;
    
private:
    static std::string generate_uuid();
};

} // namespace nanodeploy
