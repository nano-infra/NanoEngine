#include "sequence.h"
#include "sequence_metric.h"
#include <stdexcept>
#include <algorithm>
#include <memory>


namespace nanodeploy {

// BlockContext Implementation

BlockContext::BlockContext(const std::optional<std::string>& engine_id,
                           int dp_idx, int master_sp_idx,
                           int attention_sp, int attention_dp)
    : engine_id(engine_id), dp_idx(dp_idx), master_sp_idx(master_sp_idx),
      attention_sp(attention_sp), attention_dp(attention_dp) {}

std::tuple<std::optional<std::string>, int, int, int, int, 
           std::vector<std::pair<int, int>>,
           std::unordered_map<int, std::vector<int>>,
           std::unordered_map<int, int>> BlockContext::getstate() const {
    return std::make_tuple(engine_id, dp_idx, master_sp_idx, attention_sp, attention_dp,
                           block_location, sp_block_table, num_dispatched_tokens);
}

BlockContext BlockContext::setstate(const std::tuple<std::optional<std::string>, int, int, int, int, 
           std::vector<std::pair<int, int>>,
           std::unordered_map<int, std::vector<int>>,
           std::unordered_map<int, int>>& state) {
    BlockContext ctx;
    ctx.engine_id = std::get<0>(state);
    ctx.dp_idx = std::get<1>(state);
    ctx.master_sp_idx = std::get<2>(state);
    ctx.attention_sp = std::get<3>(state);
    ctx.attention_dp = std::get<4>(state);
    ctx.block_location = std::get<5>(state);
    ctx.sp_block_table = std::get<6>(state);
    ctx.num_dispatched_tokens = std::get<7>(state);
    return ctx;
}

// Sequence Implementation

std::string Sequence::generate_uuid() {
    // Simple UUID generation (not RFC compliant but sufficient for unique ID)
    // Use thread_local to avoid data races in multi-threaded environment
    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    thread_local std::uniform_int_distribution<> dis(0, 15);
    thread_local std::uniform_int_distribution<> dis2(8, 11);

    std::stringstream ss;
    ss << std::hex;
    for (int i = 0; i < 8; i++) ss << dis(gen);
    ss << "-";
    for (int i = 0; i < 4; i++) ss << dis(gen);
    ss << "-4"; // UUID version 4
    for (int i = 0; i < 3; i++) ss << dis(gen);
    ss << "-";
    ss << dis2(gen); // UUID variant
    for (int i = 0; i < 3; i++) ss << dis(gen);
    ss << "-";
    for (int i = 0; i < 12; i++) ss << dis(gen);
    return ss.str();
}

Sequence::Sequence(const std::vector<int>& token_ids,
                   double temperature,
                   int max_tokens,
                   bool ignore_eos,
                   const std::optional<std::string>& engine_id,
                   int master_sp_rank)
        : token_ids(token_ids),
            backup_engine_id(engine_id),
            active_engine_id(engine_id),
            temperature(temperature),
            max_tokens(max_tokens),
            ignore_eos(ignore_eos) {
    
    seq_id = generate_uuid();
    status = SequenceStatus::WAITING;
    num_tokens = static_cast<int>(token_ids.size());
    if (!token_ids.empty()) {
        last_token = token_ids.back();
    } else {
        last_token = -1; // Should not happen based on usage
    }
    num_prompt_tokens = num_tokens;
    num_checkpointed_tokens = num_tokens;
    num_cached_tokens = 0;

    BlockContext ctx(engine_id, -1, master_sp_rank, 1, 1);
    // Initialize sp_block_table and num_dispatched_tokens defaults
    // Python: sp_block_table=defaultdict(list), num_dispatched_tokens=defaultdict(int)
    // In C++, we just leave them empty, map[] will create default constructed value (empty vector/0)
    
    block_ctx_map[engine_id] = ctx;
}

BlockContext& Sequence::block_ctx(const std::optional<std::string>& engine_id) {
    auto eid = engine_id.has_value() ? engine_id : active_engine_id;
    auto it = block_ctx_map.find(eid);
    if (it == block_ctx_map.end()) {
        throw std::runtime_error("BlockContext not found for engine_id");
    }
    return it->second;
}

const BlockContext& Sequence::block_ctx(const std::optional<std::string>& engine_id) const {
    auto eid = engine_id.has_value() ? engine_id : active_engine_id;
    auto it = block_ctx_map.find(eid);
    if (it == block_ctx_map.end()) {
        throw std::runtime_error("BlockContext not found for engine_id");
    }
    return it->second;
}

int Sequence::dp_idx(const std::optional<std::string>& engine_id) {
    return block_ctx(engine_id).dp_idx;
}

std::vector<int>& Sequence::block_table(const std::optional<std::string>& engine_id, int sp_idx) {
    return block_ctx(engine_id).sp_block_table[sp_idx];
}

void Sequence::set_engine_id(const std::string& engine_id, int attention_dp, int attention_sp) {
    active_engine_id = engine_id;
    if (block_ctx_map.find(engine_id) != block_ctx_map.end()) {
        return;
    }
    
    BlockContext ctx(engine_id, -1, 0, attention_sp, attention_dp);
    // Initialize sp_block_table and num_dispatched_tokens for range(attention_sp)
    // This mimics Python's defaultdict behavior
    for (int i = 0; i < attention_sp; ++i) {
        ctx.sp_block_table[i] = {};
        ctx.num_dispatched_tokens[i] = 0;
    }
    block_ctx_map[engine_id] = ctx;
}

int Sequence::context_len(const std::optional<std::string>& engine_id, std::optional<int> sp_idx) {
    auto& ctx = block_ctx(engine_id);
    int idx = sp_idx.has_value() ? sp_idx.value() : ctx.master_sp_idx;
    return ctx.num_dispatched_tokens[idx];
}

void Sequence::append_token(int token_id, const std::optional<std::string>& engine_id, std::optional<int> sp_idx) {
    auto eid = engine_id.has_value() ? engine_id : active_engine_id;
    auto& ctx = block_ctx(eid);
    int idx = sp_idx.has_value() ? sp_idx.value() : ctx.master_sp_idx;
    
    token_ids.push_back(token_id);
    last_token = token_id;
    num_tokens++;
    ctx.num_dispatched_tokens[idx]++;
}

void Sequence::block_table_append(int block_id,
                                 const std::optional<std::string>& engine_id,
                                 int sp_idx) {
    block_table(engine_id, sp_idx).push_back(block_id);
}

void Sequence::block_table_clear(const std::optional<std::string>& engine_id, int sp_idx) {
    block_table(engine_id, sp_idx).clear();
}

void Sequence::block_table_set(const std::vector<int>& table,
                              const std::optional<std::string>& engine_id,
                              int sp_idx) {
    block_ctx(engine_id).sp_block_table[sp_idx] = table;
}

void Sequence::block_location_append(int sp_idx,
                                     int block_id,
                                     const std::optional<std::string>& engine_id) {
    block_ctx(engine_id).block_location.emplace_back(sp_idx, block_id);
}

void Sequence::block_location_clear(const std::optional<std::string>& engine_id) {
    block_ctx(engine_id).block_location.clear();
}

void Sequence::sp_block_table_clear(const std::optional<std::string>& engine_id) {
    block_ctx(engine_id).sp_block_table.clear();
}

void Sequence::num_dispatched_tokens_clear(const std::optional<std::string>& engine_id) {
    block_ctx(engine_id).num_dispatched_tokens.clear();
}

int Sequence::num_blocks(const std::optional<std::string>& engine_id, int sp_idx) {
    int n_tokens = block_ctx(engine_id).num_dispatched_tokens[sp_idx];
    return (n_tokens + block_size - 1) / block_size;
}

int Sequence::last_block_page_id(const std::optional<std::string>& engine_id, int sp_idx) {
    int n_tokens = block_ctx(engine_id).num_dispatched_tokens[sp_idx];
    int last_block_idx = (n_tokens - 1) / block_size;
    auto& table = block_table(engine_id, sp_idx);
    if (last_block_idx >= static_cast<int>(table.size())) {
         throw std::out_of_range("Block index out of range");
    }
    return table[last_block_idx];
}

int Sequence::last_block_num_tokens(const std::optional<std::string>& engine_id, int sp_idx) {
    int n_tokens = block_ctx(engine_id).num_dispatched_tokens[sp_idx];
    return n_tokens - (num_blocks(engine_id, sp_idx) - 1) * block_size;
}

std::vector<int> Sequence::block(int i, const std::optional<std::string>& engine_id, int sp_idx) {
    int n_blocks = num_blocks(engine_id, sp_idx);
    if (i < 0 || i >= n_blocks) {
        throw std::out_of_range("Block index out of range");
    }
    
    int start = i * block_size;
    int end = std::min((i + 1) * block_size, static_cast<int>(token_ids.size()));
    
    // Note: Python implementation slices token_ids. 
    // However, token_ids stores ALL tokens.
    // block() seems to return tokens for a specific block.
    // Wait, Python implementation:
    // return self.token_ids[i * self.block_size : (i + 1) * self.block_size]
    // This assumes token_ids corresponds to the blocks.
    
    if (start >= static_cast<int>(token_ids.size())) {
        return {};
    }
    return std::vector<int>(token_ids.begin() + start, token_ids.begin() + end);
}

std::vector<int> Sequence::prompt_token_ids() const {
    if (num_prompt_tokens > static_cast<int>(token_ids.size())) return token_ids;
    return std::vector<int>(token_ids.begin(), token_ids.begin() + num_prompt_tokens);
}

std::vector<int> Sequence::completion_token_ids() const {
    if (num_prompt_tokens >= static_cast<int>(token_ids.size())) return {};
    return std::vector<int>(token_ids.begin() + num_prompt_tokens, token_ids.end());
}

} // namespace nanodeploy
