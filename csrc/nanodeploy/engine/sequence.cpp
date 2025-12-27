#include <algorithm>
#include <memory>
#include <stdexcept>

#include "nanodeploy/metrics/sequence_metric.h"

#include "sequence.h"

namespace nanodeploy {

// BlockContext Implementation

BlockContext::BlockContext(const std::string& engine_id, int attention_sp, int attention_dp)
{
    reset(engine_id, attention_sp, attention_dp);
}

void BlockContext::reset(const std::string& engine_id, int attention_sp, int attention_dp)
{
    engine_id_     = engine_id;
    dp_idx_        = 0;
    master_sp_idx_ = 0;
    attention_sp_  = attention_sp;
    attention_dp_  = attention_dp;
    for (int i = 0; i < attention_sp; ++i) {
        (void)sp_block_table[i];
    }
    num_dispatched_tokens.resize(attention_sp, 0);
}

std::tuple<std::string,
           int,
           int,
           int,
           int,
           std::vector<std::pair<int, int>>,
           std::unordered_map<int, std::vector<int>>,
           std::vector<int>>
BlockContext::getstate() const
{
    std::unordered_map<int, std::vector<int>> sp_block_table_state;
    sp_block_table_state.reserve(sp_block_table.size());
    for (const auto& pair : sp_block_table) {
        sp_block_table_state.emplace(pair.first, std::vector<int>(pair.second.begin(), pair.second.end()));
    }

    return std::make_tuple(engine_id_,
                           dp_idx_,
                           master_sp_idx_,
                           attention_sp_,
                           attention_dp_,
                           std::vector<std::pair<int, int>>(block_location.begin(), block_location.end()),
                           std::move(sp_block_table_state),
                           num_dispatched_tokens);
}

BlockContext BlockContext::setstate(const std::tuple<std::string,
                                                     int,
                                                     int,
                                                     int,
                                                     int,
                                                     std::vector<std::pair<int, int>>,
                                                     std::unordered_map<int, std::vector<int>>,
                                                     std::vector<int>>& state)
{
    BlockContext ctx;
    ctx.engine_id_     = std::get<0>(state);
    ctx.dp_idx_        = std::get<1>(state);
    ctx.master_sp_idx_ = std::get<2>(state);
    ctx.attention_sp_  = std::get<3>(state);
    ctx.attention_dp_  = std::get<4>(state);
    ctx.block_location = BlockContext::BlockLocationList(std::get<5>(state).begin(), std::get<5>(state).end());
    ctx.sp_block_table.clear();
    for (const auto& pair : std::get<6>(state)) {
        ctx.sp_block_table[pair.first] = BlockContext::BlockIdList(pair.second.begin(), pair.second.end());
    }
    ctx.num_dispatched_tokens = std::get<7>(state);
    return ctx;
}

// Sequence Implementation

std::atomic<uint64_t> Sequence::next_seq_id_{0};

Sequence::Sequence(const std::vector<int>& token_ids, double temperature, int max_tokens, bool ignore_eos):
    token_ids(token_ids), temperature(temperature), max_tokens(max_tokens), ignore_eos(ignore_eos)
{
    this->token_ids.reserve(max_tokens);
    this->token_ids = token_ids;

    seq_id     = next_seq_id_.fetch_add(1);
    status     = SequenceStatus::WAITING;
    num_tokens = static_cast<int>(token_ids.size());
    if (!token_ids.empty()) {
        last_token = token_ids.back();
    }
    else {
        last_token = -1;  // Should not happen based on usage
    }
    num_prompt_tokens       = num_tokens;
    num_checkpointed_tokens = num_tokens;
    num_cached_tokens       = 0;
}

BlockContext& Sequence::block_ctx(BlockContextSlot slot)
{
    return slots_[(size_t)slot];
}

const BlockContext& Sequence::block_ctx(BlockContextSlot slot) const
{
    return slots_[(size_t)slot];
}

int Sequence::dp_idx(BlockContextSlot slot)
{
    return block_ctx(slot).dp_idx_;
}

BlockContext::BlockIdList& Sequence::block_table(BlockContextSlot slot, int sp_idx)
{
    return block_ctx(slot).sp_block_table[sp_idx];
}

int Sequence::context_len(BlockContextSlot slot, std::optional<int> sp_idx)
{
    auto& ctx = block_ctx(slot);
    int   idx = sp_idx.has_value() ? sp_idx.value() : ctx.master_sp_idx_;
    return ctx.num_dispatched_tokens[idx];
}

void Sequence::append_token(int token_id, BlockContextSlot slot, std::optional<int> sp_idx)
{
    auto& ctx = block_ctx(slot);
    int   idx = sp_idx.has_value() ? sp_idx.value() : ctx.master_sp_idx_;

    token_ids.push_back(token_id);
    last_token = token_id;
    num_tokens++;
    ctx.num_dispatched_tokens[idx]++;
}

int Sequence::num_blocks(BlockContextSlot slot, int sp_idx)
{
    int n_tokens = block_ctx(slot).num_dispatched_tokens[sp_idx];
    return (n_tokens + block_size - 1) / block_size;
}

int Sequence::last_block_page_id(BlockContextSlot slot, int sp_idx)
{
    int   n_tokens       = block_ctx(slot).num_dispatched_tokens[sp_idx];
    int   last_block_idx = (n_tokens - 1) / block_size;
    auto& table          = block_table(slot, sp_idx);
    if (last_block_idx >= static_cast<int>(table.size())) {
        throw std::out_of_range("Block index out of range");
    }
    return table[last_block_idx];
}

int Sequence::last_block_num_tokens(BlockContextSlot slot, int sp_idx)
{
    int n_tokens = block_ctx(slot).num_dispatched_tokens[sp_idx];
    return n_tokens - (num_blocks(slot, sp_idx) - 1) * block_size;
}

std::pair<const int*, size_t> Sequence::block_view(int i, BlockContextSlot slot, int sp_idx) const
{
    int n_blocks = const_cast<Sequence*>(this)->num_blocks(slot, sp_idx);
    if (i < 0 || i >= n_blocks) {
        throw std::out_of_range("Block index out of range");
    }

    int start = i * block_size;
    int end   = std::min((i + 1) * block_size, static_cast<int>(token_ids.size()));

    if (start >= static_cast<int>(token_ids.size())) {
        return {nullptr, 0};
    }
    return {&token_ids[start], static_cast<size_t>(end - start)};
}

std::vector<int> Sequence::block(int i, BlockContextSlot slot, int sp_idx)
{
    auto view = block_view(i, slot, sp_idx);
    if (view.second == 0)
        return {};
    return std::vector<int>(view.first, view.first + view.second);
}

std::vector<int> Sequence::prompt_token_ids() const
{
    if (num_prompt_tokens > static_cast<int>(token_ids.size()))
        return token_ids;
    return std::vector<int>(token_ids.begin(), token_ids.begin() + num_prompt_tokens);
}

std::vector<int> Sequence::completion_token_ids() const
{
    if (num_prompt_tokens >= static_cast<int>(token_ids.size()))
        return {};
    return std::vector<int>(token_ids.begin() + num_prompt_tokens, token_ids.end());
}

}  // namespace nanodeploy
