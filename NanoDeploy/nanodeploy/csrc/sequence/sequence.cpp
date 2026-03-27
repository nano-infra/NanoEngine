#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>

#include "nanodeploy/csrc/metrics/sequence_metric.h"
#include "sequence.h"
#include "sequence_generated.h"

namespace nanodeploy {

// BlockContext free functions

std::unique_ptr<BlockContext>
make_block_context(const std::string& engine_id, int group_size, int attention_dp, int num_kvcache_blocks)
{
    auto ctx = std::make_unique<BlockContext>();
    reset_block_context(*ctx, engine_id, group_size, attention_dp, num_kvcache_blocks);
    return ctx;
}

void reset_block_context(
    BlockContext& ctx, const std::string& engine_id, int group_size, int attention_dp, int num_kvcache_blocks)
{
    ctx.engine_id = engine_id;

    ctx.dp_idx          = 0;
    ctx.master_group_id = 0;
    ctx.group_size      = group_size;
    ctx.attention_dp    = attention_dp;

    ctx.num_kvcache_blocks = num_kvcache_blocks;
    ctx.state_slot         = -1;

    // Initialize group_block_table and num_dispatched_tokens
    ctx.group_block_table.clear();
    ctx.group_block_table.resize(group_size);
    for (auto& list : ctx.group_block_table) {
        list = std::make_unique<fbs::IntListT>();
    }
    ctx.num_dispatched_tokens.clear();
    ctx.num_dispatched_tokens.resize(group_size, 0);
    ctx.block_location.clear();
    ctx.endpoints.clear();  // Initialize endpoints to empty vector
}

int                   Sequence::block_size = 256;
std::atomic<uint64_t> Sequence::next_seq_id_{0};

Sequence::Sequence(const std::vector<int>& token_ids, const SamplingParams& sampling_params)
{
    data_ = std::make_unique<SequenceT>();

    data_->token_ids = token_ids;
    data_->token_ids.reserve(sampling_params.max_tokens);

    data_->seq_id     = next_seq_id_.fetch_add(1);
    data_->status     = SequenceStatus::WAITING;
    data_->num_tokens = static_cast<int>(token_ids.size());
    if (!token_ids.empty()) {
        data_->last_token = token_ids.back();
    }
    else {
        data_->last_token = -1;  // Should not happen based on usage
    }
    data_->num_prompt_tokens       = data_->num_tokens;
    data_->num_checkpointed_tokens = data_->num_tokens;
    data_->num_cached_tokens       = 0;

    // Initialize sampling_params
    data_->sampling_params  = std::make_unique<SamplingParamsT>();
    *data_->sampling_params = sampling_params;

    // Initialize slots - pre-allocate all slots to avoid nulls
    data_->slots.resize((size_t)BlockContextSlot::_COUNT);
    for (size_t i = 0; i < (size_t)BlockContextSlot::_COUNT; ++i) {
        auto ctx = std::make_unique<BlockContext>();
        // Initialize all fields to safe defaults
        ctx->engine_id          = "";
        ctx->dp_idx             = 0;
        ctx->master_group_id    = 0;
        ctx->group_size         = 0;
        ctx->attention_dp       = 0;
        ctx->num_kvcache_blocks = 0;
        ctx->state_slot         = -1;
        ctx->block_location.clear();
        ctx->num_dispatched_tokens.clear();
        ctx->group_block_table.clear();
        ctx->endpoints.clear();
        data_->slots[i] = std::move(ctx);
    }
}

std::shared_ptr<Sequence> Sequence::from_data(std::unique_ptr<SequenceT> data)
{
    auto seq   = std::make_shared<Sequence>(std::vector<int>{}, SamplingParams());
    seq->data_ = std::move(data);

    // Ensure all slots are allocated (handle deserialized data that may have nulls)
    if (seq->data_->slots.size() < (size_t)BlockContextSlot::_COUNT) {
        seq->data_->slots.resize((size_t)BlockContextSlot::_COUNT);
    }
    for (size_t i = 0; i < (size_t)BlockContextSlot::_COUNT; ++i) {
        if (!seq->data_->slots[i]) {
            auto ctx = std::make_unique<BlockContext>();
            // Initialize all fields to safe defaults
            ctx->engine_id          = "";
            ctx->dp_idx             = 0;
            ctx->master_group_id    = 0;
            ctx->group_size         = 0;
            ctx->attention_dp       = 0;
            ctx->num_kvcache_blocks = 0;
            ctx->state_slot         = -1;
            ctx->block_location.clear();
            ctx->num_dispatched_tokens.clear();
            ctx->group_block_table.clear();
            ctx->endpoints.clear();
            seq->data_->slots[i] = std::move(ctx);
        }
    }

    return seq;
}

BlockContext& Sequence::block_ctx(BlockContextSlot slot)
{
    ensure_slot(slot);
    return *get_slot(slot);
}

const BlockContext& Sequence::block_ctx(BlockContextSlot slot) const
{
    ensure_slot(slot);
    return *get_slot(slot);
}

int Sequence::dp_idx(BlockContextSlot slot)
{
    return block_ctx(slot).dp_idx;
}

int Sequence::state_slot(BlockContextSlot slot) const
{
    return block_ctx(slot).state_slot;
}

void Sequence::set_state_slot(BlockContextSlot slot, int state_slot)
{
    block_ctx(slot).state_slot = state_slot;
}

std::vector<int>& Sequence::block_table(BlockContextSlot slot, int group_id)
{
    auto& ctx = block_ctx(slot);
    if (group_id >= static_cast<int>(ctx.group_block_table.size())) {
        size_t old_size = ctx.group_block_table.size();
        ctx.group_block_table.resize(group_id + 1);
        // Initialize all new elements to avoid null pointers
        for (size_t i = old_size; i < ctx.group_block_table.size(); ++i) {
            ctx.group_block_table[i] = std::make_unique<fbs::IntListT>();
        }
    }
    if (!ctx.group_block_table[group_id]) {
        ctx.group_block_table[group_id] = std::make_unique<fbs::IntListT>();
    }
    // Return a reference to the values vector directly
    return ctx.group_block_table[group_id]->values;
}

int Sequence::context_len(BlockContextSlot slot, std::optional<int> group_id)
{
    auto& ctx = block_ctx(slot);
    int   idx = group_id.has_value() ? group_id.value() : ctx.master_group_id;
    if (idx >= static_cast<int>(ctx.num_dispatched_tokens.size())) {
        return 0;
    }
    return ctx.num_dispatched_tokens[idx];
}

void Sequence::append_token(int token_id, BlockContextSlot slot, std::optional<int> group_id)
{
    auto& ctx = block_ctx(slot);
    int   idx = group_id.has_value() ? group_id.value() : ctx.master_group_id;

    data_->token_ids.push_back(token_id);
    data_->last_token = token_id;
    data_->num_tokens++;
    if (idx >= static_cast<int>(ctx.num_dispatched_tokens.size())) {
        ctx.num_dispatched_tokens.resize(idx + 1, 0);
    }
    ctx.num_dispatched_tokens[idx]++;
}

int Sequence::num_blocks(BlockContextSlot slot, int group_id)
{
    auto& ctx = block_ctx(slot);
    if (group_id >= static_cast<int>(ctx.num_dispatched_tokens.size())) {
        return 0;
    }
    int n_tokens = ctx.num_dispatched_tokens[group_id];
    return (n_tokens + block_size - 1) / block_size;
}

int Sequence::last_block_page_id(BlockContextSlot slot, int group_id)
{
    auto& ctx = block_ctx(slot);
    if (group_id >= static_cast<int>(ctx.num_dispatched_tokens.size())) {
        throw std::out_of_range("SP index out of range (last_block_page_id): seq_id=" + std::to_string(data_->seq_id)
                                + " group_id=" + std::to_string(group_id)
                                + " num_dispatched_tokens.size()=" + std::to_string(ctx.num_dispatched_tokens.size()));
    }
    int n_tokens       = ctx.num_dispatched_tokens[group_id];
    int last_block_idx = (n_tokens - 1) / block_size;

    if (group_id >= static_cast<int>(ctx.group_block_table.size()) || !ctx.group_block_table[group_id]
        || last_block_idx >= static_cast<int>(ctx.group_block_table[group_id]->values.size())) {
        int table_size  = static_cast<int>(ctx.group_block_table.size());
        int values_size = (group_id >= 0 && group_id < table_size && ctx.group_block_table[group_id]) ?
                              static_cast<int>(ctx.group_block_table[group_id]->values.size()) :
                              -1;
        throw std::out_of_range("Block index out of range (last_block_page_id): seq_id=" + std::to_string(data_->seq_id)
                                + " group_id=" + std::to_string(group_id) + " last_block_idx="
                                + std::to_string(last_block_idx) + " num_dispatched_tokens=" + std::to_string(n_tokens)
                                + " group_block_table.size()=" + std::to_string(table_size)
                                + " group_block_table[group_id].values.size()=" + std::to_string(values_size));
    }
    return ctx.group_block_table[group_id]->values[last_block_idx];
}

int Sequence::last_block_num_tokens(BlockContextSlot slot, int group_id)
{
    auto& ctx = block_ctx(slot);
    if (group_id >= static_cast<int>(ctx.num_dispatched_tokens.size())) {
        return 0;
    }
    int n_tokens = ctx.num_dispatched_tokens[group_id];
    return n_tokens - (num_blocks(slot, group_id) - 1) * block_size;
}

std::pair<const int*, size_t> Sequence::block_view(int i, BlockContextSlot slot, int group_id) const
{
    int n_blocks = const_cast<Sequence*>(this)->num_blocks(slot, group_id);
    if (i < 0 || i >= n_blocks) {
        throw std::out_of_range("Block index out of range (block_view): seq_id=" + std::to_string(data_->seq_id)
                                + " slot=" + std::to_string(static_cast<int>(slot))
                                + " group_id=" + std::to_string(group_id) + " block_index=" + std::to_string(i)
                                + " n_blocks=" + std::to_string(n_blocks));
    }

    int start = i * block_size;
    int end   = std::min((i + 1) * block_size, static_cast<int>(data_->token_ids.size()));

    if (start >= static_cast<int>(data_->token_ids.size())) {
        return {nullptr, 0};
    }
    return {&data_->token_ids[start], static_cast<size_t>(end - start)};
}

std::vector<int> Sequence::block(int i, BlockContextSlot slot, int group_id)
{
    auto view = block_view(i, slot, group_id);
    if (view.second == 0)
        return {};
    return std::vector<int>(view.first, view.first + view.second);
}

std::vector<int> Sequence::prompt_token_ids() const
{
    if (data_->num_prompt_tokens > static_cast<int>(data_->token_ids.size()))
        return data_->token_ids;
    return std::vector<int>(data_->token_ids.begin(), data_->token_ids.begin() + data_->num_prompt_tokens);
}

std::vector<int> Sequence::completion_token_ids() const
{
    if (data_->num_prompt_tokens >= static_cast<int>(data_->token_ids.size()))
        return {};
    return std::vector<int>(data_->token_ids.begin() + data_->num_prompt_tokens, data_->token_ids.end());
}

}  // namespace nanodeploy
