#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "nanodeploy/csrc/common/xxhash.hpp"
#include "nanodeploy/csrc/sequence/sequence.h"

#include "sequence_generated.h"

#include "block_manager.h"

namespace nanodeploy {

BlockManager::BlockManager(const std::string& engine_id, int group_id, int num_blocks, int block_size):
    engine_id_(engine_id), group_id_(group_id), block_size_(block_size)
{

    blocks_.reserve(num_blocks);
    block_id_to_free_list_it_.resize(num_blocks);
    for (int i = 0; i < num_blocks; ++i) {
        blocks_.emplace_back(i, block_size);
        free_block_ids_.push_back(i);
        block_id_to_free_list_it_[i] = std::prev(free_block_ids_.end());
    }
}

int64_t BlockManager::compute_hash(const std::vector<int>& token_ids, int64_t prefix)
{
    return compute_hash(token_ids.data(), token_ids.size(), prefix);
}

int64_t BlockManager::compute_hash(const int* token_ids, size_t size, int64_t prefix)
{
    xxh::hash_state64_t state;
    if (prefix != -1) {
        state.update(&prefix, sizeof(prefix));
    }
    state.update(token_ids, size * sizeof(int));
    return static_cast<int64_t>(state.digest());
}

Block& BlockManager::allocate_block(int block_id)
{
    Block& block = blocks_[block_id];
    if (block.ref_count != 0) {
        throw std::runtime_error("Block ref_count is not 0");
    }
    block.reset();

    auto it = block_id_to_free_list_it_[block_id];
    if (it != free_block_ids_.end()) {
        free_block_ids_.erase(it);
        block_id_to_free_list_it_[block_id] = free_block_ids_.end();
    }

    used_block_ids_.insert(block_id);
    return blocks_[block_id];
}

void BlockManager::deallocate_block(int block_id)
{
    if (blocks_[block_id].ref_count != 0) {
        throw std::runtime_error("Block ref_count is not 0");
    }
    used_block_ids_.erase(block_id);
    free_block_ids_.push_back(block_id);
    block_id_to_free_list_it_[block_id] = std::prev(free_block_ids_.end());
}

int BlockManager::count_active_prefix_hits(Sequence& seq) const
{
    int64_t h          = -1;
    int     num_blocks = seq.num_blocks(BlockContextSlot::ACTIVE, group_id_);
    int     hits       = 0;

    for (int i = 0; i < num_blocks; ++i) {
        auto view = seq.block_view(i, BlockContextSlot::ACTIVE, group_id_);

        // Only full blocks can be cached
        if (view.second != static_cast<size_t>(block_size_)) {
            break;
        }
        h = compute_hash(view.first, view.second, h);

        auto it = hash_to_block_id_.find(h);
        if (it == hash_to_block_id_.end()) {
            break;
        }
        int block_id = it->second;
        // Must be actively shared (ref_count > 0) and token content must match
        if (!used_block_ids_.count(block_id)) {
            break;
        }
        const Block& blk = blocks_[block_id];
        if (blk.token_ids.size() != view.second
            || !std::equal(blk.token_ids.begin(), blk.token_ids.end(), view.first)) {
            break;
        }
        hits++;
    }
    return hits;
}

int BlockManager::can_allocate(Sequence& seq) const
{
    int n_cached      = count_active_prefix_hits(seq);
    int blocks_needed = seq.num_blocks(BlockContextSlot::ACTIVE, group_id_) - n_cached;
    if (static_cast<int>(free_block_ids_.size()) >= blocks_needed)
        return n_cached;  // success: return hit count
    return -1;            // cannot allocate
}

void BlockManager::allocate(Sequence& seq, int prefix_hint)
{
    auto& table = seq.block_table(BlockContextSlot::ACTIVE, group_id_);
    if (!table.empty()) {
        throw std::runtime_error("Block table is not empty");
    }

    int64_t h                 = -1;
    bool    cache_miss        = false;
    int     num_blocks        = seq.num_blocks(BlockContextSlot::ACTIVE, group_id_);
    int     num_prefix_cached = 0;  // consecutive leading full-block cache hits

    // If can_allocate already computed the prefix hit count we can trust it
    // for the leading `prefix_hint` full blocks — they are guaranteed to be
    // cache hits.  We still need to walk those blocks to build the hash chain
    // and populate the block table, but we can skip the per-block content
    // comparison for the first `prefix_hint` blocks.
    int fast_prefix = (prefix_hint > 0) ? prefix_hint : 0;

    for (int i = 0; i < num_blocks; ++i) {
        auto view = seq.block_view(i, BlockContextSlot::ACTIVE, group_id_);

        if (view.second == static_cast<size_t>(block_size_)) {
            h = compute_hash(view.first, view.second, h);
        }
        else {
            h = -1;
        }

        int block_id = -1;
        if (h != -1 && hash_to_block_id_.count(h)) {
            block_id = hash_to_block_id_.at(h);
        }

        // For the first `fast_prefix` blocks we know the content matches
        // (validated by count_active_prefix_hits in can_allocate).  Skip
        // the expensive element-wise comparison.
        if (!cache_miss && i < fast_prefix) {
            // Guaranteed hit — skip content check
        }
        else if (block_id == -1 || blocks_[block_id].token_ids.size() != view.second
                 || !std::equal(blocks_[block_id].token_ids.begin(), blocks_[block_id].token_ids.end(), view.first)) {
            cache_miss = true;
        }

        Block* block_ptr = nullptr;
        if (cache_miss) {
            if (free_block_ids_.empty()) {
                throw std::runtime_error("No free blocks available");
            }
            block_id  = free_block_ids_.front();
            block_ptr = &allocate_block(block_id);
        }
        else {
            if (used_block_ids_.count(block_id)) {
                block_ptr = &blocks_[block_id];
                block_ptr->ref_count++;
            }
            else {
                block_ptr = &allocate_block(block_id);
            }
            // Track consecutive leading full-block hits (prefix caching)
            if (view.second == static_cast<size_t>(block_size_)) {
                num_prefix_cached++;
            }
        }

        if (h != -1) {
            block_ptr->update(h, view.first, view.second);
            hash_to_block_id_[h] = block_id;
        }

        seq.block_ctx(BlockContextSlot::ACTIVE).block_location.emplace_back(group_id_, block_id);
        table.push_back(block_id);
    }

    // Wire prefix caching: tell the sequence how many leading tokens are already
    // in the shared KV cache so prefill can skip recomputing them.
    // Cap at num_tokens - 1 to guarantee seqlen_q >= 1: the model always needs
    // at least one Q token to produce logits for sampling.
    int cached_tokens = num_prefix_cached * block_size_;
    int max_cached    = std::max(0, seq.num_tokens() - 1);
    seq.set_num_cached_tokens(std::min(cached_tokens, max_cached));
}

void BlockManager::deallocate(Sequence& seq, BlockContextSlot slot)
{
    auto& table = seq.block_table(slot, group_id_);
    // Iterate in reverse
    for (auto it = table.rbegin(); it != table.rend(); ++it) {
        int    block_id = *it;
        Block& block    = blocks_[block_id];
        block.ref_count--;
        if (block.ref_count == 0) {
            deallocate_block(block_id);
        }
    }
    seq.set_num_cached_tokens(0);
    table.clear();
}

bool BlockManager::can_append(Sequence& seq, int num_tokens) const
{
    int num_dispatched = seq.block_ctx(BlockContextSlot::ACTIVE).num_dispatched_tokens[group_id_];
    int blocks_needed  = (num_dispatched + num_tokens + block_size_ - 1) / block_size_;
    int current_blocks = static_cast<int>(seq.block_table(BlockContextSlot::ACTIVE, group_id_).size());
    int additional     = blocks_needed - current_blocks;
    if (additional < 0)
        additional = 0;

    return static_cast<int>(free_block_ids_.size()) >= additional;
}

bool BlockManager::may_append(Sequence& seq, int num_tokens)
{
    int   current_dispatched = seq.block_ctx(BlockContextSlot::ACTIVE).num_dispatched_tokens[group_id_];
    auto& table              = seq.block_table(BlockContextSlot::ACTIVE, group_id_);
    int   current_blocks     = static_cast<int>(table.size());
    int   blocks_needed      = (current_dispatched + num_tokens + block_size_ - 1) / block_size_;

    while (current_blocks < blocks_needed) {
        if (free_block_ids_.empty()) {
            return false;
        }
        int block_id = free_block_ids_.front();
        seq.block_ctx(BlockContextSlot::ACTIVE).block_location.emplace_back(group_id_, block_id);
        allocate_block(block_id);
        table.push_back(block_id);
        current_blocks++;
    }
    return true;
}

}  // namespace nanodeploy
