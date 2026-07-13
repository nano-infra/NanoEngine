#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "nanodeploy/sequence/sequence.h"
#include "xxhash.hpp"

#include "block_manager.h"

namespace nanodeploy {

BlockManager::BlockManager(const std::string& engine_id, int sp_idx, int num_blocks, int block_size):
    engine_id_(engine_id), sp_idx_(sp_idx), block_size_(block_size)
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

bool BlockManager::can_allocate(Sequence& seq) const
{
    return static_cast<int>(free_block_ids_.size()) >= seq.num_blocks(BlockContextSlot::ACTIVE, sp_idx_);
}

void BlockManager::allocate(Sequence& seq, int token_idx_from, int token_idx_to)
{
    (void)token_idx_from;  // Unused
    (void)token_idx_to;    // Unused

    auto& table = seq.block_table(BlockContextSlot::ACTIVE, sp_idx_);
    if (!table.empty()) {
        throw std::runtime_error("Block table is not empty");
    }

    int64_t h          = -1;
    bool    cache_miss = false;
    int     num_blocks = seq.num_blocks(BlockContextSlot::ACTIVE, sp_idx_);

    for (int i = 0; i < num_blocks; ++i) {
        auto view = seq.block_view(i, BlockContextSlot::ACTIVE, sp_idx_);

        if (view.second == static_cast<size_t>(block_size_)) {
            h = compute_hash(view.first, view.second, h);
        }
        else {
            h = -1;
        }

        int block_id = -1;
        if (hash_to_block_id_.count(h)) {
            block_id = hash_to_block_id_.at(h);
        }

        if (block_id == -1 || blocks_[block_id].token_ids.size() != view.second
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
        }

        if (h != -1) {
            block_ptr->update(h, view.first, view.second);
            hash_to_block_id_[h] = block_id;
        }

        seq.block_ctx(BlockContextSlot::ACTIVE).block_location.emplace_back(sp_idx_, block_id);
        table.push_back(block_id);
    }
}

void BlockManager::allocate_uncached(Sequence& seq)
{
    auto& table = seq.block_table(BlockContextSlot::ACTIVE, sp_idx_);
    if (!table.empty()) {
        throw std::runtime_error("Block table is not empty");
    }

    int num_blocks = seq.num_blocks(BlockContextSlot::ACTIVE, sp_idx_);
    if (static_cast<int>(free_block_ids_.size()) < num_blocks) {
        throw std::runtime_error("No free blocks available");
    }
    for (int i = 0; i < num_blocks; ++i) {
        int block_id = free_block_ids_.front();
        allocate_block(block_id);
        seq.block_ctx(BlockContextSlot::ACTIVE).block_location.emplace_back(sp_idx_, block_id);
        table.push_back(block_id);
    }
}

void BlockManager::deallocate(Sequence& seq, BlockContextSlot slot)
{
    auto& table = seq.block_table(slot, sp_idx_);
    // Iterate in reverse
    for (auto it = table.rbegin(); it != table.rend(); ++it) {
        int    block_id = *it;
        Block& block    = blocks_[block_id];
        block.ref_count--;
        if (block.ref_count == 0) {
            deallocate_block(block_id);
        }
    }
    seq.num_cached_tokens = 0;
    table.clear();
}

void BlockManager::trim_blocks_to_token_count(Sequence& seq, BlockContextSlot slot, int token_count)
{
    if (token_count < 0) {
        throw std::runtime_error("token_count must be non-negative");
    }
    auto&     table       = seq.block_table(slot, sp_idx_);
    const int keep_blocks = (token_count + block_size_ - 1) / block_size_;
    if (keep_blocks > static_cast<int>(table.size())) {
        throw std::runtime_error("block table is shorter than committed token count");
    }

    auto& locations = seq.block_ctx(slot).block_location;
    while (static_cast<int>(table.size()) > keep_blocks) {
        int block_id = table.back();
        table.pop_back();

        auto location = std::find(locations.rbegin(), locations.rend(), std::make_pair(sp_idx_, block_id));
        if (location != locations.rend()) {
            locations.erase(std::next(location).base());
        }

        Block& block = blocks_[block_id];
        block.ref_count--;
        if (block.ref_count == 0) {
            deallocate_block(block_id);
        }
    }
}

bool BlockManager::can_append(Sequence& seq, int num_tokens) const
{
    int num_dispatched             = seq.block_ctx(BlockContextSlot::ACTIVE).num_dispatched_tokens[sp_idx_];
    int total_tokens_needed_before = (num_dispatched + block_size_ - 1) / block_size_;
    int total_tokens_needed_after  = (num_dispatched + num_tokens + block_size_ - 1) / block_size_;

    return static_cast<int>(free_block_ids_.size()) >= (total_tokens_needed_after - total_tokens_needed_before);
}

bool BlockManager::may_append(Sequence& seq, int num_tokens)
{
    for (int idx = 0; idx < num_tokens; ++idx) {
        auto& table = seq.block_table(BlockContextSlot::ACTIVE, sp_idx_);

        int current_dispatched = seq.block_ctx(BlockContextSlot::ACTIVE).num_dispatched_tokens[sp_idx_];

        if ((current_dispatched + idx) % block_size_ == 0) {
            if (free_block_ids_.empty()) {
                return false;
            }
            int block_id = free_block_ids_.front();
            seq.block_ctx(BlockContextSlot::ACTIVE).block_location.emplace_back(sp_idx_, block_id);
            allocate_block(block_id);
            table.push_back(block_id);
        }
        else if ((current_dispatched + idx - 1) % block_size_ == 0) {
            // Logic for updating hash of the previous block (commented out in Python)
        }
    }
    return true;
}

}  // namespace nanodeploy
