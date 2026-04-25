#include <stdexcept>

#include "nanodeploy/csrc/sequence/sequence.h"

#include "compressed_block_manager.h"

namespace nanodeploy {

CompressedBlockManager::CompressedBlockManager(
    const std::string& engine_id, int ratio, int num_pages, int page_size, int max_blocks_per_seq):
    engine_id_(engine_id),
    ratio_(ratio),
    num_pages_(num_pages),
    page_size_(page_size),
    max_blocks_per_seq_(max_blocks_per_seq)
{
    in_use_.assign(num_pages, 0);
    free_pages_.reserve(num_pages);
    // Push in reverse so that pop_back yields IDs in ascending order — handy
    // for debugging / determinism, no functional difference.
    for (int i = num_pages - 1; i >= 0; --i) {
        free_pages_.push_back(i);
    }
}

int CompressedBlockManager::allocate_page()
{
    if (free_pages_.empty()) {
        throw std::runtime_error("CompressedBlockManager: no free pages in pool (ratio=" + std::to_string(ratio_)
                                 + ")");
    }
    int id = free_pages_.back();
    free_pages_.pop_back();
    in_use_[id] = 1;
    return id;
}

void CompressedBlockManager::deallocate_page(int id)
{
    if (id < 0 || id >= num_pages_ || !in_use_[id]) {
        // Idempotent: deallocating an already-free or out-of-range page is a
        // no-op rather than an error, since deallocate(seq, slot) may be
        // called multiple times during migration / abort paths.
        return;
    }
    in_use_[id] = 0;
    free_pages_.push_back(id);
}

bool CompressedBlockManager::can_allocate(int num_blocks) const
{
    return static_cast<int>(free_pages_.size()) >= num_blocks;
}

void CompressedBlockManager::allocate(Sequence& seq, int num_blocks)
{
    auto& existing = seq.compressed_block_table(BlockContextSlot::ACTIVE, ratio_);
    if (!existing.empty()) {
        throw std::runtime_error("Sequence already has compressed blocks allocated for ratio="
                                 + std::to_string(ratio_));
    }
    if (num_blocks > max_blocks_per_seq_) {
        throw std::runtime_error("CompressedBlockManager::allocate: num_blocks " + std::to_string(num_blocks)
                                 + " exceeds max_blocks_per_seq " + std::to_string(max_blocks_per_seq_));
    }
    if (!can_allocate(num_blocks)) {
        throw std::runtime_error("CompressedBlockManager: pool exhausted");
    }

    std::vector<int> ids;
    ids.reserve(num_blocks);
    for (int i = 0; i < num_blocks; ++i) {
        ids.push_back(allocate_page());
    }
    seq.set_compressed_block_table(BlockContextSlot::ACTIVE, ratio_, std::move(ids));
}

bool CompressedBlockManager::may_append(Sequence& seq, int additional_blocks)
{
    if (additional_blocks <= 0)
        return true;
    auto& existing = seq.compressed_block_table(BlockContextSlot::ACTIVE, ratio_);
    int   new_size = static_cast<int>(existing.size()) + additional_blocks;
    if (new_size > max_blocks_per_seq_)
        return false;
    if (!can_allocate(additional_blocks))
        return false;

    std::vector<int> ids = existing;  // copy
    ids.reserve(new_size);
    for (int i = 0; i < additional_blocks; ++i) {
        ids.push_back(allocate_page());
    }
    seq.set_compressed_block_table(BlockContextSlot::ACTIVE, ratio_, std::move(ids));
    return true;
}

void CompressedBlockManager::deallocate(Sequence& seq, BlockContextSlot slot)
{
    auto& ids = seq.compressed_block_table(slot, ratio_);
    if (ids.empty())
        return;
    for (int id : ids) {
        deallocate_page(id);
    }
    seq.set_compressed_block_table(slot, ratio_, std::vector<int>{});
}

}  // namespace nanodeploy
