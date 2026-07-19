#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "nanodeploy/sequence/sequence.h"
#include "xxhash.hpp"

#include "block_manager.h"

namespace nanodeploy {

BlockManager::PreparedBlockMutation::PreparedBlockMutation(BlockManager* manager, Kind kind) noexcept:
    manager_(manager), kind_(kind)
{
}

BlockManager::PreparedBlockMutation::PreparedBlockMutation(PreparedBlockMutation&& other) noexcept
{
    take_from(std::move(other));
}

BlockManager::PreparedBlockMutation&
BlockManager::PreparedBlockMutation::operator=(PreparedBlockMutation&& other) noexcept
{
    if (this != &other) {
        abort_noexcept();
        take_from(std::move(other));
    }
    return *this;
}

BlockManager::PreparedBlockMutation::~PreparedBlockMutation() noexcept
{
    abort_noexcept();
}

void BlockManager::PreparedBlockMutation::take_from(PreparedBlockMutation&& other) noexcept
{
    manager_              = std::exchange(other.manager_, nullptr);
    kind_                 = other.kind_;
    state_                = std::exchange(other.state_, State::ABORTED);
    block_ids_            = std::move(other.block_ids_);
    allocation_snapshots_ = std::move(other.allocation_snapshots_);
    staged_free_nodes_    = std::move(other.staged_free_nodes_);
    release_node_iters_   = std::move(other.release_node_iters_);
}

void BlockManager::PreparedBlockMutation::commit_noexcept() noexcept
{
    if (state_ != State::PREPARED || manager_ == nullptr) {
        return;
    }

    if (kind_ == Kind::ALLOCATE) {
        // The blocks already carry one transaction-owned reference and are in
        // used_block_ids_. Committing only discards their detached free-list
        // nodes; attaching the IDs to a shadow/published context is the caller's
        // responsibility.
        staged_free_nodes_.clear();
    }
    else {
        // Every node needed for a refcount-1 release was allocated by prepare.
        // Install their order metadata before the noexcept list splice.
        for (int block_id : block_ids_) {
            Block& block = manager_->blocks_[block_id];
            block.ref_count--;
            if (block.ref_count == 0) {
                manager_->used_block_ids_.erase(block_id);
            }
        }
        for (auto node : release_node_iters_) {
            int block_id                                  = *node;
            manager_->free_order_keys_[block_id]          = manager_->next_free_order_key_++;
            manager_->block_id_to_free_list_it_[block_id] = node;
        }
        manager_->free_block_ids_.splice(manager_->free_block_ids_.end(), staged_free_nodes_);
    }

    for (int block_id : block_ids_) {
        manager_->prepared_block_ids_.erase(block_id);
    }
    state_   = State::COMMITTED;
    manager_ = nullptr;
}

void BlockManager::PreparedBlockMutation::abort_noexcept() noexcept
{
    if (state_ != State::PREPARED || manager_ == nullptr) {
        return;
    }

    if (kind_ == Kind::ALLOCATE) {
        for (auto& snapshot : allocation_snapshots_) {
            Block& block    = manager_->blocks_[snapshot.block_id];
            block.ref_count = snapshot.ref_count;
            block.hash      = snapshot.hash;
            block.token_ids.swap(snapshot.token_ids);
            manager_->used_block_ids_.erase(snapshot.block_id);
        }

        // free_block_ids_ is ordered by a stable per-free-event key. Reinsert
        // one detached node at a time, so disjoint concurrent preparations do
        // not invalidate a saved neighbor iterator or perturb abort ordering.
        while (!staged_free_nodes_.empty()) {
            auto node     = staged_free_nodes_.begin();
            int  block_id = *node;
            auto position = std::find_if(manager_->free_block_ids_.begin(),
                                         manager_->free_block_ids_.end(),
                                         [&](int current_id) {
                                             return manager_->free_order_keys_[current_id]
                                                    > manager_->free_order_keys_[block_id];
                                         });
            manager_->block_id_to_free_list_it_[block_id] = node;
            manager_->free_block_ids_.splice(position, staged_free_nodes_, node);
        }
    }
    else {
        // Release preparation did not alter ownership/refcounts. Destroying the
        // transaction-owned list nodes is the complete rollback.
        staged_free_nodes_.clear();
    }

    for (int block_id : block_ids_) {
        manager_->prepared_block_ids_.erase(block_id);
    }
    state_   = State::ABORTED;
    manager_ = nullptr;
}

BlockManager::PreparedBlockRebalance::PreparedBlockRebalance(BlockManager* manager) noexcept:
    manager_(manager)
{
}

BlockManager::PreparedBlockRebalance::PreparedBlockRebalance(PreparedBlockRebalance&& other) noexcept
{
    take_from(std::move(other));
}

BlockManager::PreparedBlockRebalance&
BlockManager::PreparedBlockRebalance::operator=(PreparedBlockRebalance&& other) noexcept
{
    if (this != &other) {
        abort_noexcept();
        take_from(std::move(other));
    }
    return *this;
}

BlockManager::PreparedBlockRebalance::~PreparedBlockRebalance() noexcept
{
    abort_noexcept();
}

void BlockManager::PreparedBlockRebalance::take_from(PreparedBlockRebalance&& other) noexcept
{
    manager_                    = std::exchange(other.manager_, nullptr);
    state_                      = std::exchange(other.state_, State::ABORTED);
    release_block_ids_          = std::move(other.release_block_ids_);
    allocation_block_ids_       = std::move(other.allocation_block_ids_);
    reused_block_ids_           = std::move(other.reused_block_ids_);
    newly_allocated_block_ids_  = std::move(other.newly_allocated_block_ids_);
    allocation_snapshots_       = std::move(other.allocation_snapshots_);
    detached_free_nodes_        = std::move(other.detached_free_nodes_);
    staged_release_nodes_       = std::move(other.staged_release_nodes_);
}

void BlockManager::PreparedBlockRebalance::commit_noexcept() noexcept
{
    if (state_ != State::PREPARED || manager_ == nullptr) {
        return;
    }

    // A reused block transfers its sole transaction-protected ownership
    // reference. Every other release drops one reference; refcount-1 releases
    // already have a transaction-owned free-list node ready to splice.
    for (int block_id : release_block_ids_) {
        if (std::find(reused_block_ids_.begin(), reused_block_ids_.end(), block_id)
            != reused_block_ids_.end()) {
            Block& block = manager_->blocks_[block_id];
            block.hash = -1;
            block.token_ids.clear();
            continue;
        }
        Block& block = manager_->blocks_[block_id];
        block.ref_count--;
        if (block.ref_count == 0) {
            manager_->used_block_ids_.erase(block_id);
        }
    }

    for (auto node = staged_release_nodes_.begin(); node != staged_release_nodes_.end(); ++node) {
        const int block_id                             = *node;
        manager_->free_order_keys_[block_id]          = manager_->next_free_order_key_++;
        manager_->block_id_to_free_list_it_[block_id] = node;
    }
    manager_->free_block_ids_.splice(manager_->free_block_ids_.end(), staged_release_nodes_);
    detached_free_nodes_.clear();

    for (int block_id : release_block_ids_) {
        manager_->prepared_block_ids_.erase(block_id);
    }
    for (int block_id : newly_allocated_block_ids_) {
        manager_->prepared_block_ids_.erase(block_id);
    }
    state_   = State::COMMITTED;
    manager_ = nullptr;
}

void BlockManager::PreparedBlockRebalance::abort_noexcept() noexcept
{
    if (state_ != State::PREPARED || manager_ == nullptr) {
        return;
    }

    for (auto& snapshot : allocation_snapshots_) {
        Block& block    = manager_->blocks_[snapshot.block_id];
        block.ref_count = snapshot.ref_count;
        block.hash      = snapshot.hash;
        block.token_ids.swap(snapshot.token_ids);
        manager_->used_block_ids_.erase(snapshot.block_id);
    }
    while (!detached_free_nodes_.empty()) {
        auto node     = detached_free_nodes_.begin();
        int  block_id = *node;
        auto position = std::find_if(manager_->free_block_ids_.begin(),
                                     manager_->free_block_ids_.end(),
                                     [&](int current_id) {
                                         return manager_->free_order_keys_[current_id]
                                                > manager_->free_order_keys_[block_id];
                                     });
        manager_->block_id_to_free_list_it_[block_id] = node;
        manager_->free_block_ids_.splice(position, detached_free_nodes_, node);
    }
    staged_release_nodes_.clear();

    for (int block_id : release_block_ids_) {
        manager_->prepared_block_ids_.erase(block_id);
    }
    for (int block_id : newly_allocated_block_ids_) {
        manager_->prepared_block_ids_.erase(block_id);
    }
    state_   = State::ABORTED;
    manager_ = nullptr;
}

BlockManager::BlockManager(const std::string& engine_id, int sp_idx, int num_blocks, int block_size):
    engine_id_(engine_id), sp_idx_(sp_idx), block_size_(block_size)
{

    blocks_.reserve(num_blocks);
    block_id_to_free_list_it_.resize(num_blocks);
    free_order_keys_.resize(num_blocks);
    for (int i = 0; i < num_blocks; ++i) {
        blocks_.emplace_back(i, block_size);
        free_block_ids_.push_back(i);
        block_id_to_free_list_it_[i] = std::prev(free_block_ids_.end());
        free_order_keys_[i]          = next_free_order_key_++;
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
    // unordered_set insertion is the only potentially allocating operation in
    // this path. Perform it before removing the block from the free list so an
    // allocation exception cannot strand the block in neither collection.
    used_block_ids_.insert(block_id);
    block.reset();

    auto it = block_id_to_free_list_it_[block_id];
    if (it != free_block_ids_.end()) {
        free_block_ids_.erase(it);
        block_id_to_free_list_it_[block_id] = free_block_ids_.end();
    }
    return blocks_[block_id];
}

void BlockManager::deallocate_block(int block_id)
{
    if (prepared_block_ids_.count(block_id)) {
        throw std::runtime_error("KV cache block is owned by a prepared mutation");
    }
    if (blocks_[block_id].ref_count != 0) {
        throw std::runtime_error("Block ref_count is not 0");
    }
    // Allocate the free-list node before erasing ownership. If allocation
    // fails, the block remains discoverable as used and the caller can retry.
    free_block_ids_.push_back(block_id);
    free_order_keys_[block_id] = next_free_order_key_++;
    used_block_ids_.erase(block_id);
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

        if (block_id == -1 || prepared_block_ids_.count(block_id)
            || blocks_[block_id].token_ids.size() != view.second
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
    // Reserve both destination vectors before touching the free list. The
    // subsequent push operations then cannot throw due to vector growth.
    table.reserve(num_blocks);
    seq.block_ctx(BlockContextSlot::ACTIVE)
        .block_location.reserve(seq.block_ctx(BlockContextSlot::ACTIVE).block_location.size() + num_blocks);
    for (int i = 0; i < num_blocks; ++i) {
        int block_id = free_block_ids_.front();
        allocate_block(block_id);
        seq.block_ctx(BlockContextSlot::ACTIVE).block_location.emplace_back(sp_idx_, block_id);
        table.push_back(block_id);
    }
}

BlockManager::PreparedBlockMutation BlockManager::prepare_allocate_uncached(int count)
{
    if (count < 0) {
        throw std::runtime_error("prepared allocation count must be non-negative");
    }
    if (static_cast<int>(free_block_ids_.size()) < count) {
        throw std::runtime_error("No free blocks available for prepared allocation");
    }

    std::vector<int> block_ids;
    block_ids.reserve(count);
    auto free_it = free_block_ids_.begin();
    for (int idx = 0; idx < count; ++idx, ++free_it) {
        block_ids.push_back(*free_it);
    }
    return prepare_allocate_uncached(block_ids);
}

BlockManager::PreparedBlockMutation
BlockManager::prepare_allocate_uncached(const std::vector<int>& exact_block_ids)
{
    PreparedBlockMutation mutation(this, PreparedBlockMutation::Kind::ALLOCATE);
    mutation.block_ids_ = exact_block_ids;
    mutation.allocation_snapshots_.reserve(exact_block_ids.size());

    std::vector<unsigned char> selected(blocks_.size(), 0);
    for (int block_id : exact_block_ids) {
        if (block_id < 0 || block_id >= static_cast<int>(blocks_.size())) {
            throw std::runtime_error("prepared allocation block ID is out of range");
        }
        if (selected[block_id]) {
            throw std::runtime_error("prepared allocation contains a duplicate block ID");
        }
        selected[block_id] = 1;
        if (prepared_block_ids_.count(block_id) || used_block_ids_.count(block_id)
            || blocks_[block_id].ref_count != 0 || block_id_to_free_list_it_[block_id] == free_block_ids_.end()) {
            throw std::runtime_error("prepared allocation requested a block that is not free");
        }

        PreparedBlockMutation::BlockSnapshot snapshot;
        snapshot.block_id  = block_id;
        snapshot.ref_count = blocks_[block_id].ref_count;
        snapshot.hash      = blocks_[block_id].hash;
        snapshot.token_ids = blocks_[block_id].token_ids;
        mutation.allocation_snapshots_.push_back(std::move(snapshot));
    }

    try {
        for (int block_id : exact_block_ids) {
            if (!prepared_block_ids_.insert(block_id).second || !used_block_ids_.insert(block_id).second) {
                throw std::runtime_error("prepared allocation ownership changed during prepare");
            }
        }
    }
    catch (...) {
        for (int block_id : exact_block_ids) {
            prepared_block_ids_.erase(block_id);
            used_block_ids_.erase(block_id);
        }
        throw;
    }

    // From here through PREPARED publication every operation is non-allocating.
    // Detach selected free-list nodes in list order while preserving the caller's
    // requested ID order separately in mutation.block_ids_.
    auto free_it = free_block_ids_.begin();
    while (free_it != free_block_ids_.end()) {
        auto current = free_it++;
        int  block_id = *current;
        if (!selected[block_id]) {
            continue;
        }
        block_id_to_free_list_it_[block_id] = free_block_ids_.end();
        mutation.staged_free_nodes_.splice(mutation.staged_free_nodes_.end(), free_block_ids_, current);
    }
    for (int block_id : exact_block_ids) {
        Block& block    = blocks_[block_id];
        block.ref_count = 1;
        block.hash      = -1;
        block.token_ids.clear();
    }
    mutation.state_ = PreparedBlockMutation::State::PREPARED;
    return mutation;
}

BlockManager::PreparedBlockMutation BlockManager::prepare_release(const std::vector<int>& owned_block_ids)
{
    PreparedBlockMutation mutation(this, PreparedBlockMutation::Kind::RELEASE);
    mutation.block_ids_ = owned_block_ids;
    mutation.release_node_iters_.reserve(owned_block_ids.size());

    std::vector<unsigned char> selected(blocks_.size(), 0);
    for (int block_id : owned_block_ids) {
        if (block_id < 0 || block_id >= static_cast<int>(blocks_.size())) {
            throw std::runtime_error("prepared release block ID is out of range");
        }
        if (selected[block_id]) {
            throw std::runtime_error("prepared release contains a duplicate block ID");
        }
        selected[block_id] = 1;
        if (prepared_block_ids_.count(block_id) || !used_block_ids_.count(block_id)
            || blocks_[block_id].ref_count <= 0 || block_id_to_free_list_it_[block_id] != free_block_ids_.end()) {
            throw std::runtime_error("prepared release requested a block that is not owned");
        }
        if (blocks_[block_id].ref_count == 1) {
            mutation.staged_free_nodes_.push_back(block_id);
            mutation.release_node_iters_.push_back(std::prev(mutation.staged_free_nodes_.end()));
        }
    }

    try {
        for (int block_id : owned_block_ids) {
            if (!prepared_block_ids_.insert(block_id).second) {
                throw std::runtime_error("prepared release ownership changed during prepare");
            }
        }
    }
    catch (...) {
        for (int block_id : owned_block_ids) {
            prepared_block_ids_.erase(block_id);
        }
        throw;
    }

    mutation.state_ = PreparedBlockMutation::State::PREPARED;
    return mutation;
}

BlockManager::PreparedBlockRebalance
BlockManager::prepare_rebalance(const std::vector<int>& release_block_ids, int allocation_count)
{
    if (allocation_count < 0) {
        throw std::runtime_error("prepared rebalance allocation count must be non-negative");
    }

    PreparedBlockRebalance mutation(this);
    mutation.release_block_ids_ = release_block_ids;
    mutation.allocation_block_ids_.reserve(static_cast<size_t>(allocation_count));
    mutation.reused_block_ids_.reserve(
        std::min(release_block_ids.size(), static_cast<size_t>(allocation_count)));

    std::vector<unsigned char> selected(blocks_.size(), 0);
    for (int block_id : release_block_ids) {
        if (block_id < 0 || block_id >= static_cast<int>(blocks_.size())) {
            throw std::runtime_error("prepared rebalance release block ID is out of range");
        }
        if (selected[block_id]) {
            throw std::runtime_error("prepared rebalance contains a duplicate release block ID");
        }
        selected[block_id] = 1;
        if (prepared_block_ids_.count(block_id) || !used_block_ids_.count(block_id)
            || blocks_[block_id].ref_count <= 0 || block_id_to_free_list_it_[block_id] != free_block_ids_.end()) {
            throw std::runtime_error("prepared rebalance requested a release that is not owned");
        }
        if (blocks_[block_id].ref_count == 1
            && static_cast<int>(mutation.allocation_block_ids_.size()) < allocation_count) {
            mutation.reused_block_ids_.push_back(block_id);
            mutation.allocation_block_ids_.push_back(block_id);
        }
    }

    const int free_needed = allocation_count - static_cast<int>(mutation.allocation_block_ids_.size());
    if (free_needed > static_cast<int>(free_block_ids_.size())) {
        throw std::runtime_error("No free or reusable blocks available for prepared rebalance");
    }
    mutation.newly_allocated_block_ids_.reserve(static_cast<size_t>(free_needed));
    mutation.allocation_snapshots_.reserve(static_cast<size_t>(free_needed));
    auto free_it = free_block_ids_.begin();
    for (int idx = 0; idx < free_needed; ++idx, ++free_it) {
        const int block_id = *free_it;
        mutation.newly_allocated_block_ids_.push_back(block_id);
        mutation.allocation_block_ids_.push_back(block_id);

        PreparedBlockRebalance::BlockSnapshot snapshot;
        snapshot.block_id  = block_id;
        snapshot.ref_count = blocks_[block_id].ref_count;
        snapshot.hash      = blocks_[block_id].hash;
        snapshot.token_ids = blocks_[block_id].token_ids;
        mutation.allocation_snapshots_.push_back(std::move(snapshot));
    }

    // Allocate every release-side free-list node and every bookkeeping vector
    // before reserving allocator ownership. Shared releases need no node.
    for (int block_id : release_block_ids) {
        const bool reused = std::find(mutation.reused_block_ids_.begin(),
                                      mutation.reused_block_ids_.end(),
                                      block_id)
                            != mutation.reused_block_ids_.end();
        if (!reused && blocks_[block_id].ref_count == 1) {
            mutation.staged_release_nodes_.push_back(block_id);
        }
    }

    std::vector<unsigned char> selected_free(blocks_.size(), 0);
    for (int block_id : mutation.newly_allocated_block_ids_) {
        selected_free[block_id] = 1;
    }

    try {
        for (int block_id : release_block_ids) {
            if (!prepared_block_ids_.insert(block_id).second) {
                throw std::runtime_error("prepared rebalance release ownership changed during prepare");
            }
        }
        for (int block_id : mutation.newly_allocated_block_ids_) {
            if (!prepared_block_ids_.insert(block_id).second || !used_block_ids_.insert(block_id).second) {
                throw std::runtime_error("prepared rebalance allocation ownership changed during prepare");
            }
        }
    }
    catch (...) {
        for (int block_id : release_block_ids) {
            prepared_block_ids_.erase(block_id);
        }
        for (int block_id : mutation.newly_allocated_block_ids_) {
            prepared_block_ids_.erase(block_id);
            used_block_ids_.erase(block_id);
        }
        throw;
    }

    free_it = free_block_ids_.begin();
    while (free_it != free_block_ids_.end()) {
        auto current = free_it++;
        int  block_id = *current;
        if (!selected_free[block_id]) {
            continue;
        }
        block_id_to_free_list_it_[block_id] = free_block_ids_.end();
        mutation.detached_free_nodes_.splice(mutation.detached_free_nodes_.end(), free_block_ids_, current);
    }
    for (int block_id : mutation.newly_allocated_block_ids_) {
        Block& block    = blocks_[block_id];
        block.ref_count = 1;
        block.hash      = -1;
        block.token_ids.clear();
    }

    mutation.state_ = PreparedBlockRebalance::State::PREPARED;
    return mutation;
}

void BlockManager::deallocate(Sequence& seq, BlockContextSlot slot)
{
    auto& table = seq.block_table(slot, sp_idx_);
    for (int block_id : table) {
        if (prepared_block_ids_.count(block_id)) {
            throw std::runtime_error("cannot deallocate a block owned by a prepared mutation");
        }
    }
    // Remove each successfully released ID immediately. If free-list growth
    // throws, the remaining table is an exact retry set and never contains an
    // already-free block.
    while (!table.empty()) {
        int    block_id = table.back();
        Block& block    = blocks_[block_id];
        block.ref_count--;
        if (block.ref_count == 0) {
            try {
                deallocate_block(block_id);
            }
            catch (...) {
                block.ref_count++;
                throw;
            }
        }
        table.pop_back();
    }
    seq.num_cached_tokens = 0;
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
    for (int idx = keep_blocks; idx < static_cast<int>(table.size()); ++idx) {
        if (prepared_block_ids_.count(table[idx])) {
            throw std::runtime_error("cannot trim a block owned by a prepared mutation");
        }
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

std::vector<int> BlockManager::reserve_blocks(int count)
{
    if (count < 0) {
        throw std::runtime_error("reserved block count must be non-negative");
    }
    if (static_cast<int>(free_block_ids_.size()) < count) {
        throw std::runtime_error("No free blocks available for KV consolidation");
    }

    std::vector<int> reserved;
    reserved.reserve(count);
    try {
        for (int idx = 0; idx < count; ++idx) {
            int block_id = free_block_ids_.front();
            allocate_block(block_id);
            reserved.push_back(block_id);
        }
    }
    catch (...) {
        release_blocks(reserved);
        throw;
    }
    return reserved;
}

void BlockManager::release_blocks(const std::vector<int>& block_ids)
{
    for (int block_id : block_ids) {
        if (prepared_block_ids_.count(block_id)) {
            throw std::runtime_error("cannot release a block owned by a prepared mutation");
        }
    }
    for (int block_id : block_ids) {
        if (block_id < 0 || block_id >= static_cast<int>(blocks_.size()) || !used_block_ids_.count(block_id)) {
            throw std::runtime_error("attempted to release an unowned KV cache block");
        }
        Block& block = blocks_[block_id];
        if (block.ref_count <= 0) {
            throw std::runtime_error("KV cache block has an invalid reference count");
        }
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
