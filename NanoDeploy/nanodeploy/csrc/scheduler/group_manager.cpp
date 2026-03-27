#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>

#include "nanodeploy/csrc/sequence/sequence.h"
#include "sequence_generated.h"

#include "group_manager.h"

namespace nanodeploy {

GroupManager::GroupManager(const std::string& engine_id,
                           int                group_size,
                           int                num_kvcache_blocks,
                           int                kvcache_block_size,
                           int                max_num_seqs,
                           int                max_num_batched_tokens):
    gdn_state_manager_(engine_id, 0, max_num_seqs),
    engine_id_(engine_id),
    group_size_(group_size),
    max_num_seqs_(max_num_seqs),
    max_num_batched_tokens_(max_num_batched_tokens),
    kvcache_block_size_(kvcache_block_size),
    num_kvcache_blocks_(num_kvcache_blocks),
    num_running_seqs_per_group_(group_size, 0),
    num_running_tokens_per_group_(group_size, 0)
{
    for (int i = 0; i < group_size; ++i) {
        block_manager[i] = std::make_shared<BlockManager>(engine_id, i, num_kvcache_blocks, kvcache_block_size);
    }

    initialize_dummy_seqs();
}

void GroupManager::initialize_dummy_seqs()
{
    // Use a fixed seed for reproducibility or random device
    std::random_device              rd;
    std::mt19937                    gen(rd());
    std::uniform_int_distribution<> dis(0, 7999);

    for (int group_id = 0; group_id < group_size_; ++group_id) {
        std::vector<int> token_ids = {dis(gen)};

        SamplingParams sp;
        sp.temperature = 1.0;
        sp.max_tokens  = 256;
        sp.ignore_eos  = false;

        auto dummy_seq = std::make_shared<Sequence>(token_ids, sp);
        dummy_seq->active(engine_id_, group_size_, 1, num_kvcache_blocks_);
        dummy_seq->block_ctx().master_group_id = group_id;

        dummy_seq->append_token(dis(gen), BlockContextSlot::ACTIVE, group_id);

        block_manager[group_id]->allocate(*dummy_seq);
        dummy_seqs.push_back(dummy_seq);
    }
}

int GroupManager::next_group_id()
{
    int idx           = group_rr_counter_;
    group_rr_counter_ = (group_rr_counter_ + 1) % group_size_;
    return idx;
}

bool GroupManager::can_append(Sequence& seq, int num_tokens)
{
    int master_group_id = seq.block_ctx(BlockContextSlot::ACTIVE).master_group_id;
    if (block_manager.find(master_group_id) == block_manager.end()) {
        return false;
    }
    return block_manager[master_group_id]->can_append(seq, num_tokens);
}

bool GroupManager::may_append(Sequence& seq, int num_tokens)
{
    int master_group_id = seq.block_ctx(BlockContextSlot::ACTIVE).master_group_id;
    if (block_manager.find(master_group_id) != block_manager.end()) {
        return block_manager[master_group_id]->may_append(seq, num_tokens);
    }
    return false;
}

bool GroupManager::can_allocate(Sequence&                           seq,
                                const std::unordered_map<int, int>& num_seqs,
                                const std::unordered_map<int, int>& num_batched_tokens)
{
    // Step 1: Determine min required ranks (Initial SP Size)
    int num_tokens           = seq.num_tokens();
    int num_segments         = (num_tokens + segment_size - 1) / segment_size;
    int initial_ranks_needed = std::max(1, std::min(group_size_, num_segments));

    // Step 2 & 3: Prepare and sort all ranks
    struct RankStatus {
        int       id;
        long long current_kv_load;
        int       current_batch_load;
        int       free_blocks;
    };

    std::vector<RankStatus> all_ranks;
    all_ranks.reserve(group_size_);

    for (int i = 0; i < group_size_; ++i) {
        long long tokens = num_running_tokens_per_group_[i];
        if (num_batched_tokens.count(i))
            tokens += num_batched_tokens.at(i);

        int seqs = num_running_seqs_per_group_[i];
        if (num_seqs.count(i))
            seqs += num_seqs.at(i);

        int free_blks = block_manager[i]->num_free_blocks();
        all_ranks.push_back({i, tokens, seqs, free_blks});
    }

    // Keep batch-first sorting strategy
    std::sort(all_ranks.begin(), all_ranks.end(), [](const RankStatus& a, const RankStatus& b) {
        if (a.current_batch_load != b.current_batch_load) {
            return a.current_batch_load < b.current_batch_load;
        }
        return a.free_blocks > b.free_blocks;
    });

    int needed_blocks = (num_tokens + kvcache_block_size_ - 1) / kvcache_block_size_;

    // Outer loop: Adaptively increase SP Size
    for (int current_sp_size = initial_ranks_needed; current_sp_size <= group_size_; ++current_sp_size) {

        // Step 4: Select participants for current size
        std::vector<RankStatus> participants;
        std::vector<RankStatus> candidates_pool;

        participants.reserve(current_sp_size);
        candidates_pool.reserve(group_size_ - current_sp_size);

        long long total_free_blocks_capacity = 0;

        for (int i = 0; i < group_size_; ++i) {
            if (i < current_sp_size) {
                participants.push_back(all_ranks[i]);
                total_free_blocks_capacity += all_ranks[i].free_blocks;
            }
            else {
                candidates_pool.push_back(all_ranks[i]);
            }
        }

        // Step 4.5: Swap participants if capacity insufficient
        auto sort_pool_by_mem_desc = [](const RankStatus& a, const RankStatus& b) {
            return a.free_blocks > b.free_blocks;
        };
        std::sort(candidates_pool.begin(), candidates_pool.end(), sort_pool_by_mem_desc);

        bool capacity_check_passed = true;
        while (total_free_blocks_capacity < needed_blocks) {
            if (candidates_pool.empty()) {
                capacity_check_passed = false;
                break;
            }

            auto min_mem_it = std::min_element(
                participants.begin(), participants.end(), [](const RankStatus& a, const RankStatus& b) {
                    return a.free_blocks < b.free_blocks;
                });

            const auto& rich_candidate = candidates_pool.front();

            if (rich_candidate.free_blocks <= min_mem_it->free_blocks) {
                capacity_check_passed = false;
                break;
            }

            total_free_blocks_capacity -= min_mem_it->free_blocks;
            total_free_blocks_capacity += rich_candidate.free_blocks;

            *min_mem_it = rich_candidate;
            candidates_pool.erase(candidates_pool.begin());
        }

        if (!capacity_check_passed) {
            continue;  // Try next SP Size
        }

        // Step 5: Water-filling allocation
        auto& block_ctx = seq.block_ctx(BlockContextSlot::ACTIVE);
        block_ctx.num_dispatched_tokens.assign(group_size_, 0);

        std::vector<long long> simulated_kv_loads;
        std::vector<int>       alloc_counts(participants.size(), 0);
        for (const auto& p : participants)
            simulated_kv_loads.push_back(p.current_kv_load);

        int       tokens_remaining  = num_tokens;
        const int CHUNK_SIZE        = kvcache_block_size_;
        bool      water_fill_failed = false;

        while (tokens_remaining > 0) {
            auto min_it = std::min_element(simulated_kv_loads.begin(), simulated_kv_loads.end());
            int  idx    = std::distance(simulated_kv_loads.begin(), min_it);

            int attempt_alloc        = std::min(CHUNK_SIZE, tokens_remaining);
            int rank_capacity_tokens = participants[idx].free_blocks * kvcache_block_size_;

            if (alloc_counts[idx] + attempt_alloc > rank_capacity_tokens) {
                *min_it = std::numeric_limits<long long>::max();  // Mark rank as full

                bool all_full = true;
                for (auto val : simulated_kv_loads) {
                    if (val != std::numeric_limits<long long>::max()) {
                        all_full = false;
                        break;
                    }
                }
                if (all_full) {
                    water_fill_failed = true;
                    break;
                }
                continue;
            }

            simulated_kv_loads[idx] += attempt_alloc;
            alloc_counts[idx] += attempt_alloc;
            tokens_remaining -= attempt_alloc;
        }

        if (water_fill_failed) {
            continue;  // Try larger SP Size
        }

        // Step 6 & 7: Success - write results and return

        // 1. Fill dispatch results
        for (size_t i = 0; i < participants.size(); ++i) {
            int rank_id                              = participants[i].id;
            block_ctx.num_dispatched_tokens[rank_id] = alloc_counts[i];
        }

        // 2. Select master (least loaded)
        auto min_batch_it =
            std::min_element(participants.begin(), participants.end(), [](const RankStatus& a, const RankStatus& b) {
                return a.current_batch_load < b.current_batch_load;
            });
        int master_rank           = min_batch_it->id;
        block_ctx.master_group_id = master_rank;

        // 3. Final physical check
        if (min_batch_it->current_batch_load + 1 > max_num_seqs_) {
            continue;  // Max seq limit reached, try larger SP Size
        }

        bool             physical_check_ok = true;
        std::vector<int> prefix_hints(group_size_, -1);
        for (size_t i = 0; i < participants.size(); ++i) {
            int rank_id = participants[i].id;
            int hits    = block_manager[rank_id]->can_allocate(seq);
            if (hits < 0) {
                physical_check_ok = false;
                break;
            }
            prefix_hints[rank_id] = hits;
        }

        if (physical_check_ok) {
            // *** Success! Store hints for allocate() ***
            cached_prefix_hints_ = std::move(prefix_hints);
            return true;
        }

        // Physical check failed, try larger SP Size
    }

    // All SP sizes failed
    return false;
}

std::optional<AllocResult> GroupManager::try_allocate(Sequence&                           seq,
                                                      const std::unordered_map<int, int>& num_seqs,
                                                      const std::unordered_map<int, int>& num_batched_tokens)
{
    int orig_num_tokens = seq.num_tokens();

    // Budget pre-check: skip expensive can_allocate if no tokens can be scheduled
    int budget = max_num_batched_tokens_;
    for (auto& [sp, tok] : num_batched_tokens)
        budget = std::min(budget, max_num_batched_tokens_ - tok);
    if (budget <= 0)
        return std::nullopt;

    // Set num_tokens to full prompt length for block allocation.
    // PD separation guarantees no decode traffic competes for KV cache,
    // so locking all blocks at admission eliminates mid-prefill OOM.
    int full_len = std::max(seq.num_prompt_tokens(), seq.num_checkpointed_tokens());
    seq.set_num_tokens(full_len);

    if (!can_allocate(seq, num_seqs, num_batched_tokens)) {
        seq.set_num_tokens(orig_num_tokens);
        return std::nullopt;
    }

    // Physical block allocation (sets seq.num_cached_tokens via prefix matching)
    allocate(seq);

    // Compute chunk boundary using actual prefix cache hits.
    // BlockManager::allocate caps num_cached_tokens at num_tokens - 1,
    // so full_len - num_cached >= 1 is guaranteed (combined with budget >= 1).
    int num_cached = seq.num_cached_tokens();
    int new_tokens = std::min(budget, full_len - num_cached);
    int chunk_end  = num_cached + new_tokens;

    seq.set_num_tokens(chunk_end);

    // For chunked sequences, restrict dispatch to master SP rank
    if (chunk_end < full_len) {
        auto& block_ctx    = seq.block_ctx(BlockContextSlot::ACTIVE);
        int   master_group = block_ctx.master_group_id;
        std::fill(block_ctx.num_dispatched_tokens.begin(), block_ctx.num_dispatched_tokens.end(), 0);
        block_ctx.num_dispatched_tokens[master_group] = chunk_end;
    }

    return AllocResult{chunk_end, new_tokens};
}

void GroupManager::allocate(Sequence& seq)
{
    auto& block_ctx       = seq.block_ctx(BlockContextSlot::ACTIVE);
    int   master_group_id = block_ctx.master_group_id;

    // Use prefix hints cached by can_allocate (if available) to skip
    // redundant hash scans inside BlockManager::allocate.
    auto hints = std::move(cached_prefix_hints_);
    cached_prefix_hints_.clear();

    auto get_hint = [&](int group_id) -> int {
        return (group_id < static_cast<int>(hints.size())) ? hints[group_id] : -1;
    };

    for (int group_id = 0; group_id < group_size_; ++group_id) {
        if (group_id != master_group_id) {
            block_manager[group_id]->allocate(seq, get_hint(group_id));
        }
    }
    block_manager[master_group_id]->allocate(seq, get_hint(master_group_id));

    // Assign a GDN state slot (index into conv/recurrent state buffers).
    // state_manager_ is a free-list over [0, max_num_seqs_); slot max_num_seqs_
    // is the reserved dummy slot and is never allocated here.
    gdn_state_manager_.allocate(seq);

    num_running_seqs_++;
    num_running_tokens_ += seq.num_tokens();
    num_running_seqs_per_group_[master_group_id]++;
    num_running_tokens_per_group_[master_group_id] += seq.num_tokens();
}

void GroupManager::deallocate(Sequence& seq, BlockContextSlot slot)
{
    for (int group_id = 0; group_id < group_size_; ++group_id) {
        block_manager[group_id]->deallocate(seq, slot);
    }

    // Free the GDN state slot so it can be reused by future sequences.
    gdn_state_manager_.deallocate(seq, slot);

    auto& block_ctx       = seq.block_ctx(BlockContextSlot::ACTIVE);
    int   master_group_id = block_ctx.master_group_id;
    block_ctx.group_block_table.clear();
    block_ctx.block_location.clear();
    std::fill(block_ctx.num_dispatched_tokens.begin(), block_ctx.num_dispatched_tokens.end(), 0);

    num_running_seqs_--;
    num_running_tokens_ -= seq.num_tokens();
    num_running_seqs_per_group_[master_group_id]--;
    num_running_tokens_per_group_[master_group_id] -= seq.num_tokens();
}

}  // namespace nanodeploy
