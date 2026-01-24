#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>

#include "nanodeploy/sequence/sequence.h"

#include "sp_state_manager.h"

namespace nanodeploy {

SPStateManager::SPStateManager(const std::string& engine_id,
                               int                attention_sp,
                               int                num_kvcache_blocks,
                               int                kvcache_block_size,
                               int                max_num_seqs,
                               int                max_num_batched_tokens):
    engine_id_(engine_id),
    attention_sp_(attention_sp),
    max_num_seqs_(max_num_seqs),
    max_num_batched_tokens_(max_num_batched_tokens),
    kvcache_block_size_(kvcache_block_size),
    num_running_seqs_per_sp_(attention_sp, 0),
    num_running_tokens_per_sp_(attention_sp, 0)
{
    for (int i = 0; i < attention_sp; ++i) {
        block_manager[i] = std::make_shared<BlockManager>(engine_id, i, num_kvcache_blocks, kvcache_block_size);
    }

    initialize_dummy_seqs();
}

void SPStateManager::initialize_dummy_seqs()
{
    // Use a fixed seed for reproducibility or random device
    std::random_device              rd;
    std::mt19937                    gen(rd());
    std::uniform_int_distribution<> dis(0, 7999);

    for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
        std::vector<int> token_ids = {dis(gen)};

        auto dummy_seq = std::make_shared<Sequence>(token_ids,
                                                    1.0,   // temperature
                                                    256,   // max_tokens
                                                    false  // ignore_eos
        );
        dummy_seq->active(engine_id_, attention_sp_, 1);
        dummy_seq->block_ctx().master_sp_idx_ = sp_idx;

        dummy_seq->append_token(dis(gen), BlockContextSlot::ACTIVE, sp_idx);

        block_manager[sp_idx]->allocate(*dummy_seq);
        dummy_seqs.push_back(dummy_seq);
    }
}

int SPStateManager::next_sp_idx()
{
    int idx        = sp_rr_counter_;
    sp_rr_counter_ = (sp_rr_counter_ + 1) % attention_sp_;
    return idx;
}

bool SPStateManager::can_append(Sequence& seq, int num_tokens)
{
    int master_sp_idx = seq.block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_;
    if (block_manager.find(master_sp_idx) == block_manager.end()) {
        return false;
    }
    return block_manager[master_sp_idx]->can_append(seq, num_tokens);
}

bool SPStateManager::may_append(Sequence& seq, int num_tokens)
{
    int master_sp_idx = seq.block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_;
    if (block_manager.find(master_sp_idx) != block_manager.end()) {
        return block_manager[master_sp_idx]->may_append(seq, num_tokens);
    }
    return false;
}

bool SPStateManager::can_allocate(Sequence&                           seq,
                                  const std::unordered_map<int, int>& num_seqs,
                                  const std::unordered_map<int, int>& num_batched_tokens)
{
    // Step 1: Determine min required ranks (Initial SP Size)
    int num_tokens           = seq.num_tokens;
    int num_segments         = (num_tokens + segment_size - 1) / segment_size;
    int initial_ranks_needed = std::max(1, std::min(attention_sp_, num_segments));

    // Step 2 & 3: Prepare and sort all ranks
    struct RankStatus {
        int       id;
        long long current_kv_load;
        int       current_batch_load;
        int       free_blocks;
    };

    std::vector<RankStatus> all_ranks;
    all_ranks.reserve(attention_sp_);

    for (int i = 0; i < attention_sp_; ++i) {
        long long tokens = num_running_tokens_per_sp_[i];
        if (num_batched_tokens.count(i))
            tokens += num_batched_tokens.at(i);

        int seqs = num_running_seqs_per_sp_[i];
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
    for (int current_sp_size = initial_ranks_needed; current_sp_size <= attention_sp_; ++current_sp_size) {

        // Step 4: Select participants for current size
        std::vector<RankStatus> participants;
        std::vector<RankStatus> candidates_pool;

        participants.reserve(current_sp_size);
        candidates_pool.reserve(attention_sp_ - current_sp_size);

        long long total_free_blocks_capacity = 0;

        for (int i = 0; i < attention_sp_; ++i) {
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
        block_ctx.num_dispatched_tokens.assign(attention_sp_, 0);

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
        int master_rank          = min_batch_it->id;
        block_ctx.master_sp_idx_ = master_rank;

        // 3. Final physical check
        if (min_batch_it->current_batch_load + 1 > max_num_seqs_) {
            continue;  // Max seq limit reached, try larger SP Size
        }

        bool physical_check_ok = true;
        for (size_t i = 0; i < participants.size(); ++i) {
            int rank_id = participants[i].id;
            if (!block_manager[rank_id]->can_allocate(seq)) {
                physical_check_ok = false;
                break;
            }
        }

        if (physical_check_ok) {
            // *** Success! ***
            return true;
        }

        // Physical check failed, try larger SP Size
    }

    // All SP sizes failed
    return false;
}

void SPStateManager::allocate(Sequence& seq)
{
    auto& block_ctx     = seq.block_ctx(BlockContextSlot::ACTIVE);
    int   master_sp_idx = block_ctx.master_sp_idx_;

    for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
        if (sp_idx != master_sp_idx) {
            block_manager[sp_idx]->allocate(seq);
        }
    }
    block_manager[master_sp_idx]->allocate(seq);

    num_running_seqs_++;
    num_running_tokens_ += seq.num_tokens;
    num_running_seqs_per_sp_[master_sp_idx]++;
    num_running_tokens_per_sp_[master_sp_idx] += seq.num_tokens;
}

void SPStateManager::deallocate(Sequence& seq, BlockContextSlot slot)
{
    for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
        block_manager[sp_idx]->deallocate(seq, slot);
    }

    auto& block_ctx     = seq.block_ctx(BlockContextSlot::ACTIVE);
    int   master_sp_idx = block_ctx.master_sp_idx_;
    block_ctx.sp_block_table.clear();
    block_ctx.block_location.clear();
    std::fill(block_ctx.num_dispatched_tokens.begin(), block_ctx.num_dispatched_tokens.end(), 0);

    num_running_seqs_--;
    num_running_tokens_ -= seq.num_tokens;
    num_running_seqs_per_sp_[master_sp_idx]--;
    num_running_tokens_per_sp_[master_sp_idx] -= seq.num_tokens;
}

}  // namespace nanodeploy
