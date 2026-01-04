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
                               int                max_num_batched_tokens,
                               int                max_num_recv_seqs):
    engine_id_(engine_id),
    attention_sp_(attention_sp),
    max_num_seqs_(max_num_seqs),
    max_num_batched_tokens_(max_num_batched_tokens),
    max_num_recv_seqs_(max_num_recv_seqs),
    kvcache_block_size_(kvcache_block_size),
    num_running_seqs_per_sp_(attention_sp, 0),
    num_running_tokens_per_sp_(attention_sp, 0),
    num_recv_seqs_per_sp_(attention_sp, 0)
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
    // =========================================================================
    // Branch 1: Sequence Parallel (SP > 1)
    // Strategy: LeastBatch Master + MostFree Helpers + Water-filling
    // =========================================================================
    if (attention_sp_ > 1) {
        // Step 1: Gather rank status
        struct RankStatus {
            int       id;
            long long current_kv_load;   // Current tokens
            int       current_batch_load; // Current seqs
            int       current_recv_load;  // Current receiving seqs
            int       free_blocks;
        };

        std::vector<RankStatus> all_ranks;
        all_ranks.reserve(attention_sp_);

        for (int i = 0; i < attention_sp_; ++i) {
            long long tokens = num_running_tokens_per_sp_[i];
            if (num_batched_tokens.count(i)) tokens += num_batched_tokens.at(i);

            int seqs = num_running_seqs_per_sp_[i];
            if (num_seqs.count(i)) seqs += num_seqs.at(i);
            
            int recvs = num_recv_seqs_per_sp_[i];

            int free_blks = block_manager[i]->num_free_blocks();
            all_ranks.push_back({i, tokens, seqs, recvs, free_blks});
        }

        int num_tokens = seq.num_tokens;
        int needed_blocks = (num_tokens + kvcache_block_size_ - 1) / kvcache_block_size_;
        
        // Calculate minimum required parallelism
        int num_segments = (num_tokens + segment_size - 1) / segment_size;
        // [Fix Warning] Use size_t
        size_t initial_ranks_needed = static_cast<size_t>(std::max(1, std::min(attention_sp_, num_segments)));

        // Step 2: Filter and sort potential Masters
        std::vector<RankStatus> candidate_masters;
        for (const auto& r : all_ranks) {
            // Only ranks with available batch capacity can be Master
            if (r.current_batch_load + 1 <= max_num_seqs_) {
                candidate_masters.push_back(r);
            }
        }

        // Master sort: LeastBatch first
        std::sort(candidate_masters.begin(), candidate_masters.end(),
                  [](const RankStatus& a, const RankStatus& b) {
                      if (a.current_batch_load != b.current_batch_load) {
                          return a.current_batch_load < b.current_batch_load;
                      }
                      return a.free_blocks > b.free_blocks;
                  });

        if (candidate_masters.empty()) {
            return false; 
        }

        auto& block_ctx = seq.block_ctx(BlockContextSlot::ACTIVE);

        // Step 3: Try each candidate Master with Helpers
        for (const auto& master : candidate_masters) {
            // 3.1 Prepare Helpers (Receivers)
            std::vector<RankStatus> candidate_helpers;
            candidate_helpers.reserve(attention_sp_ - 1);

            for (const auto& r : all_ranks) {
                if (r.id == master.id) continue;
                if (r.current_recv_load < max_num_recv_seqs_) {
                    candidate_helpers.push_back(r);
                }
            }

            // Helper sort: MostFree first (Best Fit)
            std::sort(candidate_helpers.begin(), candidate_helpers.end(),
                      [](const RankStatus& a, const RankStatus& b) {
                          return a.free_blocks > b.free_blocks;
                      });

            // 3.2 Build Participants (Master + Helpers)
            std::vector<RankStatus> participants;
            participants.push_back(master);
            
            long long current_capacity = master.free_blocks;
            bool enough_memory = (current_capacity >= needed_blocks);
            bool enough_ranks  = (participants.size() >= initial_ranks_needed);

            size_t helper_idx = 0;
            while ((!enough_memory || !enough_ranks) && helper_idx < candidate_helpers.size()) {
                const auto& helper = candidate_helpers[helper_idx++];
                participants.push_back(helper);
                current_capacity += helper.free_blocks;
                
                enough_memory = (current_capacity >= needed_blocks);
                enough_ranks  = (participants.size() >= initial_ranks_needed);
            }

            if (!enough_memory) {
                continue; 
            }

            // Step 4: Water-filling simulation
            block_ctx.num_dispatched_tokens.assign(attention_sp_, 0);
            std::vector<long long> simulated_kv_loads;
            std::vector<int> alloc_counts(participants.size(), 0);
            
            for(const auto& p : participants) {
                simulated_kv_loads.push_back(p.current_kv_load);
            }

            int tokens_remaining = num_tokens;
            const int CHUNK_SIZE = kvcache_block_size_; 
            bool water_fill_failed = false;

            while (tokens_remaining > 0) {
                auto min_it = std::min_element(simulated_kv_loads.begin(), simulated_kv_loads.end());
                int idx = std::distance(simulated_kv_loads.begin(), min_it);
                
                int rank_capacity_tokens = participants[idx].free_blocks * kvcache_block_size_;
                int attempt_alloc = std::min(CHUNK_SIZE, tokens_remaining);

                if (alloc_counts[idx] + attempt_alloc > rank_capacity_tokens) {
                    *min_it = std::numeric_limits<long long>::max(); 
                    bool all_full = true;
                    for(auto val : simulated_kv_loads) {
                        if (val != std::numeric_limits<long long>::max()) {
                            all_full = false; break;
                        }
                    }
                    if (all_full) {
                        water_fill_failed = true;
                        break;
                    }
                    continue;
                }

                simulated_kv_loads[idx] += attempt_alloc;
                alloc_counts[idx]       += attempt_alloc;
                tokens_remaining        -= attempt_alloc;
            }

            if (water_fill_failed) {
                continue; 
            }

            // Step 5: Commit allocation plan
            for (size_t i = 0; i < participants.size(); ++i) {
                int rank_id = participants[i].id;
                block_ctx.num_dispatched_tokens[rank_id] = alloc_counts[i];
            }
            
            block_ctx.master_sp_idx_ = master.id;

            // Physical check
            bool physical_check_ok = true;
            for (size_t i = 0; i < participants.size(); ++i) {
                int rank_id = participants[i].id;
                if (alloc_counts[i] > 0) {
                     if (!block_manager[rank_id]->can_allocate(seq)) {
                        physical_check_ok = false;
                        break;
                    }
                }
            }

            if (physical_check_ok) {
                return true;
            }
        }

        return false;
    } 
    // =========================================================================
    // Branch 2: Pure Data Parallel (SP = 1)
    // Logic: Direct rank constraint check without complex splitting
    // =========================================================================
    else {
        auto& block_ctx = seq.block_ctx(BlockContextSlot::ACTIVE);
        // Reset state
        block_ctx.num_dispatched_tokens.assign(attention_sp_, 0);

        // SP=1: attention_sp_ is 1, sp_idx typically 0
        // Use next_sp_idx() to maintain round-robin logic
        int master_rank = next_sp_idx(); 

        // 1. Check Batch Size constraint
        int running_master_count = num_running_seqs_per_sp_[master_rank];
        
        auto it_seqs = num_seqs.find(master_rank);
        int current_num_seqs = (it_seqs != num_seqs.end()) ? it_seqs->second : 0;

        if (current_num_seqs + running_master_count + 1 > max_num_seqs_) {
            return false;
        }

        // 2. Check Batched Tokens constraint
        auto it_tokens = num_batched_tokens.find(master_rank);
        int current_batched_tokens = (it_tokens != num_batched_tokens.end()) ? it_tokens->second : 0;
        
        if (current_batched_tokens + seq.num_tokens >= max_num_batched_tokens_) {
            return false;
        }

        // 3. Prepare allocation info
        // SP=1: Master handles all tokens, no helpers
        block_ctx.master_sp_idx_ = master_rank;
        block_ctx.num_dispatched_tokens[master_rank] = seq.num_tokens;

        // 4. Physical memory check
        if (!block_manager[master_rank]->can_allocate(seq)) {
            return false;
        }

        return true;
    }
}

void SPStateManager::allocate(Sequence& seq)
{
    auto& block_ctx     = seq.block_ctx(BlockContextSlot::ACTIVE);
    int   master_sp_idx = block_ctx.master_sp_idx_;

    for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
        // [修改] 更新 Recv 计数
        // 只有确实分到了 token 且不是 Master 的才算 Receiver
        if (block_ctx.num_dispatched_tokens[sp_idx] > 0) {
            if (sp_idx != master_sp_idx) {
                 num_recv_seqs_per_sp_[sp_idx]++;
            }
        }

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
    auto& block_ctx     = seq.block_ctx(slot);
    int   master_sp_idx = block_ctx.master_sp_idx_;

    // [修改] 在清理 block_ctx 之前，先减少 Recv 计数
    for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
        if (block_ctx.num_dispatched_tokens[sp_idx] > 0) {
            if (sp_idx != master_sp_idx) {
                num_recv_seqs_per_sp_[sp_idx]--;
            }
        }
    }

    for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
        block_manager[sp_idx]->deallocate(seq, slot);
    }

    block_ctx.sp_block_table.clear();
    block_ctx.block_location.clear();
    std::fill(block_ctx.num_dispatched_tokens.begin(), block_ctx.num_dispatched_tokens.end(), 0);

    num_running_seqs_--;
    num_running_tokens_ -= seq.num_tokens;
    num_running_seqs_per_sp_[master_sp_idx]--;
    num_running_tokens_per_sp_[master_sp_idx] -= seq.num_tokens;
}

}  // namespace nanodeploy