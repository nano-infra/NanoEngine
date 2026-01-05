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
    // Strategy: Adaptive SP Size + LeastBatch Sort + Memory Swap
    // =========================================================================
    if (attention_sp_ > 1) {
        // Debug logging setup
        std::vector<std::string> fail_reasons;
        bool debug_enabled = false; 

        // Step 1: Determine min required ranks (Initial SP Size)
        int num_tokens = seq.num_tokens;
        int num_segments = (num_tokens + segment_size - 1) / segment_size;
        int initial_ranks_needed = std::max(1, std::min(attention_sp_, num_segments));
        int needed_blocks = (num_tokens + kvcache_block_size_ - 1) / kvcache_block_size_;

        if (debug_enabled) {
             std::stringstream ss;
             ss << "Req(DP=" << dp_idx_ << "): seq_id=" << seq.seq_id << ", tokens=" << num_tokens 
                << ", blocks=" << needed_blocks << ", init_ranks=" << initial_ranks_needed;
             fail_reasons.push_back(ss.str());
        }

        // Step 2 & 3: Prepare and sort all ranks
        struct RankStatus {
            int id;
            long long current_kv_load;
            int current_batch_load;
            int free_blocks;
        };

        std::vector<RankStatus> all_ranks;
        all_ranks.reserve(attention_sp_);

        for (int i = 0; i < attention_sp_; ++i) {
            long long tokens = num_running_tokens_per_sp_[i];
            if (num_batched_tokens.count(i)) tokens += num_batched_tokens.at(i);
            
            int seqs = num_running_seqs_per_sp_[i];
            if (num_seqs.count(i)) seqs += num_seqs.at(i);
            
            int free_blks = block_manager[i]->num_free_blocks();
            all_ranks.push_back({i, tokens, seqs, free_blks});
        }

        // Keep batch-first sorting (LeastBatch)
        std::sort(all_ranks.begin(), all_ranks.end(), 
                  [](const RankStatus& a, const RankStatus& b) {
                      if (a.current_batch_load != b.current_batch_load) {
                          return a.current_batch_load < b.current_batch_load;
                      }
                      return a.free_blocks > b.free_blocks;
                  });

        // Outer loop: Adaptively increase SP Size
        for (int current_sp_size = initial_ranks_needed; current_sp_size <= attention_sp_; ++current_sp_size) {
            
            // Step 4: Select participants for current size
            std::vector<RankStatus> participants;
            std::vector<RankStatus> candidates_pool;
            
            participants.reserve(current_sp_size);
            candidates_pool.reserve(attention_sp_ - current_sp_size);

            long long total_free_blocks_capacity = 0;

            for(int i = 0; i < attention_sp_; ++i) {
                if (i < current_sp_size) {
                    participants.push_back(all_ranks[i]);
                    total_free_blocks_capacity += all_ranks[i].free_blocks;
                } else {
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

                auto min_mem_it = std::min_element(participants.begin(), participants.end(), 
                    [](const RankStatus& a, const RankStatus& b) {
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
                if (debug_enabled) {
                    std::stringstream ss;
                    ss << "  [SP=" << current_sp_size << "] Cap Fail: Available " << total_free_blocks_capacity << " < Needed " << needed_blocks;
                    ss << ". Participants: [";
                    for (size_t i = 0; i < participants.size(); ++i) {
                        ss << participants[i].id << "(" << participants[i].free_blocks << ")";
                        if (i < participants.size() - 1) ss << ", ";
                    }
                    ss << "]";
                    fail_reasons.push_back(ss.str());
                }
                continue; // Try next SP Size
            }

            // Step 5: Water-filling allocation
            auto& block_ctx = seq.block_ctx(BlockContextSlot::ACTIVE);
            block_ctx.num_dispatched_tokens.assign(attention_sp_, 0);

            std::vector<long long> simulated_kv_loads;
            std::vector<int> alloc_counts(participants.size(), 0);
            for(const auto& p : participants) simulated_kv_loads.push_back(p.current_kv_load);
            
            int tokens_remaining = num_tokens;
            const int CHUNK_SIZE = kvcache_block_size_; 
            bool water_fill_failed = false;

            while (tokens_remaining > 0) {
                auto min_it = std::min_element(simulated_kv_loads.begin(), simulated_kv_loads.end());
                int idx = std::distance(simulated_kv_loads.begin(), min_it);
                
                int attempt_alloc = std::min(CHUNK_SIZE, tokens_remaining);
                int rank_capacity_tokens = participants[idx].free_blocks * kvcache_block_size_;
                
                if (alloc_counts[idx] + attempt_alloc > rank_capacity_tokens) {
                    *min_it = std::numeric_limits<long long>::max(); // Mark rank as full
                    
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
                 if (debug_enabled) {
                    std::stringstream ss;
                    ss << "  [SP=" << current_sp_size << "] WaterFill Fail: Fragmentation or per-rank memory limits.";
                    ss << " Participants: [";
                    for (size_t i = 0; i < participants.size(); ++i) {
                        ss << participants[i].id << "(" << participants[i].free_blocks << ")";
                        if (i < participants.size() - 1) ss << ", ";
                    }
                    ss << "]";
                    fail_reasons.push_back(ss.str());
                }
                continue; // Try larger SP Size
            }

            // Step 6: Fill dispatch results
            for (size_t i = 0; i < participants.size(); ++i) {
                int rank_id = participants[i].id;
                block_ctx.num_dispatched_tokens[rank_id] = alloc_counts[i];
            }

            // Step 7: Select Master with RECV Constraints
            std::vector<RankStatus> eligible_masters;
            
            for (const auto& candidate_m : participants) {
                // Constraint 1: Master load check
                if (candidate_m.current_batch_load + 1 > max_num_seqs_) {
                    if (debug_enabled) {
                        std::stringstream ss;
                        ss << "    MasterReject(Rank" << candidate_m.id << "): BatchLoad " << candidate_m.current_batch_load + 1 << " > " << max_num_seqs_;
                        fail_reasons.push_back(ss.str());
                    }
                    continue;
                }

                // Constraint 2: Other participants as Receivers
                bool others_ok = true;
                for (const auto& p : participants) {
                    if (p.id == candidate_m.id) continue; // Skip Master itself
                    
                    // Check if p can accept another Recv request
                    if (num_recv_seqs_per_sp_[p.id] >= max_num_recv_seqs_) {
                        if (debug_enabled) {
                            std::stringstream ss;
                            ss << "    MasterReject(Rank" << candidate_m.id << "): Peer Rank" << p.id 
                               << " RecvFull (" << num_recv_seqs_per_sp_[p.id] << " >= " << max_num_recv_seqs_ << ")";
                            fail_reasons.push_back(ss.str());
                        }
                        others_ok = false;
                        break;
                    }
                }

                if (others_ok) {
                    eligible_masters.push_back(candidate_m);
                }
            }

            if (eligible_masters.empty()) {
                if (debug_enabled) {
                    std::stringstream ss;
                    ss << "  [SP=" << current_sp_size << "] No Eligible Master found.";
                    fail_reasons.push_back(ss.str());
                }
                continue; 
            }

            // Select Master with least load
            auto best_master_it = std::min_element(eligible_masters.begin(), eligible_masters.end(),
                [](const RankStatus& a, const RankStatus& b) {
                    return a.current_batch_load < b.current_batch_load;
                });
            
            int master_rank = best_master_it->id;
            block_ctx.master_sp_idx_ = master_rank;

            // Final physical check
            bool physical_check_ok = true;
            for (size_t i = 0; i < participants.size(); ++i) {
                int rank_id = participants[i].id;
                // Only check ranks with allocated tokens
                if (block_ctx.num_dispatched_tokens[rank_id] > 0) {
                     if (!block_manager[rank_id]->can_allocate(seq)) {
                        if (debug_enabled) {
                            std::stringstream ss;
                            ss << "  [SP=" << current_sp_size << "] PhysicalAlloc Fail at Rank " << rank_id;
                            fail_reasons.push_back(ss.str());
                        }
                        physical_check_ok = false;
                        break;
                    }
                }
            }

            if (physical_check_ok) {
                // *** Success! ***
                return true;
            }
            
            // Physical check failed, try larger SP Size
        }

        // All SP sizes failed
        if (debug_enabled) {
            std::cerr << "\n[SP_ALLOC_FAIL] EngineID: " << engine_id_ 
                      << " DP_Idx: " << dp_idx_ 
                      << " Failed to allocate seq " << seq.seq_id << std::endl;
            for (const auto& reason : fail_reasons) {
                std::cerr << reason << std::endl;
            }
            std::cerr << "[SP_ALLOC_FAIL] End Report\n" << std::endl;
        }

        return false;
    }
    // =========================================================================
    // Branch 2: Pure Data Parallel (SP = 1)
    // Strategy: Simple Check (Next RR -> Load Check -> Block Check)
    // =========================================================================
    else {
        auto& block_ctx = seq.block_ctx(BlockContextSlot::ACTIVE);
        // Reset state
        block_ctx.num_dispatched_tokens.assign(attention_sp_, 0);

        int master_rank = next_sp_idx(); 

        // 1. Batch Size Constraint (Max Seqs)
        int running_master_count = num_running_seqs_per_sp_[master_rank];
        
        auto it_seqs = num_seqs.find(master_rank);
        int current_num_seqs = (it_seqs != num_seqs.end()) ? it_seqs->second : 0;

        if (current_num_seqs + running_master_count + 1 > max_num_seqs_) {
            return false;
        }

        // 2. Batched Tokens Constraint
        auto it_tokens = num_batched_tokens.find(master_rank);
        int current_batched_tokens = (it_tokens != num_batched_tokens.end()) ? it_tokens->second : 0;
        
        if (current_batched_tokens + seq.num_tokens >= max_num_batched_tokens_) {
            return false;
        }

        // 3. Prepare allocation info
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