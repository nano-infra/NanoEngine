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
                               int                max_num_recv_seqs,
                               double             reserved_blocks_per_req,
                               bool               enable_dynamic_sp_size,
                               bool               enable_non_uniform_split,
                               const std::string& sp_master_selector) :
    engine_id_(engine_id),
    attention_sp_(attention_sp),
    max_num_seqs_(max_num_seqs),
    max_num_batched_tokens_(max_num_batched_tokens),
    max_num_recv_seqs_(max_num_recv_seqs),
    reserved_blocks_per_req_(reserved_blocks_per_req),
    kvcache_block_size_(kvcache_block_size),
    num_recv_seqs_per_sp_(attention_sp, 0),
    enable_dynamic_sp_size_(enable_dynamic_sp_size),
    enable_non_uniform_split_(enable_non_uniform_split)
{
    // Initialize Strategy
    if (sp_master_selector == "LeastBatch") {
        master_selector_ = SPMasterSelector::LeastBatch;
    } else if (sp_master_selector == "LeastCache") {
        master_selector_ = SPMasterSelector::LeastCache;
    } else {
        master_selector_ = SPMasterSelector::RoundRobin;
    }

    // Initialize Running Load Counter
    master_seq_counts_.assign(attention_sp_, 0);

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

int SPStateManager::select_master_rank()
{
    if (master_selector_ == SPMasterSelector::RoundRobin) {
        int idx = sp_rr_counter_;
        sp_rr_counter_ = (sp_rr_counter_ + 1) % attention_sp_;
        return idx;
    } 
    else if (master_selector_ == SPMasterSelector::LeastBatch) {
        int best_idx = 0;
        int min_load = std::numeric_limits<int>::max();

        for (int i = 0; i < attention_sp_; ++i) {

            int current_load = master_seq_counts_[i]; 

            if (current_load < min_load) {
                min_load = current_load;
                best_idx = i;
            }
        }
        return best_idx;
    } 
    else if (master_selector_ == SPMasterSelector::LeastCache) {
        int best_idx = 0;
        int max_free = -1;

        for (int i = 0; i < attention_sp_; ++i) {

            int free_blocks = block_manager[i]->num_free_blocks();
            
            if (free_blocks > max_free) {
                max_free = free_blocks;
                best_idx = i;
            }
        }
        return best_idx;
    }
    return 0; // Fallback
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
    if (attention_sp_ > 1) {
        // ==========================================
        //  Optimized Strategy (Dynamic SP Size)
        // ==========================================
        auto& block_ctx = seq.block_ctx(BlockContextSlot::ACTIVE);
        
        int num_tokens            = seq.num_tokens;
        int num_segments          = (num_tokens + segment_size - 1) / segment_size;
        int num_segments_per_rank = (num_segments + attention_sp_ - 1) / attention_sp_;
        int initial_num_ranks     = (num_segments + num_segments_per_rank - 1) / num_segments_per_rank;

        if (num_segments_per_rank == 0) num_segments_per_rank = 1;
        if (initial_num_ranks == 0) initial_num_ranks = 1;

        int master_rank = select_master_rank();
        if (master_seq_counts_[master_rank] + 1 > max_num_seqs_) return false;

        auto it_tokens              = num_batched_tokens.find(master_rank);
        int  current_batched_tokens = (it_tokens != num_batched_tokens.end()) ? it_tokens->second : 0;
        if (current_batched_tokens + seq.num_tokens >= max_num_batched_tokens_) return false;

        // Rank selection preparation
        std::vector<std::pair<int, int>> rank_free_count;
        for (const auto& [rank, bm] : block_manager) {
            if (rank != master_rank) {
                rank_free_count.push_back({rank, bm->num_free_blocks()});
            }
        }
        std::sort(rank_free_count.begin(),
                  rank_free_count.end(),
                  [](const std::pair<int, int>& a, const std::pair<int, int>& b) { return a.second > b.second; });

        int start_ranks = initial_num_ranks;
        int end_ranks   = enable_dynamic_sp_size_ ? attention_sp_ : initial_num_ranks;

        for (int target_num_ranks = start_ranks; target_num_ranks <= end_ranks; ++target_num_ranks) {
            
            block_ctx.num_dispatched_tokens.assign(attention_sp_, 0);
            block_ctx.master_sp_idx_ = master_rank;

            // Select participating ranks
            std::vector<int> top_most_free_ranks;
            int              ranks_to_pick = std::min((int)rank_free_count.size(), target_num_ranks - 1);
            for (int i = 0; i < ranks_to_pick; ++i) {
                top_most_free_ranks.push_back(rank_free_count[i].first);
            }
            top_most_free_ranks.push_back(master_rank);

            // =================================================================
            // [New Feature] Non-Uniform Split (Water-filling / Valley-filling)
            // =================================================================
            if (enable_non_uniform_split_) {
                // 1. Collect free blocks info for all participating ranks (including master)
                std::vector<std::pair<int, int>> sorted_ranks; // {sp_idx, free_blocks}
                for (int sp_idx : top_most_free_ranks) {
                    sorted_ranks.push_back({sp_idx, block_manager[sp_idx]->num_free_blocks()});
                }
                
                // 2. Sort participating ranks by free blocks descending (richest first)
                std::sort(sorted_ranks.begin(), sorted_ranks.end(),
                          [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                              return a.second > b.second;
                          });

                long long total_tokens_needed = seq.num_tokens;
                long long final_target_free_tokens = 0;
                int k = 0; // Number of ranks contributing to "water-filling"

                // 3. Find the optimal "water level" (target free tokens)
                // We greedily check if the top k ranks can absorb the load such that
                // their remaining capacity is balanced.
                for (k = 1; k <= (int)sorted_ranks.size(); ++k) {
                    long long sum_free_tokens = 0;
                    for (int i = 0; i < k; ++i) {
                        sum_free_tokens += (long long)sorted_ranks[i].second * kvcache_block_size_;
                    }

                    // If we use top k ranks, what would be the equalized remaining capacity?
                    long long remaining_after_alloc = sum_free_tokens - total_tokens_needed;
                    long long target_free = remaining_after_alloc / k;

                    // If we are at the last rank, or if the calculated target level is 
                    // higher than the next rank's capacity (meaning next rank doesn't need to help),
                    // then we found our split point.
                    if (k == (int)sorted_ranks.size()) {
                        final_target_free_tokens = target_free;
                        break;
                    } else {
                        long long next_rank_free = (long long)sorted_ranks[k].second * kvcache_block_size_;
                        if (target_free >= next_rank_free) {
                            final_target_free_tokens = target_free;
                            break;
                        }
                    }
                }

                // 4. Assign tokens based on the target level
                long long allocated_sum = 0;
                for (int i = 0; i < (int)sorted_ranks.size(); ++i) {
                    int sp_idx = sorted_ranks[i].first;
                    long long current_free = (long long)sorted_ranks[i].second * kvcache_block_size_;
                    
                    // Alloc = Current - Target
                    long long alloc = current_free - final_target_free_tokens;
                    
                    if (alloc < 0) alloc = 0;
                    if (alloc > current_free) alloc = current_free; // Safety cap

                    block_ctx.num_dispatched_tokens[sp_idx] = (int)alloc;
                    allocated_sum += alloc;
                }

                // 5. Handle integer division remainders
                long long remainder = total_tokens_needed - allocated_sum;
                int idx = 0;
                
                // If we allocated too few (remainder > 0), distribute to the richest ranks
                while (remainder > 0) {
                    block_ctx.num_dispatched_tokens[sorted_ranks[idx].first]++;
                    remainder--;
                    idx = (idx + 1) % k;
                }
                
                // If we allocated too many (remainder < 0), take back from richest ranks
                // (This can happen if target calculation slightly overshoots due to integer math)
                while (remainder < 0) {
                    if (block_ctx.num_dispatched_tokens[sorted_ranks[idx].first] > 0) {
                        block_ctx.num_dispatched_tokens[sorted_ranks[idx].first]--;
                        remainder++;
                    }
                    idx = (idx + 1) % k;
                }
            } 
            else {
                // =================================================================
                // Standard Feature: Uniform Split
                // =================================================================
                int total_token_unalloc = seq.num_tokens;
                for (int sp_idx : top_most_free_ranks) {
                    int tokens_to_dispatch                  = std::min(total_token_unalloc, num_segments_per_rank * segment_size);
                    block_ctx.num_dispatched_tokens[sp_idx] = tokens_to_dispatch;
                    total_token_unalloc -= tokens_to_dispatch;
                }
            }

            // Reservation Check
            std::vector<int> master_req_counts(attention_sp_, 0);
            for (const auto& running_seq : running) {
                int m_idx = running_seq->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_;
                if (m_idx >= 0 && m_idx < attention_sp_) master_req_counts[m_idx]++;
            }
            for (const auto& [m_idx, count] : num_seqs) {
                if (m_idx >= 0 && m_idx < attention_sp_) master_req_counts[m_idx] += count;
            }
            master_req_counts[master_rank]++;

            bool memory_check_passed = true;
            for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
                if (block_ctx.num_dispatched_tokens[sp_idx] > 0 || sp_idx == master_rank) {
                    int free_blocks = block_manager[sp_idx]->num_free_blocks();
                    int prefill_tokens = block_ctx.num_dispatched_tokens[sp_idx];
                    int prefill_blocks_needed = (prefill_tokens + kvcache_block_size_ - 1) / kvcache_block_size_;

                    double needed_float = master_req_counts[sp_idx] * reserved_blocks_per_req_;
                    int reservation_blocks_needed = static_cast<int>(std::ceil(needed_float));

                    if (free_blocks < prefill_blocks_needed + reservation_blocks_needed) {
                        memory_check_passed = false;
                        break;
                    }
                    
                    if (!block_manager[sp_idx]->can_allocate(seq)) {
                        memory_check_passed = false;
                        break;
                    }
                }
            }

            if (memory_check_passed) {
                return true;
            }
        }
        return false;
    } 
    else {
        // ==========================================
        //  Naive Strategy (SP=1)
        // ==========================================
        auto& block_ctx = seq.block_ctx(BlockContextSlot::ACTIVE);
        block_ctx.num_dispatched_tokens.assign(attention_sp_, 0);

        int num_tokens            = seq.num_tokens;
        int num_segments          = (num_tokens + segment_size - 1) / segment_size;
        int num_segments_per_rank = (num_segments + attention_sp_ - 1) / attention_sp_;
        int num_ranks             = (num_segments + num_segments_per_rank - 1) / num_segments_per_rank;

        if (num_segments_per_rank == 0)
            num_segments_per_rank = 1;
        if (num_ranks == 0)
            num_ranks = 1;

        int master_rank = select_master_rank();
        if (master_seq_counts_[master_rank] + 1 > max_num_seqs_) {
            return false;
        }

        auto it_tokens              = num_batched_tokens.find(master_rank);
        int  current_batched_tokens = (it_tokens != num_batched_tokens.end()) ? it_tokens->second : 0;
        if (current_batched_tokens + seq.num_tokens >= max_num_batched_tokens_) {
            return false;
        }

        std::vector<std::pair<int, int>> rank_free_count;
        for (const auto& [rank, bm] : block_manager) {
            if (rank != master_rank) {
                rank_free_count.push_back({rank, bm->num_free_blocks()});
            }
        }

        std::sort(rank_free_count.begin(),
                  rank_free_count.end(),
                  [](const std::pair<int, int>& a, const std::pair<int, int>& b) { return a.second > b.second; });

        std::vector<int> top_most_free_ranks;
        int              ranks_to_pick = std::min((int)rank_free_count.size(), num_ranks - 1);
        for (int i = 0; i < ranks_to_pick; ++i) {
            top_most_free_ranks.push_back(rank_free_count[i].first);
        }
        top_most_free_ranks.push_back(master_rank);

        block_ctx.master_sp_idx_ = master_rank;
        int total_token_unalloc  = seq.num_tokens;

        for (int sp_idx : top_most_free_ranks) {
            int tokens_to_dispatch                  = std::min(total_token_unalloc, num_segments_per_rank * segment_size);
            block_ctx.num_dispatched_tokens[sp_idx] = tokens_to_dispatch;
            total_token_unalloc -= tokens_to_dispatch;
        }

        std::vector<int> master_req_counts(attention_sp_, 0);

        for (const auto& running_seq : running) {
            int m_idx = running_seq->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_;
            if (m_idx >= 0 && m_idx < attention_sp_) {
                master_req_counts[m_idx]++;
            }
        }

        for (const auto& [m_idx, count] : num_seqs) {
            if (m_idx >= 0 && m_idx < attention_sp_) {
                master_req_counts[m_idx] += count;
            }
        }

        master_req_counts[master_rank]++;

        for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
            int free_blocks = block_manager[sp_idx]->num_free_blocks();
            
            int prefill_tokens = block_ctx.num_dispatched_tokens[sp_idx];
            int prefill_blocks_needed = (prefill_tokens + kvcache_block_size_ - 1) / kvcache_block_size_;

            double needed_float = master_req_counts[sp_idx] * reserved_blocks_per_req_;
            int reservation_blocks_needed = static_cast<int>(std::ceil(needed_float));

            if (free_blocks < prefill_blocks_needed + reservation_blocks_needed) {
                return false;
            }
        }

        for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
            if (!block_manager[sp_idx]->can_allocate(seq)) {
                return false;
            }
        }

        return true;
    }
}

void SPStateManager::allocate(Sequence& seq)
{
    auto& block_ctx     = seq.block_ctx(BlockContextSlot::ACTIVE);
    int   master_sp_idx = block_ctx.master_sp_idx_;

    if (master_sp_idx >= 0 && master_sp_idx < attention_sp_) {
        master_seq_counts_[master_sp_idx]++;
    }

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
}

void SPStateManager::deallocate(Sequence& seq, BlockContextSlot slot)
{
    auto& block_ctx     = seq.block_ctx(slot);
    int   master_sp_idx = block_ctx.master_sp_idx_;

    if (master_sp_idx >= 0 && master_sp_idx < attention_sp_) {
        if (master_seq_counts_[master_sp_idx] > 0) {
            master_seq_counts_[master_sp_idx]--;
        }
    }

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
}

}  // namespace nanodeploy