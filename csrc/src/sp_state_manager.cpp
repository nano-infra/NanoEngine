#include "sp_state_manager.h"
#include <algorithm>
#include <random>
#include <iostream>

namespace nanodeploy {

SPStateManager::SPStateManager(const std::optional<std::string>& engine_id,
                               int attention_sp,
                               int num_kvcache_blocks,
                               int kvcache_block_size,
                               int max_num_seqs,
                               int max_num_batched_tokens)
    : engine_id_(engine_id),
      attention_sp_(attention_sp),
      max_num_seqs_(max_num_seqs),
      max_num_batched_tokens_(max_num_batched_tokens) {
    
    for (int i = 0; i < attention_sp; ++i) {
        block_manager[i] = std::make_shared<BlockManager>(
            engine_id, i, num_kvcache_blocks, kvcache_block_size
        );
    }
    
    initialize_dummy_seqs();
}

void SPStateManager::initialize_dummy_seqs() {
    // Use a fixed seed for reproducibility or random device
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 7999);

    for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
        std::vector<int> token_ids = {dis(gen)};
        
        auto dummy_seq = std::make_shared<Sequence>(
            token_ids,
            1.0, // temperature
            256, // max_tokens
            false, // ignore_eos
            engine_id_,
            sp_idx // master_sp_rank
        );
        
        dummy_seq->append_token(dis(gen), engine_id_, sp_idx);
        
        block_manager[sp_idx]->allocate(*dummy_seq);
        dummy_seqs.push_back(dummy_seq);
    }
}

int SPStateManager::next_sp_idx() {
    int idx = sp_rr_counter_;
    sp_rr_counter_ = (sp_rr_counter_ + 1) % attention_sp_;
    return idx;
}

bool SPStateManager::can_append(Sequence& seq, int num_tokens) {
    int master_sp_idx = seq.block_ctx(engine_id_).master_sp_idx;
    if (block_manager.find(master_sp_idx) == block_manager.end()) {
        return false;
    }
    return block_manager[master_sp_idx]->can_append(seq, num_tokens);
}

void SPStateManager::may_append(Sequence& seq, int num_tokens) {
    int master_sp_idx = seq.block_ctx(engine_id_).master_sp_idx;
    if (block_manager.find(master_sp_idx) != block_manager.end()) {
        block_manager[master_sp_idx]->may_append(seq, num_tokens);
    }
}

bool SPStateManager::can_allocate(Sequence& seq,
                                  const std::unordered_map<int, int>& num_seqs,
                                  const std::unordered_map<int, int>& num_batched_tokens) {
    // Step 1: cal num_blocks and num_blocks_per_rank
    auto& block_ctx = seq.block_ctx(engine_id_);
    block_ctx.num_dispatched_tokens.clear();
    
    int num_tokens = seq.num_tokens;
    int num_segments = (num_tokens + segment_size - 1) / segment_size;
    int num_segments_per_rank = (num_segments + attention_sp_ - 1) / attention_sp_;
    int num_ranks = (num_segments + num_segments_per_rank - 1) / num_segments_per_rank;
    
    // Handle division by zero if num_segments_per_rank is 0 (empty sequence?)
    // Assuming num_tokens > 0, so num_segments >= 1.
    if (num_segments_per_rank == 0) num_segments_per_rank = 1; 
    if (num_ranks == 0) num_ranks = 1;

    int master_rank = next_sp_idx();
    
    // Check constraints
    auto it_seqs = num_seqs.find(master_rank);
    int current_num_seqs = (it_seqs != num_seqs.end()) ? it_seqs->second : 0;
    if (current_num_seqs >= max_num_seqs_) {
        return false;
    }
    
    auto it_tokens = num_batched_tokens.find(master_rank);
    int current_batched_tokens = (it_tokens != num_batched_tokens.end()) ? it_tokens->second : 0;
    if (current_batched_tokens + seq.num_tokens >= max_num_batched_tokens_) {
        return false;
    }
    
    // Rank selection logic
    std::vector<std::pair<int, int>> rank_free_count;
    for (const auto& [rank, bm] : block_manager) {
        if (rank != master_rank) {
            rank_free_count.push_back({rank, bm->num_free_blocks()});
        }
    }
    
    std::sort(rank_free_count.begin(), rank_free_count.end(), 
              [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                  return a.second < b.second;
              });
              
    std::vector<int> top_least_free_ranks;
    int ranks_to_pick = std::min((int)rank_free_count.size(), num_ranks - 1);
    for (int i = 0; i < ranks_to_pick; ++i) {
        top_least_free_ranks.push_back(rank_free_count[i].first);
    }
    top_least_free_ranks.push_back(master_rank);
    
    // Step 2: allocation setup
    block_ctx.master_sp_idx = master_rank;
    int total_token_unalloc = seq.num_tokens;
    
    for (int sp_idx : top_least_free_ranks) {
        int tokens_to_dispatch = std::min(total_token_unalloc, num_segments_per_rank * segment_size);
        block_ctx.num_dispatched_tokens[sp_idx] = tokens_to_dispatch;
        total_token_unalloc -= tokens_to_dispatch;
    }
    
    // Check if all involved block managers can allocate
    for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
        if (!block_manager[sp_idx]->can_allocate(seq)) {
            return false;
        }
    }
    
    return true;
}

void SPStateManager::allocate(Sequence& seq) {
    auto& block_ctx = seq.block_ctx(engine_id_);
    int master_sp_idx = block_ctx.master_sp_idx;
    
    for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
        if (sp_idx != master_sp_idx) {
            block_manager[sp_idx]->allocate(seq);
        }
    }
    block_manager[master_sp_idx]->allocate(seq);
}

void SPStateManager::deallocate(Sequence& seq) {
    for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
        block_manager[sp_idx]->deallocate(seq);
    }
    
    auto& block_ctx = seq.block_ctx(engine_id_);
    block_ctx.sp_block_table.clear();
    block_ctx.block_location.clear();
    block_ctx.num_dispatched_tokens.clear();
}

} // namespace nanodeploy
