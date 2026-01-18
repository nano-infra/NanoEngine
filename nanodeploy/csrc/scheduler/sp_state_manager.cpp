#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>

#include "nanodeploy/csrc/sequence/sequence.h"

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
    // ==========================================
    // Step 1: Determine SP Size (Number of Ranks)
    // ==========================================
    // Use segment logic to calculate minimum ranks needed to minimize communication overhead.
    int num_tokens   = seq.num_tokens;
    int num_segments = (num_tokens + segment_size - 1) / segment_size;

    int num_ranks_needed = std::max(1, std::min(attention_sp_, num_segments));

    // ==========================================
    // Step 2: Get Current Load Info for All Ranks
    // ==========================================
    struct RankLoadInfo {
        int       id;
        long long current_tokens;  // KV Cache load
        int       current_seqs;    // Master load
    };
    std::vector<RankLoadInfo> all_ranks;
    all_ranks.reserve(attention_sp_);

    for (int i = 0; i < attention_sp_; ++i) {
        // 1. Calculate Token load: running + scheduled (in queue)
        long long tokens = num_running_tokens_per_sp_[i];
        if (num_batched_tokens.count(i)) {
            tokens += num_batched_tokens.at(i);
        }

        // 2. Calculate Sequence load: running + scheduled (in queue)
        int seqs = num_running_seqs_per_sp_[i];
        if (num_seqs.count(i)) {
            seqs += num_seqs.at(i);
        }

        all_ranks.push_back({i, tokens, seqs});
    }

    // ==========================================
    // Step 3: Select Participants (Prioritize KV Cache)
    // ==========================================
    // Sort by token load ascending; pick emptiest ranks first.
    std::sort(all_ranks.begin(), all_ranks.end(), [](const RankLoadInfo& a, const RankLoadInfo& b) {
        return a.current_tokens < b.current_tokens;
    });

    // Select top K ranks as participants.
    std::vector<RankLoadInfo> participants;
    participants.reserve(num_ranks_needed);
    for (int i = 0; i < num_ranks_needed; ++i) {
        participants.push_back(all_ranks[i]);
    }

    // ==========================================
    // Step 4: Distribute Tokens (Water-Filling Algorithm)
    // ==========================================
    auto& block_ctx = seq.block_ctx(BlockContextSlot::ACTIVE);
    block_ctx.num_dispatched_tokens.assign(attention_sp_, 0);

    std::vector<long long> simulated_loads;
    for (const auto& p : participants) {
        simulated_loads.push_back(p.current_tokens);
    }
    std::vector<int> alloc_counts(participants.size(), 0);

    int tokens_remaining = num_tokens;

    // Use kvcache_block_size_ as unit to avoid fragmented blocks and wasted memory.
    const int CHUNK_SIZE = kvcache_block_size_;

    while (tokens_remaining > 0) {
        // 1. Find rank with lowest simulated load.
        auto min_it = std::min_element(simulated_loads.begin(), simulated_loads.end());
        int  idx    = std::distance(simulated_loads.begin(), min_it);

        // 2. Allocate chunk (or remaining tokens).
        int current_alloc = std::min(CHUNK_SIZE, tokens_remaining);

        simulated_loads[idx] += current_alloc;
        alloc_counts[idx] += current_alloc;
        tokens_remaining -= current_alloc;
    }

    // Apply allocation results to block_ctx.
    for (size_t i = 0; i < participants.size(); ++i) {
        int rank_id                              = participants[i].id;
        block_ctx.num_dispatched_tokens[rank_id] = alloc_counts[i];
    }

    // ==========================================
    // Step 5: Select Master (Load Balancing)
    // ==========================================
    // Choose participant with fewest sequences as Master.
    auto min_seq_it =
        std::min_element(participants.begin(), participants.end(), [](const RankLoadInfo& a, const RankLoadInfo& b) {
            return a.current_seqs < b.current_seqs;
        });

    int master_rank          = min_seq_it->id;
    block_ctx.master_sp_idx_ = master_rank;

    // ==========================================
    // Step 6: Resource and Physical Memory Checks
    // ==========================================

    // 1. Check Master's Max Seqs limit using estimated load.
    if (min_seq_it->current_seqs + 1 > max_num_seqs_) {
        return false;
    }

    // 2. Check Master's Max Batched Tokens limit.
    long long master_pending_tokens = num_batched_tokens.count(master_rank) ? num_batched_tokens.at(master_rank) : 0;
    // Maintain original logic: throttle if Master is overloaded, even if tokens are distributed.
    if (master_pending_tokens + seq.num_tokens >= max_num_batched_tokens_) {
        return false;
    }

    // 3. Check physical memory (BlockManager) for all participants.
    // Crucial: validates logical calculation against actual free blocks.
    for (size_t i = 0; i < participants.size(); ++i) {
        int rank_id = participants[i].id;
        // BlockManager checks based on num_dispatched_tokens set above.
        if (!block_manager[rank_id]->can_allocate(seq)) {
            return false;
        }
    }

    return true;
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
