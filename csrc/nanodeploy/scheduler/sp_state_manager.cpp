#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <set>

#include "nanodeploy/sequence/sequence.h"

#include "sp_state_manager.h"

namespace nanodeploy {

namespace {

std::string trim_copy(const std::string& input)
{
    const auto start = input.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) {
        return "";
    }
    const auto end = input.find_last_not_of(" \t\n\r");
    return input.substr(start, end - start + 1);
}

std::vector<SPBucketInterval> parse_bucket_policy(const std::string& text, int max_sp)
{
    std::vector<SPBucketInterval> intervals;
    const std::string trimmed = trim_copy(text);
    if (trimmed.empty()) {
        return intervals;
    }

    std::stringstream ss(trimmed);
    std::string item;
    int prev_high = std::numeric_limits<int>::min();
    while (std::getline(ss, item, ';')) {
        item = trim_copy(item);
        if (item.empty()) {
            continue;
        }
        const auto colon = item.find(':');
        const auto dash = item.find('-', colon == std::string::npos ? 0 : colon + 1);
        if (colon == std::string::npos || dash == std::string::npos) {
            throw std::runtime_error("Invalid dynamic_sp_bucket_policy item: " + item);
        }
        SPBucketInterval interval;
        interval.sp_size = std::stoi(trim_copy(item.substr(0, colon)));
        interval.seq_len_low = std::stoi(trim_copy(item.substr(colon + 1, dash - colon - 1)));
        interval.seq_len_high = std::stoi(trim_copy(item.substr(dash + 1)));
        if (interval.sp_size < 1 || interval.sp_size > max_sp) {
            throw std::runtime_error("Bucket sp_size out of range: " + item);
        }
        if (interval.seq_len_low < 0 || interval.seq_len_high < interval.seq_len_low) {
            throw std::runtime_error("Invalid bucket seq range: " + item);
        }
        if (!intervals.empty() && interval.seq_len_low <= prev_high) {
            throw std::runtime_error("dynamic_sp_bucket_policy ranges must be strictly increasing");
        }
        prev_high = interval.seq_len_high;
        intervals.push_back(interval);
    }
    return intervals;
}

const char* dynamic_sp_size_strategy_name(DynamicSPSizeStrategy strategy)
{
    switch (strategy) {
        case DynamicSPSizeStrategy::Legacy:
            return "legacy";
        case DynamicSPSizeStrategy::Bucket:
            return "bucket";
    }
    return "unknown";
}

}  // namespace

SPStateManager::SPStateManager(const std::string& engine_id,
                               int                attention_sp,
                               int                num_kvcache_blocks,
                               int                kvcache_block_size,
                               int                max_num_seqs,
                               int                max_num_batched_tokens,
                               int                max_num_recv_seqs,
                               double             reserved_blocks_per_req,
                               int                segment_size,
                               const std::string& dynamic_sp_size_strategy,
                               bool               enable_dynamic_sp_bucket_policy,
                               const std::string& dynamic_sp_bucket_policy,
                               bool               enable_non_uniform_split,
                               const std::string& sp_master_selector,
                               int                fixed_sp_size) :
    engine_id_(engine_id),
    attention_sp_(attention_sp),
    max_num_seqs_(max_num_seqs),
    max_num_batched_tokens_(max_num_batched_tokens),
    max_num_recv_seqs_(max_num_recv_seqs),
    reserved_blocks_per_req_(reserved_blocks_per_req),
    kvcache_block_size_(kvcache_block_size),
    segment_size_(segment_size),
    dynamic_sp_size_strategy_(DynamicSPSizeStrategy::Legacy),
    enable_dynamic_sp_bucket_policy_(enable_dynamic_sp_bucket_policy),
    num_recv_seqs_per_sp_(attention_sp, 0),
    enable_non_uniform_split_(enable_non_uniform_split),
    fixed_sp_size_(fixed_sp_size)
{
    // Initialize Strategy
    if (sp_master_selector == "LeastBatch") {
        master_selector_ = SPMasterSelector::LeastBatch;
    } else if (sp_master_selector == "LeastCache") {
        master_selector_ = SPMasterSelector::LeastCache;
    } else {
        master_selector_ = SPMasterSelector::RoundRobin;
    }

    if (dynamic_sp_size_strategy == "legacy") {
        dynamic_sp_size_strategy_ = DynamicSPSizeStrategy::Legacy;
    } else if (dynamic_sp_size_strategy == "bucket") {
        dynamic_sp_size_strategy_ = DynamicSPSizeStrategy::Bucket;
    } else {
        throw std::runtime_error(
            "Unsupported dynamic_sp_size_strategy: " + dynamic_sp_size_strategy);
    }
    dynamic_sp_bucket_policy_ = parse_bucket_policy(dynamic_sp_bucket_policy, attention_sp_);

    if (attention_sp_ <= 0) {
        throw std::runtime_error("attention_sp must be positive to prevent division by zero");
    }
    if (kvcache_block_size_ <= 0) {
        throw std::runtime_error("kvcache_block_size must be positive to prevent division by zero");
    }
    if (fixed_sp_size_ < 0 || fixed_sp_size_ > attention_sp_) {
        throw std::runtime_error("fixed_sp_size must be in [0, attention_sp]");
    }
    if (fixed_sp_size_ > 0
        && (dynamic_sp_size_strategy_ != DynamicSPSizeStrategy::Legacy
            || enable_dynamic_sp_bucket_policy_)) {
        throw std::runtime_error(
            "fixed_sp_size cannot be combined with dynamic SP size strategies");
    }

    // Initialize Running Load Counter
    master_seq_counts_.assign(attention_sp_, 0);

    for (int i = 0; i < attention_sp; ++i) {
        block_manager[i] = std::make_shared<BlockManager>(engine_id, i, num_kvcache_blocks, kvcache_block_size);
    }

    initialize_dummy_seqs();

    std::cerr << "[SPStateManager] Initialized with attention_sp=" << attention_sp_ 
              << ", kvcache_block_size=" << kvcache_block_size_
              << ", reserved_blocks_per_req=" << reserved_blocks_per_req_ 
              << ", segment_size=" << segment_size_
              << ", fixed_sp_size=" << fixed_sp_size_
              << ", dynamic_sp_size_strategy=" << dynamic_sp_size_strategy_name(dynamic_sp_size_strategy_)
              << ", enable_dynamic_sp_bucket_policy=" << enable_dynamic_sp_bucket_policy_
              << std::endl;

}

int SPStateManager::effective_target_sp_size(int requested_sp_size, int num_tokens) const
{
    int target_sp_size = std::max(1, std::min(requested_sp_size, attention_sp_));
    if (fixed_sp_size_ > 0) {
        target_sp_size = std::min(target_sp_size, std::max(1, num_tokens));
    }
    return target_sp_size;
}

std::optional<int> SPStateManager::select_bucket_sp_size(int seq_len) const
{
    if (!enable_dynamic_sp_bucket_policy_) {
        return std::nullopt;
    }
    for (const auto& interval : dynamic_sp_bucket_policy_) {
        if (interval.seq_len_low <= seq_len && seq_len <= interval.seq_len_high) {
            return interval.sp_size;
        }
    }
    return std::nullopt;
}

void SPStateManager::initialize_dummy_seqs()
{
    constexpr int kControlToken = 0;
    constexpr int kDecodeQuantum = 16;

    for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
        std::vector<int> token_ids = {kControlToken};

        auto dummy_seq = std::make_shared<Sequence>(token_ids,
                                                    1.0,   // temperature
                                                    kDecodeQuantum,
                                                    true   // ignore_eos
        );
        dummy_seq->active(engine_id_, attention_sp_, 1);
        dummy_seq->block_ctx().master_sp_idx_ = sp_idx;

        // The first dispatched token is the deterministic bootstrap token.
        // Reserve the complete 16-forward quantum before the object can be
        // used so every worker receives a structurally valid block table.
        dummy_seq->append_token(kControlToken, BlockContextSlot::ACTIVE, sp_idx);
        dummy_seq->num_bootstrap_tokens = 1;

        block_manager[sp_idx]->allocate(*dummy_seq);
        if (!block_manager[sp_idx]->may_append(*dummy_seq, kDecodeQuantum)) {
            throw std::runtime_error(
                "Insufficient KV blocks for the permanent control dummy on SP rank "
                + std::to_string(sp_idx));
        }
        dummy_seqs.push_back(dummy_seq);
    }
}

bool SPStateManager::can_fit_lifetime(const Sequence& seq, int additional_master_tokens) const
{
    if (additional_master_tokens < 0) {
        return false;
    }

    const auto& block_ctx = seq.block_ctx(BlockContextSlot::ACTIVE);
    const int master_sp_idx = block_ctx.master_sp_idx_;
    if (master_sp_idx < 0 || master_sp_idx >= attention_sp_
        || static_cast<int>(block_ctx.num_dispatched_tokens.size()) != attention_sp_) {
        return false;
    }

    for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
        const int control_blocks = num_control_dummy_blocks(sp_idx);
        const int service_blocks = block_manager.at(sp_idx)->num_blocks() - control_blocks;
        const int extra_tokens = sp_idx == master_sp_idx ? additional_master_tokens : 0;
        const int required_tokens = block_ctx.num_dispatched_tokens[sp_idx] + extra_tokens;
        const int required_blocks =
            (required_tokens + kvcache_block_size_ - 1) / kvcache_block_size_;
        if (required_blocks > service_blocks) {
            return false;
        }
    }
    return true;
}

bool SPStateManager::is_control_dummy(const std::shared_ptr<Sequence>& seq) const
{
    return std::find(dummy_seqs.begin(), dummy_seqs.end(), seq) != dummy_seqs.end();
}

int SPStateManager::num_control_dummy_blocks(int sp_idx) const
{
    if (sp_idx < -1 || sp_idx >= attention_sp_) {
        throw std::out_of_range("control dummy SP rank is out of range");
    }

    int total = 0;
    for (const auto& dummy_seq : dummy_seqs) {
        const auto& block_ctx = dummy_seq->block_ctx(BlockContextSlot::ACTIVE);
        if (sp_idx == -1) {
            for (const auto& table : block_ctx.sp_block_table) {
                total += static_cast<int>(table.size());
            }
        }
        else {
            total += static_cast<int>(block_ctx.sp_block_table[sp_idx].size());
        }
    }
    return total;
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

void SPStateManager::apply_planned_placement(Sequence& seq, const PlannedPlacement& placement)
{
    auto& block_ctx = seq.block_ctx(BlockContextSlot::ACTIVE);
    block_ctx.master_sp_idx_ = placement.master_sp_idx;
    block_ctx.num_dispatched_tokens = placement.num_dispatched_tokens;
    block_ctx.block_location.clear();
    block_ctx.sp_block_table.assign(attention_sp_, {});

    if (master_selector_ == SPMasterSelector::RoundRobin) {
        sp_rr_counter_ = (placement.master_sp_idx + 1) % attention_sp_;
    }
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
        int num_segments          = (num_tokens + segment_size_ - 1) / segment_size_;
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
                  [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                      return a.second != b.second ? a.second > b.second
                                                 : a.first < b.first;
                  });

        int start_ranks = initial_num_ranks;
        int end_ranks   = initial_num_ranks;
        bool recompute_segments_for_forced_sp = false;
        if (fixed_sp_size_ > 0) {
            const int forced_num_ranks = effective_target_sp_size(fixed_sp_size_, seq.num_tokens);
            start_ranks = forced_num_ranks;
            end_ranks = forced_num_ranks;
        } else if (dynamic_sp_size_strategy_ == DynamicSPSizeStrategy::Bucket) {
            const int forced_num_ranks = std::max(
                1,
                std::min(
                    attention_sp_,
                    select_bucket_sp_size(seq.num_tokens).value_or(initial_num_ranks)));
            start_ranks = forced_num_ranks;
            end_ranks = forced_num_ranks;
            recompute_segments_for_forced_sp = true;
        }

        for (int target_num_ranks = start_ranks; target_num_ranks <= end_ranks; ++target_num_ranks) {
            int target_num_segments_per_rank = num_segments_per_rank;
            if (recompute_segments_for_forced_sp) {
                target_num_segments_per_rank = (num_segments + target_num_ranks - 1) / target_num_ranks;
                if (target_num_segments_per_rank == 0) {
                    target_num_segments_per_rank = 1;
                }
            }
            
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
            if (enable_non_uniform_split_ && fixed_sp_size_ == 0) {
                // 1. Collect free blocks info for all participating ranks (including master)
                std::vector<std::pair<int, int>> sorted_ranks; // {sp_idx, free_blocks}
                for (int sp_idx : top_most_free_ranks) {
                    sorted_ranks.push_back({sp_idx, block_manager[sp_idx]->num_free_blocks()});
                }
                
                // 2. Sort participating ranks by free blocks descending (richest first)
                std::sort(sorted_ranks.begin(), sorted_ranks.end(),
                          [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                              return a.second != b.second ? a.second > b.second
                                                         : a.first < b.first;
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
                if (fixed_sp_size_ > 0) {
                    int participating_ranks = static_cast<int>(top_most_free_ranks.size());
                    if (participating_ranks <= 0) {
                        return false;
                    }
                    int base_tokens = seq.num_tokens / participating_ranks;
                    int extra_tokens = seq.num_tokens % participating_ranks;
                    for (int i = 0; i < participating_ranks; ++i) {
                        int sp_idx = top_most_free_ranks[i];
                        block_ctx.num_dispatched_tokens[sp_idx] =
                            base_tokens + (i < extra_tokens ? 1 : 0);
                    }
                } else {
                    int total_token_unalloc = seq.num_tokens;
                    for (int sp_idx : top_most_free_ranks) {
                        int tokens_to_dispatch = std::min(
                            total_token_unalloc,
                            target_num_segments_per_rank * segment_size_);
                        block_ctx.num_dispatched_tokens[sp_idx] = tokens_to_dispatch;
                        total_token_unalloc -= tokens_to_dispatch;
                    }
                }
            }

            // The sequence already includes the sampled token that will be
            // consumed by the first decode forward. Its KV slot must live on
            // the master rank. Under load, non-uniform water-filling can give
            // every existing token to ranks with more free blocks and leave
            // the selected master at zero. Move one pending-token slot back
            // to the master while preserving the total placement size.
            if (block_ctx.num_dispatched_tokens[master_rank] == 0) {
                int donor_rank = -1;
                for (int sp_idx : top_most_free_ranks) {
                    if (sp_idx == master_rank) {
                        continue;
                    }
                    if (
                        donor_rank < 0
                        || block_ctx.num_dispatched_tokens[sp_idx]
                            > block_ctx.num_dispatched_tokens[donor_rank]
                    ) {
                        donor_rank = sp_idx;
                    }
                }
                if (
                    donor_rank < 0
                    || block_ctx.num_dispatched_tokens[donor_rank] <= 0
                ) {
                    return false;
                }
                block_ctx.num_dispatched_tokens[donor_rank]--;
                block_ctx.num_dispatched_tokens[master_rank] = 1;
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
                    
                    if (fixed_sp_size_ == 0 && sp_idx != master_rank
                        && block_ctx.num_dispatched_tokens[sp_idx] > 0) {
                        if (num_recv_seqs_per_sp_[sp_idx] >= max_num_recv_seqs_) {
                            memory_check_passed = false;
                            break;
                        }
                    }

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
        int num_segments          = (num_tokens + segment_size_ - 1) / segment_size_;
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
                  [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                      return a.second != b.second ? a.second > b.second
                                                 : a.first < b.first;
                  });

        std::vector<int> top_most_free_ranks;
        int              ranks_to_pick = std::min((int)rank_free_count.size(), num_ranks - 1);
        for (int i = 0; i < ranks_to_pick; ++i) {
            top_most_free_ranks.push_back(rank_free_count[i].first);
        }
        top_most_free_ranks.push_back(master_rank);

        block_ctx.master_sp_idx_ = master_rank;
        int total_token_unalloc  = seq.num_tokens;

        for (int sp_idx : top_most_free_ranks) {
            int tokens_to_dispatch                  = std::min(total_token_unalloc, num_segments_per_rank * segment_size_);
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

    block_ctx.sp_block_table.assign(attention_sp_, {});
    block_ctx.block_location.clear();
    std::fill(block_ctx.num_dispatched_tokens.begin(), block_ctx.num_dispatched_tokens.end(), 0);

    num_running_seqs_--;
    num_running_tokens_ -= seq.num_tokens;
}

}  // namespace nanodeploy
