#include "sp_size_policy.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

namespace nanodeploy {

SPSizePolicy::SPSizePolicy(const SPSizeConfig& config) : config_(config)
{
}

SPSizeDecision SPSizePolicy::determine_sp_size(
    int                      num_tokens,
    const std::vector<int>&  free_blocks_per_rank,
    const std::vector<int>&  used_blocks_per_rank,
    const std::vector<int>&  long_used_per_rank,
    const std::vector<int>&  batch_size_per_rank,
    int                      total_blocks_per_rank,
    LoadStatistics&          stats,
    int                      max_sp,
    int                      block_size) const
{
    SPSizeDecision decision;
    decision.sp_size = 1;
    decision.due_to_imbalance = false;
    decision.due_to_pressure = false;
    
    // Always use LeastBatch for master rank selection
    decision.master_rank = select_master_rank_least_batch(batch_size_per_rank);

    // 1. Evaluate request characteristics for "Life-cycle" estimation
    // Predict future growth: Prompt + half of expected output length (based on steady-state expectation)
    int estimated_total_tokens = num_tokens + static_cast<int>(stats.avg_output_length() / 2.0f);
    int required_blocks = (estimated_total_tokens + block_size - 1) / block_size;
    
    // 2. Identify system-wide "Short Request" profile and workload reservation
    int   avg_short_blocks    = stats.estimate_short_req_blocks(block_size);
    float avg_short_bs_rank   = stats.avg_short_batch_size();
    
    // Effective capacity reservation per rank to maintain short request throughput
    // reserve = avg_short_batch_size * avg_short_request_size
    int reserve_per_rank = static_cast<int>(std::ceil(avg_short_bs_rank * avg_short_blocks));

    // Use segment-based method if configured (fallback)
    if (config_.mode == SPSizeMode::Segment) {
        decision.sp_size = determine_sp_size_segment(num_tokens, max_sp);
        // Manual uniform split for segment mode
        decision.dispatch_tokens.assign(free_blocks_per_rank.size(), 0);
        int segment_size = config_.segment_size;
        int remaining = num_tokens;
        // Simplified segment distribution for compatibility
        for (int i = 0; i < decision.sp_size; ++i) {
            int rank = (decision.master_rank + i) % max_sp;
            int alloc = std::min(remaining, segment_size);
            decision.dispatch_tokens[rank] = alloc;
            remaining -= alloc;
        }
        return decision;
    }

    // === Path A: Workload-Aware Reservation (Maintain Communication Balance) ===
    // Goal: Choose smallest SP such that the long request load doesn't squeeze out short requests
    // We simulate water-filling using real long_used_per_rank data
    
    int best_sp_for_balance = 1;
    std::vector<int> best_dispatch;

    for (int sp = 1; sp <= max_sp; ++sp) {
        // Pick participating ranks: master + sp-1 ranks with most effective free space
        std::vector<std::pair<int, int>> rank_effective_free;
        for (int i = 0; i < max_sp; ++i) {
            if (i == decision.master_rank) continue;
            // Effective free for long requests = total - reserve - long_used
            int eff_free = total_blocks_per_rank - reserve_per_rank - long_used_per_rank[i];
            rank_effective_free.push_back({i, std::max(0, std::min(eff_free, free_blocks_per_rank[i]))});
        }
        std::sort(rank_effective_free.begin(), rank_effective_free.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        std::vector<int> participating_ranks = {decision.master_rank};
        for (int i = 0; i < sp - 1 && i < (int)rank_effective_free.size(); ++i) {
            participating_ranks.push_back(rank_effective_free[i].first);
        }

        // Simulate Water-filling distribution
        std::vector<int> simulated_dispatch = simulate_water_filling(
            num_tokens, long_used_per_rank, participating_ranks, block_size);
        
        // Check if this distribution fits within the effective capacity of ALL participating ranks
        bool fits = true;
        for (int rank_idx : participating_ranks) {
            int tokens = simulated_dispatch[rank_idx];
            int blocks = (tokens + block_size - 1) / block_size;
            int eff_limit = total_blocks_per_rank - reserve_per_rank;
            
            // Check against both physical free and workload-aware reservation
            if (long_used_per_rank[rank_idx] + blocks > eff_limit || 
                blocks > free_blocks_per_rank[rank_idx]) {
                fits = false;
                break;
            }
        }

        if (fits) {
            best_sp_for_balance = sp;
            best_dispatch = simulated_dispatch;
            break;
        }
        if (sp == max_sp) {
            best_sp_for_balance = max_sp;
            best_dispatch = simulated_dispatch;
        }
    }

    decision.sp_size = best_sp_for_balance;
    decision.dispatch_tokens = best_dispatch;
    if (decision.sp_size > 1) decision.due_to_pressure = true;

    // === Path B: Attention Balance (Computational Bottleneck) ===
    // If Path A didn't lead to max_sp, check if increasing SP further benefits Attention latency.
    // Added Persistence Check to prevent selfish SP decisions in high-load serving.
    if (decision.sp_size < max_sp) {
        float current_imbalance = stats.kvcache_imbalance_ratio();
        float avg_imbalance = stats.avg_kvcache_imbalance();
        float threshold = stats.imbalance_threshold();

        // Path B only operates if the imbalance is both EXTREME and PERSISTENT.
        // This ensures we don't waste bandwidth on imbalances that could be naturally 
        // balanced by future DP-routed requests.
        bool is_extremely_imbalanced = current_imbalance > threshold * 1.5f;
        bool is_persistently_imbalanced = avg_imbalance > threshold;

        if (is_extremely_imbalanced && is_persistently_imbalanced) {
            int optimal_sp_for_latency = calculate_optimal_sp_for_balance(
                num_tokens, used_blocks_per_rank, block_size, max_sp);
            
            if (optimal_sp_for_latency > decision.sp_size) {
                // Re-simulate with the new sp size
                std::vector<std::pair<int, int>> rank_used;
                for (int i = 0; i < max_sp; ++i) {
                    if (i == decision.master_rank) continue;
                    rank_used.push_back({i, used_blocks_per_rank[i]});
                }
                std::sort(rank_used.begin(), rank_used.end(),
                          [](const auto& a, const auto& b) { return a.second < b.second; });

                std::vector<int> participating_ranks = {decision.master_rank};
                for (int i = 0; i < optimal_sp_for_latency - 1 && i < (int)rank_used.size(); ++i) {
                    participating_ranks.push_back(rank_used[i].first);
                }

                std::vector<int> latency_dispatch = simulate_water_filling(
                    num_tokens, used_blocks_per_rank, participating_ranks, block_size);
                
                // Verify physical capacity for the latency-driven decision
                bool physical_fits = true;
                for (int rank_idx : participating_ranks) {
                    int blocks = (latency_dispatch[rank_idx] + block_size - 1) / block_size;
                    if (blocks > free_blocks_per_rank[rank_idx]) {
                        physical_fits = false;
                        break;
                    }
                }

                if (physical_fits) {
                    decision.sp_size = optimal_sp_for_latency;
                    decision.dispatch_tokens = latency_dispatch;
                    decision.due_to_imbalance = true;
                }
            }
        }
    }

    return decision;
}

std::vector<int> SPSizePolicy::simulate_water_filling(
    int                     num_tokens,
    const std::vector<int>& base_used_per_rank,
    const std::vector<int>& participating_ranks,
    int                     block_size) const
{
    std::vector<int> dispatch(base_used_per_rank.size(), 0);
    if (participating_ranks.empty()) return dispatch;

    // Sort participating ranks by base usage ascending
    std::vector<std::pair<int, int>> sorted_ranks;
    for (int rank_idx : participating_ranks) {
        sorted_ranks.push_back({rank_idx, base_used_per_rank[rank_idx]});
    }
    std::sort(sorted_ranks.begin(), sorted_ranks.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    long long total_needed = num_tokens;
    int k = 0;
    long long final_target_level = 0;

    // Find water level
    for (k = 1; k <= (int)sorted_ranks.size(); ++k) {
        long long current_sum_blocks = 0;
        for (int i = 0; i < k; ++i) current_sum_blocks += sorted_ranks[i].second;
        
        long long needed_blocks = (total_needed + block_size - 1) / block_size;
        long long target_level = (current_sum_blocks + needed_blocks + k - 1) / k;

        if (k == (int)sorted_ranks.size() || target_level <= sorted_ranks[k].second) {
            final_target_level = target_level;
            break;
        }
    }

    // Allocate based on target level
    long long allocated_tokens = 0;
    for (int i = 0; i < k; ++i) {
        int sp_idx = sorted_ranks[i].first;
        long long diff_blocks = final_target_level - sorted_ranks[i].second;
        long long alloc = diff_blocks * block_size;
        
        dispatch[sp_idx] = (int)std::min(alloc, total_needed - allocated_tokens);
        allocated_tokens += dispatch[sp_idx];
    }

    // Distribute remainder tokens
    int remainder = (int)(total_needed - allocated_tokens);
    int idx = 0;
    while (remainder > 0) {
        dispatch[sorted_ranks[idx % k].first]++;
        remainder--;
        idx++;
    }

    return dispatch;
}

int SPSizePolicy::determine_sp_size_segment(int num_tokens, int max_sp) const
{
    int segment_size          = config_.segment_size;
    int num_segments          = (num_tokens + segment_size - 1) / segment_size;
    int num_segments_per_rank = (num_segments + max_sp - 1) / max_sp;
    if (num_segments_per_rank == 0) num_segments_per_rank = 1;
    int num_ranks             = (num_segments + num_segments_per_rank - 1) / num_segments_per_rank;

    return std::min(num_ranks, max_sp);
}

int SPSizePolicy::select_master_rank_least_batch(const std::vector<int>& batch_size_per_rank) const
{
    if (batch_size_per_rank.empty()) return 0;
    return (int)(std::min_element(batch_size_per_rank.begin(), batch_size_per_rank.end()) - batch_size_per_rank.begin());
}

int SPSizePolicy::calculate_optimal_sp_for_balance(
    int                     num_tokens,
    const std::vector<int>& used_blocks_per_rank,
    int                     block_size,
    int                     max_sp) const
{
    if (max_sp <= 1) return 1;
    
    float best_latency = std::numeric_limits<float>::max();
    int best_sp = 1;
    
    for (int sp = 1; sp <= max_sp; ++sp) {
        // Pick best sp ranks for latency (lowest used)
        std::vector<int> rank_indices(used_blocks_per_rank.size());
        std::iota(rank_indices.begin(), rank_indices.end(), 0);
        std::sort(rank_indices.begin(), rank_indices.end(), 
                 [&](int a, int b){ return used_blocks_per_rank[a] < used_blocks_per_rank[b]; });
        
        std::vector<int> participating;
        for(int i=0; i<sp; ++i) participating.push_back(rank_indices[i]);

        std::vector<int> simulated_tokens = simulate_water_filling(num_tokens, used_blocks_per_rank, participating, block_size);
        std::vector<int> simulated_used = used_blocks_per_rank;
        for(size_t i=0; i<simulated_used.size(); ++i) {
            simulated_used[i] += (simulated_tokens[i] + block_size - 1) / block_size;
        }
        
        float attn_lat = estimate_attention_latency(simulated_used);
        float comm_lat = estimate_sp_comm_overhead(sp, num_tokens);
        float total = attn_lat + comm_lat;
        
        if (total < best_latency) {
            best_latency = total;
            best_sp = sp;
        }
    }
    return best_sp;
}

float SPSizePolicy::estimate_attention_latency(const std::vector<int>& kvcache_per_rank) const
{
    if (kvcache_per_rank.empty()) return 0.0f;
    int max_kvcache = *std::max_element(kvcache_per_rank.begin(), kvcache_per_rank.end());
    return config_.attn_cost_per_token * max_kvcache;
}

float SPSizePolicy::estimate_sp_comm_overhead(int sp_size, int num_tokens) const
{
    if (sp_size <= 1) return 0.0f;
    return config_.sp_comm_alpha * (sp_size - 1) + config_.sp_comm_beta * num_tokens;
}

}  // namespace nanodeploy
