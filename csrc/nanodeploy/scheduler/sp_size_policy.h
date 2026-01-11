#pragma once

#include <algorithm>
#include <climits>
#include <string>
#include <vector>

#include "load_statistics.h"

namespace nanodeploy {

class Sequence;

/**
 * @brief Mode for determining SP (Sequence Parallel) size
 */
enum class SPSizeMode {
    Segment,    ///< Original segment_size based strategy (for compatibility)
    LoadAware,  ///< New load-aware strategy based on memory pressure and KVCache balance
};

/**
 * @brief Configuration for SP size policy
 * 
 * Note: Most thresholds are now learned from traces, not manually configured.
 * The config only contains non-adaptive parameters.
 */
struct SPSizeConfig {
    SPSizeMode mode = SPSizeMode::Segment;

    // Communication cost model parameters (can be calibrated per-hardware)
    // SP overhead = sp_comm_alpha * sp_size + sp_comm_beta * data_size
    float sp_comm_alpha = 0.001f;  // Per-rank fixed overhead (ms)
    float sp_comm_beta  = 0.00001f;  // Per-token communication cost (ms/token)
    
    // Attention cost model parameter
    // Attention time = attn_cost_per_token * kvcache_tokens (ms)
    float attn_cost_per_token = 0.0001f;  // ms per KVCache token

    // Initial statistics (will be updated from trace)
    float initial_avg_prompt_length = 1024.0f;
    float initial_avg_output_length = 256.0f;
    
    int stats_window_size = 1000;

    // Segment mode parameters (for backward compatibility)
    int segment_size = 65536;
};

/**
 * @brief Result of SP size decision with additional metadata
 */
struct SPSizeDecision {
    int  sp_size;           ///< Recommended SP size
    int  master_rank;       ///< Recommended master rank (using LeastBatch)
    bool due_to_imbalance;  ///< True if SP was increased due to KVCache imbalance
    bool due_to_pressure;   ///< True if SP was increased due to memory pressure
    
    std::vector<int> dispatch_tokens; ///< Detailed token distribution per rank
};

/**
 * @brief SPSizePolicy determines the optimal SP size for each request.
 */
class SPSizePolicy {
public:
    explicit SPSizePolicy(const SPSizeConfig& config = SPSizeConfig());

    /**
     * @brief Determine the optimal SP size and master rank for a sequence
     * 
     * @param num_tokens Number of tokens in the sequence (prompt length)
     * @param free_blocks_per_rank Free blocks on each SP rank
     * @param used_blocks_per_rank Used blocks on each SP rank (total current usage)
     * @param long_used_per_rank Blocks used by long requests specifically (on-the-fly calculated)
     * @param batch_size_per_rank Current batch size on each rank
     * @param total_blocks_per_rank Total blocks per rank
     * @param stats Load statistics (includes learned thresholds)
     * @param max_sp Maximum allowed SP size
     * @param block_size KVCache block size in tokens
     * @return SPSizeDecision with sp_size, master_rank and dispatch_tokens
     */
    SPSizeDecision determine_sp_size(
        int                      num_tokens,
        const std::vector<int>&  free_blocks_per_rank,
        const std::vector<int>&  used_blocks_per_rank,
        const std::vector<int>&  long_used_per_rank,
        const std::vector<int>&  batch_size_per_rank,
        int                      total_blocks_per_rank,
        LoadStatistics&          stats,
        int                      max_sp,
        int                      block_size) const;

    /**
     * @brief Calculate SP size using the original segment-based method
     */
    int determine_sp_size_segment(int num_tokens, int max_sp) const;

    /**
     * @brief Select master rank using LeastBatch strategy
     */
    int select_master_rank_least_batch(const std::vector<int>& batch_size_per_rank) const;

    // === Accessors ===
    const SPSizeConfig& config() const { return config_; }
    SPSizeConfig&       config() { return config_; }
    
    SPSizeMode mode() const { return config_.mode; }
    void       set_mode(SPSizeMode mode) { config_.mode = mode; }

private:
    SPSizeConfig config_;

    /**
     * @brief Calculate optimal SP size to minimize total latency
     * 
     * Models: total_latency = attention_latency + sp_communication_overhead
     * Uses water-filling approach to balance KVCache across participating ranks.
     */
    int calculate_optimal_sp_for_balance(
        int                     num_tokens,
        const std::vector<int>& used_blocks_per_rank,
        int                     block_size,
        int                     max_sp) const;

    /**
     * @brief Simulate non-uniform token distribution (Water-filling)
     * 
     * @param num_tokens Tokens to distribute
     * @param base_used_per_rank Base usage (e.g. long_used or total_used)
     * @param participating_ranks Indices of ranks participating in SP
     * @param block_size Tokens per block
     * @return Vector of tokens dispatched to each rank
     */
    std::vector<int> simulate_water_filling(
        int                     num_tokens,
        const std::vector<int>& base_used_per_rank,
        const std::vector<int>& participating_ranks,
        int                     block_size) const;

    /**
     * @brief Estimate attention latency for a given KVCache distribution
     */
    float estimate_attention_latency(const std::vector<int>& kvcache_per_rank) const;

    /**
     * @brief Estimate SP communication overhead
     */
    float estimate_sp_comm_overhead(int sp_size, int num_tokens) const;
};

}  // namespace nanodeploy
