#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <numeric>
#include <vector>

namespace nanodeploy {

/**
 * @brief LoadStatistics maintains comprehensive statistics for load-aware SP decisions.
 * 
 * This class tracks:
 * 1. Request characteristics (prompt/output lengths)
 * 2. System load (waiting queue sizes, arrival rates)
 * 3. KVCache distribution across ranks (for imbalance detection)
 * 
 * All thresholds are learned from historical traces, not manually configured.
 */
class LoadStatistics {
public:
    /**
     * @brief Construct a new LoadStatistics object
     * 
     * @param attention_sp Number of SP ranks
     * @param initial_avg_prompt_length Initial estimate (will be updated from trace)
     * @param initial_avg_output_length Initial estimate (will be updated from trace)
     * @param window_size Size of sliding window for statistics
     */
    LoadStatistics(int   attention_sp             = 1,
                   float initial_avg_prompt_length = 1024.0f,
                   float initial_avg_output_length = 256.0f,
                   int   window_size              = 1000);

    // === Request Statistics ===
    
    void record_request(int prompt_length, int output_length);
    
    float avg_prompt_length() const;
    float avg_output_length() const;
    
    float avg_short_prompt_length() const;
    float avg_short_output_length() const;

    float prompt_length_p50() const;
    float prompt_length_p90() const;
    float prompt_length_percentile(float percentile) const;
    
    bool is_long_request(int prompt_length) const;
    int  estimate_short_req_blocks(int block_size) const;

    // === System Load Statistics ===
    
    /**
     * @brief Record the current waiting queue size at each scheduling step
     */
    void record_waiting_queue_size(int queue_size);
    
    /**
     * @brief Record the current number of running short requests
     */
    void record_short_batch_size(int short_seq_count);

    float avg_short_batch_size() const;

    /**
     * @brief Record request arrival for QPS calculation
     */
    void record_arrival();
    
    /**
     * @brief Get the estimated number of expected waiting requests
     * 
     * This replaces the fixed target_short_req_capacity with a learned value.
     */
    float expected_waiting_requests() const;
    
    /**
     * @brief Get the current request arrival rate (requests per second)
     */
    float arrival_rate() const;

    // === KVCache Distribution Statistics ===
    
    /**
     * @brief Record the current KVCache usage per rank
     * 
     * @param used_blocks_per_rank Vector of used block counts per rank
     */
    void record_kvcache_distribution(const std::vector<int>& used_blocks_per_rank);
    
    /**
     * @brief Calculate KVCache imbalance ratio
     * 
     * Returns max(used) / avg(used), where 1.0 means perfectly balanced.
     * Higher values indicate more imbalance.
     */
    float kvcache_imbalance_ratio() const;
    
    /**
     * @brief Calculate KVCache coefficient of variation (CV)
     * 
     * CV = std_dev / mean, normalized measure of dispersion.
     */
    float kvcache_cv() const;
    
    /**
     * @brief Get the historical average KVCache imbalance
     */
    float avg_kvcache_imbalance() const;
    
    /**
     * @brief Get current KVCache per rank (most recent snapshot)
     */
    const std::vector<int>& current_kvcache_per_rank() const { return current_kvcache_per_rank_; }

    // === Adaptive Threshold Learning ===
    
    /**
     * @brief Get the learned "long request" threshold multiplier
     * 
     * This is adaptively learned based on when SP actually helps performance.
     * Default starts at 3.0x but adjusts based on trace.
     */
    float long_req_threshold() const { return learned_long_req_threshold_; }
    
    /**
     * @brief Get the learned KVCache imbalance threshold
     * 
     * When imbalance exceeds this, SP should be considered even under low load.
     */
    float imbalance_threshold() const { return learned_imbalance_threshold_; }
    
    /**
     * @brief Update learned thresholds based on observed performance
     * 
     * @param sp_size The SP size that was used
     * @param was_beneficial Whether using this SP size improved performance
     */
    void update_learned_thresholds(int sp_size, bool was_beneficial);

    // === Accessors ===
    
    size_t num_samples() const { return prompt_length_window_.size(); }
    bool   has_sufficient_samples() const { return prompt_length_window_.size() >= static_cast<size_t>(min_samples_for_stats_); }
    int    attention_sp() const { return attention_sp_; }

private:
    double current_time_seconds() const;

    // Configuration
    int   attention_sp_;
    float initial_avg_prompt_length_;
    float initial_avg_output_length_;
    int   window_size_;
    int   min_samples_for_stats_ = 10;

    // Request statistics
    std::deque<int> prompt_length_window_;
    std::deque<int> output_length_window_;
    long long       prompt_length_sum_ = 0;
    long long       output_length_sum_ = 0;

    std::deque<int> short_prompt_length_window_;
    std::deque<int> short_output_length_window_;
    long long       short_prompt_length_sum_ = 0;
    long long       short_output_length_sum_ = 0;

    // System load statistics
    std::deque<int>    waiting_queue_sizes_;
    std::deque<int>    short_batch_size_window_;
    long long          short_batch_size_sum_ = 0;
    std::deque<double> arrival_timestamps_;
    
    // KVCache distribution tracking
    std::vector<int>              current_kvcache_per_rank_;
    std::deque<float>             kvcache_imbalance_history_;
    
    // Learned thresholds (adaptive)
    float learned_long_req_threshold_  = 3.0f;   // Multiplier for avg_prompt_length
    float learned_imbalance_threshold_ = 1.5f;   // KVCache imbalance ratio threshold
    
    // For online threshold learning
    int   sp_decision_count_      = 0;
    int   beneficial_sp_count_    = 0;
    float running_benefit_ratio_  = 0.5f;
};

}  // namespace nanodeploy
