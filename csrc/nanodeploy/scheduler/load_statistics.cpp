#include "load_statistics.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace nanodeploy {

LoadStatistics::LoadStatistics(int   attention_sp,
                               float initial_avg_prompt_length,
                               float initial_avg_output_length,
                               int   window_size) :
    attention_sp_(attention_sp),
    initial_avg_prompt_length_(initial_avg_prompt_length),
    initial_avg_output_length_(initial_avg_output_length),
    window_size_(window_size),
    current_kvcache_per_rank_(attention_sp, 0)
{
}

double LoadStatistics::current_time_seconds() const
{
    auto now = std::chrono::steady_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
}

// === Request Statistics ===

void LoadStatistics::record_request(int prompt_length, int output_length)
{
    prompt_length_window_.push_back(prompt_length);
    output_length_window_.push_back(output_length);
    prompt_length_sum_ += prompt_length;
    output_length_sum_ += output_length;

    if (static_cast<int>(prompt_length_window_.size()) > window_size_) {
        prompt_length_sum_ -= prompt_length_window_.front();
        output_length_sum_ -= output_length_window_.front();
        prompt_length_window_.pop_front();
        output_length_window_.pop_front();
    }

    // Also record as a short request if it qualifies
    if (!is_long_request(prompt_length)) {
        short_prompt_length_window_.push_back(prompt_length);
        short_output_length_window_.push_back(output_length);
        short_prompt_length_sum_ += prompt_length;
        short_output_length_sum_ += output_length;

        if (static_cast<int>(short_prompt_length_window_.size()) > window_size_) {
            short_prompt_length_sum_ -= short_prompt_length_window_.front();
            short_output_length_sum_ -= short_output_length_window_.front();
            short_prompt_length_window_.pop_front();
            short_output_length_window_.pop_front();
        }
    }
}

float LoadStatistics::avg_prompt_length() const
{
    if (prompt_length_window_.empty()) {
        return initial_avg_prompt_length_;
    }
    return static_cast<float>(prompt_length_sum_) / prompt_length_window_.size();
}

float LoadStatistics::avg_output_length() const
{
    if (output_length_window_.empty()) {
        return initial_avg_output_length_;
    }
    return static_cast<float>(output_length_sum_) / output_length_window_.size();
}

float LoadStatistics::avg_short_prompt_length() const
{
    if (short_prompt_length_window_.empty()) {
        return initial_avg_prompt_length_ * 0.5f; // Assume short is half of overall avg if no data
    }
    return static_cast<float>(short_prompt_length_sum_) / short_prompt_length_window_.size();
}

float LoadStatistics::avg_short_output_length() const
{
    if (short_output_length_window_.empty()) {
        return initial_avg_output_length_ * 0.8f; // Assume short is slightly less than overall avg
    }
    return static_cast<float>(short_output_length_sum_) / short_output_length_window_.size();
}

float LoadStatistics::prompt_length_percentile(float percentile) const
{
    if (prompt_length_window_.empty()) {
        return initial_avg_prompt_length_;
    }

    std::vector<int> sorted(prompt_length_window_.begin(), prompt_length_window_.end());
    std::sort(sorted.begin(), sorted.end());

    float index = (percentile / 100.0f) * (sorted.size() - 1);
    int   lower = static_cast<int>(std::floor(index));
    int   upper = static_cast<int>(std::ceil(index));

    if (lower == upper || upper >= static_cast<int>(sorted.size())) {
        return static_cast<float>(sorted[lower]);
    }

    float fraction = index - lower;
    return sorted[lower] * (1.0f - fraction) + sorted[upper] * fraction;
}

float LoadStatistics::prompt_length_p50() const
{
    return prompt_length_percentile(50.0f);
}

float LoadStatistics::prompt_length_p90() const
{
    return prompt_length_percentile(90.0f);
}

bool LoadStatistics::is_long_request(int prompt_length) const
{
    float avg = avg_prompt_length();
    return static_cast<float>(prompt_length) > learned_long_req_threshold_ * avg;
}

int LoadStatistics::estimate_short_req_blocks(int block_size) const
{
    float avg_short_len = avg_short_prompt_length() + avg_short_output_length();
    return static_cast<int>(std::ceil(avg_short_len / block_size));
}

// === System Load Statistics ===

void LoadStatistics::record_waiting_queue_size(int queue_size)
{
    waiting_queue_sizes_.push_back(queue_size);
    if (static_cast<int>(waiting_queue_sizes_.size()) > window_size_) {
        waiting_queue_sizes_.pop_front();
    }
}

void LoadStatistics::record_short_batch_size(int short_seq_count)
{
    short_batch_size_window_.push_back(short_seq_count);
    short_batch_size_sum_ += short_seq_count;

    if (static_cast<int>(short_batch_size_window_.size()) > window_size_) {
        short_batch_size_sum_ -= short_batch_size_window_.front();
        short_batch_size_window_.pop_front();
    }
}

float LoadStatistics::avg_short_batch_size() const
{
    if (short_batch_size_window_.empty()) {
        return 0.0f;
    }
    float avg_total = static_cast<float>(short_batch_size_sum_) / short_batch_size_window_.size();
    return avg_total / attention_sp_;
}

void LoadStatistics::record_arrival()
{
    double now = current_time_seconds();
    arrival_timestamps_.push_back(now);
    
    // Keep only recent arrivals (last 60 seconds)
    while (!arrival_timestamps_.empty() && 
           (now - arrival_timestamps_.front()) > 60.0) {
        arrival_timestamps_.pop_front();
    }
}

float LoadStatistics::expected_waiting_requests() const
{
    if (waiting_queue_sizes_.empty()) {
        return 10.0f;  // Default estimate
    }
    
    // Use the 75th percentile of historical waiting queue sizes
    // This gives a reasonable upper estimate without being too conservative
    std::vector<int> sorted(waiting_queue_sizes_.begin(), waiting_queue_sizes_.end());
    std::sort(sorted.begin(), sorted.end());
    
    size_t p75_idx = static_cast<size_t>(0.75 * (sorted.size() - 1));
    float p75_value = static_cast<float>(sorted[p75_idx]);
    
    // Also consider arrival rate - if requests are arriving fast, expect more waiting
    float rate = arrival_rate();
    if (rate > 0) {
        // Estimate: avg_wait_time * arrival_rate
        // Assume avg processing time is proportional to avg prompt length
        float avg_process_time = avg_prompt_length() / 1000.0f;  // rough estimate in seconds
        float little_law_estimate = rate * avg_process_time;
        
        // Blend historical and arrival-rate based estimates
        return std::max(p75_value, little_law_estimate);
    }
    
    return std::max(p75_value, 5.0f);  // At least 5
}

float LoadStatistics::arrival_rate() const
{
    if (arrival_timestamps_.size() < 2) {
        return 0.0f;
    }
    
    double time_span = arrival_timestamps_.back() - arrival_timestamps_.front();
    if (time_span <= 0) {
        return 0.0f;
    }
    
    return static_cast<float>(arrival_timestamps_.size() - 1) / time_span;
}

// === KVCache Distribution Statistics ===

void LoadStatistics::record_kvcache_distribution(const std::vector<int>& used_blocks_per_rank)
{
    current_kvcache_per_rank_ = used_blocks_per_rank;
    
    // Calculate and record imbalance
    float imbalance = kvcache_imbalance_ratio();
    kvcache_imbalance_history_.push_back(imbalance);
    
    if (static_cast<int>(kvcache_imbalance_history_.size()) > window_size_) {
        kvcache_imbalance_history_.pop_front();
    }
}

float LoadStatistics::kvcache_imbalance_ratio() const
{
    if (current_kvcache_per_rank_.empty()) {
        return 1.0f;  // No imbalance
    }
    
    long long sum = 0;
    int max_used = 0;
    for (int used : current_kvcache_per_rank_) {
        sum += used;
        max_used = std::max(max_used, used);
    }
    
    if (sum == 0) {
        return 1.0f;  // No KVCache used, no imbalance
    }
    
    float avg = static_cast<float>(sum) / current_kvcache_per_rank_.size();
    if (avg == 0) {
        return 1.0f;
    }
    
    return static_cast<float>(max_used) / avg;
}

float LoadStatistics::kvcache_cv() const
{
    if (current_kvcache_per_rank_.size() < 2) {
        return 0.0f;
    }
    
    // Calculate mean
    double sum = 0;
    for (int used : current_kvcache_per_rank_) {
        sum += used;
    }
    double mean = sum / current_kvcache_per_rank_.size();
    
    if (mean == 0) {
        return 0.0f;
    }
    
    // Calculate variance
    double variance = 0;
    for (int used : current_kvcache_per_rank_) {
        double diff = used - mean;
        variance += diff * diff;
    }
    variance /= current_kvcache_per_rank_.size();
    
    // CV = std_dev / mean
    return static_cast<float>(std::sqrt(variance) / mean);
}

float LoadStatistics::avg_kvcache_imbalance() const
{
    if (kvcache_imbalance_history_.empty()) {
        return 1.0f;
    }
    
    float sum = 0;
    for (float imb : kvcache_imbalance_history_) {
        sum += imb;
    }
    return sum / kvcache_imbalance_history_.size();
}

// === Adaptive Threshold Learning ===

void LoadStatistics::update_learned_thresholds(int sp_size, bool was_beneficial)
{
    (void)sp_size;  // Reserved for future use (e.g., per-SP-size tracking)
    
    sp_decision_count_++;
    if (was_beneficial) {
        beneficial_sp_count_++;
    }
    
    // Update running benefit ratio with exponential moving average
    // This ratio represents how often SP helps balance the system
    float alpha = 0.1f;  // Learning rate for the benefit ratio
    float benefit = was_beneficial ? 1.0f : 0.0f;
    running_benefit_ratio_ = alpha * benefit + (1.0f - alpha) * running_benefit_ratio_;
    
    // Adjust thresholds based on the benefit ratio
    // If SP is frequently beneficial (> 70%), we should be more aggressive (lower thresholds)
    // If SP is rarely beneficial (< 30%), we should be more conservative (raise thresholds)
    
    if (sp_decision_count_ >= 20) {  // Start adjusting after a small warmup
        if (running_benefit_ratio_ > 0.7f) {
            // SP is often beneficial, lower thresholds to use SP more often
            learned_long_req_threshold_ = std::max(1.5f, learned_long_req_threshold_ * 0.98f);
            learned_imbalance_threshold_ = std::max(1.1f, learned_imbalance_threshold_ * 0.98f);
        } else if (running_benefit_ratio_ < 0.3f) {
            // SP is rarely beneficial, raise thresholds to use SP more sparingly
            learned_long_req_threshold_ = std::min(5.0f, learned_long_req_threshold_ * 1.02f);
            learned_imbalance_threshold_ = std::min(2.5f, learned_imbalance_threshold_ * 1.02f);
        }
    }
}

}  // namespace nanodeploy
