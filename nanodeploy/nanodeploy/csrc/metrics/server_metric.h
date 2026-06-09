#pragma once
#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace nanodeploy {

class ServerMetric {
public:
    ServerMetric();

    // Core update methods
    void update_running_requests(int count);
    void update_waiting_requests(int count);
    void update_waiting_migration_requests(int count);
    void add_completed_request();

    void update_waiting_blocks(int head_blocks, int total_blocks);

    void add_tokens(long long num_prompt = 0, long long num_generated = 0);

    void record_prefill_throughput(long long num_tokens, double duration);
    void record_decode_throughput(long long num_tokens, double duration);

    void update_token_usage(int dp_idx, long long num_tokens);
    void update_group_stats(const std::vector<std::vector<int>>& group_send_counts,
                            const std::vector<std::vector<int>>& group_recv_counts);

    // Properties (Getters)
    std::optional<double> avg_prefill_throughput() const;
    std::optional<double> avg_decode_throughput() const;
    std::optional<double> current_prefill_throughput() const;
    std::optional<double> current_decode_throughput() const;

    long long total_token_usage() const;
    double    uptime() const;

    // Logging
    std::string get_metric_report(bool include_detailed = false) const;

    // Public fields
    long long total_tokens           = 0;
    long long total_prompt_tokens    = 0;
    long long total_generated_tokens = 0;

    int num_running_requests           = 0;
    int num_waiting_requests           = 0;
    int num_waiting_migration_requests = 0;
    int num_completed_requests         = 0;

    int num_waiting_head_blocks  = 0;
    int num_waiting_total_blocks = 0;

    std::vector<double>                prefill_throughput_samples;
    std::vector<double>                decode_throughput_samples;
    std::unordered_map<int, long long> token_usage_by_dp;          // dp_idx -> count
    std::unordered_map<int, long long> group_send_request_counts;  // group_id -> count
    std::unordered_map<int, long long> group_recv_request_counts;  // group_id -> count

    double start_time;

private:
    static double current_time();
};

}  // namespace nanodeploy
