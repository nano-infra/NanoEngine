#include "server_metric.h"
#include <algorithm>
#include <iomanip>
#include <memory>
#include <numeric>
#include <sstream>

namespace nanodeploy {

double ServerMetric::current_time()
{
    auto now = std::chrono::system_clock::now();
    return std::chrono::duration<double>(now.time_since_epoch()).count();
}

ServerMetric::ServerMetric()
{
    start_time = current_time();
}

void ServerMetric::update_running_requests(int count)
{
    num_running_requests = count;
}

void ServerMetric::update_waiting_requests(int count)
{
    num_waiting_requests = count;
}

void ServerMetric::update_waiting_migration_requests(int count)
{
    num_waiting_migration_requests = count;
}

void ServerMetric::add_completed_request()
{
    num_completed_requests++;
}

void ServerMetric::update_waiting_blocks(int head_blocks, int total_blocks)
{
    num_waiting_head_blocks  = head_blocks;
    num_waiting_total_blocks = total_blocks;
}

void ServerMetric::add_tokens(long long num_prompt, long long num_generated)
{
    total_prompt_tokens += num_prompt;
    total_generated_tokens += num_generated;
    total_tokens += (num_prompt + num_generated);
}

void ServerMetric::record_prefill_throughput(long long num_tokens, double duration)
{
    if (duration > 0) {
        double throughput = static_cast<double>(num_tokens) / duration;
        prefill_throughput_samples.push_back(throughput);
    }
}

void ServerMetric::record_decode_throughput(long long num_tokens, double duration)
{
    if (duration > 0) {
        double throughput = static_cast<double>(num_tokens) / duration;
        decode_throughput_samples.push_back(throughput);
    }
}

void ServerMetric::update_token_usage(int dp_idx, long long num_tokens)
{
    token_usage_by_dp[dp_idx] += num_tokens;
}

void ServerMetric::update_group_stats(const std::vector<std::vector<int>>& group_send_counts,
                                      const std::vector<std::vector<int>>& group_recv_counts)
{
    // Clear old stats? Or accumulate?
    // The requirement is "statistic of each rank's ... send and receive request quantity".
    // Usually metrics accumulate over time.
    // However, if we want to print per-step stats in Python, we do that from ScheduleResult.
    // If we want total accumulated stats in metrics, we accumulate here.
    // The user said "add metric and log logic", suggesting both per-step logging and accumulated metrics.

    // Let's implement accumulation here.
    for (size_t dp_idx = 0; dp_idx < group_send_counts.size(); ++dp_idx) {
        for (size_t group_id = 0; group_id < group_send_counts[dp_idx].size(); ++group_id) {
            // Mapping: global group_id?
            // Wait, group_send_counts is [dp_idx][group_id].
            // If we want per-rank stats, we need to map (dp_idx, group_id) to a unique rank ID?
            // Or just store by group_id if user implies group ranks within a DP group?
            // Given "group_batch_sizes" is [dp][group], we should probably aggregate by group index if group is uniform
            // across DP? Or more likely, the user wants stats per logical rank.

            // NOTE: In `llm_engine.py`, `group_batch_sizes` is `[dp_idx][group_id]`.
            // Let's store aggregated stats for simplicity, or flattened.
            // Let's map global group_rank = dp_idx * group_size + group_id.
            // But `server_metric.h` has `std::unordered_map<int, long long> group_send_request_counts;`.
            // Let's use `dp_idx * group_size + group_id` as keys.
            // Using size of vector to infer group_size.

            int global_group_rank = dp_idx * group_send_counts[dp_idx].size() + group_id;
            group_send_request_counts[global_group_rank] += group_send_counts[dp_idx][group_id];
            group_recv_request_counts[global_group_rank] += group_recv_counts[dp_idx][group_id];
        }
    }
}

std::optional<double> ServerMetric::avg_prefill_throughput() const
{
    if (prefill_throughput_samples.empty())
        return std::nullopt;
    double sum = std::accumulate(prefill_throughput_samples.begin(), prefill_throughput_samples.end(), 0.0);
    return sum / prefill_throughput_samples.size();
}

std::optional<double> ServerMetric::avg_decode_throughput() const
{
    if (decode_throughput_samples.empty())
        return std::nullopt;
    double sum = std::accumulate(decode_throughput_samples.begin(), decode_throughput_samples.end(), 0.0);
    return sum / decode_throughput_samples.size();
}

std::optional<double> ServerMetric::current_prefill_throughput() const
{
    if (prefill_throughput_samples.empty())
        return std::nullopt;
    return prefill_throughput_samples.back();
}

std::optional<double> ServerMetric::current_decode_throughput() const
{
    if (decode_throughput_samples.empty())
        return std::nullopt;
    return decode_throughput_samples.back();
}

long long ServerMetric::total_token_usage() const
{
    long long total = 0;
    for (const auto& kv : token_usage_by_dp) {
        total += kv.second;
    }
    return total;
}

double ServerMetric::uptime() const
{
    return current_time() - start_time;
}

std::string ServerMetric::get_metric_report(bool include_detailed) const
{
    std::stringstream ss;

    double prefill_tput = 0.0;
    if (!prefill_throughput_samples.empty())
        prefill_tput = prefill_throughput_samples.back();

    double decode_tput = 0.0;
    if (!decode_throughput_samples.empty())
        decode_tput = decode_throughput_samples.back();

    ss << "ServerMetric - "
       << "Running/Waiting/Waiting migration: " << num_running_requests << "/" << num_waiting_requests << "/"
       << num_waiting_migration_requests << ", "
       << "Completed: " << num_completed_requests << ", "
       << "Tokens: " << total_tokens << " (prompt: " << total_prompt_tokens << ", gen: " << total_generated_tokens
       << "), "
       << "Throughput: Prefill " << std::fixed << std::setprecision(0) << prefill_tput << " tok/s, "
       << "Decode " << decode_tput << " tok/s";

    if (include_detailed && !token_usage_by_dp.empty()) {
        ss << "\nDetailed Token Usage:";
        for (const auto& kv : token_usage_by_dp) {
            ss << "\n  DP[" << kv.first << "] token usage: " << kv.second;
        }
    }

    if (include_detailed && !group_send_request_counts.empty()) {
        ss << "\nDetailed Group Request Stats (Send/Recv):";
        // Iterate through sorted keys for stable output?
        std::vector<int> rank_ids;
        for (const auto& kv : group_send_request_counts)
            rank_ids.push_back(kv.first);
        std::sort(rank_ids.begin(), rank_ids.end());

        for (int rank : rank_ids) {
            long long send = group_send_request_counts.at(rank);
            long long recv = 0;
            if (group_recv_request_counts.count(rank)) {
                recv = group_recv_request_counts.at(rank);
            }
            ss << "\n  Rank[" << rank << "]: Send=" << send << ", Recv=" << recv;
        }
    }

    return ss.str();
}

}  // namespace nanodeploy
