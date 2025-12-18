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
    token_usage_by_dp[dp_idx] = num_tokens;
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

    return ss.str();
}

}  // namespace nanodeploy
