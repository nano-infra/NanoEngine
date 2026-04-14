#include "sequence_metric.h"
#include <cmath>
#include <iostream>

namespace nanodeploy {

SequenceMetric::SequenceMetric(uint64_t seq_id, int num_prompt_tokens):
    seq_id(seq_id), num_prompt_tokens(num_prompt_tokens)
{
}

double SequenceMetric::current_time()
{
    using namespace std::chrono;
    return duration_cast<duration<double>>(high_resolution_clock::now().time_since_epoch()).count();
}

void SequenceMetric::record_arrival()
{
    if (!arrival_time.has_value()) {
        arrival_time = current_time();
    }
}

void SequenceMetric::record_first_scheduled()
{
    if (!first_scheduled_time.has_value()) {
        first_scheduled_time = current_time();
    }
}

void SequenceMetric::record_decode_arrival()
{
    if (!decode_arrival_time.has_value()) {
        decode_arrival_time = current_time();
    }
}

void SequenceMetric::record_decode_scheduled()
{
    if (!decode_scheduled_time.has_value()) {
        decode_scheduled_time = current_time();
    }
}

void SequenceMetric::record_first_token()
{
    if (!first_token_time.has_value()) {
        first_token_time = current_time();
        last_token_time  = first_token_time;
    }
}

void SequenceMetric::record_token()
{
    double now = current_time();
    if (last_token_time.has_value()) {
        double itl = (now - last_token_time.value()) * 1000.0;  // ms
        itl_samples.push_back(itl);
    }
    last_token_time = now;
    num_generated_tokens++;
}

void SequenceMetric::record_step_tokens(int num_tokens, double step_itl_ms)
{
    // Record ITL samples based on step timing instead of per-token timing
    // step_itl_ms = step_duration_ms / loop_count (fair share per token slot)
    
    // If this is the first batch of tokens, the first token is TTFT, not ITL.
    // So we record (num_tokens - 1) ITL samples.
    int samples_to_record = num_tokens;
    if (num_generated_tokens == 0) {
        samples_to_record = std::max(0, num_tokens - 1);
    }

    for (int i = 0; i < samples_to_record; ++i) {
        itl_samples.push_back(step_itl_ms);
    }
    num_generated_tokens += num_tokens;
    last_token_time = current_time();
}

void SequenceMetric::record_completion()
{
    completion_time = current_time();
}

void SequenceMetric::on_preemption()
{
    // Clear ITL history to avoid outliers across preemption gap
    itl_samples.clear();

    // Reset scheduling timers so they can be re-recorded upon re-scheduling
    first_scheduled_time  = std::nullopt;
    decode_scheduled_time = std::nullopt;

    // Reset token timers to strictly measure performance of the new run
    first_token_time = std::nullopt;
    last_token_time  = std::nullopt;

    // Reset generated token count for the new run
    num_generated_tokens = 0;
}

std::optional<double> SequenceMetric::ttft() const
{
    if (!first_token_time.has_value() || !arrival_time.has_value()) {
        return std::nullopt;
    }
    return (first_token_time.value() - arrival_time.value()) * 1000.0;
}

std::optional<double> SequenceMetric::e2e_latency() const
{
    if (!completion_time.has_value() || !arrival_time.has_value()) {
        return std::nullopt;
    }
    return (completion_time.value() - arrival_time.value()) * 1000.0;
}
// loop_count bias
std::optional<double> SequenceMetric::avg_tpot_with_queueing() const
{
    auto e2e = e2e_latency();
    if (!e2e.has_value() || num_generated_tokens == 0) {
        return std::nullopt;
    }
    return e2e.value() / num_generated_tokens;
}
// loop_count bias
std::optional<double> SequenceMetric::avg_tpot_wo_queueing() const
{
    // if (!completion_time.has_value() || !first_token_time.has_value() || num_generated_tokens <= 1) {
    //     return std::nullopt;
    // }
    // return ((completion_time.value() - first_token_time.value()) * 1000.0) / (num_generated_tokens - 1);
    if (!completion_time.has_value() || !first_scheduled_time.has_value() || num_generated_tokens == 0) {
        return std::nullopt;
    }
    return ((completion_time.value() - first_scheduled_time.value()) * 1000.0) / num_generated_tokens;
}

std::optional<double> SequenceMetric::queueing_time_ms() const
{
    if (!first_scheduled_time.has_value() || !arrival_time.has_value()) {
        return std::nullopt;
    }
    return (first_scheduled_time.value() - arrival_time.value()) * 1000.0;
}

std::optional<double> SequenceMetric::decode_queue_time_ms() const
{
    if (!decode_scheduled_time.has_value() || !decode_arrival_time.has_value()) {
        return std::nullopt;
    }
    return (decode_scheduled_time.value() - decode_arrival_time.value()) * 1000.0;
}

std::optional<double> SequenceMetric::avg_itl() const
{
    if (itl_samples.empty()) {
        return std::nullopt;
    }
    double sum = std::accumulate(itl_samples.begin(), itl_samples.end(), 0.0);
    return sum / itl_samples.size();
}

std::optional<double> SequenceMetric::avg_itl_exclude_first() const
{
    if (itl_samples.size() <= 1) {
        return std::nullopt;
    }
    double sum = std::accumulate(itl_samples.begin() + 1, itl_samples.end(), 0.0);
    return sum / (itl_samples.size() - 1);
}

std::optional<double> SequenceMetric::avg_itl_with_decode_queue() const
{
    auto itl = avg_itl();
    auto decode_queue = decode_queue_time_ms();
    
    if (!itl.has_value() || !decode_queue.has_value() || num_generated_tokens == 0) {
        return std::nullopt;
    }
    
    return itl.value() + (decode_queue.value() / num_generated_tokens);
}

std::optional<double> SequenceMetric::p50_itl() const
{
    if (itl_samples.empty()) {
        return std::nullopt;
    }
    std::vector<double> sorted_itl = itl_samples;
    size_t              n          = sorted_itl.size() / 2;
    std::nth_element(sorted_itl.begin(), sorted_itl.begin() + n, sorted_itl.end());
    return sorted_itl[n];
}

std::optional<double> SequenceMetric::p99_itl() const
{
    if (itl_samples.empty()) {
        return std::nullopt;
    }
    std::vector<double> sorted_itl = itl_samples;
    size_t              n          = static_cast<size_t>(sorted_itl.size() * 0.99);
    if (n >= sorted_itl.size())
        n = sorted_itl.size() - 1;
    std::nth_element(sorted_itl.begin(), sorted_itl.begin() + n, sorted_itl.end());
    return sorted_itl[n];
}

void SequenceMetric::log_metrics() const
{
    // Implementation can be added if needed, or rely on Python side logging
}

std::tuple<uint64_t,
           std::optional<double>,
           std::optional<double>,
           std::optional<double>,
           std::optional<double>,
           std::optional<double>,
           std::optional<double>,
           std::optional<double>,
           int,
           int,
           std::vector<double>>
SequenceMetric::getstate() const
{
    return std::make_tuple(seq_id,
                           arrival_time,
                           first_scheduled_time,
                           decode_arrival_time,
                           decode_scheduled_time,
                           first_token_time,
                           completion_time,
                           last_token_time,
                           num_prompt_tokens,
                           num_generated_tokens,
                           itl_samples);
}

std::shared_ptr<SequenceMetric> SequenceMetric::setstate(const std::tuple<uint64_t,
                                                                          std::optional<double>,
                                                                          std::optional<double>,
                                                                          std::optional<double>,
                                                                          std::optional<double>,
                                                                          std::optional<double>,
                                                                          std::optional<double>,
                                                                          std::optional<double>,
                                                                          int,
                                                                          int,
                                                                          std::vector<double>>& state)
{
    auto metric                   = std::make_shared<SequenceMetric>(std::get<0>(state));
    metric->arrival_time          = std::get<1>(state);
    metric->first_scheduled_time  = std::get<2>(state);
    metric->decode_arrival_time   = std::get<3>(state);
    metric->decode_scheduled_time = std::get<4>(state);
    metric->first_token_time      = std::get<5>(state);
    metric->completion_time       = std::get<6>(state);
    metric->last_token_time       = std::get<7>(state);
    metric->num_prompt_tokens     = std::get<8>(state);
    metric->num_generated_tokens  = std::get<9>(state);
    metric->itl_samples           = std::get<10>(state);

    return metric;
}

}  // namespace nanodeploy
