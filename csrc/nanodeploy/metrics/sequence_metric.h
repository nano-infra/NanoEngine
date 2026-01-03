#pragma once
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

namespace nanodeploy {

class SequenceMetric {
public:
    explicit SequenceMetric(uint64_t seq_id, int num_prompt_tokens = 0);

    void record_arrival();
    void record_first_scheduled();
    void record_decode_arrival();
    void record_decode_scheduled();
    void record_first_token();
    void record_token();
    void record_completion();

    std::optional<double> ttft() const;
    std::optional<double> e2e_latency() const;
    std::optional<double> avg_tpot_with_queueing() const;
    std::optional<double> avg_tpot_wo_queueing() const;
    std::optional<double> queueing_time_ms() const;
    std::optional<double> decode_queue_time_ms() const;
    std::optional<double> avg_itl() const;
    std::optional<double> p50_itl() const;
    std::optional<double> p99_itl() const;

    void log_metrics() const;

    // Public fields (match Python dataclass mutability)
    uint64_t              seq_id;
    std::optional<double> arrival_time;
    std::optional<double> first_scheduled_time;
    std::optional<double> decode_arrival_time;
    std::optional<double> decode_scheduled_time;
    std::optional<double> first_token_time;
    std::optional<double> completion_time;
    int                   num_prompt_tokens    = 0;
    int                   num_generated_tokens = 0;
    std::vector<double>   itl_samples;
    std::optional<double> last_token_time;

    // For pickle support
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
    getstate() const;

    static std::shared_ptr<SequenceMetric> setstate(const std::tuple<uint64_t,
                                                                     std::optional<double>,
                                                                     std::optional<double>,
                                                                     std::optional<double>,
                                                                     std::optional<double>,
                                                                     std::optional<double>,
                                                                     std::optional<double>,
                                                                     std::optional<double>,
                                                                     int,
                                                                     int,
                                                                     std::vector<double>>& state);

private:
    static double current_time();
};

}  // namespace nanodeploy
