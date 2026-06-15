#pragma once
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

namespace dlengine {

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
    // Called once per prefill step the sequence participates in (each chunked
    // prefill step, including the final chunk that emits the first token). The
    // gap since the previous chunk (or since first_scheduled for the first
    // chunk) is appended to prefill_chunk_samples.
    void record_prefill_chunk();

    std::optional<double> ttft() const;
    std::optional<double> e2e_latency() const;
    std::optional<double> avg_tpot_with_queueing() const;
    std::optional<double> avg_tpot_wo_queueing() const;
    std::optional<double> queueing_time_ms() const;
    std::optional<double> decode_queue_time_ms() const;
    std::optional<double> avg_itl() const;
    std::optional<double> p50_itl() const;
    std::optional<double> p99_itl() const;
    // Total prefill compute time (first_scheduled -> first_token), excluding
    // queue wait. Sum of prefill_chunk_samples for a non-preempted sequence.
    std::optional<double> prefill_time_ms() const;

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
    // Chunked-prefill accounting (see record_prefill_chunk()).
    int                   num_prefill_chunks = 0;
    std::vector<double>   prefill_chunk_samples;  // per-chunk latency (ms)
    std::optional<double> last_chunk_time;

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
               std::vector<double>,
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
                                                                     std::vector<double>,
                                                                     int,
                                                                     std::vector<double>>& state);

private:
    static double current_time();
};

}  // namespace dlengine
