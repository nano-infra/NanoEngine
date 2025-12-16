#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

// -------------------------------------------------------------------------
// 0. 辅助工具与 Opaque 声明
// -------------------------------------------------------------------------

inline double get_time_sec()
{
    using namespace std::chrono;
    return duration<double>(system_clock::now().time_since_epoch()).count();
}

inline std::string generate_uuid()
{
    py::object uuid_mod = py::module::import("uuid");
    return uuid_mod.attr("uuid4")().attr("__str__")().cast<std::string>();
}

enum class SequenceStatus {
    WAITING        = 1,
    RUNNING        = 2,
    FINISHED       = 3,
    TO_BE_MIGRATED = 4
};

struct BlockContext;

PYBIND11_MAKE_OPAQUE(std::vector<int>);
PYBIND11_MAKE_OPAQUE(std::map<int, int>);
PYBIND11_MAKE_OPAQUE(std::map<int, std::vector<int>>);
PYBIND11_MAKE_OPAQUE(std::map<std::optional<std::string>, BlockContext>);

// -------------------------------------------------------------------------
// 1. Metrics 类定义
// -------------------------------------------------------------------------

class SequenceMetric {
public:
    std::string           seq_id;
    std::optional<double> arrival_time;
    std::optional<double> decode_first_scheduled_time;
    std::optional<double> first_token_time;
    std::optional<double> completion_time;
    std::optional<double> last_token_time;

    int                 num_prompt_tokens    = 0;
    int                 num_generated_tokens = 0;
    std::vector<double> itl_samples;

    SequenceMetric() = default;
    SequenceMetric(std::string id, int prompt_len): seq_id(id), num_prompt_tokens(prompt_len) {}

    void record_arrival()
    {
        if (!arrival_time.has_value())
            arrival_time = get_time_sec();
    }

    void record_first_scheduled()
    {
        if (!decode_first_scheduled_time.has_value())
            decode_first_scheduled_time = get_time_sec();
    }

    void on_token_generated()
    {
        double now = get_time_sec();

        if (num_generated_tokens == 0) {
            if (!first_token_time.has_value()) {
                first_token_time = now;
            }
            last_token_time      = now;
            num_generated_tokens = 1;
        }
        else {
            if (last_token_time.has_value()) {
                double itl = (now - last_token_time.value()) * 1000.0;
                itl_samples.push_back(itl);
            }
            last_token_time = now;
            num_generated_tokens++;
        }
    }

    void record_completion()
    {
        completion_time = get_time_sec();
    }

    std::optional<double> ttft()
    {
        if (!first_token_time.has_value() || !arrival_time.has_value())
            return std::nullopt;
        return (first_token_time.value() - arrival_time.value()) * 1000.0;
    }

    std::optional<double> e2e_latency()
    {
        if (!completion_time.has_value() || !arrival_time.has_value())
            return std::nullopt;
        return (completion_time.value() - arrival_time.value()) * 1000.0;
    }

    std::optional<double> avg_itl()
    {
        if (itl_samples.empty())
            return std::nullopt;
        double sum = std::accumulate(itl_samples.begin(), itl_samples.end(), 0.0);
        return sum / itl_samples.size();
    }

    py::object get_itl_stats()
    {
        if (itl_samples.empty())
            return py::none();
        std::vector<double> sorted = itl_samples;
        std::sort(sorted.begin(), sorted.end());
        size_t n   = sorted.size();
        double p50 = sorted[n / 2];
        double p99 = sorted[(size_t)(n * 0.99)];
        return py::make_tuple(p50, p99);
    }

    std::optional<double> avg_tpot_with_queueing()
    {
        if (num_generated_tokens == 0 || !completion_time.has_value() || !arrival_time.has_value())
            return std::nullopt;
        return ((completion_time.value() - arrival_time.value()) * 1000.0) / num_generated_tokens;
    }

    std::optional<double> avg_tpot_wo_queueing()
    {
        if (num_generated_tokens == 0 || !completion_time.has_value() || !decode_first_scheduled_time.has_value())
            return std::nullopt;
        return ((completion_time.value() - decode_first_scheduled_time.value()) * 1000.0) / num_generated_tokens;
    }
};

class ServerMetric {
public:
    int64_t total_tokens                   = 0;
    int64_t total_prompt_tokens            = 0;
    int64_t total_generated_tokens         = 0;
    int     num_running_requests           = 0;
    int     num_waiting_requests           = 0;
    int     num_waiting_migration_requests = 0;
    int     num_completed_requests         = 0;

    std::vector<double> prefill_throughput_samples;
    std::vector<double> decode_throughput_samples;
    std::map<int, int>  token_usage_by_dp;
    double              start_time;

    ServerMetric()
    {
        start_time = get_time_sec();
    }

    void add_tokens(int num_prompt, int num_generated)
    {
        total_prompt_tokens += num_prompt;
        total_generated_tokens += num_generated;
        total_tokens += (num_prompt + num_generated);
    }

    void update_running_requests(int count)
    {
        num_running_requests = count;
    }
    void update_waiting_requests(int count)
    {
        num_waiting_requests = count;
    }
    void update_waiting_migration_requests(int count)
    {
        num_waiting_migration_requests = count;
    }
    void add_completed_request()
    {
        num_completed_requests++;
    }

    void record_prefill_throughput(int num_tokens, double duration)
    {
        if (duration > 0) {
            prefill_throughput_samples.push_back(num_tokens / duration);
        }
    }

    void record_decode_throughput(int num_tokens, double duration)
    {
        if (duration > 0) {
            decode_throughput_samples.push_back(num_tokens / duration);
        }
    }

    void update_token_usage(int dp_idx, int tokens)
    {
        token_usage_by_dp[dp_idx] = tokens;
    }

    std::optional<double> avg_prefill_throughput()
    {
        if (prefill_throughput_samples.empty())
            return std::nullopt;
        double sum = std::accumulate(prefill_throughput_samples.begin(), prefill_throughput_samples.end(), 0.0);
        return sum / prefill_throughput_samples.size();
    }

    std::optional<double> avg_decode_throughput()
    {
        if (decode_throughput_samples.empty())
            return std::nullopt;
        double sum = std::accumulate(decode_throughput_samples.begin(), decode_throughput_samples.end(), 0.0);
        return sum / decode_throughput_samples.size();
    }

    std::optional<double> current_prefill_throughput()
    {
        if (prefill_throughput_samples.empty())
            return std::nullopt;
        return prefill_throughput_samples.back();
    }

    std::optional<double> current_decode_throughput()
    {
        if (decode_throughput_samples.empty())
            return std::nullopt;
        return decode_throughput_samples.back();
    }

    int64_t total_token_usage()
    {
        int64_t sum = 0;
        for (auto const& [key, val] : token_usage_by_dp) {
            sum += val;
        }
        return sum;
    }

    double uptime()
    {
        return get_time_sec() - start_time;
    }

    py::dict get_summary()
    {
        py::dict d;
        d["uptime_seconds"]         = uptime();
        d["total_requests"]         = num_completed_requests;
        d["running_requests"]       = num_running_requests;
        d["waiting_requests"]       = num_waiting_requests;
        d["total_tokens"]           = total_tokens;
        d["total_prompt_tokens"]    = total_prompt_tokens;
        d["total_generated_tokens"] = total_generated_tokens;

        auto avg_pre = avg_prefill_throughput();
        d["avg_prefill_throughput"] =
            avg_pre.has_value() ? py::object(py::float_(avg_pre.value())) : py::object(py::none());

        auto avg_dec = avg_decode_throughput();
        d["avg_decode_throughput"] =
            avg_dec.has_value() ? py::object(py::float_(avg_dec.value())) : py::object(py::none());

        auto cur_pre = current_prefill_throughput();
        d["current_prefill_throughput"] =
            cur_pre.has_value() ? py::object(py::float_(cur_pre.value())) : py::object(py::none());

        auto cur_dec = current_decode_throughput();
        d["current_decode_throughput"] =
            cur_dec.has_value() ? py::object(py::float_(cur_dec.value())) : py::object(py::none());

        d["total_token_usage"] = total_token_usage();
        return d;
    }
};

// -------------------------------------------------------------------------
// 2. 核心数据结构 (BlockContext, Sequence)
// -------------------------------------------------------------------------

struct BlockContext {
    std::optional<std::string>       engine_id;
    int                              dp_idx;
    int                              master_sp_idx;
    int                              attention_sp;
    int                              attention_dp;
    std::vector<std::pair<int, int>> block_location;
    std::map<int, int>               num_dispatched_tokens;
    std::map<int, std::vector<int>>  sp_block_table;

    BlockContext() = default;
    BlockContext(std::optional<std::string> eid, int dp, int m_sp, int att_sp, int att_dp):
        engine_id(eid), dp_idx(dp), master_sp_idx(m_sp), attention_sp(att_sp), attention_dp(att_dp)
    {
    }

    bool operator==(const BlockContext& other) const
    {
        return engine_id == other.engine_id && dp_idx == other.dp_idx && master_sp_idx == other.master_sp_idx
               && attention_sp == other.attention_sp && attention_dp == other.attention_dp
               && block_location == other.block_location && num_dispatched_tokens == other.num_dispatched_tokens
               && sp_block_table == other.sp_block_table;
    }
};

class Sequence {
private:
    Sequence() = default;

public:
    static constexpr int               block_size = 256;
    static inline std::atomic<int64_t> global_counter{0};

    std::string                seq_id;
    SequenceStatus             status = SequenceStatus::WAITING;
    std::vector<int>           token_ids;
    int                        last_token              = -1;
    int64_t                    num_tokens              = 0;
    int64_t                    num_prompt_tokens       = 0;
    int64_t                    num_checkpointed_tokens = 0;
    int64_t                    num_cached_tokens       = 0;
    std::optional<std::string> backup_engine_id;
    std::optional<std::string> active_engine_id;

    std::map<std::optional<std::string>, BlockContext> block_ctx_map;

    std::shared_ptr<SequenceMetric> metric;
    int                             pending_token_count = 0;

    std::optional<float> temperature;
    std::optional<int>   max_tokens;
    bool                 ignore_eos = false;

    static std::shared_ptr<Sequence> create_empty()
    {
        struct MakeSharedEnabler: public Sequence {};
        return std::make_shared<MakeSharedEnabler>();
    }

    Sequence(std::vector<int>           t_ids,
             py::object                 sampling_params,
             std::optional<std::string> engine_id,
             int                        master_sp_rank)
    {
        if (sampling_params.is_none()) {
            try {
                py::module sp_mod = py::module::import("nanodeploy.sampling_params");
                sampling_params   = sp_mod.attr("SamplingParams")();
            }
            catch (...) {
            }
        }
        if (!sampling_params.is_none()) {
            if (py::hasattr(sampling_params, "temperature"))
                this->temperature = sampling_params.attr("temperature").cast<std::optional<float>>();
            if (py::hasattr(sampling_params, "max_tokens"))
                this->max_tokens = sampling_params.attr("max_tokens").cast<std::optional<int>>();
            if (py::hasattr(sampling_params, "ignore_eos"))
                this->ignore_eos = sampling_params.attr("ignore_eos").cast<bool>();
            else
                this->ignore_eos = false;
        }
        else {
            this->ignore_eos = false;
        }

        this->seq_id    = generate_uuid();
        this->status    = SequenceStatus::WAITING;
        this->token_ids = t_ids;
        if (!t_ids.empty())
            this->last_token = t_ids.back();
        else
            this->last_token = -1;

        this->num_tokens              = t_ids.size();
        this->num_prompt_tokens       = t_ids.size();
        this->num_checkpointed_tokens = t_ids.size();
        this->num_cached_tokens       = 0;
        this->backup_engine_id        = engine_id;
        this->active_engine_id        = engine_id;

        this->metric = std::make_shared<SequenceMetric>(this->seq_id, this->num_tokens);

        BlockContext ctx(engine_id, -1, master_sp_rank, 1, 1);

        ctx.num_dispatched_tokens[master_sp_rank] = 0;

        this->block_ctx_map[engine_id] = ctx;
    }

    int dp_idx(std::optional<std::string> engine_id)
    {
        return block_ctx(engine_id).dp_idx;
    }

    BlockContext& block_ctx(std::optional<std::string> engine_id)
    {
        if (!engine_id.has_value())
            engine_id = active_engine_id;
        return block_ctx_map[engine_id];
    }

    std::vector<int>& block_table(std::optional<std::string> engine_id, int sp_idx)
    {
        return block_ctx(engine_id).sp_block_table[sp_idx];
    }

    void set_engine_id(std::string engine_id, int attention_dp, int attention_sp)
    {
        this->active_engine_id = engine_id;
        if (block_ctx_map.count(engine_id))
            return;

        BlockContext ctx(engine_id, -1, 0, attention_sp, attention_dp);

        for (int i = 0; i < attention_sp; ++i) {
            ctx.sp_block_table[i] = {};

            ctx.num_dispatched_tokens[i] = 0;
        }

        block_ctx_map[engine_id] = ctx;
    }

    int context_len(std::optional<std::string> engine_id, std::optional<int> sp_idx)
    {
        BlockContext& ctx = block_ctx(engine_id);
        int           idx = sp_idx.value_or(ctx.master_sp_idx);
        return ctx.num_dispatched_tokens[idx];
    }

    bool is_finished() const
    {
        return status == SequenceStatus::FINISHED;
    }
    int64_t num_completed_tokens() const
    {
        return num_tokens - num_prompt_tokens;
    }
    int64_t num_generated_tokens_since_checkpoint() const
    {
        return num_tokens - num_checkpointed_tokens;
    }

    py::list prompt_token_ids() const
    {
        py::list L;
        size_t   end = (num_prompt_tokens >= (int64_t)token_ids.size()) ? token_ids.size() : num_prompt_tokens;
        for (size_t i = 0; i < end; ++i)
            L.append(token_ids[i]);
        return L;
    }

    py::list completion_token_ids() const
    {
        py::list L;
        if (num_prompt_tokens >= (int64_t)token_ids.size())
            return L;
        for (size_t i = num_prompt_tokens; i < token_ids.size(); ++i)
            L.append(token_ids[i]);
        return L;
    }

    int64_t num_cached_blocks() const
    {
        return num_cached_tokens / block_size;
    }

    int num_blocks(std::optional<std::string> engine_id, int sp_idx)
    {
        int tokens = block_ctx(engine_id).num_dispatched_tokens[sp_idx];
        return (tokens + block_size - 1) / block_size;
    }

    int last_block_page_id(std::optional<std::string> engine_id, int sp_idx)
    {
        int   tokens         = block_ctx(engine_id).num_dispatched_tokens[sp_idx];
        int   last_block_idx = (tokens - 1) / block_size;
        auto& table          = block_ctx(engine_id).sp_block_table[sp_idx];
        if (last_block_idx < 0 || last_block_idx >= (int)table.size()) {
            throw py::index_error("Block table index out of range in last_block_page_id");
        }
        return table[last_block_idx];
    }

    int last_block_num_tokens(std::optional<std::string> engine_id, int sp_idx)
    {
        int tokens = block_ctx(engine_id).num_dispatched_tokens[sp_idx];
        return tokens - (num_blocks(engine_id, sp_idx) - 1) * block_size;
    }

    py::list block(int i, std::optional<std::string> engine_id, int sp_idx)
    {
        py::list L;
        if (i < 0)
            return L;
        size_t start = static_cast<size_t>(i) * block_size;
        size_t end   = static_cast<size_t>(i + 1) * block_size;
        if (end > token_ids.size())
            end = token_ids.size();
        if (start >= token_ids.size())
            return L;
        for (size_t k = start; k < end; ++k)
            L.append(token_ids[k]);
        return L;
    }

    void append_token_unsafe(int token_id, BlockContext* ctx, int sp_idx)
    {
        token_ids.push_back(token_id);
        last_token = token_id;
        num_tokens++;
        pending_token_count++;

        ctx->num_dispatched_tokens[sp_idx]++;

        if (metric) {
            metric->on_token_generated();
        }
    }

    void append_token(int token_id, std::optional<std::string> engine_id, std::optional<int> sp_idx)
    {
        if (!engine_id.has_value())
            engine_id = active_engine_id;
        BlockContext& ctx = block_ctx(engine_id);
        int           idx = sp_idx.value_or(ctx.master_sp_idx);

        token_ids.push_back(token_id);
        last_token = token_id;
        num_tokens++;
        pending_token_count++;
        ctx.num_dispatched_tokens[idx]++;

        if (metric) {
            metric->on_token_generated();
        }
    }
};

void bind_scheduler_ops(py::module& m);
