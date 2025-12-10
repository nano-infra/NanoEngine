#include "sequence_core.h"

// =========================================================================
// 新增: 二进制序列化辅助工具 (扩展支持 Metric)
// =========================================================================

class BinaryBuffer {
public:
    std::vector<char> data;
    size_t            read_pos = 0;

    BinaryBuffer()
    {
        // [优化] 预分配 1MB 空间 (根据你的实际负载调整，1MB 通常足够容纳数千个请求的基础信息)
        // 这避免了 vector 在追加数据时的多次扩容和内存拷贝
        data.reserve(1024 * 1024);
    }

    void write(const void* ptr, size_t size)
    {
        const char* src = static_cast<const char*>(ptr);
        data.insert(data.end(), src, src + size);
    }

    template<typename T>
    void write_val(const T& val)
    {
        write(&val, sizeof(T));
    }

    void write_string(const std::string& str)
    {
        size_t len = str.size();
        write_val(len);
        write(str.data(), len);
    }

    void write_opt_string(const std::optional<std::string>& opt)
    {
        bool has_val = opt.has_value();
        write_val(has_val);
        if (has_val)
            write_string(opt.value());
    }

    void write_opt_double(const std::optional<double>& opt)
    {
        bool has_val = opt.has_value();
        write_val(has_val);
        if (has_val)
            write_val(opt.value());
    }

    template<typename T>
    void write_vec(const std::vector<T>& vec)
    {
        size_t len = vec.size();
        write_val(len);
        if (len > 0)
            write(vec.data(), len * sizeof(T));
    }

    void read(void* ptr, size_t size)
    {
        if (read_pos + size > data.size())
            throw std::runtime_error("Buffer overflow");
        std::memcpy(ptr, data.data() + read_pos, size);
        read_pos += size;
    }

    template<typename T>
    T read_val()
    {
        T val;
        read(&val, sizeof(T));
        return val;
    }

    std::string read_string()
    {
        size_t      len = read_val<size_t>();
        std::string str(len, '\0');
        read(&str[0], len);
        return str;
    }

    std::optional<std::string> read_opt_string()
    {
        bool has_val = read_val<bool>();
        if (has_val)
            return read_string();
        return std::nullopt;
    }

    // [Added] 支持 optional double 读取
    std::optional<double> read_opt_double()
    {
        bool has_val = read_val<bool>();
        if (has_val)
            return read_val<double>();
        return std::nullopt;
    }

    template<typename T>
    std::vector<T> read_vec()
    {
        size_t         len = read_val<size_t>();
        std::vector<T> vec(len);
        if (len > 0)
            read(vec.data(), len * sizeof(T));
        return vec;
    }
};

// =========================================================================
// 新增: SequenceBatch 类
// 用于在 Ray 传输时批量、零拷贝地序列化 Sequence 列表
// =========================================================================

class SequenceBatch {
public:
    std::vector<std::shared_ptr<Sequence>> sequences;
    bool                                   is_decode_mode = false;

    SequenceBatch() = default;

    SequenceBatch(const py::list& seq_list, bool is_decode = false)
    {
        this->is_decode_mode = is_decode;
        sequences.reserve(seq_list.size());
        for (auto handle : seq_list) {
            sequences.push_back(handle.cast<std::shared_ptr<Sequence>>());
        }
    }

    size_t size() const
    {
        return sequences.size();
    }

    std::shared_ptr<Sequence> get_item(size_t index)
    {
        if (index >= sequences.size())
            throw py::index_error();
        return sequences[index];
    }

    // 完整支持 Metric 序列化
    std::string serialize(bool include_metrics = true)
    {
        BinaryBuffer buf;

        buf.write_val(sequences.size());
        buf.write_val(is_decode_mode);

        for (const auto& seq : sequences) {
            buf.write_string(seq->seq_id);
            buf.write_val(seq->status);

            // 如果是 Decode 模式，且 token_ids 不为空，只传最后一个 token
            if (is_decode_mode && !seq->token_ids.empty()) {
                buf.write_val(seq->token_ids.back());
            }
            else {
                // Prefill 模式，或者空序列：传输全量 Vector
                buf.write_vec(seq->token_ids);
            }

            buf.write_val(seq->num_tokens);

            buf.write_val(seq->num_prompt_tokens);
            buf.write_val(seq->num_checkpointed_tokens);
            buf.write_val(seq->num_cached_tokens);
            buf.write_opt_string(seq->backup_engine_id);
            buf.write_opt_string(seq->active_engine_id);
            buf.write_val(seq->last_token);

            bool has_temp = seq->temperature.has_value();
            buf.write_val(has_temp);
            if (has_temp)
                buf.write_val(seq->temperature.value());

            bool has_max = seq->max_tokens.has_value();
            buf.write_val(has_max);
            if (has_max)
                buf.write_val(seq->max_tokens.value());

            buf.write_val(seq->ignore_eos);

            buf.write_val(seq->block_ctx_map.size());
            for (const auto& kv : seq->block_ctx_map) {
                buf.write_opt_string(kv.first);
                const auto& ctx = kv.second;
                buf.write_opt_string(ctx.engine_id);
                buf.write_val(ctx.dp_idx);
                buf.write_val(ctx.master_sp_idx);
                buf.write_val(ctx.attention_sp);
                buf.write_val(ctx.attention_dp);
                size_t loc_len = ctx.block_location.size();
                buf.write_val(loc_len);
                if (loc_len > 0)
                    buf.write(ctx.block_location.data(), loc_len * sizeof(std::pair<int, int>));
                buf.write_val(ctx.num_dispatched_tokens.size());
                for (auto const& [k, v] : ctx.num_dispatched_tokens) {
                    buf.write_val(k);
                    buf.write_val(v);
                }
                buf.write_val(ctx.sp_block_table.size());
                for (auto const& [k, v] : ctx.sp_block_table) {
                    buf.write_val(k);
                    buf.write_vec(v);
                }
            }

            if (include_metrics) {
                bool has_metric = (seq->metric != nullptr);
                buf.write_val(has_metric);
                if (has_metric) {
                    auto& m = seq->metric;
                    buf.write_string(m->seq_id);
                    buf.write_opt_double(m->arrival_time);
                    buf.write_opt_double(m->decode_first_scheduled_time);
                    buf.write_opt_double(m->first_token_time);
                    buf.write_opt_double(m->completion_time);
                    buf.write_opt_double(m->last_token_time);
                    buf.write_val(m->num_prompt_tokens);
                    buf.write_val(m->num_generated_tokens);
                    buf.write_vec(m->itl_samples);
                }
            }
            else {
                buf.write_val(false);
            }
        }
        return std::string(buf.data.begin(), buf.data.end());
    }

    static std::shared_ptr<SequenceBatch> deserialize(const std::string& bytes)
    {
        auto         batch = std::make_shared<SequenceBatch>();
        BinaryBuffer buf;
        buf.data.assign(bytes.begin(), bytes.end());

        size_t count             = buf.read_val<size_t>();
        bool   is_decode_payload = buf.read_val<bool>();

        batch->sequences.reserve(count);

        for (size_t i = 0; i < count; ++i) {
            auto seq = Sequence::create_empty();

            seq->seq_id = buf.read_string();
            seq->status = buf.read_val<SequenceStatus>();

            if (is_decode_payload) {
                int last_token = buf.read_val<int>();
                seq->token_ids.push_back(last_token);
            }
            else {
                seq->token_ids = std::move(buf.read_vec<int>());
            }

            seq->num_tokens = buf.read_val<int64_t>();

            seq->num_prompt_tokens       = buf.read_val<int64_t>();
            seq->num_checkpointed_tokens = buf.read_val<int64_t>();
            seq->num_cached_tokens       = buf.read_val<int64_t>();
            seq->backup_engine_id        = buf.read_opt_string();
            seq->active_engine_id        = buf.read_opt_string();
            seq->last_token              = buf.read_val<int>();

            if (buf.read_val<bool>())
                seq->temperature = buf.read_val<float>();
            if (buf.read_val<bool>())
                seq->max_tokens = buf.read_val<int>();
            seq->ignore_eos = buf.read_val<bool>();

            size_t map_size = buf.read_val<size_t>();
            for (size_t j = 0; j < map_size; ++j) {
                std::optional<std::string> key = buf.read_opt_string();
                BlockContext               ctx;
                ctx.engine_id     = buf.read_opt_string();
                ctx.dp_idx        = buf.read_val<int>();
                ctx.master_sp_idx = buf.read_val<int>();
                ctx.attention_sp  = buf.read_val<int>();
                ctx.attention_dp  = buf.read_val<int>();
                size_t loc_len    = buf.read_val<size_t>();
                ctx.block_location.resize(loc_len);
                if (loc_len > 0)
                    buf.read(ctx.block_location.data(), loc_len * sizeof(std::pair<int, int>));
                size_t dispatched_len = buf.read_val<size_t>();
                for (size_t k = 0; k < dispatched_len; ++k) {
                    int dk                        = buf.read_val<int>();
                    int dv                        = buf.read_val<int>();
                    ctx.num_dispatched_tokens[dk] = dv;
                }
                size_t table_len = buf.read_val<size_t>();
                for (size_t k = 0; k < table_len; ++k) {
                    int tk                 = buf.read_val<int>();
                    ctx.sp_block_table[tk] = buf.read_vec<int>();
                }
                seq->block_ctx_map[key] = ctx;
            }

            bool has_metric = buf.read_val<bool>();
            if (has_metric) {
                auto m                         = std::make_shared<SequenceMetric>();
                m->seq_id                      = buf.read_string();
                m->arrival_time                = buf.read_opt_double();
                m->decode_first_scheduled_time = buf.read_opt_double();
                m->first_token_time            = buf.read_opt_double();
                m->completion_time             = buf.read_opt_double();
                m->last_token_time             = buf.read_opt_double();
                m->num_prompt_tokens           = buf.read_val<int>();
                m->num_generated_tokens        = buf.read_val<int>();
                m->itl_samples                 = buf.read_vec<double>();
                seq->metric                    = m;
            }
            else {
                seq->metric = nullptr;
            }

            batch->sequences.push_back(seq);
        }
        return batch;
    }
};

struct PostProcessOps {
    std::vector<std::pair<int, Sequence*>> finished;
    std::vector<std::pair<int, Sequence*>> migrated;
};

// -------------------------------------------------------------------------
// 3. 核心 Postprocess 函数
// -------------------------------------------------------------------------

PostProcessOps postprocess_step(py::list    dp_seqs_list,
                                py::list    dp_token_ids_list,
                                py::list    dp_dummy_seqs_list,
                                std::string engine_id,
                                int         eos_id,
                                bool        is_prefill,
                                py::object  metrics_manager)
{
    PostProcessOps             ops;
    std::optional<std::string> opt_engine_id = engine_id;

    for (size_t dp_idx = 0; dp_idx < dp_seqs_list.size(); ++dp_idx) {
        py::list sp_seqs      = dp_seqs_list[dp_idx].cast<py::list>();
        py::list sp_token_ids = dp_token_ids_list[dp_idx].cast<py::list>();

        std::unordered_set<Sequence*> dummies;
        py::list                      current_dummies = dp_dummy_seqs_list[dp_idx].cast<py::list>();
        for (auto item : current_dummies) {
            dummies.insert(item.cast<Sequence*>());
        }

        for (size_t sp_idx = 0; sp_idx < sp_seqs.size(); ++sp_idx) {
            py::list seqs            = sp_seqs[sp_idx].cast<py::list>();
            py::list batch_token_ids = sp_token_ids[sp_idx].cast<py::list>();
            int      current_sp_idx  = (int)sp_idx;

            size_t batch_size = seqs.size();
            for (size_t i = 0; i < batch_size; ++i) {
                Sequence* seq = seqs[i].cast<Sequence*>();
                if (dummies.count(seq))
                    continue;

                auto it = seq->block_ctx_map.find(opt_engine_id);
                if (it == seq->block_ctx_map.end()) {
                    it = seq->block_ctx_map.find(seq->active_engine_id);
                }
                BlockContext* ctx_ptr = &(it->second);

                py::list loop_token_ids = batch_token_ids[i].cast<py::list>();
                bool     stop_sequence  = false;

                for (auto t : loop_token_ids) {
                    if (stop_sequence)
                        break;

                    int token_id = t.cast<int>();

                    seq->append_token_unsafe(token_id, ctx_ptr, current_sp_idx);

                    bool is_eos     = (!seq->ignore_eos && token_id == eos_id);
                    bool is_max_len = false;
                    if (seq->max_tokens.has_value()) {
                        is_max_len = ((seq->num_tokens - seq->num_prompt_tokens) == seq->max_tokens.value());
                    }

                    if (is_eos || is_max_len) {
                        seq->status = SequenceStatus::FINISHED;
                        ops.finished.push_back({(int)dp_idx, seq});
                        stop_sequence = true;
                    }
                    else if (is_prefill) {
                        seq->status           = SequenceStatus::TO_BE_MIGRATED;
                        seq->backup_engine_id = seq->active_engine_id;
                        seq->active_engine_id = std::nullopt;
                        ops.migrated.push_back({(int)dp_idx, seq});
                        stop_sequence = true;
                    }
                }
            }
        }
    }
    return ops;
}

// -------------------------------------------------------------------------
// 4. Pybind11 模块绑定
// -------------------------------------------------------------------------

PYBIND11_MODULE(_core, m)
{
    bind_scheduler_ops(m);

    py::bind_vector<std::vector<int>>(m, "IntVector")
        .def(py::pickle(
            [](const std::vector<int>& v) {
                py::list l;
                for (int item : v) {
                    l.append(item);
                }
                return l;
            },
            [](py::list l) {
                std::vector<int> v;
                v.reserve(l.size());
                for (auto item : l) {
                    v.push_back(item.cast<int>());
                }
                return v;
            }));

    py::bind_map<std::map<int, int>>(m, "IntIntMap")
        .def("clear", [](std::map<int, int>& m) { m.clear(); })
        .def(py::pickle(
            [](const std::map<int, int>& map) {
                py::dict d;
                for (const auto& kv : map) {
                    d[py::int_(kv.first)] = py::int_(kv.second);
                }
                return d;
            },
            [](py::dict d) {
                std::map<int, int> map;
                for (auto item : d) {
                    map[item.first.cast<int>()] = item.second.cast<int>();
                }
                return map;
            }));

    py::bind_map<std::map<int, std::vector<int>>>(m, "IntVectorMap")
        .def("clear", [](std::map<int, std::vector<int>>& m) { m.clear(); })
        .def(py::pickle(
            [](const std::map<int, std::vector<int>>& map) {
                py::dict d;
                for (const auto& kv : map) {
                    py::list l;
                    for (int x : kv.second) {
                        l.append(x);
                    }
                    d[py::int_(kv.first)] = l;
                }
                return d;
            },
            [](py::dict d) {
                std::map<int, std::vector<int>> map;
                for (auto item : d) {
                    std::vector<int> v;
                    py::list         l = item.second.cast<py::list>();
                    v.reserve(l.size());
                    for (auto x : l) {
                        v.push_back(x.cast<int>());
                    }
                    map[item.first.cast<int>()] = v;
                }
                return map;
            }));

    // 绑定 BlockContextMap 并增加 Pickle 支持
    py::bind_map<std::map<std::optional<std::string>, BlockContext>>(m, "BlockContextMap")
        .def(py::pickle(
            [](const std::map<std::optional<std::string>, BlockContext>& map) {
                py::dict d;
                for (const auto& kv : map) {
                    d[py::cast(kv.first)] = py::cast(kv.second);
                }
                return d;
            },
            [](py::dict d) {
                std::map<std::optional<std::string>, BlockContext> map;
                for (auto item : d) {
                    map[item.first.cast<std::optional<std::string>>()] = item.second.cast<BlockContext>();
                }
                return map;
            }));

    // 绑定 Metrics
    py::class_<SequenceMetric, std::shared_ptr<SequenceMetric>>(m, "SequenceMetric")
        .def(py::init<std::string, int>())
        .def_readwrite("seq_id", &SequenceMetric::seq_id)
        .def_readwrite("arrival_time", &SequenceMetric::arrival_time)
        .def_readwrite("decode_first_scheduled_time", &SequenceMetric::decode_first_scheduled_time)
        .def_readwrite("first_token_time", &SequenceMetric::first_token_time)
        .def_readwrite("completion_time", &SequenceMetric::completion_time)
        .def_readwrite("num_prompt_tokens", &SequenceMetric::num_prompt_tokens)
        .def_readwrite("num_generated_tokens", &SequenceMetric::num_generated_tokens)
        .def_readwrite("itl_samples", &SequenceMetric::itl_samples)  // Exposed
        .def("record_arrival", &SequenceMetric::record_arrival)
        .def("record_first_scheduled", &SequenceMetric::record_first_scheduled)
        .def("record_token", &SequenceMetric::on_token_generated)
        .def("record_first_token", &SequenceMetric::on_token_generated)
        .def("record_completion", &SequenceMetric::record_completion)
        .def_property_readonly("ttft", &SequenceMetric::ttft)
        .def_property_readonly("e2e_latency", &SequenceMetric::e2e_latency)
        .def_property_readonly("avg_itl", &SequenceMetric::avg_itl)
        .def_property_readonly("avg_tpot_with_queueing", &SequenceMetric::avg_tpot_with_queueing)
        .def_property_readonly("avg_tpot_wo_queueing", &SequenceMetric::avg_tpot_wo_queueing)
        .def("get_itl_stats", &SequenceMetric::get_itl_stats)
        // [Added] 标准 Pickle 支持，用于非 Batch 模式的传输
        .def(py::pickle(
            [](const SequenceMetric& m) {
                return py::make_tuple(m.seq_id,
                                      m.arrival_time,
                                      m.decode_first_scheduled_time,
                                      m.first_token_time,
                                      m.completion_time,
                                      m.last_token_time,
                                      m.num_prompt_tokens,
                                      m.num_generated_tokens,
                                      m.itl_samples);
            },
            [](py::tuple t) {
                SequenceMetric m;
                m.seq_id                      = t[0].cast<std::string>();
                m.arrival_time                = t[1].cast<std::optional<double>>();
                m.decode_first_scheduled_time = t[2].cast<std::optional<double>>();
                m.first_token_time            = t[3].cast<std::optional<double>>();
                m.completion_time             = t[4].cast<std::optional<double>>();
                m.last_token_time             = t[5].cast<std::optional<double>>();
                m.num_prompt_tokens           = t[6].cast<int>();
                m.num_generated_tokens        = t[7].cast<int>();
                m.itl_samples                 = t[8].cast<std::vector<double>>();
                return m;
            }));

    py::class_<ServerMetric, std::shared_ptr<ServerMetric>>(m, "ServerMetric")
        .def(py::init<>())
        .def_readwrite("total_tokens", &ServerMetric::total_tokens)
        .def_readwrite("total_prompt_tokens", &ServerMetric::total_prompt_tokens)
        .def_readwrite("total_generated_tokens", &ServerMetric::total_generated_tokens)
        .def_readwrite("num_running_requests", &ServerMetric::num_running_requests)
        .def_readwrite("num_waiting_requests", &ServerMetric::num_waiting_requests)
        .def_readwrite("num_waiting_migration_requests", &ServerMetric::num_waiting_migration_requests)
        .def_readwrite("num_completed_requests", &ServerMetric::num_completed_requests)
        .def_readwrite("prefill_throughput_samples", &ServerMetric::prefill_throughput_samples)
        .def_readwrite("decode_throughput_samples", &ServerMetric::decode_throughput_samples)
        .def_readonly("start_time", &ServerMetric::start_time)
        .def_readwrite("token_usage_by_dp", &ServerMetric::token_usage_by_dp)

        .def("add_tokens", &ServerMetric::add_tokens, py::arg("num_prompt") = 0, py::arg("num_generated") = 0)
        .def("update_token_usage", &ServerMetric::update_token_usage)
        .def("update_running_requests", &ServerMetric::update_running_requests)
        .def("update_waiting_requests", &ServerMetric::update_waiting_requests)
        .def("update_waiting_migration_requests", &ServerMetric::update_waiting_migration_requests)
        .def("add_completed_request", &ServerMetric::add_completed_request)
        .def("record_prefill_throughput", &ServerMetric::record_prefill_throughput)
        .def("record_decode_throughput", &ServerMetric::record_decode_throughput)
        .def("get_summary", &ServerMetric::get_summary)

        .def_property_readonly("avg_prefill_throughput", &ServerMetric::avg_prefill_throughput)
        .def_property_readonly("avg_decode_throughput", &ServerMetric::avg_decode_throughput)
        .def_property_readonly("current_prefill_throughput", &ServerMetric::current_prefill_throughput)
        .def_property_readonly("current_decode_throughput", &ServerMetric::current_decode_throughput)
        .def_property_readonly("total_token_usage", &ServerMetric::total_token_usage)
        .def_property_readonly("uptime", &ServerMetric::uptime);

    py::class_<PostProcessOps>(m, "PostProcessOps")
        .def_readwrite("finished", &PostProcessOps::finished)
        .def_readwrite("migrated", &PostProcessOps::migrated);

    m.def("postprocess_step",
          &postprocess_step,
          py::return_value_policy::reference,
          py::arg("dp_seqs_list"),
          py::arg("dp_token_ids_list"),
          py::arg("dp_dummy_seqs_list"),
          py::arg("engine_id"),
          py::arg("eos_id"),
          py::arg("is_prefill"),
          py::arg("metrics_manager") = py::none(),
          "Optimized C++ implementation of scheduler postprocess loop");

    py::enum_<SequenceStatus>(m, "SequenceStatus")
        .value("WAITING", SequenceStatus::WAITING)
        .value("RUNNING", SequenceStatus::RUNNING)
        .value("FINISHED", SequenceStatus::FINISHED)
        .value("TO_BE_MIGRATED", SequenceStatus::TO_BE_MIGRATED);

    py::class_<BlockContext>(m, "BlockContext")
        .def(py::init<>())
        .def_readwrite("engine_id", &BlockContext::engine_id)
        .def_readwrite("dp_idx", &BlockContext::dp_idx)
        .def_readwrite("master_sp_idx", &BlockContext::master_sp_idx)
        .def_readwrite("attention_sp", &BlockContext::attention_sp)
        .def_readwrite("attention_dp", &BlockContext::attention_dp)
        .def_readwrite("block_location", &BlockContext::block_location)
        .def_readwrite("num_dispatched_tokens", &BlockContext::num_dispatched_tokens)
        .def_readwrite("sp_block_table", &BlockContext::sp_block_table)
        .def(py::pickle(
            [](const BlockContext& b) {
                return py::make_tuple(b.engine_id,
                                      b.dp_idx,
                                      b.master_sp_idx,
                                      b.attention_sp,
                                      b.attention_dp,
                                      b.block_location,
                                      b.num_dispatched_tokens,
                                      b.sp_block_table);
            },
            [](py::tuple t) {
                if (t.size() != 8)
                    throw std::runtime_error("Invalid BlockContext state");
                BlockContext b;
                b.engine_id             = t[0].cast<std::optional<std::string>>();
                b.dp_idx                = t[1].cast<int>();
                b.master_sp_idx         = t[2].cast<int>();
                b.attention_sp          = t[3].cast<int>();
                b.attention_dp          = t[4].cast<int>();
                b.block_location        = t[5].cast<std::vector<std::pair<int, int>>>();
                b.num_dispatched_tokens = t[6].cast<std::map<int, int>>();
                b.sp_block_table        = t[7].cast<std::map<int, std::vector<int>>>();
                return b;
            }));

    py::class_<Sequence, std::shared_ptr<Sequence>>(m, "Sequence")
        .def(py::init([](const py::object&          tokens,
                         py::object                 sampling_params,
                         std::optional<std::string> engine_id,
                         int                        master_sp_rank) {
                 std::vector<int> t_ids;
                 if (py::isinstance<py::list>(tokens)) {
                     auto l = tokens.cast<py::list>();
                     t_ids.reserve(l.size());
                     for (auto item : l) {
                         t_ids.push_back(item.cast<int>());
                     }
                 }
                 else {
                     t_ids = tokens.cast<std::vector<int>>();
                 }
                 return new Sequence(t_ids, sampling_params, engine_id, master_sp_rank);
             }),
             py::arg("token_ids"),
             py::arg("sampling_params") = py::none(),
             py::arg("engine_id")       = py::none(),
             py::arg("master_sp_rank")  = 0)

        .def_property_readonly_static("block_size", [](py::object) { return Sequence::block_size; })
        .def_readwrite("seq_id", &Sequence::seq_id)
        .def_readwrite("status", &Sequence::status)
        .def_readwrite("token_ids", &Sequence::token_ids)
        .def_readwrite("last_token", &Sequence::last_token)
        .def_readwrite("num_tokens", &Sequence::num_tokens)
        .def_readwrite("num_prompt_tokens", &Sequence::num_prompt_tokens)
        .def_readwrite("num_checkpointed_tokens", &Sequence::num_checkpointed_tokens)
        .def_readwrite("num_cached_tokens", &Sequence::num_cached_tokens)
        .def_readwrite("backup_engine_id", &Sequence::backup_engine_id)
        .def_readwrite("active_engine_id", &Sequence::active_engine_id)
        .def_readwrite("block_ctx_map", &Sequence::block_ctx_map)
        .def_readwrite("metric", &Sequence::metric)
        .def_readwrite("pending_token_count", &Sequence::pending_token_count)
        .def_readwrite("temperature", &Sequence::temperature)
        .def_readwrite("max_tokens", &Sequence::max_tokens)
        .def_readwrite("ignore_eos", &Sequence::ignore_eos)
        .def("dp_idx", &Sequence::dp_idx)
        .def("block_ctx",
             &Sequence::block_ctx,
             py::arg("engine_id") = py::none(),
             py::return_value_policy::reference_internal)
        .def("block_table",
             &Sequence::block_table,
             py::arg("engine_id") = py::none(),
             py::arg("sp_idx")    = 0,
             py::return_value_policy::reference_internal)
        .def("set_engine_id",
             &Sequence::set_engine_id,
             py::arg("engine_id"),
             py::arg("attention_dp") = 1,
             py::arg("attention_sp") = 1)
        .def("context_len", &Sequence::context_len, py::arg("engine_id") = py::none(), py::arg("sp_idx") = py::none())
        .def("__len__", [](const Sequence& s) { return s.num_tokens; })
        .def("__getitem__",
             [](const Sequence& s, int i) {
                 if (i < 0)
                     i += s.token_ids.size();
                 if (i < 0 || i >= (int)s.token_ids.size())
                     throw py::index_error();
                 return s.token_ids[i];
             })
        .def_property_readonly("is_finished", &Sequence::is_finished)
        .def_property_readonly("num_completed_tokens", &Sequence::num_completed_tokens)
        .def_property_readonly("num_generated_tokens_since_checkpoint",
                               &Sequence::num_generated_tokens_since_checkpoint)
        .def_property_readonly("prompt_token_ids", &Sequence::prompt_token_ids)
        .def_property_readonly("completion_token_ids", &Sequence::completion_token_ids)
        .def_property_readonly("num_cached_blocks", &Sequence::num_cached_blocks)
        .def("num_blocks", &Sequence::num_blocks)
        .def("last_block_page_id", &Sequence::last_block_page_id)
        .def("last_block_num_tokens", &Sequence::last_block_num_tokens)
        .def("block", &Sequence::block)
        .def("append_token",
             &Sequence::append_token,
             py::arg("token_id"),
             py::arg("engine_id") = py::none(),
             py::arg("sp_idx")    = py::none())
        .def(py::pickle(
            [](const Sequence& s) {
                py::object token_state;
                if (s.num_generated_tokens_since_checkpoint() == 0)
                    token_state = py::cast(s.token_ids);
                else
                    token_state = py::cast(s.last_token);
                return py::make_tuple(s.num_tokens,
                                      s.num_checkpointed_tokens,
                                      s.num_cached_tokens,
                                      s.backup_engine_id,
                                      s.active_engine_id,
                                      s.block_ctx_map,
                                      s.temperature,
                                      token_state,
                                      s.seq_id,
                                      // [Changed] 传输 metric 对象
                                      s.metric);
            },
            [](py::tuple t) {
                // [Optimized] 使用 create_empty 避免开销
                auto s_ptr = Sequence::create_empty();

                s_ptr->num_tokens              = t[0].cast<int64_t>();
                s_ptr->num_checkpointed_tokens = t[1].cast<int64_t>();
                s_ptr->num_cached_tokens       = t[2].cast<int64_t>();
                s_ptr->backup_engine_id        = t[3].cast<std::optional<std::string>>();
                s_ptr->active_engine_id        = t[4].cast<std::optional<std::string>>();
                s_ptr->block_ctx_map           = t[5].cast<std::map<std::optional<std::string>, BlockContext>>();
                s_ptr->temperature             = t[6].cast<std::optional<float>>();

                // 处理 token/last_token 逻辑
                if (py::isinstance<py::list>(t[7])) {
                    auto l = t[7].cast<py::list>();
                    s_ptr->token_ids.reserve(l.size());
                    for (auto item : l)
                        s_ptr->token_ids.push_back(item.cast<int>());
                }
                else if (py::isinstance<py::int_>(t[7])) {
                    s_ptr->last_token = t[7].cast<int>();
                }

                s_ptr->seq_id = t[8].cast<std::string>();

                // [Changed] 恢复 metric
                if (t.size() > 9 && !t[9].is_none()) {
                    s_ptr->metric = t[9].cast<std::shared_ptr<SequenceMetric>>();
                }
                else {
                    s_ptr->metric = nullptr;
                }

                return s_ptr;
            }));

    py::class_<SequenceBatch, std::shared_ptr<SequenceBatch>>(m, "SequenceBatch")
        .def(py::init<const py::list&, bool>(), py::arg("seq_list"), py::arg("is_decode") = false)
        .def("__len__", &SequenceBatch::size)
        .def("__getitem__", &SequenceBatch::get_item)
        .def(
            "__iter__",
            [](SequenceBatch& s) { return py::make_iterator(s.sequences.begin(), s.sequences.end()); },
            py::keep_alive<0, 1>())
        .def(
            "serialize",
            [](SequenceBatch& self, bool include_metrics) {
                std::string s = self.serialize(include_metrics);
                return py::bytes(s);
            },
            py::arg("include_metrics") = true)
        .def_static("deserialize", &SequenceBatch::deserialize)
        .def(py::pickle(
            [](SequenceBatch& s) {
                // 1. 定义一个 C++ 字符串用于接收数据
                std::string data;

                {
                    py::gil_scoped_release release;

                    data = s.serialize(false);
                }

                return py::make_tuple(py::bytes(data));
            },
            [](py::tuple t) {
                if (t.size() != 1)
                    throw std::runtime_error("Invalid state");

                std::string                    bytes = t[0].cast<std::string>();
                std::shared_ptr<SequenceBatch> batch;

                {
                    py::gil_scoped_release release;

                    batch = SequenceBatch::deserialize(bytes);
                }

                return batch;
            }));
}
