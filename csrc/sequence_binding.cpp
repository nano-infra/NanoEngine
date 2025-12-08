#include <atomic>
#include <iostream>
#include <map>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <string>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<int>);
PYBIND11_MAKE_OPAQUE(std::map<int, int>);
PYBIND11_MAKE_OPAQUE(std::map<int, std::vector<int>>);

std::string generate_uuid()
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
};

class Sequence {
public:
    static constexpr int               block_size = 256;
    static inline std::atomic<int64_t> global_counter{0};

    std::string                                        seq_id;
    SequenceStatus                                     status;
    std::vector<int>                                   token_ids;
    int                                                last_token;
    int64_t                                            num_tokens;
    int64_t                                            num_prompt_tokens;
    int64_t                                            num_checkpointed_tokens;
    int64_t                                            num_cached_tokens;
    std::optional<std::string>                         backup_engine_id;
    std::optional<std::string>                         active_engine_id;
    std::map<std::optional<std::string>, BlockContext> block_ctx_map;
    py::object                                         metric;
    std::optional<float>                               temperature;
    std::optional<int>                                 max_tokens;
    bool                                               ignore_eos;

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
        this->metric                  = py::none();

        BlockContext ctx(engine_id, -1, master_sp_rank, 1, 1);
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
        for (int i = 0; i < attention_sp; ++i)
            ctx.sp_block_table[i] = {};
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

    void append_token(int token_id, std::optional<std::string> engine_id, std::optional<int> sp_idx)
    {
        if (!engine_id.has_value())
            engine_id = active_engine_id;
        BlockContext& ctx = block_ctx(engine_id);
        int           idx = sp_idx.value_or(ctx.master_sp_idx);
        token_ids.push_back(token_id);
        last_token = token_id;
        num_tokens++;
        ctx.num_dispatched_tokens[idx]++;
    }
};

struct PostProcessOps {
    std::vector<std::pair<int, Sequence*>> finished;  // (dp_idx, seq)
    std::vector<std::pair<int, Sequence*>> migrated;  // (dp_idx, seq)
};

PostProcessOps postprocess_step(py::list    dp_seqs_list,
                                py::list    dp_token_ids_list,
                                py::list    dp_dummy_seqs_list,
                                std::string engine_id,
                                int         eos_id,
                                bool        is_prefill,
                                py::object  metrics_manager)
{
    PostProcessOps ops;

    // 遍历 DP Rank
    for (size_t dp_idx = 0; dp_idx < dp_seqs_list.size(); ++dp_idx) {
        py::list sp_seqs      = dp_seqs_list[dp_idx].cast<py::list>();
        py::list sp_token_ids = dp_token_ids_list[dp_idx].cast<py::list>();

        std::unordered_set<Sequence*> dummies;
        py::list                      current_dummies = dp_dummy_seqs_list[dp_idx].cast<py::list>();
        for (auto item : current_dummies) {
            dummies.insert(item.cast<Sequence*>());
        }

        // 遍历 SP Rank
        for (size_t sp_idx = 0; sp_idx < sp_seqs.size(); ++sp_idx) {
            py::list seqs            = sp_seqs[sp_idx].cast<py::list>();
            py::list batch_token_ids = sp_token_ids[sp_idx].cast<py::list>();

            // 遍历 Batch 内的 Sequence
            for (size_t i = 0; i < seqs.size(); ++i) {
                Sequence* seq = seqs[i].cast<Sequence*>();
                if (dummies.count(seq))
                    continue;

                py::list loop_token_ids = batch_token_ids[i].cast<py::list>();
                bool     stop_sequence  = false;

                for (auto t : loop_token_ids) {
                    if (stop_sequence)
                        break;

                    int token_id = t.cast<int>();
                    seq->append_token(token_id, engine_id, (int)sp_idx);

                    // Metric 更新
                    if (!metrics_manager.is_none() && !seq->metric.is_none()) {
                        auto metric  = seq->metric;
                        int  num_gen = metric.attr("num_generated_tokens").cast<int>();
                        if (num_gen == 0) {
                            metric.attr("record_first_token")();
                            metric.attr("num_generated_tokens") = 1;
                        }
                        else {
                            metric.attr("record_token")();
                        }
                    }

                    bool is_eos     = (!seq->ignore_eos && token_id == eos_id);
                    bool is_max_len = false;
                    if (seq->max_tokens.has_value()) {
                        is_max_len = (seq->num_completed_tokens() == seq->max_tokens.value());
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

PYBIND11_MODULE(_core, m)
{
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
          "Core C++ implementation of scheduler postprocess loop");

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

    py::class_<Sequence>(m, "Sequence")
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
                                      s.metric);
            },
            [](py::tuple t) {
                std::vector<int> dummy_ids;
                if (py::isinstance<py::list>(t[7])) {
                    auto l = t[7].cast<py::list>();
                    dummy_ids.reserve(l.size());
                    for (auto item : l)
                        dummy_ids.push_back(item.cast<int>());
                }
                else if (!py::isinstance<py::int_>(t[7]) && !py::isinstance<py::none>(t[7])) {
                    try {
                        dummy_ids = t[7].cast<std::vector<int>>();
                    }
                    catch (...) {
                    }
                }
                Sequence s(dummy_ids, py::none(), std::nullopt, 0);
                s.num_tokens              = t[0].cast<int64_t>();
                s.num_checkpointed_tokens = t[1].cast<int64_t>();
                s.num_cached_tokens       = t[2].cast<int64_t>();
                s.backup_engine_id        = t[3].cast<std::optional<std::string>>();
                s.active_engine_id        = t[4].cast<std::optional<std::string>>();
                s.block_ctx_map           = t[5].cast<std::map<std::optional<std::string>, BlockContext>>();
                s.temperature             = t[6].cast<std::optional<float>>();
                if (py::isinstance<py::int_>(t[7]))
                    s.last_token = t[7].cast<int>();
                s.seq_id = t[8].cast<std::string>();
                s.metric = t[9];
                return s;
            }));
}
