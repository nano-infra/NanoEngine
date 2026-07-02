#include "dlengine/csrc/metrics/server_metric.h"
#include "dlengine/csrc/scheduler/scheduler.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
using namespace dlengine;

void bind_scheduler_utils(py::module_& m)
{
    py::enum_<RoutingStrategy>(m, "RoutingStrategy")
        .value("RoundRobin", RoutingStrategy::RoundRobin)
        .value("LeastBatch", RoutingStrategy::LeastBatch)
        .value("LeastCache", RoutingStrategy::LeastCache)
        .value("SessionPrefix", RoutingStrategy::SessionPrefix)
        .export_values()
        .def_static("__class_getitem__",
                    [](const std::string& name) {
                        if (name == "RoundRobin")
                            return RoutingStrategy::RoundRobin;
                        if (name == "LeastBatch")
                            return RoutingStrategy::LeastBatch;
                        if (name == "LeastCache")
                            return RoutingStrategy::LeastCache;
                        if (name == "SessionPrefix")
                            return RoutingStrategy::SessionPrefix;
                        throw py::key_error(name);
                    })
        .def_property_readonly_static("__members__", [](py::object /* self */) {
            py::dict members;
            members["RoundRobin"]    = RoutingStrategy::RoundRobin;
            members["LeastBatch"]    = RoutingStrategy::LeastBatch;
            members["LeastCache"]    = RoutingStrategy::LeastCache;
            members["SessionPrefix"] = RoutingStrategy::SessionPrefix;
            return members;
        });

    // Bind the ScheduleResult struct
    py::class_<ScheduleResult>(m, "ScheduleResult")
        .def_readwrite("dp_seqs", &ScheduleResult::dp_seqs)
        .def_readwrite("dp_group_seqs", &ScheduleResult::dp_group_seqs)
        .def_readwrite("filtered_dp_group_seqs", &ScheduleResult::filtered_dp_group_seqs)
        .def_readwrite("is_prefill", &ScheduleResult::is_prefill)
        .def_readonly("group_send_counts", &ScheduleResult::group_send_counts)
        .def_readonly("group_recv_counts", &ScheduleResult::group_recv_counts)
        // .def_readonly("group_comm_matrix", &ScheduleResult::group_comm_matrix)
        .def_readonly("group_q_matrix", &ScheduleResult::group_q_matrix)
        // .def_readonly("group_res_matrix", &ScheduleResult::group_res_matrix);
        .def_readonly("waiting_head_blocks", &ScheduleResult::waiting_head_blocks)
        .def_readonly("waiting_total_blocks", &ScheduleResult::waiting_total_blocks);

    py::class_<StepMetricSnapshot>(m, "StepMetricSnapshot")
        .def_readonly("prefill_tokens", &StepMetricSnapshot::prefill_tokens)
        .def_readonly("decode_tokens", &StepMetricSnapshot::decode_tokens)
        .def_readonly("real_bs", &StepMetricSnapshot::real_bs)
        .def_readonly("prefill_tokens_per_dp", &StepMetricSnapshot::prefill_tokens_per_dp)
        .def_readonly("decode_tokens_per_dp", &StepMetricSnapshot::decode_tokens_per_dp)
        .def_readonly("prefix_cached_tokens_per_dp", &StepMetricSnapshot::prefix_cached_tokens_per_dp)
        .def_readonly("prefix_prompt_tokens_per_dp", &StepMetricSnapshot::prefix_prompt_tokens_per_dp);

    py::class_<SchedulerMetricSnapshot>(m, "SchedulerMetricSnapshot")
        .def_readonly("running_per_dp", &SchedulerMetricSnapshot::running_per_dp)
        .def_readonly("total_waiting", &SchedulerMetricSnapshot::total_waiting)
        .def_readonly("total_waiting_migration", &SchedulerMetricSnapshot::total_waiting_migration)
        .def_readonly("waiting_migration_head_tokens", &SchedulerMetricSnapshot::waiting_migration_head_tokens)
        .def_readonly("total_blocks_per_dp", &SchedulerMetricSnapshot::total_blocks_per_dp)
        .def_readonly("used_blocks_per_dp", &SchedulerMetricSnapshot::used_blocks_per_dp)
        .def_readonly("free_blocks", &SchedulerMetricSnapshot::free_blocks);

    py::class_<GqaCacheSpec>(m, "GqaCacheSpec")
        .def(py::init<>())
        .def_readwrite("num_pages", &GqaCacheSpec::num_pages)
        .def_readwrite("page_size", &GqaCacheSpec::page_size)
        .def_readwrite("max_blocks_per_seq", &GqaCacheSpec::max_blocks_per_seq)
        .def_readwrite("num_layers", &GqaCacheSpec::num_layers)
        .def_readwrite("num_kv_heads", &GqaCacheSpec::num_kv_heads)
        .def_readwrite("head_dim", &GqaCacheSpec::head_dim);

    py::class_<MlaCacheSpec>(m, "MlaCacheSpec")
        .def(py::init<>())
        .def_readwrite("num_pages", &MlaCacheSpec::num_pages)
        .def_readwrite("page_size", &MlaCacheSpec::page_size)
        .def_readwrite("max_blocks_per_seq", &MlaCacheSpec::max_blocks_per_seq)
        .def_readwrite("num_layers", &MlaCacheSpec::num_layers)
        .def_readwrite("kv_lora_rank", &MlaCacheSpec::kv_lora_rank)
        .def_readwrite("qk_rope_head_dim", &MlaCacheSpec::qk_rope_head_dim)
        .def_readwrite("head_dim", &MlaCacheSpec::head_dim);

    py::class_<GdnCacheSpec>(m, "GdnCacheSpec")
        .def(py::init<>())
        .def_readwrite("num_pages", &GdnCacheSpec::num_pages)
        .def_readwrite("page_size", &GdnCacheSpec::page_size)
        .def_readwrite("max_blocks_per_seq", &GdnCacheSpec::max_blocks_per_seq)
        .def_readwrite("state_slots", &GdnCacheSpec::state_slots)
        .def_readwrite("state_bytes", &GdnCacheSpec::state_bytes);

    py::class_<HcaCacheSpec>(m, "HcaCacheSpec")
        .def(py::init<>())
        .def_readwrite("bytes_per_token", &HcaCacheSpec::bytes_per_token)
        .def_readwrite("block_size_multiple", &HcaCacheSpec::block_size_multiple)
        .def_readwrite("compression_ratio", &HcaCacheSpec::compression_ratio)
        .def_readwrite("num_pages", &HcaCacheSpec::num_pages)
        .def_readwrite("page_size", &HcaCacheSpec::page_size)
        .def_readwrite("max_blocks_per_seq", &HcaCacheSpec::max_blocks_per_seq);

    py::class_<CsaCacheSpec>(m, "CsaCacheSpec")
        .def(py::init<>())
        .def_readwrite("compressor_head_dim", &CsaCacheSpec::compressor_head_dim)
        .def_readwrite("compression_ratio", &CsaCacheSpec::compression_ratio)
        .def_readwrite("num_pages", &CsaCacheSpec::num_pages)
        .def_readwrite("page_size", &CsaCacheSpec::page_size)
        .def_readwrite("max_blocks_per_seq", &CsaCacheSpec::max_blocks_per_seq);

    py::class_<IndexerCacheSpec>(m, "IndexerCacheSpec")
        .def(py::init<>())
        .def_readwrite("num_pages", &IndexerCacheSpec::num_pages)
        .def_readwrite("page_size", &IndexerCacheSpec::page_size)
        .def_readwrite("max_blocks_per_seq", &IndexerCacheSpec::max_blocks_per_seq)
        .def_readwrite("index_head_dim", &IndexerCacheSpec::index_head_dim)
        .def_readwrite("bytes_per_token", &IndexerCacheSpec::bytes_per_token);

    py::class_<HiSparseCacheSpec>(m, "HiSparseCacheSpec")
        .def(py::init<>())
        .def_readwrite("max_num_seqs", &HiSparseCacheSpec::max_num_seqs)
        .def_readwrite("device_buffer_size", &HiSparseCacheSpec::device_buffer_size)
        .def_readwrite("host_to_device_ratio", &HiSparseCacheSpec::host_to_device_ratio)
        .def_readwrite("swap_in_block_size", &HiSparseCacheSpec::swap_in_block_size)
        .def_readwrite("dummy_slot", &HiSparseCacheSpec::dummy_slot);

    py::enum_<CachePlanFlag>(m, "CachePlanFlag")
        .value("Gqa", CachePlanFlag::Gqa)
        .value("Mla", CachePlanFlag::Mla)
        .value("Gdn", CachePlanFlag::Gdn)
        .value("Hca", CachePlanFlag::Hca)
        .value("Csa", CachePlanFlag::Csa)
        .value("Indexer", CachePlanFlag::Indexer)
        .value("Hisparse", CachePlanFlag::Hisparse)
        .export_values();

    m.def("cache_plan_flag", &cache_plan_flag, py::arg("flag"));

    py::class_<CachePlan>(m, "CachePlan")
        .def(py::init([](uint32_t flags) {
                 CachePlan c;
                 c.flags                   = flags;
                 c.hca.bytes_per_token     = c.has_hca() ? 584 : -1;
                 c.hca.block_size_multiple = c.has_hca() ? 64 : -1;
                 c.csa.compressor_head_dim = c.has_csa() ? 512 : -1;
                 return c;
             }),
             py::arg("flags") = 0)
        .def("has_flag", &CachePlan::has_flag, py::arg("flag"))
        .def("set_flag", &CachePlan::set_flag, py::arg("flag"))
        .def("has_gqa", &CachePlan::has_gqa)
        .def("has_mla", &CachePlan::has_mla)
        .def("has_gdn", &CachePlan::has_gdn)
        .def("has_hca", &CachePlan::has_hca)
        .def("has_csa", &CachePlan::has_csa)
        .def("has_indexer", &CachePlan::has_indexer)
        .def("has_hisparse", &CachePlan::has_hisparse)
        .def("cache_mode", &CachePlan::cache_mode)
        .def("to_json", [](const CachePlan& c) { return c.to_json_string(); })
        .def("to_json_string", &CachePlan::to_json_string, py::arg("indent") = -1)
        .def_property_readonly("has_linear_attention", &CachePlan::has_linear_attention)
        .def_readwrite("flags", &CachePlan::flags)
        .def_readwrite("gqa", &CachePlan::gqa)
        .def_readwrite("mla", &CachePlan::mla)
        .def_readwrite("gdn", &CachePlan::gdn)
        .def_readwrite("hca", &CachePlan::hca)
        .def_readwrite("csa", &CachePlan::csa)
        .def_readwrite("indexer", &CachePlan::indexer)
        .def_readwrite("hisparse", &CachePlan::hisparse)
        .def_property(
            "linear_attention",
            [](CachePlan& c) -> GdnCacheSpec& { return c.gdn; },
            [](CachePlan& c, const GdnCacheSpec& spec) { c.gdn = spec; },
            py::return_value_policy::reference_internal)
        .def_property(
            "dsv4_compressed",
            [](CachePlan& c) -> CsaCacheSpec& { return c.csa; },
            [](CachePlan& c, const CsaCacheSpec& spec) { c.csa = spec; },
            py::return_value_policy::reference_internal)
        .def(py::pickle(
            [](const CachePlan& c) {
                return py::make_tuple(c.flags,
                                      c.gqa.num_pages,
                                      c.gqa.page_size,
                                      c.gqa.max_blocks_per_seq,
                                      c.gqa.num_layers,
                                      c.gqa.num_kv_heads,
                                      c.gqa.head_dim,
                                      c.mla.num_pages,
                                      c.mla.page_size,
                                      c.mla.max_blocks_per_seq,
                                      c.mla.num_layers,
                                      c.mla.kv_lora_rank,
                                      c.mla.qk_rope_head_dim,
                                      c.mla.head_dim,
                                      c.gdn.num_pages,
                                      c.gdn.page_size,
                                      c.gdn.max_blocks_per_seq,
                                      c.gdn.state_slots,
                                      c.gdn.state_bytes,
                                      c.hca.bytes_per_token,
                                      c.hca.block_size_multiple,
                                      c.hca.compression_ratio,
                                      c.hca.num_pages,
                                      c.hca.page_size,
                                      c.hca.max_blocks_per_seq,
                                      c.csa.compressor_head_dim,
                                      c.csa.compression_ratio,
                                      c.csa.num_pages,
                                      c.csa.page_size,
                                      c.csa.max_blocks_per_seq,
                                      c.indexer.num_pages,
                                      c.indexer.page_size,
                                      c.indexer.max_blocks_per_seq,
                                      c.indexer.index_head_dim,
                                      c.indexer.bytes_per_token,
                                      c.hisparse.max_num_seqs,
                                      c.hisparse.device_buffer_size,
                                      c.hisparse.host_to_device_ratio,
                                      c.hisparse.swap_in_block_size,
                                      c.hisparse.dummy_slot);
            },
            [](py::tuple t) {
                if (t.size() != 40) {
                    throw std::runtime_error("Invalid CachePlan pickle state");
                }
                CachePlan c;
                c.flags                         = t[0].cast<uint32_t>();
                c.gqa.num_pages                 = t[1].cast<int>();
                c.gqa.page_size                 = t[2].cast<int>();
                c.gqa.max_blocks_per_seq        = t[3].cast<int>();
                c.gqa.num_layers                = t[4].cast<int>();
                c.gqa.num_kv_heads              = t[5].cast<int>();
                c.gqa.head_dim                  = t[6].cast<int>();
                c.mla.num_pages                 = t[7].cast<int>();
                c.mla.page_size                 = t[8].cast<int>();
                c.mla.max_blocks_per_seq        = t[9].cast<int>();
                c.mla.num_layers                = t[10].cast<int>();
                c.mla.kv_lora_rank              = t[11].cast<int>();
                c.mla.qk_rope_head_dim          = t[12].cast<int>();
                c.mla.head_dim                  = t[13].cast<int>();
                c.gdn.num_pages                 = t[14].cast<int>();
                c.gdn.page_size                 = t[15].cast<int>();
                c.gdn.max_blocks_per_seq        = t[16].cast<int>();
                c.gdn.state_slots               = t[17].cast<int>();
                c.gdn.state_bytes               = t[18].cast<int>();
                c.hca.bytes_per_token           = t[19].cast<int>();
                c.hca.block_size_multiple       = t[20].cast<int>();
                c.hca.compression_ratio         = t[21].cast<int>();
                c.hca.num_pages                 = t[22].cast<int>();
                c.hca.page_size                 = t[23].cast<int>();
                c.hca.max_blocks_per_seq        = t[24].cast<int>();
                c.csa.compressor_head_dim       = t[25].cast<int>();
                c.csa.compression_ratio         = t[26].cast<int>();
                c.csa.num_pages                 = t[27].cast<int>();
                c.csa.page_size                 = t[28].cast<int>();
                c.csa.max_blocks_per_seq        = t[29].cast<int>();
                c.indexer.num_pages             = t[30].cast<int>();
                c.indexer.page_size             = t[31].cast<int>();
                c.indexer.max_blocks_per_seq    = t[32].cast<int>();
                c.indexer.index_head_dim        = t[33].cast<int>();
                c.indexer.bytes_per_token       = t[34].cast<int>();
                c.hisparse.max_num_seqs         = t[35].cast<int>();
                c.hisparse.device_buffer_size   = t[36].cast<int>();
                c.hisparse.host_to_device_ratio = t[37].cast<int>();
                c.hisparse.swap_in_block_size   = t[38].cast<int>();
                c.hisparse.dummy_slot           = t[39].cast<int>();
                return c;
            }));

    py::class_<SchedulerConfig>(m, "SchedulerConfig")
        .def(py::init<>())
        .def(py::init([](const std::string&      engine_id,
                         int                     num_speculative_tokens,
                         int                     max_num_seqs,
                         int                     max_num_batched_tokens,
                         int                     max_model_len,
                         const std::vector<int>& eos_ids,
                         int                     attention_dp,
                         int                     group_size,
                         int                     num_kvcache_blocks,
                         int                     kvcache_block_size,
                         const std::string&      mode,
                         const CachePlan&        cache_plan,
                         RoutingStrategy         routing_strategy,
                         int                     gdn_state_cache_slots) {
                 SchedulerConfig c;
                 c.engine_id              = engine_id;
                 c.num_speculative_tokens = num_speculative_tokens;
                 c.max_num_seqs           = max_num_seqs;
                 c.max_num_batched_tokens = max_num_batched_tokens;
                 c.max_model_len          = max_model_len;
                 c.eos_ids                = eos_ids;
                 c.attention_dp           = attention_dp;
                 c.group_size             = group_size;
                 c.num_kvcache_blocks     = num_kvcache_blocks;
                 c.kvcache_block_size     = kvcache_block_size;
                 c.mode                   = mode;
                 c.routing_strategy       = routing_strategy;
                 c.gdn_state_cache_slots  = gdn_state_cache_slots;
                 c.cache_plan             = cache_plan;
                 return c;
             }),
             py::arg("engine_id"),
             py::arg("num_speculative_tokens"),
             py::arg("max_num_seqs"),
             py::arg("max_num_batched_tokens"),
             py::arg("max_model_len"),
             py::arg("eos_ids"),
             py::arg("attention_dp"),
             py::arg("group_size"),
             py::arg("num_kvcache_blocks"),
             py::arg("kvcache_block_size"),
             py::arg("mode"),
             py::arg("cache_plan")            = CachePlan{},
             py::arg("routing_strategy")      = RoutingStrategy::RoundRobin,
             py::arg("gdn_state_cache_slots") = 0)
        .def_readwrite("engine_id", &SchedulerConfig::engine_id)
        .def_readwrite("num_speculative_tokens", &SchedulerConfig::num_speculative_tokens)
        .def_readwrite("max_num_seqs", &SchedulerConfig::max_num_seqs)
        .def_readwrite("max_num_batched_tokens", &SchedulerConfig::max_num_batched_tokens)
        .def_readwrite("max_model_len", &SchedulerConfig::max_model_len)
        .def_readwrite("eos_ids", &SchedulerConfig::eos_ids)
        .def_readwrite("attention_dp", &SchedulerConfig::attention_dp)
        .def_readwrite("group_size", &SchedulerConfig::group_size)
        .def_readwrite("num_kvcache_blocks", &SchedulerConfig::num_kvcache_blocks)
        .def_readwrite("kvcache_block_size", &SchedulerConfig::kvcache_block_size)
        .def_readwrite("mode", &SchedulerConfig::mode)
        .def_readwrite("routing_strategy", &SchedulerConfig::routing_strategy)
        .def_readwrite("gdn_state_cache_slots", &SchedulerConfig::gdn_state_cache_slots)
        .def_readwrite("cache_plan", &SchedulerConfig::cache_plan);

    // Bind the Scheduler class
    py::class_<Scheduler, std::shared_ptr<Scheduler>>(m, "Scheduler")
        .def(py::init<const SchedulerConfig&>(), py::arg("config"))
        .def(py::init<const std::string&,
                      int,
                      int,
                      int,
                      int,
                      std::vector<int>,
                      int,
                      int,
                      int,
                      int,
                      const std::string&>(),
             py::arg("engine_id"),
             py::arg("num_speculative_tokens"),
             py::arg("max_num_seqs"),
             py::arg("max_num_batched_tokens"),
             py::arg("max_model_len"),
             py::arg("eos_ids"),
             py::arg("attention_dp"),
             py::arg("group_size"),
             py::arg("num_kvcache_blocks"),
             py::arg("kvcache_block_size"),
             py::arg("mode"))

        // Cross-request prefix caching toggle (disabled for linear-attention)
        .def("set_prefix_caching_enabled", &Scheduler::set_prefix_caching_enabled, py::arg("enabled"))

        // Session-scoped GatedDeltaNet state caching (warm-session capacity)
        .def("set_session_cache_slots", &Scheduler::set_session_cache_slots, py::arg("capacity"))

        // Queue management
        .def("add", &Scheduler::add, py::arg("seq"))

        // Scheduling
        .def("schedule", &Scheduler::schedule, py::call_guard<py::gil_scoped_release>())

        // Postprocessing
        .def("postprocess",
             &Scheduler::postprocess,
             py::arg("dp_seqs"),
             py::arg("dp_token_ids"),
             py::arg("update_metrics") = true,
             // Per-token logprobs aligned with ``dp_token_ids``. Empty
             // (default) means the rollout didn't request logprobs and
             // the scheduler skips populating Sequence.completion_logprobs.
             py::arg("dp_token_logprobs") = std::vector<std::vector<std::vector<float>>>{},
             py::call_guard<py::gil_scoped_release>())

        // State queries
        .def("is_finished", &Scheduler::is_finished)
        .def("num_waiting", &Scheduler::num_waiting)
        .def("metric_snapshot", &Scheduler::metric_snapshot)
        .def("update_server_metric", &Scheduler::update_server_metric, py::arg("metric"), py::arg("schedule_result"))
        .def("record_step_metric",
             &Scheduler::record_step_metric,
             py::arg("metric"),
             py::arg("schedule_result"),
             py::arg("dp_group_token_ids") = std::vector<std::vector<std::vector<int>>>{})
        .def("prefix_cached_tokens", &Scheduler::prefix_cached_tokens, py::arg("seq_id"))
        .def("clear_finished_metric_state", &Scheduler::clear_finished_metric_state, py::arg("seq_id"))

        // Preemption
        .def("preempt", &Scheduler::preempt, py::arg("dp_idx"), py::arg("seq"))

        // Migration management
        .def("free_to_be_migrated",
             py::overload_cast<std::shared_ptr<Sequence>>(&Scheduler::free_to_be_migrated),
             py::arg("seq"))
        .def("free_to_be_migrated",
             py::overload_cast<const std::vector<std::shared_ptr<Sequence>>&>(&Scheduler::free_to_be_migrated),
             py::arg("seqs"))

        // Abort: stop generating for a sequence and free its KV blocks.
        .def("abort", &Scheduler::abort, py::arg("seq_id"))

        // Public member access
        .def_readonly("group_size", &Scheduler::group_size_)
        .def_readonly("attention_dp", &Scheduler::attention_dp_)
        .def_readwrite("routing_strategy", &Scheduler::routing_strategy);
}
