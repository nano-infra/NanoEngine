#include "nanodeploy/scheduler/sp_state_manager.h"
#include "opaque_types.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
using namespace nanodeploy;

// Bind the map and deque types
PYBIND11_MAKE_OPAQUE(std::unordered_map<int, std::shared_ptr<BlockManager>>);
PYBIND11_MAKE_OPAQUE(std::deque<std::shared_ptr<Sequence>>);

void bind_sp_state_manager(py::module_& m)
{
    py::class_<SPStateManager::LSDecodeMasterPlan>(m, "LSDecodeMasterPlan")
        .def(py::init<>())
        .def_readonly("success", &SPStateManager::LSDecodeMasterPlan::success)
        .def_readonly("failure_reason", &SPStateManager::LSDecodeMasterPlan::failure_reason)
        .def_readonly("scale_reason", &SPStateManager::LSDecodeMasterPlan::scale_reason)
        .def_readonly("allocation", &SPStateManager::LSDecodeMasterPlan::allocation)
        .def_readonly("master_ranks", &SPStateManager::LSDecodeMasterPlan::master_ranks)
        .def_readonly("master_batch_sizes", &SPStateManager::LSDecodeMasterPlan::master_batch_sizes)
        .def_readonly("sequence_master_ranks", &SPStateManager::LSDecodeMasterPlan::sequence_master_ranks)
        .def_readonly("new_allocation_ranks", &SPStateManager::LSDecodeMasterPlan::new_allocation_ranks)
        .def_readonly("group_used_kv_tokens", &SPStateManager::LSDecodeMasterPlan::group_used_kv_tokens)
        .def_readonly("group_used_kv_blocks", &SPStateManager::LSDecodeMasterPlan::group_used_kv_blocks);

    py::class_<SPStateManager::KVTokenRangeMove>(m, "KVTokenRangeMove")
        .def_readonly("seq_id", &SPStateManager::KVTokenRangeMove::seq_id)
        .def_readonly("dp_idx", &SPStateManager::KVTokenRangeMove::dp_idx)
        .def_readonly("src_sp_rank", &SPStateManager::KVTokenRangeMove::src_sp_rank)
        .def_readonly("dst_sp_rank", &SPStateManager::KVTokenRangeMove::dst_sp_rank)
        .def_readonly("src_block_id", &SPStateManager::KVTokenRangeMove::src_block_id)
        .def_readonly("src_token_offset", &SPStateManager::KVTokenRangeMove::src_token_offset)
        .def_readonly("dst_block_id", &SPStateManager::KVTokenRangeMove::dst_block_id)
        .def_readonly("dst_token_offset", &SPStateManager::KVTokenRangeMove::dst_token_offset)
        .def_readonly("num_tokens", &SPStateManager::KVTokenRangeMove::num_tokens);

    py::class_<SPStateManager::LSKVConsolidationPlan, std::shared_ptr<SPStateManager::LSKVConsolidationPlan>>(
        m, "LSKVConsolidationPlan")
        .def_readonly("success", &SPStateManager::LSKVConsolidationPlan::success)
        .def_readonly("failure_reason", &SPStateManager::LSKVConsolidationPlan::failure_reason)
        .def_readonly("transaction_id", &SPStateManager::LSKVConsolidationPlan::transaction_id)
        .def_readonly("group_id", &SPStateManager::LSKVConsolidationPlan::group_id)
        .def_readonly("dp_idx", &SPStateManager::LSKVConsolidationPlan::dp_idx)
        .def_readonly("source_rank", &SPStateManager::LSKVConsolidationPlan::source_rank)
        .def_readonly("retained_ranks", &SPStateManager::LSKVConsolidationPlan::retained_ranks)
        .def_readonly("moves", &SPStateManager::LSKVConsolidationPlan::moves)
        .def_readonly("num_tokens", &SPStateManager::LSKVConsolidationPlan::num_tokens);

    // Bind RoutingStrategy
    py::enum_<RoutingStrategy>(m, "RoutingStrategy")
        .value("RoundRobin", RoutingStrategy::RoundRobin)
        .value("LeastBatch", RoutingStrategy::LeastBatch)
        .value("LeastCache", RoutingStrategy::LeastCache)
        .value("VLLMLoadBalance", RoutingStrategy::VLLMLoadBalance)
        .export_values()
        .def_static("__class_getitem__",
                    [](const std::string& name) {
                        if (name == "RoundRobin")
                            return RoutingStrategy::RoundRobin;
                        if (name == "LeastBatch")
                            return RoutingStrategy::LeastBatch;
                        if (name == "LeastCache")
                            return RoutingStrategy::LeastCache;
                        if (name == "VLLMLoadBalance")
                            return RoutingStrategy::VLLMLoadBalance;
                        throw py::key_error(name);
                    })
        .def_property_readonly_static("__members__", [](py::object /* self */) {
            py::dict m;
            m["RoundRobin"]      = RoutingStrategy::RoundRobin;
            m["LeastBatch"]      = RoutingStrategy::LeastBatch;
            m["LeastCache"]      = RoutingStrategy::LeastCache;
            m["VLLMLoadBalance"] = RoutingStrategy::VLLMLoadBalance;
            return m;
        });
    py::enum_<SPMasterSelector>(m, "SPMasterSelector")
        .value("RoundRobin", SPMasterSelector::RoundRobin)
        .value("LeastBatch", SPMasterSelector::LeastBatch)
        .value("LeastCache", SPMasterSelector::LeastCache)
        .export_values();

    // Bind BlockManagerMap
    py::bind_map<std::unordered_map<int, std::shared_ptr<BlockManager>>>(m, "BlockManagerMap");

    // Bind SequenceDeque
    py::class_<std::deque<std::shared_ptr<Sequence>>>(m, "SequenceDeque")
        .def(py::init<>())
        .def("append", [](std::deque<std::shared_ptr<Sequence>>& d, std::shared_ptr<Sequence> s) { d.push_back(s); })
        .def("popleft",
             [](std::deque<std::shared_ptr<Sequence>>& d) {
                 if (d.empty())
                     throw py::index_error();
                 auto s = d.front();
                 d.pop_front();
                 return s;
             })
        .def("pop",
             [](std::deque<std::shared_ptr<Sequence>>& d) {
                 if (d.empty())
                     throw py::index_error();
                 auto s = d.back();
                 d.pop_back();
                 return s;
             })
        .def("extendleft",
             [](std::deque<std::shared_ptr<Sequence>>& d, py::object iterable) {
                 for (auto item : iterable) {
                     d.push_front(item.cast<std::shared_ptr<Sequence>>());
                 }
             })
        .def("remove",
             [](std::deque<std::shared_ptr<Sequence>>& d, std::shared_ptr<Sequence> s) {
                 auto it = std::find(d.begin(), d.end(), s);
                 if (it != d.end()) {
                     d.erase(it);
                 }
                 else {
                     throw py::value_error("list.remove(x): x not in list");
                 }
             })
        .def("__getitem__",
             [](const std::deque<std::shared_ptr<Sequence>>& d, int idx) {
                 if (idx < 0)
                     idx += d.size();
                 if (idx < 0 || idx >= (int)d.size())
                     throw py::index_error();
                 return d[idx];
             })
        .def("__setitem__",
             [](std::deque<std::shared_ptr<Sequence>>& d, int idx, std::shared_ptr<Sequence> s) {
                 if (idx < 0)
                     idx += d.size();
                 if (idx < 0 || idx >= (int)d.size())
                     throw py::index_error();
                 d[idx] = s;
             })
        .def("__len__", [](const std::deque<std::shared_ptr<Sequence>>& d) { return d.size(); })
        .def("__bool__", [](const std::deque<std::shared_ptr<Sequence>>& d) { return !d.empty(); })
        .def(
            "__iter__",
            [](std::deque<std::shared_ptr<Sequence>>& d) { return py::make_iterator(d.begin(), d.end()); },
            py::keep_alive<0, 1>());  // Bind SPStateManager
    py::class_<SPStateManager, std::shared_ptr<SPStateManager>>(m, "SPStateManager")
        .def(py::init([](const std::string& engine_id,
                         int                attention_sp,
                         int                num_kvcache_blocks,
                         int                kvcache_block_size,
                         int                max_num_seqs,
                         int                max_num_batched_tokens,
                         int                max_num_recv_seqs,
                         double             reserved_blocks_per_req,
                         int                segment_size,
                         bool               enable_dynamic_sp_size,
                         const std::string& dynamic_sp_size_strategy,
                         int                dynamic_sp_long_request_threshold,
                         int                dynamic_sp_long_request_size,
                         bool               enable_dynamic_sp_bucket_policy,
                         const std::string& dynamic_sp_bucket_policy,
                         double             attention_cost_a,
                         double             attention_cost_b,
                         double             q_cost_a,
                         double             q_cost_b,
                         double             res_cost_a,
                         double             res_cost_b,
                         double             lse_cost_a,
                         double             lse_cost_b,
                         int                q_bytes_per_edge,
                         int                res_bytes_per_edge,
                         int                lse_bytes_per_edge,
                         bool               enable_non_uniform_split,
                         const std::string& sp_master_selector,
                         bool               sp_debug,
                         int                fixed_sp_size) {
                 return std::make_shared<SPStateManager>(engine_id,
                                                         attention_sp,
                                                         num_kvcache_blocks,
                                                         kvcache_block_size,
                                                         max_num_seqs,
                                                         max_num_batched_tokens,
                                                         max_num_recv_seqs,
                                                         reserved_blocks_per_req,
                                                         segment_size,
                                                         enable_dynamic_sp_size,
                                                         dynamic_sp_size_strategy,
                                                         dynamic_sp_long_request_threshold,
                                                         dynamic_sp_long_request_size,
                                                         enable_dynamic_sp_bucket_policy,
                                                         dynamic_sp_bucket_policy,
                                                         attention_cost_a,
                                                         attention_cost_b,
                                                         q_cost_a,
                                                         q_cost_b,
                                                         res_cost_a,
                                                         res_cost_b,
                                                         lse_cost_a,
                                                         lse_cost_b,
                                                         q_bytes_per_edge,
                                                         res_bytes_per_edge,
                                                         lse_bytes_per_edge,
                                                         enable_non_uniform_split,
                                                         sp_master_selector,
                                                         sp_debug,
                                                         fixed_sp_size);
             }),
             py::arg("engine_id"),
             py::arg("attention_sp"),
             py::arg("num_kvcache_blocks"),
             py::arg("kvcache_block_size"),
             py::arg("max_num_seqs"),
             py::arg("max_num_batched_tokens"),
             py::arg("max_num_recv_seqs"),
             py::arg("reserved_blocks_per_req"),
             py::arg("segment_size") = 65536,
             py::arg("enable_dynamic_sp_size"),
             py::arg("dynamic_sp_size_strategy")          = "legacy",
             py::arg("dynamic_sp_long_request_threshold") = 100000,
             py::arg("dynamic_sp_long_request_size")      = 0,
             py::arg("enable_dynamic_sp_bucket_policy")   = false,
             py::arg("dynamic_sp_bucket_policy")          = "",
             py::arg("attention_cost_a")                  = 1.0,
             py::arg("attention_cost_b")                  = 0.0,
             py::arg("q_cost_a")                          = 1.0,
             py::arg("q_cost_b")                          = 0.0,
             py::arg("res_cost_a")                        = 1.0,
             py::arg("res_cost_b")                        = 0.0,
             py::arg("lse_cost_a")                        = 1.0,
             py::arg("lse_cost_b")                        = 0.0,
             py::arg("q_bytes_per_edge")                  = 1,
             py::arg("res_bytes_per_edge")                = 1,
             py::arg("lse_bytes_per_edge")                = 1,
             py::arg("enable_non_uniform_split"),
             py::arg("sp_master_selector"),
             py::arg("sp_debug")      = false,
             py::arg("fixed_sp_size") = 0)

        .def_property_readonly("is_empty", &SPStateManager::is_empty)
        .def_property_readonly("num_running_seqs", &SPStateManager::num_running_seqs)
        .def_property_readonly("num_running_tokens", &SPStateManager::num_running_tokens)
        .def("num_recv_seqs_per_sp", &SPStateManager::num_recv_seqs_per_sp, py::arg("sp_idx"))
        .def("master_seq_count", &SPStateManager::master_seq_count, py::arg("sp_idx"))

        .def("can_append", &SPStateManager::can_append, py::arg("seq"), py::arg("num_tokens") = 1)
        .def("may_append", &SPStateManager::may_append, py::arg("seq"), py::arg("num_tokens") = 1)
        .def("can_append_on_sp",
             &SPStateManager::can_append_on_sp,
             py::arg("seq"),
             py::arg("sp_idx"),
             py::arg("num_tokens") = 1)
        .def("may_append_on_sp",
             &SPStateManager::may_append_on_sp,
             py::arg("seq"),
             py::arg("sp_idx"),
             py::arg("num_tokens") = 1)

        .def("can_allocate",
             &SPStateManager::can_allocate,
             py::arg("seq"),
             py::arg("num_seqs"),
             py::arg("num_batched_tokens"))

        .def("allocate", &SPStateManager::allocate, py::arg("seq"))
        .def("allocate_ls_initial", &SPStateManager::allocate_ls_initial, py::arg("seq"))
        .def("allocate_ls_initial_batch",
             &SPStateManager::allocate_ls_initial_batch,
             py::arg("batch"),
             py::arg("failure_after_allocations_for_test") = -1)
        .def("deallocate", &SPStateManager::deallocate, py::arg("seq"), py::arg("slot"))
        .def("set_decode_master", &SPStateManager::set_decode_master, py::arg("seq"), py::arg("master_sp_idx"))
        .def("estimate_pending_append_capacity",
             &SPStateManager::estimate_pending_append_capacity,
             py::arg("rank"),
             py::arg("requests"),
             py::arg("group_sequences"))
        .def("plan_iteration_masters_source_greedy",
             &SPStateManager::plan_iteration_masters_source_greedy,
             py::arg("requests"),
             py::arg("allocation"),
             py::arg("extra_ranks"),
             py::arg("batch_per_master"),
             py::arg("enable_memory_scale_up") = true)
        .def(
            "validate_iteration_master_plan",
            [](const SPStateManager&                         manager,
               const std::vector<std::shared_ptr<Sequence>>& requests,
               const SPStateManager::LSDecodeMasterPlan&     plan) {
                std::string error;
                bool        valid = manager.validate_iteration_master_plan(requests, plan, &error);
                return py::make_tuple(valid, error);
            },
            py::arg("requests"),
            py::arg("plan"))
        .def("reassign_pending_append",
             &SPStateManager::reassign_pending_append,
             py::arg("seq"),
             py::arg("target_sp_idx"))
        .def("commit_iteration_master_plan",
             &SPStateManager::commit_iteration_master_plan,
             py::arg("requests"),
             py::arg("plan"))
        .def("group_used_kv_tokens", &SPStateManager::group_used_kv_tokens, py::arg("seqs"))
        .def("group_used_kv_blocks", &SPStateManager::group_used_kv_blocks, py::arg("seqs"))
        .def("get_active_master_count", &SPStateManager::get_active_master_count, py::arg("seqs"))
        .def("get_kv_participant_count", &SPStateManager::get_kv_participant_count, py::arg("seqs"))
        .def("rebuild_decode_role_counters", &SPStateManager::rebuild_decode_role_counters)
        .def("add_running_tokens", &SPStateManager::add_running_tokens, py::arg("sp_idx"), py::arg("count"))

        .def_readwrite("block_manager", &SPStateManager::block_manager)
        .def_readwrite("running", &SPStateManager::running)
        .def_readwrite("dummy_seqs", &SPStateManager::dummy_seqs)
        .def_readwrite("routing_strategy", &SPStateManager::routing_strategy)
        // Expose waiting queues for decentralized scheduler mode (per-rank control)
        .def_readwrite("waiting", &SPStateManager::waiting)
        .def_readwrite("waiting_migration", &SPStateManager::waiting_migration);
}
