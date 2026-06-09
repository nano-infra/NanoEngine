#include "nanodeploy/csrc/metrics/server_metric.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace nanodeploy;

void bind_server_metric(py::module_& m)
{
    py::class_<ServerMetric, std::shared_ptr<ServerMetric>>(m, "ServerMetric")
        .def(py::init<>())
        // Read-write fields
        .def_readwrite("total_tokens", &ServerMetric::total_tokens)
        .def_readwrite("total_prompt_tokens", &ServerMetric::total_prompt_tokens)
        .def_readwrite("total_generated_tokens", &ServerMetric::total_generated_tokens)
        .def_readwrite("num_running_requests", &ServerMetric::num_running_requests)
        .def_readwrite("num_waiting_requests", &ServerMetric::num_waiting_requests)
        .def_readwrite("num_waiting_migration_requests", &ServerMetric::num_waiting_migration_requests)
        .def_readwrite("num_completed_requests", &ServerMetric::num_completed_requests)
        .def_readwrite("prefill_throughput_samples", &ServerMetric::prefill_throughput_samples)
        .def_readwrite("decode_throughput_samples", &ServerMetric::decode_throughput_samples)
        .def_readwrite("token_usage_by_dp", &ServerMetric::token_usage_by_dp)
        .def_readwrite("start_time", &ServerMetric::start_time)

        // Update methods
        .def("update_running_requests", &ServerMetric::update_running_requests, py::arg("count"))
        .def("update_waiting_requests", &ServerMetric::update_waiting_requests, py::arg("count"))
        .def("update_waiting_migration_requests", &ServerMetric::update_waiting_migration_requests, py::arg("count"))
        .def("add_completed_request", &ServerMetric::add_completed_request)
        .def("update_waiting_blocks",
             &ServerMetric::update_waiting_blocks,
             py::arg("head_blocks"),
             py::arg("total_blocks"))
        .def_readwrite("num_waiting_head_blocks", &ServerMetric::num_waiting_head_blocks)
        .def_readwrite("num_waiting_total_blocks", &ServerMetric::num_waiting_total_blocks)
        .def("add_tokens", &ServerMetric::add_tokens, py::arg("num_prompt") = 0, py::arg("num_generated") = 0)
        .def("record_prefill_throughput",
             &ServerMetric::record_prefill_throughput,
             py::arg("num_tokens"),
             py::arg("duration"))
        .def("record_decode_throughput",
             &ServerMetric::record_decode_throughput,
             py::arg("num_tokens"),
             py::arg("duration"))
        .def("update_token_usage", &ServerMetric::update_token_usage, py::arg("dp_idx"), py::arg("num_tokens"))
        .def("update_group_stats",
             &ServerMetric::update_group_stats,
             py::arg("group_send_counts"),
             py::arg("group_recv_counts"))

        // Properties
        .def_property_readonly("avg_prefill_throughput", &ServerMetric::avg_prefill_throughput)
        .def_property_readonly("avg_decode_throughput", &ServerMetric::avg_decode_throughput)
        .def_property_readonly("current_prefill_throughput", &ServerMetric::current_prefill_throughput)
        .def_property_readonly("current_decode_throughput", &ServerMetric::current_decode_throughput)
        .def_property_readonly("total_token_usage", &ServerMetric::total_token_usage)
        .def_property_readonly("uptime", &ServerMetric::uptime)

        // Logging
        .def("get_metric_report", &ServerMetric::get_metric_report, py::arg("include_detailed") = false)

        // get_summary
        .def("get_summary", [](const ServerMetric& self) {
            py::dict summary;
            summary["uptime_seconds"]             = self.uptime();
            summary["total_requests"]             = self.num_completed_requests;
            summary["running_requests"]           = self.num_running_requests;
            summary["waiting_requests"]           = self.num_waiting_requests;
            summary["total_tokens"]               = self.total_tokens;
            summary["total_prompt_tokens"]        = self.total_prompt_tokens;
            summary["total_generated_tokens"]     = self.total_generated_tokens;
            summary["avg_prefill_throughput"]     = self.avg_prefill_throughput();
            summary["avg_decode_throughput"]      = self.avg_decode_throughput();
            summary["current_prefill_throughput"] = self.current_prefill_throughput();
            summary["current_decode_throughput"]  = self.current_decode_throughput();
            summary["total_token_usage"]          = self.total_token_usage();
            summary["group_send_counts"]          = self.group_send_request_counts;
            summary["group_recv_counts"]          = self.group_recv_request_counts;
            summary["waiting_head_blocks"]        = self.num_waiting_head_blocks;
            summary["waiting_total_blocks"]       = self.num_waiting_total_blocks;
            return summary;
        });
}
