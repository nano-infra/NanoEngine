#include "nanodeploy/metrics/sequence_metric.h"
#include "nanodeploy/metrics/server_metric.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace nanodeploy;

void bind_sequence_metric(py::module_& m)
{
    py::class_<SequenceMetric, std::shared_ptr<SequenceMetric>>(m, "SequenceMetric")
        .def(py::init<const std::string&, int>(), py::arg("seq_id"), py::arg("num_prompt_tokens") = 0)

        // Dataclass-like fields (must be writable; scheduler mutates these)
        .def_readwrite("seq_id", &SequenceMetric::seq_id)
        .def_readwrite("arrival_time", &SequenceMetric::arrival_time)
        .def_readwrite("first_scheduled_time", &SequenceMetric::first_scheduled_time)
        .def_readwrite("decode_arrival_time", &SequenceMetric::decode_arrival_time)
        .def_readwrite("decode_scheduled_time", &SequenceMetric::decode_scheduled_time)
        .def_readwrite("first_token_time", &SequenceMetric::first_token_time)
        .def_readwrite("completion_time", &SequenceMetric::completion_time)
        .def_readwrite("num_prompt_tokens", &SequenceMetric::num_prompt_tokens)
        .def_readwrite("num_generated_tokens", &SequenceMetric::num_generated_tokens)
        .def_readwrite("itl_samples", &SequenceMetric::itl_samples)
        .def_readwrite("last_token_time", &SequenceMetric::last_token_time)
        .def("record_arrival", &SequenceMetric::record_arrival)
        .def("record_first_scheduled", &SequenceMetric::record_first_scheduled)
        .def("record_decode_arrival", &SequenceMetric::record_decode_arrival)
        .def("record_decode_scheduled", &SequenceMetric::record_decode_scheduled)
        .def("record_first_token", &SequenceMetric::record_first_token)
        .def("record_token", &SequenceMetric::record_token)
        .def("record_completion", &SequenceMetric::record_completion)

        .def_property_readonly("ttft", &SequenceMetric::ttft)
        .def_property_readonly("e2e_latency", &SequenceMetric::e2e_latency)
        .def_property_readonly("avg_tpot_with_queueing", &SequenceMetric::avg_tpot_with_queueing)
        .def_property_readonly("avg_tpot_wo_queueing", &SequenceMetric::avg_tpot_wo_queueing)
        .def_property_readonly("queueing_time_ms", &SequenceMetric::queueing_time_ms)
        .def_property_readonly("decode_queue_time_ms", &SequenceMetric::decode_queue_time_ms)
        .def_property_readonly("avg_itl", &SequenceMetric::avg_itl)
        .def_property_readonly("p50_itl", &SequenceMetric::p50_itl)
        .def_property_readonly("p99_itl", &SequenceMetric::p99_itl)

        .def("log_metrics", &SequenceMetric::log_metrics)
        .def(py::pickle([](const SequenceMetric& p) { return p.getstate(); },
                        [](const std::tuple<std::string,
                                            std::optional<double>,
                                            std::optional<double>,
                                            std::optional<double>,
                                            std::optional<double>,
                                            std::optional<double>,
                                            std::optional<double>,
                                            std::optional<double>,
                                            int,
                                            int,
                                            std::vector<double>>& t) { return SequenceMetric::setstate(t); }));

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
            return summary;
        });
}
