#include "nanodeploy/csrc/metrics/sequence_metric.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace nanodeploy;

void bind_sequence_metric(py::module_& m)
{
    py::class_<SequenceMetric, std::shared_ptr<SequenceMetric>>(m, "SequenceMetric")
        .def(py::init<uint64_t, int>(), py::arg("seq_id"), py::arg("num_prompt_tokens") = 0)

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
                        [](const std::tuple<uint64_t,
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
}
