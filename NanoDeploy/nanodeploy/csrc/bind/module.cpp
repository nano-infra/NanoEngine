#include <pybind11/pybind11.h>

namespace py = pybind11;

#include "nanocommon/logging.h"
// Sequence-related bindings from NanoSequence
#include "nanosequence/csrc/bind/metric_binding.h"
#include "nanosequence/csrc/bind/sequence_binding.h"
void bind_server_metric(py::module_& m);
void bind_block_manager(py::module_& m);
void bind_sp_state_manager(py::module_& m);
void bind_scheduler_utils(py::module_& m);
void bind_model_runner_utils(py::module_& m);

PYBIND11_MODULE(_nanodeploy_cpp, m)
{
    m.doc() = "NanoDeploy C++ Backend";

    // Sequence and SequenceMetric bindings from NanoSequence
    bind_sequence(m);
    bind_sequence_metric(m);

    // NanoDeploy-specific bindings
    bind_server_metric(m);
    bind_block_manager(m);
    bind_sp_state_manager(m);
    bind_scheduler_utils(m);
    bind_model_runner_utils(m);

    m.def("set_log_level", &nanocommon::set_log_level, "Set C++ backend log level");
}
