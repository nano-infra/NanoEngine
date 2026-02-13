#include <pybind11/pybind11.h>

namespace py = pybind11;

#include "nanocommon/logging.h"
#include "nanosequence/csrc/bind/metric_binding.h"
#include "nanosequence/csrc/bind/sequence_binding.h"

PYBIND11_MODULE(_nanosequence_cpp, m)
{
    m.doc() = "NanoSequence C++ Backend";

    bind_sequence_metric(m);
    bind_sequence(m);

    m.def("set_log_level", &nanocommon::set_log_level, "Set C++ backend log level");
}
