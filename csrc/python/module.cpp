#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_sequence_metric(py::module_& m);
void bind_sequence(py::module_& m);
void bind_block_manager(py::module_& m);
void bind_sp_state_manager(py::module_& m);
void bind_scheduler_utils(py::module_& m);

PYBIND11_MODULE(_nanodeploy_cpp, m)
{
    m.doc() = "NanoDeploy C++ Backend";

    bind_sequence_metric(m);
    bind_sequence(m);
    bind_block_manager(m);
    bind_sp_state_manager(m);
    bind_scheduler_utils(m);
}