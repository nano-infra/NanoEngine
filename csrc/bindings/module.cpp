#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_sequence_metric(py::module_& m);
void bind_sequence(py::module_& m);

PYBIND11_MODULE(_nanodeploy_cpp, m) {
    m.doc() = "NanoDeploy C++ Backend";
    
    bind_sequence_metric(m);
    bind_sequence(m);
}
