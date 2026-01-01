#include "nanodeploy/proto/serialization.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace nanodeploy;

void bind_proto(py::module_& m)
{
    // Directly accepts address and size
    m.def("serialize", &serialize_sequences, 
          py::arg("data_ptr"), 
          py::arg("buffer_size"), 
          py::arg("seqs"), 
          py::arg("is_prefill"));

    m.def("deserialize", &deserialize_sequences, py::arg("data_ptr"), py::arg("data_len"));
}
