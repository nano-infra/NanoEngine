#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_sequence_metric(py::module_& m);
