#include "nanodeploy/csrc/core/config.h"
#include "nanodeploy/csrc/engine/engine.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace nanodeploy;

void bind_simple_engine(py::module_& m)
{
    py::class_<SimpleEngine>(m, "SimpleEngine")
        .def(py::init<>())
        .def("init", &SimpleEngine::init)
        .def("shutdown", &SimpleEngine::shutdown)
        .def("add_request", &SimpleEngine::add_request)
        .def("is_finished", &SimpleEngine::is_finished)
        .def("step", &SimpleEngine::step)
        .def("generate", &SimpleEngine::generate);
}
