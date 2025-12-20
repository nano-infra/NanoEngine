#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nanodeploy/engine/scheduler_utils.h"

namespace py = pybind11;
using namespace nanodeploy;

void bind_scheduler_utils(py::module_& m)
{
    m.def("postprocess_sequences", 
          &postprocess_sequences,
          py::arg("worker_states"),
          py::arg("dp_seqs"),
          py::arg("dp_token_ids"),
          py::arg("engine_id"),
          py::arg("eos_id"),
          py::arg("is_prefill"),
          py::arg("update_metrics") = true
    );
}