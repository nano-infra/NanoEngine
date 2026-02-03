#include <pybind11/attr.h>
#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>

#if defined(BUILD_INTRA_OPS) || defined(BUILD_INTER_OPS)
#include <torch/torch.h>

#ifdef BUILD_INTRA_OPS
#include "ops/intra_ll/all_to_all/all_to_all_intra_ll_buffer.h"
#endif

#ifdef BUILD_INTER_OPS
#include "ops/inter_ll/all_gather_inter_ll/all_gather_inter_ll_buffer.h"
#endif

#endif

#include "dlslime/csrc/logging.h"
#include "nanocommon/json.hpp"
#include "nanocommon/pybind_json/pybind_json.hpp"

using json = nlohmann::json;

namespace py = pybind11;

#ifdef BUILD_INTRA_OPS
#define BUILD_INTRA_OPS_ENABLED true
#else
#define BUILD_INTRA_OPS_ENABLED false
#endif

#ifdef BUILD_INTER_OPS
#define BUILD_INTER_OPS_ENABLED true
#else
#define BUILD_INTER_OPS_ENABLED false
#endif

#define EXPOSE_BUILD_FLAG(m, flag) m.attr("_" #flag) = flag##_ENABLED

PYBIND11_MODULE(_nanoccl_c, m)
{
    EXPOSE_BUILD_FLAG(m, BUILD_INTRA_OPS);
    EXPOSE_BUILD_FLAG(m, BUILD_INTER_OPS);

#ifdef BUILD_INTRA_OPS
    py::class_<dlslime::AllToAllIntraLLBuffer>(m, "AllToAllIntraLLBuffer")
        .def(py::init<int32_t, int32_t, int32_t, int32_t, int64_t>())
        .def("buffer_info", &dlslime::AllToAllIntraLLBuffer::buffer_info)
        .def("connect_full_mesh", &dlslime::AllToAllIntraLLBuffer::connectFullMesh)
        .def("get_local_buffer", &dlslime::AllToAllIntraLLBuffer::getLocalBuffer)
        .def("get_buffer_size_hint", &dlslime::AllToAllIntraLLBuffer::get_buffer_size_hint)
        .def("set_max_bs", &dlslime::AllToAllIntraLLBuffer::setMaxBs)
        .def("all_to_all_ll",
             &dlslime::AllToAllIntraLLBuffer::allToAllLL2D,
             py::arg("x"),
             py::arg("is_transpose") = false,
             py::arg("mask")         = py::none(),
             py::arg("offsets")      = py::none(),
             "AllToAll with optional mask and offsets");
#endif

#ifdef BUILD_INTER_OPS
    py::class_<dlslime::AllGatherInterLLBuffer>(m, "AllGatherInterLLBuffer")
        .def(py::init<int64_t, int64_t, torch::Dtype, int64_t, int64_t, int64_t>())
        .def(py::init<int64_t, int64_t, torch::Dtype, int64_t, int64_t, int64_t, bool>())
        .def("buffer_info", &dlslime::AllGatherInterLLBuffer::bufferInfo)
        .def("connect_full_mesh", &dlslime::AllGatherInterLLBuffer::connectFullMesh)
        .def("all_gather_ll", &dlslime::AllGatherInterLLBuffer::allGatherLL)
        .def("all_gather_ll_hook", &dlslime::AllGatherInterLLBuffer::allGatherLLHook);
#endif
}
