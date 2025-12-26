#include "nanodeploy/rpc/rpc_endpoint.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace nanodeploy;

void bind_rpc_endpoint(py::module_& m) {
    py::class_<RpcEndpoint>(m, "RpcEndpoint")
        .def(py::init<>())
        // 关键：将 data() 指针转为 python int，方便传递给 RDMA
        .def("data", [](RpcEndpoint& self) {
            return reinterpret_cast<uint64_t>(self.data());
        })
        .def("size", &RpcEndpoint::size)
        // 关键：Python 传入 int 地址，转为 C++ 指针
        .def("set_buffer", &RpcEndpoint::set_buffer, py::arg("ptr"), py::arg("size"))
        
        .def("feed_sequences", &RpcEndpoint::feed_sequences)
        .def("sequences", &RpcEndpoint::sequences)
        
        // 序列化接口
        .def("serialize_for_prefill", &RpcEndpoint::serialize_for_prefill)
        .def("serialize_for_decode", &RpcEndpoint::serialize_for_decode)
        .def("serialize_for_migrate", &RpcEndpoint::serialize_for_migrate)
        
        // 反序列化接口
        .def("deserialize_for_prefill", &RpcEndpoint::deserialize_for_prefill)
        .def("deserialize_for_decode", &RpcEndpoint::deserialize_for_decode)
        .def("deserialize_for_migrate", &RpcEndpoint::deserialize_for_migrate);
}