#include "nanodeploy/csrc/core/config.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace nanodeploy::core;

void bind_config(py::module_& m)
{
    py::class_<ModelConfig>(m, "ModelConfig")
        .def(py::init<>())
        .def_readwrite("model_type", &ModelConfig::model_type)
        .def_readwrite("hidden_size", &ModelConfig::hidden_size)
        // ... (Binding all ModelConfig fields might be overkill if not used in Python, but good for completeness)
        .def_readwrite("rope_theta", &ModelConfig::rope_theta);

    py::class_<EngineConfig>(m, "EngineConfig")
        .def(py::init<>())
        .def_readwrite("model", &EngineConfig::model)
        .def_readwrite("loop_count", &EngineConfig::loop_count)
        .def_readwrite("max_num_batched_tokens", &EngineConfig::max_num_batched_tokens)
        .def_readwrite("max_num_seqs", &EngineConfig::max_num_seqs)
        .def_readwrite("max_num_send_seqs", &EngineConfig::max_num_send_seqs)
        .def_readwrite("max_num_recv_seqs", &EngineConfig::max_num_recv_seqs)
        .def_readwrite("max_model_len", &EngineConfig::max_model_len)
        .def_readwrite("attention_tp", &EngineConfig::attention_tp)
        .def_readwrite("attention_sp", &EngineConfig::attention_sp)
        .def_readwrite("attention_dp", &EngineConfig::attention_dp)
        .def_readwrite("ffn_ep", &EngineConfig::ffn_ep)
        .def_readwrite("ffn_tp", &EngineConfig::ffn_tp)
        .def_readwrite("ffn_dp", &EngineConfig::ffn_dp)
        .def_readwrite("num_kvcache_blocks", &EngineConfig::num_kvcache_blocks)
        .def_readwrite("kvcache_block_size", &EngineConfig::kvcache_block_size)
        .def_readwrite("eos", &EngineConfig::eos)
        .def_readwrite("engine_id", &EngineConfig::engine_id)
        .def_readwrite("mode", &EngineConfig::mode)
        .def_readwrite("master_address", &EngineConfig::master_address)
        .def_readwrite("spoke_hub_addr", &EngineConfig::spoke_hub_addr);
}
