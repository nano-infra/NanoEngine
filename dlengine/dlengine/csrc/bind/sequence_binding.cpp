#include <utility>

#include <flatbuffers/flatbuffers.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "dlengine/csrc/metrics/sequence_metric.h"
#include "dlengine/csrc/sequence/sequence.h"
#include "dlengine/csrc/sequence/serialization.h"
#include "sequence_generated.h"

namespace py = pybind11;
using namespace dlengine;

void bind_sequence(py::module_& m)
{
    // Directly accepts address and size
    m.def("serialize",
          &serialize_sequences,
          py::arg("data_ptr"),
          py::arg("buffer_size"),
          py::arg("seqs"),
          py::arg("is_prefill"));

    m.def("deserialize", &deserialize_sequences, py::arg("data_ptr"), py::arg("data_len"));

    py::class_<SamplingParams>(m, "SamplingParams")
        .def(py::init<>())
        .def(py::init([](double temperature, int32_t max_tokens, bool ignore_eos, bool return_completion_logprobs) {
                 SamplingParams sp;
                 sp.temperature                = temperature;
                 sp.max_tokens                 = max_tokens;
                 sp.ignore_eos                 = ignore_eos;
                 sp.return_completion_logprobs = return_completion_logprobs;
                 return sp;
             }),
             py::arg("temperature")                = 1.0,
             py::arg("max_tokens")                 = 256,
             py::arg("ignore_eos")                 = false,
             py::arg("return_completion_logprobs") = false)
        .def_readwrite("temperature", &SamplingParams::temperature)
        .def_readwrite("max_tokens", &SamplingParams::max_tokens)
        .def_readwrite("ignore_eos", &SamplingParams::ignore_eos)
        .def_readwrite("return_completion_logprobs", &SamplingParams::return_completion_logprobs);

    py::class_<Sequence, std::shared_ptr<Sequence>>(m, "Sequence")
        .def(py::init<const std::vector<int>&, const SamplingParams&>(),
             py::arg("token_ids"),
             py::arg("sampling_params") = SamplingParams())
        .def("migrate_engine_id", &Sequence::migrate_engine_id)

        .def_property("seq_id", &Sequence::seq_id, &Sequence::set_seq_id)
        .def_property_readonly("last_token", &Sequence::last_token)
        .def_property_readonly("num_tokens", &Sequence::num_tokens)
        .def_property("affinity_key", &Sequence::affinity_key, &Sequence::set_affinity_key)
        .def_readwrite("metric", &Sequence::metric)
        .def_property_readonly("sampling_params", &Sequence::sampling_params)

        .def_property_readonly("is_finished", &Sequence::is_finished)
        .def_property_readonly("is_to_be_migrated", &Sequence::is_to_be_migrated)
        .def_property_readonly("prompt_token_ids", &Sequence::prompt_token_ids)
        .def_property_readonly("completion_token_ids", &Sequence::completion_token_ids)
        .def_property_readonly("completion_logprobs", &Sequence::completion_logprobs)

        // Vision slot management (EP separated mode)
        .def("clear_vision_slots", &Sequence::clear_vision_slots)
        .def_property_readonly("vision_slots",
                               [](const Sequence& s) {
                                   py::list result;
                                   for (const auto& vs : s.vision_slots()) {
                                       if (!vs)
                                           continue;
                                       py::dict d;
                                       d["encoder_engine_id"]   = vs->encoder_engine_id;
                                       d["slot_idx"]            = vs->slot_idx;
                                       d["num_tokens"]          = vs->num_tokens;
                                       d["hidden_size"]         = vs->hidden_size;
                                       d["max_tokens_per_slot"] = vs->max_tokens_per_slot;
                                       result.append(d);
                                   }
                                   return result;
                               })

        .def_readwrite_static("block_size", &Sequence::block_size)
        .def_static("set_block_size", &Sequence::set_block_size, py::arg("block_size"));
}
