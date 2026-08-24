#include <algorithm>
#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "nanodeploy/metrics/sequence_metric.h"

#include "nanodeploy/sequence/sequence.h"
#include "nanodeploy/sequence/serialization.h"

#include "opaque_types.h"

namespace py = pybind11;
using namespace nanodeploy;

namespace {

BlockContext::BlockIdList block_id_list_from_iterable(const py::iterable& it)
{
    BlockContext::BlockIdList out;
    for (auto item : it) {
        out.push_back(item.cast<int>());
    }
    return out;
}

}  // namespace

void bind_sequence(py::module_& m)
{
    // Bind wrapper containers used for mutable proxy views.
    // These are intentionally distinct from std::vector<int> used by token_ids.
    auto block_id_list = py::bind_vector<BlockContext::BlockIdList>(m, "BlockIdList");
    block_id_list.def(py::init<>()).def(py::init([](py::iterable it) { return block_id_list_from_iterable(it); }));
    py::implicitly_convertible<py::list, BlockContext::BlockIdList>();

    py::bind_vector<BlockContext::BlockLocationList>(m, "BlockLocationList").def(py::init<>());

    // Directly accepts address and size
    m.def("serialize",
          &serialize_sequences,
          py::arg("data_ptr"),
          py::arg("buffer_size"),
          py::arg("seqs"),
          py::arg("is_prefill"),
          py::arg("sp_rank") = -1,
          py::arg("sp_size") = -1);

    m.def("deserialize", &deserialize_sequences, py::arg("data_ptr"), py::arg("data_len"));

    m.def("serialize_sequence_payload", [](const std::shared_ptr<Sequence>& seq) {
        const std::vector<std::shared_ptr<Sequence>> seqs{seq};
        const size_t payload_size = serialized_sequences_size(seqs, true);
        py::bytes payload = py::reinterpret_steal<py::bytes>(
            PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(payload_size)));
        const auto data_ptr = reinterpret_cast<uintptr_t>(PyBytes_AS_STRING(payload.ptr()));
        const size_t written = serialize_sequences(data_ptr, payload_size, seqs, true);
        if (written != payload_size) {
            throw std::runtime_error("serialized Sequence payload size mismatch");
        }
        return payload;
    });

    m.def("deserialize_sequence_payload", [](const py::buffer& payload) {
        const py::buffer_info view = payload.request();
        if (view.itemsize != 1 || view.ndim != 1 || view.strides[0] != 1) {
            throw py::value_error("Sequence payload must be a contiguous byte buffer");
        }
        auto sequences = deserialize_sequences(reinterpret_cast<uintptr_t>(view.ptr),
                                               static_cast<size_t>(view.size));
        if (sequences.size() != 1) {
            throw py::value_error("Sequence payload must contain exactly one Sequence");
        }
        return sequences.front();
    });

    // Wrapper class for sp_block_table to provide defaultdict(list) behavior.
    py::class_<BlockContext::SpBlockTable>(m, "DefaultListDict")
        .def(py::init<>())
        .def(
            "__getitem__",
            [](BlockContext::SpBlockTable& self, int key) -> BlockContext::BlockIdList& {
                // Mimic defaultdict(list): create empty list for missing keys.
                return self[key];
            },
            py::return_value_policy::reference_internal)
        .def("__setitem__", [](BlockContext::SpBlockTable& self, int key, py::iterable value) {
            self[key] = block_id_list_from_iterable(value);
        });

    py::enum_<SequenceStatus>(m, "SequenceStatus")
        .value("WAITING", SequenceStatus::WAITING)
        .value("RUNNING", SequenceStatus::RUNNING)
        .value("FINISHED", SequenceStatus::FINISHED)
        .value("TO_BE_MIGRATED", SequenceStatus::TO_BE_MIGRATED)
        .export_values();

    py::enum_<BlockContextSlot>(m, "BlockContextSlot")
        .value("ACTIVE", BlockContextSlot::ACTIVE)
        .value("MIGRATE", BlockContextSlot::MIGRATE)
        .value("SWAP", BlockContextSlot::SWAP)
        .export_values();

    // Wrapper class for num_dispatched_tokens to provide defaultdict(int) behavior
    py::class_<std::unordered_map<int, int>>(m, "DefaultIntDict")
        .def(py::init<>())
        .def(
            "__getitem__",
            [](std::unordered_map<int, int>& self, int key) -> int& {
                // This mimics defaultdict(int) - returns 0 for missing keys and creates entry
                return self[key];
            },
            py::return_value_policy::reference_internal)
        .def("__setitem__", [](std::unordered_map<int, int>& self, int key, int value) { self[key] = value; })
        .def("__contains__",
             [](const std::unordered_map<int, int>& self, int key) { return self.find(key) != self.end(); })
        .def("keys",
             [](const std::unordered_map<int, int>& self) {
                 py::list keys;
                 for (const auto& pair : self) {
                     keys.append(pair.first);
                 }
                 return keys;
             })
        .def("items",
             [](const std::unordered_map<int, int>& self) {
                 py::list items;
                 for (const auto& pair : self) {
                     items.append(py::make_tuple(pair.first, pair.second));
                 }
                 return items;
             })
        .def("clear", [](std::unordered_map<int, int>& self) { self.clear(); })
        .def(py::pickle(
            [](const std::unordered_map<int, int>& self) {
                py::dict d;
                for (const auto& pair : self) {
                    d[py::cast(pair.first)] = py::cast(pair.second);
                }
                return d;
            },
            [](py::dict d) {
                std::unordered_map<int, int> self;
                for (auto item : d) {
                    self[item.first.cast<int>()] = item.second.cast<int>();
                }
                return self;
            }));

    py::class_<BlockContext>(m, "BlockContext")
        .def(py::init<>())
        .def_readwrite("engine_id", &BlockContext::engine_id_)
        .def_readwrite("dp_idx", &BlockContext::dp_idx_)
        .def_readwrite("master_sp_idx", &BlockContext::master_sp_idx_)
        .def_readwrite("attention_sp", &BlockContext::attention_sp_)
        .def_readwrite("attention_dp", &BlockContext::attention_dp_)
        .def_property("num_dispatched_tokens",
                      [](const BlockContext& self) { return self.num_dispatched_tokens; },
                      [](BlockContext& self, const std::vector<int>& value) { self.num_dispatched_tokens = value; })
        .def_property(
            "block_location",
            [](BlockContext& self) -> BlockContext::BlockLocationList& { return self.block_location; },
            [](BlockContext& self, const BlockContext::BlockLocationList& value) { self.block_location = value; },
            py::return_value_policy::reference_internal)
        .def_property(
            "sp_block_table",
            [](BlockContext& self) -> BlockContext::SpBlockTable& { return self.sp_block_table; },
            [](BlockContext& self, const BlockContext::SpBlockTable& value) { self.sp_block_table = value; },
            py::return_value_policy::reference_internal)
        .def("reset", &BlockContext::reset, py::arg("engine_id"), py::arg("attention_sp"), py::arg("attention_dp"))
        .def(py::pickle([](const BlockContext& p) { return p.getstate(); },
                        [](const std::tuple<std::string,
                                            int,
                                            int,
                                            int,
                                            int,
                                            std::vector<std::pair<int, int>>,
                                            std::vector<std::vector<int>>,
                                            std::vector<int>>& t) { return BlockContext::setstate(t); }));

    py::class_<Sequence, std::shared_ptr<Sequence>>(m, "Sequence")
        .def(py::init<const std::vector<int>&, double, int, bool>(),
             py::arg("token_ids"),
             py::arg("temperature") = 1.0,
             py::arg("max_tokens")  = 256,
             py::arg("ignore_eos")  = false)
        .def("active", &Sequence::active, py::arg("engine_id"), py::arg("attention_sp"), py::arg("attention_dp"))
        .def("migrate", &Sequence::migrate)
        .def("context_len", &Sequence::context_len, py::arg("engine_id"), py::arg("sp_idx") = std::nullopt)
        .def("append_token",
             &Sequence::append_token,
             py::arg("token_id"),
             py::arg("slot"),
             py::arg("sp_idx") = std::nullopt)
        .def("block_ctx",
             static_cast<BlockContext& (Sequence::*)(BlockContextSlot)>(&Sequence::block_ctx),
             py::arg("slot") = BlockContextSlot::ACTIVE,
             py::return_value_policy::reference_internal)
        .def("block_table",
             &Sequence::block_table,
             py::arg("slot"),
             py::arg("sp_idx") = 0,
             py::return_value_policy::reference_internal)
        .def("dp_idx", &Sequence::dp_idx, py::arg("slot"))
        .def("num_blocks", &Sequence::num_blocks, py::arg("slot"), py::arg("sp_idx"))
        .def("last_block_page_id", &Sequence::last_block_page_id, py::arg("slot"), py::arg("sp_idx"))
        .def("last_block_num_tokens", &Sequence::last_block_num_tokens, py::arg("slot"), py::arg("sp_idx"))
        .def("block", &Sequence::block, py::arg("i"), py::arg("slot"), py::arg("sp_idx"))

        .def_readwrite("seq_id", &Sequence::seq_id)
        .def_readwrite("status", &Sequence::status)
        .def_readwrite("token_ids", &Sequence::token_ids)
        .def_readwrite("last_token", &Sequence::last_token)
        .def_readwrite("num_tokens", &Sequence::num_tokens)
        .def_readwrite("num_prompt_tokens", &Sequence::num_prompt_tokens)
        .def_readwrite("num_bootstrap_tokens", &Sequence::num_bootstrap_tokens)
        .def_readwrite("num_checkpointed_tokens", &Sequence::num_checkpointed_tokens)
        .def_readwrite("num_cached_tokens", &Sequence::num_cached_tokens)
        .def_readwrite("metric", &Sequence::metric)
        .def_readwrite("temperature", &Sequence::temperature)
        .def_readwrite("max_tokens", &Sequence::max_tokens)
        .def_readwrite("ignore_eos", &Sequence::ignore_eos)

        .def_property_readonly("is_finished", &Sequence::is_finished)
        .def_property_readonly("num_completed_tokens", &Sequence::num_completed_tokens)
        .def_property_readonly("num_generated_tokens_since_checkpoint",
                               &Sequence::num_generated_tokens_since_checkpoint)
        .def_property_readonly("prompt_token_ids", &Sequence::prompt_token_ids)
        .def_property_readonly("completion_token_ids", &Sequence::completion_token_ids)
        .def_property_readonly("num_cached_blocks", &Sequence::num_cached_blocks)
        .def_property_readonly("materialized_token_count", [](const Sequence& sequence) {
            return sequence.token_ids.size();
        })
        .def("first_invalid_prompt_token", [](const Sequence& sequence, int vocab_size) -> py::object {
            const size_t prompt_size = std::min(static_cast<size_t>(std::max(sequence.num_prompt_tokens, 0)),
                                                sequence.token_ids.size());
            for (size_t index = 0; index < prompt_size; ++index) {
                const int token_id = sequence.token_ids[index];
                if (token_id < 0 || token_id >= vocab_size) {
                    return py::int_(token_id);
                }
            }
            return py::none();
        }, py::arg("vocab_size"))

        .def("__len__", [](const Sequence& s) { return s.num_tokens; })
        .def("__getitem__",
             [](const Sequence& s, py::object key) -> py::object {
                 if (py::isinstance<py::slice>(key)) {
                     py::slice slice_obj = key.cast<py::slice>();
                     size_t    start, stop, step, slicelength;
                     if (!slice_obj.compute(s.token_ids.size(), &start, &stop, &step, &slicelength)) {
                         throw py::error_already_set();
                     }
                     std::vector<int> result;
                     result.reserve(slicelength);
                     for (size_t i = 0; i < slicelength; ++i) {
                         result.push_back(s.token_ids[start]);
                         start += step;
                     }
                     return py::cast(result);
                 }
                 else {
                     int idx = key.cast<int>();
                     if (idx < 0)
                         idx += s.token_ids.size();
                     if (idx < 0 || idx >= static_cast<int>(s.token_ids.size()))
                         throw py::index_error();
                     return py::cast(s.token_ids[idx]);
                 }
             })

        .def(py::pickle(
            [](const Sequence& p) {  // __getstate__
                // (num_tokens, num_checkpointed_tokens, num_cached_tokens,
                // num_bootstrap_tokens, block_ctx_map, temperature,
                // token_ids/last_token)
                std::vector<int> last_element;
                if (p.num_generated_tokens_since_checkpoint() == 0) {
                    last_element = p.token_ids;
                }
                else {
                    last_element = {p.last_token};
                }

                return std::make_tuple(p.num_tokens,
                                       p.num_checkpointed_tokens,
                                       p.num_cached_tokens,
                                       p.num_bootstrap_tokens,
                                       p.slots_,
                                       p.temperature,
                                       last_element);
            },
            [](const std::tuple<int,
                                int,
                                int,
                                int,
                                std::array<BlockContext, (size_t)BlockContextSlot::_COUNT>,
                                double,
                                std::vector<int>>& t) {  // __setstate__
                // We need to reconstruct the object.
                // Since we don't have a constructor that takes all these, we create a dummy one and fill it.
                // Or we can use the existing constructor and then overwrite fields.
                // But the existing constructor requires token_ids.

                // Let's extract token_ids from the last element if possible.
                std::vector<int> last_element = std::get<6>(t);
                std::vector<int> initial_tokens;

                // If num_generated_tokens_since_checkpoint == 0, last_element is token_ids.
                // We can check num_tokens vs num_checkpointed_tokens.
                int num_tokens              = std::get<0>(t);
                int num_checkpointed_tokens = std::get<1>(t);

                if (num_tokens - num_checkpointed_tokens == 0) {
                    initial_tokens = last_element;
                }

                auto seq                     = std::make_shared<Sequence>(initial_tokens);
                seq->num_tokens              = std::get<0>(t);
                seq->num_checkpointed_tokens = std::get<1>(t);
                seq->num_cached_tokens       = std::get<2>(t);
                seq->num_bootstrap_tokens    = std::get<3>(t);

                seq->slots_ = std::move(std::get<4>(t));

                seq->temperature = std::get<5>(t);

                if (num_tokens - num_checkpointed_tokens != 0) {
                    if (!last_element.empty()) {
                        seq->last_token = last_element[0];
                    }
                }

                return seq;
            }))

        .def_readonly_static("block_size", &Sequence::block_size);
}
