#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "sequence.h"
#include "sequence_metric.h"

namespace py = pybind11;
using namespace nanodeploy;

PYBIND11_MAKE_OPAQUE(std::unordered_map<int, int>);

void bind_sequence(py::module_& m) {
    py::enum_<SequenceStatus>(m, "SequenceStatus")
        .value("WAITING", SequenceStatus::WAITING)
        .value("RUNNING", SequenceStatus::RUNNING)
        .value("FINISHED", SequenceStatus::FINISHED)
        .value("TO_BE_MIGRATED", SequenceStatus::TO_BE_MIGRATED)
        .export_values();
    
    // Wrapper class for num_dispatched_tokens to provide defaultdict(int) behavior
    py::class_<std::unordered_map<int, int>>(m, "DefaultIntDict")
        .def("__getitem__", [](std::unordered_map<int, int>& self, int key) -> int& {
            // This mimics defaultdict(int) - returns 0 for missing keys and creates entry
            return self[key];
        }, py::return_value_policy::reference_internal)
        .def("__setitem__", [](std::unordered_map<int, int>& self, int key, int value) {
            self[key] = value;
        })
        .def("__contains__", [](const std::unordered_map<int, int>& self, int key) {
            return self.find(key) != self.end();
        })
        .def("keys", [](const std::unordered_map<int, int>& self) {
            py::list keys;
            for (const auto& pair : self) {
                keys.append(pair.first);
            }
            return keys;
        })
        .def("items", [](const std::unordered_map<int, int>& self) {
            py::list items;
            for (const auto& pair : self) {
                items.append(py::make_tuple(pair.first, pair.second));
            }
            return items;
        })
        .def("clear", [](std::unordered_map<int, int>& self) {
            self.clear();
        })
        .def(py::pickle(
            [](const std::unordered_map<int, int> &self) {
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
            }
        ));
    
    py::class_<BlockContext>(m, "BlockContext")
        .def(py::init<>())
        .def(py::init<const std::optional<std::string>&, int, int, int, int>(),
             py::arg("engine_id"), py::arg("dp_idx"), py::arg("master_sp_idx"),
             py::arg("attention_sp"), py::arg("attention_dp"))
        .def_readwrite("engine_id", &BlockContext::engine_id)
        .def_readwrite("dp_idx", &BlockContext::dp_idx)
        .def_readwrite("master_sp_idx", &BlockContext::master_sp_idx)
        .def_readwrite("attention_sp", &BlockContext::attention_sp)
        .def_readwrite("attention_dp", &BlockContext::attention_dp)
        .def_readwrite("block_location", &BlockContext::block_location)
        .def_readwrite("sp_block_table", &BlockContext::sp_block_table)
        .def_property("num_dispatched_tokens",
            [](BlockContext& self) -> std::unordered_map<int, int>& { 
                return self.num_dispatched_tokens; 
            },
            [](BlockContext& self, const std::unordered_map<int, int>& value) {
                self.num_dispatched_tokens = value;
            },
            py::return_value_policy::reference_internal)
        .def(py::pickle(
            [](const BlockContext &p) {
                return p.getstate();
            },
            [](const std::tuple<std::optional<std::string>, int, int, int, int, 
               std::vector<std::pair<int, int>>,
               std::unordered_map<int, std::vector<int>>,
               std::unordered_map<int, int>>& t) {
                return BlockContext::setstate(t);
            }
        ));
    
    py::class_<Sequence, std::shared_ptr<Sequence>>(m, "Sequence")
        .def(py::init<const std::vector<int>&, double, int, bool,
                      const std::optional<std::string>&, int>(),
             py::arg("token_ids"),
             py::arg("temperature") = 1.0,
             py::arg("max_tokens") = 256,
             py::arg("ignore_eos") = false,
             py::arg("engine_id") = std::nullopt,
             py::arg("master_sp_rank") = 0)
        
        .def("set_engine_id", &Sequence::set_engine_id,
             py::arg("engine_id"),
             py::arg("attention_dp") = 1,
             py::arg("attention_sp") = 1)
        .def("context_len", &Sequence::context_len,
             py::arg("engine_id") = std::nullopt,
             py::arg("sp_idx") = std::nullopt)
        .def("append_token", &Sequence::append_token,
             py::arg("token_id"),
             py::arg("engine_id") = std::nullopt,
             py::arg("sp_idx") = std::nullopt)

           .def("block_table_append", &Sequence::block_table_append,
               py::arg("block_id"),
               py::arg("engine_id") = std::nullopt,
               py::arg("sp_idx") = 0)
           .def("block_table_clear", &Sequence::block_table_clear,
               py::arg("engine_id") = std::nullopt,
               py::arg("sp_idx") = 0)
           .def("block_table_set", &Sequence::block_table_set,
               py::arg("table"),
               py::arg("engine_id") = std::nullopt,
               py::arg("sp_idx") = 0)

           .def("block_location_append", &Sequence::block_location_append,
               py::arg("sp_idx"),
               py::arg("block_id"),
               py::arg("engine_id") = std::nullopt)
           .def("block_location_clear", &Sequence::block_location_clear,
               py::arg("engine_id") = std::nullopt)

           .def("sp_block_table_clear", &Sequence::sp_block_table_clear,
               py::arg("engine_id") = std::nullopt)
           .def("num_dispatched_tokens_clear", &Sequence::num_dispatched_tokens_clear,
               py::arg("engine_id") = std::nullopt)
        .def("block_ctx", 
             static_cast<BlockContext& (Sequence::*)(const std::optional<std::string>&)>
             (&Sequence::block_ctx),
             py::arg("engine_id") = std::nullopt,
             py::return_value_policy::reference_internal)
        .def("block_table", &Sequence::block_table,
             py::arg("engine_id") = std::nullopt,
             py::arg("sp_idx") = 0,
             py::return_value_policy::reference_internal)
        .def("dp_idx", &Sequence::dp_idx, py::arg("engine_id"))
        .def("num_blocks", &Sequence::num_blocks, py::arg("engine_id"), py::arg("sp_idx"))
        .def("last_block_page_id", &Sequence::last_block_page_id, py::arg("engine_id"), py::arg("sp_idx"))
        .def("last_block_num_tokens", &Sequence::last_block_num_tokens, py::arg("engine_id"), py::arg("sp_idx"))
        .def("block", &Sequence::block, py::arg("i"), py::arg("engine_id"), py::arg("sp_idx"))
        
        .def_readwrite("seq_id", &Sequence::seq_id)
        .def_readwrite("status", &Sequence::status)
        .def_readwrite("token_ids", &Sequence::token_ids)
        .def_readwrite("last_token", &Sequence::last_token)
        .def_readwrite("num_tokens", &Sequence::num_tokens)
        .def_readwrite("num_prompt_tokens", &Sequence::num_prompt_tokens)
        .def_readwrite("num_checkpointed_tokens", &Sequence::num_checkpointed_tokens)
        .def_readwrite("num_cached_tokens", &Sequence::num_cached_tokens)
        .def_readwrite("backup_engine_id", &Sequence::backup_engine_id)
        .def_readwrite("active_engine_id", &Sequence::active_engine_id)
        .def_readwrite("block_ctx_map", &Sequence::block_ctx_map)
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
        
        .def("__len__", [](const Sequence& s) { return s.num_tokens; })
        .def("__getitem__", [](const Sequence& s, py::object key) -> py::object {
            if (py::isinstance<py::slice>(key)) {
                py::slice slice_obj = key.cast<py::slice>();
                size_t start, stop, step, slicelength;
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
            } else {
                int idx = key.cast<int>();
                if (idx < 0) idx += s.token_ids.size();
                if (idx < 0 || idx >= static_cast<int>(s.token_ids.size())) throw py::index_error();
                return py::cast(s.token_ids[idx]);
            }
        })
        
        .def(py::pickle(
            [](const Sequence &p) { // __getstate__
                // (num_tokens, num_checkpointed_tokens, num_cached_tokens, backup_engine_id, active_engine_id, block_ctx_map, temperature, token_ids/last_token)
                std::vector<int> last_element;
                if (p.num_generated_tokens_since_checkpoint() == 0) {
                    last_element = p.token_ids;
                } else {
                    last_element = {p.last_token};
                }
                
                // Convert block_ctx_map to list of tuples to avoid map conversion issues
                py::list block_ctx_list;
                for (const auto& pair : p.block_ctx_map) {
                    block_ctx_list.append(py::make_tuple(pair.first, pair.second));
                }

                return std::make_tuple(
                    p.num_tokens,
                    p.num_checkpointed_tokens,
                    p.num_cached_tokens,
                    p.backup_engine_id,
                    p.active_engine_id,
                    block_ctx_list,
                    p.temperature,
                    last_element
                );
            },
            [](const std::tuple<int, int, int, std::optional<std::string>, std::optional<std::string>,
                                py::list,
                                double, std::vector<int>> &t) { // __setstate__
                
                // We need to reconstruct the object.
                // Since we don't have a constructor that takes all these, we create a dummy one and fill it.
                // Or we can use the existing constructor and then overwrite fields.
                // But the existing constructor requires token_ids.
                
                // Let's extract token_ids from the last element if possible.
                std::vector<int> last_element = std::get<7>(t);
                std::vector<int> initial_tokens;
                
                // If num_generated_tokens_since_checkpoint == 0, last_element is token_ids.
                // We can check num_tokens vs num_checkpointed_tokens.
                int num_tokens = std::get<0>(t);
                int num_checkpointed_tokens = std::get<1>(t);
                
                if (num_tokens - num_checkpointed_tokens == 0) {
                    initial_tokens = last_element;
                }
                
                auto seq = std::make_shared<Sequence>(initial_tokens);
                seq->num_tokens = std::get<0>(t);
                seq->num_checkpointed_tokens = std::get<1>(t);
                seq->num_cached_tokens = std::get<2>(t);
                seq->backup_engine_id = std::get<3>(t);
                seq->active_engine_id = std::get<4>(t);
                
                // Reconstruct block_ctx_map
                py::list block_ctx_list = std::get<5>(t);
                seq->block_ctx_map.clear(); // Clear default one
                for (auto item : block_ctx_list) {
                    auto tuple = item.cast<py::tuple>();
                    auto eid = tuple[0].cast<std::optional<std::string>>();
                    auto ctx = tuple[1].cast<BlockContext>();
                    seq->block_ctx_map[eid] = ctx;
                }

                seq->temperature = std::get<6>(t);
                
                if (num_tokens - num_checkpointed_tokens != 0) {
                    if (!last_element.empty()) {
                        seq->last_token = last_element[0];
                    }
                }
                
                return seq;
            }
        ))
        
        .def_readonly_static("block_size", &Sequence::block_size);
}
