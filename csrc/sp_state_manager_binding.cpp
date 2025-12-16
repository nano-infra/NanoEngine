#include "sp_state_manager_core.h"

void bind_sp_state_manager(py::module& m)
{
    py::class_<NDSPStateManager>(m, "SPStateManager")
        .def(py::init<std::optional<std::string>, int, int, int, int, int>(),
             py::arg("engine_id"),
             py::arg("attention_sp"),
             py::arg("num_kvcache_blocks"),
             py::arg("kvcache_block_size"),
             py::arg("max_num_seqs"),
             py::arg("max_num_batched_tokens"))
        .def_property_readonly("is_empty", &NDSPStateManager::is_empty)
        .def_property_readonly(
            "block_manager",
            [](NDSPStateManager& self) {
                py::dict d;
                for (int i = 0; i < self.attention_sp(); ++i) {
                    d[py::int_(i)] = py::cast(&self.block_manager_at(i),
                                             py::return_value_policy::reference_internal,
                                             py::cast(&self));
                }
                return d;
            })
        .def_property_readonly("dummy_seqs", &NDSPStateManager::dummy_seqs)
        .def("dummy_seq_at", &NDSPStateManager::dummy_seq_at, py::arg("sp_idx"))
        .def("running_has_any", &NDSPStateManager::running_has_any)
        .def("running_size", &NDSPStateManager::running_size)
        .def("running_append", &NDSPStateManager::running_append, py::arg("seq"))
        .def("running_popleft", &NDSPStateManager::running_popleft)
        .def("running_pop", &NDSPStateManager::running_pop)
        .def("running_extendleft", &NDSPStateManager::running_extendleft, py::arg("seqs"))
        .def("running_remove_seq_ids", &NDSPStateManager::running_remove_seq_ids, py::arg("seq_ids"))
        .def("running_snapshot", &NDSPStateManager::running_snapshot)
        .def("can_append", &NDSPStateManager::can_append, py::arg("seq"), py::arg("num_tokens") = 1)
        .def("may_append", &NDSPStateManager::may_append, py::arg("seq"), py::arg("num_tokens") = 1)
        .def("can_allocate", &NDSPStateManager::can_allocate, py::arg("seq"), py::arg("num_seqs"), py::arg("num_batched_tokens"))
        .def("allocate", &NDSPStateManager::allocate, py::arg("seq"))
        .def("schedule_decode", &NDSPStateManager::schedule_decode, py::arg("loop_count"))
        .def("deallocate", &NDSPStateManager::deallocate, py::arg("seq"));
}
