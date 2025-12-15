#include "sequence_core.h"

struct ScheduleInputs {
    std::vector<std::vector<Sequence*>> filtered_dp_sp_seqs;
    std::vector<std::vector<Sequence*>> dp_sp_seqs;
    std::vector<std::vector<Sequence*>> dp_sp_tp_seqs;
    std::vector<int>                    dp_batch_sizes;
    std::vector<std::vector<int>>       sp_batch_sizes;
};

ScheduleInputs prepare_step_inputs(py::list dp_seqs_list, std::string engine_id, int sp_size, int tp_size)
{
    ScheduleInputs             results;
    std::optional<std::string> opt_engine_id = engine_id;
    size_t                     dp_size       = dp_seqs_list.size();

    results.filtered_dp_sp_seqs.reserve(dp_size * sp_size);
    results.dp_sp_seqs.reserve(dp_size * sp_size);
    results.dp_sp_tp_seqs.reserve(dp_size * sp_size * tp_size);
    results.dp_batch_sizes.reserve(dp_size);
    results.sp_batch_sizes.reserve(dp_size);

    // [Loop 1] 遍历 DP Groups
    for (size_t dp_idx = 0; dp_idx < dp_size; ++dp_idx) {
        py::list               current_dp_seqs_py = dp_seqs_list[dp_idx].cast<py::list>();
        std::vector<Sequence*> current_dp_seqs;
        current_dp_seqs.reserve(current_dp_seqs_py.size());
        for (auto item : current_dp_seqs_py) {
            current_dp_seqs.push_back(item.cast<Sequence*>());
        }

        // [Output 1] dp_batch_sizes
        results.dp_batch_sizes.push_back((int)current_dp_seqs.size());

        std::vector<int> current_sp_batch_sizes;
        current_sp_batch_sizes.reserve(sp_size);

        // [Loop 2] 遍历 SP Ranks
        for (int sp_idx = 0; sp_idx < sp_size; ++sp_idx) {

            // [Output 2] dp_sp_seqs
            results.dp_sp_seqs.push_back(current_dp_seqs);

            // [Output 3] filtered_dp_sp_seqs
            std::vector<Sequence*> filtered_seqs;
            filtered_seqs.reserve(current_dp_seqs.size() / sp_size + 2);

            for (Sequence* seq : current_dp_seqs) {
                auto it = seq->block_ctx_map.find(opt_engine_id);
                if (it == seq->block_ctx_map.end()) {
                    it = seq->block_ctx_map.find(seq->active_engine_id);
                }

                if (it != seq->block_ctx_map.end()) {
                    if (it->second.master_sp_idx == sp_idx) {
                        filtered_seqs.push_back(seq);
                    }
                }
            }

            current_sp_batch_sizes.push_back((int)filtered_seqs.size());

            results.filtered_dp_sp_seqs.push_back(std::move(filtered_seqs));
        }

        results.sp_batch_sizes.push_back(std::move(current_sp_batch_sizes));
    }

    // [Loop 3] 构建 dp_sp_tp_seqs
    for (const auto& seqs : results.dp_sp_seqs) {
        for (int i = 0; i < tp_size; ++i) {
            results.dp_sp_tp_seqs.push_back(seqs);
        }
    }

    return results;
}

void bind_scheduler_ops(py::module& m)
{
    py::class_<ScheduleInputs>(m, "ScheduleInputs")
        .def_readwrite("filtered_dp_sp_seqs", &ScheduleInputs::filtered_dp_sp_seqs)
        .def_readwrite("dp_sp_seqs", &ScheduleInputs::dp_sp_seqs)
        .def_readwrite("dp_sp_tp_seqs", &ScheduleInputs::dp_sp_tp_seqs)
        .def_readwrite("dp_batch_sizes", &ScheduleInputs::dp_batch_sizes)
        .def_readwrite("sp_batch_sizes", &ScheduleInputs::sp_batch_sizes);

    m.def("prepare_step_inputs",
          &prepare_step_inputs,
          py::return_value_policy::move,
          py::arg("dp_seqs_list"),
          py::arg("engine_id"),
          py::arg("sp_size"),
          py::arg("tp_size"),
          "Optimized input preparation for scheduler step");
}
