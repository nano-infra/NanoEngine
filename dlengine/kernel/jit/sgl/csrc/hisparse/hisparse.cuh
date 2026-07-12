#include <sgl_kernel/hisparse/hisparse.cuh>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

struct HiSparseRingMappingKernel {
    static void run(tvm::ffi::TensorView slots,
                    tvm::ffi::TensorView positions,
                    tvm::ffi::TensorView output,
                    tvm::ffi::TensorView num_real_reqs,
                    int64_t              max_num_seqs,
                    int64_t              tokens_per_seq)
    {
        using namespace host;

        auto N      = SymbolicSize{"num_tokens"};
        auto device = SymbolicDevice{};
        device.set_options<kDLCUDA>();
        TensorMatcher({N}).with_dtype<int64_t>().with_device(device).verify(slots);
        TensorMatcher({N}).with_dtype<int64_t>().with_device(device).verify(positions);
        TensorMatcher({N}).with_dtype<int32_t>().with_device(device).verify(output);
        TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(num_real_reqs);

        const auto n = N.unwrap();
        if (n == 0) {
            return;
        }
        const auto stream = LaunchKernel::resolve_device(device.unwrap());
        sgl_kernel::hisparse::launch_build_ring_slot_mapping(static_cast<const int64_t*>(slots.data_ptr()),
                                                             static_cast<const int64_t*>(positions.data_ptr()),
                                                             static_cast<int32_t*>(output.data_ptr()),
                                                             static_cast<const int32_t*>(num_real_reqs.data_ptr()),
                                                             n,
                                                             max_num_seqs,
                                                             tokens_per_seq,
                                                             stream);
    }
};

}  // namespace
