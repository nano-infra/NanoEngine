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

struct HiSparseMLASlotLoadKernel {
    static void run(tvm::ffi::TensorView indices,
                    tvm::ffi::TensorView request_slots,
                    tvm::ffi::TensorView cold,
                    tvm::ffi::TensorView hot,
                    tvm::ffi::TensorView output,
                    tvm::ffi::TensorView hot_output_slots,
                    tvm::ffi::TensorView num_real_reqs,
                    int64_t              max_num_seqs,
                    int64_t              hot_capacity,
                    int64_t              slot_stride_tokens)
    {
        using namespace host;
        auto R    = SymbolicSize{"num_rows"};
        auto K    = SymbolicSize{"topk"};
        auto B    = SymbolicSize{"num_blocks"};
        auto HB   = SymbolicSize{"hot_blocks"};
        auto P    = SymbolicSize{"block_size"};
        auto H    = SymbolicSize{"num_heads"};
        auto D    = SymbolicSize{"head_dim"};
        auto CSB  = SymbolicSize{"cold_block_stride"};
        auto CST  = SymbolicSize{"cold_token_stride"};
        auto HSB  = SymbolicSize{"hot_block_stride"};
        auto HST  = SymbolicSize{"hot_token_stride"};
        auto cuda = SymbolicDevice{};
        cuda.set_options<kDLCUDA>();
        TensorMatcher({R, K}).with_dtype<int32_t>().with_device(cuda).verify(indices).verify(output);
        TensorMatcher({R}).with_dtype<int64_t>().with_device(cuda).verify(request_slots);
        TensorMatcher({R}).with_dtype<int32_t>().with_device(cuda).verify(hot_output_slots);
        TensorMatcher({1}).with_dtype<int32_t>().with_device(cuda).verify(num_real_reqs);
        // cold intentionally remains a CPU tensor: PeerAgent registration pins
        // and CUDA-maps it for direct kernel reads.
        TensorMatcher({B, P, H, D}).with_strides({CSB, CST, D, 1}).verify(cold);
        TensorMatcher({HB, P, H, D}).with_strides({HSB, HST, D, 1}).with_device(cuda).verify(hot);
        RuntimeCheck(cold.dtype() == hot.dtype(), "cold/hot dtype mismatch");
        const int64_t element_bytes = cold.dtype().bits / 8;
        const int64_t item_bytes    = H.unwrap() * D.unwrap() * element_bytes;
        const auto    stream        = LaunchKernel::resolve_device(cuda.unwrap());
        sgl_kernel::hisparse::launch_load_mla_slot(static_cast<const int32_t*>(indices.data_ptr()),
                                                   static_cast<const int64_t*>(request_slots.data_ptr()),
                                                   cold.data_ptr(),
                                                   hot.data_ptr(),
                                                   static_cast<int32_t*>(output.data_ptr()),
                                                   static_cast<int32_t*>(hot_output_slots.data_ptr()),
                                                   static_cast<const int32_t*>(num_real_reqs.data_ptr()),
                                                   R.unwrap(),
                                                   K.unwrap(),
                                                   max_num_seqs,
                                                   hot_capacity,
                                                   slot_stride_tokens,
                                                   P.unwrap(),
                                                   cold.strides()[0] * element_bytes,
                                                   cold.strides()[1] * element_bytes,
                                                   hot.strides()[0] * element_bytes,
                                                   hot.strides()[1] * element_bytes,
                                                   item_bytes,
                                                   stream);
    }
};

struct HiSparseMLASlotWritebackKernel {
    static void run(tvm::ffi::TensorView logical_slots,
                    tvm::ffi::TensorView hot_slots,
                    tvm::ffi::TensorView hot,
                    tvm::ffi::TensorView cold,
                    tvm::ffi::TensorView num_real_reqs)
    {
        using namespace host;
        auto R    = SymbolicSize{"num_rows"};
        auto B    = SymbolicSize{"num_blocks"};
        auto HB   = SymbolicSize{"hot_blocks"};
        auto P    = SymbolicSize{"block_size"};
        auto H    = SymbolicSize{"num_heads"};
        auto D    = SymbolicSize{"head_dim"};
        auto HSB  = SymbolicSize{"hot_block_stride"};
        auto HST  = SymbolicSize{"hot_token_stride"};
        auto CSB  = SymbolicSize{"cold_block_stride"};
        auto CST  = SymbolicSize{"cold_token_stride"};
        auto cuda = SymbolicDevice{};
        cuda.set_options<kDLCUDA>();
        TensorMatcher({R}).with_dtype<int32_t>().with_device(cuda).verify(logical_slots).verify(hot_slots);
        TensorMatcher({1}).with_dtype<int32_t>().with_device(cuda).verify(num_real_reqs);
        TensorMatcher({HB, P, H, D}).with_strides({HSB, HST, D, 1}).with_device(cuda).verify(hot);
        TensorMatcher({B, P, H, D}).with_strides({CSB, CST, D, 1}).verify(cold);
        RuntimeCheck(cold.dtype() == hot.dtype(), "cold/hot dtype mismatch");
        const int64_t element_bytes = cold.dtype().bits / 8;
        const int64_t item_bytes    = H.unwrap() * D.unwrap() * element_bytes;
        const auto    stream        = LaunchKernel::resolve_device(cuda.unwrap());
        sgl_kernel::hisparse::launch_writeback_mla_slot(static_cast<const int32_t*>(logical_slots.data_ptr()),
                                                        static_cast<const int32_t*>(hot_slots.data_ptr()),
                                                        hot.data_ptr(),
                                                        cold.data_ptr(),
                                                        static_cast<const int32_t*>(num_real_reqs.data_ptr()),
                                                        R.unwrap(),
                                                        P.unwrap(),
                                                        hot.strides()[0] * element_bytes,
                                                        hot.strides()[1] * element_bytes,
                                                        cold.strides()[0] * element_bytes,
                                                        cold.strides()[1] * element_bytes,
                                                        item_bytes,
                                                        stream);
    }
};

}  // namespace
