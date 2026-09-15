#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cstdint>

namespace {

constexpr uint32_t kSMEM = 32 * 1024 * sizeof(uint32_t);  // 128KB: exact through 16K contexts
// Use a bounded threshold candidate buffer to preserve kernel occupancy.
constexpr uint32_t kRaggedSMEM = 8 * 1024 * sizeof(uint32_t);  // 32KB

struct TopKParams {
    const float* __restrict__ scores;
    const int32_t* __restrict__ seq_lens;
    const int32_t* __restrict__ page_table;
    int32_t* __restrict__ page_indices;
    int32_t* __restrict__ raw_indices;  // optional: output raw abs position indices before page transform
    const int64_t score_stride;
    const int64_t page_table_stride;
    uint32_t      page_bits;
};

struct RaggedTopKParams {
    const float* __restrict__ scores;
    const int32_t* __restrict__ seq_lens;
    const int32_t* __restrict__ row_starts;
    const int32_t* __restrict__ offsets;
    int32_t* __restrict__ indices;
    const int64_t score_stride;
};

SGL_DEVICE uint8_t convert_to_uint8(float x)
{
    __half   h    = __float2half_rn(x);
    uint16_t bits = __half_as_ushort(h);
    uint16_t key  = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
    return static_cast<uint8_t>(key >> 8);
}

SGL_DEVICE uint32_t convert_to_uint32(float x)
{
    uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

SGL_DEVICE int32_t page_to_indices(const int32_t* __restrict__ page_table, uint32_t i, uint32_t page_bits)
{
    const uint32_t mask = (1u << page_bits) - 1u;
    return (page_table[i >> page_bits] << page_bits) | (i & mask);
}

template<uint32_t kTopK, uint32_t kTopKBlockSize>
[[maybe_unused]]
SGL_DEVICE void naive_transform(const float* __restrict__,  // unused
                                const int32_t* __restrict__ page_table,
                                int32_t* __restrict__ indices,
                                int32_t* __restrict__ raw_indices,  // optional: output raw abs position indices
                                const uint32_t length,
                                const uint32_t page_bits)
{
    for (uint32_t i = threadIdx.x; i < kTopK; i += kTopKBlockSize) {
        if (i < length) {
            indices[i] = page_to_indices(page_table, i, page_bits);
            if (raw_indices != nullptr) {
                raw_indices[i] = i;
            }
        }
        else {
            indices[i] = -1;  // fill invalid indices to -1
            if (raw_indices != nullptr) {
                raw_indices[i] = -1;
            }
        }
    }
}

template<uint32_t kTopK, uint32_t kTopKBlockSize, uint32_t kRadixSMEM = kSMEM>
[[maybe_unused]]
SGL_DEVICE void radix_topk(const float* __restrict__ input, int32_t* __restrict__ output, const uint32_t length)
{
    constexpr uint32_t RADIX           = 256;
    constexpr uint32_t BLOCK_SIZE      = kTopKBlockSize;
    constexpr uint32_t SMEM_INPUT_SIZE = kRadixSMEM / (2 * sizeof(int32_t));

    alignas(128) __shared__ uint32_t _s_histogram_buf[2][RADIX + 32];
    alignas(128) __shared__ uint32_t s_counter;
    alignas(128) __shared__ uint32_t s_threshold_bin_id;
    alignas(128) __shared__ uint32_t s_num_input[2];
    alignas(128) __shared__ int32_t  s_last_remain;

    extern __shared__ uint32_t s_input_idx[];

    const uint32_t tx          = threadIdx.x;
    uint32_t       remain_topk = kTopK;
    auto&          s_histogram = _s_histogram_buf[0];

    const auto run_cumsum = [&] {
#pragma unroll 8
        for (int32_t i = 0; i < 8; ++i) {
            static_assert(1 << 8 == RADIX);
            if (tx < RADIX) {
                const auto j     = 1 << i;
                const auto k     = i & 1;
                auto       value = _s_histogram_buf[k][tx];
                if (tx + j < RADIX) {
                    value += _s_histogram_buf[k][tx + j];
                }
                _s_histogram_buf[k ^ 1][tx] = value;
            }
            __syncthreads();
        }
    };

    // stage 1: 8bit coarse histogram
    if (tx < RADIX + 1)
        s_histogram[tx] = 0;
    __syncthreads();
    for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
        const auto bin = convert_to_uint8(input[idx]);
        ::atomicAdd(&s_histogram[bin], 1);
    }
    __syncthreads();
    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > remain_topk && s_histogram[tx + 1] <= remain_topk) {
        s_threshold_bin_id = tx;
        s_num_input[0]     = 0;
        s_counter          = 0;
    }
    __syncthreads();

    const auto threshold_bin = s_threshold_bin_id;
    remain_topk -= s_histogram[threshold_bin + 1];
    if (remain_topk == 0) {
        for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
            const uint32_t bin = convert_to_uint8(input[idx]);
            if (bin > threshold_bin) {
                const auto pos = ::atomicAdd(&s_counter, 1);
                output[pos]    = idx;
            }
        }
        __syncthreads();
        return;
    }
    else {
        __syncthreads();
        if (tx < RADIX + 1) {
            s_histogram[tx] = 0;
        }
        __syncthreads();

        for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
            const float    raw_input = input[idx];
            const uint32_t bin       = convert_to_uint8(raw_input);
            if (bin > threshold_bin) {
                const auto pos = ::atomicAdd(&s_counter, 1);
                output[pos]    = idx;
            }
            else if (bin == threshold_bin) {
                const auto pos = ::atomicAdd(&s_num_input[0], 1);
                if (pos < SMEM_INPUT_SIZE) {
                    [[likely]] s_input_idx[pos] = idx;
                    const auto bin              = convert_to_uint32(raw_input);
                    const auto sub_bin          = (bin >> 24) & 0xFF;
                    ::atomicAdd(&s_histogram[sub_bin], 1);
                }
            }
        }
        __syncthreads();
    }

    // stage 2: refine with 8bit radix passes
#pragma unroll 4
    for (int round = 0; round < 4; ++round) {
        const auto r_idx = round % 2;

        // clip here to prevent overflow
        const auto raw_num_input = s_num_input[r_idx];
        const auto num_input     = raw_num_input < SMEM_INPUT_SIZE ? raw_num_input : SMEM_INPUT_SIZE;

        run_cumsum();
        if (tx < RADIX && s_histogram[tx] > remain_topk && s_histogram[tx + 1] <= remain_topk) {
            s_threshold_bin_id     = tx;
            s_num_input[r_idx ^ 1] = 0;
            s_last_remain          = remain_topk - s_histogram[tx + 1];
        }
        __syncthreads();

        const auto threshold_bin = s_threshold_bin_id;
        remain_topk -= s_histogram[threshold_bin + 1];

        if (remain_topk == 0) {
            for (uint32_t i = tx; i < num_input; i += BLOCK_SIZE) {
                const auto idx    = s_input_idx[r_idx * SMEM_INPUT_SIZE + i];
                const auto offset = 24 - round * 8;
                const auto bin    = (convert_to_uint32(input[idx]) >> offset) & 0xFF;
                if (bin > threshold_bin) {
                    const auto pos = ::atomicAdd(&s_counter, 1);
                    output[pos]    = idx;
                }
            }
            __syncthreads();
            break;
        }
        else {
            __syncthreads();
            if (tx < RADIX + 1) {
                s_histogram[tx] = 0;
            }
            __syncthreads();
            for (uint32_t i = tx; i < num_input; i += BLOCK_SIZE) {
                const auto idx       = s_input_idx[r_idx * SMEM_INPUT_SIZE + i];
                const auto raw_input = input[idx];
                const auto offset    = 24 - round * 8;
                const auto bin       = (convert_to_uint32(raw_input) >> offset) & 0xFF;
                if (bin > threshold_bin) {
                    const auto pos = ::atomicAdd(&s_counter, 1);
                    output[pos]    = idx;
                }
                else if (bin == threshold_bin) {
                    if (round == 3) {
                        const auto pos = ::atomicAdd(&s_last_remain, -1);
                        if (pos > 0) {
                            output[kTopK - pos] = idx;
                        }
                    }
                    else {
                        const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
                        if (pos < SMEM_INPUT_SIZE) {
                            /// NOTE: (dark) fuse the histogram computation here
                            [[likely]] s_input_idx[(r_idx ^ 1) * SMEM_INPUT_SIZE + pos] = idx;
                            const auto bin                                              = convert_to_uint32(raw_input);
                            const auto sub_bin                                          = (bin >> (offset - 8)) & 0xFF;
                            ::atomicAdd(&s_histogram[sub_bin], 1);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
}

template<uint32_t kTopK, uint32_t kTopKBlockSize, bool kUsePDL>
__global__ void topk_transform(const __grid_constant__ TopKParams params)
{
    const auto& [scores,
                 seq_lens,
                 page_table,
                 page_indices,
                 raw_indices,  // pointers
                 score_stride,
                 page_table_stride,
                 page_bits  // sizes
    ]                      = params;
    const uint32_t work_id = blockIdx.x;

    /// NOTE: dangerous prefetch seq_len before PDL wait
    const uint32_t seq_len         = seq_lens[work_id];
    const auto     score_ptr       = scores + work_id * score_stride;
    const auto     page_ptr        = page_table + work_id * page_table_stride;
    const auto     indices_ptr     = page_indices + work_id * kTopK;
    const auto     raw_indices_ptr = raw_indices != nullptr ? raw_indices + work_id * kTopK : nullptr;

    device::PDLWaitPrimary<kUsePDL>();

    if (seq_len <= kTopK) {
        naive_transform<kTopK, kTopKBlockSize>(score_ptr, page_ptr, indices_ptr, raw_indices_ptr, seq_len, page_bits);
    }
    else {
        __shared__ int32_t s_topk_indices[kTopK];
        radix_topk<kTopK, kTopKBlockSize>(score_ptr, s_topk_indices, seq_len);
        for (uint32_t i = threadIdx.x; i < kTopK; i += kTopKBlockSize) {
            indices_ptr[i] = page_to_indices(page_ptr, s_topk_indices[i], page_bits);
            if (raw_indices_ptr != nullptr) {
                raw_indices_ptr[i] = s_topk_indices[i];
            }
        }
    }

    device::PDLTriggerSecondary<kUsePDL>();
}

template<uint32_t kTopK, uint32_t kTopKBlockSize, bool kUsePDL>
__global__ void topk_transform_ragged(const __grid_constant__ RaggedTopKParams params)
{
    const auto& [scores, seq_lens, row_starts, offsets, indices, score_stride] = params;
    const uint32_t work_id                                                     = blockIdx.x;
    const uint32_t seq_len                                                     = seq_lens[work_id];
    const uint32_t row_start                                                   = row_starts[work_id];
    const int32_t  offset                                                      = offsets[work_id];
    const auto     score_ptr                                                   = scores + work_id * score_stride;
    const auto     indices_ptr                                                 = indices + work_id * kTopK;

    device::PDLWaitPrimary<kUsePDL>();

    if (seq_len <= kTopK) {
        for (uint32_t i = threadIdx.x; i < kTopK; i += kTopKBlockSize) {
            indices_ptr[i] = i < seq_len ? static_cast<int32_t>(i) + offset : -1;
        }
    }
    else {
        __shared__ int32_t s_topk_indices[kTopK];
        radix_topk<kTopK, kTopKBlockSize, kRaggedSMEM>(score_ptr + row_start, s_topk_indices, seq_len);
        for (uint32_t i = threadIdx.x; i < kTopK; i += kTopKBlockSize) {
            indices_ptr[i] = s_topk_indices[i] + offset;
        }
    }

    device::PDLTriggerSecondary<kUsePDL>();
}

template<auto* f, size_t kMaxDynamicSMEM>
void setup_kernel_smem_once(host::DebugInfo where = {})
{
    [[maybe_unused]]
    static const auto result = [] {
        const auto fptr = std::bit_cast<const void*>(f);
        return ::cudaFuncSetAttribute(fptr, ::cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxDynamicSMEM);
    }();
    host::RuntimeDeviceCheck(result, where);
}

template<uint32_t kTopK, bool kUsePDL>
struct TopKKernel {
    static_assert(kTopK == 512 || kTopK == 2048, "supported top-k sizes are 512 and 2048");
    static constexpr uint32_t kTopKBlockSize = kTopK < 1024 ? kTopK : 1024;
    static constexpr auto     kernel         = topk_transform<kTopK, kTopKBlockSize, kUsePDL>;
    static constexpr auto     ragged_kernel  = topk_transform_ragged<kTopK, kTopKBlockSize, kUsePDL>;

    static void transform(const tvm::ffi::TensorView                     scores,
                          const tvm::ffi::TensorView                     seq_lens,
                          const tvm::ffi::TensorView                     page_table,
                          const tvm::ffi::TensorView                     page_indices,
                          const uint32_t                                 page_size,
                          const tvm::ffi::Optional<tvm::ffi::TensorView> raw_indices)
    {
        using namespace host;
        auto B      = SymbolicSize{"batch_size"};
        auto S      = SymbolicSize{"score_stride"};
        auto P      = SymbolicSize{"page_table_stride"};
        auto device = SymbolicDevice{};
        device.set_options<kDLCUDA>();

        TensorMatcher({B, -1})  // strided scores
            .with_strides({S, 1})
            .with_dtype<float>()
            .with_device(device)
            .verify(scores);
        TensorMatcher({B})  // seq_lens, must be contiguous
            .with_dtype<int32_t>()
            .with_device(device)
            .verify(seq_lens);
        TensorMatcher({B, -1})  // strided page table
            .with_strides({P, 1})
            .with_dtype<int32_t>()
            .with_device(device)
            .verify(page_table);
        TensorMatcher({B, kTopK})  // output, must be contiguous
            .with_dtype<int32_t>()
            .with_device(device)
            .verify(page_indices);

        int32_t* raw_indices_ptr = nullptr;
        if (raw_indices.has_value()) {
            TensorMatcher({B, kTopK})  // optional raw indices output, must be contiguous
                .with_dtype<int32_t>()
                .with_device(device)
                .verify(raw_indices.value());
            raw_indices_ptr = static_cast<int32_t*>(raw_indices.value().data_ptr());
        }

        RuntimeCheck(std::has_single_bit(page_size), "page_size must be power of 2");
        const auto page_bits  = static_cast<uint32_t>(std::countr_zero(page_size));
        const auto batch_size = static_cast<uint32_t>(B.unwrap());
        const auto params     = TopKParams{
                .scores            = static_cast<float*>(scores.data_ptr()),
                .seq_lens          = static_cast<int32_t*>(seq_lens.data_ptr()),
                .page_table        = static_cast<int32_t*>(page_table.data_ptr()),
                .page_indices      = static_cast<int32_t*>(page_indices.data_ptr()),
                .raw_indices       = raw_indices_ptr,
                .score_stride      = S.unwrap(),
                .page_table_stride = P.unwrap(),
                .page_bits         = page_bits,
        };
        constexpr auto kSMEM_ = kSMEM + sizeof(int32_t);  // align up a little
        setup_kernel_smem_once<kernel, kSMEM_>();
        LaunchKernel(batch_size, kTopKBlockSize, device.unwrap(), kSMEM_).enable_pdl(kUsePDL)(kernel, params);
    }

    static void transform_ragged(const tvm::ffi::TensorView scores,
                                 const tvm::ffi::TensorView seq_lens,
                                 const tvm::ffi::TensorView row_starts,
                                 const tvm::ffi::TensorView offsets,
                                 const tvm::ffi::TensorView indices)
    {
        using namespace host;
        auto B      = SymbolicSize{"batch_size"};
        auto S      = SymbolicSize{"score_stride"};
        auto device = SymbolicDevice{};
        device.set_options<kDLCUDA>();

        TensorMatcher({B, -1}).with_strides({S, 1}).with_dtype<float>().with_device(device).verify(scores);
        for (const auto& tensor : {seq_lens, row_starts, offsets}) {
            TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(tensor);
        }
        TensorMatcher({B, kTopK}).with_dtype<int32_t>().with_device(device).verify(indices);

        const auto batch_size = static_cast<uint32_t>(B.unwrap());
        const auto params     = RaggedTopKParams{
                .scores       = static_cast<float*>(scores.data_ptr()),
                .seq_lens     = static_cast<int32_t*>(seq_lens.data_ptr()),
                .row_starts   = static_cast<int32_t*>(row_starts.data_ptr()),
                .offsets      = static_cast<int32_t*>(offsets.data_ptr()),
                .indices      = static_cast<int32_t*>(indices.data_ptr()),
                .score_stride = S.unwrap(),
        };
        constexpr auto kRaggedSMEM_ = kRaggedSMEM + sizeof(int32_t);
        setup_kernel_smem_once<ragged_kernel, kRaggedSMEM_>();
        LaunchKernel(batch_size, kTopKBlockSize, device.unwrap(), kRaggedSMEM_)
            .enable_pdl(kUsePDL)(ragged_kernel, params);
    }
};

}  // namespace
