#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

using deepseek_v4::fp8::cast_to_ue8m0;
using deepseek_v4::fp8::inv_scale_ue8m0;
using deepseek_v4::fp8::pack_fp8;

constexpr uint32_t kFusedQBlockSize = 128;
constexpr uint32_t kFusedQNumWarps  = kFusedQBlockSize / device::kWarpThreads;

#define Q_KERNEL __global__ __launch_bounds__(kFusedQBlockSize, 16)

template<int64_t kRopeDim>
SGL_DEVICE device::AlignedVector<float, 4> load_rope_first_cos_sin(const float* __restrict__ cos_sin_cache,
                                                                   int32_t lane_id)
{
    constexpr int64_t               kHalfRopeDim = kRopeDim / 2;
    const int32_t                   pair0        = lane_id * 2;
    const int32_t                   pair1        = pair0 + 1;
    device::AlignedVector<float, 4> freq;
    freq[0] = cos_sin_cache[pair0];
    freq[1] = cos_sin_cache[kHalfRopeDim + pair0];
    freq[2] = cos_sin_cache[pair1];
    freq[3] = cos_sin_cache[kHalfRopeDim + pair1];
    return freq;
}

struct FusedQIndexerRopeHadamardQuantParams {
    const void* __restrict__ q_input;  // (B, num_heads, 128) DType
    void* __restrict__ q_fp8;          // (B, num_heads, 128) fp8_e4m3
    // weights_out[b, h] = weight[b, h] * weight_scale * q_scale[b, h].
    // q_scale is computed internally and not exposed -- the only consumer of
    // it is `weights_out`.
    const void* __restrict__ weight;  // (B, num_heads) DType
    float* __restrict__ weights_out;  // (B, num_heads) fp32 (== (B, H, 1) flat)
    float weight_scale;               // scalar c4_indexer.weight_scale
    // Template-dependent layout:
    //   kRopeFirst=false: (max_pos, 64) fp32 interleaved [cos0, sin0, ...]
    //   kRopeFirst=true : (max_pos, 64) fp32 halves [cos..., sin...]
    const float* __restrict__ rope_cache;
    const void* __restrict__ positions;  // (B,) PosT
    // Row stride for `weight` (caller passes the non-contiguous wk slice directly).
    int64_t  weight_stride_batch;
    uint32_t batch_size;
    uint32_t num_heads;
};

template<typename DType, typename PosT, bool kUsePDL, bool kRopeFirst = false, bool kHadamard = true>
Q_KERNEL void fused_q_indexer_rope_hadamard_quant(const __grid_constant__ FusedQIndexerRopeHadamardQuantParams params)
{
    using namespace device;

    constexpr int64_t  kHeadDim  = 128;
    constexpr int64_t  kRopeDim  = 64;
    constexpr int64_t  kVecSize  = 4;
    constexpr uint32_t kRopeSize = kRopeDim / kVecSize;  // = 16
    static_assert(kHeadDim == kWarpThreads * kVecSize);
    static_assert(kRopeDim == kWarpThreads * 2);
    static_assert(kRopeSize <= kWarpThreads);

    using Storage    = AlignedVector<DType, kVecSize>;
    using Float4     = AlignedVector<float, kVecSize>;
    using OutStorage = AlignedVector<fp8x2_e4m3_t, 2>;  // 4 fp8 / lane

    const auto warp_id = threadIdx.x / kWarpThreads;
    const auto lane_id = threadIdx.x % kWarpThreads;
    const auto work_id = blockIdx.x * kFusedQNumWarps + warp_id;
    // V4 ropes the trailing kRopeDim dims (kRopeFirst=false); V3.2 ropes the
    // leading kRopeDim dims (kRopeFirst=true). Select the owning lanes per layout.
    const bool is_rope_lane = kRopeFirst ? (lane_id < kRopeSize) : (lane_id >= kWarpThreads - kRopeSize);

    const uint32_t total_works = params.batch_size * params.num_heads;
    if (work_id >= total_works)
        return;

    const uint32_t batch_id   = work_id / params.num_heads;
    const auto     input_ptr  = static_cast<const DType*>(params.q_input) + work_id * kHeadDim;
    const auto     position   = static_cast<int32_t>(static_cast<const PosT*>(params.positions)[batch_id]);
    const auto     rope_cache = params.rope_cache + position * kRopeDim;

    // Lane 0 prefetches the weight scalar for this (token, head) work item.
    // Weight is (B, num_heads) DType; we need one scalar per warp -- offload
    // the load to lane 0 only. The multiply + store happens once the q_scale
    // is known (part 4).

    PDLWaitPrimary<kUsePDL>();
    Float4         data, freq;
    const uint32_t head_id = work_id - batch_id * params.num_heads;
    const auto     weight_val =
        cast<float>(static_cast<const DType*>(params.weight)[batch_id * params.weight_stride_batch + head_id]);

    // part 1: load (no norm). Each lane owns a 4-elem pack.
    {
        Storage input_vec;
        input_vec.load(input_ptr, lane_id);
        if (is_rope_lane) {
            if constexpr (kRopeFirst) {
                freq = load_rope_first_cos_sin<kRopeDim>(rope_cache, lane_id);
            }
            else {
                freq.load(rope_cache, lane_id - (kWarpThreads - kRopeSize));
            }
        }
#pragma unroll
        for (int i = 0; i < kVecSize; ++i) {
            data[i] = cast<float>(input_vec[i]);
        }
    }

    // part 2: rope on rope lanes only (4 elems / lane = 2 (real, imag) pairs).
    if (is_rope_lane) {
        const auto x_real = data[0];
        const auto x_imag = data[1];
        const auto y_real = data[2];
        const auto y_imag = data[3];
        const auto fxr    = freq[0];
        const auto fxi    = freq[1];
        const auto fyr    = freq[2];
        const auto fyi    = freq[3];
        data[0]           = x_real * fxr - x_imag * fxi;
        data[1]           = x_real * fxi + x_imag * fxr;
        data[2]           = y_real * fyr - y_imag * fyi;
        data[3]           = y_real * fyi + y_imag * fyr;
    }

    PDLTriggerSecondary<kUsePDL>();

    // DLEngine's V3.2 path converts the leading RoPE block from interleaved
    // [real0, imag0, ...] to NeoX half layout [real..., imag...] before the
    // Hadamard transform. Reproduce that permutation across the warp.
    if constexpr (kRopeFirst) {
        Float4 reordered;
#pragma unroll
        for (int i = 0; i < kVecSize; ++i) {
            const uint32_t dst      = lane_id * kVecSize + i;
            const uint32_t src      = dst < kRopeDim ? dst * 2 : (dst - kRopeDim) * 2 + 1;
            const uint32_t src_lane = src / kVecSize;
            const uint32_t src_item = src % kVecSize;
            const float    v0       = __shfl_sync(0xFFFFFFFFu, data[0], src_lane, kWarpThreads);
            const float    v1       = __shfl_sync(0xFFFFFFFFu, data[1], src_lane, kWarpThreads);
            const float    v2       = __shfl_sync(0xFFFFFFFFu, data[2], src_lane, kWarpThreads);
            const float    v3       = __shfl_sync(0xFFFFFFFFu, data[3], src_lane, kWarpThreads);
            reordered[i]            = src_item == 0 ? v0 : src_item == 1 ? v1 : src_item == 2 ? v2 : v3;
        }
        data = reordered;
    }
    // Match the former kernel boundary: RoPE was materialized as BF16 before
    // fast_hadamard_transform reloaded it.
#pragma unroll
    for (int i = 0; i < kVecSize; ++i)
        data[i] = cast<float>(cast<DType>(data[i]));

    // part 3: 128-point Hadamard (2 local stages + 5 cross-lane shfl_xor stages).
    // Same recipe as `fused_norm_rope_indexer`; see comments there for the
    // butterfly invariants and the early-return safety argument. V3.2 omits the
    // rotation (kHadamard=false): it is logit-preserving (H orthonormal, applied
    // to both q and k), so dropping it only trades fp8 quant accuracy.
    if constexpr (kHadamard) {
        {
            const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
            data[0] = a0 + a1;
            data[1] = a0 - a1;
            data[2] = a2 + a3;
            data[3] = a2 - a3;
        }
        {
            const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
            data[0] = a0 + a2;
            data[1] = a1 + a3;
            data[2] = a0 - a2;
            data[3] = a1 - a3;
        }
#pragma unroll
        for (uint32_t mask = 1; mask < kWarpThreads; mask <<= 1) {
#pragma unroll
            for (int i = 0; i < kVecSize; ++i) {
                const float other = __shfl_xor_sync(0xFFFFFFFFu, data[i], mask, kWarpThreads);
                data[i]           = (lane_id & mask) ? (other - data[i]) : (data[i] + other);
            }
        }
        const float kHadamardScale = math::rsqrt(static_cast<float>(kHeadDim));
#pragma unroll
        for (int i = 0; i < kVecSize; ++i)
            data[i] *= kHadamardScale;
    }

    // Match the second former boundary: Hadamard output was BF16 before quant.
#pragma unroll
    for (int i = 0; i < kVecSize; ++i)
        data[i] = cast<float>(cast<DType>(data[i]));

    {
        float local_max = math::abs(data[0]);
#pragma unroll
        for (int i = 1; i < kVecSize; ++i) {
            local_max = math::max(local_max, math::abs(data[i]));
        }
        const auto abs_max     = warp::reduce_max(local_max);
        const auto scale_raw   = fmaxf(1e-4f, abs_max) / math::FP8_E4M3_MAX;
        const auto scale_ue8m0 = cast_to_ue8m0(scale_raw);
        const auto scale       = __uint_as_float(scale_ue8m0 << 23);
        const auto inv_scale   = inv_scale_ue8m0(scale_ue8m0);
        OutStorage result;
        result[0] = pack_fp8(data[0] * inv_scale, data[1] * inv_scale);
        result[1] = pack_fp8(data[2] * inv_scale, data[3] * inv_scale);

        // q_fp8 row pointer: 128 fp8 / row = 32 OutStorage / row, one per lane.
        auto out_row = static_cast<uint8_t*>(params.q_fp8) + work_id * kHeadDim;
        result.store(out_row, lane_id);
        params.weights_out[work_id] = weight_val * params.weight_scale * scale;
    }
}

template<typename DType, bool kUsePDL, bool kRopeFirst = false, bool kHadamard = true>
struct FusedQIndexerRopeHadamardQuantKernel {
    template<typename PosT>
    static constexpr auto kernel = fused_q_indexer_rope_hadamard_quant<DType, PosT, kUsePDL, kRopeFirst, kHadamard>;

    static void forward(const tvm::ffi::TensorView q_input,
                        const tvm::ffi::TensorView q_fp8,
                        const tvm::ffi::TensorView weight,
                        const tvm::ffi::TensorView weights_out,
                        double                     weight_scale,
                        const tvm::ffi::TensorView rope_cache,
                        const tvm::ffi::TensorView positions)
    {
        using namespace host;
        constexpr int64_t kHeadDim = 128;
        constexpr int64_t kRopeDim = 64;

        auto B       = SymbolicSize{"batch_size"};
        auto H       = SymbolicSize{"num_heads"};
        auto device_ = SymbolicDevice{};
        device_.set_options<kDLCUDA>();

        // Caller path is `wq_b(q_lora).view(-1, H, D)` -> contiguous; the kernel
        // assumes a flat `(B*H, kHeadDim)` layout for both q_input and q_fp8.
        // Pin the head/innermost strides; assert the batch stride below.
        TensorMatcher({B, H, kHeadDim})  //
            .with_strides({-1, kHeadDim, 1})
            .with_dtype<DType>()
            .with_device(device_)
            .verify(q_input);
        TensorMatcher({B, H, kHeadDim})  //
            .with_strides({-1, kHeadDim, 1})
            .with_dtype<fp8_e4m3_t>()
            .with_device(device_)
            .verify(q_fp8);
        TensorMatcher({B, H})  //
            .with_strides({-1, 1})
            .with_dtype<DType>()
            .with_device(device_)
            .verify(weight);
        TensorMatcher({B, H, 1})  //
            .with_dtype<float>()
            .with_device(device_)
            .verify(weights_out);
        TensorMatcher({-1, kRopeDim})  //
            .with_dtype<float>()
            .with_device(device_)
            .verify(rope_cache);
        auto pos_dtype = SymbolicDType{};
        TensorMatcher({B})  //
            .with_dtype<int32_t, int64_t>(pos_dtype)
            .with_device(device_)
            .verify(positions);

        const auto batch_size = static_cast<uint32_t>(B.unwrap());
        const auto num_heads  = static_cast<uint32_t>(H.unwrap());
        if (batch_size == 0)
            return;

        // The kernel computes row pointers as `base + work_id * kHeadDim`, so
        // both inputs must be contiguous in (batch, head, elem) order.
        const int64_t expected_batch_stride = static_cast<int64_t>(num_heads) * kHeadDim;
        RuntimeCheck(q_input.stride(0) == expected_batch_stride,
                     "q_input must be contiguous (B, H, kHeadDim); got stride[0]=",
                     q_input.stride(0));
        RuntimeCheck(q_fp8.stride(0) == expected_batch_stride,
                     "q_fp8 must be contiguous (B, H, kHeadDim); got stride[0]=",
                     q_fp8.stride(0));

        const auto params = FusedQIndexerRopeHadamardQuantParams{
            .q_input             = q_input.data_ptr(),
            .q_fp8               = q_fp8.data_ptr(),
            .weight              = weight.data_ptr(),
            .weights_out         = static_cast<float*>(weights_out.data_ptr()),
            .weight_scale        = static_cast<float>(weight_scale),
            .rope_cache          = static_cast<const float*>(rope_cache.data_ptr()),
            .positions           = positions.data_ptr(),
            .weight_stride_batch = weight.stride(0),
            .batch_size          = batch_size,
            .num_heads           = num_heads,
        };
        const auto total_works = batch_size * num_heads;
        const auto num_blocks  = div_ceil(total_works, kFusedQNumWarps);
        const auto k_int32     = kernel<int32_t>;
        const auto k_int64     = kernel<int64_t>;
        const auto k           = pos_dtype.is_type<int32_t>() ? k_int32 : k_int64;
        LaunchKernel(num_blocks, kFusedQBlockSize, device_.unwrap())  //
            .enable_pdl(kUsePDL)(k, params);
    }
};

}  // namespace
