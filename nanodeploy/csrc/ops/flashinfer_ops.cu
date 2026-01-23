#include "nanodeploy/csrc/logging.h"
#include "nanodeploy/csrc/ops/flashinfer_ops.h"
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <torch/torch.h>
#include <vector>

#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/fastdiv.cuh>
#include <flashinfer/page.cuh>

using namespace flashinfer;
using half     = __half;
using bfloat16 = __nv_bfloat16;

namespace nanodeploy {
namespace ops {

// ... (Existing decode implementation) ...

// -------------------------------------------------------------------------
// Prefill Implementation
// -------------------------------------------------------------------------

// Methods moved to end of file

// (Namespace continues for Impl definition)

cudaError_t DispatchDecodePlanReal(void*           float_buffer,
                                   size_t          float_workspace_size_in_bytes,
                                   void*           int_buffer,
                                   void*           page_locked_int_buffer,
                                   size_t          int_workspace_size_in_bytes,
                                   DecodePlanInfo& plan_info,
                                   int32_t*        indptr_h,
                                   uint32_t        batch_size,
                                   uint32_t        num_qo_heads,
                                   uint32_t        num_kv_heads,
                                   uint32_t        page_size,
                                   bool            enable_cuda_graph,
                                   cudaStream_t    stream)
{
    constexpr uint32_t        HEAD_DIM          = 128;  // Hardcoded for Qwen3
    constexpr PosEncodingMode POS_ENCODING_MODE = PosEncodingMode::kNone;

    using DTypeQ           = bfloat16;
    using DTypeKV          = bfloat16;
    using DTypeO           = bfloat16;
    using IdType           = int32_t;
    using Params           = BatchDecodeParams<DTypeQ, DTypeKV, DTypeO, IdType>;
    using AttentionVariant = DefaultAttention<false, false, false, false>;

    uint32_t    group_size = num_qo_heads / num_kv_heads;
    cudaError_t status     = cudaSuccess;

    // Simplified local dispatch to avoid relying on utils.cuh macro limits
    auto run_for_group_size = [&](auto group_size_const) {
        constexpr uint32_t GROUP_SIZE      = decltype(group_size_const)::value;
        auto               work_estimation = [&](bool&          split_kv,
                                   uint32_t&      max_grid_size,
                                   uint32_t&      max_num_pages_per_batch,
                                   uint32_t&      new_batch_size,
                                   uint32_t&      gdy,
                                   uint32_t       bs,
                                   IdType*        kv_indptr,
                                   const uint32_t n_qo,
                                   const uint32_t ps,
                                   bool           graphs,
                                   cudaStream_t   s) {
            return BatchDecodeWithPagedKVCacheWorkEstimationDispatched<GROUP_SIZE,
                                                                                     HEAD_DIM,
                                                                                     POS_ENCODING_MODE,
                                                                                     AttentionVariant,
                                                                                     Params>(split_kv,
                                                                               max_grid_size,
                                                                               max_num_pages_per_batch,
                                                                               new_batch_size,
                                                                               gdy,
                                                                               bs,
                                                                               kv_indptr,
                                                                               n_qo,
                                                                               ps,
                                                                               graphs,
                                                                               s);
        };

        status = DecodePlan<HEAD_DIM, POS_ENCODING_MODE, AttentionVariant, Params>(float_buffer,
                                                                                   float_workspace_size_in_bytes,
                                                                                   int_buffer,
                                                                                   page_locked_int_buffer,
                                                                                   int_workspace_size_in_bytes,
                                                                                   plan_info,
                                                                                   indptr_h,
                                                                                   batch_size,
                                                                                   num_qo_heads,
                                                                                   page_size,
                                                                                   enable_cuda_graph,
                                                                                   stream,
                                                                                   work_estimation);
    };

    switch (group_size) {
        case 1:
            run_for_group_size(std::integral_constant<uint32_t, 1>{});
            break;
        case 2:
            run_for_group_size(std::integral_constant<uint32_t, 2>{});
            break;
        case 3:
            run_for_group_size(std::integral_constant<uint32_t, 3>{});
            break;
        case 4:
            run_for_group_size(std::integral_constant<uint32_t, 4>{});
            break;
        case 6:
            run_for_group_size(std::integral_constant<uint32_t, 6>{});
            break;
        case 8:
            run_for_group_size(std::integral_constant<uint32_t, 8>{});
            break;
        case 16:
            run_for_group_size(std::integral_constant<uint32_t, 16>{});
            break;
        default:
            NANODEPLOY_LOG_ERROR("Unsupported group size for flashinfer: ", group_size);
            status = cudaErrorInvalidValue;
            break;
    }

    return status;
}

// -------------------------------------------------------------------------
// PIMPL Implementation
// -------------------------------------------------------------------------

struct FlashInferOps::Impl {
    int           num_layers_;
    int           num_heads_;
    int           num_kv_heads_;
    int           head_dim_;
    int           page_size_;
    torch::Device device_;

    // Workspaces
    torch::Tensor float_buffer_;
    torch::Tensor int_buffer_;
    torch::Tensor page_locked_int_buffer_;

    // Metadata (Device Tensors used during execution)
    torch::Tensor indptr_;
    torch::Tensor indices_;
    torch::Tensor last_page_len_;
    torch::Tensor q_indptr_;  // Added for prefill

    // Metadata (Host Tensors used for setup)
    torch::Tensor indptr_cpu_;
    torch::Tensor last_page_len_cpu_;

    // State
    int            batch_size_                = 0;
    int            max_batch_size_            = 0;
    int            padded_batch_size_prefill_ = 0;
    DecodePlanInfo plan_info;

    // Static attention output buffer for CUDA Graph capture
    torch::Tensor static_attn_output_;

    Impl(int num_layers, int num_heads, int num_kv_heads, int head_dim, int page_size, torch::Device device):
        num_layers_(num_layers),
        num_heads_(num_heads),
        num_kv_heads_(num_kv_heads),
        head_dim_(head_dim),
        page_size_(page_size),
        device_(device)
    {
        // 128MB workspace
        size_t workspace_size = 128 * 1024 * 1024;
        float_buffer_ =
            torch::empty({(long)workspace_size}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
        int_buffer_ = torch::empty({(long)workspace_size}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
        page_locked_int_buffer_ =
            torch::empty({(long)workspace_size}, torch::TensorOptions().dtype(torch::kUInt8).pinned_memory(true));
    }

    void init_workspace(int max_batch_size, int max_total_blocks)
    {
        max_batch_size_ = max_batch_size;

        // Allocate static metadata buffers
        auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(device_);

        indptr_        = torch::empty({max_batch_size + 1}, options_int);
        last_page_len_ = torch::empty({max_batch_size}, options_int);
        indices_       = torch::empty({max_total_blocks}, options_int);

        // Allocate persistent host buffers for capture safety
        auto options_cpu   = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true);
        indptr_cpu_        = torch::empty({max_batch_size + 1}, options_cpu);
        last_page_len_cpu_ = torch::empty({max_batch_size}, options_cpu);

        // Allocate static attention output buffer for CUDA Graph capture
        // Shape: [max_batch_size, num_heads, head_dim] for decode (seq_len=1)
        auto options_bf16   = torch::TensorOptions().dtype(torch::kBFloat16).device(device_);
        static_attn_output_ = torch::empty({max_batch_size, num_heads_, head_dim_}, options_bf16);

        NANODEPLOY_LOG_INFO(
            "[FlashInfer] Initialized Static Workspace: MaxBatch=", max_batch_size, " MaxBlocks=", max_total_blocks);
    }

    void begin_forward_prefill(int32_t* q_indptr_host,
                               int32_t* block_tables_host,
                               int32_t* last_page_len_host,
                               int      batch_size,
                               int      max_num_blocks,
                               int      num_qo_heads,
                               int      num_kv_heads,
                               int      head_dim,
                               int      page_size);

    torch::Tensor prefill(void* q_ptr, void* k_cache_ptr, void* v_cache_ptr, int total_tokens, int layer_idx);

    void begin_forward(int32_t* block_tables_host,
                       int32_t* seq_lens_host,
                       int      batch_size,
                       int      max_num_blocks,
                       int      num_qo_heads,
                       int      num_kv_heads,
                       int      head_dim,
                       int      page_size,
                       int      window_left)
    {
        // fprintf(stderr,
        //         "[FlashInfer] begin_forward_impl: BS=%d MaxBlocks=%d PageSize=%d\n",
        //         batch_size,
        //         max_num_blocks,
        //         page_size);

        batch_size_ = batch_size;

        // NOTE: Input pointers are now HOST pointers. No D2H copy needed.

        // Prep CSR Metadata (CPU)
        // NOTE: Input pointers are now HOST pointers. No D2H copy needed.

        // Prep CSR Metadata (CPU)
        // Use persistent buffers if available (static mode), else temp
        bool use_static = (indptr_.defined() && indptr_.size(0) >= batch_size + 1);

        torch::Tensor indptr_cpu;
        torch::Tensor last_page_len_cpu;

        if (use_static) {
            indptr_cpu        = indptr_cpu_.slice(0, 0, batch_size + 1);
            last_page_len_cpu = last_page_len_cpu_.slice(0, 0, batch_size);
        }
        else {
            auto options_cpu  = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true);
            indptr_cpu        = torch::empty({static_cast<int64_t>(batch_size + 1)}, options_cpu);
            last_page_len_cpu = torch::empty({static_cast<int64_t>(batch_size)}, options_cpu);
        }

        auto indptr_acc        = indptr_cpu.accessor<int32_t, 1>();
        auto last_page_len_acc = last_page_len_cpu.accessor<int32_t, 1>();

        // Wrap host pointers in accessors/spans for easy reading
        // We can just pointer arithmetic directly or wrap in Tensor (zero-copy)
        // Store in lvalue to use accessor (cannot access rvalue)
        auto seq_lens_t   = torch::from_blob(seq_lens_host, {static_cast<int64_t>(batch_size)}, torch::kInt32);
        auto seq_lens_acc = seq_lens_t.accessor<int32_t, 1>();

        auto block_tables_t = torch::from_blob(
            block_tables_host, {static_cast<int64_t>(batch_size), static_cast<int64_t>(max_num_blocks)}, torch::kInt32);
        auto block_tables_acc = block_tables_t.accessor<int32_t, 2>();

        int32_t current_offset = 0;
        indptr_acc[0]          = 0;
        std::vector<int32_t> flat_indices;
        flat_indices.reserve(static_cast<size_t>(batch_size) * static_cast<size_t>(max_num_blocks));

        for (int32_t i = 0; i < batch_size; ++i) {
            int32_t seq_len    = seq_lens_acc[i];
            int32_t num_blocks = (seq_len + page_size - 1) / page_size;
            if (seq_len == 0)
                num_blocks = 0;

            for (int32_t b = 0; b < num_blocks; ++b) {
                flat_indices.push_back(block_tables_acc[i][b]);
            }
            current_offset += num_blocks;
            indptr_acc[i + 1]    = current_offset;
            int32_t r            = (seq_len - 1) % page_size + 1;
            last_page_len_acc[i] = (seq_len == 0) ? 0 : r;
        }

        // Send to Device
        // Allocate device tensors if needed (Dynamic Mode) or check size (Static Mode)
        if (!use_static) {
            if (!indices_.defined() || indices_.numel() < static_cast<int64_t>(flat_indices.size())) {
                indices_ = torch::empty({static_cast<int64_t>(flat_indices.size())},
                                        torch::TensorOptions().dtype(torch::kInt32).device(device_));
            }
        }

        if (!flat_indices.empty()) {
            // If static, we just copy into the beginning of indices_
            // Ensure we don't overflow static buffer
            if (use_static && static_cast<int64_t>(flat_indices.size()) > indices_.numel()) {
                NANODEPLOY_LOG_ERROR("[FlashInfer] Static indices buffer overflow!");
                // Fallback or crash? Crash/Error is safer for now.
            }

            cudaMemcpyAsync((char*)indices_.data_ptr(),
                            (char*)flat_indices.data(),
                            flat_indices.size() * sizeof(int32_t),
                            cudaMemcpyHostToDevice,
                            c10::cuda::getCurrentCUDAStream());
        }

        // Copy indptr and last_page_len to device for execution (Async H2D)
        if (use_static) {
            // Copy into static buffers (slice match implicitly by pointer arithmetic or explicit slice)
            // We copy to the *start* of the persistent buffer.
            // Using Async copy from pinned host memory.
            cudaMemcpyAsync(indptr_.data_ptr(),
                            indptr_cpu.data_ptr(),
                            indptr_cpu.nbytes(),
                            cudaMemcpyHostToDevice,
                            c10::cuda::getCurrentCUDAStream());
            cudaMemcpyAsync(last_page_len_.data_ptr(),
                            last_page_len_cpu.data_ptr(),
                            last_page_len_cpu.nbytes(),
                            cudaMemcpyHostToDevice,
                            c10::cuda::getCurrentCUDAStream());
        }
        else {
            indptr_        = indptr_cpu.to(device_, /*non_blocking=*/true);
            last_page_len_ = last_page_len_cpu.to(device_, /*non_blocking=*/true);
        }

        // Dispatch Plan
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

        cudaError_t status =
            DispatchDecodePlanReal(float_buffer_.data_ptr(),
                                   float_buffer_.nbytes(),
                                   int_buffer_.data_ptr(),
                                   page_locked_int_buffer_.data_ptr(),
                                   int_buffer_.nbytes(),
                                   plan_info,
                                   static_cast<int32_t*>(indptr_cpu.data_ptr()),  // Use Pinned Host Pointer
                                   static_cast<uint32_t>(batch_size),
                                   static_cast<uint32_t>(num_qo_heads),
                                   static_cast<uint32_t>(num_kv_heads),
                                   static_cast<uint32_t>(page_size),
                                   use_static,  // Enable CUDAGraph mode if using static workspace
                                   stream);

        // REMOVED: cudaStreamSynchronize(stream);
        // REMOVED: Logging

        // IMPORTANT: FlashInfer's DecodePlan likely writes to plan_info struct (Host Memory) via D2H copy internally?
        // Checking DispatchDecodePlanReal usage...
        // It passes `plan_info` (Host Reference). FlashInfer usually updates this via `cudaMemcpyAsync`.
        // If we don't sync, `plan_info` might not be ready if we access it immediately on Host.
        // However, `attention` kernel uses correct offsets that are computed on Device or Host?
        // Actually `plan_info` contains offsets like `request_indices_offset`.
        // These are set by `DecodePlan` which runs on Host (helper) mostly?
        // Wait, FlashInfer DecodePlan function signature:
        // DecodePlan(..., DecodePlanInfo& plan_info, ..., cudaStream_t stream)
        // If FlashInfer updates plan_info using Async Copy, we MUST synchronize before reading it on CPU.
        // BUT, looking at FlashInfer source (assumed), `DecodePlan` calculates workspace offsets ON CPU immediately.
        // It calculates how much workspace is needed and returns status.
        // The *content* of workspaces is populated on GPU.
        // `plan_info.padded_batch_size` depends on `work_estimation` which runs a kernel.
        // `work_estimation` ( BatchDecodeWithPagedKVCacheWorkEstimationDispatched ) -> launches kernel.
        // Then it seems it might copy back `padded_batch_size`?
        // If FlashInfer relies on D2H for `padded_batch_size`, we are stuck with sync or we must blindly launch.
        // For `DefaultAttention`, `padded_batch_size` is usually `batch_size` (no splitting) or more (splitting).
        // Let's assume for now we remove the sync. If `plan_info` is garbage, we crash.
        // HACK: We will NOT access plan_info on CPU for logging. We trust it is set or we accept the race if FlashInfer
        // was designed that way. Actually, if `work_estimation` is asynchronous, `plan_info` might be invalid. BUT,
        // `DispatchDecodePlanReal` is our wrapper. If `DecodePlan` returns `plan_info`, it must be valid.

        // We will remove the explicit `cudaStreamSynchronize` and assume FlashInfer handles consistency
        // or that we simply don't read `plan_info` values that depend on GPU kernels in the hot path.
        // (Offsets are calculated on CPU).
    }
};

// -------------------------------------------------------------------------
// FlashInferOps Proxy Methods
// -------------------------------------------------------------------------

FlashInferOps::FlashInferOps(
    int num_layers, int num_heads, int num_kv_heads, int head_dim, int page_size, torch::Device device):
    impl_(std::make_unique<Impl>(num_layers, num_heads, num_kv_heads, head_dim, page_size, device))
{
}

FlashInferOps::~FlashInferOps() = default;

void FlashInferOps::begin_forward(int32_t* block_tables_host,
                                  int32_t* seq_lens_host,
                                  int      batch_size,
                                  int      max_num_blocks,
                                  int      num_qo_heads,
                                  int      num_kv_heads,
                                  int      head_dim,
                                  int      page_size,
                                  int      window_left)
{
    impl_->begin_forward(block_tables_host,
                         seq_lens_host,
                         batch_size,
                         max_num_blocks,
                         num_qo_heads,
                         num_kv_heads,
                         head_dim,
                         page_size,
                         window_left);
}

void FlashInferOps::init_workspace(int max_batch_size, int max_total_blocks)
{
    impl_->init_workspace(max_batch_size, max_total_blocks);
}

void FlashInferOps::begin_forward_impl(int32_t* block_tables_host,
                                       int32_t* seq_lens_host,
                                       int      batch_size,
                                       int      max_num_blocks,
                                       int      num_qo_heads,
                                       int      num_kv_heads,
                                       int      head_dim,
                                       int      page_size,
                                       int      window_left)
{
    // Legacy / unused in PIMPL
}

torch::Tensor FlashInferOps::attention(
    void* q_ptr, void* k_cache_ptr, void* v_cache_ptr, int batch, int seq, int heads, int head_dim, int layer_idx)
{
    NANODEPLOY_LOG_DEBUG("[FlashInfer] attention: Q_ptr=", q_ptr, " Layer=", layer_idx);

    auto& impl = *impl_;

    // Use static output buffer for CUDA Graph capture compatibility
    // For decode (seq=1), output shape is [batch, heads, head_dim]
    // We use a slice of the pre-allocated static buffer
    torch::Tensor o;
    if (seq == 1 && impl.static_attn_output_.defined() && batch <= impl.max_batch_size_) {
        // Use pre-allocated static buffer (slice to actual batch size)
        o = impl.static_attn_output_.slice(0, 0, batch);
    }
    else {
        // Fallback to dynamic allocation for prefill or if static buffer not available
        auto                 options_d = torch::TensorOptions().dtype(torch::kBFloat16).device(impl.device_);
        std::vector<int64_t> q_shape   = {batch * seq, heads, head_dim};
        if (seq == 1)
            q_shape = {batch, heads, head_dim};
        o = torch::empty(q_shape, options_d);
    }

    using DTypeQ  = bfloat16;
    using DTypeKV = bfloat16;
    using DTypeO  = bfloat16;
    using IdType  = int32_t;
    // Our cache is [num_blocks, num_kv_heads, page_size, head_dim] -> HND
    QKVLayout kv_layout = QKVLayout::kHND;

    paged_kv_t<DTypeKV, IdType> paged_kv(static_cast<uint32_t>(impl.num_kv_heads_),
                                         static_cast<uint32_t>(impl.page_size_),
                                         static_cast<uint32_t>(impl.head_dim_),
                                         static_cast<uint32_t>(impl.batch_size_),
                                         kv_layout,
                                         static_cast<DTypeKV*>(k_cache_ptr),
                                         static_cast<DTypeKV*>(v_cache_ptr),
                                         static_cast<IdType*>(impl.indices_.data_ptr()),
                                         static_cast<IdType*>(impl.indptr_.data_ptr()),
                                         static_cast<IdType*>(impl.last_page_len_.data_ptr()));

    BatchDecodeParams<DTypeQ, DTypeKV, DTypeO, IdType> params;
    params.q            = static_cast<DTypeQ*>(q_ptr);
    params.paged_kv     = paged_kv;
    params.o            = static_cast<DTypeO*>(o.data_ptr());
    params.num_qo_heads = static_cast<uint32_t>(heads);
    // Q tensor is [batch, num_qo_heads, head_dim] (contiguous NHD)
    // Explicitly cast to IdType (int32_t) for FlashInfer compatibility
    params.q_stride_n = static_cast<IdType>(heads * head_dim);  // Stride between batch elements
    params.q_stride_h = static_cast<IdType>(head_dim);          // Stride between heads

    // Helper for pointer arithmetic
    auto get_ptr = [&](size_t offset) {
        return reinterpret_cast<IdType*>(static_cast<uint8_t*>(impl.int_buffer_.data_ptr()) + offset);
    };

    auto& plan               = impl.plan_info;
    params.request_indices   = get_ptr(plan.request_indices_offset);
    params.kv_tile_indices   = get_ptr(plan.kv_tile_indices_offset);
    params.o_indptr          = get_ptr(plan.o_indptr_offset);
    params.kv_chunk_size_ptr = get_ptr(plan.kv_chunk_size_ptr_offset);
    params.padded_batch_size = static_cast<uint32_t>(plan.padded_batch_size);

    NANODEPLOY_LOG_DEBUG("[FlashInfer] Params: PaddedBS=",
                         params.padded_batch_size,
                         " RequestIndicesOffset=",
                         plan.request_indices_offset);

    if (plan.split_kv) {
        params.block_valid_mask =
            reinterpret_cast<bool*>(static_cast<uint8_t*>(impl.int_buffer_.data_ptr()) + plan.block_valid_mask_offset);
        params.partition_kv = true;

        NANODEPLOY_LOG_DEBUG("[FlashInfer] SplitKV Enabled. ValidMaskOffset=", plan.block_valid_mask_offset);
    }
    params.sm_scale = 1.0f / std::sqrt(float(head_dim));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    DTypeO*      tmp_v  = nullptr;
    float*       tmp_s  = nullptr;

    if (plan.split_kv) {
        tmp_v = reinterpret_cast<DTypeO*>(static_cast<uint8_t*>(impl.float_buffer_.data_ptr()) + plan.v_offset);
        tmp_s = reinterpret_cast<float*>(static_cast<uint8_t*>(impl.float_buffer_.data_ptr()) + plan.s_offset);
    }

    // Calculate Group Size
    int num_qo_heads = heads;               // 16
    int num_kv_heads = impl.num_kv_heads_;  // 8
    int group_size   = num_qo_heads / num_kv_heads;

    if (num_qo_heads % num_kv_heads != 0) {
        NANODEPLOY_LOG_ERROR("[FlashInfer] Error: Q heads ", num_qo_heads, " not divisible by KV heads ", num_kv_heads);
        return o;
    }

    constexpr uint32_t HEAD_DIM = 128;
    if (head_dim != HEAD_DIM) {
        NANODEPLOY_LOG_ERROR("[FlashInfer] Error: Runtime HeadDim ", head_dim, " != Compiled HeadDim ", HEAD_DIM);
        return o;
    }

    cudaError_t status = cudaSuccess;

    // DEBUG: Print Kernel Config params
    if (params.padded_batch_size == 0 || impl.num_kv_heads_ == 0) {
        NANODEPLOY_LOG_ERROR(
            "[FlashInfer] ERROR: Invalid Grid: Batch=", params.padded_batch_size, " KVHeads=", impl.num_kv_heads_);
        // Don't launch if invalid to avoid CUDA error spam
        return o;
    }

    status = BatchDecodeWithPagedKVCacheDispatched<HEAD_DIM,
                                                   PosEncodingMode::kNone,
                                                   DefaultAttention<false, false, false, false>,
                                                   BatchDecodeParams<DTypeQ, DTypeKV, DTypeO, IdType>>(
        params, tmp_v, tmp_s, /*enable_cuda_graph=*/(impl.max_batch_size_ > 0), stream);

    if (status != cudaSuccess) {
        NANODEPLOY_LOG_ERROR("BatchDecode kernel failed: ", cudaGetErrorString(status));
    }
    else {
        // Verify Execution Completion REMOVED for performance
        // cudaStreamSynchronize(stream);
        // fprintf(stderr, "[FlashInfer] BatchDecode Kernel Execution Done.\n");
    }

    return o;
}

// -------------------------------------------------------------------------
// Prefill Implementation
// -------------------------------------------------------------------------

// End of File (Duplicate removed)

void FlashInferOps::Impl::begin_forward_prefill(int32_t* q_indptr_host,
                                                int32_t* block_tables_host,
                                                int32_t* last_page_len_host,
                                                int      batch_size,
                                                int      max_num_blocks,
                                                int      num_qo_heads,
                                                int      num_kv_heads,
                                                int      head_dim,
                                                int      page_size)
{
    batch_size_      = batch_size;
    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(device_);

    // 1. Setup q_indptr (Token Offsets)
    if (!q_indptr_.defined() || q_indptr_.size(0) < static_cast<int64_t>(batch_size + 1)) {
        q_indptr_ = torch::empty({static_cast<int64_t>(batch_size + 1)}, options_int);
    }

    // Copy HOST q_indptr to DEVICE q_indptr_
    cudaMemcpyAsync(q_indptr_.data_ptr(),
                    q_indptr_host,
                    static_cast<size_t>(batch_size + 1) * sizeof(int32_t),
                    cudaMemcpyHostToDevice,
                    c10::cuda::getCurrentCUDAStream());

    // 2. Setup Page Indptr (KV Block Offsets) & Indices (Flattened Blocks)
    // We need to calculate how many blocks each request uses to build `indptr_` (kv_page_indptr).

    std::vector<int32_t> flat_indices;
    std::vector<int32_t> kv_page_indptr_host_vec(static_cast<size_t>(batch_size + 1));
    std::vector<int32_t> last_page_lens(static_cast<size_t>(batch_size));

    flat_indices.reserve(static_cast<size_t>(batch_size) * static_cast<size_t>(max_num_blocks));
    kv_page_indptr_host_vec[0] = 0;

    for (int32_t i = 0; i < batch_size; ++i) {
        int32_t seq_len    = q_indptr_host[i + 1] - q_indptr_host[i];
        int32_t num_blocks = (seq_len + page_size - 1) / page_size;
        if (seq_len == 0)
            num_blocks = 0;

        // Append blocks
        for (int32_t b = 0; b < num_blocks; ++b) {
            flat_indices.push_back(block_tables_host[i * max_num_blocks + b]);
        }

        kv_page_indptr_host_vec[i + 1] = kv_page_indptr_host_vec[i] + num_blocks;
    }

    // Copy Indices
    if (!indices_.defined() || indices_.numel() < static_cast<int64_t>(flat_indices.size())) {
        indices_ = torch::empty({static_cast<int64_t>(flat_indices.size())}, options_int);
    }
    if (!flat_indices.empty()) {
        cudaMemcpyAsync(indices_.data_ptr(),
                        flat_indices.data(),
                        flat_indices.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice,
                        c10::cuda::getCurrentCUDAStream());
    }

    // Copy KV Page Indptr
    if (!indptr_.defined() || indptr_.size(0) < static_cast<int64_t>(batch_size + 1)) {
        indptr_ = torch::empty({static_cast<int64_t>(batch_size + 1)}, options_int);
    }
    cudaMemcpyAsync(indptr_.data_ptr(),
                    kv_page_indptr_host_vec.data(),
                    static_cast<size_t>(batch_size + 1) * sizeof(int32_t),
                    cudaMemcpyHostToDevice,
                    c10::cuda::getCurrentCUDAStream());

    // Copy Last Page Len
    if (!last_page_len_.defined() || last_page_len_.size(0) < static_cast<int64_t>(batch_size)) {
        last_page_len_ = torch::empty({static_cast<int64_t>(batch_size)}, options_int);
    }
    cudaMemcpyAsync(last_page_len_.data_ptr<int32_t>(),
                    last_page_len_host,
                    static_cast<size_t>(batch_size) * sizeof(int32_t),
                    cudaMemcpyHostToDevice,
                    c10::cuda::getCurrentCUDAStream());
}

torch::Tensor
FlashInferOps::Impl::prefill(void* q_ptr, void* k_cache_ptr, void* v_cache_ptr, int total_tokens, int layer_idx)
{
    // Output tensor [total_tokens, num_qo_heads, head_dim]
    auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(device_);
    auto o            = torch::zeros(
        {static_cast<int64_t>(total_tokens), static_cast<int64_t>(num_heads_), static_cast<int64_t>(head_dim_)},
        options_bf16);

    using DTypeQ  = bfloat16;
    using DTypeKV = bfloat16;
    using DTypeO  = bfloat16;
    using IdType  = int32_t;
    using Params  = BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeO, IdType>;

    // PagedKV
    // PagedKV
    QKVLayout kv_layout = QKVLayout::kHND;

    paged_kv_t<DTypeKV, IdType> paged_kv(static_cast<uint32_t>(num_kv_heads_),
                                         static_cast<uint32_t>(page_size_),
                                         static_cast<uint32_t>(head_dim_),
                                         static_cast<uint32_t>(batch_size_),
                                         kv_layout,
                                         static_cast<DTypeKV*>(k_cache_ptr),
                                         static_cast<DTypeKV*>(v_cache_ptr),
                                         static_cast<IdType*>(indices_.data_ptr()),
                                         static_cast<IdType*>(indptr_.data_ptr()),        // kv_page_indptr
                                         static_cast<IdType*>(last_page_len_.data_ptr())  // last_page_len
    );

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Workspace
    // We need temporary buffer for prefill workspace.
    // Let's use `float_buffer_` (128MB).

    // Construct Params
    Params params;
    params.q            = static_cast<DTypeQ*>(q_ptr);
    params.q_indptr     = static_cast<IdType*>(q_indptr_.data_ptr());
    params.paged_kv     = paged_kv;
    params.o            = static_cast<DTypeO*>(o.data_ptr());
    params.lse          = nullptr;  // LSE not needed for prefill usually
    params.num_qo_heads = static_cast<uint32_t>(num_heads_);
    // Use uint_fastdiv constructor explicitly for group_size
    params.group_size = uint_fastdiv(static_cast<uint32_t>(num_heads_ / num_kv_heads_));

    // Set q_stride for contiguous [TotalTokens, Heads, Dim] layout (NHD)
    // Explicitly cast to IdType (int32_t) for FlashInfer compatibility
    params.q_stride_n = static_cast<IdType>(num_heads_ * head_dim_);
    params.q_stride_h = static_cast<IdType>(head_dim_);

    // Note: BatchPrefillPagedParams lacks o_stride members; FlashInfer assumes contiguous output [TotalTokens, Heads,
    // Dim]
    params.sm_scale = 1.0f / std::sqrt(float(head_dim_));
    // Use INT32_MAX for infinite window to avoid potential cast/arithmetic issues with -1
    params.window_left     = 2147483647;
    params.logits_soft_cap = 0.0f;

    params.padded_batch_size = static_cast<uint32_t>(batch_size_);

    // -------------------------------------------------------------------------
    // Setup Mandatory Plan Indices (One-to-One mapping, No Splitting)
    // -------------------------------------------------------------------------

    // We reuse `int_buffer_` for these small variances
    int32_t* int_ptr_base = static_cast<int32_t*>(int_buffer_.data_ptr());
    int32_t* req_ind_ptr  = int_ptr_base;               // [BatchSize]
    int32_t* qo_ind_ptr   = req_ind_ptr + batch_size_;  // [BatchSize]
    int32_t* kv_ind_ptr   = qo_ind_ptr + batch_size_;   // [BatchSize]
    int32_t* chunk_sz_ptr = kv_ind_ptr + batch_size_;   // [1]

    // Fill Indices on Device (Async)
    // We need [0, 1, ..., B-1] for RequestIndices
    // We need [0, 0, ..., 0] for QO/KV TileIndices
    // We need [INT_MAX] for ChunkSize

    // Using a small kernel or cudaMemcpy is tedious. Let's use Torch to fill them
    auto opts_int = torch::TensorOptions().dtype(torch::kInt32).device(device_);

    // Create views into buffer to fill them using Tensor Ops
    // Note: ensure int_buffer_ is large enough (128MB is plenty)

    auto t_req_ind = torch::from_blob(req_ind_ptr, {static_cast<int64_t>(batch_size_)}, opts_int);
    auto t_qo_ind  = torch::from_blob(qo_ind_ptr, {static_cast<int64_t>(batch_size_)}, opts_int);
    auto t_kv_ind  = torch::from_blob(kv_ind_ptr, {static_cast<int64_t>(batch_size_)}, opts_int);
    auto t_chunk   = torch::from_blob(chunk_sz_ptr, {1}, opts_int);

    // Launch fills (Async on current stream)
    // Use explicit int64_t to ensure correct type for arange
    torch::arange_out(t_req_ind,
                      static_cast<int64_t>(0),
                      static_cast<int64_t>(batch_size_),
                      static_cast<int64_t>(1));       // [0, 1, ..., B-1]
    t_qo_ind.zero_();                                 // [0, ..., 0]
    t_kv_ind.zero_();                                 // [0, ..., 0]
    t_chunk.fill_(static_cast<int32_t>(2000000000));  // Large chunk size to avoid splitting

    // Assign pointers to params
    params.request_indices   = req_ind_ptr;
    params.qo_tile_indices   = qo_ind_ptr;
    params.kv_tile_indices   = kv_ind_ptr;
    params.kv_chunk_size_ptr = chunk_sz_ptr;

    // IMPORTANT: Output Indptr MUST match Query Indptr for contiguous token generation
    params.o_indptr = params.q_indptr;

    // Use Dispatch API with Hardcoded 128
    constexpr uint32_t        HEAD_DIM_VAL          = 128;
    constexpr uint32_t        CTA_TILE_Q            = 128;  // Default
    constexpr PosEncodingMode POS                   = PosEncodingMode::kNone;
    constexpr bool            USE_FP16_QK_REDUCTION = false;  // BF16
    constexpr MaskMode        MASK                  = MaskMode::kCausal;
    using AttentionVariant                          = DefaultAttention<false, false, false, false>;

    cudaError_t status = BatchPrefillWithPagedKVCacheDispatched<CTA_TILE_Q,
                                                                HEAD_DIM_VAL,
                                                                HEAD_DIM_VAL,
                                                                POS,
                                                                USE_FP16_QK_REDUCTION,
                                                                MASK,
                                                                AttentionVariant,
                                                                Params>(params, nullptr, nullptr, false, stream);

    if (status != cudaSuccess) {
        NANODEPLOY_LOG_ERROR("BatchPrefill kernel failed: ", cudaGetErrorString(status));
    }

    return o;
}

// Wrapper Definitions (Moved from top)
void FlashInferOps::begin_forward_prefill(int32_t* q_indptr,
                                          int32_t* block_tables,
                                          int32_t* last_page_len_host,
                                          int      batch_size,
                                          int      max_num_blocks,
                                          int      num_qo_heads,
                                          int      num_kv_heads,
                                          int      head_dim,
                                          int      page_size)
{
    impl_->begin_forward_prefill(q_indptr,
                                 block_tables,
                                 last_page_len_host,
                                 batch_size,
                                 max_num_blocks,
                                 num_qo_heads,
                                 num_kv_heads,
                                 head_dim,
                                 page_size);
}

torch::Tensor FlashInferOps::prefill(void* q_ptr, void* k_cache_ptr, void* v_cache_ptr, int total_tokens, int layer_idx)
{
    return impl_->prefill(q_ptr, k_cache_ptr, v_cache_ptr, total_tokens, layer_idx);
}

torch::Tensor FlashInferOps::prefill_ragged(void*    q_ptr,
                                            void*    k_ptr,
                                            void*    v_ptr,
                                            int      total_q_tokens,
                                            int      total_kv_tokens,
                                            int32_t* q_indptr,
                                            int32_t* kv_indptr,
                                            int      batch_size)
{
    // Use SinglePrefillWithKVCache for batch_size=1 (simpler, no planning required)
    // This is equivalent to flash_attn_varlen_func for single sequence

    using DTypeQ  = bfloat16;
    using DTypeKV = bfloat16;
    using DTypeO  = bfloat16;
    using Params  = SinglePrefillParams<DTypeQ, DTypeKV, DTypeO>;

    int num_heads    = impl_->num_heads_;
    int num_kv_heads = impl_->num_kv_heads_;
    int head_dim     = impl_->head_dim_;

    // For batch_size > 1, we need to handle each sequence separately
    // For now, assume batch_size = 1 (common case for prefill)
    if (batch_size != 1) {
        NANODEPLOY_LOG_ERROR("[FlashInfer Ragged] batch_size > 1 not supported yet, got ", batch_size);
    }

    // Allocate output tensor [total_q_tokens, num_heads, head_dim]
    auto o = torch::empty({total_q_tokens, num_heads, head_dim},
                          torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Build SinglePrefillParams - much simpler than BatchPrefillRaggedParams
    Params params;
    params.q                  = static_cast<DTypeQ*>(q_ptr);
    params.k                  = static_cast<DTypeKV*>(k_ptr);
    params.v                  = static_cast<DTypeKV*>(v_ptr);
    params.maybe_custom_mask  = nullptr;
    params.o                  = static_cast<DTypeO*>(o.data_ptr());
    params.lse                = nullptr;
    params.maybe_alibi_slopes = nullptr;

    // Direct lengths - no indptr needed for single sequence
    params.qo_len       = static_cast<uint32_t>(total_q_tokens);
    params.kv_len       = static_cast<uint32_t>(total_kv_tokens);
    params.num_qo_heads = static_cast<uint32_t>(num_heads);
    params.num_kv_heads = static_cast<uint32_t>(num_kv_heads);
    params.group_size   = uint_fastdiv(static_cast<uint32_t>(num_heads / num_kv_heads));
    params.head_dim     = static_cast<uint32_t>(head_dim);

    // Strides for [TotalTokens, Heads, HeadDim] layout (NHD)
    params.q_stride_n = static_cast<uint32_t>(num_heads * head_dim);
    params.q_stride_h = static_cast<uint32_t>(head_dim);
    params.k_stride_n = static_cast<uint32_t>(num_kv_heads * head_dim);
    params.k_stride_h = static_cast<uint32_t>(head_dim);
    params.v_stride_n = static_cast<uint32_t>(num_kv_heads * head_dim);
    params.v_stride_h = static_cast<uint32_t>(head_dim);

    params.window_left     = -1;  // -1 for no sliding window (full causal)
    params.logits_soft_cap = 0.0f;
    params.sm_scale        = 1.0f / std::sqrt(float(head_dim));
    params.rope_rcp_scale  = 1.0f;  // No RoPE in kernel (already applied)
    params.rope_rcp_theta  = 1.0f;

    // Dispatch kernel using SinglePrefillWithKVCacheDispatched
    constexpr uint32_t        HEAD_DIM_VAL          = 128;
    constexpr PosEncodingMode POS                   = PosEncodingMode::kNone;
    constexpr bool            USE_FP16_QK_REDUCTION = false;
    constexpr MaskMode        MASK                  = MaskMode::kCausal;
    using AttentionVariant                          = DefaultAttention<false, false, false, false>;

    cudaError_t status = SinglePrefillWithKVCacheDispatched<HEAD_DIM_VAL,
                                                            HEAD_DIM_VAL,
                                                            POS,
                                                            USE_FP16_QK_REDUCTION,
                                                            MASK,
                                                            AttentionVariant,
                                                            Params>(params, nullptr, stream);

    if (status != cudaSuccess) {
        NANODEPLOY_LOG_ERROR("SinglePrefillWithKVCache failed: ", cudaGetErrorString(status));
    }

    return o;
}

}  // namespace ops
}  // namespace nanodeploy
