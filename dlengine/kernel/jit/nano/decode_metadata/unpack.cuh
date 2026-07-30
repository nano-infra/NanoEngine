#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace {

constexpr uint32_t kDecodeMetadataMagic   = 0x444d444c;
constexpr uint16_t kDecodeMetadataVersion = 1;

struct alignas(16) DecodeMetadataHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t flags;
    uint32_t payload_bytes;
    uint32_t num_seqs;
    uint32_t max_num_seqs;
    uint32_t max_num_blocks;
    uint32_t block_size;
    uint32_t input_ids_offset;
    uint32_t positions_offset;
    uint32_t temperatures_offset;
    uint32_t state_slots_offset;
    uint32_t hisparse_slots_offset;
    uint32_t block_row_offsets_offset;
    uint32_t block_ids_offset;
    uint32_t seq_ids_offset;
    uint32_t block_count;
};

static_assert(sizeof(DecodeMetadataHeader) == 64);

__global__ void decode_metadata_unpack_kernel(const uint8_t* __restrict__ payload,
                                              const int32_t dummy_state_slot,
                                              int64_t* __restrict__ input_ids_out,
                                              int64_t* __restrict__ positions_out,
                                              float* __restrict__ temperatures_out,
                                              int64_t* __restrict__ state_slots_out,
                                              int32_t* __restrict__ slot_mapping_out,
                                              int32_t* __restrict__ context_lens_out,
                                              int32_t* __restrict__ block_tables_out,
                                              const uint32_t sp_size,
                                              int32_t* __restrict__ plan_indptr_out,
                                              int32_t* __restrict__ plan_indices_out,
                                              int32_t* __restrict__ plan_last_page_len_out,
                                              const uint32_t graph_batch_size)
{
    const auto* header = reinterpret_cast<const DecodeMetadataHeader*>(payload);
    if (header->magic != kDecodeMetadataMagic || header->version != kDecodeMetadataVersion) {
        return;
    }
    const uint32_t index          = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t num_seqs       = header->num_seqs;
    const uint32_t max_num_seqs   = header->max_num_seqs;
    const uint32_t max_num_blocks = header->max_num_blocks;
    const uint32_t block_size     = header->block_size;

    const auto* input_ids    = reinterpret_cast<const int64_t*>(payload + header->input_ids_offset);
    const auto* positions    = reinterpret_cast<const int64_t*>(payload + header->positions_offset);
    const auto* temperatures = reinterpret_cast<const float*>(payload + header->temperatures_offset);
    const auto* state_slots  = reinterpret_cast<const int64_t*>(payload + header->state_slots_offset);
    const auto* row_offsets  = reinterpret_cast<const uint32_t*>(payload + header->block_row_offsets_offset);
    const auto* block_ids    = reinterpret_cast<const int32_t*>(payload + header->block_ids_offset);

    if (index < max_num_seqs) {
        const bool    active    = index < num_seqs;
        const int64_t position  = active ? positions[index] : 0;
        input_ids_out[index]    = active ? input_ids[index] : 0;
        positions_out[index]    = position;
        temperatures_out[index] = active ? temperatures[index] : 1.0f;

        const int64_t state_slot = active ? state_slots[index] : dummy_state_slot;
        state_slots_out[index]   = state_slot >= 0 && state_slot < dummy_state_slot ? state_slot : dummy_state_slot;
        // Inactive graph rows use length 1 so FlashInfer can plan a valid
        // padded bucket. Eager consumers slice to num_seqs.
        context_lens_out[index] = active ? static_cast<int32_t>(position + 1) : 1;

        int32_t physical_slot = -1;
        if (active && position >= 0) {
            const uint32_t logical_block = static_cast<uint32_t>(position) / block_size;
            const uint32_t row_begin     = row_offsets[index];
            const uint32_t row_end       = row_offsets[index + 1];
            if (logical_block < row_end - row_begin) {
                physical_slot = block_ids[row_begin + logical_block] * static_cast<int32_t>(block_size)
                                + static_cast<int32_t>(position % block_size);
            }
        }
        slot_mapping_out[index] = physical_slot;
    }

    const uint64_t table_elements = static_cast<uint64_t>(sp_size) * max_num_seqs * max_num_blocks;
    if (index < table_elements) {
        const uint32_t column   = index % max_num_blocks;
        const uint32_t row      = (index / max_num_blocks) % max_num_seqs;
        int32_t        block_id = 0;
        if (row < num_seqs) {
            const uint32_t row_begin = row_offsets[row];
            const uint32_t row_end   = row_offsets[row + 1];
            if (column < row_end - row_begin) {
                block_id = block_ids[row_begin + column];
            }
        }
        block_tables_out[index] = block_id;
    }

    if (plan_indptr_out != nullptr && index <= graph_batch_size) {
        int32_t page_offset = 0;
        for (uint32_t row = 0; row < index; ++row) {
            page_offset += row < num_seqs ? static_cast<int32_t>((positions[row] + block_size) / block_size) : 1;
        }
        plan_indptr_out[index] = page_offset;
    }
    if (plan_indices_out != nullptr) {
        uint32_t packed_offset = index;
        int32_t  block_id      = 0;
        bool     valid         = false;
        for (uint32_t row = 0; row < graph_batch_size; ++row) {
            const uint32_t pages =
                row < num_seqs ? static_cast<uint32_t>((positions[row] + block_size) / block_size) : 1;
            if (packed_offset < pages) {
                block_id = row < num_seqs ? block_ids[row_offsets[row] + packed_offset] : 0;
                valid    = true;
                break;
            }
            packed_offset -= pages;
        }
        if (valid) {
            plan_indices_out[index] = block_id;
        }
    }
    if (plan_last_page_len_out != nullptr && index < graph_batch_size) {
        if (index < num_seqs) {
            const int32_t context_len     = static_cast<int32_t>(positions[index] + 1);
            const int32_t remainder       = context_len % static_cast<int32_t>(block_size);
            plan_last_page_len_out[index] = remainder == 0 ? static_cast<int32_t>(block_size) : remainder;
        }
        else {
            plan_last_page_len_out[index] = 1;
        }
    }
}

struct DecodeMetadataUnpack {
    static int64_t register_mapped(tvm::ffi::TensorView payload)
    {
        using namespace host;
        auto bytes = SymbolicSize{"payload_bytes"};
        TensorMatcher({bytes}).with_dtype<uint8_t>().verify(payload);
        RuntimeCheck(payload.device().device_type == kDLCPU, "mapped payload must be a CPU tensor");

        int            device = 0;
        cudaDeviceProp properties{};
        RuntimeDeviceCheck(cudaGetDevice(&device));
        RuntimeDeviceCheck(cudaGetDeviceProperties(&properties, device));
        RuntimeCheck(properties.canMapHostMemory != 0, "CUDA device cannot map host memory");
        RuntimeDeviceCheck(cudaHostRegister(payload.data_ptr(), bytes.unwrap(), cudaHostRegisterMapped));
        void* device_pointer = nullptr;
        RuntimeDeviceCheck(cudaHostGetDevicePointer(&device_pointer, payload.data_ptr(), 0));
        return reinterpret_cast<int64_t>(device_pointer);
    }

    static void unregister_mapped(tvm::ffi::TensorView payload)
    {
        using namespace host;
        auto bytes = SymbolicSize{"payload_bytes"};
        TensorMatcher({bytes}).with_dtype<uint8_t>().verify(payload);
        RuntimeDeviceCheck(cudaHostUnregister(payload.data_ptr()));
    }

    static void launch(int64_t              mapped_device_pointer,
                       int32_t              dummy_state_slot,
                       tvm::ffi::TensorView input_ids,
                       tvm::ffi::TensorView positions,
                       tvm::ffi::TensorView temperatures,
                       tvm::ffi::TensorView state_slots,
                       tvm::ffi::TensorView slot_mapping,
                       tvm::ffi::TensorView context_lens,
                       tvm::ffi::TensorView block_tables)
    {
        using namespace host;
        auto max_num_seqs   = SymbolicSize{"max_num_seqs"};
        auto sp_size        = SymbolicSize{"sp_size"};
        auto max_num_blocks = SymbolicSize{"max_num_blocks"};
        auto cuda           = SymbolicDevice{};
        cuda.set_options<kDLCUDA>();
        TensorMatcher({max_num_seqs})
            .with_dtype<int64_t>()
            .with_device(cuda)
            .verify(input_ids)
            .verify(positions)
            .verify(state_slots);
        TensorMatcher({max_num_seqs}).with_dtype<float>().with_device(cuda).verify(temperatures);
        TensorMatcher({max_num_seqs}).with_dtype<int32_t>().with_device(cuda).verify(slot_mapping);
        TensorMatcher({sp_size, max_num_seqs}).with_dtype<int32_t>().with_device(cuda).verify(context_lens);
        TensorMatcher({sp_size, max_num_seqs, max_num_blocks})
            .with_dtype<int32_t>()
            .with_device(cuda)
            .verify(block_tables);

        const uint64_t work_items = std::max<uint64_t>(
            max_num_seqs.unwrap(), sp_size.unwrap() * max_num_seqs.unwrap() * max_num_blocks.unwrap());
        constexpr uint32_t threads = 256;
        const uint32_t     blocks  = static_cast<uint32_t>((work_items + threads - 1) / threads);
        const auto         stream  = LaunchKernel::resolve_device(cuda.unwrap());
        decode_metadata_unpack_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(mapped_device_pointer),
            dummy_state_slot,
            static_cast<int64_t*>(input_ids.data_ptr()),
            static_cast<int64_t*>(positions.data_ptr()),
            static_cast<float*>(temperatures.data_ptr()),
            static_cast<int64_t*>(state_slots.data_ptr()),
            static_cast<int32_t*>(slot_mapping.data_ptr()),
            static_cast<int32_t*>(context_lens.data_ptr()),
            static_cast<int32_t*>(block_tables.data_ptr()),
            static_cast<uint32_t>(sp_size.unwrap()),
            nullptr,
            nullptr,
            nullptr,
            0);
    }

    static void launch_graph(int64_t              mapped_device_pointer,
                             int32_t              dummy_state_slot,
                             tvm::ffi::TensorView input_ids,
                             tvm::ffi::TensorView positions,
                             tvm::ffi::TensorView temperatures,
                             tvm::ffi::TensorView state_slots,
                             tvm::ffi::TensorView slot_mapping,
                             tvm::ffi::TensorView context_lens,
                             tvm::ffi::TensorView block_tables,
                             tvm::ffi::TensorView plan_indptr,
                             tvm::ffi::TensorView plan_indices,
                             tvm::ffi::TensorView plan_last_page_len)
    {
        using namespace host;
        auto max_num_seqs   = SymbolicSize{"max_num_seqs"};
        auto sp_size        = SymbolicSize{"sp_size"};
        auto max_num_blocks = SymbolicSize{"max_num_blocks"};
        auto graph_bs       = SymbolicSize{"graph_bs"};
        auto indptr_size    = SymbolicSize{"indptr_size"};
        auto plan_capacity  = SymbolicSize{"plan_capacity"};
        auto cuda           = SymbolicDevice{};
        cuda.set_options<kDLCUDA>();
        TensorMatcher({max_num_seqs})
            .with_dtype<int64_t>()
            .with_device(cuda)
            .verify(input_ids)
            .verify(positions)
            .verify(state_slots);
        TensorMatcher({max_num_seqs}).with_dtype<float>().with_device(cuda).verify(temperatures);
        TensorMatcher({max_num_seqs}).with_dtype<int32_t>().with_device(cuda).verify(slot_mapping);
        TensorMatcher({sp_size, max_num_seqs}).with_dtype<int32_t>().with_device(cuda).verify(context_lens);
        TensorMatcher({sp_size, max_num_seqs, max_num_blocks})
            .with_dtype<int32_t>()
            .with_device(cuda)
            .verify(block_tables);
        TensorMatcher({indptr_size}).with_dtype<int32_t>().with_device(cuda).verify(plan_indptr);
        TensorMatcher({plan_capacity}).with_dtype<int32_t>().with_device(cuda).verify(plan_indices);
        TensorMatcher({graph_bs}).with_dtype<int32_t>().with_device(cuda).verify(plan_last_page_len);
        RuntimeCheck(indptr_size.unwrap() == graph_bs.unwrap() + 1,
                     "FlashInfer plan indptr must have graph_bs + 1 elements");

        const uint64_t work_items =
            std::max<uint64_t>(std::max<uint64_t>(max_num_seqs.unwrap(),
                                                  sp_size.unwrap() * max_num_seqs.unwrap() * max_num_blocks.unwrap()),
                               std::max<uint64_t>(graph_bs.unwrap() + 1, plan_capacity.unwrap()));
        constexpr uint32_t threads = 256;
        const uint32_t     blocks  = static_cast<uint32_t>((work_items + threads - 1) / threads);
        const auto         stream  = LaunchKernel::resolve_device(cuda.unwrap());
        decode_metadata_unpack_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(mapped_device_pointer),
            dummy_state_slot,
            static_cast<int64_t*>(input_ids.data_ptr()),
            static_cast<int64_t*>(positions.data_ptr()),
            static_cast<float*>(temperatures.data_ptr()),
            static_cast<int64_t*>(state_slots.data_ptr()),
            static_cast<int32_t*>(slot_mapping.data_ptr()),
            static_cast<int32_t*>(context_lens.data_ptr()),
            static_cast<int32_t*>(block_tables.data_ptr()),
            static_cast<uint32_t>(sp_size.unwrap()),
            static_cast<int32_t*>(plan_indptr.data_ptr()),
            static_cast<int32_t*>(plan_indices.data_ptr()),
            static_cast<int32_t*>(plan_last_page_len.data_ptr()),
            static_cast<uint32_t>(graph_bs.unwrap()));
    }
};

}  // namespace
