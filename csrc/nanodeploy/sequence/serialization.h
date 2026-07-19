#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "sequence.h"

namespace nanodeploy {

// Raw Sequence payloads are an internal lockstep ABI. Legacy and mixed-version
// payloads are rejected rather than guessed or partially decoded.
inline constexpr uint64_t kSequenceSerializationMagic      = 0x4E44534551524157ULL;  // "NDSEQRAW"
inline constexpr uint32_t kSequenceSerializationVersion    = 1;
inline constexpr size_t   kSequenceSerializationHeaderSize =
    sizeof(kSequenceSerializationMagic) + sizeof(kSequenceSerializationVersion);

// Shared by the raw and pickle schemas so both reject malformed topology,
// block-table and inactive-context state at the same ABI boundary.
void validate_serializable_block_context(const BlockContext& context);

// assigned_dp is persistent LS ownership. A detached WAITING request may have
// an assigned pool while its default ACTIVE context remains uninitialized; once
// that context is dimensioned or the request enters an active lifecycle state,
// its DP must agree with the canonical assignment.
void validate_serializable_sequence_context_ownership(int                 assigned_dp,
                                                      SequenceStatus      status,
                                                      const BlockContext& active_context);

/**
 * @brief 序列化一组 Sequence
 * @param data_ptr 目标缓冲区的起始物理/虚拟地址
 * @param buffer_size 缓冲区总长度（用于安全检查）
 * @param seqs 要序列化的数据
 * @param is_prefill 是否为预填充阶段（影响序列化格式）
 * @param sp_rank 目标SP rank（仅在Decode优化路径中使用，用于裁剪目标rank无关的重字段）
 * @param sp_size SP world size（仅在Decode优化路径中使用）
 * @return size_t 实际写入的字节总数
 */
size_t serialize_sequences(uintptr_t                                     data_ptr,
                           size_t                                        buffer_size,
                           const std::vector<std::shared_ptr<Sequence>>& seqs,
                           bool                                          is_prefill,
                           int                                           sp_rank = -1,
                           int                                           sp_size = -1);

/**
 * @brief 反序列化一组 Sequence
 * @param data_ptr 源数据缓冲区的起始地址
 * @param data_len 有效数据长度
 * @return std::vector<std::shared_ptr<Sequence>> 还原出的对象列表
 */
std::vector<std::shared_ptr<Sequence>> deserialize_sequences(uintptr_t data_ptr, size_t data_len);

}  // namespace nanodeploy
