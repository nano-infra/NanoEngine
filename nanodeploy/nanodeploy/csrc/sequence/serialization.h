#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "sequence.h"

namespace nanodeploy {

/**
 * @brief Serialize a single Sequence to a byte buffer
 * @param seq The sequence to serialize
 * @return std::vector<uint8_t> Serialized bytes
 */
std::vector<uint8_t> serialize_sequence(const Sequence& seq);

/**
 * @brief Deserialize a single Sequence from a byte buffer
 * @param buffer Pointer to buffer
 * @param size Buffer size
 * @return std::shared_ptr<Sequence> Deserialized sequence
 */
std::shared_ptr<Sequence> deserialize_sequence(const uint8_t* buffer, size_t size);

/**
 * @brief 序列化一组 Sequence
 * @param data_ptr 目标缓冲区的起始物理/虚拟地址
 * @param buffer_size 缓冲区总长度（用于安全检查）
 * @param seqs 要序列化的数据
 * @param is_prefill 是否为预填充阶段（影响序列化格式）
 * @return size_t 实际写入的字节总数
 */
size_t serialize_sequences(uintptr_t                                     data_ptr,
                           size_t                                        buffer_size,
                           const std::vector<std::shared_ptr<Sequence>>& seqs,
                           bool                                          is_prefill);

/**
 * @brief 反序列化一组 Sequence
 * @param data_ptr 源数据缓冲区的起始地址
 * @param data_len 有效数据长度
 * @return std::vector<std::shared_ptr<Sequence>> 还原出的对象列表
 */
std::vector<std::shared_ptr<Sequence>> deserialize_sequences(uintptr_t data_ptr, size_t data_len);

/**
 * @brief Internal helper - creates Sequence directly from unpacked FlatBuffer data
 * @param data Unpacked SequenceT data
 * @return std::shared_ptr<Sequence> The created sequence
 */
std::shared_ptr<Sequence> sequence_from_data(std::unique_ptr<SequenceT> data);

}  // namespace nanodeploy
