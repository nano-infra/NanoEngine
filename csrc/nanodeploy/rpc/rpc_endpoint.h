#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "nanodeploy/engine/sequence.h"
#include <dlslime/engine/rdma/rdma_endpoint.h>

namespace nanodeploy {

class RpcEndpoint {
public:
    RpcEndpoint() = default;
    ~RpcEndpoint() = default;

    // 获取数据指针（用于发送端）
    void* data()
    {
        return data_;
    }
    
    // 获取数据大小
    size_t size()
    {
        return size_;
    }

    // 设置外部 buffer（用于接收端，接收 RDMA Pinned Memory）
    // ptr 是 Python 传递过来的 uint64 地址
    void set_buffer(uint64_t ptr, size_t size);

    std::vector<std::shared_ptr<Sequence>>& sequences()
    {
        return seqs_;
    }

    // Python -> C++: 传入需要序列化的 Sequence 对象
    int32_t feed_sequences(std::vector<std::shared_ptr<Sequence>> seqs);

    // 序列化接口
    int32_t serialize_for_prefill();
    int32_t serialize_for_decode();
    int32_t serialize_for_migrate();

    // 反序列化接口
    int32_t deserialize_for_prefill();
    int32_t deserialize_for_decode();
    int32_t deserialize_for_migrate();

private:
    void* data_ = nullptr;
    size_t size_ = 0;
    bool   own_data_ = false; // 标记 data_ 是否指向 buffer_

    // 内部 buffer，用于存储序列化后的数据
    std::vector<uint8_t> buffer_;

    std::vector<std::shared_ptr<Sequence>> seqs_;

    // 预留的 RDMA Endpoint 指针，暂时保留以兼容原有结构
    std::shared_ptr<dlslime::RDMAEndpoint> rdma_endpoint_;
};

}  // namespace nanodeploy