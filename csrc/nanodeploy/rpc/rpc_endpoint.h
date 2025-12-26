#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "nanodeploy/engine/sequence.h"
#include <dlslime/engine/rdma/rdma_endpoint.h>

namespace nanodeploy {
class RpcEndpoint {
public:
    RpcEndpoint() {}
    ~RpcEndpoint() {}

    void* data()
    {
        return data_;
    }
    size_t size()
    {
        return size_;
    }

    std::vector<std::shared_ptr<Sequence>>& sequences()
    {
        return seqs_;
    }

    int32_t feed_sequences(std::vector<std::shared_ptr<Sequence>> seqs);

    int32_t serialize_for_prefill();

    int32_t serialize_for_decode();

    int32_t serialize_for_migrate();

    int32_t deserialize_for_prefill();

    int32_t deserialize_for_decode();

    int32_t deserialize_for_migrate();

private:
    void*  data_;
    size_t size_;

    std::vector<std::shared_ptr<Sequence>> seqs_;

    std::shared_ptr<dlslime::RDMAEndpoint> rdma_endpoint_;
};
}  // namespace nanodeploy
