#pragma once

#include <torch/torch.h>

#include "nanodeploy/sequence/serialization.h"
#include "nanodeploy/worker/dummy_runner.h"
#include "spoke/serializer.h"

namespace spoke {

template<>
inline nanodeploy::RunReq Unpack<nanodeploy::RunReq>(const std::string& data)
{
    return nanodeploy::deserialize_sequences(reinterpret_cast<uintptr_t>(data.data()), data.size());
}

template<>
inline std::string Pack<nanodeploy::RunResp>(const nanodeploy::RunResp& data)
{
    // Use torch pickle for tensor serialization
    std::vector<char> buffer = torch::pickle_save(data);
    return std::string(buffer.begin(), buffer.end());
}

// Client side might need Pack<RunReq> too?
// Wait, Client calls `callRemote<Req, Resp>`.
// callRemote calls `Pack(req)`.
// Server calls `Unpack<Req>(data)`.
// Server handler returns Resp.
// Server calls `Pack(resp)`.
// Client calls `Unpack<Resp>(data)`.
// So we need Pack/Unpack for BOTH types if using symmetric definition or specific ones.
// Default serializer might handle basic types, but these are custom.

template<>
inline std::string Pack<nanodeploy::RunReq>(const nanodeploy::RunReq& data)
{
    // We need to implement serialize_sequences wrapper here
    // But serialize_sequences writes to a buffer.
    // We need to allocate a buffer.

    // Let's use a temporary vector or similar.
    // nanodeploy::serialize_sequences(uintptr_t data_ptr, size_t buffer_size, ...)

    // Estimate size? Or use a growing buffer?
    // For now, let's just allocate a large enough buffer or improve serialize_sequences API later.
    // Assuming 512KB is enough for metadata of sequences for now?
    std::vector<char> buf(4 * 1024 * 1024);  // 4MB
    size_t            size = nanodeploy::serialize_sequences((uintptr_t)buf.data(), buf.size(), data, true);
    return std::string(buf.data(), size);
}

template<>
inline nanodeploy::RunResp Unpack<nanodeploy::RunResp>(const std::string& data)
{
    std::vector<char> buf(data.begin(), data.end());
    torch::IValue     ivalue = torch::pickle_load(buf);
    return ivalue.toTensor();
}

}  // namespace spoke
