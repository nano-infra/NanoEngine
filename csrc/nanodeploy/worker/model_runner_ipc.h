#pragma once

#include <torch/torch.h>
#include <vector>

#include "nanodeploy/sequence/sequence.h"
#include "nanodeploy/sequence/serialization.h"
#include "spoke/csrc/serializer.h"

namespace nanodeploy {

struct ModelRunReq {
    std::vector<std::shared_ptr<Sequence>> seqs;
    bool                                   is_prefill;
};

struct ModelInitReq {
    std::string config_path;
    std::string weight_map_path;
    std::string weight_path;
    int         rank;
    int         world_size;
    int         tp_degree;
    int         pp_degree;
    int         dp_degree;
};

struct ModelRunResp {
    torch::Tensor tensor;
};
using ModelInitResp = bool;

}  // namespace nanodeploy

namespace spoke {

// === ModelRunReq ===
template<>
struct Serializer<nanodeploy::ModelRunReq> {
    static constexpr size_t kMaxBufSize = 64 * 1024 * 1024;  // Increase buffer for large batches

    static std::string pack(const nanodeploy::ModelRunReq& data)
    {
        std::vector<char> buf(kMaxBufSize);
        // Serialize sequences
        size_t sz = nanodeploy::serialize_sequences((uintptr_t)buf.data(), buf.size(), data.seqs, data.is_prefill);

        // Append is_prefill boolean
        if (sz + 1 > buf.size()) {
            // ...
        }
        buf[sz] = data.is_prefill ? 1 : 0;

        return std::string(buf.data(), sz + 1);
    }

    static nanodeploy::ModelRunReq unpack(const std::string& data)
    {
        nanodeploy::ModelRunReq req;
        if (data.size() < 1)
            return req;

        // Last byte is is_prefill
        req.is_prefill = (data.back() == 1);

        // Deserialize sequences (excluding last byte)
        req.seqs = nanodeploy::deserialize_sequences(reinterpret_cast<uintptr_t>(data.data()), data.size() - 1);
        return req;
    }

    static size_t size(const nanodeploy::ModelRunReq& data)
    {
        return pack(data).size();
    }

    static void packTo(const nanodeploy::ModelRunReq& data, char* buf)
    {
        std::string packed = pack(data);
        std::memcpy(buf, packed.data(), packed.size());
    }

    static nanodeploy::ModelRunReq unpackFrom(const char* buf, size_t len)
    {
        // Re-use unpack logic by creating string wrapper (copy overhead but safe)
        // Optimization: Create unpackFrom capable deserialize_sequences
        return unpack(std::string(buf, len));
    }
};

// === ModelRunResp (Tensor) ===
// Reuse the logic from dummy_runner_ipc for Tensor, or redefine here if compatible.
// Since Serialize<torch::Tensor> isn't specialized, we specialize nanodeploy::RunResp which is Tensor.
// Here we specialize nanodeploy::ModelRunResp which is also Tensor.

template<>
struct Serializer<nanodeploy::ModelRunResp> {
    static std::string pack(const nanodeploy::ModelRunResp& data)
    {
        std::vector<char> buffer = torch::pickle_save(data.tensor);
        return std::string(buffer.begin(), buffer.end());
    }
    static nanodeploy::ModelRunResp unpack(const std::string& data)
    {
        std::vector<char> buf(data.begin(), data.end());
        torch::IValue     ivalue = torch::pickle_load(buf);
        return {ivalue.toTensor()};
    }
    static size_t size(const nanodeploy::ModelRunResp& data)
    {
        std::vector<char> buffer = torch::pickle_save(data.tensor);
        return buffer.size();
    }
    static void packTo(const nanodeploy::ModelRunResp& data, char* buf)
    {
        std::vector<char> buffer = torch::pickle_save(data.tensor);
        std::memcpy(buf, buffer.data(), buffer.size());
    }
    static nanodeploy::ModelRunResp unpackFrom(const char* buf, size_t len)
    {
        std::vector<char> vec(buf, buf + len);
        torch::IValue     ivalue = torch::pickle_load(vec);
        return {ivalue.toTensor()};
    }
};

// === ModelInitReq ===
template<>
struct Serializer<nanodeploy::ModelInitReq> {
    static size_t size(const nanodeploy::ModelInitReq& data)
    {
        return sizeof(int) * 6 + data.config_path.size();  // rank, ws, tp, pp, dp, size_of_str
    }
    static void packTo(const nanodeploy::ModelInitReq& data, char* buf)
    {
        char* ptr = buf;
        int   sz;

        sz         = (int)data.config_path.size();
        *(int*)ptr = sz;
        ptr += sizeof(int);
        memcpy(ptr, data.config_path.data(), sz);
        ptr += sz;

        *(int*)ptr = data.rank;
        ptr += sizeof(int);
        *(int*)ptr = data.world_size;
        ptr += sizeof(int);

        *(int*)ptr = data.tp_degree;
        ptr += sizeof(int);
        *(int*)ptr = data.pp_degree;
        ptr += sizeof(int);
        *(int*)ptr = data.dp_degree;
    }
    static nanodeploy::ModelInitReq unpackFrom(const char* buf, size_t)
    {
        nanodeploy::ModelInitReq req;
        const char*              ptr = buf;
        int                      sz;

        sz = *(int*)ptr;
        ptr += sizeof(int);
        req.config_path = std::string(ptr, sz);
        ptr += sz;

        req.rank = *(int*)ptr;
        ptr += sizeof(int);
        req.world_size = *(int*)ptr;
        ptr += sizeof(int);

        req.tp_degree = *(int*)ptr;
        ptr += sizeof(int);
        req.pp_degree = *(int*)ptr;
        ptr += sizeof(int);
        req.dp_degree = *(int*)ptr;

        return req;
    }
    static std::string pack(const nanodeploy::ModelInitReq& data)
    {
        std::string s;
        s.resize(size(data));
        packTo(data, s.data());
        return s;
    }
    static nanodeploy::ModelInitReq unpack(const std::string& data)
    {
        return unpackFrom(data.data(), data.size());
    }
};

// === ModelInitResp (bool) ===
template<>
struct Serializer<nanodeploy::ModelInitResp> {
    static std::string pack(const nanodeploy::ModelInitResp& data)
    {
        return data ? "1" : "0";
    }
    static nanodeploy::ModelInitResp unpack(const std::string& data)
    {
        return data == "1";
    }
    static size_t size(const nanodeploy::ModelInitResp& /*data*/)
    {
        return 1;
    }
    static void packTo(const nanodeploy::ModelInitResp& data, char* buf)
    {
        *buf = data ? '1' : '0';
    }
    static nanodeploy::ModelInitResp unpackFrom(const char* buf, size_t)
    {
        return *buf == '1';
    }
};

}  // namespace spoke
