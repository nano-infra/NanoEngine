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

    int attention_tp = 1;
    int attention_dp = 1;
    int attention_sp = 1;

    int ffn_tp = 1;
    int ffn_dp = 1;
    int ffn_ep = 1;
};

struct ModelRunResp {
    torch::Tensor tensor;
};
using ModelInitResp = bool;

// DeepEP sync structures
struct DeepEpInfoResp {
    int         device_id = -1;
    std::string ipc_handle;         // Binary data of IPC handle
    std::string nvshmem_unique_id;  // Optional, only for root rank
    int         num_rdma_ranks = 1;
    int         rdma_rank      = 0;
    int         root_rdma_rank = 0;
};

struct DeepEpSyncReq {
    std::vector<int>         device_ids;
    std::vector<std::string> ipc_handles;             // One per rank
    std::string              root_nvshmem_unique_id;  // Optional
};

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
        // rank, ws, tp, pp, dp, att_tp, att_dp, att_sp, ffn_tp, ffn_dp, ffn_ep, size_of_str
        return sizeof(int) * 12 + data.config_path.size();
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
        ptr += sizeof(int);

        *(int*)ptr = data.attention_tp;
        ptr += sizeof(int);
        *(int*)ptr = data.attention_dp;
        ptr += sizeof(int);
        *(int*)ptr = data.attention_sp;
        ptr += sizeof(int);

        *(int*)ptr = data.ffn_tp;
        ptr += sizeof(int);
        *(int*)ptr = data.ffn_dp;
        ptr += sizeof(int);
        *(int*)ptr = data.ffn_ep;
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
        ptr += sizeof(int);

        req.attention_tp = *(int*)ptr;
        ptr += sizeof(int);
        req.attention_dp = *(int*)ptr;
        ptr += sizeof(int);
        req.attention_sp = *(int*)ptr;
        ptr += sizeof(int);

        req.ffn_tp = *(int*)ptr;
        ptr += sizeof(int);
        req.ffn_dp = *(int*)ptr;
        ptr += sizeof(int);
        req.ffn_ep = *(int*)ptr;

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

// === DeepEpInfoResp ===
template<>
struct Serializer<nanodeploy::DeepEpInfoResp> {
    static size_t size(const nanodeploy::DeepEpInfoResp& data)
    {
        // device_id, num_rdma_ranks, rdma_rank, root_rdma_rank, ipc_handle_len, nvshmem_id_len, data...
        return sizeof(int) * 6 + data.ipc_handle.size() + data.nvshmem_unique_id.size();
    }
    static void packTo(const nanodeploy::DeepEpInfoResp& data, char* buf)
    {
        char* ptr  = buf;
        *(int*)ptr = data.device_id;
        ptr += sizeof(int);
        *(int*)ptr = data.num_rdma_ranks;
        ptr += sizeof(int);
        *(int*)ptr = data.rdma_rank;
        ptr += sizeof(int);
        *(int*)ptr = data.root_rdma_rank;
        ptr += sizeof(int);
        *(int*)ptr = (int)data.ipc_handle.size();
        ptr += sizeof(int);
        memcpy(ptr, data.ipc_handle.data(), data.ipc_handle.size());
        ptr += data.ipc_handle.size();
        *(int*)ptr = (int)data.nvshmem_unique_id.size();
        ptr += sizeof(int);
        memcpy(ptr, data.nvshmem_unique_id.data(), data.nvshmem_unique_id.size());
    }
    static nanodeploy::DeepEpInfoResp unpackFrom(const char* buf, size_t)
    {
        nanodeploy::DeepEpInfoResp resp;
        const char*                ptr = buf;
        resp.device_id                 = *(int*)ptr;
        ptr += sizeof(int);
        resp.num_rdma_ranks = *(int*)ptr;
        ptr += sizeof(int);
        resp.rdma_rank = *(int*)ptr;
        ptr += sizeof(int);
        resp.root_rdma_rank = *(int*)ptr;
        ptr += sizeof(int);
        int ipc_len = *(int*)ptr;
        ptr += sizeof(int);
        resp.ipc_handle = std::string(ptr, ipc_len);
        ptr += ipc_len;
        int nvshmem_len = *(int*)ptr;
        ptr += sizeof(int);
        resp.nvshmem_unique_id = std::string(ptr, nvshmem_len);
        return resp;
    }
    static std::string pack(const nanodeploy::DeepEpInfoResp& data)
    {
        std::string s;
        s.resize(size(data));
        packTo(data, s.data());
        return s;
    }
    static nanodeploy::DeepEpInfoResp unpack(const std::string& data)
    {
        return unpackFrom(data.data(), data.size());
    }
};

// === DeepEpSyncReq ===
template<>
struct Serializer<nanodeploy::DeepEpSyncReq> {
    static size_t size(const nanodeploy::DeepEpSyncReq& data)
    {
        size_t sz = sizeof(int) * 2;  // num_devices, root_id_len
        sz += sizeof(int) * data.device_ids.size();
        for (const auto& h : data.ipc_handles) {
            sz += sizeof(int) + h.size();
        }
        sz += data.root_nvshmem_unique_id.size();
        return sz;
    }
    static void packTo(const nanodeploy::DeepEpSyncReq& data, char* buf)
    {
        char* ptr  = buf;
        *(int*)ptr = (int)data.device_ids.size();
        ptr += sizeof(int);
        for (int id : data.device_ids) {
            *(int*)ptr = id;
            ptr += sizeof(int);
        }
        for (const auto& h : data.ipc_handles) {
            *(int*)ptr = (int)h.size();
            ptr += sizeof(int);
            memcpy(ptr, h.data(), h.size());
            ptr += h.size();
        }
        *(int*)ptr = (int)data.root_nvshmem_unique_id.size();
        ptr += sizeof(int);
        memcpy(ptr, data.root_nvshmem_unique_id.data(), data.root_nvshmem_unique_id.size());
    }
    static nanodeploy::DeepEpSyncReq unpackFrom(const char* buf, size_t)
    {
        nanodeploy::DeepEpSyncReq req;
        const char*               ptr         = buf;
        int                       num_devices = *(int*)ptr;
        ptr += sizeof(int);
        req.device_ids.resize(num_devices);
        for (int i = 0; i < num_devices; ++i) {
            req.device_ids[i] = *(int*)ptr;
            ptr += sizeof(int);
        }
        req.ipc_handles.resize(num_devices);
        for (int i = 0; i < num_devices; ++i) {
            int len = *(int*)ptr;
            ptr += sizeof(int);
            req.ipc_handles[i] = std::string(ptr, len);
            ptr += len;
        }
        int root_id_len = *(int*)ptr;
        ptr += sizeof(int);
        req.root_nvshmem_unique_id = std::string(ptr, root_id_len);
        return req;
    }
    static std::string pack(const nanodeploy::DeepEpSyncReq& data)
    {
        std::string s;
        s.resize(size(data));
        packTo(data, s.data());
        return s;
    }
    static nanodeploy::DeepEpSyncReq unpack(const std::string& data)
    {
        return unpackFrom(data.data(), data.size());
    }
};

}  // namespace spoke
