#pragma once

#include <torch/torch.h>

#include "nanodeploy/csrc/sequence/serialization.h"
#include "nanodeploy/csrc/worker/dummy_runner.h"
#include "spoke/csrc/serializer.h"

namespace nanodeploy {
struct RunAllReduceReq {};
}  // namespace nanodeploy

namespace spoke {

// === RunAllReduceReq (Empty) ===
template<>
struct Serializer<nanodeploy::RunAllReduceReq> {
    static std::string pack(const nanodeploy::RunAllReduceReq&)
    {
        return "";
    }
    static nanodeploy::RunAllReduceReq unpack(const std::string&)
    {
        return {};
    }
    static size_t size(const nanodeploy::RunAllReduceReq&)
    {
        return 0;
    }
    static void                        packTo(const nanodeploy::RunAllReduceReq&, char*) {}
    static nanodeploy::RunAllReduceReq unpackFrom(const char*, size_t)
    {
        return {};
    }
};

// === RunReq ===
template<>
struct Serializer<nanodeploy::RunReq> {
    static constexpr size_t kMaxBufSize = 4 * 1024 * 1024;

    static std::string pack(const nanodeploy::RunReq& data)
    {
        std::vector<char> buf(kMaxBufSize);
        size_t            sz = nanodeploy::serialize_sequences((uintptr_t)buf.data(), buf.size(), data, true);
        return std::string(buf.data(), sz);
    }
    static nanodeploy::RunReq unpack(const std::string& data)
    {
        return nanodeploy::deserialize_sequences(reinterpret_cast<uintptr_t>(data.data()), data.size());
    }
    static size_t size(const nanodeploy::RunReq& data)
    {
        // Use pack() to get exact size - this is called before packTo
        return pack(data).size();
    }
    static void packTo(const nanodeploy::RunReq& data, char* buf)
    {
        // buf size is from size() call, serialize directly into it
        // Note: serialize_sequences needs enough space, but we trust size() was called first
        std::string packed = pack(data);
        std::memcpy(buf, packed.data(), packed.size());
    }
    static nanodeploy::RunReq unpackFrom(const char* buf, size_t len)
    {
        return nanodeploy::deserialize_sequences(reinterpret_cast<uintptr_t>(buf), len);
    }
};

// === RunResp (Tensor) ===
template<>
struct Serializer<nanodeploy::RunResp> {
    static std::string pack(const nanodeploy::RunResp& data)
    {
        std::vector<char> buffer = torch::pickle_save(data);
        return std::string(buffer.begin(), buffer.end());
    }
    static nanodeploy::RunResp unpack(const std::string& data)
    {
        std::vector<char> buf(data.begin(), data.end());
        torch::IValue     ivalue = torch::pickle_load(buf);
        return ivalue.toTensor();
    }
    static size_t size(const nanodeploy::RunResp& data)
    {
        std::vector<char> buffer = torch::pickle_save(data);
        return buffer.size();
    }
    static void packTo(const nanodeploy::RunResp& data, char* buf)
    {
        std::vector<char> buffer = torch::pickle_save(data);
        std::memcpy(buf, buffer.data(), buffer.size());
    }
    static nanodeploy::RunResp unpackFrom(const char* buf, size_t len)
    {
        std::vector<char> vec(buf, buf + len);
        torch::IValue     ivalue = torch::pickle_load(vec);
        return ivalue.toTensor();
    }
};

// === DistConfig ===
template<>
struct Serializer<nanodeploy::DistConfig> {
    static size_t size(const nanodeploy::DistConfig& data)
    {
        return sizeof(int) * 3 + data.master_addr.size() + sizeof(int);
    }
    static void packTo(const nanodeploy::DistConfig& data, char* buf)
    {
        char* ptr  = buf;
        *(int*)ptr = data.rank;
        ptr += sizeof(int);
        *(int*)ptr = data.world_size;
        ptr += sizeof(int);
        int addr_len = data.master_addr.size();
        *(int*)ptr   = addr_len;
        ptr += sizeof(int);
        std::memcpy(ptr, data.master_addr.data(), addr_len);
        ptr += addr_len;
        *(int*)ptr = data.master_port;
    }
    static nanodeploy::DistConfig unpackFrom(const char* buf, size_t)
    {
        nanodeploy::DistConfig cfg;
        const char*            ptr = buf;
        cfg.rank                   = *(int*)ptr;
        ptr += sizeof(int);
        cfg.world_size = *(int*)ptr;
        ptr += sizeof(int);
        int addr_len = *(int*)ptr;
        ptr += sizeof(int);
        cfg.master_addr = std::string(ptr, addr_len);
        ptr += addr_len;
        cfg.master_port = *(int*)ptr;
        return cfg;
    }
    static std::string pack(const nanodeploy::DistConfig& data)
    {
        std::string s;
        s.resize(size(data));
        packTo(data, s.data());
        return s;
    }
    static nanodeploy::DistConfig unpack(const std::string& data)
    {
        return unpackFrom(data.data(), data.size());
    }
};

// === DistResp ===
template<>
struct Serializer<nanodeploy::DistResp> {
    static size_t size(const nanodeploy::DistResp& data)
    {
        return sizeof(bool) + sizeof(int) + data.message.size();
    }
    static void packTo(const nanodeploy::DistResp& data, char* buf)
    {
        char* ptr   = buf;
        *(bool*)ptr = data.success;
        ptr += sizeof(bool);
        int msg_len = data.message.size();
        *(int*)ptr  = msg_len;
        ptr += sizeof(int);
        std::memcpy(ptr, data.message.data(), msg_len);
    }
    static nanodeploy::DistResp unpackFrom(const char* buf, size_t)
    {
        nanodeploy::DistResp resp;
        const char*          ptr = buf;
        resp.success             = *(bool*)ptr;
        ptr += sizeof(bool);
        int msg_len = *(int*)ptr;
        ptr += sizeof(int);
        resp.message = std::string(ptr, msg_len);
        return resp;
    }
    static std::string pack(const nanodeploy::DistResp& data)
    {
        std::string s;
        s.resize(size(data));
        packTo(data, s.data());
        return s;
    }
    static nanodeploy::DistResp unpack(const std::string& data)
    {
        return unpackFrom(data.data(), data.size());
    }
};

}  // namespace spoke
