#pragma once

#include "spoke/csrc/serializer.h"
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace nanodeploy {

// --- Requests/Responses Definitions ---

struct DeepGemmInitReq {
    int rank;
    int world_size;
};

struct DeepGemmInitResp {
    bool        success;
    std::string message;
};

enum class DeepGemmTestMode : int {
    kFp8Gemm = 0,
    kMaskedGroupGemm = 1
};

struct DeepGemmTestReq {
    int mode = 0; // 0: Fp8Gemm, 1: MaskedGroupGemm
    int m = 4096;
    int n = 4096;
    int k = 4096;
    int num_groups = 128; // For masked grouped gemm
    int seed = 1;
    int warmup_iters = 5;
    int test_iters = 20;
};

struct DeepGemmTestResp {
    bool        success;
    double      lat_us;
    std::string message;
};

}  // namespace nanodeploy

namespace spoke {

// Reuse helper for string serialization from deep_ep_ipc.h if possible, 
// but since headers might be included separately, better to redefine or make common utils.
// For now, I'll assume simple implementation here or duplicate helper.
// Actually deep_ep_ipc.h had them inline. I will copy them for safety/independence.

inline void serialize_str_gemm(std::string& buf, const std::string& str)
{
    size_t sz       = str.size();
    size_t old_size = buf.size();
    buf.resize(old_size + sizeof(size_t) + sz);
    char* ptr = &buf[old_size];
    std::memcpy(ptr, &sz, sizeof(size_t));
    if (sz > 0)
        std::memcpy(ptr + sizeof(size_t), str.data(), sz);
}

inline void deserialize_str_gemm(const char*& ptr, std::string& str)
{
    size_t sz;
    std::memcpy(&sz, ptr, sizeof(size_t));
    ptr += sizeof(size_t);
    str.resize(sz);
    if (sz > 0) {
        std::memcpy(str.data(), ptr, sz);
        ptr += sz;
    }
}

inline size_t packed_size_str_gemm(const std::string& str)
{
    return sizeof(size_t) + str.size();
}

inline void pack_to_str_gemm(char*& ptr, const std::string& str)
{
    size_t sz = str.size();
    std::memcpy(ptr, &sz, sizeof(size_t));
    ptr += sizeof(size_t);
    if (sz > 0) {
        std::memcpy(ptr, str.data(), sz);
        ptr += sz;
    }
}

// DeepGemmInitReq
template<>
struct Serializer<nanodeploy::DeepGemmInitReq> {
    static std::string pack(const nanodeploy::DeepGemmInitReq& obj)
    {
        std::string s;
        s.resize(sizeof(obj));
        std::memcpy(s.data(), &obj, sizeof(obj));
        return s;
    }
    static nanodeploy::DeepGemmInitReq unpack(const std::string& data)
    {
        return unpackFrom(data.data(), data.size());
    }
    static nanodeploy::DeepGemmInitReq unpackFrom(const char* buf, size_t len)
    {
        nanodeploy::DeepGemmInitReq obj;
        if (len >= sizeof(obj))
            std::memcpy(&obj, buf, sizeof(obj));
        return obj;
    }
    static size_t size(const nanodeploy::DeepGemmInitReq&)
    {
        return sizeof(nanodeploy::DeepGemmInitReq);
    }
    static void packTo(const nanodeploy::DeepGemmInitReq& obj, char* buf)
    {
        std::memcpy(buf, &obj, sizeof(nanodeploy::DeepGemmInitReq));
    }
};

// DeepGemmInitResp
template<>
struct Serializer<nanodeploy::DeepGemmInitResp> {
    static std::string pack(const nanodeploy::DeepGemmInitResp& obj)
    {
        std::string s;
        s.resize(sizeof(bool));
        std::memcpy(s.data(), &obj.success, sizeof(bool));
        serialize_str_gemm(s, obj.message);
        return s;
    }
    static nanodeploy::DeepGemmInitResp unpack(const std::string& data)
    {
        return unpackFrom(data.data(), data.size());
    }
    static nanodeploy::DeepGemmInitResp unpackFrom(const char* buf, size_t len)
    {
        nanodeploy::DeepGemmInitResp obj;
        const char* ptr = buf;
        if (len < sizeof(bool)) return obj;
        std::memcpy(&obj.success, ptr, sizeof(bool));
        ptr += sizeof(bool);
        deserialize_str_gemm(ptr, obj.message);
        return obj;
    }
    static size_t size(const nanodeploy::DeepGemmInitResp& obj)
    {
        return sizeof(bool) + packed_size_str_gemm(obj.message);
    }
    static void packTo(const nanodeploy::DeepGemmInitResp& obj, char* buf)
    {
        char* ptr = buf;
        std::memcpy(ptr, &obj.success, sizeof(bool));
        ptr += sizeof(bool);
        pack_to_str_gemm(ptr, obj.message);
    }
};

// DeepGemmTestReq
template<>
struct Serializer<nanodeploy::DeepGemmTestReq> {
    static std::string pack(const nanodeploy::DeepGemmTestReq& obj)
    {
        std::string s;
        s.resize(sizeof(obj));
        std::memcpy(s.data(), &obj, sizeof(obj));
        return s;
    }
    static nanodeploy::DeepGemmTestReq unpack(const std::string& data)
    {
        return unpackFrom(data.data(), data.size());
    }
    static nanodeploy::DeepGemmTestReq unpackFrom(const char* buf, size_t len)
    {
        nanodeploy::DeepGemmTestReq obj;
        if (len >= sizeof(obj))
            std::memcpy(&obj, buf, sizeof(obj));
        return obj;
    }
    static size_t size(const nanodeploy::DeepGemmTestReq&)
    {
        return sizeof(nanodeploy::DeepGemmTestReq);
    }
    static void packTo(const nanodeploy::DeepGemmTestReq& obj, char* buf)
    {
        std::memcpy(buf, &obj, sizeof(nanodeploy::DeepGemmTestReq));
    }
};

// DeepGemmTestResp
template<>
struct Serializer<nanodeploy::DeepGemmTestResp> {
    static std::string pack(const nanodeploy::DeepGemmTestResp& obj)
    {
        std::string s;
        s.resize(sizeof(bool) + sizeof(double));
        char* ptr = s.data();
        std::memcpy(ptr, &obj.success, sizeof(bool));
        ptr += sizeof(bool);
        std::memcpy(ptr, &obj.lat_us, sizeof(double));
        ptr += sizeof(double);
        serialize_str_gemm(s, obj.message);
        return s;
    }
    static nanodeploy::DeepGemmTestResp unpack(const std::string& data)
    {
        return unpackFrom(data.data(), data.size());
    }
    static nanodeploy::DeepGemmTestResp unpackFrom(const char* buf, size_t len)
    {
        nanodeploy::DeepGemmTestResp obj;
        const char* ptr = buf;
        if (len < sizeof(bool) + sizeof(double)) return obj;
        std::memcpy(&obj.success, ptr, sizeof(bool));
        ptr += sizeof(bool);
        std::memcpy(&obj.lat_us, ptr, sizeof(double));
        ptr += sizeof(double);
        deserialize_str_gemm(ptr, obj.message);
        return obj;
    }
    static size_t size(const nanodeploy::DeepGemmTestResp& obj)
    {
        return sizeof(bool) + sizeof(double) + packed_size_str_gemm(obj.message);
    }
    static void packTo(const nanodeploy::DeepGemmTestResp& obj, char* buf)
    {
        char* ptr = buf;
        std::memcpy(ptr, &obj.success, sizeof(bool));
        ptr += sizeof(bool);
        std::memcpy(ptr, &obj.lat_us, sizeof(double));
        ptr += sizeof(double);
        pack_to_str_gemm(ptr, obj.message);
    }
};

} // namespace spoke
