#pragma once

#include "spoke/csrc/serializer.h"
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace nanodeploy {

// --- Requests/Responses Definitions ---

struct DeepEPInitReq {
    int     rank;
    int     world_size;
    int64_t num_nvl_bytes    = 1024 * 1024 * 128;  // 128MB
    int64_t num_rdma_bytes   = 1024 * 1024 * 128;  // 128MB
    bool    low_latency_mode = true;
};

struct DeepEPInitResp {
    bool        success;
    std::string message;
};

struct DeepEPInfoReq {
    int dummy = 0;
};

struct DeepEPInfoResp {
    std::vector<uint8_t> ipc_handle;
    std::vector<uint8_t> nvshmem_unique_id;  // Only rank 0 has meaningful data
};

struct DeepEPSyncReq {
    // Flattened handles: [h0_byte0...h0_byteN, h1_byte0...]
    std::vector<uint8_t> all_handles_flat;
    std::vector<uint8_t> nvshmem_unique_id;
    int                  handle_size;
    std::vector<int32_t> valid_mask;  // 1 if handle exists, 0 otherwise
};

struct DeepEPSyncResp {
    bool        success;
    std::string message;
};

struct DeepEPTestReq {
    int  num_tokens  = 32;    // Reduced to fit in 1GB buffer
    int  hidden      = 2048;  // Reduced to fit in 1GB buffer
    int  num_experts = 64;    // Reduced to fit in 1GB buffer (8 local experts per rank)
    int  num_topk    = 8;
    bool use_fp8     = false;
    int  seed        = 1;
};

struct DeepEPTestResp {
    bool        success;
    double      dispatch_lat_us;
    double      combine_lat_us;
    std::string message;
};

}  // namespace nanodeploy

// --- Serialization Specializations ---
// We place these in the 'spoke' namespace so ADL or template lookup works,
// or simply specialize them in global scope if Spoke allows.
// Spoke serializer is in `spoke` namespace.

namespace spoke {

// Helper for string/vector serialization
template<typename T>
void serialize_vec(std::string& buf, const std::vector<T>& vec)
{
    size_t sz       = vec.size();
    size_t old_size = buf.size();
    buf.resize(old_size + sizeof(size_t) + sz * sizeof(T));
    char* ptr = &buf[old_size];
    std::memcpy(ptr, &sz, sizeof(size_t));
    if (sz > 0)
        std::memcpy(ptr + sizeof(size_t), vec.data(), sz * sizeof(T));
}

template<typename T>
void deserialize_vec(const char*& ptr, std::vector<T>& vec)
{
    size_t sz;
    std::memcpy(&sz, ptr, sizeof(size_t));
    ptr += sizeof(size_t);
    vec.resize(sz);
    if (sz > 0) {
        std::memcpy(vec.data(), ptr, sz * sizeof(T));
        ptr += sz * sizeof(T);
    }
}

template<typename T>
size_t packed_size_vec(const std::vector<T>& vec)
{
    return sizeof(size_t) + vec.size() * sizeof(T);
}

template<typename T>
void pack_to_vec(char*& ptr, const std::vector<T>& vec)
{
    size_t sz = vec.size();
    std::memcpy(ptr, &sz, sizeof(size_t));
    ptr += sizeof(size_t);
    if (sz > 0) {
        std::memcpy(ptr, vec.data(), sz * sizeof(T));
        ptr += sz * sizeof(T);
    }
}

inline void serialize_str(std::string& buf, const std::string& str)
{
    size_t sz       = str.size();
    size_t old_size = buf.size();
    buf.resize(old_size + sizeof(size_t) + sz);
    char* ptr = &buf[old_size];
    std::memcpy(ptr, &sz, sizeof(size_t));
    if (sz > 0)
        std::memcpy(ptr + sizeof(size_t), str.data(), sz);
}

inline void deserialize_str(const char*& ptr, std::string& str)
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

inline size_t packed_size_str(const std::string& str)
{
    return sizeof(size_t) + str.size();
}

inline void pack_to_str(char*& ptr, const std::string& str)
{
    size_t sz = str.size();
    std::memcpy(ptr, &sz, sizeof(size_t));
    ptr += sizeof(size_t);
    if (sz > 0) {
        std::memcpy(ptr, str.data(), sz);
        ptr += sz;
    }
}

// DeepEPInitReq is POD-ish, but has bool. Default serializer uses memcpy which is fine for POD struct.
// But let's be explicit to avoid padding issues if any.
template<>
struct Serializer<nanodeploy::DeepEPInitReq> {
    static std::string pack(const nanodeploy::DeepEPInitReq& obj)
    {
        std::string s;
        s.resize(sizeof(obj));
        std::memcpy(s.data(), &obj, sizeof(obj));
        return s;
    }
    static nanodeploy::DeepEPInitReq unpack(const std::string& data)
    {
        nanodeploy::DeepEPInitReq obj;
        if (data.size() >= sizeof(obj))
            std::memcpy(&obj, data.data(), sizeof(obj));
        return obj;
    }
    static size_t size(const nanodeploy::DeepEPInitReq&)
    {
        return sizeof(nanodeploy::DeepEPInitReq);
    }
    static void packTo(const nanodeploy::DeepEPInitReq& obj, char* buf)
    {
        std::memcpy(buf, &obj, sizeof(nanodeploy::DeepEPInitReq));
    }
    static nanodeploy::DeepEPInitReq unpackFrom(const char* buf, size_t len)
    {
        nanodeploy::DeepEPInitReq obj;
        if (len >= sizeof(obj))
            std::memcpy(&obj, buf, sizeof(obj));
        return obj;
    }
};

template<>
struct Serializer<nanodeploy::DeepEPInitResp> {
    static std::string pack(const nanodeploy::DeepEPInitResp& obj)
    {
        std::string s;
        s.resize(sizeof(bool));
        std::memcpy(s.data(), &obj.success, sizeof(bool));
        serialize_str(s, obj.message);
        return s;
    }
    static nanodeploy::DeepEPInitResp unpack(const std::string& data)
    {
        return unpackFrom(data.data(), data.size());
    }
    static nanodeploy::DeepEPInitResp unpackFrom(const char* buf, size_t len)
    {
        nanodeploy::DeepEPInitResp obj;
        const char*                ptr = buf;
        if (len < sizeof(bool))
            return obj;
        std::memcpy(&obj.success, ptr, sizeof(bool));
        ptr += sizeof(bool);
        deserialize_str(ptr, obj.message);
        return obj;
    }
    static size_t size(const nanodeploy::DeepEPInitResp& obj)
    {
        return sizeof(bool) + packed_size_str(obj.message);
    }
    static void packTo(const nanodeploy::DeepEPInitResp& obj, char* buf)
    {
        char* ptr = buf;
        std::memcpy(ptr, &obj.success, sizeof(bool));
        ptr += sizeof(bool);
        pack_to_str(ptr, obj.message);
    }
};

template<>
struct Serializer<nanodeploy::DeepEPInfoReq> {
    static std::string pack(const nanodeploy::DeepEPInfoReq& obj)
    {
        return std::string((char*)&obj, sizeof(obj));
    }
    static nanodeploy::DeepEPInfoReq unpack(const std::string& data)
    {
        return unpackFrom(data.data(), data.size());
    }
    static nanodeploy::DeepEPInfoReq unpackFrom(const char* buf, size_t len)
    {
        nanodeploy::DeepEPInfoReq obj;
        if (len >= sizeof(obj))
            std::memcpy(&obj, buf, sizeof(obj));
        return obj;
    }
    static size_t size(const nanodeploy::DeepEPInfoReq&)
    {
        return sizeof(nanodeploy::DeepEPInfoReq);
    }
    static void packTo(const nanodeploy::DeepEPInfoReq& obj, char* buf)
    {
        std::memcpy(buf, &obj, sizeof(nanodeploy::DeepEPInfoReq));
    }
};

template<>
struct Serializer<nanodeploy::DeepEPInfoResp> {
    static std::string pack(const nanodeploy::DeepEPInfoResp& obj)
    {
        std::string s;
        serialize_vec(s, obj.ipc_handle);
        serialize_vec(s, obj.nvshmem_unique_id);
        return s;
    }
    static nanodeploy::DeepEPInfoResp unpack(const std::string& data)
    {
        return unpackFrom(data.data(), data.size());
    }
    static nanodeploy::DeepEPInfoResp unpackFrom(const char* buf, size_t len)
    {
        (void)len;  // Unused
        nanodeploy::DeepEPInfoResp obj;
        const char*                ptr = buf;
        deserialize_vec(ptr, obj.ipc_handle);
        deserialize_vec(ptr, obj.nvshmem_unique_id);
        return obj;
    }
    static size_t size(const nanodeploy::DeepEPInfoResp& obj)
    {
        return packed_size_vec(obj.ipc_handle) + packed_size_vec(obj.nvshmem_unique_id);
    }
    static void packTo(const nanodeploy::DeepEPInfoResp& obj, char* buf)
    {
        char* ptr = buf;
        pack_to_vec(ptr, obj.ipc_handle);
        pack_to_vec(ptr, obj.nvshmem_unique_id);
    }
};

template<>
struct Serializer<nanodeploy::DeepEPSyncReq> {
    static std::string pack(const nanodeploy::DeepEPSyncReq& obj)
    {
        std::string s;
        s.resize(sizeof(int));
        std::memcpy(s.data(), &obj.handle_size, sizeof(int));
        serialize_vec(s, obj.all_handles_flat);
        serialize_vec(s, obj.nvshmem_unique_id);
        serialize_vec(s, obj.valid_mask);
        return s;
    }
    static nanodeploy::DeepEPSyncReq unpack(const std::string& data)
    {
        return unpackFrom(data.data(), data.size());
    }
    static nanodeploy::DeepEPSyncReq unpackFrom(const char* buf, size_t len)
    {
        nanodeploy::DeepEPSyncReq obj;
        const char*               ptr = buf;
        if (len < sizeof(int))
            return obj;
        std::memcpy(&obj.handle_size, ptr, sizeof(int));
        ptr += sizeof(int);
        deserialize_vec(ptr, obj.all_handles_flat);
        deserialize_vec(ptr, obj.nvshmem_unique_id);
        deserialize_vec(ptr, obj.valid_mask);
        return obj;
    }
    static size_t size(const nanodeploy::DeepEPSyncReq& obj)
    {
        return sizeof(int) + packed_size_vec(obj.all_handles_flat) + packed_size_vec(obj.nvshmem_unique_id)
               + packed_size_vec(obj.valid_mask);
    }
    static void packTo(const nanodeploy::DeepEPSyncReq& obj, char* buf)
    {
        char* ptr = buf;
        std::memcpy(ptr, &obj.handle_size, sizeof(int));
        ptr += sizeof(int);
        pack_to_vec(ptr, obj.all_handles_flat);
        pack_to_vec(ptr, obj.nvshmem_unique_id);
        pack_to_vec(ptr, obj.valid_mask);
    }
};

template<>
struct Serializer<nanodeploy::DeepEPSyncResp> {
    static std::string pack(const nanodeploy::DeepEPSyncResp& obj)
    {
        std::string s;
        s.resize(sizeof(bool));
        std::memcpy(s.data(), &obj.success, sizeof(bool));
        serialize_str(s, obj.message);
        return s;
    }
    static nanodeploy::DeepEPSyncResp unpack(const std::string& data)
    {
        return unpackFrom(data.data(), data.size());
    }
    static nanodeploy::DeepEPSyncResp unpackFrom(const char* buf, size_t len)
    {
        nanodeploy::DeepEPSyncResp obj;
        const char*                ptr = buf;
        if (len < sizeof(bool))
            return obj;
        std::memcpy(&obj.success, ptr, sizeof(bool));
        ptr += sizeof(bool);
        deserialize_str(ptr, obj.message);
        return obj;
    }
    static size_t size(const nanodeploy::DeepEPSyncResp& obj)
    {
        return sizeof(bool) + packed_size_str(obj.message);
    }
    static void packTo(const nanodeploy::DeepEPSyncResp& obj, char* buf)
    {
        char* ptr = buf;
        std::memcpy(ptr, &obj.success, sizeof(bool));
        ptr += sizeof(bool);
        pack_to_str(ptr, obj.message);
    }
};

template<>
struct Serializer<nanodeploy::DeepEPTestReq> {
    static std::string pack(const nanodeploy::DeepEPTestReq& obj)
    {
        std::string s;
        s.resize(sizeof(obj));
        std::memcpy(s.data(), &obj, sizeof(obj));
        return s;
    }
    static nanodeploy::DeepEPTestReq unpack(const std::string& data)
    {
        return unpackFrom(data.data(), data.size());
    }
    static nanodeploy::DeepEPTestReq unpackFrom(const char* buf, size_t len)
    {
        nanodeploy::DeepEPTestReq obj;
        if (len >= sizeof(obj))
            std::memcpy(&obj, buf, sizeof(obj));
        return obj;
    }
    static size_t size(const nanodeploy::DeepEPTestReq&)
    {
        return sizeof(nanodeploy::DeepEPTestReq);
    }
    static void packTo(const nanodeploy::DeepEPTestReq& obj, char* buf)
    {
        std::memcpy(buf, &obj, sizeof(nanodeploy::DeepEPTestReq));
    }
};

template<>
struct Serializer<nanodeploy::DeepEPTestResp> {
    static std::string pack(const nanodeploy::DeepEPTestResp& obj)
    {
        std::string s;
        s.resize(sizeof(bool) + 2 * sizeof(double));
        char* ptr = s.data();
        std::memcpy(ptr, &obj.success, sizeof(bool));
        ptr += sizeof(bool);
        std::memcpy(ptr, &obj.dispatch_lat_us, sizeof(double));
        ptr += sizeof(double);
        std::memcpy(ptr, &obj.combine_lat_us, sizeof(double));
        ptr += sizeof(double);
        serialize_str(s, obj.message);
        return s;
    }
    static nanodeploy::DeepEPTestResp unpack(const std::string& data)
    {
        return unpackFrom(data.data(), data.size());
    }
    static nanodeploy::DeepEPTestResp unpackFrom(const char* buf, size_t len)
    {
        nanodeploy::DeepEPTestResp obj;
        const char*                ptr = buf;
        if (len < sizeof(bool) + 2 * sizeof(double))
            return obj;
        std::memcpy(&obj.success, ptr, sizeof(bool));
        ptr += sizeof(bool);
        std::memcpy(&obj.dispatch_lat_us, ptr, sizeof(double));
        ptr += sizeof(double);
        std::memcpy(&obj.combine_lat_us, ptr, sizeof(double));
        ptr += sizeof(double);
        deserialize_str(ptr, obj.message);
        return obj;
    }
    static size_t size(const nanodeploy::DeepEPTestResp& obj)
    {
        return sizeof(bool) + 2 * sizeof(double) + packed_size_str(obj.message);
    }
    static void packTo(const nanodeploy::DeepEPTestResp& obj, char* buf)
    {
        char* ptr = buf;
        std::memcpy(ptr, &obj.success, sizeof(bool));
        ptr += sizeof(bool);
        std::memcpy(ptr, &obj.dispatch_lat_us, sizeof(double));
        ptr += sizeof(double);
        std::memcpy(ptr, &obj.combine_lat_us, sizeof(double));
        ptr += sizeof(double);
        pack_to_str(ptr, obj.message);
    }
};

}  // namespace spoke
