#pragma once
#include "spoke/csrc/serializer.h"
#include <cstring>
#include <string>
#include <vector>

namespace nanodeploy {

struct EngineInitReq {
    char config_path[256]  = "";
    int  tp                = 1;
    int  pp                = 1;
    int  dp                = 1;
    char hub_ip[64]        = "127.0.0.1";
    int  hub_port          = 8888;
    int  attention_tp      = 1;
    int  attention_dp      = 1;
    int  attention_sp      = 1;
    int  ffn_tp            = 1;
    int  ffn_dp            = 1;
    int  ffn_ep            = 1;
    bool enable_rdma       = true;
    bool enable_cuda_graph = false;
};

struct EngineAddReq {
    std::vector<int> prompt_ids;
    int              max_new_tokens;
    uint32_t         slot_id;  // [New] Client-side mailbox ID
};

// [New] Unsolicited Stream Message
struct StreamToken {
    uint64_t         seq_id;
    std::vector<int> tokens;
    bool             finished;
};

struct EngineStepResp {
    bool finished;
};

struct EngineGetFinishedResp {
    std::vector<std::pair<uint64_t, std::vector<int>>> results;
};

}  // namespace nanodeploy

namespace spoke {

// EngineInitReq - POD type, use default Serializer (memcpy)
// Already handled by default template

// EngineAddReq - has vector, need custom
template<>
struct Serializer<nanodeploy::EngineAddReq> {
    static std::string pack(const nanodeploy::EngineAddReq& v)
    {
        std::string s;
        size_t      len = v.prompt_ids.size();
        s.append((char*)&len, sizeof(len));
        s.append((char*)v.prompt_ids.data(), len * sizeof(int));
        s.append((char*)&v.max_new_tokens, sizeof(int));
        s.append((char*)&v.slot_id, sizeof(uint32_t));
        return s;
    }
    static nanodeploy::EngineAddReq unpack(const std::string& s)
    {
        nanodeploy::EngineAddReq v;
        size_t                   len;
        const char*              ptr = s.data();
        std::memcpy(&len, ptr, sizeof(len));
        ptr += sizeof(len);
        v.prompt_ids.resize(len);
        std::memcpy(v.prompt_ids.data(), ptr, len * sizeof(int));
        ptr += len * sizeof(int);
        std::memcpy(&v.max_new_tokens, ptr, sizeof(int));
        ptr += sizeof(int);
        if ((size_t)(ptr - s.data()) < s.size()) {
            std::memcpy(&v.slot_id, ptr, sizeof(uint32_t));
        }
        else {
            v.slot_id = 0;
        }
        return v;
    }
    static size_t size(const nanodeploy::EngineAddReq& v)
    {
        return sizeof(size_t) + v.prompt_ids.size() * sizeof(int) + sizeof(int) + sizeof(uint32_t);
    }
    static void packTo(const nanodeploy::EngineAddReq& v, char* buf)
    {
        size_t len = v.prompt_ids.size();
        std::memcpy(buf, &len, sizeof(len));
        buf += sizeof(len);
        std::memcpy(buf, v.prompt_ids.data(), len * sizeof(int));
        buf += len * sizeof(int);
        std::memcpy(buf, &v.max_new_tokens, sizeof(int));
        buf += sizeof(int);
        std::memcpy(buf, &v.slot_id, sizeof(uint32_t));
    }
    static nanodeploy::EngineAddReq unpackFrom(const char* buf, size_t len)
    {
        return unpack(std::string(buf, len));
    }
};

// StreamToken - has vector, need custom
template<>
struct Serializer<nanodeploy::StreamToken> {
    static std::string pack(const nanodeploy::StreamToken& v)
    {
        std::string s;
        s.append((char*)&v.seq_id, sizeof(v.seq_id));
        size_t len = v.tokens.size();
        s.append((char*)&len, sizeof(len));
        s.append((char*)v.tokens.data(), len * sizeof(int));
        s.append((char*)&v.finished, sizeof(bool));
        return s;
    }
    static nanodeploy::StreamToken unpack(const std::string& s)
    {
        nanodeploy::StreamToken v;
        const char*             ptr = s.data();
        std::memcpy(&v.seq_id, ptr, sizeof(v.seq_id));
        ptr += sizeof(v.seq_id);
        size_t len;
        std::memcpy(&len, ptr, sizeof(len));
        ptr += sizeof(len);
        v.tokens.resize(len);
        std::memcpy(v.tokens.data(), ptr, len * sizeof(int));
        ptr += len * sizeof(int);
        std::memcpy(&v.finished, ptr, sizeof(bool));
        return v;
    }
    static size_t size(const nanodeploy::StreamToken& v)
    {
        return sizeof(v.seq_id) + sizeof(size_t) + v.tokens.size() * sizeof(int) + sizeof(bool);
    }
    static void packTo(const nanodeploy::StreamToken& v, char* buf)
    {
        std::memcpy(buf, &v.seq_id, sizeof(v.seq_id));
        buf += sizeof(v.seq_id);
        size_t len = v.tokens.size();
        std::memcpy(buf, &len, sizeof(len));
        buf += sizeof(len);
        std::memcpy(buf, v.tokens.data(), len * sizeof(int));
        buf += len * sizeof(int);
        std::memcpy(buf, &v.finished, sizeof(bool));
    }
    static nanodeploy::StreamToken unpackFrom(const char* buf, size_t len)
    {
        return unpack(std::string(buf, len));
    }
};

// EngineGetFinishedResp - has nested vectors
template<>
struct Serializer<nanodeploy::EngineGetFinishedResp> {
    static std::string pack(const nanodeploy::EngineGetFinishedResp& v)
    {
        std::string s;
        size_t      count = v.results.size();
        s.append((char*)&count, sizeof(count));
        for (const auto& pair : v.results) {
            s.append((char*)&pair.first, sizeof(uint64_t));
            size_t vec_len = pair.second.size();
            s.append((char*)&vec_len, sizeof(size_t));
            s.append((char*)pair.second.data(), vec_len * sizeof(int));
        }
        return s;
    }
    static nanodeploy::EngineGetFinishedResp unpack(const std::string& s)
    {
        nanodeploy::EngineGetFinishedResp v;
        const char*                       ptr = s.data();
        size_t                            count;
        std::memcpy(&count, ptr, sizeof(count));
        ptr += sizeof(count);
        v.results.resize(count);
        for (size_t i = 0; i < count; ++i) {
            std::memcpy(&v.results[i].first, ptr, sizeof(uint64_t));
            ptr += sizeof(uint64_t);
            size_t vec_len;
            std::memcpy(&vec_len, ptr, sizeof(size_t));
            ptr += sizeof(size_t);
            v.results[i].second.resize(vec_len);
            std::memcpy(v.results[i].second.data(), ptr, vec_len * sizeof(int));
            ptr += vec_len * sizeof(int);
        }
        return v;
    }
    static size_t size(const nanodeploy::EngineGetFinishedResp& v)
    {
        size_t sz = sizeof(size_t);
        for (const auto& p : v.results) {
            sz += sizeof(uint64_t) + sizeof(size_t) + p.second.size() * sizeof(int);
        }
        return sz;
    }
    static void packTo(const nanodeploy::EngineGetFinishedResp& v, char* buf)
    {
        std::string s = pack(v);
        std::memcpy(buf, s.data(), s.size());
    }
    static nanodeploy::EngineGetFinishedResp unpackFrom(const char* buf, size_t len)
    {
        return unpack(std::string(buf, len));
    }
};

}  // namespace spoke
