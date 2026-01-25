#pragma once

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "serializer.h"
#include "types.h"

// Always include RDMA headers as DLSlime is now required
#include "dlslime/engine/rdma/rdma_endpoint.h"
#include "dlslime/engine/rdma/rdma_future.h"
#include "dlslime/json.hpp"

namespace spoke {

// [New] Zero-Copy Message View
struct MessageView {
    const char* data;
    size_t      size;
    bool        owned;

    MessageView(const std::string& s): data(s.data()), size(s.size()), owned(false) {}
    MessageView(const char* d, size_t s): data(d), size(s), owned(false) {}

    std::string toString() const
    {
        return std::string(data, size);
    }
};

using MethodHandler = std::function<std::string(MessageView)>;

class Actor {
public:
    Actor(const std::string& id, int rx, int tx): actor_id_(id), rx_fd_(rx), tx_fd_(tx)
    {
        int el = 0;
        if (const char* e = std::getenv("NANODEPLOY_ENGINE_LOG_LEVEL"))
            el = std::atoi(e);
        int rl = 0;
        if (const char* r = std::getenv("NANODEPLOY_RUNNER_LOG_LEVEL"))
            rl = std::atoi(r);
        // Enable debug logs if either level is >= 2 (DEBUG)
        debug_mode_ = (el >= 2 || rl >= 2);
    }

    virtual ~Actor()
    {
        is_active_ = false;
        if (rdma_thread_.joinable()) {
            rdma_thread_.join();
        }
    }

    void run()
    {
        while (is_active_) {
            PipeHeader hdr;
            if (read(rx_fd_, &hdr, sizeof(PipeHeader)) <= 0)
                break;

            std::string body;
            body.resize(hdr.data_size);
            if (hdr.data_size > 0) {
                size_t total = 0;
                while (total < hdr.data_size) {
                    int r = read(rx_fd_, body.data() + total, hdr.data_size - total);
                    if (r <= 0)
                        break;
                    total += r;
                }
            }

            {
                std::lock_guard<std::mutex> lk(actor_mtx_);
                onMessage(hdr, body);
            }
        }
    }

    // [New] Push unsolicited message to Client (Stream)
    // Uses RDMA if available, else Socket
    bool pushToClient(uint32_t slot_id, const std::string& data)
    {
        // 1. Try RDMA
        if (rdma_ep_ && client_remote_addr_ > 0) {
            try {
                size_t total_size = 12 + data.size();
                if (total_size > rdma_buf_size_) {
                    std::cerr << "[Actor] Push RDMA buffer too small!" << std::endl;
                    // TODO: Realloc? But realloc needs Client cooperation (RPC).
                    // Push should be fast. Fallback to socket?
                    goto push_socket;
                }

                uint32_t* resp_ptr = (uint32_t*)rdma_buffer_;
                resp_ptr[0]        = static_cast<uint32_t>(Action::kStreamPush);
                resp_ptr[1]        = slot_id;
                resp_ptr[2]        = (uint32_t)data.size();

                if (!data.empty()) {
                    memcpy(rdma_buffer_ + 12, data.data(), data.size());
                }

                std::vector<dlslime::assign_tuple_t> chunks;
                chunks.emplace_back((uintptr_t)rdma_buffer_, client_remote_addr_, 0, 0, total_size);

                auto write_fut = rdma_ep_->writeWithImm(chunks, slot_id, nullptr);  // imm_data=slot_id? or dummy?
                // Note: writeWithImm(..., imm_data) -> imm_data is usually 32-bit.
                // Client side immRecv doesn't return imm_data in future?
                // DLSlime immRecv() returns Future. wait() returns void.
                // But DLSlime RDMAEndpoint usually doesn't expose the Imm data value to user easily in current API?
                // Wait, in Client::rdmaRecvLoop we read from buffer. The Imm is just a signal.
                // The data is in the buffer.

                write_fut->wait();
                return true;
            }
            catch (...) {
                // Fallback
            }
        }

    push_socket:
        // 2. Fallback to Socket
        // Response header: Action=kStreamPush, Seq=slot_id
        PipeHeader                  resp_hdr{Action::kStreamPush, slot_id, (uint32_t)data.size()};
        std::lock_guard<std::mutex> lk(actor_mtx_);  // Reuse actor mutex for tx_fd_ safety?
        if (write(tx_fd_, &resp_hdr, sizeof(PipeHeader)) < 0)
            return false;
        if (!data.empty()) {
            if (write(tx_fd_, data.data(), data.size()) < 0)
                return false;
        }
        return true;
    }

protected:
    void registerMethod(Action action, MethodHandler handler)
    {
        std::lock_guard<std::mutex> lk(actor_mtx_);
        handlers_[action] = std::move(handler);
    }

private:
    // Unified logic: Returns response string
    std::string processRequest(const PipeHeader& hdr, MessageView body)
    {
        if (hdr.action == Action::kRdmaRealloc) {
            try {
                size_t requested_size = std::stoull(body.toString());
                if (requested_size > rdma_buf_size_) {
                    reallocRDMA(requested_size);
                }
                // Return new info
                auto      local_info      = rdma_ep_->endpointInfo();
                uintptr_t addr            = (uintptr_t)rdma_buffer_;
                local_info["buffer_addr"] = addr;
                local_info["buffer_size"] = rdma_buf_size_;
                local_info["rkey"]        = local_info["mr_info"][std::to_string(addr)]["rkey"];
                return local_info.dump();
            }
            catch (...) {
                std::cerr << "[Actor] Realloc failed" << std::endl;
                return "";
            }
        }

        auto it = handlers_.find(hdr.action);
        if (it != handlers_.end()) {
            return it->second(body);
        }
        return "";
    }

    void onMessage(const PipeHeader& hdr, const std::string& body)
    {
        if (hdr.action == Action::kStop) {
            fflush(stdout);
            fflush(stderr);
            _exit(0);
        }

        if (hdr.action == Action::kInitRDMA) {
            if (debug_mode_)
                std::cout << "[Actor] Received kInitRDMA for Seq: " << hdr.seq_id << std::endl;
            std::string res = setupRDMA(body);
            PipeHeader  resp_hdr{Action::kInit, hdr.seq_id, (uint32_t)res.size()};
            if (write(tx_fd_, &resp_hdr, sizeof(PipeHeader)) < 0) {
                std::cerr << "[Actor] Failed to write response header" << std::endl;
            }
            if (!res.empty()) {
                if (write(tx_fd_, res.data(), res.size()) < 0) {
                    std::cerr << "[Actor] Failed to write response body" << std::endl;
                }
            }
            if (debug_mode_)
                std::cout << "[Actor] kInitRDMA response written for Seq: " << hdr.seq_id << std::endl;
            return;
        }

        // Use unified processing
        std::string res = processRequest(hdr, MessageView(body));
        if (debug_mode_)
            std::cout << "[Actor] Processed request. Result size: " << res.size() << ". Writing response..."
                      << std::endl;

        PipeHeader resp_hdr{Action::kInit, hdr.seq_id, (uint32_t)res.size()};
        if (write(tx_fd_, &resp_hdr, sizeof(PipeHeader)) < 0) {
            std::cerr << "[Actor] Failed to write response header" << std::endl;
        }
        if (!res.empty()) {
            if (write(tx_fd_, res.data(), res.size()) < 0) {
                std::cerr << "[Actor] Failed to write response body" << std::endl;
            }
        }
        if (debug_mode_)
            std::cout << "[Actor] Response written to FD " << tx_fd_ << std::endl;
    }

    std::string setupRDMA(const std::string& remote_info_str)
    {
        try {
            rdma_ep_         = std::make_shared<dlslime::RDMAEndpoint>();
            auto remote_json = dlslime::json::parse(remote_info_str);
            rdma_ep_->connect(remote_json);

            // Store Client's Remote Info for replying
            client_remote_addr_ = remote_json["buffer_addr"].get<uintptr_t>();
            client_remote_size_ = remote_json["buffer_size"].get<size_t>();

            reallocRDMA(4 * 1024 * 1024);
            rdma_thread_ = std::thread(&Actor::rdmaPollLoop, this);

            auto      local_info      = rdma_ep_->endpointInfo();
            uintptr_t addr            = (uintptr_t)rdma_buffer_;
            local_info["buffer_addr"] = addr;
            local_info["buffer_size"] = rdma_buf_size_;
            local_info["rkey"]        = local_info["mr_info"][std::to_string(addr)]["rkey"];
            return local_info.dump();
        }
        catch (const std::exception& e) {
            std::cerr << "[Actor] setupRDMA failed: " << e.what() << std::endl;
            return "";
        }
    }

    void reallocRDMA(size_t new_size)
    {
        if (rdma_buffer_) {
            // Deregister old MR? DLSlime handles it if we destroy pool or overwrite.
            // But simple pointer delete is risky if MR is live.
            // For now, assume single realloc or leak old MR handle (not memory).
            delete[] rdma_buffer_;
        }
        rdma_buf_size_ = new_size;
        rdma_buffer_   = new char[rdma_buf_size_];  // Use posix_memalign for RDMA?
        uintptr_t addr = (uintptr_t)rdma_buffer_;
        rdma_ep_->registerOrAccessMemoryRegion(addr, addr, 0, rdma_buf_size_);
        std::cout << "[Actor] Reallocated RDMA Buffer to " << (rdma_buf_size_ / 1024 / 1024) << "MB" << std::endl;
    }

    void rdmaPollLoop()
    {
        std::cout << "[Actor] RDMA Poll Loop Started. EP: " << rdma_ep_.get() << std::endl;
        while (is_active_) {
            try {
                // std::cout << "[Actor] Waiting for RDMA Imm..." << std::endl;
                auto fut = rdma_ep_->immRecv();
                fut->wait();
                if (!is_active_)
                    break;

                uint32_t*  ptr = (uint32_t*)rdma_buffer_;
                PipeHeader hdr;
                hdr.action    = static_cast<Action>(ptr[0]);
                hdr.seq_id    = ptr[1];
                hdr.data_size = ptr[2];

                if (debug_mode_)
                    std::cout << "⚡ [Actor] RDMA Request: Action=" << ptr[0] << ", Seq=" << hdr.seq_id << std::endl;

                MessageView body(rdma_buffer_ + 12, hdr.data_size);

                std::string res;
                {
                    std::lock_guard<std::mutex> lk(actor_mtx_);
                    res = processRequest(hdr, body);
                }

                // SEND REPLY VIA RDMA TO CLIENT
                // We use the same buffer for Input and Output for simplicity (Ping-Pong)?
                // No, we should use a separate offset or overwrite input if processed.
                // Overwriting is safe since we consumed inputs.

                // Response Format: [Action][SeqID][Size][Body]
                // We can reuse the start of buffer.
                uint32_t* resp_ptr = (uint32_t*)rdma_buffer_;
                resp_ptr[0]        = static_cast<uint32_t>(Action::kInit);  // Generic OK
                resp_ptr[1]        = hdr.seq_id;
                resp_ptr[2]        = (uint32_t)res.size();

                if (!res.empty()) {
                    memcpy(rdma_buffer_ + 12, res.data(), res.size());
                }

                // RDMA Write Back
                size_t total_size = 12 + res.size();

                std::vector<dlslime::assign_tuple_t> chunks;
                // Remote Addr: Client expects response at offset 0?
                // Client `writeWithImm` sends to Actor offset 0.
                // Client `immRecv`... doesn't specify offset, it just receives "Imm Data".
                // But data needs to be written somewhere.
                // Client is listening. We must Write to Client's buffer first.
                // Where? Client defaults to offset 0 too.
                chunks.emplace_back((uintptr_t)rdma_buffer_, client_remote_addr_, 0, 0, total_size);

                // Write with Imm (Signal completion to Client)
                auto write_fut = rdma_ep_->writeWithImm(chunks, hdr.seq_id, nullptr);
                write_fut->wait();
            }
            catch (const std::exception& e) {
                if (is_active_) {
                    std::cerr << "[Actor] RDMA Poll Loop Exception: " << e.what() << std::endl;
                }
                if (!is_active_)
                    break;
            }
            catch (...) {
                if (is_active_) {
                    std::cerr << "[Actor] RDMA Poll Loop Unknown Exception" << std::endl;
                }
                if (!is_active_)
                    break;
            }
        }
        std::cout << "[Actor] RDMA Poll Loop Exiting" << std::endl;
    }

    std::string actor_id_;
    int         rx_fd_, tx_fd_;
    bool        is_active_ = true;

    std::shared_ptr<dlslime::RDMAEndpoint> rdma_ep_;
    char*                                  rdma_buffer_   = nullptr;
    size_t                                 rdma_buf_size_ = 0;

    // Remote Client Info
    uintptr_t client_remote_addr_ = 0;
    size_t    client_remote_size_ = 0;

    std::thread rdma_thread_;
    std::mutex  actor_mtx_;

    std::unordered_map<Action, MethodHandler> handlers_;
    bool                                      debug_mode_ = false;
};

using ActorFactoryFunc = std::function<std::unique_ptr<Actor>(std::string, int, int)>;

class ActorFactory {
public:
    static ActorFactory& instance()
    {
        static ActorFactory inst;
        return inst;
    }
    void registerType(const std::string& type, ActorFactoryFunc func)
    {
        creators_[type] = func;
    }
    std::unique_ptr<Actor> create(const std::string& type, const std::string& id, int rx, int tx)
    {
        if (creators_.count(type))
            return creators_[type](id, rx, tx);
        return nullptr;
    }

    std::vector<std::string> getRegisteredTypes() const
    {
        std::vector<std::string> types;
        for (const auto& [type, _] : creators_) {
            types.push_back(type);
        }
        return types;
    }

private:
    std::map<std::string, ActorFactoryFunc> creators_;
};

#define SPOKE_REGISTER_ACTOR(Type, ClassName)                                                                          \
    struct Reg_##ClassName {                                                                                           \
        Reg_##ClassName()                                                                                              \
        {                                                                                                              \
            spoke::ActorFactory::instance().registerType(                                                              \
                Type, [](std::string id, int rx, int tx) { return std::make_unique<ClassName>(id, rx, tx); });         \
        }                                                                                                              \
    } reg_##ClassName##_inst_;

#define SPOKE_METHOD(ClassName, MethodName, ActionID, ReqType, RespType)                                               \
    struct Reg_##MethodName {                                                                                          \
        Reg_##MethodName(ClassName* obj)                                                                               \
        {                                                                                                              \
            obj->registerMethod(ActionID, [obj](spoke::MessageView raw) -> std::string {                               \
                ReqType  req  = spoke::UnpackFrom<ReqType>(raw.data, raw.size);                                        \
                RespType resp = obj->MethodName(req);                                                                  \
                return spoke::Pack(resp);                                                                              \
            });                                                                                                        \
        }                                                                                                              \
    } reg_##MethodName##_inst_{this};                                                                                  \
    RespType MethodName(ReqType val)

}  // namespace spoke
