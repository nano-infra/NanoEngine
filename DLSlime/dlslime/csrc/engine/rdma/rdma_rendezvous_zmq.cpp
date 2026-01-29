#ifdef BUILD_RDMA_RENDEZVOUS_ZMQ

#include "rdma_rendezvous_zmq.h"

#include <zmq.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include "dlslime/csrc/logging.h"

namespace dlslime {

namespace {

std::string ensure_tcp_addr(const std::string& addr)
{
    if (addr.find("://") != std::string::npos)
        return addr;
    return "tcp://" + addr;
}

std::string bind_addr(const std::string& addr)
{
    std::string a = ensure_tcp_addr(addr);
    // 0.0.0.0:port -> *:port for ZMQ bind (keep colon and port; "0.0.0.0:" is 8 chars)
    size_t pos = a.find("0.0.0.0:");
    if (pos != std::string::npos)
        a.replace(pos, 8, "*:");
    else if (a.find("0.0.0.0") != std::string::npos)
        a.replace(a.find("0.0.0.0"), 8, "*");
    return a;
}

// Parse "tcp://*:50001" or "*:50001" -> port number. Returns -1 on failure.
int parse_port_from_bind_addr(const std::string& addr)
{
    std::string a     = ensure_tcp_addr(addr);
    size_t      colon = a.rfind(':');
    if (colon == std::string::npos || colon + 1 >= a.size())
        return -1;
    const char* p    = a.c_str() + colon + 1;
    char*       end  = nullptr;
    long        port = std::strtol(p, &end, 10);
    if (end == p || *end != '\0' || port <= 0 || port > 65535)
        return -1;
    return static_cast<int>(port);
}

// Create TCP listen socket with SO_REUSEADDR so port can be reused immediately after close.
// Returns fd or -1 on error. Caller must close(fd) on failure; on success ZMQ takes ownership via ZMQ_USE_FD.
int create_listen_fd_with_reuseaddr(const std::string& zmq_bind_addr)
{
    int port = parse_port_from_bind_addr(zmq_bind_addr);
    if (port <= 0)
        return -1;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0)
        return -1;
    int on = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on)) != 0) {
        close(fd);
        return -1;
    }
    struct sockaddr_in sa = {};
    sa.sin_family         = AF_INET;
    sa.sin_port           = htons(static_cast<uint16_t>(port));
    sa.sin_addr.s_addr    = INADDR_ANY;
    if (bind(fd, reinterpret_cast<struct sockaddr*>(&sa), sizeof(sa)) != 0) {
        close(fd);
        return -1;
    }
    if (listen(fd, 100) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

// Receive multipart: [method, body]. Returns (method, body) or throws.
std::pair<std::string, std::string> recv_multipart(void* socket)
{
    std::string method;
    std::string body;
    zmq_msg_t   msg;
    zmq_msg_init(&msg);
    if (zmq_msg_recv(&msg, socket, 0) < 0) {
        zmq_msg_close(&msg);
        throw std::runtime_error("zmq recv method failed");
    }
    int         more = zmq_msg_more(&msg);
    const char* data = static_cast<const char*>(zmq_msg_data(&msg));
    size_t      size = zmq_msg_size(&msg);
    method.assign(data, size);
    zmq_msg_close(&msg);
    if (!more)
        return {method, ""};
    zmq_msg_init(&msg);
    if (zmq_msg_recv(&msg, socket, 0) < 0) {
        zmq_msg_close(&msg);
        throw std::runtime_error("zmq recv body failed");
    }
    data = static_cast<const char*>(zmq_msg_data(&msg));
    size = zmq_msg_size(&msg);
    body.assign(data, size);
    zmq_msg_close(&msg);
    return {method, body};
}

void send_multipart(void* socket, const std::string& part1, const std::string& part2)
{
    if (zmq_send(socket, part1.data(), part1.size(), ZMQ_SNDMORE) < 0)
        throw std::runtime_error("zmq send part1 failed");
    if (zmq_send(socket, part2.data(), part2.size(), 0) < 0)
        throw std::runtime_error("zmq send part2 failed");
}

void send_response(void* socket, const std::string& body)
{
    if (zmq_send(socket, body.data(), body.size(), 0) < 0)
        throw std::runtime_error("zmq send response failed");
}

void send_error(void* socket, const std::string& msg)
{
    std::string err = "ERROR";
    if (zmq_send(socket, err.data(), err.size(), ZMQ_SNDMORE) < 0)
        throw std::runtime_error("zmq send error tag failed");
    if (zmq_send(socket, msg.data(), msg.size(), 0) < 0)
        throw std::runtime_error("zmq send error msg failed");
}

}  // namespace

// ---------------------------------------------------------------------------
// RdmaRendezvousBackend
// ---------------------------------------------------------------------------

RdmaRendezvousBackend::RdmaRendezvousBackend(std::shared_ptr<RDMAEndpoint> endpoint): endpoint_(std::move(endpoint)) {}

json RdmaRendezvousBackend::get_endpoint_info()
{
    if (!endpoint_)
        throw std::runtime_error("get_endpoint_info: broker backend has no endpoint");
    return endpoint_->endpointInfo();
}

json RdmaRendezvousBackend::handshake_request(const json& req)
{
    if (!endpoint_)
        throw std::runtime_error("handshake_request: broker backend has no endpoint");
    json ep = req.contains("endpoint_info") ? req["endpoint_info"] : req;
    endpoint_->connect(ep);
    return get_endpoint_info();
}

json RdmaRendezvousBackend::register_shared_buffer(const json& req)
{
    std::string buffer_id = req.value("buffer_id", std::string{});
    json        mr_info;
    mr_info["mr_key"] = req.value("mr_key", static_cast<uint64_t>(0));
    mr_info["addr"]   = req.value("addr", static_cast<uint64_t>(0));
    mr_info["rkey"]   = req.value("rkey", static_cast<uint32_t>(0));
    mr_info["length"] = req.value("length", static_cast<uint64_t>(0));
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_buffers_[buffer_id] = mr_info;
        cond_.notify_all();
    }
    return json{};
}

json RdmaRendezvousBackend::get_local_buffer(const json& req)
{
    std::string                 buffer_id = req.value("buffer_id", std::string{});
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = local_buffers_.find(buffer_id);
    if (it == local_buffers_.end())
        throw std::runtime_error("GetLocalBuffer: buffer_id not published: " + buffer_id);
    return it->second;
}

void RdmaRendezvousBackend::register_local_buffer(const std::string& buffer_id, const json& mr_info)
{
    std::lock_guard<std::mutex> lock(mutex_);
    local_buffers_[buffer_id] = mr_info;
}

std::pair<std::string, json> RdmaRendezvousBackend::get_pending_shared_buffer(std::string* buffer_id,
                                                                              double       timeout_sec)
{
    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_sec > 0 ? timeout_sec : 30.0);
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
        if (buffer_id == nullptr || buffer_id->empty()) {
            if (!pending_buffers_.empty()) {
                auto        it  = pending_buffers_.begin();
                std::string bid = it->first;
                json        j   = it->second;
                pending_buffers_.erase(it);
                return {bid, j};
            }
        }
        else {
            auto it = pending_buffers_.find(*buffer_id);
            if (it != pending_buffers_.end()) {
                json j = it->second;
                pending_buffers_.erase(it);
                return {*buffer_id, j};
            }
        }
        if (timeout_sec > 0 && std::chrono::steady_clock::now() >= deadline)
            return {"", json::object()};
        cond_.wait_for(lock, std::chrono::milliseconds(100));
    }
}

void RdmaRendezvousBackend::lazy_handshake_request(const std::string& initiator_id,
                                                   const std::string& peer_id,
                                                   const json&        my_endpoint_info,
                                                   const std::string& initiator_broker_addr)
{
    std::lock_guard<std::mutex> lock(mutex_);
    pending_lazy_requests_[peer_id].emplace_back(initiator_id, my_endpoint_info, initiator_broker_addr);
    cond_.notify_all();
}

json RdmaRendezvousBackend::get_pending_lazy_handshakes(const std::string& peer_id)
{
    std::lock_guard<std::mutex> lock(mutex_);
    json                        arr = json::array();
    auto                        it  = pending_lazy_requests_.find(peer_id);
    if (it != pending_lazy_requests_.end()) {
        for (auto& t : it->second)
            arr.push_back(json{{"initiator_id", std::get<0>(t)},
                               {"endpoint_info", std::get<1>(t)},
                               {"initiator_broker_addr", std::get<2>(t)}});
        pending_lazy_requests_.erase(it);
    }
    return arr;
}

void RdmaRendezvousBackend::lazy_handshake_response(const std::string& initiator_id,
                                                    const std::string& peer_id,
                                                    const json&        my_endpoint_info)
{
    std::lock_guard<std::mutex> lock(mutex_);
    pending_lazy_responses_[{initiator_id, peer_id}] = my_endpoint_info;
    cond_.notify_all();
}

json RdmaRendezvousBackend::get_lazy_handshake_response(const std::string& initiator_id,
                                                        const std::string& peer_id,
                                                        double             timeout_sec)
{
    auto                         key = std::make_pair(initiator_id, peer_id);
    std::unique_lock<std::mutex> lock(mutex_);
    auto                         it = pending_lazy_responses_.find(key);
    if (it != pending_lazy_responses_.end()) {
        json j = std::move(it->second);
        pending_lazy_responses_.erase(it);
        return j;
    }
    // Non-blocking when timeout_sec <= 0: broker returns immediately so REP can serve peer.
    if (timeout_sec <= 0)
        return json::object();
    auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_sec);
    while (true) {
        if (std::chrono::steady_clock::now() >= deadline)
            return json::object();
        cond_.wait_for(lock, std::chrono::milliseconds(100));
        it = pending_lazy_responses_.find(key);
        if (it != pending_lazy_responses_.end()) {
            json j = std::move(it->second);
            pending_lazy_responses_.erase(it);
            return j;
        }
    }
}

void RdmaRendezvousBackend::register_connect_request(const std::string& my_id,
                                                     const std::string& peer_id,
                                                     const json&        endpoint_info)
{
    std::lock_guard<std::mutex> lock(mutex_);
    symmetric_connects_[peer_id][my_id] = endpoint_info;
    cond_.notify_all();
}

json RdmaRendezvousBackend::get_peer_info(const std::string& peer_id, const std::string& other_id, double timeout_sec)
{
    std::unique_lock<std::mutex> lock(mutex_);
    auto                         pit = symmetric_connects_.find(peer_id);
    if (pit != symmetric_connects_.end()) {
        auto oit = pit->second.find(other_id);
        if (oit != pit->second.end()) {
            json j = std::move(oit->second);
            pit->second.erase(oit);
            if (pit->second.empty())
                symmetric_connects_.erase(pit);
            return j;
        }
    }
    if (timeout_sec <= 0)
        return json::object();
    auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_sec);
    while (true) {
        if (std::chrono::steady_clock::now() >= deadline)
            return json::object();
        cond_.wait_for(lock, std::chrono::milliseconds(100));
        pit = symmetric_connects_.find(peer_id);
        if (pit != symmetric_connects_.end()) {
            auto oit = pit->second.find(other_id);
            if (oit != pit->second.end()) {
                json j = std::move(oit->second);
                pit->second.erase(oit);
                if (pit->second.empty())
                    symmetric_connects_.erase(pit);
                return j;
            }
        }
    }
}

void RdmaRendezvousBackend::register_buffer(const std::string& endpoint_id,
                                            const std::string& buffer_id,
                                            const json&        mr_info)
{
    std::lock_guard<std::mutex> lock(mutex_);
    endpoint_buffers_[endpoint_id][buffer_id] = mr_info;
    cond_.notify_all();
}

json RdmaRendezvousBackend::get_remote_buffer(const std::string& remote_endpoint_id,
                                              const std::string& buffer_id,
                                              double             timeout_sec)
{
    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_sec > 0 ? timeout_sec : 30.0);
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
        auto eit = endpoint_buffers_.find(remote_endpoint_id);
        if (eit != endpoint_buffers_.end()) {
            auto bit = eit->second.find(buffer_id);
            if (bit != eit->second.end()) {
                return bit->second;
            }
        }
        if (timeout_sec > 0 && std::chrono::steady_clock::now() >= deadline)
            return json::object();
        cond_.wait_for(lock, std::chrono::milliseconds(100));
    }
}

// ---------------------------------------------------------------------------

void RdmaRendezvousBackend::trigger_ensure_peer(const std::string& client_addr)
{
    if (ensure_peer_cb_)
        ensure_peer_cb_(client_addr);
}

// ---------------------------------------------------------------------------
// ZmqRendezvousServer
// ---------------------------------------------------------------------------

ZmqRendezvousServer::ZmqRendezvousServer(std::shared_ptr<RdmaRendezvousBackend> backend, const std::string& addr):
    backend_(std::move(backend)), addr_(bind_addr(addr))
{
}

ZmqRendezvousServer::~ZmqRendezvousServer()
{
    stop();
}

void ZmqRendezvousServer::start()
{
    if (thread_)
        return;
    stop_.store(false);
    thread_ = std::make_unique<std::thread>(&ZmqRendezvousServer::run, this);
}

void ZmqRendezvousServer::stop()
{
    stop_.store(true);
    if (thread_ && thread_->joinable())
        thread_->join();
    thread_.reset();
    if (zmq_socket_) {
        zmq_close(zmq_socket_);
        zmq_socket_ = nullptr;
    }
    if (zmq_ctx_) {
        zmq_ctx_term(zmq_ctx_);
        zmq_ctx_ = nullptr;
    }
}

void ZmqRendezvousServer::run()
{
    zmq_ctx_    = zmq_ctx_new();
    zmq_socket_ = zmq_socket(zmq_ctx_, ZMQ_REP);
    if (!zmq_socket_) {
        SLIME_LOG_ERROR("ZMQ REP socket create failed");
        return;
    }
    if (zmq_bind(zmq_socket_, addr_.c_str()) != 0) {
        SLIME_LOG_ERROR("ZMQ bind failed: ", addr_.c_str());
        zmq_close(zmq_socket_);
        zmq_socket_ = nullptr;
        zmq_ctx_term(zmq_ctx_);
        zmq_ctx_ = nullptr;
        return;
    }
    int timeout_ms = 1000;
    zmq_setsockopt(zmq_socket_, ZMQ_RCVTIMEO, &timeout_ms, sizeof(timeout_ms));
    while (!stop_.load()) {
        bool request_received = false;
        try {
            auto [method, body_str] = recv_multipart(zmq_socket_);
            request_received        = true;  // client is waiting for a response
            json req                = body_str.empty() ? json::object() : json::parse(body_str);
            json resp;
            if (method == "GetEndpointInfo") {
                resp = backend_->get_endpoint_info();
            }
            else if (method == "Handshake") {
                resp = backend_->handshake_request(req);
            }
            else if (method == "RegisterSharedBuffer") {
                resp = backend_->register_shared_buffer(req);
            }
            else if (method == "GetLocalBuffer") {
                resp = backend_->get_local_buffer(req);
            }
            else if (method == "LazyHandshakeRequest") {
                backend_->lazy_handshake_request(req.value("initiator_id", std::string{}),
                                                 req.value("peer_id", std::string{}),
                                                 req.value("my_endpoint_info", json::object()),
                                                 req.value("initiator_broker_addr", std::string{}));
                resp = json::object();
            }
            else if (method == "GetPendingLazyHandshakes") {
                resp = backend_->get_pending_lazy_handshakes(req.value("peer_id", std::string{}));
            }
            else if (method == "LazyHandshakeResponse") {
                backend_->lazy_handshake_response(req.value("initiator_id", std::string{}),
                                                  req.value("peer_id", std::string{}),
                                                  req.value("my_endpoint_info", json::object()));
                resp = json::object();
            }
            else if (method == "GetLazyHandshakeResponse") {
                resp = backend_->get_lazy_handshake_response(req.value("initiator_id", std::string{}),
                                                             req.value("peer_id", std::string{}),
                                                             req.value("timeout_sec", 30.0));
            }
            else if (method == "RegisterConnectRequest") {
                backend_->register_connect_request(req.value("my_id", std::string{}),
                                                   req.value("peer_id", std::string{}),
                                                   req.value("endpoint_info", json::object()));
                resp = json::object();
            }
            else if (method == "GetPeerInfo") {
                resp = backend_->get_peer_info(req.value("peer_id", std::string{}),
                                               req.value("other_id", std::string{}),
                                               req.value("timeout_sec", 30.0));
            }
            else if (method == "RegisterBuffer") {
                backend_->register_buffer(req.value("endpoint_id", std::string{}),
                                          req.value("buffer_id", std::string{}),
                                          req.value("mr_info", json::object()));
                resp = json::object();
            }
            else if (method == "GetRemoteBuffer") {
                resp = backend_->get_remote_buffer(req.value("remote_endpoint_id", std::string{}),
                                                   req.value("buffer_id", std::string{}),
                                                   req.value("timeout_sec", 30.0));
            }
            else if (method == "EnsurePeer") {
                backend_->trigger_ensure_peer(req.value("client_addr", std::string{}));
                resp = json::object();
            }
            else {
                send_error(zmq_socket_, "unknown method: " + method);
                continue;
            }
            send_response(zmq_socket_, resp.dump());
        }
        catch (const std::exception& e) {
            if (!stop_.load() && request_received) {
                try {
                    send_error(zmq_socket_, e.what());
                }
                catch (const std::exception& send_ex) {
                    SLIME_LOG_ERROR("Failed to send error response (client may have disconnected): ", send_ex.what());
                    break;
                }
                catch (...) {
                    SLIME_LOG_ERROR("Failed to send error response (client may have disconnected)");
                    break;
                }
            }
        }
        catch (...) {
            if (!stop_.load() && request_received) {
                try {
                    send_error(zmq_socket_, "unknown error");
                }
                catch (const std::exception& send_ex) {
                    SLIME_LOG_ERROR("Failed to send error response (client may have disconnected): ", send_ex.what());
                    break;
                }
                catch (...) {
                    SLIME_LOG_ERROR("Failed to send error response (client may have disconnected)");
                    break;
                }
            }
        }
    }
    if (zmq_socket_) {
        zmq_close(zmq_socket_);
        zmq_socket_ = nullptr;
    }
    if (zmq_ctx_) {
        zmq_ctx_term(zmq_ctx_);
        zmq_ctx_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// ZmqRendezvousStub
// ---------------------------------------------------------------------------

ZmqRendezvousStub::ZmqRendezvousStub(const std::string& remote_addr): remote_addr_(ensure_tcp_addr(remote_addr))
{
    zmq_ctx_    = zmq_ctx_new();
    zmq_socket_ = zmq_socket(zmq_ctx_, ZMQ_REQ);
    if (!zmq_socket_) {
        zmq_ctx_term(zmq_ctx_);
        zmq_ctx_ = nullptr;
        throw std::runtime_error("ZMQ REQ socket create failed");
    }
    if (zmq_connect(zmq_socket_, remote_addr_.c_str()) != 0) {
        zmq_close(zmq_socket_);
        zmq_ctx_term(zmq_ctx_);
        zmq_socket_ = nullptr;
        zmq_ctx_    = nullptr;
        throw std::runtime_error("ZMQ connect failed: " + remote_addr_);
    }
}

ZmqRendezvousStub::~ZmqRendezvousStub()
{
    close();
}

void ZmqRendezvousStub::close()
{
    if (zmq_socket_) {
        zmq_close(zmq_socket_);
        zmq_socket_ = nullptr;
    }
    if (zmq_ctx_) {
        zmq_ctx_term(zmq_ctx_);
        zmq_ctx_ = nullptr;
    }
}

json ZmqRendezvousStub::call(const std::string& method, const json& req)
{
    std::string body = req.dump();
    send_multipart(zmq_socket_, method, body);
    auto [part1, part2] = recv_multipart(zmq_socket_);
    if (part1 == "ERROR") {
        throw std::runtime_error("ZMQ RPC error: " + part2);
    }
    return part1.empty() ? json::object() : json::parse(part1);
}

json ZmqRendezvousStub::GetEndpointInfo()
{
    return call("GetEndpointInfo", json::object());
}

json ZmqRendezvousStub::Handshake(const json& endpoint_info)
{
    return call("Handshake", json{{"endpoint_info", endpoint_info}});
}

void ZmqRendezvousStub::RegisterSharedBuffer(const json& req)
{
    (void)call("RegisterSharedBuffer", req);
}

json ZmqRendezvousStub::GetLocalBuffer(const std::string& buffer_id)
{
    return call("GetLocalBuffer", json{{"buffer_id", buffer_id}});
}

void ZmqRendezvousStub::RequestLazyHandshake(const std::string& initiator_id,
                                             const std::string& peer_id,
                                             const json&        my_endpoint_info,
                                             const std::string& initiator_broker_addr)
{
    (void)call("LazyHandshakeRequest",
               json{{"initiator_id", initiator_id},
                    {"peer_id", peer_id},
                    {"my_endpoint_info", my_endpoint_info},
                    {"initiator_broker_addr", initiator_broker_addr}});
}

json ZmqRendezvousStub::GetPendingLazyHandshakes(const std::string& peer_id)
{
    return call("GetPendingLazyHandshakes", json{{"peer_id", peer_id}});
}

void ZmqRendezvousStub::RespondLazyHandshake(const std::string& initiator_id,
                                             const std::string& peer_id,
                                             const json&        my_endpoint_info)
{
    (void)call("LazyHandshakeResponse",
               json{{"initiator_id", initiator_id}, {"peer_id", peer_id}, {"my_endpoint_info", my_endpoint_info}});
}

json ZmqRendezvousStub::GetLazyHandshakeResponse(const std::string& initiator_id,
                                                 const std::string& peer_id,
                                                 double             timeout_sec)
{
    return call("GetLazyHandshakeResponse",
                json{{"initiator_id", initiator_id}, {"peer_id", peer_id}, {"timeout_sec", timeout_sec}});
}

void ZmqRendezvousStub::RegisterConnectRequest(const std::string& my_id,
                                               const std::string& peer_id,
                                               const json&        endpoint_info)
{
    (void)call("RegisterConnectRequest",
               json{{"my_id", my_id}, {"peer_id", peer_id}, {"endpoint_info", endpoint_info}});
}

json ZmqRendezvousStub::GetPeerInfo(const std::string& peer_id, const std::string& other_id, double timeout_sec)
{
    return call("GetPeerInfo", json{{"peer_id", peer_id}, {"other_id", other_id}, {"timeout_sec", timeout_sec}});
}

void ZmqRendezvousStub::RegisterBuffer(const std::string& endpoint_id,
                                       const std::string& buffer_id,
                                       const json&        mr_info)
{
    (void)call("RegisterBuffer", json{{"endpoint_id", endpoint_id}, {"buffer_id", buffer_id}, {"mr_info", mr_info}});
}

json ZmqRendezvousStub::GetRemoteBuffer(const std::string& remote_endpoint_id,
                                        const std::string& buffer_id,
                                        double             timeout_sec)
{
    return call(
        "GetRemoteBuffer",
        json{{"remote_endpoint_id", remote_endpoint_id}, {"buffer_id", buffer_id}, {"timeout_sec", timeout_sec}});
}

void ZmqRendezvousStub::EnsurePeer(const std::string& client_addr)
{
    (void)call("EnsurePeer", json{{"client_addr", client_addr}});
}

}  // namespace dlslime

#endif  // BUILD_RDMA_RENDEZVOUS_ZMQ
