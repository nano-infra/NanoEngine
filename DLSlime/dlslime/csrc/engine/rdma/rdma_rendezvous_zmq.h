#pragma once

#ifdef BUILD_RDMA_RENDEZVOUS_ZMQ

#include <atomic>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "nanocommon/json.hpp"
#include "rdma_endpoint.h"

namespace dlslime {

class RdmaRendezvousBackend;
class ZmqRendezvousServer;
class ZmqRendezvousStub;

/** Backend for RDMA rendezvous: endpoint info, handshake, shared buffer registration. Uses JSON.
 *  For lazy handshake broker, pass endpoint=nullptr; only lazy handshake methods are valid then. */
class RdmaRendezvousBackend {
public:
    explicit RdmaRendezvousBackend(std::shared_ptr<RDMAEndpoint> endpoint);
    json                         get_endpoint_info();
    json                         handshake_request(const json& req);
    json                         register_shared_buffer(const json& req);
    json                         get_local_buffer(const json& req);
    void                         register_local_buffer(const std::string& buffer_id, const json& mr_info);
    std::pair<std::string, json> get_pending_shared_buffer(std::string* buffer_id, double timeout_sec);

    /** Lazy handshake: initiator requests connection to peer; broker stores (endpoint_info, initiator_broker_addr). */
    void lazy_handshake_request(const std::string& initiator_id,
                                const std::string& peer_id,
                                const json&        my_endpoint_info,
                                const std::string& initiator_broker_addr);
    /** Lazy handshake: peer polls pending requests; returns list of {initiator_id, endpoint_info,
     * initiator_broker_addr}, consumed. */
    json get_pending_lazy_handshakes(const std::string& peer_id);
    /** Lazy handshake: peer responds with my endpoint_info after connecting to initiator. */
    void
    lazy_handshake_response(const std::string& initiator_id, const std::string& peer_id, const json& my_endpoint_info);
    /** Lazy handshake: initiator blocks until peer responded or timeout_sec; returns peer endpoint_info or empty. */
    json get_lazy_handshake_response(const std::string& initiator_id, const std::string& peer_id, double timeout_sec);

    /** Symmetric connect: register my endpoint_info on peer's broker (peer_id = broker owner). No initiator/target. */
    void register_connect_request(const std::string& my_id, const std::string& peer_id, const json& endpoint_info);
    /** Symmetric connect: wait for other_id's endpoint_info on my broker (peer_id = me); returns and consumes. */
    json get_peer_info(const std::string& peer_id, const std::string& other_id, double timeout_sec);

    /** Register buffer by (endpoint_id, buffer_id); broker stores mr_info for others to get. */
    /** Register buffer by (endpoint_id, buffer_id); broker stores mr_info for others to get. */
    void register_buffer(const std::string& endpoint_id, const std::string& buffer_id, const json& mr_info);
    /** Get remote buffer mr_info by (remote_endpoint_id, buffer_id); blocks until registered or timeout. */
    json get_remote_buffer(const std::string& remote_endpoint_id, const std::string& buffer_id, double timeout_sec);

    /** Ensure Peer Callback: triggered when we receive an EnsurePeer request. */
    void set_ensure_peer_callback(std::function<void(const std::string&)> cb)
    {
        ensure_peer_cb_ = std::move(cb);
    }
    /** Triggers the callback if set. */
    void trigger_ensure_peer(const std::string& client_addr);

private:
    std::shared_ptr<RDMAEndpoint> endpoint_;
    std::map<std::string, json>   pending_buffers_;
    std::map<std::string, json>   local_buffers_;
    std::mutex                    mutex_;
    std::condition_variable       cond_;
    // Lazy handshake: peer_id -> [(initiator_id, endpoint_info, initiator_broker_addr)]
    std::map<std::string, std::vector<std::tuple<std::string, json, std::string>>> pending_lazy_requests_;
    // (initiator_id, peer_id) -> peer endpoint_info
    std::map<std::pair<std::string, std::string>, json> pending_lazy_responses_;
    // Symmetric connect: peer_id -> (other_id -> endpoint_info)
    std::map<std::string, std::map<std::string, json>> symmetric_connects_;
    // endpoint_id -> (buffer_id -> mr_info) for broker buffer lookup
    std::map<std::string, std::map<std::string, json>> endpoint_buffers_;
    // Ensure Peer Callback
    std::function<void(const std::string&)> ensure_peer_cb_;
};

/** ZMQ REP server running in a thread; dispatches by method name, JSON body. */
class ZmqRendezvousServer {
public:
    ZmqRendezvousServer(std::shared_ptr<RdmaRendezvousBackend> backend, const std::string& addr);
    ~ZmqRendezvousServer();
    void start();
    void stop();

private:
    void                                   run();
    std::shared_ptr<RdmaRendezvousBackend> backend_;
    std::string                            addr_;
    void*                                  zmq_ctx_{nullptr};
    void*                                  zmq_socket_{nullptr};
    std::unique_ptr<std::thread>           thread_;
    std::atomic<bool>                      stop_{false};
};

/** ZMQ REQ client stub: GetEndpointInfo, Handshake, RegisterSharedBuffer, GetLocalBuffer, lazy handshake. JSON. */
class ZmqRendezvousStub {
public:
    explicit ZmqRendezvousStub(const std::string& remote_addr, int timeout_ms = 1000);
    ~ZmqRendezvousStub();
    json GetEndpointInfo();
    json Handshake(const json& endpoint_info);
    void RegisterSharedBuffer(const json& req);
    json GetLocalBuffer(const std::string& buffer_id);
    void RequestLazyHandshake(const std::string& initiator_id,
                              const std::string& peer_id,
                              const json&        my_endpoint_info,
                              const std::string& initiator_broker_addr);
    json GetPendingLazyHandshakes(const std::string& peer_id);
    void
    RespondLazyHandshake(const std::string& initiator_id, const std::string& peer_id, const json& my_endpoint_info);
    json GetLazyHandshakeResponse(const std::string& initiator_id, const std::string& peer_id, double timeout_sec);
    void RegisterConnectRequest(const std::string& my_id, const std::string& peer_id, const json& endpoint_info);
    json GetPeerInfo(const std::string& peer_id, const std::string& other_id, double timeout_sec);
    void RegisterBuffer(const std::string& endpoint_id, const std::string& buffer_id, const json& mr_info);
    json GetRemoteBuffer(const std::string& remote_endpoint_id, const std::string& buffer_id, double timeout_sec);
    /** Ensure Peer: force remote broker to check/create peer. */
    void EnsurePeer(const std::string& client_addr);
    void close();

private:
    json        call(const std::string& method, const json& req);
    std::string remote_addr_;
    void*       zmq_ctx_{nullptr};
    void*       zmq_socket_{nullptr};
};

}  // namespace dlslime

#endif  // BUILD_RDMA_RENDEZVOUS_ZMQ
