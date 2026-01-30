#pragma once

#ifdef BUILD_RDMA_RENDEZVOUS_ZMQ

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "dlslime/csrc/engine/assignment.h"
#include "rdma_future.h"
#include "rdma_lazy_peer.h"
#include "rdma_rendezvous_zmq.h"

namespace dlslime {

/** RDMA Peer Agent: Manages rendezvous, lazy peer creation, and buffer registration.
 *  Formerly "Broker". Encapsulates one-to-one or one-to-many connections via lazy peer.
 */
class RdmaPeerAgent {
public:
    explicit RdmaPeerAgent(const std::string& bind_addr);
    ~RdmaPeerAgent();

    void                     Connect(const std::string& remote_broker_addr,
                                     const std::string& device_name        = "",
                                     int32_t            ib_port            = 1,
                                     const std::string& link_type          = "RoCE",
                                     int                timeout_ms         = 1000,
                                     bool               skip_remote_ensure = false);
    void                     RegisterBuffer(const std::string& buffer_id, void* ptr, size_t size);
    std::pair<void*, size_t> AllocAndRegisterBuffer(const std::string& buffer_id, size_t size);
    uintptr_t                GetLocalMrKey(const std::string& remote_id_or_addr, const std::string& buffer_id);
    uintptr_t                GetRemoteMrKey(const std::string& remote_id_or_addr, const std::string& buffer_id);

    std::shared_ptr<ReadWriteFuture>
    read(const std::string& remote_id_or_addr, const std::vector<assign_tuple_t>& assign, void* stream = nullptr);
    std::shared_ptr<ReadWriteFuture>
    write(const std::string& remote_id_or_addr, const std::vector<assign_tuple_t>& assign, void* stream = nullptr);

    std::shared_ptr<RDMAEndpoint> GetEndpoint(const std::string& remote_broker_addr);

    void Close();
    void stop();

    std::string client_addr() const
    {
        return client_addr_;
    }

private:
    void EnsurePeer(const std::string& device_name, int32_t ib_port, const std::string& link_type);

    std::string                            client_addr_;
    std::shared_ptr<RdmaRendezvousBackend> backend_;
    std::shared_ptr<ZmqRendezvousServer>   server_;
    // We use unique_ptr for peer because it's owned by this agent (mostly).
    // Wait, RdmaLazyPeer is used as shared_ptr in GetEndpoint (impl) or exposed?
    // checking rdma_broker.cpp (old) / rdma_peer_agent.cpp:
    // In cpp: peer_ = std::make_unique<RdmaLazyPeer>(...)
    // So unique_ptr is correct based on initialization.
    std::unique_ptr<RdmaLazyPeer> peer_;

    std::string peer_device_name_;
    int32_t     peer_ib_port_   = 1;
    std::string peer_link_type_ = "RoCE";
    std::mutex  mutex_;
};

}  // namespace dlslime

#endif  // BUILD_RDMA_RENDEZVOUS_ZMQ
