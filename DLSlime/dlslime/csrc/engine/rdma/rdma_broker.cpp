#ifdef BUILD_RDMA_RENDEZVOUS_ZMQ

#include "rdma_broker.h"

#include "rendezvous/client_addr.h"

namespace dlslime {

RdmaBroker::RdmaBroker(const std::string& bind_addr):
    client_addr_(rendezvous::client_addr_from_bind(bind_addr)),
    backend_(std::make_shared<RdmaRendezvousBackend>(nullptr)),
    server_(std::make_shared<ZmqRendezvousServer>(backend_, bind_addr))
{
    backend_->set_ensure_peer_callback([this](const std::string& client_addr) {
        if (!client_addr.empty()) {
            // Triggered by remote EnsurePeer RPC.
            // Connect back to allowing handshake, but skip contacting remote to avoid infinite loop.
            // Run in background thread to avoid deadlock (A waits for EnsurePeer return, B waits for A connect)
            std::thread([this, client_addr]() {
                Connect(client_addr, "", 1, "RoCE", /*skip_remote_ensure=*/true);
            }).detach();
        }
        else {
            EnsurePeer("", 1, "RoCE");
        }
    });
    server_->start();
}

RdmaBroker::~RdmaBroker()
{
    Close();  // 断掉所有连接，释放内部 peer 及其 RDMA endpoints；peer->Close() 会释放所有 ZMQ stub (REQ socket+ctx)
    server_->stop();  // 释放 server 的 ZMQ (REP socket+ctx)
}

void RdmaBroker::EnsurePeer(const std::string& device_name, int32_t ib_port, const std::string& link_type)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (peer_)
        return;
    std::string id    = rendezvous::default_peer_id_from_client_addr(client_addr_);
    peer_             = std::make_unique<RdmaLazyPeer>(client_addr_, id, device_name, ib_port, link_type);
    peer_device_name_ = device_name;
    peer_ib_port_     = ib_port;
    peer_link_type_   = link_type;
}

void RdmaBroker::Connect(const std::string& remote_broker_addr,
                         const std::string& device_name,
                         int32_t            ib_port,
                         const std::string& link_type,
                         bool               skip_remote_ensure)
{
    // Smart Connect: Force remote broker to ensure it has a peer.
    // This allows single-sided connection initiation.
    if (!skip_remote_ensure) {
        try {
            ZmqRendezvousStub stub(remote_broker_addr);
            stub.EnsurePeer(client_addr_);
        }
        catch (const std::exception& e) {
            SLIME_LOG_ERROR("Smart Connect (EnsurePeer) failed: ", e.what());
            // Fallthrough? If remote is old, it might not support this.
            // But if remote is not started, we'll fail later anyway.
        }
    }

    EnsurePeer(device_name, ib_port, link_type);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (peer_)
            peer_->Connect(remote_broker_addr);
    }
}

void RdmaBroker::RegisterBuffer(const std::string& buffer_id, void* ptr, size_t size)
{
    EnsurePeer(peer_device_name_, peer_ib_port_, peer_link_type_);
    peer_->RegisterBuffer(buffer_id, ptr, size);
}

std::pair<void*, size_t> RdmaBroker::AllocAndRegisterBuffer(const std::string& buffer_id, size_t size)
{
    EnsurePeer(peer_device_name_, peer_ib_port_, peer_link_type_);
    return peer_->AllocAndRegisterBuffer(buffer_id, size);
}

uintptr_t RdmaBroker::GetLocalMrKey(const std::string& remote_id_or_addr, const std::string& buffer_id)
{
    if (!peer_)
        throw std::runtime_error("RdmaBroker: not connected (call Connect first)");
    return peer_->GetLocalMrKey(remote_id_or_addr, buffer_id);
}

uintptr_t RdmaBroker::GetRemoteMrKey(const std::string& remote_id_or_addr, const std::string& buffer_id)
{
    if (!peer_)
        throw std::runtime_error("RdmaBroker: not connected (call Connect first)");
    return peer_->GetRemoteMrKey(remote_id_or_addr, buffer_id);
}

std::shared_ptr<ReadWriteFuture>
RdmaBroker::read(const std::string& remote_id_or_addr, const std::vector<assign_tuple_t>& assign, void* stream)
{
    if (!peer_)
        throw std::runtime_error("RdmaBroker: not connected (call Connect first)");
    return peer_->read(remote_id_or_addr, assign, stream);
}

std::shared_ptr<ReadWriteFuture>
RdmaBroker::write(const std::string& remote_id_or_addr, const std::vector<assign_tuple_t>& assign, void* stream)
{
    if (!peer_)
        throw std::runtime_error("RdmaBroker: not connected (call Connect first)");
    return peer_->write(remote_id_or_addr, assign, stream);
}

void RdmaBroker::Close()
{
    if (peer_) {
        peer_->Close();
        peer_.reset();
    }
}

RdmaLazyPeer* RdmaBroker::peer(const std::string& my_id,
                               const std::string& device_name,
                               int32_t            ib_port,
                               const std::string& link_type)
{
    EnsurePeer(device_name, ib_port, link_type);
    return peer_.get();
}

void RdmaBroker::stop()
{
    Close();
    server_->stop();
}

}  // namespace dlslime

#endif  // BUILD_RDMA_RENDEZVOUS_ZMQ
