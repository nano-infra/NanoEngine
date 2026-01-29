#pragma once

#ifdef BUILD_RDMA_RENDEZVOUS_ZMQ

#include <memory>
#include <string>
#include <vector>

#include "dlslime/csrc/engine/assignment.h"
#include "rdma_future.h"
#include "rdma_lazy_peer.h"
#include "rdma_rendezvous_zmq.h"

namespace dlslime {

/** RDMA 本进程 broker：无 endpoint，仅 handshake/buffer 路由。与另一 broker 永远只有一条链路。
 *  Peer 惰性：首次 connect 时创建内部 RdmaLazyPeer；connect/write/read 等直接挂在 broker 上。 */
class RdmaBroker {
public:
    explicit RdmaBroker(const std::string& bind_addr);
    ~RdmaBroker();

    /** 惰性：首次调用时创建内部 peer；与每个 remote broker 仅一条链路。connect 幂等。 */
    void                     Connect(const std::string& remote_broker_addr,
                                     const std::string& device_name        = "",
                                     int32_t            ib_port            = 1,
                                     const std::string& link_type          = "RoCE",
                                     bool               skip_remote_ensure = false);
    void                     RegisterBuffer(const std::string& buffer_id, void* ptr, size_t size);
    std::pair<void*, size_t> AllocAndRegisterBuffer(const std::string& buffer_id, size_t size);
    uintptr_t                GetLocalMrKey(const std::string& remote_id_or_addr, const std::string& buffer_id);
    uintptr_t                GetRemoteMrKey(const std::string& remote_id_or_addr, const std::string& buffer_id);
    std::shared_ptr<ReadWriteFuture>
    read(const std::string& remote_id_or_addr, const std::vector<assign_tuple_t>& assign, void* stream = nullptr);
    std::shared_ptr<ReadWriteFuture>
         write(const std::string& remote_id_or_addr, const std::vector<assign_tuple_t>& assign, void* stream = nullptr);
    void Close();

    /** 兼容旧 API：返回内部惰性 peer（每 broker 一个）。 */
    RdmaLazyPeer* peer(const std::string& my_id       = "",
                       const std::string& device_name = "",
                       int32_t            ib_port     = 1,
                       const std::string& link_type   = "RoCE");

    void        stop();
    std::string client_addr() const
    {
        return client_addr_;
    }

private:
    void EnsurePeer(const std::string& device_name, int32_t ib_port, const std::string& link_type);

    std::string                            client_addr_;
    std::shared_ptr<RdmaRendezvousBackend> backend_;
    std::shared_ptr<ZmqRendezvousServer>   server_;
    std::unique_ptr<RdmaLazyPeer>          peer_;
    std::string                            peer_device_name_;
    int32_t                                peer_ib_port_   = 1;
    std::string                            peer_link_type_ = "RoCE";
    std::mutex                             mutex_;
};

}  // namespace dlslime

#endif  // BUILD_RDMA_RENDEZVOUS_ZMQ
