#pragma once

#ifdef BUILD_RDMA_RENDEZVOUS_ZMQ

#include <condition_variable>
#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "dlslime/csrc/engine/assignment.h"
#include "nanocommon/json.hpp"
#include "rdma_endpoint.h"
#include "rdma_future.h"
#include "rdma_rendezvous_zmq.h"
#include "rendezvous/client_addr.h"

namespace dlslime {

/** Lazy peer: my_broker_addr + my_id; each process has its own broker (P2P). connect(remote_id, remote_broker_addr). */
class RdmaLazyPeer {
public:
    RdmaLazyPeer(const std::string& my_broker_addr,
                 const std::string& my_id,
                 const std::string& device_name = "",
                 int32_t            ib_port     = 1,
                 const std::string& link_type   = "RoCE");

    /** Connect to remote_id; remote_broker_addr is the broker of the remote peer (where to post handshake / get
     * buffer). */
    void Connect(const std::string& remote_id, const std::string& remote_broker_addr);
    /** Connect to remote: single arg = remote_broker_addr, remote_id 自动取端口（与 broker 默认 peer id 一致）. */
    void Connect(const std::string& remote_broker_addr);
    void RegisterBuffer(const std::string& buffer_id, void* ptr, size_t size);
    /** Page-aligned alloc + RegisterBuffer. Returns (ptr, aligned_size). */
    std::pair<void*, size_t> AllocAndRegisterBuffer(const std::string& buffer_id, size_t size);

    /** Get local mr_key for buffer_id. remote 可为 remote_id 或 remote_broker_addr（含 ':' 则按端口推导 id）. */
    uintptr_t GetLocalMrKey(const std::string& remote_id_or_addr, const std::string& buffer_id);
    /** Get remote mr_key for buffer_id. remote 可为 remote_id 或 remote_broker_addr. */
    uintptr_t GetRemoteMrKey(const std::string& remote_id_or_addr, const std::string& buffer_id);

    std::shared_ptr<RDMAEndpoint> GetEndpoint(const std::string& remote_id_or_addr);

    /** RDMA read from remote. remote 可为 remote_id 或 remote_broker_addr. */
    std::shared_ptr<ReadWriteFuture>
    read(const std::string& remote_id_or_addr, const std::vector<assign_tuple_t>& assign, void* stream = nullptr);
    /** RDMA write to remote. remote 可为 remote_id 或 remote_broker_addr. */
    std::shared_ptr<ReadWriteFuture>
    write(const std::string& remote_id_or_addr, const std::vector<assign_tuple_t>& assign, void* stream = nullptr);

    void Close();

    static constexpr double LAZY_TIMEOUT   = 60.0;
    static constexpr double BUFFER_TIMEOUT = 30.0;

private:
    /** Create stub for current thread (ZMQ sockets are thread-affine). Caller must hold mutex_. */
    void EnsureMyBrokerStub();
    void EnsureConnect(const std::string& remote_id, const std::string& remote_broker_addr);
    /** Stub to remote broker for current thread. Caller must hold mutex_. */
    ZmqRendezvousStub* GetOrCreateStubToRemoteBroker(const std::string& remote_id,
                                                     const std::string& remote_broker_addr);
    ZmqRendezvousStub* GetStubToRemoteBroker(const std::string& remote_id);
    static json        GetMrForPtr(const std::shared_ptr<RDMAEndpoint>& ep, uintptr_t ptr);
    /** remote_id_or_addr 含 ':' 则按 default_peer_id_from_client_addr 推导，否则原样. */
    static std::string ResolveRemoteId(const std::string& remote_id_or_addr);

    mutable std::mutex      mutex_;
    std::condition_variable cond_;
    std::set<std::string>   connecting_;
    std::string             my_id_;
    std::string             device_name_;
    int32_t                 ib_port_;
    std::string             link_type_;
    std::string             my_broker_addr_;
    /** Per-thread stub to my broker (ZMQ socket must be used from creating thread). */
    std::map<std::thread::id, std::unique_ptr<ZmqRendezvousStub>> stub_my_broker_per_thread_;
    /** Per-thread: remote_id -> stub to that peer's broker. */
    std::map<std::thread::id, std::map<std::string, std::unique_ptr<ZmqRendezvousStub>>>
        remote_broker_stubs_per_thread_;
    /** remote_id -> remote broker addr (set when stub first created), so any thread can create its stub in
     * GetRemoteMrKey. */
    std::map<std::string, std::string>                   remote_broker_addrs_;
    std::map<std::string, std::shared_ptr<RDMAEndpoint>> endpoints_;

    struct LocalBuffer {
        uintptr_t                        ptr;
        size_t                           size;
        std::map<std::string, uintptr_t> mr_key_per_peer;
    };
    std::map<std::string, LocalBuffer>                       local_buffers_;
    std::map<std::string, std::pair<uintptr_t, size_t>>      pending_buffers_;
    std::map<std::pair<std::string, std::string>, uintptr_t> remote_buffers_;
};

}  // namespace dlslime

#endif  // BUILD_RDMA_RENDEZVOUS_ZMQ
