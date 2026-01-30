#ifdef BUILD_RDMA_RENDEZVOUS_ZMQ

#include "rdma_lazy_peer.h"

#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <thread>
#include <unistd.h>

#include "dlslime/csrc/engine/assignment.h"
#include "rdma_future.h"
#include "rdma_utils.h"

namespace dlslime {

static size_t page_size()
{
    long p = sysconf(_SC_PAGESIZE);
    return p > 0 ? static_cast<size_t>(p) : 4096;
}

RdmaLazyPeer::RdmaLazyPeer(const std::string& my_broker_addr,
                           const std::string& my_id,
                           const std::string& device_name,
                           int32_t            ib_port,
                           const std::string& link_type):
    my_id_(my_id), ib_port_(ib_port), link_type_(link_type), my_broker_addr_(my_broker_addr)
{
    device_name_ = device_name;
    if (device_name_.empty()) {
        auto devs = available_nic();
        if (!devs.empty())
            device_name_ = devs[0];
    }
    // Stub created lazily in EnsureMyBrokerStub() so ZMQ socket is used from same thread that created it
}

void RdmaLazyPeer::EnsureMyBrokerStub()
{
    std::thread::id tid  = std::this_thread::get_id();
    auto&           stub = stub_my_broker_per_thread_[tid];
    if (!stub)
        stub = std::make_unique<ZmqRendezvousStub>(my_broker_addr_);
}

ZmqRendezvousStub* RdmaLazyPeer::GetStubToRemoteBroker(const std::string& remote_id)
{
    std::thread::id tid  = std::this_thread::get_id();
    auto            it_t = remote_broker_stubs_per_thread_.find(tid);
    if (it_t == remote_broker_stubs_per_thread_.end())
        return nullptr;
    auto it = it_t->second.find(remote_id);
    if (it != it_t->second.end())
        return it->second.get();
    return nullptr;
}

ZmqRendezvousStub* RdmaLazyPeer::GetOrCreateStubToRemoteBroker(const std::string& remote_id,
                                                               const std::string& remote_broker_addr)
{
    std::thread::id tid   = std::this_thread::get_id();
    auto&           stubs = remote_broker_stubs_per_thread_[tid];
    auto            it    = stubs.find(remote_id);
    if (it != stubs.end())
        return it->second.get();
    stubs[remote_id] = std::make_unique<ZmqRendezvousStub>(remote_broker_addr);
    return stubs[remote_id].get();
}

void RdmaLazyPeer::EnsureConnect(const std::string& remote_id, const std::string& remote_broker_addr)
{
    ZmqRendezvousStub*            stub_my     = nullptr;
    ZmqRendezvousStub*            stub_remote = nullptr;
    std::shared_ptr<RDMAEndpoint> ep;
    std::string                   dev;

    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (endpoints_.find(remote_id) != endpoints_.end())
            return;  // idempotent: already connected
        while (connecting_.count(remote_id) != 0)
            cond_.wait(lock);
        connecting_.insert(remote_id);
        EnsureMyBrokerStub();
        stub_my                         = stub_my_broker_per_thread_[std::this_thread::get_id()].get();
        remote_broker_addrs_[remote_id] = remote_broker_addr;
        stub_remote                     = GetOrCreateStubToRemoteBroker(remote_id, remote_broker_addr);
    }

    auto devices = available_nic();
    if (devices.empty()) {
        std::lock_guard<std::mutex> lock(mutex_);
        connecting_.erase(remote_id);
        cond_.notify_all();
        throw std::runtime_error("RdmaLazyPeer: no RDMA devices");
    }
    size_t n_dev = devices.size();
    size_t idx   = (static_cast<size_t>(std::hash<std::string>{}(my_id_ + remote_id)) % n_dev);
    dev          = devices[idx];

    // my broker.
    ep = std::make_shared<RDMAEndpoint>(dev, ib_port_, link_type_);
    stub_remote->RegisterConnectRequest(my_id_, remote_id, ep->endpointInfo());

    json peer_info;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(LAZY_TIMEOUT);
    while (std::chrono::steady_clock::now() < deadline) {
        peer_info = stub_my->GetPeerInfo(my_id_, remote_id, 0);
        if (!peer_info.empty())
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (peer_info.empty()) {
        std::lock_guard<std::mutex> lock(mutex_);
        connecting_.erase(remote_id);
        cond_.notify_all();
        throw std::runtime_error("RdmaLazyPeer: get_peer_info timed out (remote peer may not have called connect)");
    }
    ep->connect(peer_info);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        connecting_.erase(remote_id);
        cond_.notify_all();
        endpoints_[remote_id] = ep;

        for (auto& kv : pending_buffers_) {
            uintptr_t ptr  = kv.second.first;
            size_t    size = kv.second.second;
            ep->registerOrAccessMemoryRegion(ptr, ptr, 0, size);
        }
        json mr_info_map = ep->endpointInfo().value("mr_info", json::object());
        for (const auto& kv : pending_buffers_) {
            const std::string& buf_id = kv.first;
            uintptr_t          ptr    = kv.second.first;
            size_t             size   = kv.second.second;
            json               mr     = mr_info_map.value(std::to_string(ptr), json::object());
            if (!mr.empty())
                stub_my_broker_per_thread_[std::this_thread::get_id()]->RegisterBuffer(my_id_, buf_id, mr);
            LocalBuffer lb;
            lb.ptr                        = ptr;
            lb.size                       = size;
            lb.mr_key_per_peer[remote_id] = mr.value("mr_key", static_cast<uintptr_t>(0));
            local_buffers_[buf_id]        = std::move(lb);
        }
        pending_buffers_.clear();

        for (auto& kv : local_buffers_) {
            const std::string& buf_id = kv.first;
            LocalBuffer&       lb     = kv.second;
            if (lb.mr_key_per_peer.count(remote_id))
                continue;
            ep->registerOrAccessMemoryRegion(lb.ptr, lb.ptr, 0, lb.size);
            json mr = GetMrForPtr(ep, lb.ptr);
            if (!mr.empty())
                lb.mr_key_per_peer[remote_id] = mr.value("mr_key", static_cast<uintptr_t>(0));
        }
    }
}

json RdmaLazyPeer::GetMrForPtr(const std::shared_ptr<RDMAEndpoint>& ep, uintptr_t ptr)
{
    json info    = ep->endpointInfo();
    json mr_info = info.value("mr_info", json::object());
    return mr_info.value(std::to_string(ptr), json::object());
}

std::string RdmaLazyPeer::ResolveRemoteId(const std::string& remote_id_or_addr)
{
    if (remote_id_or_addr.find(':') != std::string::npos)
        return rendezvous::default_peer_id_from_client_addr(remote_id_or_addr);
    return remote_id_or_addr;
}

void RdmaLazyPeer::Connect(const std::string& remote_id, const std::string& remote_broker_addr)
{
    EnsureConnect(remote_id, remote_broker_addr);
}

void RdmaLazyPeer::Connect(const std::string& remote_broker_addr)
{
    Connect(ResolveRemoteId(remote_broker_addr), remote_broker_addr);
}

void RdmaLazyPeer::RegisterBuffer(const std::string& buffer_id, void* ptr, size_t size)
{
    std::lock_guard<std::mutex> lock(mutex_);
    EnsureMyBrokerStub();
    uintptr_t uptr = reinterpret_cast<uintptr_t>(ptr);
    if (endpoints_.empty()) {
        pending_buffers_[buffer_id] = {uptr, size};
        return;
    }
    for (auto& kv : endpoints_)
        kv.second->registerOrAccessMemoryRegion(uptr, uptr, 0, size);
    auto first_ep = endpoints_.begin()->second;
    json mr       = GetMrForPtr(first_ep, uptr);
    if (mr.empty())
        throw std::runtime_error("RdmaLazyPeer: mr_info not found for ptr");
    stub_my_broker_per_thread_[std::this_thread::get_id()]->RegisterBuffer(my_id_, buffer_id, mr);
    LocalBuffer lb;
    lb.ptr  = uptr;
    lb.size = size;
    for (auto& kv : endpoints_) {
        json m                       = GetMrForPtr(kv.second, uptr);
        lb.mr_key_per_peer[kv.first] = m.value("mr_key", static_cast<uintptr_t>(0));
    }
    local_buffers_[buffer_id] = std::move(lb);
}

std::pair<void*, size_t> RdmaLazyPeer::AllocAndRegisterBuffer(const std::string& buffer_id, size_t size)
{
    size_t pgsz         = page_size();
    size_t aligned_size = ((size + pgsz - 1) / pgsz) * pgsz;
    void*  ptr          = nullptr;
    if (posix_memalign(&ptr, pgsz, aligned_size) != 0)
        throw std::runtime_error("RdmaLazyPeer: posix_memalign failed");
    RegisterBuffer(buffer_id, ptr, aligned_size);
    return {ptr, aligned_size};
}

uintptr_t RdmaLazyPeer::GetLocalMrKey(const std::string& remote_id_or_addr, const std::string& buffer_id)
{
    std::string                 remote_id = ResolveRemoteId(remote_id_or_addr);
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        lit = local_buffers_.find(buffer_id);
    if (lit == local_buffers_.end())
        throw std::runtime_error("RdmaLazyPeer: local buffer not registered: " + buffer_id);
    auto pit = lit->second.mr_key_per_peer.find(remote_id);
    if (pit == lit->second.mr_key_per_peer.end() || pit->second == 0) {
        SLIME_LOG_ERROR("RdmaLazyPeer: local buffer not registered on endpoint to peer {}", remote_id);
        return (uintptr_t)nullptr;
    }
    return pit->second;
}

std::shared_ptr<RDMAEndpoint> RdmaLazyPeer::GetEndpoint(const std::string& remote_id_or_addr)
{
    std::string                 remote_id = ResolveRemoteId(remote_id_or_addr);
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = endpoints_.find(remote_id);
    if (it != endpoints_.end())
        return it->second;
    return nullptr;
}

uintptr_t RdmaLazyPeer::GetRemoteMrKey(const std::string& remote_id_or_addr, const std::string& buffer_id)
{
    std::string                   remote_id = ResolveRemoteId(remote_id_or_addr);
    std::shared_ptr<RDMAEndpoint> ep;
    ZmqRendezvousStub*            stub_remote = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        key = std::make_pair(remote_id, buffer_id);
        auto                        it  = remote_buffers_.find(key);
        if (it != remote_buffers_.end())
            return it->second;
        ep = endpoints_[remote_id];
        if (!ep)
            throw std::runtime_error("RdmaLazyPeer: not connected to remote " + remote_id + " (call Connect first)");
        stub_remote = GetStubToRemoteBroker(remote_id);
        if (!stub_remote) {
            auto it_addr = remote_broker_addrs_.find(remote_id);
            if (it_addr != remote_broker_addrs_.end())
                stub_remote = GetOrCreateStubToRemoteBroker(remote_id, it_addr->second);
        }
        if (!stub_remote)
            throw std::runtime_error("RdmaLazyPeer: not connected to remote " + remote_id + " (call Connect first)");
    }
    json mr_info = stub_remote->GetRemoteBuffer(remote_id, buffer_id, BUFFER_TIMEOUT);
    if (mr_info.empty())
        throw std::runtime_error("RdmaLazyPeer: get_remote_buffer timed out or empty");
    uintptr_t mr_key = mr_info.value("mr_key", static_cast<uintptr_t>(0));
    ep->registerOrAccessRemoteMemoryRegion(mr_key, mr_info);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        remote_buffers_[{remote_id, buffer_id}] = mr_key;
    }
    return mr_key;
}

std::shared_ptr<ReadWriteFuture>
RdmaLazyPeer::read(const std::string& remote_id_or_addr, const std::vector<assign_tuple_t>& assign, void* stream)
{
    std::string                   remote_id = ResolveRemoteId(remote_id_or_addr);
    std::shared_ptr<RDMAEndpoint> ep;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        it = endpoints_.find(remote_id);
        if (it == endpoints_.end())
            throw std::runtime_error("RdmaLazyPeer: not connected to remote " + remote_id + " (call Connect first)");
        ep = it->second;
    }
    return ep->read(assign, stream);
}

std::shared_ptr<ReadWriteFuture>
RdmaLazyPeer::write(const std::string& remote_id_or_addr, const std::vector<assign_tuple_t>& assign, void* stream)
{
    std::string                   remote_id = ResolveRemoteId(remote_id_or_addr);
    std::shared_ptr<RDMAEndpoint> ep;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        it = endpoints_.find(remote_id);
        if (it == endpoints_.end())
            throw std::runtime_error("RdmaLazyPeer: not connected to remote " + remote_id + " (call Connect first)");
        ep = it->second;
    }
    return ep->write(assign, stream);
}

void RdmaLazyPeer::Close()
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& kv : stub_my_broker_per_thread_)
        if (kv.second)
            kv.second->close();
    for (auto& kv_t : remote_broker_stubs_per_thread_)
        for (auto& kv : kv_t.second)
            if (kv.second)
                kv.second->close();
}

}  // namespace dlslime

#endif  // BUILD_RDMA_RENDEZVOUS_ZMQ
