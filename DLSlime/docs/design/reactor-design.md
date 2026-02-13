# Reactor Model Design Sketch (For Future Reference)

## Overview

This document outlines how to convert RdmaLazyPeer to a reactor/mailbox model **IF** profiling shows lock contention issues.

## Architecture

```
┌─────────────────────────────────────────────────┐
│              Multiple Threads                    │
└──────────────┬──────────────────────────────────┘
               │ (Mailbox::send)
               ▼
┌──────────────────────────────────────────────────┐
│         Lock-Free Mailbox (SPMC Queue)           │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │
│  │ Msg │→│ Msg │→│ Msg │→│ Msg │→│ Msg │       │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘       │
└──────────────┬───────────────────────────────────┘
               │ (dequeue)
               ▼
┌──────────────────────────────────────────────────┐
│           Reactor Thread (Event Loop)            │
│  ┌────────────────────────────────────────────┐  │
│  │ while (running) {                          │  │
│  │   msg = mailbox.dequeue(timeout);         │  │
│  │   dispatch(msg);                          │  │
│  │   process_rdma_completions();             │  │
│  │   process_timers();                       │  │
│  │ }                                         │  │
│  └────────────────────────────────────────────┘  │
│  State (no locks needed!):                       │
│  - peer_connections_ (map)                       │
│  - local_buffers_ (map)                          │
│  - stub_pool_ (still thread-safe for responses)  │
└──────────────────────────────────────────────────┘
```

## Message Protocol

```cpp
enum class MessageType {
    CONNECT,
    DISCONNECT,
    REGISTER_BUFFER,
    GET_REMOTE_KEY,
    SHUTDOWN
};

struct Message {
    MessageType type;
    std::string remote_id;
    std::string remote_addr;
    std::string buffer_id;
    void* buffer_ptr;
    size_t buffer_size;

    // For responses
    std::promise<bool> result_promise;  // Or callback
};

class RdmaReactor {
public:
    // Public API (thread-safe, non-blocking)
    std::future<bool> Connect(const std::string& remote_id,
                              const std::string& remote_addr) {
        Message msg{MessageType::CONNECT, remote_id, remote_addr};
        std::future<bool> result = msg.result_promise.get_future();
        mailbox_.send(std::move(msg));
        return result;
    }

private:
    void run() {
        while (running_) {
            // Process messages
            while (auto msg = mailbox_.try_dequeue()) {
                dispatch(*msg);
            }

            // Process RDMA completions (non-blocking)
            poll_rdma_completions();

            // Process timers
            process_timers();

            // Brief sleep if no work
            if (idle) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }
    }

    void dispatch(Message& msg) {
        switch (msg.type) {
            case MessageType::CONNECT:
                handle_connect(msg);
                break;
            case MessageType::REGISTER_BUFFER:
                handle_register_buffer(msg);
                break;
            // ...
        }
    }

    void handle_connect(Message& msg) {
        // All state access is single-threaded, no locks!
        try {
            bool success = do_connect(msg.remote_id, msg.remote_addr);
            msg.result_promise.set_value(success);
        } catch (const std::exception& e) {
            msg.result_promise.set_exception(std::current_exception());
        }
    }

    // Lock-free SPMC queue
    moodycamel::ConcurrentQueue<Message> mailbox_;

    // State (no locks needed - single writer)
    std::map<std::string, PeerConnection> peer_connections_;
    std::map<std::string, LocalBuffer> local_buffers_;

    std::thread reactor_thread_;
    std::atomic<bool> running_{true};
};
```

## Implementation Steps

### Phase 1: Add Mailbox (Hybrid Model)

```cpp
class RdmaLazyPeer {
private:
    // Keep existing direct-call methods for data path
    std::shared_ptr<RDMAEndpoint> GetEndpoint(const std::string& id);

    // Add mailbox for control plane
    void Connect_Internal(Message msg);  // Runs on reactor thread

    std::future<bool> Connect(const std::string& id, const std::string& addr) {
        Message msg{MessageType::CONNECT, id, addr};
        auto future = msg.result_promise.get_future();
        control_mailbox_.send(std::move(msg));
        return future;
    }

    moodycamel::ConcurrentQueue<Message> control_mailbox_;
    std::thread control_thread_;
};
```

### Phase 2: Move Data Path to Reactor

```cpp
class RdmaLazyPeer {
public:
    // Even read/write go through mailbox
    std::future<void> write(const std::string& remote_id,
                           const std::vector<assign_tuple_t>& assign) {
        Message msg{MessageType::WRITE, remote_id, assign};
        auto future = msg.result_promise.get_future();
        mailbox_.send(std::move(msg));
        return future;
    }
};
```

### Phase 3: Integrate with RDMA Completion Events

```cpp
class RdmaReactor {
private:
    void run() {
        struct epoll_event events[64];

        while (running_) {
            // Poll both mailbox and RDMA completion channel
            int n = epoll_wait(epoll_fd_, events, 64, timeout_ms);

            for (int i = 0; i < n; i++) {
                if (events[i].data.ptr == &mailbox_) {
                    process_messages();
                } else if (events[i].data.ptr == rdma_completion_channel_) {
                    process_rdma_completions();
                }
            }
        }
    }
};
```

## Benefits of Full Reactor

1. **No lock contention**: Single writer to all state
2. **Better cache locality**: All state on one thread
3. **Deterministic**: Easier to reason about and test
4. **Scalable**: Can handle 1000+ connections efficiently
5. **Composable**: Easy to add timers, retries, etc.

## Drawbacks

1. **Latency**: Extra message passing hop (~1-5μs)
2. **Complexity**: Callback-based API is harder to use
3. **Debugging**: Harder to follow async flow
4. **Migration effort**: Significant code change

## Alternative: Hybrid Model (Recommended)

Keep fast path direct, slow path through reactor:

```cpp
class RdmaLazyPeer {
public:
    // Fast path: direct call (read lock only)
    std::shared_ptr<RDMAEndpoint> GetEndpoint(const std::string& id) {
        std::shared_lock lock(mutex_);
        return peer_connections_[id].endpoint;
    }

    // Data path: direct call
    std::future<void> write(const std::string& id, ...) {
        auto ep = GetEndpoint(id);  // Fast!
        return ep->write(...);
    }

    // Slow path: mailbox for complex state changes
    std::future<bool> Connect(const std::string& id, ...) {
        Message msg{MessageType::CONNECT, id, ...};
        // Goes through reactor
        return send_to_reactor(std::move(msg));
    }
};
```

## When to Use Each

| Operation      | Current (V2)  | Hybrid  | Full Reactor |
| -------------- | ------------- | ------- | ------------ |
| GetEndpoint    | Direct        | Direct  | Mailbox      |
| read/write     | Direct        | Direct  | Mailbox      |
| Connect        | Direct + lock | Mailbox | Mailbox      |
| RegisterBuffer | Direct + lock | Mailbox | Mailbox      |
| Close          | Direct + lock | Mailbox | Mailbox      |

## Performance Comparison

```
Operation latency (estimated):

Current V2:
  GetEndpoint: 50ns (atomic read + map lookup)
  read/write:  500ns (function call + queue + RDMA post)
  Connect:     10ms (network RTT + lock wait)

Hybrid:
  GetEndpoint: 50ns (same as V2)
  read/write:  500ns (same as V2)
  Connect:     10ms + 2μs mailbox (negligible)

Full Reactor:
  GetEndpoint: 50ns + 2μs mailbox (~40x slower)
  read/write:  500ns + 2μs mailbox (~5x slower)
  Connect:     10ms + 2μs mailbox (negligible)
```

## Recommendation

**Start with V2, profile, then consider Hybrid if needed.**

Only go Full Reactor if:

- 1000+ concurrent connections
- Lock contention >10% of runtime
- Need for complex async orchestration
