# Walkthrough - Fixing RDMA Multi-QP Hang and OOM

## Problem Description

When running `dlslime_torch_dist_sendrecv_bench.py` with `SLIME_QP_NUM=4`, the benchmark would hang or crash, while it worked fine with `SLIME_QP_NUM=1` or `2`.
The `endpoint_sendrecv_bench.py` (C++ benchmark) sometimes passed but `torch_bench` (Python wrapper) consistently failed.

## Root Cause Analysis

### 1. Race Condition in Context Assignments

**Diagnosis**: In `RDMAMsgEndpoint::sendProcess` (and `recvProcess`), we iterated over all QPs to post operations.

```cpp
for (size_t qpi = 0; qpi < num_qp_; ++qpi) {
    // RESET writes to the SAME assignment object
    s_ctx->data_send_assign_.reset(..., qpi, ...);
    // POST passes pointer to this object
    data_channel_->post_rc_oneside_batch(qpi, &(s_ctx->data_send_assign_));
}
```

Because `post_rc_oneside_batch` is asynchronous (or at least the completion handling is), reusing the *same* `data_send_assign_` object for multiple QPs meant that:

1. QP #0's callback/metadata was overwritten by QP #1's `reset`.
2. When QP #0 completed, it invoked the callback from QP #N (or corrupted state).
3. The signal mask `comm_done` never reached `expected_mask`, causing a hang.

### 2. Memory Exhaustion / Corruption

**Diagnosis**: To fix the race condition, we changed `RDMAAssign data_send_assign_` to an array `RDMAAssign data_send_assigns_[64]`.
However, `RDMAAssign` was defined as:

```cpp
struct RDMAAssign {
    // Assignment batch_[4096]; // Static array!
    // ...
};
```

With `sizeof(Assignment) ≈ 64 bytes`, one `RDMAAssign` was ~256 KB.
With 1024 slots * 2 (Send/Recv) * 64 QPs, the total memory required was:
`1024 * 2 * 64 * 256KB ≈ 32 GB`.
This massive allocation caused allocation failures or memory corruption/thrashing, leading to immediate hangs or crashes after the first fix.

## Implementation Details

### Fix 1: Isolation via Arrays

We modified `SendContext` and `RecvContext` to store an array of assignments, ensuring each QP has its own independent state.

**File**: `rdma_msg_endpoint.h`

```cpp
struct SendContext {
    // ...
    RDMAAssign data_send_assigns_[64]; // One per QP (up to MAX)
};
```

**File**: `rdma_msg_endpoint.cpp`

```cpp
// sendProcess
for (size_t qpi = 0; qpi < num_qp_; ++qpi) {
    // ...
    s_ctx->data_send_assigns_[qpi].reset(...); // Use specific slot
    data_channel_->post_rc_oneside_batch(qpi, &(s_ctx->data_send_assigns_[qpi]));
}
```

### Fix 2: Dynamic Memory for RDMAAssign

We refactored `RDMAAssign` to use `std::vector` instead of a large static array. This reduced the object size from ~256 KB to ~64 bytes (plus dynamic heap usage, which is small since we typically only use batch size = 1).

**File**: `rdma_assignment.h`

```cpp
struct RDMAAssign {
    // Assignment batch_[4096]; // REMOVED
    AssignmentBatch batch_;     // std::vector<Assignment>
};
```

**File**: `rdma_assignment.cpp` / `rdma_io_endpoint.cpp`
Updated all call sites to use `batch_.size()`, `batch_.resize()`, `batch_.emplace_back()`, removing direct array indexing and `memcpy`.

## Verification Results

Benchmark `SLIME_QP_NUM=4` now runs successfully with full performance.

**Performance (2 GPUs, H200)**:

| Message Size | Bandwidth  |
| :----------- | :--------- |
| 1 MB         | ~18 GB/s   |
| 16 MB        | ~43 GB/s   |
| 512 MB       | ~48.7 GB/s |

This confirms effective multi-QP utilization without hangs.
