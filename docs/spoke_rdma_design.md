# Spoke Deep Integration: RDMA Zero-Copy Serialization

This document details the "True Zero-Copy" (Zero-ish) serialization path implemented in Spoke.

## 1. Overview

We have eliminated the intermediate `std::string` copies in the critical RDMA path.

### Previous Path (One Copy)

1. User Data -> `Pack()` -> `std::string` (Heap Alloc + Copy)
2. `std::string` -> `memcpy` -> `Pinned Buffer`
3. RDMA Transfer
4. `Pinned Buffer` -> `std::string` (Heap Alloc + Copy)
5. `std::string` -> `Unpack()` -> User Data

### New Path (Direct Serialization)

1. User Data -> `PackTo()` -> `Pinned Buffer` (Direct Write)
2. RDMA Transfer
3. `Pinned Buffer` -> `MessageView` (No Alloc)
4. `MessageView` -> `UnpackFrom()` -> User Data

**Result**: We have removed **2 unnecessary heap allocations and memory copies** per request.

## 2. API Changes

### Client Side

Transparent to the user. `callRemote` automatically uses `PackedSize` and `PackTo` when RDMA is active.

### Actor Side

Handlers now receive `spoke::MessageView` instead of `std::string` (internally). The `SPOKE_METHOD` macro has been updated to use `UnpackFrom` with this view.

```cpp
// Internal Handler Signature
using MethodHandler = std::function<std::string(MessageView)>;
```

## 3. Performance Impact

- **Latency**: Significant reduction for small messages (due to alloc avoidance) and large messages (due to copy avoidance).
- **CPU**: Reduced CPU usage on both Client and Actor by bypassing the memory allocator.

## 4. Remaining Copy

There is still **1 Copy** from User Data to Pinned Buffer (Client) and Pinned Buffer to User Data (Actor).
To achieve **True Hardware Zero Copy** (DMA directly from User `std::vector`), we would need a custom Allocator that acts on Pinned Memory directly. This is a potential future optimization.
