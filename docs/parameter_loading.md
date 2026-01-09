# NanoDeploy Parameter Loading Architecture

NanoDeploy implements a high-performance parameter loading system designed to minimize cold start latency for inference services. It supports both efficient local loading via memory mapping and reduced-latency remote transmission using RDMA.

## Local Loading Strategy

For local model weights, NanoDeploy utilizes a specialized `SafeTensorLoader` that leverages system-level optimizations for speed and memory efficiency.

### Key Features

- **Format Support**: Native support for HuggingFace `safetensors` format, avoiding the overhead of ensuring pickle safety or conversion.
- **Memory Mapping (`mmap`)**: Weights are mapped directly into the process address space. This allows for:
  - **Zero-Copy Loading**: Data is not copied from kernel to userspace buffers; it is accessed directly from the file cache.
  - **Instant Access**: Large models (70B+) can be "loaded" in milliseconds, as pages are faulted in only when accessed (lazy loading) or prefetched efficiently.
  - **Shared Memory**: Multiple workers on the same node can share the read-only physical memory pages of the model weights, significantly reducing total memory footprint.

### Implementation

The core logic is encapsulated in `nanodeploy::SafeTensorLoader` (C++):

```cpp
// Direct mmap of the .safetensors file
addr_ = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);

// Zero-copy tensor creation from blob
auto options = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
torch::Tensor t = torch::from_blob(data_ptr, shape, options);
```

## Distributed Loading (NanoStore Integration)

For distributed inference scenarios where weights must be fetched from a centralized Parameter Server (PS), NanoDeploy integrates with **NanoStore**.

### Architecture

- **Transport Layer**: Built on **DLSlime**, utilizing **RDMA** (Remote Direct Memory Access) for high-throughput, low-latency transmission.
- **Protocol**: Uses a customized protocol (`ps_request.py`) to negotiate memory regions and initiate one-sided RDMA operations.
- **Efficiency**:
  - **Bypass CPU**: RDMA allows direct placement of weight data into the receiver's memory (RAM or potentially GPU memory via GPUDirect), bypassing the receiver's CPU.
  - **Parallelism**: Parameter transmission is parallelized across multiple network interface cards (NICs) if available.

### Workflow

1. **Request**: Worker requests specific layers or tensors from the NanoStore PS.
2. **Registration**: `BufferManager` on PS and Worker register memory regions.
3. **Transfer**: Data is pushed/pulled via RDMA WRITE/READ operations.
4. **Inference**: Weights are immediately available for computation upon transfer completion.

## Future Optimize

- **GPUDirect Storage (GDS)**: Enabling direct loading from NVMe to GPU memory.
- **Peer-to-Peer Migration**: Using Spoke + DLSlime to migrate KV cache and weights between workers for load balancing.
