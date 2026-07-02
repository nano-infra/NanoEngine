#include <arpa/inet.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>

#define CUDA_CHECK(cmd) do {                                      \
  cudaError_t e = (cmd);                                           \
  if (e != cudaSuccess) {                                          \
    fprintf(stderr, "CUDA failure %s:%d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(e));                                \
    std::abort();                                                  \
  }                                                               \
} while (0)

#define CU_CHECK(cmd) do {                                      \
  CUresult e = (cmd);                                           \
  if (e != CUDA_SUCCESS) {                                      \
    const char* s = nullptr;                                    \
    cuGetErrorString(e, &s);                                    \
    fprintf(stderr, "CUDA driver failure %s:%d: %s\n", __FILE__, __LINE__, \
            s ? s : "unknown");                                \
    std::abort();                                               \
  }                                                            \
} while (0)

#define NCCL_CHECK(cmd) do {                                      \
  ncclResult_t r = (cmd);                                          \
  if (r != ncclSuccess) {                                          \
    fprintf(stderr, "NCCL failure %s:%d: %s\n", __FILE__, __LINE__, \
            ncclGetErrorString(r));                                \
    std::abort();                                                  \
  }                                                               \
} while (0)

static int env_int(const char* name, int def) {
  const char* v = std::getenv(name);
  return v ? std::atoi(v) : def;
}

static std::string env_str(const char* name, const char* def) {
  const char* v = std::getenv(name);
  return v ? std::string(v) : std::string(def);
}

static void write_all(int fd, const void* buf, size_t n) {
  const char* p = static_cast<const char*>(buf);
  while (n) {
    ssize_t rc = ::write(fd, p, n);
    if (rc <= 0) throw std::runtime_error("write failed");
    p += rc;
    n -= static_cast<size_t>(rc);
  }
}

static void read_all(int fd, void* buf, size_t n) {
  char* p = static_cast<char*>(buf);
  while (n) {
    ssize_t rc = ::read(fd, p, n);
    if (rc <= 0) throw std::runtime_error("read failed");
    p += rc;
    n -= static_cast<size_t>(rc);
  }
}

static int connect_to(const std::string& host, int port) {
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* res = nullptr;
  const std::string port_s = std::to_string(port);
  if (getaddrinfo(host.c_str(), port_s.c_str(), &hints, &res) != 0)
    throw std::runtime_error("getaddrinfo failed");
  int fd = -1;
  for (addrinfo* p = res; p; p = p->ai_next) {
    fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (fd < 0) continue;
    if (connect(fd, p->ai_addr, p->ai_addrlen) == 0) break;
    close(fd);
    fd = -1;
  }
  freeaddrinfo(res);
  if (fd < 0) throw std::runtime_error("connect failed");
  return fd;
}

static int listen_on(int port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) throw std::runtime_error("socket failed");
  int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(static_cast<uint16_t>(port));
  if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0)
    throw std::runtime_error("bind failed");
  if (listen(fd, 128) != 0) throw std::runtime_error("listen failed");
  return fd;
}

static void broadcast_nccl_id(int rank, int world, ncclUniqueId* id) {
  // Minimal TCP bootstrap so the binary can be launched by torchrun --no-python
  // without depending on MPI, PyTorch, or DeepEP runtime code.
  const std::string master = env_str("MASTER_ADDR", "127.0.0.1");
  const int port = env_int("MASTER_PORT", 29500) + 137;
  if (rank == 0) {
    int lfd = listen_on(port);
    for (int i = 1; i < world; ++i) {
      int cfd = accept(lfd, nullptr, nullptr);
      if (cfd < 0) throw std::runtime_error("accept failed");
      write_all(cfd, id, sizeof(*id));
      close(cfd);
    }
    close(lfd);
  } else {
    int fd = -1;
    for (int tries = 0; tries < 200; ++tries) {
      try {
        fd = connect_to(master, port);
        break;
      } catch (...) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    }
    if (fd < 0) throw std::runtime_error("could not connect to rank0 bootstrap");
    read_all(fd, id, sizeof(*id));
    close(fd);
  }
}

static void* alloc_host_numa(size_t requested, size_t* actual_size) {
  // NCCL symmetric host windows require CUDA VMM HOST_NUMA memory. Plain
  // malloc/cudaHostAlloc is not enough for ncclCommWindowRegister here.
  CUdevice current_dev;
  int cuda_dev = 0;
  int numa = 0;
  CUmemAllocationProp prop{};
  CUmemAccessDesc access[2]{};
  CUmemGenericAllocationHandle handle;
  size_t granularity = 0;
  CUdeviceptr ptr = 0;

  CUDA_CHECK(cudaGetDevice(&cuda_dev));
  CU_CHECK(cuDeviceGet(&current_dev, cuda_dev));
  if (cuDeviceGetAttribute(&numa, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, current_dev) != CUDA_SUCCESS || numa < 0)
    numa = 0;

  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  prop.location.id = numa;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  size_t size = ((requested + granularity - 1) / granularity) * granularity;
  CU_CHECK(cuMemCreate(&handle, size, &prop, 0));
  CU_CHECK(cuMemAddressReserve(&ptr, size, granularity, 0, 0));
  CU_CHECK(cuMemMap(ptr, size, 0, handle, 0));

  access[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access[0].location.id = cuda_dev;
  access[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  access[1].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  access[1].location.id = numa;
  access[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CU_CHECK(cuMemSetAccess(ptr, size, access, 2));
  CU_CHECK(cuMemRelease(handle));
  *actual_size = size;
  return reinterpret_cast<void*>(ptr);
}

static void free_host_numa(void* p) {
  if (!p) return;
  size_t size = 0;
  CU_CHECK(cuMemGetAddressRange(nullptr, &size, reinterpret_cast<CUdeviceptr>(p)));
  CU_CHECK(cuMemUnmap(reinterpret_cast<CUdeviceptr>(p), size));
  CU_CHECK(cuMemAddressFree(reinterpret_cast<CUdeviceptr>(p), size));
}

__global__ void init_u32(uint32_t* ptr, int n, uint32_t base) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) ptr[i] = base + static_cast<uint32_t>(i);
}

__global__ void hbm_to_remote_dram_kernel(ncclDevComm dev_comm, ncclWindow_t dev_win,
                                          ncclWindow_t host_win, int nbytes, int peer) {
  ncclGin gin{dev_comm, 0};
  uint64_t signal_value = gin.readSignal(0);
  ncclGinBarrierSession<ncclCoopCta> bar{ncclCoopCta(), gin, ncclTeamTagWorld(), 0};
  // The world GIN barrier keeps both ranks inside the kernel while one-sided
  // network operations are issued and completed.
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire, ncclGinFenceLevel::Relaxed);
  // RDMA write: local HBM window is the source, peer HOST_NUMA DRAM window is
  // the destination. SegmentMixed tells NCCL that at least one window may be
  // CPU-backed HOST_NUMA memory rather than device-only memory.
  gin.put(ncclTeamWorld(dev_comm), peer,
          host_win, 0,
          dev_win, 0,
          static_cast<size_t>(nbytes),
          ncclGin_None{},
          ncclGin_None{},
          ncclCoopCta{},
          ncclGin_None{},
          cuda::thread_scope_thread,
          cuda::thread_scope_system,
          ncclGinOptFlagsDefault,
          ncclGin_SegmentMixed{});
  gin.flush(ncclCoopCta());
  bar.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
}

__global__ void remote_dram_to_hbm_kernel(ncclDevComm dev_comm, ncclWindow_t host_win,
                                          ncclWindow_t dev_win, int nbytes, int peer) {
  ncclGin gin{dev_comm, 0};
  ncclGinBarrierSession<ncclCoopCta> bar{ncclCoopCta(), gin, ncclTeamTagWorld(), 0};
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire, ncclGinFenceLevel::Relaxed);
  // RDMA read/get: peer HOST_NUMA DRAM window is the source, local HBM window is
  // the destination.
  gin.get(ncclTeamWorld(dev_comm), peer,
          host_win, 0,
          dev_win, 0,
          static_cast<size_t>(nbytes),
          ncclCoopCta{},
          ncclGin_None{},
          ncclGinOptFlagsDefault,
          ncclGin_SegmentMixed{});
  gin.flush(ncclCoopCta());
  bar.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
}

int main(int argc, char** argv) {
  const int rank = env_int("RANK", 0);
  const int world = env_int("WORLD_SIZE", 1);
  const int local_rank = env_int("LOCAL_RANK", rank);
  const int count = argc > 1 ? std::atoi(argv[1]) : 1024;
  const int iters = argc > 2 ? std::atoi(argv[2]) : 100;
  const size_t bytes = static_cast<size_t>(count) * sizeof(uint32_t);
  if (world < 2) {
    fprintf(stderr, "Need WORLD_SIZE >= 2\n");
    return 1;
  }

  setenv("NCCL_ELASTIC_BUFFER_REGISTER", "1", 0);
  CU_CHECK(cuInit(0));
  CUDA_CHECK(cudaSetDevice(local_rank));

  ncclUniqueId id;
  if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&id));
  broadcast_nccl_id(rank, world, &id);

  ncclComm_t comm;
  NCCL_CHECK(ncclCommInitRank(&comm, world, id, rank));

  ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
  NCCL_CHECK(ncclCommQueryProperties(comm, &props));
  printf("rank %d props: deviceApi=%d ginType=%d railedGinType=%d hostRma=%d nRanks=%d\n",
         rank, props.deviceApiSupport, props.ginType, props.railedGinType,
         props.hostRmaSupport, props.nRanks);
  fflush(stdout);

  void* d_buf = nullptr;
  NCCL_CHECK(ncclMemAlloc(&d_buf, bytes));
  size_t host_alloc_size = 0;
  uint32_t* h_buf = static_cast<uint32_t*>(alloc_host_numa(bytes, &host_alloc_size));

  for (int i = 0; i < count; ++i) h_buf[i] = 0xdead0000u + static_cast<uint32_t>(rank);
  init_u32<<<(count + 255) / 256, 256>>>(static_cast<uint32_t*>(d_buf), count,
                                         0x10000000u + static_cast<uint32_t>(rank) * 0x10000u);
  CUDA_CHECK(cudaDeviceSynchronize());

  ncclWindow_t dev_win = nullptr, host_win = nullptr;
  // Symmetric windows export both buffers to NCCL Device API. Device kernels use
  // these opaque window handles instead of raw pointers for remote put/get.
  NCCL_CHECK(ncclCommWindowRegister(comm, d_buf, bytes, &dev_win, NCCL_WIN_COLL_SYMMETRIC));
  NCCL_CHECK(ncclCommWindowRegister(comm, h_buf, bytes, &host_win, NCCL_WIN_COLL_SYMMETRIC));
  if (!dev_win || !host_win) {
    fprintf(stderr, "rank %d failed to register NCCL windows: dev=%p host=%p\n", rank, dev_win, host_win);
    return 2;
  }

  ncclDevComm_t dev_comm;
  ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  // Request one GIN context and one world barrier/signal slot. With
  // NCCL_GIN_TYPE=3 this exercises the GDAKI GPU-initiated networking path.
  reqs.ginContextCount = 1;
  reqs.ginExclusiveContexts = true;
  reqs.ginQueueDepth = props.ginType == NCCL_GIN_TYPE_PROXY ? 0 : 1024;
  reqs.ginSignalCount = 1;
  reqs.worldGinBarrierCount = 1;
  reqs.ginConnectionType = props.ginType != NCCL_GIN_TYPE_NONE ? NCCL_GIN_CONNECTION_FULL : NCCL_GIN_CONNECTION_RAIL;
  NCCL_CHECK(ncclDevCommCreate(comm, &reqs, &dev_comm));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  int peer = (rank + 1) % world;
  int prev = (rank + world - 1) % world;

  hbm_to_remote_dram_kernel<<<1, 128, 0, stream>>>(dev_comm, dev_win, host_win, static_cast<int>(bytes), peer);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));

  bool ok_put = true;
  for (int i = 0; i < count; ++i) {
    uint32_t expected = 0x10000000u + static_cast<uint32_t>(prev) * 0x10000u + static_cast<uint32_t>(i);
    if (h_buf[i] != expected) {
      fprintf(stderr, "rank %d HBM->DRAM mismatch i=%d got=0x%x expected=0x%x\n", rank, i, h_buf[i], expected);
      ok_put = false;
      break;
    }
  }
  printf("rank %d HBM->remote-DRAM %s\n", rank, ok_put ? "PASSED" : "FAILED");
  fflush(stdout);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));
  // Simple smoke bandwidth: includes kernel launch, GIN barrier, and flush cost.
  // It proves the path works and gives a rough throughput number, not a tuned
  // peak RDMA benchmark.
  for (int i = 0; i < iters; ++i) {
    hbm_to_remote_dram_kernel<<<1, 128, 0, stream>>>(dev_comm, dev_win, host_win, static_cast<int>(bytes), peer);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float put_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&put_ms, start, stop));
  double put_gbps = (static_cast<double>(bytes) * iters) / (put_ms / 1e3) / 1e9;
  printf("rank %d HBM->remote-DRAM bandwidth %.3f GB/s over %d iters, %.2f MiB each\n",
         rank, put_gbps, iters, static_cast<double>(bytes) / (1024.0 * 1024.0));
  fflush(stdout);

  for (int i = 0; i < count; ++i)
    h_buf[i] = 0x20000000u + static_cast<uint32_t>(rank) * 0x10000u + static_cast<uint32_t>(i);
  CUDA_CHECK(cudaMemsetAsync(d_buf, 0, bytes, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  remote_dram_to_hbm_kernel<<<1, 128, 0, stream>>>(dev_comm, host_win, dev_win, static_cast<int>(bytes), peer);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));

  uint32_t* out = static_cast<uint32_t*>(std::malloc(bytes));
  CUDA_CHECK(cudaMemcpy(out, d_buf, bytes, cudaMemcpyDeviceToHost));
  bool ok_get = true;
  for (int i = 0; i < count; ++i) {
    uint32_t expected = 0x20000000u + static_cast<uint32_t>(peer) * 0x10000u + static_cast<uint32_t>(i);
    if (out[i] != expected) {
      fprintf(stderr, "rank %d remote-DRAM->HBM mismatch i=%d got=0x%x expected=0x%x\n", rank, i, out[i], expected);
      ok_get = false;
      break;
    }
  }
  printf("rank %d remote-DRAM->HBM %s\n", rank, ok_get ? "PASSED" : "FAILED");
  fflush(stdout);

  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < iters; ++i) {
    remote_dram_to_hbm_kernel<<<1, 128, 0, stream>>>(dev_comm, host_win, dev_win, static_cast<int>(bytes), peer);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float get_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&get_ms, start, stop));
  double get_gbps = (static_cast<double>(bytes) * iters) / (get_ms / 1e3) / 1e9;
  printf("rank %d remote-DRAM->HBM bandwidth %.3f GB/s over %d iters, %.2f MiB each\n",
         rank, get_gbps, iters, static_cast<double>(bytes) / (1024.0 * 1024.0));
  fflush(stdout);

  NCCL_CHECK(ncclCommWindowDeregister(comm, host_win));
  NCCL_CHECK(ncclCommWindowDeregister(comm, dev_win));
  NCCL_CHECK(ncclDevCommDestroy(comm, &dev_comm));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaStreamDestroy(stream));
  free(out);
  free_host_numa(h_buf);
  NCCL_CHECK(ncclMemFree(d_buf));
  NCCL_CHECK(ncclCommFinalize(comm));
  NCCL_CHECK(ncclCommDestroy(comm));
  return (ok_put && ok_get) ? 0 : 3;
}
