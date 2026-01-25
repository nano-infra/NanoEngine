#pragma once
#include <cstdint>
#include <vector>

namespace spoke {

enum class Action : uint32_t {
    kInit            = 0x01,
    kInitRDMA        = 0x08,
    kRdmaRealloc     = 0x09,  // [New] Request remote buffer expansion
    kStop            = 0xFF,
    kNetSpawn        = 0x04,
    kHubRegisterNode = 0x30,
    kHubFindNode     = 0x31,
    kNetAllocate     = 0x32,  // [New] V2 Allocation
    kNetLaunch       = 0x33,  // [New] V2 Launch
    kNetRelease      = 0x34,  // [New] V2 Release
    kStreamPush      = 0x40,  // [New] Unsolicited Stream Message (Server -> Client)
    kUserActionStart = 0x10
};

// 网络包头 (Header)
struct NetHeader {
    uint32_t magic = 0x504F4B45;
    uint32_t meta_size;
    uint32_t data_size;
};

// 请求元数据 (Meta)
struct NetMeta {
    Action   action;
    uint32_t seq_id;
    char     actor_id[32];
    char     actor_type[32];
};

// 响应元数据 (RespMeta)
struct NetRespMeta {
    uint32_t seq_id;
    int      status;
    Action   action;  // [New] Added action field for async push
};

// IPC 管道消息头
struct PipeHeader {
    Action   action;
    uint32_t seq_id;
    uint32_t data_size;
};

// ==========================================
// [v2 Upgrade] Resource Management
// ==========================================
struct ResourceSpec {
    int32_t num_gpus = 0;
    int32_t num_cpus = 0;  // Reserved
};

struct RegisterNodeReq {
    char         ip[64];
    int32_t      port;
    ResourceSpec capacity;
};

struct FindNodeReq {
    ResourceSpec requirements;
};

// [New] Gang Allocation Request
struct AllocateReq {
    uint32_t     num_nodes;           // Number of nodes required
    uint32_t     actors_per_node;     // Actors per node (Total actors = num_nodes * actors_per_node)
    ResourceSpec res_per_actor;       // Resource requirement per actor
    bool         strict_pack = true;  // Require exact packing (true for typical TP/PP)
    char         master_node_ip[64];  // [New] Affinity: Prefer this node if available
};

// Detail for a single allocated slot
struct AllocatedSlot {
    char    node_ip[64];
    int32_t node_port;
    int32_t global_rank;  // Global rank in the gang (0..N-1)
    int32_t local_rank;   // Local rank in the node (0..M-1)
    int32_t gpu_id;       // Physical GPU ID assigned
};

// Response from Hub
struct AllocateResp {
    char     ticket_id[64];  // Allocation UUID
    uint32_t num_members;    // Total slots
    // Followed by num_members * AllocatedSlot (Variable length body)
};

// [New] Launch Request (Client -> Hub)
struct LaunchReq {
    char     ticket_id[64];
    uint32_t global_rank;
    // Followed by args (std::vector<string> serialized)
};

// [New] Spawn Request Body (Hub -> Agent)
// Replaces the empty body of kNetSpawn for V2 calls
struct SpawnActorReq {
    char         ticket_id[64];
    ResourceSpec resources;
    int32_t      assigned_gpu_ids[8];  // Fixed size for simplicity, supports up to 8 GPUs
    // Followed by args...
};

// [New] Release Request (Client -> Hub)
struct ReleaseReq {
    char ticket_id[64];
};

}  // namespace spoke
