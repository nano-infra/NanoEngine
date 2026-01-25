#pragma once

#include <algorithm>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include "spoke/csrc/types.h"

namespace spoke {

struct NodeInfo {
    std::string  ip;
    int          port;
    int          load;   // Active actors count
    ResourceSpec total;  // Total hardware capacity
    ResourceSpec used;   // Currently allocated
};

// [New] Internal struct to track allocation details
struct AllocationState {
    std::string                ticket_id;
    std::vector<AllocatedSlot> slots;
    ResourceSpec               res_per_actor;

    struct LaunchedActor {
        std::string actor_id;
        std::string node_ip;
        int         node_port;
    };
    std::vector<LaunchedActor> launched_actors;
};

class Hub {
public:
    void registerNode(const std::string& ip, int port, const ResourceSpec& capacity)
    {
        std::lock_guard<std::mutex> lk(mtx_);
        // Check for existing node (update if exists)
        for (auto& node : nodes_) {
            if (node.ip == ip && node.port == port) {
                node.total = capacity;  // Update capacity
                std::cout << "[Hub] Updated Node: " << ip << ":" << port << " GPUs=" << capacity.num_gpus << std::endl;
                return;
            }
        }
        NodeInfo info;
        info.ip    = ip;
        info.port  = port;
        info.load  = 0;
        info.total = capacity;
        info.used  = {0, 0};

        nodes_.push_back(info);
        std::cout << "[Hub] Registered Node: " << ip << ":" << port << " GPUs=" << capacity.num_gpus << std::endl;
    }

    // Schedule a node based on resource requirements (Single Node)
    bool scheduleNode(const ResourceSpec& req, NodeInfo* out_node)
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (nodes_.empty())
            return false;

        // Strategy: Find node with enough FREE resources and minimal load
        // Free = Total - Used

        NodeInfo* best_candidate = nullptr;

        for (auto& node : nodes_) {
            int free_gpus = node.total.num_gpus - node.used.num_gpus;

            if (free_gpus >= req.num_gpus) {
                if (!best_candidate || node.load < best_candidate->load) {
                    best_candidate = &node;
                }
            }
        }

        if (best_candidate) {
            *out_node = *best_candidate;
            // Tentatively reserve resources (Simplified: Assuming atomic commit happens here)
            best_candidate->load++;
            best_candidate->used.num_gpus += req.num_gpus;

            std::cout << "[Hub] Scheduled " << best_candidate->ip
                      << " (Free GPUs: " << (best_candidate->total.num_gpus - best_candidate->used.num_gpus) << ")"
                      << std::endl;
            return true;
        }

        std::cerr << "[Hub] Scheduling Failed: No node has " << req.num_gpus << " free GPUs." << std::endl;
        return false;
    }

    // [New] Multi-Node Allocation
    bool gangAllocate(const AllocateReq& req, std::string& out_ticket, std::vector<AllocatedSlot>& out_slots)
    {
        std::lock_guard<std::mutex> lk(mtx_);

        // 1. Find 'num_nodes' that satisfy 'actors_per_node' * 'res_per_actor'
        std::vector<NodeInfo*> candidate_nodes;

        // Affinity Logic: If master_node_ip is set, try to find it first.
        std::string master_ip = req.master_node_ip;
        if (!master_ip.empty()) {
            for (auto& node : nodes_) {
                if (node.ip == master_ip) {
                    int free_gpus = node.total.num_gpus - node.used.num_gpus;
                    int req_gpus  = req.actors_per_node * req.res_per_actor.num_gpus;
                    if (free_gpus >= req_gpus) {
                        candidate_nodes.push_back(&node);
                        std::cout << "[Hub] Affinity match found: " << master_ip << std::endl;
                    }
                    else {
                        std::cout << "[Hub] Affinity node " << master_ip << " has insufficient resources." << std::endl;
                    }
                    break;
                }
            }
        }

        // Simple greedy: Find remaining nodes that fit
        for (auto& node : nodes_) {
            // Skip if already added (affinity node)
            bool already_added = false;
            for (auto* c : candidate_nodes) {
                if (c->ip == node.ip && c->port == node.port) {
                    already_added = true;
                    break;
                }
            }
            if (already_added)
                continue;

            int free_gpus = node.total.num_gpus - node.used.num_gpus;
            int req_gpus  = req.actors_per_node * req.res_per_actor.num_gpus;

            if (free_gpus >= req_gpus) {
                candidate_nodes.push_back(&node);
                if (candidate_nodes.size() == req.num_nodes)
                    break;
            }
        }

        if (candidate_nodes.size() < req.num_nodes) {
            std::cerr << "[Hub] Gang Allocation Failed: Need " << req.num_nodes << " nodes with "
                      << (req.actors_per_node * req.res_per_actor.num_gpus) << " GPUs each. Found "
                      << candidate_nodes.size() << std::endl;
            return false;
        }

        // 2. Commit Reservation
        std::string ticket      = generateTicket();
        int         global_rank = 0;

        for (auto* node : candidate_nodes) {
            // Assign slots in this node
            for (uint32_t i = 0; i < req.actors_per_node; ++i) {
                AllocatedSlot slot;
                strncpy(slot.node_ip, node->ip.c_str(), 63);
                slot.node_port   = node->port;
                slot.global_rank = global_rank++;
                slot.local_rank  = i;

                // Assign specific GPU ID (Simple linear allocation from current usage)
                // Assuming used_gpus are [0, 1, ..., used-1], we assign [used, used+1, ...]
                // IMPORTANT: This assumes simple fragmentation. Real GPU allocator needs bitmap.
                if (req.res_per_actor.num_gpus > 0) {
                    slot.gpu_id = node->used.num_gpus + (i * req.res_per_actor.num_gpus);
                }
                else {
                    slot.gpu_id = -1;  // No GPU assigned
                }
                // Note: Only supporting contiguous 1-GPU allocation for now.

                out_slots.push_back(slot);
            }

            // Update Node State
            node->load += req.actors_per_node;
            node->used.num_gpus += (req.actors_per_node * req.res_per_actor.num_gpus);

            std::cout << "[Hub] Gang Reserved on " << node->ip << ": " << req.actors_per_node
                      << " actors. Free GPUs: " << (node->total.num_gpus - node->used.num_gpus) << std::endl;
        }

        AllocationState state;
        state.ticket_id      = ticket;
        state.slots          = out_slots;
        state.res_per_actor  = req.res_per_actor;
        allocations_[ticket] = state;

        out_ticket = ticket;
        return true;
    }

    // [New] Launch Actor based on Ticket
    bool launchActor(const LaunchReq&   req,
                     const std::string& actor_type,
                     const std::string& actor_id,
                     const std::string& args_body)
    {
        std::lock_guard<std::mutex> lk(mtx_);

        // 1. Verify Ticket
        if (allocations_.find(req.ticket_id) == allocations_.end()) {
            std::cerr << "[Hub] Launch Failed: Invalid Ticket " << req.ticket_id << std::endl;
            return false;
        }

        auto& state = allocations_[req.ticket_id];
        if (req.global_rank >= state.slots.size()) {
            std::cerr << "[Hub] Launch Failed: Rank " << req.global_rank << " out of bounds" << std::endl;
            return false;
        }

        const auto& slot = state.slots[req.global_rank];

        // 2. Connect to Agent
        int                agent_sock = socket(AF_INET, SOCK_STREAM, 0);
        struct sockaddr_in sa;
        sa.sin_family = AF_INET;
        sa.sin_port   = htons(slot.node_port);
        inet_pton(AF_INET, slot.node_ip, &sa.sin_addr);

        if (connect(agent_sock, (struct sockaddr*)&sa, sizeof(sa)) < 0) {
            close(agent_sock);
            std::cerr << "[Hub] Launch Failed: Cannot connect to Agent " << slot.node_ip << ":" << slot.node_port
                      << std::endl;
            return false;
        }

        // 3. Construct SpawnActorReq (The Payload for Agent)
        SpawnActorReq spawn_req;
        memset(&spawn_req, 0, sizeof(spawn_req));
        strncpy(spawn_req.ticket_id, req.ticket_id, 63);
        spawn_req.resources = state.res_per_actor;

        // Fill assigned GPUs
        // TODO: Support assigning multiple GPUs if res_per_actor > 1
        for (int i = 0; i < 8; ++i)
            spawn_req.assigned_gpu_ids[i] = -1;

        if (state.res_per_actor.num_gpus > 0) {
            spawn_req.assigned_gpu_ids[0] = slot.gpu_id;
        }

        // Concatenate SpawnReq + Args
        std::vector<char> payload(sizeof(SpawnActorReq) + args_body.size());
        memcpy(payload.data(), &spawn_req, sizeof(SpawnActorReq));
        if (!args_body.empty()) {
            memcpy(payload.data() + sizeof(SpawnActorReq), args_body.data(), args_body.size());
        }

        // 4. Send kNetSpawn to Agent
        NetMeta meta{Action::kNetSpawn, 0, "", ""};
        strncpy(meta.actor_id, actor_id.c_str(), 31);
        strncpy(meta.actor_type, actor_type.c_str(), 31);

        NetHeader hdr{0x504F4B45, sizeof(NetMeta), (uint32_t)payload.size()};

        send(agent_sock, &hdr, sizeof(hdr), 0);
        send(agent_sock, &meta, sizeof(meta), 0);
        send(agent_sock, payload.data(), payload.size(), 0);

        // 5. Wait for Agent Ack
        NetHeader rh;
        recv(agent_sock, &rh, sizeof(rh), MSG_WAITALL);
        NetRespMeta rm;
        recv(agent_sock, &rm, sizeof(rm), MSG_WAITALL);

        close(agent_sock);

        if (rm.status > 0) {
            state.launched_actors.push_back({actor_id, slot.node_ip, slot.node_port});
            std::cout << "[Hub] Launched Rank " << req.global_rank << " (" << actor_id << ") on " << slot.node_ip
                      << " GPU:" << slot.gpu_id << std::endl;
            return true;
        }

        return false;
    }

    // [New] Release allocated resources
    bool gangRelease(const std::string& ticket_id)
    {
        std::lock_guard<std::mutex> lk(mtx_);
        auto                        it = allocations_.find(ticket_id);
        if (it == allocations_.end()) {
            std::cerr << "[Hub] Release Failed: Invalid Ticket " << ticket_id << std::endl;
            return false;
        }

        AllocationState& state = it->second;
        std::cout << "[Hub] Releasing resources for Ticket: " << ticket_id << std::endl;

        // 1. Physically STOP all launched actors
        for (const auto& la : state.launched_actors) {
            std::cout << "[Hub]   -> Stopping actor: " << la.actor_id << " on " << la.node_ip << ":" << la.node_port
                      << std::endl;

            int                agent_sock = socket(AF_INET, SOCK_STREAM, 0);
            struct sockaddr_in sa;
            sa.sin_family = AF_INET;
            sa.sin_port   = htons(la.node_port);
            inet_pton(AF_INET, la.node_ip.c_str(), &sa.sin_addr);

            if (connect(agent_sock, (struct sockaddr*)&sa, sizeof(sa)) >= 0) {
                NetMeta meta{Action::kStop, 0, "", ""};
                strncpy(meta.actor_id, la.actor_id.c_str(), 31);
                NetHeader hdr{0x504F4B45, sizeof(NetMeta), 0};

                send(agent_sock, &hdr, sizeof(hdr), 0);
                send(agent_sock, &meta, sizeof(meta), 0);
                close(agent_sock);
            }
            else {
                std::cerr << "[Hub]   -> Failed to connect to Agent to stop actor " << la.actor_id << std::endl;
                close(agent_sock);
            }
        }

        // 2. Update node resources (Bookkeeping)
        for (const auto& slot : state.slots) {
            for (auto& node : nodes_) {
                if (node.ip == slot.node_ip && node.port == slot.node_port) {
                    node.load--;
                    node.used.num_gpus -= state.res_per_actor.num_gpus;
                    std::cout << "[Hub]   -> Node " << node.ip << ":" << node.port << " recovered "
                              << state.res_per_actor.num_gpus << " GPU(s). "
                              << "Free GPUs: " << (node.total.num_gpus - node.used.num_gpus) << std::endl;
                    break;
                }
            }
        }
        allocations_.erase(it);
        return true;
    }

private:
    std::string generateTicket()
    {
        static std::random_device              rd;
        static std::mt19937                    gen(rd());
        static std::uniform_int_distribution<> dis(0, 15);
        const char*                            hex = "0123456789abcdef";
        std::string                            s   = "ticket_";
        for (int i = 0; i < 8; ++i)
            s += hex[dis(gen)];
        return s;
    }

    std::vector<NodeInfo>                  nodes_;
    std::map<std::string, AllocationState> allocations_;
    std::mutex                             mtx_;
};

}  // namespace spoke
