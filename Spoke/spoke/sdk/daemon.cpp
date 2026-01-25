#include "daemon.h"
#include <arpa/inet.h>
#include <cstdlib>  // For getenv
#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "spoke/csrc/agent.h"
#include "spoke/csrc/types.h"

namespace spoke {

// Global map for actor ownership
static std::mutex                 client_map_mtx;
static std::map<std::string, int> actor_owner_map;  // ActorID -> ClientFD

std::string expand_path(const std::string& path)
{
    if (!path.empty() && path[0] == '~') {
        const char* home = std::getenv("HOME");
        if (home) {
            return std::string(home) + path.substr(1);
        }
    }
    return path;
}

void register_to_hub(const std::string& hub_ip, int hub_port, int my_port, const ResourceSpec& capacity)
{
    int                sock = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in sa;
    sa.sin_family = AF_INET;
    sa.sin_port   = htons(hub_port);
    inet_pton(AF_INET, hub_ip.c_str(), &sa.sin_addr);
    if (connect(sock, (struct sockaddr*)&sa, sizeof(sa)) < 0) {
        close(sock);
        std::cerr << "[Daemon] Failed to connect to Hub at " << hub_ip << ":" << hub_port << std::endl;
        return;
    }

    RegisterNodeReq req;
    strncpy(req.ip, "127.0.0.1", sizeof(req.ip) - 1);  // Ideally, get actual IP
    req.port     = my_port;
    req.capacity = capacity;

    NetMeta   meta{Action::kHubRegisterNode, 0, "", ""};
    NetHeader hdr{0x504F4B45, sizeof(NetMeta), sizeof(RegisterNodeReq)};

    send(sock, &hdr, sizeof(hdr), 0);
    send(sock, &meta, sizeof(meta), 0);
    send(sock, &req, sizeof(req), 0);

    NetHeader rh;
    recv(sock, &rh, sizeof(rh), MSG_WAITALL);
    NetRespMeta rm;
    recv(sock, &rm, sizeof(rm), MSG_WAITALL);
    if (rm.status > 0)
        std::cout << "[Daemon] Registered to Hub with " << capacity.num_gpus << " GPUs." << std::endl;
    else
        std::cerr << "[Daemon] Registration failed." << std::endl;
    close(sock);
}

void handle_client(int client_fd, Agent* agent)
{
    while (true) {
        NetHeader hdr;
        if (recv(client_fd, &hdr, sizeof(NetHeader), MSG_WAITALL) <= 0)
            break;
        NetMeta meta;
        if (hdr.meta_size > 0)
            recv(client_fd, &meta, hdr.meta_size, MSG_WAITALL);
        std::string body;
        body.resize(hdr.data_size);
        if (hdr.data_size > 0)
            recv(client_fd, body.data(), hdr.data_size, MSG_WAITALL);

        if (meta.action == Action::kNetSpawn) {
            std::cout << "[Daemon] Spawning: " << meta.actor_type << std::endl;

            // Check for V2 Spawn Body
            std::vector<int> gpu_ids;
            std::string      ticket_id = "";
            if (body.size() >= sizeof(SpawnActorReq)) {
                // V2 Launch (with Resource Spec)
                const auto* req = reinterpret_cast<const SpawnActorReq*>(body.data());
                ticket_id       = req->ticket_id;
                for (int i = 0; i < 8; ++i) {
                    if (req->assigned_gpu_ids[i] >= 0) {
                        gpu_ids.push_back(req->assigned_gpu_ids[i]);
                    }
                }
                // Args are after the struct, if any (not implemented yet)
            }
            else {
                // Legacy V1 Spawn (Empty Body or just string args)
                // Default: no GPU isolation
            }

            agent->spawnActor(meta.actor_type, meta.actor_id, gpu_ids, ticket_id);

            // [New] Register owner
            {
                std::lock_guard<std::mutex> lk(client_map_mtx);
                actor_owner_map[meta.actor_id] = client_fd;
            }

            std::cout << "[Daemon] Spawned actor: " << meta.actor_id
                      << " Ticket: " << (ticket_id.empty() ? "N/A" : ticket_id) << " GPUs: " << gpu_ids.size()
                      << std::endl;

            // Send Ack
            NetHeader   rh{0x504F4B45, sizeof(NetRespMeta), 0};
            NetRespMeta rm{meta.seq_id, 1, Action::kNetSpawn};  // Success

            send(client_fd, &rh, sizeof(rh), 0);
            send(client_fd, &rm, sizeof(rm), 0);
        }
        else {
            // [Fix] Update actor owner to the REAL client (not Hub) when we receive direct RPC
            // This ensures push messages go to the actual client, not the Hub that forwarded spawn
            {
                std::lock_guard<std::mutex> lk(client_map_mtx);
                if (actor_owner_map.count(meta.actor_id) == 0 || actor_owner_map[meta.actor_id] != client_fd) {
                    std::cout << "[Daemon] Updating owner of " << meta.actor_id << " to client_fd=" << client_fd
                              << std::endl;
                    actor_owner_map[meta.actor_id] = client_fd;
                }
            }

            // [Fix] For user actions, dispatch asynchronously and handle responses in background
            // This allows collective operations (like NVSHMEM barriers) to work correctly
            std::cout << "[Daemon] Forwarding action " << static_cast<int>(meta.action) << " to actor " << meta.actor_id
                      << " (Seq: " << meta.seq_id << ")" << std::endl;

            auto future = agent->callRemoteRaw(meta.actor_id, meta.action, body);

            // Capture by value for thread safety
            uint32_t    seq_id_copy    = meta.seq_id;
            int         client_fd_copy = client_fd;
            std::string actor_id_copy  = meta.actor_id;

            std::thread([future = std::move(future), seq_id_copy, client_fd_copy, actor_id_copy]() mutable {
                std::string res = future.get();
                std::cout << "[Daemon] Received response from actor " << actor_id_copy << " (Seq: " << seq_id_copy
                          << ", Size: " << res.size() << "). Sending back to client..." << std::endl;

                NetHeader   rh{0x504F4B45, sizeof(NetRespMeta), (uint32_t)res.size()};
                NetRespMeta rm{seq_id_copy, 1, Action::kUserActionStart};  // Default action for generic response?
                // Note: Generic response doesn't really need action, but struct now has it.
                // Action::kUserActionStart (16) is safe dummy? or kInit?

                // Use a static mutex to serialize sends to same client_fd
                static std::mutex           send_mtx;
                std::lock_guard<std::mutex> lk(send_mtx);

                if (send(client_fd_copy, &rh, sizeof(rh), 0) < 0)
                    perror("[Daemon] Failed to send header");
                if (send(client_fd_copy, &rm, sizeof(rm), 0) < 0)
                    perror("[Daemon] Failed to send meta");
                if (!res.empty())
                    if (send(client_fd_copy, res.data(), res.size(), 0) < 0)
                        perror("[Daemon] Failed to send body");

                std::cout << "[Daemon] Response sent to client (Seq: " << seq_id_copy << ")" << std::endl;
            }).detach();
        }
    }
    close(client_fd);
}

int run_daemon(int argc, char** argv)
{
    int          my_port  = 4469;
    std::string  hub_ip   = "";
    int          hub_port = 8888;
    ResourceSpec capacity{0, 0};
    std::string  log_dir = "~/.spoke/logs";

    // Argument Parsing (Simple)
    // Usage: ./agent <hub_ip> <hub_port> [--port <p>] [--gpus <n>] [--log-dir <dir>]
    std::vector<std::string> args;
    for (int i = 0; i < argc; ++i)
        args.push_back(argv[i]);

    // Positional fallback for hub connection
    if (args.size() > 2 && args[1].rfind("--", 0) != 0) {
        hub_ip   = args[1];
        hub_port = std::stoi(args[2]);
    }

    for (size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--port" && i + 1 < args.size()) {
            my_port = std::stoi(args[i + 1]);
            i++;
        }
        else if (args[i] == "--hub-ip" && i + 1 < args.size()) {
            hub_ip = args[i + 1];
            i++;
        }
        else if (args[i] == "--hub-port" && i + 1 < args.size()) {
            hub_port = std::stoi(args[i + 1]);
            i++;
        }
        else if (args[i] == "--gpus" && i + 1 < args.size()) {
            capacity.num_gpus = std::stoi(args[i + 1]);
            i++;
        }
        else if (args[i] == "--log-dir" && i + 1 < args.size()) {
            log_dir = args[i + 1];
            i++;
        }
    }

    Agent local_agent;
    local_agent.setLogDir(expand_path(log_dir));

    // [New] Forward unsolicited messages (e.g. Streams) to *ALL* connected clients?
    // Problem: Daemon handles multiple clients. We don't know which client the message belongs to.
    // However, in V2, tickets/slots map to clients?
    // Simplification: In this architecture, usually one client controls the actors it spawned.
    // BUT here we have separate threads per client connection (`handle_client`).
    // We need to map Actor -> Client FD to forward push messages.
    // Since we don't have that map easily, we might need to broadcast or just store "owner" of an actor.
    //
    // Currently `handle_client` spawns actors. We can track mapping there.
    // But `local_agent` is shared.
    // Let's add a global map: ActorID -> ClientFD (protected by mutex)
    // And use `send` (thread-safe with lock) to forward.

    // NOTE: Use global client_map_mtx and actor_owner_map defined at namespace level

    local_agent.setOnMessage([&](const std::string& actor_id, const PipeHeader& ph, const std::string& body) {
        std::cout << "[Daemon] Received push from actor " << actor_id << " Action=" << static_cast<int>(ph.action)
                  << " Seq=" << ph.seq_id << " BodySize=" << body.size() << std::endl;
        std::lock_guard<std::mutex> lk(client_map_mtx);
        if (actor_owner_map.count(actor_id)) {
            int client_fd = actor_owner_map[actor_id];
            std::cout << "[Daemon] Found owner for " << actor_id << ", client_fd=" << client_fd << std::endl;

            // Forward Push Message
            // Now NetRespMeta has action field.
            NetHeader   rh{0x504F4B45, sizeof(NetRespMeta), (uint32_t)body.size()};
            NetRespMeta rm{ph.seq_id, 1, ph.action};

            // Use a global mutex for all sends to this client
            static std::mutex           send_mtx;
            std::lock_guard<std::mutex> slk(send_mtx);

            ssize_t r1 = send(client_fd, &rh, sizeof(rh), 0);
            ssize_t r2 = send(client_fd, &rm, sizeof(rm), 0);
            ssize_t r3 = 0;
            if (!body.empty())
                r3 = send(client_fd, body.data(), body.size(), 0);

            std::cout << "[Daemon] Push sent to client_fd=" << client_fd << " (hdr=" << r1 << " meta=" << r2
                      << " body=" << r3 << ")" << std::endl;
        }
        else {
            std::cerr << "[Daemon] Warning: No owner found for actor " << actor_id
                      << ". Map size=" << actor_owner_map.size() << std::endl;
        }
    });

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt       = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in addr;
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(my_port);

    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        std::cerr << "[Daemon] Bind failed on port " << my_port << std::endl;
        return 1;
    }
    listen(server_fd, 5);
    std::cout << "[Daemon] Listening on " << my_port << "..." << std::endl;

    if (!hub_ip.empty())
        register_to_hub(hub_ip, hub_port, my_port, capacity);

    // [New] Print registered actor types as a helpful tip
    auto registered_types = ActorFactory::instance().getRegisteredTypes();
    if (!registered_types.empty()) {
        std::cout << "💡 Tip: The following " << registered_types.size()
                  << " Actor type(s) are registered and ready to be spawned:" << std::endl;
        int count = 0;
        for (const auto& type : registered_types) {
            std::cout << "   -> " << type << std::endl;
            if (++count >= 10)
                break;
        }
        if (registered_types.size() > 10) {
            std::cout << "   ... and " << (registered_types.size() - 10) << " more." << std::endl;
        }
    }

    // Setup signal handling for graceful shutdown
    static bool g_running = true;
    signal(SIGPIPE, SIG_IGN);  // Prevent crash on write to dead child/client
    signal(SIGINT, [](int) {
        std::cout << "\n[Daemon] Shutting down..." << std::endl;
        g_running = false;
    });

    while (g_running) {
        // Use select/poll with timeout to allow checking g_running
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(server_fd, &fds);
        struct timeval tv {
            1, 0
        };  // 1s timeout

        int ret = select(server_fd + 1, &fds, nullptr, nullptr, &tv);
        if (ret > 0 && FD_ISSET(server_fd, &fds)) {
            int cfd = accept(server_fd, nullptr, nullptr);
            if (cfd >= 0)
                std::thread(handle_client, cfd, &local_agent).detach();
        }
    }
    return 0;
}

}  // namespace spoke
