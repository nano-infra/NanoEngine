#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include "hub.h"
#include "spoke/csrc/types.h"

spoke::Hub g_hub;

#include <arpa/inet.h>
#include <ifaddrs.h>

std::string get_local_ip()
{
    struct ifaddrs* ifAddrStruct = NULL;
    struct ifaddrs* ifa          = NULL;
    std::string     ip           = "127.0.0.1";
    getifaddrs(&ifAddrStruct);
    for (ifa = ifAddrStruct; ifa != NULL; ifa = ifa->ifa_next) {
        if (!ifa->ifa_addr)
            continue;
        if (ifa->ifa_addr->sa_family == AF_INET) {  // IPv4
            char buf[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &((struct sockaddr_in*)ifa->ifa_addr)->sin_addr, buf, INET_ADDRSTRLEN);
            std::string name = ifa->ifa_name;
            if (name != "lo") {
                ip = buf;
                break;
            }
        }
    }
    if (ifAddrStruct)
        freeifaddrs(ifAddrStruct);
    return ip;
}

void handle_connection(int fd)
{
    using namespace spoke;
    NetHeader hdr;
    if (recv(fd, &hdr, sizeof(NetHeader), MSG_WAITALL) <= 0) {
        close(fd);
        return;
    }
    NetMeta  meta;
    uint32_t mlen = (hdr.meta_size > 0) ? hdr.meta_size : sizeof(NetMeta);
    if (recv(fd, &meta, mlen, MSG_WAITALL) <= 0) {
        close(fd);
        return;
    }
    std::string req_body;
    if (hdr.data_size > 0) {
        req_body.resize(hdr.data_size);
        recv(fd, req_body.data(), hdr.data_size, MSG_WAITALL);
    }

    std::string resp_body;
    int         status = 0;

    if (meta.action == Action::kHubRegisterNode) {
        if (req_body.size() >= sizeof(RegisterNodeReq)) {
            const auto* req = reinterpret_cast<const RegisterNodeReq*>(req_body.data());
            g_hub.registerNode(req->ip, req->port, req->capacity);
            status = 1;
        }
        else {
            std::cerr << "[Hub] Invalid RegisterNodeReq size" << std::endl;
            status = -1;
        }
    }
    else if (meta.action == Action::kHubFindNode) {
        NodeInfo     node;
        ResourceSpec req_spec = {0, 0};

        if (req_body.size() >= sizeof(FindNodeReq)) {
            const auto* req = reinterpret_cast<const FindNodeReq*>(req_body.data());
            req_spec        = req->requirements;
        }

        if (g_hub.scheduleNode(req_spec, &node)) {
            status    = 1;
            resp_body = node.ip + ":" + std::to_string(node.port);
        }
        else {
            status = -1;
        }
    }
    else if (meta.action == Action::kNetAllocate) {
        if (req_body.size() >= sizeof(AllocateReq)) {
            const auto* req = reinterpret_cast<const AllocateReq*>(req_body.data());

            std::string                ticket;
            std::vector<AllocatedSlot> slots;

            if (g_hub.gangAllocate(*req, ticket, slots)) {
                status = 1;

                // Serialize Response
                // Header: AllocateResp (fixed size)
                // Body: vector<AllocatedSlot> (variable size)
                AllocateResp resp;
                memset(&resp, 0, sizeof(resp));
                strncpy(resp.ticket_id, ticket.c_str(), 63);
                resp.num_members = (uint32_t)slots.size();

                resp_body.resize(sizeof(AllocateResp) + slots.size() * sizeof(AllocatedSlot));
                memcpy(resp_body.data(), &resp, sizeof(AllocateResp));
                memcpy(resp_body.data() + sizeof(AllocateResp), slots.data(), slots.size() * sizeof(AllocatedSlot));
            }
            else {
                status = -1;  // Allocation failed
            }
        }
        else {
            std::cerr << "[Hub] Invalid AllocateReq size" << std::endl;
            status = -1;
        }
    }
    else if (meta.action == Action::kNetLaunch) {
        // req_body = LaunchReq + (Optional Args Body)
        if (req_body.size() >= sizeof(LaunchReq)) {
            const auto* req = reinterpret_cast<const LaunchReq*>(req_body.data());

            std::string args_body;
            if (req_body.size() > sizeof(LaunchReq)) {
                args_body = req_body.substr(sizeof(LaunchReq));
            }

            if (g_hub.launchActor(*req, meta.actor_type, meta.actor_id, args_body)) {
                status = 1;
            }
            else {
                status = -1;
            }
        }
        else {
            std::cerr << "[Hub] Invalid LaunchReq size" << std::endl;
            status = -1;
        }
    }
    else if (meta.action == Action::kNetRelease) {
        if (req_body.size() >= sizeof(ReleaseReq)) {
            const auto* req = reinterpret_cast<const ReleaseReq*>(req_body.data());
            if (g_hub.gangRelease(req->ticket_id)) {
                status = 1;
            }
            else {
                status = -1;
            }
        }
        else {
            std::cerr << "[Hub] Invalid ReleaseReq size" << std::endl;
            status = -1;
        }
    }

    NetHeader   rh{0x504F4B45, sizeof(NetRespMeta), (uint32_t)resp_body.size()};
    NetRespMeta rm{meta.seq_id, status};
    send(fd, &rh, sizeof(rh), 0);
    send(fd, &rm, sizeof(rm), 0);
    if (!resp_body.empty())
        send(fd, resp_body.data(), resp_body.size(), 0);
    close(fd);
}

int main(int argc, char** argv)
{
    int port = 8888;
    if (argc > 1)
        port = atoi(argv[1]);
    int sfd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in addr;
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(port);
    (void)bind(sfd, (struct sockaddr*)&addr, sizeof(addr));
    listen(sfd, 5);
    std::string hub_ip = get_local_ip();
    std::cout << "🚀 [Spoke Hub] Ready and listening on port: " << port << std::endl;
    std::cout << "💡 Tip: To scale your cluster, run the following command on other nodes to register an Agent:"
              << std::endl;
    std::cout << "   ./bin/spoke_agent " << hub_ip << " " << port << " --port 4469 --gpus 8" << std::endl;

    while (true) {
        int cfd = accept(sfd, nullptr, nullptr);
        if (cfd >= 0)
            std::thread(handle_connection, cfd).detach();
    }
    return 0;
}
