#pragma once
#include "actor.h"
#include "serializer.h"
#include "types.h"
#include <atomic>
#include <chrono>
#include <cstdio>  // For fprintf
#include <ctime>
#include <fcntl.h>
#include <filesystem>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <signal.h>
#include <sstream>
#include <string>
#include <sys/poll.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace spoke {

struct ActorHandle {
    pid_t pid;
    int   tx_fd;
    int   rx_fd;
};

class Agent {
public:
    Agent(): monitor_thread_(&Agent::monitor, this) {}

    void setLogDir(const std::string& dir)
    {
        log_dir_ = dir;
        // Remove trailing slash if present for consistency, though filesystem handles it
        if (!log_dir_.empty() && log_dir_.back() == '/') {
            log_dir_.pop_back();
        }
    }

    ~Agent()
    {
        running_ = false;
        if (monitor_thread_.joinable())
            monitor_thread_.join();
        for (auto& [id, h] : registry_) {
            close(h.tx_fd);
            close(h.rx_fd);
            kill(h.pid, SIGTERM);
            waitpid(h.pid, nullptr, 0);
        }
    }

    pid_t spawnActor(const std::string&      type,
                     const std::string&      id,
                     const std::vector<int>& gpu_ids  = {},
                     const std::string&      group_id = "")
    {
        // Enforce serialized forking to avoid multi-thread fork hazards
        std::lock_guard<std::mutex> fork_lk(spawn_mtx_);

        // ... (rest of the initial logic before fork)
        // Fix: Cleanup existing actor if ID collision
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (registry_.find(id) != registry_.end()) {
                auto& h = registry_[id];
                close(h.tx_fd);
                close(h.rx_fd);
                kill(h.pid, SIGTERM);
                waitpid(h.pid, nullptr, 0);
                registry_.erase(id);
                std::cout << "[Agent] Cleaned up existing actor: " << id << std::endl;
            }
        }

        int p2c[2], c2p[2];
        if (pipe(p2c) != 0 || pipe(c2p) != 0)
            return -1;
        std::cout << "[Agent] Forking for actor: " << id << "..." << std::endl;
        pid_t pid = fork();
        if (pid == 0) {
            // ... child ...
            // Child process: Exec into a new image to clear all threads/mutexes
            close(p2c[1]);
            close(c2p[0]);

            char    exe_path[1024];
            ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
            if (len == -1) {
                perror("readlink");
                exit(1);
            }
            exe_path[len] = '\0';

            std::string rx_str = std::to_string(p2c[0]);
            std::string tx_str = std::to_string(c2p[1]);

            // Args: <exe> --worker <type> <id> <rx> <tx> NULL
            std::vector<char*> args;
            args.push_back(exe_path);
            args.push_back(strdup("--worker"));
            args.push_back(strdup(type.c_str()));
            args.push_back(strdup(id.c_str()));
            args.push_back(strdup(rx_str.c_str()));
            args.push_back(strdup(tx_str.c_str()));
            args.push_back(nullptr);

            prctl(PR_SET_PDEATHSIG, SIGTERM);

            {
                // Generate filename: <log_dir>/<group_id>/<id>_<timestamp>.log
                std::string current_log_dir = log_dir_;
                if (!group_id.empty()) {
                    current_log_dir += "/" + group_id;
                }

                std::filesystem::create_directories(current_log_dir);

                auto              now       = std::chrono::system_clock::now();
                auto              in_time_t = std::chrono::system_clock::to_time_t(now);
                std::stringstream ss;
                ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");

                std::string log_file = current_log_dir + "/" + id + "_" + ss.str() + ".log";

                int fd = open(log_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
                if (fd >= 0) {
                    // Redirect stdout and stderr
                    if (dup2(fd, STDOUT_FILENO) < 0)
                        perror("dup2 stdout");
                    if (dup2(fd, STDERR_FILENO) < 0)
                        perror("dup2 stderr");
                    close(fd);

                    // Force flush to ensure file is written to immediately
                    setbuf(stdout, NULL);
                    setbuf(stderr, NULL);

                    std::cout << "==================================================" << std::endl;
                    std::cout << " Spoke Actor Log: " << id << std::endl;
                    if (!group_id.empty()) {
                        std::cout << " Group/Ticket:    " << group_id << std::endl;
                    }
                    std::cout << " Started at:      " << ss.str() << std::endl;
                    std::cout << "==================================================" << std::endl;
                }
                else {
                    fprintf(stderr, "[Spoke] Failed to open log file %s\n", log_file.c_str());
                }
            }

            // GPU Isolation Logic
            // 1. If gpu_ids are provided, restrict visibility to those IDs.
            // 2. If gpu_ids are empty BUT we have a ticket/group_id (Managed Mode),
            //    it means explicit 0-GPU allocation -> Disable CUDA.
            // 3. If gpu_ids are empty AND no ticket (Legacy Mode),
            //    inherit parent's CUDA_VISIBLE_DEVICES (no isolation).

            if (!gpu_ids.empty()) {
                std::string gpu_str = "";
                for (size_t i = 0; i < gpu_ids.size(); ++i) {
                    gpu_str += std::to_string(gpu_ids[i]);
                    if (i < gpu_ids.size() - 1)
                        gpu_str += ",";
                }
                setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID", 1);
                setenv("CUDA_VISIBLE_DEVICES", gpu_str.c_str(), 1);
                // Also set ROCR_VISIBLE_DEVICES for AMD, just in case
                setenv("ROCR_VISIBLE_DEVICES", gpu_str.c_str(), 1);
            }
            else if (!group_id.empty()) {
                // Managed mode with 0 GPUs assigned -> Explicitly disable CUDA
                setenv("CUDA_VISIBLE_DEVICES", "", 1);
                setenv("ROCR_VISIBLE_DEVICES", "", 1);
                std::cout << "[Agent] Managed Mode (Ticket: " << group_id << "): Disabling CUDA for " << id
                          << std::endl;
            }
            else {
                // Legacy mode: Inherit environment (do nothing)
            }

            execv(exe_path, args.data());

            // If execv returns, it failed
            perror("execv failed");
            exit(1);
        }
        std::cout << "[Agent] Forked PID: " << pid << ". Updating registry..." << std::endl;
        close(p2c[0]);  // Close Parent Read from Child (Wait, p2c is Parent->Child)
        close(c2p[1]);  // c2p: 0=Read, 1=Write. Child writes to 1. Parent reads from 0.

        std::cout << "[Agent] Parent Pipes: TX=" << p2c[1] << " (p2c[1]), RX=" << c2p[0] << " (c2p[0])" << std::endl;

        std::lock_guard<std::mutex> lk(mtx_);
        registry_[id] = {pid, p2c[1], c2p[0]};
        std::cout << "[Agent] Actor spawned successfully: " << id << " with RX_FD=" << c2p[0] << std::endl;
        return pid;
    }

    template<typename T>
    pid_t spawnActor(const std::string& id)
    {
        std::lock_guard<std::mutex> fork_lk(spawn_mtx_);

        int p2c[2], c2p[2];
        if (pipe(p2c) != 0 || pipe(c2p) != 0)
            return -1;
        pid_t pid = fork();
        if (pid == 0) {
            close(p2c[1]);
            close(c2p[0]);
            // 直接构造 T，不查工厂
            T instance(id, p2c[0], c2p[1]);
            instance.run();
            exit(0);
        }
        close(p2c[0]);
        close(c2p[1]);
        std::lock_guard<std::mutex> lk(mtx_);
        registry_[id] = {pid, p2c[1], c2p[0]};
        return pid;
    }

    std::future<std::string> callRemoteRaw(const std::string& id, Action act, const std::string& data)
    {
        auto     promise = std::make_shared<std::promise<std::string>>();
        auto     future  = promise->get_future();
        uint32_t s       = seq_++;

        std::lock_guard<std::mutex> lk(mtx_);
        if (registry_.find(id) == registry_.end()) {
            std::cerr << "[Agent] ERROR: Actor " << id << " not found in registry!" << std::endl;
            promise->set_value("");
            return future;
        }
        promises_[s] = promise;

        PipeHeader ph{act, s, (uint32_t)data.size()};
        int        fd = registry_[id].tx_fd;

        // [Trace] Log forwarding
        std::cout << "[Agent] Forwarding Action " << static_cast<int>(act) << " to Actor " << id << " (Seq: " << s
                  << ", Size: " << data.size() << ")" << std::endl;

        if (write(fd, &ph, sizeof(ph)) < 0) {
            std::cerr << "[Agent] Failed to write header to actor " << id << std::endl;
            promises_.erase(s);
            promise->set_value("");
            return future;
        }
        if (!data.empty()) {
            if (write(fd, data.data(), data.size()) < 0) {
                std::cerr << "[Agent] Failed to write body to actor " << id << std::endl;
                promises_.erase(s);
                promise->set_value("");
                return future;
            }
        }

        return future;
    }

    // ==========================================
    // [v1 修复] 本地泛型调用辅助 (供 smoke_v1 使用)
    // ==========================================
    template<typename ReqT, typename RespT>
    std::future<RespT> callRemote(const std::string& id, Action act, const ReqT& req)
    {
        // 1. 序列化请求
        std::string req_raw = Pack(req);

        return std::async(std::launch::async, [this, id, act, req_raw]() {
            auto        fut      = this->callRemoteRaw(id, act, req_raw);
            std::string resp_raw = fut.get();
            return Unpack<RespT>(resp_raw);
        });
    }

    void setOnMessage(std::function<void(const std::string&, const PipeHeader&, const std::string&)> cb)
    {
        std::lock_guard<std::mutex> lk(mtx_);
        on_message_callback_ = cb;
    }

private:
    bool read_exact(int fd, void* buf, size_t size)
    {
        size_t total = 0;
        char*  p     = static_cast<char*>(buf);
        while (total < size) {
            ssize_t n = read(fd, p + total, size - total);
            if (n <= 0)
                return false;
            total += n;
        }
        return true;
    }

    void monitor()
    {
        while (running_) {
            std::vector<pollfd> pfds;
            {
                std::lock_guard<std::mutex> lk(mtx_);
                for (auto& [id, h] : registry_)
                    pfds.push_back({h.rx_fd, POLLIN, 0});
            }
            if (pfds.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }

            int ret = poll(pfds.data(), pfds.size(), 100);
            if (ret <= 0)
                continue;

            for (auto& p : pfds) {
                if (p.revents & (POLLIN | POLLHUP | POLLERR)) {
                    PipeHeader ph;
                    if (!read_exact(p.fd, &ph, sizeof(ph))) {
                        // EOF or error
                        handle_actor_death(p.fd);
                        continue;
                    }

                    std::string body;
                    if (ph.data_size > 0) {
                        body.resize(ph.data_size);
                        if (!read_exact(p.fd, body.data(), ph.data_size)) {
                            handle_actor_death(p.fd);
                            continue;
                        }
                    }

                    std::lock_guard<std::mutex> lk(mtx_);
                    if (promises_.count(ph.seq_id)) {
                        std::cout << "[Agent] Received response for Seq: " << ph.seq_id << " (Size: " << body.size()
                                  << "). Fulfilling promise." << std::endl;
                        promises_[ph.seq_id]->set_value(body);
                        promises_.erase(ph.seq_id);
                    }
                    // [New] Handle unsolicited push messages (Action::kStreamPush)
                    // These should be forwarded to the Daemon/Client, but Agent class here is generic.
                    // However, in the current architecture, Daemon receives this via `callRemote` promise mechanism?
                    // NO, `callRemote` is Request-Response. Unsolicited Push needs a different path.
                    // If Agent is used by Daemon, Daemon calls `callRemoteRaw` which registers a promise.
                    // But `kStreamPush` originates from Actor without a Request.
                    // We need a callback or mechanism to forward this to Daemon.
                    else {
                        // Temporary: Log but don't drop if we can figure out how to route it.
                        // But since we are in `Agent` class which doesn't know about Daemon's clients...
                        // Wait, Daemon uses Agent. Daemon has `spawnActor`.
                        // Daemon listens to Agent? No, Agent runs `monitor` loop.
                        // We need a callback `on_message` in Agent to bubble up unsolicited messages.

                        if (on_message_callback_) {
                            // Need to find ActorID from p.fd
                            std::string actor_id;
                            for (auto& [id, h] : registry_) {
                                if (h.rx_fd == p.fd) {
                                    actor_id = id;
                                    break;
                                }
                            }
                            if (!actor_id.empty()) {
                                on_message_callback_(actor_id, ph, body);
                            }
                        }
                        else {
                            std::cerr << "[Agent] Warning: Received unexpected message from actor. Seq: " << ph.seq_id
                                      << ", Action: " << static_cast<int>(ph.action) << std::endl;
                        }
                    }
                }
            }
        }
    }

    void handle_actor_death(int fd)
    {
        std::lock_guard<std::mutex> lk(mtx_);
        std::string                 dead_id;
        for (auto& [id, h] : registry_) {
            if (h.rx_fd == fd) {
                dead_id = id;
                break;
            }
        }
        if (!dead_id.empty()) {
            std::cout << "[Agent] Detected death of actor: " << dead_id << std::endl;
            close(registry_[dead_id].tx_fd);
            close(registry_[dead_id].rx_fd);
            registry_.erase(dead_id);
        }
    }

    std::map<std::string, ActorHandle>                                             registry_;
    std::map<uint32_t, std::shared_ptr<std::promise<std::string>>>                 promises_;
    std::mutex                                                                     mtx_;
    std::mutex                                                                     spawn_mtx_;  // Serialize forks
    std::atomic<uint32_t>                                                          seq_{0};
    std::atomic<bool>                                                              running_{true};
    std::thread                                                                    monitor_thread_;
    std::string                                                                    log_dir_ = ".spoke/logs";
    std::function<void(const std::string&, const PipeHeader&, const std::string&)> on_message_callback_;
};

}  // namespace spoke
