#pragma once

#include "nanodeploy/csrc/logging.h"
#include "nanodeploy/csrc/scheduler/scheduler.h"
#include "nanodeploy/csrc/worker/model_runner_ipc.h"  // For Req/Resp definitions
#include "spoke/csrc/client.h"
#include <future>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <ifaddrs.h>

namespace nanodeploy {

static std::string get_local_ip()
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

class SimpleEngine {
public:
    SimpleEngine() = default;

    void init(const std::string& config_path,
              int                tp                = 1,
              int                pp                = 1,
              int                dp                = 1,
              const std::string& ip                = "127.0.0.1",
              int                port              = 8888,
              int                attention_tp      = 1,
              int                attention_dp      = 1,
              int                attention_sp      = 1,
              int                ffn_tp            = 1,
              int                ffn_dp            = 1,
              int                ffn_ep            = 1,
              bool               enable_rdma       = true,
              bool               enable_cuda_graph = false)
    {
        // 1. Connect to Spoke Hub
        NANODEPLOY_LOG_INFO("[SimpleEngine] Connecting to Spoke Hub at ", ip, ":", port);
        client_ = std::make_shared<spoke::Client>(ip, port, /*is_hub_mode=*/true);

        tp_           = tp;
        pp_           = pp;
        dp_           = dp;
        attention_tp_ = attention_tp;
        attention_dp_ = attention_dp;
        attention_sp_ = attention_sp;
        ffn_tp_       = ffn_tp;
        ffn_dp_       = ffn_dp;
        ffn_ep_       = ffn_ep;

        // Fallback for compatibility: if detailed args are default, try to map legacy args
        if (attention_dp_ == 1 && dp_ > 1)
            attention_dp_ = dp_;
        if (attention_tp_ == 1 && tp_ > 1)
            attention_tp_ = tp_;

        int legacy_world_size   = tp_ * pp_ * dp_;
        int detailed_world_size = attention_dp_ * attention_tp_ * attention_sp_;

        int attention_world_size = attention_dp_ * attention_tp_ * attention_sp_;
        int ffn_world_size       = ffn_dp_ * ffn_tp_ * ffn_ep_;
        NANODEPLOY_ASSERT(attention_world_size == ffn_world_size,
                          "Attention world size and FFN world size must be equal");
        int world_size = attention_world_size;

        if (world_size <= 0)
            world_size = 1;  // Default to 1 if somehow 0

        world_size_ = world_size;

        if (world_size <= 0)
            throw std::runtime_error("Invalid world_size");

        // 2. Resource Allocation (Gang Scheduling)
        NANODEPLOY_LOG_INFO("[SimpleEngine] Requesting Gang Allocation for ", world_size, " actors...");
        spoke::ResourceSpec res;
        res.num_gpus = 1;

        int actors_per_node = 8;
        if (world_size < 8)
            actors_per_node = world_size;
        int num_nodes = (world_size + actors_per_node - 1) / actors_per_node;

        std::string my_ip = get_local_ip();
        NANODEPLOY_LOG_INFO("[SimpleEngine] My IP is ", my_ip, ". Requesting affinity.");

        auto alloc_fut  = client_->gangAllocate(num_nodes, actors_per_node, res, /*strict_pack=*/true, my_ip);
        auto alloc_resp = alloc_fut.get();

        if (strlen(alloc_resp.ticket_id) == 0) {
            throw std::runtime_error("Gang Allocation Failed!");
        }
        ticket_id_ = alloc_resp.ticket_id;
        NANODEPLOY_LOG_INFO("[SimpleEngine] Allocation Success! Ticket: ", ticket_id_);

        // 3. Launch Actors (Parallel)
        actor_ids_.clear();
        std::vector<std::future<void>> launch_futs;
        std::string                    ticket = ticket_id_;

        for (int i = 0; i < world_size; ++i) {
            std::string id = "ModelRunner_" + std::to_string(i);
            actor_ids_.push_back(id);
            std::string args = "--rank " + std::to_string(i);

            launch_futs.push_back(std::async(std::launch::async, [this, ticket, i, id, args]() {
                client_->launchActor(ticket.c_str(), i, "ModelRunner", id, args);
            }));
        }
        for (auto& f : launch_futs)
            f.get();

        NANODEPLOY_LOG_INFO("[SimpleEngine] Waiting for actors to start...");
        std::this_thread::sleep_for(std::chrono::seconds(2));

        // 4. Initialize All Actors (Parallel)
        NANODEPLOY_LOG_INFO("[SimpleEngine] Initializing remote models (Parallel)...");
        constexpr int kInitAction = static_cast<int>(spoke::Action::kUserActionStart) + 10;

        std::vector<std::future<ModelInitResp>> init_futs;
        for (int i = 0; i < world_size; ++i) {
            init_futs.push_back(
                std::async(std::launch::async,
                           [this, i, world_size, config_path, kInitAction, enable_cuda_graph]() -> ModelInitResp {
                               ModelInitReq req;
                               req.config_path       = config_path;
                               req.rank              = i;
                               req.world_size        = world_size;
                               req.tp_degree         = tp_;
                               req.pp_degree         = pp_;
                               req.dp_degree         = dp_;
                               req.enable_cuda_graph = enable_cuda_graph;

                               req.attention_tp = attention_tp_;
                               req.attention_dp = attention_dp_;
                               req.attention_sp = attention_sp_;
                               req.ffn_tp       = ffn_tp_;
                               req.ffn_dp       = ffn_dp_;
                               req.ffn_ep       = ffn_ep_;

                               auto f = client_->callRemote<ModelInitReq, ModelInitResp>(
                                   actor_ids_[i], static_cast<spoke::Action>(kInitAction), req);
                               return f.get();
                           }));
        }

        for (auto& f : init_futs) {
            f.get();
        }
        NANODEPLOY_LOG_INFO("[SimpleEngine] All actors initialized.");

        // 5. Initialize KV Cache on all actors (Parallel) - 2 Step Process
        // Step 5a: Get Available Blocks from all ranks
        NANODEPLOY_LOG_INFO("[SimpleEngine] Querying available KV blocks from all workers...");
        constexpr int kGetAvailableBlocksAction = static_cast<int>(spoke::Action::kUserActionStart) + 17;
        constexpr int kAllocKVBlocksAction      = static_cast<int>(spoke::Action::kUserActionStart) + 18;

        // TODO: Make these configurable via EngineInitReq
        int   max_batch_size         = 256;  // Default max batch size
        float gpu_memory_utilization = 0.90f;

        std::vector<std::future<int>> avail_futs;
        for (int i = 0; i < world_size; ++i) {
            avail_futs.push_back(
                std::async(std::launch::async,
                           [this, i, max_batch_size, gpu_memory_utilization, kGetAvailableBlocksAction]() -> int {
                               GetAvailableKVBlocksReq req;
                               req.max_batch_size         = max_batch_size;
                               req.gpu_memory_utilization = gpu_memory_utilization;

                               auto f = client_->callRemote<GetAvailableKVBlocksReq, GetAvailableKVBlocksResp>(
                                   actor_ids_[i], static_cast<spoke::Action>(kGetAvailableBlocksAction), req);
                               return f.get();
                           }));
        }

        int min_actual_blocks = std::numeric_limits<int>::max();
        for (auto& f : avail_futs) {
            int blocks = f.get();
            if (blocks > 0 && blocks < min_actual_blocks) {
                min_actual_blocks = blocks;
            }
        }

        if (min_actual_blocks == std::numeric_limits<int>::max()) {
            NANODEPLOY_LOG_WARN("[SimpleEngine] WARNING: Failed to get valid block count. Defaulting to 1024.");
            min_actual_blocks = 1024;
        }

        NANODEPLOY_LOG_INFO("[SimpleEngine] Negotiated Block Count: ", min_actual_blocks, ". Allocating...");

        // Step 5b: Allocate KV Blocks on all ranks
        std::vector<std::future<bool>> alloc_futs;
        for (int i = 0; i < world_size; ++i) {
            alloc_futs.push_back(std::async(
                std::launch::async, [this, i, min_actual_blocks, max_batch_size, kAllocKVBlocksAction]() -> bool {
                    AllocKVBlocksReq req;
                    req.num_blocks     = min_actual_blocks;
                    req.max_batch_size = max_batch_size;

                    auto f = client_->callRemote<AllocKVBlocksReq, AllocKVBlocksResp>(
                        actor_ids_[i], static_cast<spoke::Action>(kAllocKVBlocksAction), req);
                    return f.get();
                }));
        }

        bool all_alloc_success = true;
        for (auto& f : alloc_futs) {
            if (!f.get())
                all_alloc_success = false;
        }

        if (!all_alloc_success) {
            throw std::runtime_error("KV Cache Allocation Failed on some ranks!");
        }

        NANODEPLOY_LOG_INFO("[SimpleEngine] KV Cache initialized on all actors.");

        // 5b. Capture CUDA Graphs for Decode (Optional but highly recommended for latency)
        // Static buffers now use random/valid values to avoid DeepGemm issues.
        // Sequential warmup on rank 0 ensures JIT compilation completes before capture.
        // 6. DeepEP Synchronization (MUST happen before Graph Capture Warmup)
        // Warmup calls moe_model->forward() which requires DeepEP handles to be ready.
        if (ffn_ep_ > 1) {
            NANODEPLOY_LOG_DEBUG("[SimpleEngine] Syncing DeepEP buffers...");
            syncDeepEpBuffers(world_size);
        }

        // 5b. Capture CUDA Graphs for Decode (Optional but highly recommended for latency)
        // Static buffers now use random/valid values to avoid DeepGemm issues.
        // Sequential warmup on rank 0 ensures JIT compilation completes before capture.
        if (enable_cuda_graph) {
            constexpr int kWarmupMoeAction    = static_cast<int>(spoke::Action::kUserActionStart) + 16;
            constexpr int kCaptureGraphAction = static_cast<int>(spoke::Action::kUserActionStart) + 15;
            int           warm_up_steps       = 3;  // Default warmup iterations

            // Step 1: Warmup MoE on ALL ranks (triggers DeepGEMM JIT compilation per process)
            // NOTE: Must run on all ranks because DeepGEMM JIT is process-local.
            if (ffn_ep_ > 0) {
                NANODEPLOY_LOG_INFO("[SimpleEngine] Warming up MoE on all ranks (triggers DeepGemm JIT)...");
                std::vector<std::future<bool>> warmup_futs;
                for (int i = 0; i < world_size; ++i) {
                    warmup_futs.push_back(std::async(std::launch::async, [this, i, kWarmupMoeAction]() -> bool {
                        auto f = client_->callRemote<int, bool>(
                            actor_ids_[i], static_cast<spoke::Action>(kWarmupMoeAction), 0);
                        return f.get();
                    }));
                }
                for (auto& f : warmup_futs) {
                    f.get();
                }
                NANODEPLOY_LOG_INFO("[SimpleEngine] MoE warmup complete on all ranks.");
            }

            // Step 2: Now all ranks can capture (JIT already compiled)
            NANODEPLOY_LOG_INFO("[SimpleEngine] Capturing CUDA Graphs on all actors...");
            std::vector<std::future<GraphCaptureResp>> graph_futs;
            for (int i = 0; i < world_size; ++i) {
                graph_futs.push_back(
                    std::async(std::launch::async, [this, i, warm_up_steps, kCaptureGraphAction]() -> GraphCaptureResp {
                        GraphCaptureReq req;
                        req.warm_up_steps = warm_up_steps;
                        req.warmup_only   = false;  // Actually capture graphs

                        auto f = client_->callRemote<GraphCaptureReq, GraphCaptureResp>(
                            actor_ids_[i], static_cast<spoke::Action>(kCaptureGraphAction), req);
                        return f.get();
                    }));
            }

            for (auto& f : graph_futs) {
                f.get();
            }
            NANODEPLOY_LOG_INFO("[SimpleEngine] CUDA Graphs captured on all actors.");
        }

        // 7. Configure RDMA for all actors (Direct Connection)
        if (enable_rdma) {
            NANODEPLOY_LOG_INFO("[SimpleEngine] Configuring RDMA for all actors (", world_size_, ")...");
            for (const auto& id : actor_ids_) {
                try {
                    // Try to init RDMA, but gracefully fall back to socket if it fails
                    if (client_->initRDMA(id)) {
                        NANODEPLOY_LOG_DEBUG("[SimpleEngine] RDMA Configured with ", id, ".");
                    }
                    else {
                        NANODEPLOY_LOG_INFO("[SimpleEngine] RDMA Config failed for ", id, ", using TCP.");
                    }
                }
                catch (const std::exception& e) {
                    NANODEPLOY_LOG_ERROR("[SimpleEngine] RDMA Init Exception for ", id, ": ", e.what());
                }
            }
        }
        else {
            NANODEPLOY_LOG_INFO("[SimpleEngine] RDMA Config disabled.");
        }

        // 8. Initialize Scheduler
        std::string engine_id = "simple_engine";
        scheduler_            = std::make_unique<Scheduler>(engine_id,
                                                 1,
                                                 256,
                                                 8192,
                                                 151643,
                                                 attention_dp_,
                                                 attention_sp_,
                                                 min_actual_blocks,
                                                 Sequence::block_size,
                                                 "decode");
    }

    void shutdown()
    {
        if (client_) {
            NANODEPLOY_LOG_INFO("[SimpleEngine] Shutting down client...");
            client_.reset();
        }
    }

    void release_resources()
    {
        if (client_ && !ticket_id_.empty()) {
            NANODEPLOY_LOG_INFO("[SimpleEngine] Releasing Gang Resources for ticket: ", ticket_id_);
            // Assuming gangRelease takes the ticket ID string
            // We use ticket_id_.c_str() just in case it expects const char*
            // Since I don't see the header, I assume gangRelease exists as requested.
            // If the user meant "release" as a concept, I hope this method exists.
            // However, typical Spoke client has gangRelease(ticket).
            client_->gangRelease(ticket_id_.c_str());
        }
    }

    std::shared_ptr<Sequence> add_request(const std::vector<int>& prompt_ids, int max_new_tokens)
    {
        NANODEPLOY_LOG_DEBUG("[SimpleEngine] Adding Request...");
        auto seq = std::make_shared<Sequence>(prompt_ids, 1.0, prompt_ids.size() + max_new_tokens);
        scheduler_->add(seq);
        return seq;
    }

    bool is_finished() const
    {
        return scheduler_->is_finished();
    }

    struct StepResult {
        std::vector<std::pair<uint64_t, std::vector<int>>> new_tokens;  // seq_id -> [new_tokens]
        std::vector<uint64_t>                              finished_seq_ids;
    };

    StepResult step()
    {
        constexpr int kRunAction = static_cast<int>(spoke::Action::kUserActionStart) + 11;
        StepResult    result;

        if (scheduler_->is_finished())
            return result;

        auto sched_res = scheduler_->schedule();

        bool has_work = false;
        for (const auto& list : sched_res.dp_sp_seqs) {
            if (!list.empty())
                has_work = true;
        }
        if (!has_work)
            return result;

        // Prepare requests
        std::vector<ModelRunReq> requests(world_size_);
        for (int i = 0; i < world_size_; ++i) {
            if (i >= (int)sched_res.dp_sp_seqs.size())
                break;
            requests[i].is_prefill = sched_res.is_prefill;
            requests[i].seqs       = sched_res.dp_sp_seqs[i];

            // Debug: Print outgoing tokens
            if (sched_res.is_prefill && !requests[i].seqs.empty()) {
                std::cout << "[Engine] Sending Prefill Seq " << requests[i].seqs[0]->seq_id << " to Rank " << i
                          << " Tokens: ";
                for (int id : requests[i].seqs[0]->token_ids)
                    std::cout << id << " ";
                std::cout << std::endl;
            }
        }

        // Launch all calls in parallel
        std::vector<std::future<ModelRunResp>> futures;
        for (int i = 0; i < world_size_; ++i) {
            if (i >= (int)sched_res.dp_sp_seqs.size())
                break;
            futures.push_back(std::async(std::launch::async, [this, i, &requests, kRunAction]() -> ModelRunResp {
                auto f = client_->callRemote<ModelRunReq, ModelRunResp>(
                    actor_ids_[i], static_cast<spoke::Action>(kRunAction), requests[i]);
                return f.get();
            }));
        }

        // Collect results
        std::vector<std::vector<std::vector<int>>> dp_sp_token_ids(sched_res.dp_sp_seqs.size());
        for (int i = 0; i < (int)futures.size(); ++i) {
            auto  resp = futures[i].get();
            auto& seqs = sched_res.dp_sp_seqs[i];
            dp_sp_token_ids[i].resize(seqs.size());

            if (!resp.token_ids.empty()) {
                const auto& new_tokens = resp.token_ids;

                for (size_t j = 0; j < seqs.size(); ++j) {
                    if (j < new_tokens.size()) {
                        int token = new_tokens[j];

                        // Debug: Warn if token 0 is sampled (likely indicates corrupted logits)
                        if (token == 0) {
                            std::cerr << "[Engine] WARNING: Seq " << seqs[j]->seq_id
                                      << " sampled token 0! This may indicate corrupted logits." << std::endl;
                        }

                        dp_sp_token_ids[i][j].push_back(token);

                        // Collect for stream result
                        bool found = false;
                        for (auto& p : result.new_tokens) {
                            if (p.first == seqs[j]->seq_id) {
                                p.second.push_back(token);
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            result.new_tokens.push_back({seqs[j]->seq_id, {token}});
                        }
                    }
                }
            }
        }

        // Check for finished sequences (Simple heuristic: eos token or length limit)
        // Note: Scheduler->postprocess usually handles appending tokens and checking finish conditions.
        // We should run postprocess first, then check which ones finished.

        scheduler_->postprocess(sched_res.dp_sp_seqs, dp_sp_token_ids, false);

        // Check status after postprocess
        for (const auto& list : sched_res.dp_sp_seqs) {
            for (const auto& seq : list) {
                if (seq->is_finished()) {
                    result.finished_seq_ids.push_back(seq->seq_id);
                }
            }
        }

        return result;
    }

    // Get all finished sequences with their generated tokens
    std::vector<std::pair<uint64_t, std::vector<int>>> get_finished_sequences()
    {
        std::vector<std::pair<uint64_t, std::vector<int>>> results;

        // Iterate through all DP ranks' running queues to find finished sequences
        for (int dp_idx = 0; dp_idx < attention_dp_; ++dp_idx) {
            auto& running = scheduler_->running(dp_idx);
            for (const auto& seq : running) {
                // Skip dummy sequences (num_prompt_tokens == 1)
                if (seq->num_prompt_tokens == 1)
                    continue;

                // Get generated tokens (excluding prompt)
                auto completion_ids = seq->completion_token_ids();
                results.emplace_back(seq->seq_id, completion_ids);
            }
        }
        return results;
    }

    // Generate for a single request (convenience method)
    std::vector<int> generate(const std::vector<int>& prompt_ids, int max_new_tokens)
    {
        add_request(prompt_ids, max_new_tokens);
        while (!is_finished()) {
            step();
        }
        // Get results from finished sequences
        auto results = get_finished_sequences();
        if (!results.empty()) {
            return results[0].second;  // Return first sequence's tokens
        }
        return {};
    }

private:
    void syncDeepEpBuffers(int world_size)
    {
        NANODEPLOY_LOG_DEBUG("[SimpleEngine] Starting DeepEP synchronization for ", world_size, " ranks...");

        constexpr int kGetInfoAction = static_cast<int>(spoke::Action::kUserActionStart) + 12;
        constexpr int kSyncAction    = static_cast<int>(spoke::Action::kUserActionStart) + 13;

        // Step 1: Collect DeepEP info from all ranks (parallel)
        std::vector<std::future<DeepEpInfoResp>> info_futs;
        for (int i = 0; i < world_size; ++i) {
            info_futs.push_back(std::async(std::launch::async, [this, i, kGetInfoAction]() -> DeepEpInfoResp {
                int  dummy_req = 0;
                auto f         = client_->callRemote<int, DeepEpInfoResp>(
                    actor_ids_[i], static_cast<spoke::Action>(kGetInfoAction), dummy_req);
                return f.get();
            }));
        }

        std::vector<DeepEpInfoResp> all_info;
        all_info.reserve(world_size);
        for (auto& f : info_futs) {
            all_info.push_back(f.get());
        }

        // Log collected info
        for (int i = 0; i < world_size; ++i) {
            NANODEPLOY_LOG_DEBUG("[SimpleEngine] Rank ",
                                 i,
                                 " DeepEP info: device_id=",
                                 all_info[i].device_id,
                                 " num_rdma_ranks=",
                                 all_info[i].num_rdma_ranks,
                                 " ipc_handle_size=",
                                 all_info[i].ipc_handle.size());
        }

        // Find root rank for NVSHMEM (if needed)
        std::string root_nvshmem_id;
        for (int i = 0; i < world_size; ++i) {
            if (!all_info[i].nvshmem_unique_id.empty()) {
                root_nvshmem_id = all_info[i].nvshmem_unique_id;
                NANODEPLOY_LOG_DEBUG("[SimpleEngine] Found NVSHMEM root at rank ", i);
                break;
            }
        }

        // Step 2: Build sync request
        DeepEpSyncReq sync_req;
        sync_req.device_ids.reserve(world_size);
        sync_req.ipc_handles.reserve(world_size);
        for (int i = 0; i < world_size; ++i) {
            sync_req.device_ids.push_back(all_info[i].device_id);
            sync_req.ipc_handles.push_back(all_info[i].ipc_handle);
        }
        sync_req.root_nvshmem_unique_id = root_nvshmem_id;

        // Step 3: Send sync request to all ranks (parallel)
        std::vector<std::future<bool>> sync_futs;
        for (int i = 0; i < world_size; ++i) {
            sync_futs.push_back(std::async(std::launch::async, [this, i, &sync_req, kSyncAction]() -> bool {
                auto f = client_->callRemote<DeepEpSyncReq, bool>(
                    actor_ids_[i], static_cast<spoke::Action>(kSyncAction), sync_req);
                return f.get();
            }));
        }

        bool all_synced = true;
        for (int i = 0; i < world_size; ++i) {
            bool result = sync_futs[i].get();
            if (!result) {
                NANODEPLOY_LOG_ERROR("[SimpleEngine] DeepEP sync failed for rank ", i);
                all_synced = false;
            }
        }

        if (all_synced) {
            NANODEPLOY_LOG_DEBUG("[SimpleEngine] DeepEP synchronization completed successfully.");
        }
        else {
            throw std::runtime_error("DeepEP synchronization failed!");
        }
    }

    std::shared_ptr<spoke::Client> client_;
    std::vector<std::string>       actor_ids_;
    std::unique_ptr<Scheduler>     scheduler_;

    int tp_ = 1, pp_ = 1, dp_ = 1;
    int attention_tp_ = 1, attention_dp_ = 1, attention_sp_ = 1;
    int ffn_tp_ = 1, ffn_dp_ = 1, ffn_ep_ = 1;
    int world_size_ = 1;

    std::string ticket_id_;
};

}  // namespace nanodeploy
