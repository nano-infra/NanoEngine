#pragma once

#include "nanodeploy/logging.h"
#include "nanodeploy/scheduler/scheduler.h"
#include "nanodeploy/worker/model_runner_ipc.h"  // For Req/Resp definitions
#include "spoke/csrc/client.h"
#include <future>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

namespace nanodeploy {

class SimpleEngine {
public:
    SimpleEngine() = default;

    void init(const std::string& config_path,
              int                tp           = 1,
              int                pp           = 1,
              int                dp           = 1,
              const std::string& ip           = "127.0.0.1",
              int                port         = 8888,
              int                attention_tp = 1,
              int                attention_dp = 1,
              int                attention_sp = 1,
              int                ffn_tp       = 1,
              int                ffn_dp       = 1,
              int                ffn_ep       = 1)
    {
        // 1. Connect to Spoke Hub
        std::cout << "[SimpleEngine] Connecting to Spoke Hub at " << ip << ":" << port << std::endl;
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
        std::cout << "[SimpleEngine] Requesting Gang Allocation for " << world_size << " actors..." << std::endl;
        spoke::ResourceSpec res;
        res.num_gpus = 1;

        int actors_per_node = 8;
        if (world_size < 8)
            actors_per_node = world_size;
        int num_nodes = (world_size + actors_per_node - 1) / actors_per_node;

        auto alloc_fut  = client_->gangAllocate(num_nodes, actors_per_node, res, /*strict_pack=*/true);
        auto alloc_resp = alloc_fut.get();

        if (strlen(alloc_resp.ticket_id) == 0) {
            throw std::runtime_error("Gang Allocation Failed!");
        }
        std::cout << "[SimpleEngine] Allocation Success! Ticket: " << alloc_resp.ticket_id << std::endl;

        // 3. Launch Actors (Parallel)
        actor_ids_.clear();
        std::vector<std::future<void>> launch_futs;
        std::string                    ticket = alloc_resp.ticket_id;

        for (int i = 0; i < world_size; ++i) {
            std::string id = "ModelRunner:" + std::to_string(i);
            actor_ids_.push_back(id);
            std::string args = "--rank " + std::to_string(i);

            launch_futs.push_back(std::async(std::launch::async, [this, ticket, i, id, args]() {
                client_->launchActor(ticket.c_str(), i, "ModelRunner", id, args);
            }));
        }
        for (auto& f : launch_futs)
            f.get();

        std::cout << "[SimpleEngine] Waiting for actors to start..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2));

        // 4. Initialize All Actors (Parallel)
        std::cout << "[SimpleEngine] Initializing remote models (Parallel)..." << std::endl;
        constexpr int kInitAction = static_cast<int>(spoke::Action::kUserActionStart) + 10;

        std::vector<std::future<ModelInitResp>> init_futs;
        for (int i = 0; i < world_size; ++i) {
            init_futs.push_back(
                std::async(std::launch::async, [this, i, world_size, config_path, kInitAction]() -> ModelInitResp {
                    ModelInitReq req;
                    req.config_path = config_path;
                    req.rank        = i;
                    req.world_size  = world_size;
                    req.tp_degree   = tp_;
                    req.pp_degree   = pp_;
                    req.dp_degree   = dp_;

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
        std::cout << "[SimpleEngine] All actors initialized." << std::endl;

        // 5. DeepEP Synchronization (for multi-GPU MoE)
        if (ffn_ep_ > 1) {
            syncDeepEpBuffers(world_size);
        }

        // 6. Skip RDMA for multi-actor setup (RDMA is per-actor, not global)
        // TODO: Implement per-actor RDMA channels if needed for performance
        // For now, use socket path for all actors to ensure correct routing
        if (world_size == 1) {
            if (!client_->initRDMA(actor_ids_[0])) {
                std::cerr << "[SimpleEngine] Failed to init RDMA for " << actor_ids_[0] << std::endl;
                throw std::runtime_error("RDMA Init Failed");
            }
            std::cout << "[SimpleEngine] RDMA Configured with Rank 0." << std::endl;
        }
        else {
            std::cout << "[SimpleEngine] Skipping RDMA (multi-actor mode uses socket path)." << std::endl;
        }

        // 6. Initialize Scheduler
        std::string engine_id = "simple_engine";
        scheduler_            = std::make_unique<Scheduler>(
            engine_id, 1, 256, 8192, 151643, attention_dp_, attention_sp_, 1024, Sequence::block_size, "decode");
    }

    void shutdown()
    {
        if (client_) {
            std::cout << "[SimpleEngine] Shutting down client..." << std::endl;
            client_.reset();
        }
    }

    std::shared_ptr<Sequence> add_request(const std::vector<int>& prompt_ids, int max_new_tokens)
    {
        std::cout << "[SimpleEngine] Adding Request..." << std::endl;
        auto seq = std::make_shared<Sequence>(prompt_ids, 1.0, prompt_ids.size() + max_new_tokens);
        scheduler_->add(seq);
        return seq;
    }

    bool is_finished() const
    {
        return scheduler_->is_finished();
    }

    void step()
    {
        constexpr int kRunAction = static_cast<int>(spoke::Action::kUserActionStart) + 11;

        if (scheduler_->is_finished())
            return;

        auto sched_res = scheduler_->schedule();

        bool has_work = false;
        for (const auto& list : sched_res.dp_sp_seqs) {
            if (!list.empty())
                has_work = true;
        }
        if (!has_work)
            return;

        // Prepare requests
        std::vector<ModelRunReq> requests(world_size_);
        for (int i = 0; i < world_size_; ++i) {
            if (i >= (int)sched_res.dp_sp_seqs.size())
                break;
            requests[i].is_prefill = sched_res.is_prefill;
            requests[i].seqs       = sched_res.dp_sp_seqs[i];
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

            if (resp.tensor.numel() > 0) {
                auto logits = resp.tensor;
                if (logits.dim() == 3)
                    logits = logits.squeeze(1);
                auto next_tokens = torch::argmax(logits, -1).cpu();
                auto access      = next_tokens.accessor<int64_t, 1>();

                for (int j = 0; j < (int)seqs.size(); ++j) {
                    if (j < next_tokens.size(0)) {
                        dp_sp_token_ids[i][j].push_back((int)access[j]);
                    }
                }
            }
        }

        scheduler_->postprocess(sched_res.dp_sp_seqs, dp_sp_token_ids, false);
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
        std::cout << "[SimpleEngine] Starting DeepEP synchronization for " << world_size << " ranks..." << std::endl;

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
            std::cout << "[SimpleEngine] Rank " << i << " DeepEP info: device_id=" << all_info[i].device_id
                      << " num_rdma_ranks=" << all_info[i].num_rdma_ranks
                      << " ipc_handle_size=" << all_info[i].ipc_handle.size() << std::endl;
        }

        // Find root rank for NVSHMEM (if needed)
        std::string root_nvshmem_id;
        for (int i = 0; i < world_size; ++i) {
            if (!all_info[i].nvshmem_unique_id.empty()) {
                root_nvshmem_id = all_info[i].nvshmem_unique_id;
                std::cout << "[SimpleEngine] Found NVSHMEM root at rank " << i << std::endl;
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
                std::cerr << "[SimpleEngine] DeepEP sync failed for rank " << i << std::endl;
                all_synced = false;
            }
        }

        if (all_synced) {
            std::cout << "[SimpleEngine] DeepEP synchronization completed successfully." << std::endl;
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
};

}  // namespace nanodeploy
