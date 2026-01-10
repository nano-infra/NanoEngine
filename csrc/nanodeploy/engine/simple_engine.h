#pragma once

#include "nanodeploy/scheduler/scheduler.h"
#include "nanodeploy/worker/model_runner_ipc.h"  // For Req/Resp definitions
#include "spoke/csrc/client.h"
#include <iostream>
#include <memory>
#include <vector>

namespace nanodeploy {

class SimpleEngine {
public:
    SimpleEngine() = default;

    void init(const std::string& config_path,
              int                tp   = 1,
              int                pp   = 1,
              int                dp   = 1,
              const std::string& ip   = "127.0.0.1",
              int                port = 8888)
    {
        // 1. Connect to Spoke Hub
        std::cout << "[SimpleEngine] Connecting to Spoke Hub at " << ip << ":" << port << std::endl;
        client_ = std::make_shared<spoke::Client>(ip, port, /*is_hub_mode=*/true);

        int world_size = tp * pp * dp;

        // 2. Resource Allocation (Gang Scheduling)
        std::cout << "[SimpleEngine] Requesting Gang Allocation for " << world_size << " actors (TP=" << tp << ")..."
                  << std::endl;
        spoke::ResourceSpec res;
        res.num_gpus = 1;  // Assume 1 GPU per actor

        // Request resources: We want to pack as many as possible, but strict packing for TP is crucial.
        // For now, we ask for 1 node if world_size <= 8, else multiple.
        // But better: Let Hub handle packing, we just ask for world_size slots.
        // Wait, gangAllocate takes (num_nodes, actors_per_node, ...).
        // We should calculate num_nodes based on world_size and available resources?
        // Or just ask for "enough nodes".
        // Current API: gangAllocate(uint32_t num_nodes, uint32_t actors_per_node, ...)
        // This imposes a rigid structure (equal actors per node).
        // Let's assume we fit into minimal nodes.
        // E.g. TP=2 -> 1 Node, 2 Actors.
        // TP=8 -> 1 Node, 8 Actors.
        // TP=16 -> 2 Nodes, 8 Actors.

        int actors_per_node = 8;  // Default assumption for GPUs
        if (world_size < 8)
            actors_per_node = world_size;
        int num_nodes = (world_size + actors_per_node - 1) / actors_per_node;

        // Strict pack true ensures they are packed.
        auto alloc_fut  = client_->gangAllocate(num_nodes, actors_per_node, res, /*strict_pack=*/true);
        auto alloc_resp = alloc_fut.get();

        if (strlen(alloc_resp.ticket_id) == 0) {
            throw std::runtime_error("Gang Allocation Failed!");
        }
        std::cout << "[SimpleEngine] Allocation Success! Ticket: " << alloc_resp.ticket_id << std::endl;

        // 3. Topology Validation
        // TODO: Validate that TP groups are on the same node.
        // alloc_resp contains slots?
        // The current Client::gangAllocate implementation in client.h parses AllocateResp BUT NOT the trailing slots.
        // Wait, looking at client.h:
        // "Body = AllocateResp + Slots..."
        // "memcpy(&resp, body.data(), sizeof(AllocateResp));"
        // It DOES NOT copy the slots into the returned structure if AllocateResp doesn't have a flexible array member
        // pointer. The AllocateResp struct in types.h likely has `num_members` but the slots follow it. The
        // Client::gangAllocate returns `AllocateResp` by value, slicing off the slots if they are not part of the
        // struct definition or managed. I need to update Client::gangAllocate or access the slots.

        // For now, I will skip strict topology validation in Engine and trust Hub's strict_pack=true.
        // (The User Plan step 3 says "Launch Actors: Iterate through slots...").
        // I need the slots to know where to connect?
        // Actually, launchActor uses `ticket_id` + `global_rank`. The Hub knows where they are.
        // Initialization (RPC) uses `actor_id` routing via Hub.
        // So I don't strictly need the slots locally IF I route via Hub.

        // 4. Launch Actors
        std::vector<std::string> actor_ids;
        for (int i = 0; i < world_size; ++i) {
            std::string id = "ModelRunner:" + std::to_string(i);
            actor_ids.push_back(id);

            // Launch via Hub
            // Args: None for now, or maybe rank?
            // Passing rank via args is useful for the process to know who it is immediately.
            std::string args = "--rank " + std::to_string(i);
            // NOTE: Client::launchActor takes serialized args.

            // Async launch
            client_->launchActor(alloc_resp.ticket_id, i, "ModelRunner", id, args);
        }

        // Wait for startup (simple sleep for now, or retry connect)
        std::cout << "[SimpleEngine] Waiting for actors to start..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2));

        // 5. Initialize All Actors
        std::cout << "[SimpleEngine] Initializing remote models..." << std::endl;
        constexpr int kInitAction = static_cast<int>(spoke::Action::kUserActionStart) + 10;

        std::vector<std::future<ModelInitResp>> init_futs;
        for (int i = 0; i < world_size; ++i) {
            ModelInitReq req;
            req.config_path = config_path;
            req.rank        = i;
            req.world_size  = world_size;
            req.tp_degree   = tp;
            req.pp_degree   = pp;
            req.dp_degree   = dp;

            init_futs.push_back(client_->callRemote<ModelInitReq, ModelInitResp>(
                actor_ids[i], static_cast<spoke::Action>(kInitAction), req));
        }

        for (auto& f : init_futs) {
            f.get();
        }
        std::cout << "[SimpleEngine] All actors initialized." << std::endl;

        // 6. Init RDMA with Rank 0 (Driver -> Rank 0)
        actor_id_ = actor_ids[0];  // Driver talks to Rank 0
        if (!client_->initRDMA(actor_id_)) {
            std::cerr << "[SimpleEngine] Failed to init RDMA for " << actor_id_ << std::endl;
            throw std::runtime_error("RDMA Init Failed");
        }
        std::cout << "[SimpleEngine] RDMA Configured with Rank 0." << std::endl;

        // 7. Initialize Scheduler (Config Hardcoded for Single GPU Demo)
        std::string engine_id              = "simple_engine";
        int         loop_count             = 1;
        int         max_num_seqs           = 256;
        int         max_num_batched_tokens = 8192;
        int         eos                    = 151643;  // Qwen2 EOS
        int         attention_dp           = 1;
        int         attention_sp           = 1;
        int         num_kv_blocks          = 1024;
        int         block_size             = Sequence::block_size;  // Sync with Sequence
        std::string mode                   = "decode";

        scheduler_ = std::make_unique<Scheduler>(engine_id,
                                                 loop_count,
                                                 max_num_seqs,
                                                 max_num_batched_tokens,
                                                 eos,
                                                 attention_dp,
                                                 attention_sp,
                                                 num_kv_blocks,
                                                 block_size,
                                                 mode);
    }

    void shutdown()
    {
        if (client_) {
            std::cout << "[SimpleEngine] Shutting down client..." << std::endl;
            client_.reset();
        }
    }

    // Step-by-Step API
    void add_request(const std::vector<int>& prompt_ids, int max_new_tokens)
    {
        std::cout << "[SimpleEngine] Adding Request..." << std::endl;
        auto seq =
            std::make_shared<Sequence>(prompt_ids, /*temp=*/1.0, /*max_tokens=*/prompt_ids.size() + max_new_tokens);
        scheduler_->add(seq);
    }

    bool is_finished() const
    {
        return scheduler_->is_finished();
    }

    std::vector<int> step()
    {
        std::vector<int> step_tokens;
        constexpr int    kRunAction = static_cast<int>(spoke::Action::kUserActionStart) + 11;

        if (scheduler_->is_finished()) {
            return step_tokens;
        }

        auto  sched_res  = scheduler_->schedule();
        auto& batch_seqs = sched_res.dp_sp_seqs[0];

        if (sched_res.is_prefill) {
            std::cout << "[SimpleEngine] Scheduled PREFILL." << std::endl;
        }
        else {
            std::cout << "[SimpleEngine] Scheduled DECODE." << std::endl;
        }

        if (batch_seqs.empty())
            return step_tokens;

        // Prepare Request
        ModelRunReq req;
        req.is_prefill = sched_res.is_prefill;
        req.seqs       = batch_seqs;

        // Call Remote
        auto resp =
            client_->callRemote<ModelRunReq, ModelRunResp>(actor_id_, static_cast<spoke::Action>(kRunAction), req)
                .get();

        if (resp.tensor.numel() == 0)
            return step_tokens;

        // Extract outputs (Assuming [batch, vocab] logits or similar)
        auto logits = resp.tensor;
        // For decode phase with seq_len=1, logits may be [B, 1, V] or [B, V]
        // Squeeze to ensure [B, V] before argmax
        if (logits.dim() == 3) {
            logits = logits.squeeze(1);  // [B, S, V] -> [B, V] when S=1
        }
        auto next_tokens = torch::argmax(logits, -1);  // [batch_size]

        // Postprocess preparation
        std::vector<std::vector<std::vector<int>>> dp_sp_token_ids(1);
        dp_sp_token_ids[0].resize(1);

        auto next_tokens_cpu = next_tokens.cpu();
        auto access          = next_tokens_cpu.accessor<int64_t, 1>();

        for (int k = 0; k < batch_seqs.size(); ++k) {
            int token = (int)access[k];
            dp_sp_token_ids[0][0].push_back(token);

            // Note: For multi-seq, this returns mixed tokens.
            // Caller should track by seq_id if needed, but for now we return all generated in this step.
            step_tokens.push_back(token);
            std::cout << token << " " << std::flush;
        }

        scheduler_->postprocess(sched_res.dp_sp_seqs, dp_sp_token_ids, /*update_metrics=*/false);
        return step_tokens;
    }

    std::vector<int> generate(const std::vector<int>& prompt_ids, int max_new_tokens)
    {
        std::cout << "[SimpleEngine] Starting Scheduled Generation..." << std::endl;

        add_request(prompt_ids, max_new_tokens);

        std::vector<int> output_tokens;

        while (!is_finished()) {
            auto new_tokens = step();
            output_tokens.insert(output_tokens.end(), new_tokens.begin(), new_tokens.end());
        }
        std::cout << std::endl;
        return output_tokens;
    }

private:
    std::shared_ptr<spoke::Client> client_;
    std::string                    actor_id_;
    std::unique_ptr<Scheduler>     scheduler_;
};

}  // namespace nanodeploy
