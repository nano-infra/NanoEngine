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

    void init(const std::string& config_path, const std::string& ip = "127.0.0.1", int port = 9000)
    {
        // 1. Connect to Spoke Daemon
        std::cout << "[SimpleEngine] Connecting to Spoke Daemon at " << ip << ":" << port << std::endl;
        client_ = std::make_shared<spoke::Client>(ip, port, /*enable_rdma=*/true);

        // 2. Spawn ModelRunner Actor
        actor_id_ = "ModelRunner:0";
        std::cout << "[SimpleEngine] Spawning actor " << actor_id_ << "..." << std::endl;
        client_->spawnRemote("ModelRunner", actor_id_);

        // 3. Init RDMA
        if (!client_->initRDMA(actor_id_)) {
            std::cerr << "[SimpleEngine] Failed to init RDMA for " << actor_id_ << std::endl;
            throw std::runtime_error("RDMA Init Failed");
        }
        std::cout << "[SimpleEngine] RDMA Configured." << std::endl;

        // 4. Send Init RPC
        ModelInitReq init_req;
        init_req.config_path = config_path;
        init_req.rank        = 0;
        init_req.world_size  = 1;

        std::cout << "[SimpleEngine] Initializing remote model..." << std::endl;
        constexpr int kInitAction = static_cast<int>(spoke::Action::kUserActionStart) + 10;
        auto          resp =
            client_
                ->callRemote<ModelInitReq, ModelInitResp>(actor_id_, static_cast<spoke::Action>(kInitAction), init_req)
                .get();

        std::cout << "[SimpleEngine] Init done." << std::endl;

        // 5. Initialize Scheduler (Config Hardcoded for Single GPU Demo)
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
        } else {
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
