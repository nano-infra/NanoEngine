#pragma once
#include "nanodeploy/csrc/engine/engine.h"
#include "nanodeploy/csrc/engine/engine_ipc.h"
#include "spoke/csrc/actor.h"

namespace nanodeploy {

class EngineActor: public spoke::Actor {
public:
    EngineActor(const std::string& id, int rx, int tx): spoke::Actor(id, rx, tx)
    {
        // Register RPC methods
        SPOKE_METHOD(EngineActor,
                     init,
                     static_cast<spoke::Action>(static_cast<int>(spoke::Action::kUserActionStart) + 1),
                     EngineInitReq,
                     bool);
        SPOKE_METHOD(EngineActor,
                     add_request,
                     static_cast<spoke::Action>(static_cast<int>(spoke::Action::kUserActionStart) + 2),
                     EngineAddReq,
                     bool);
        // Deprecated manual control methods
        SPOKE_METHOD(EngineActor,
                     step,
                     static_cast<spoke::Action>(static_cast<int>(spoke::Action::kUserActionStart) + 3),
                     int,
                     EngineStepResp);
        SPOKE_METHOD(EngineActor,
                     get_finished,
                     static_cast<spoke::Action>(static_cast<int>(spoke::Action::kUserActionStart) + 4),
                     int,
                     EngineGetFinishedResp);
        SPOKE_METHOD(EngineActor,
                     release,
                     static_cast<spoke::Action>(static_cast<int>(spoke::Action::kUserActionStart) + 5),
                     int,
                     bool);
    }

    ~EngineActor()
    {
        stop_loop_ = true;
        if (loop_thread_.joinable())
            loop_thread_.join();
    }

    bool init(EngineInitReq req)
    {
        NANODEPLOY_LOG_INFO("[EngineActor] Initializing SimpleEngine...");
        engine_ = std::make_unique<SimpleEngine>();
        try {
            engine_->init(req.config_path,
                          req.tp,
                          req.pp,
                          req.dp,
                          req.hub_ip,
                          req.hub_port,
                          req.attention_tp,
                          req.attention_dp,
                          req.attention_sp,
                          req.ffn_tp,
                          req.ffn_dp,
                          req.ffn_ep,
                          req.enable_rdma,
                          req.enable_cuda_graph);

            // Start background loop
            stop_loop_   = false;
            loop_thread_ = std::thread(&EngineActor::loop, this);
            return true;
        }
        catch (const std::exception& e) {
            NANODEPLOY_LOG_ERROR("[EngineActor] Init failed: ", e.what());
            return false;
        }
    }

    bool add_request(EngineAddReq req)
    {
        if (!engine_)
            return false;

        NANODEPLOY_LOG_DEBUG("[EngineActor] Adding request with slot_id=",
                             req.slot_id,
                             " Prompts=",
                             req.prompt_ids.size(),
                             " First=",
                             (req.prompt_ids.empty() ? -1 : req.prompt_ids[0]));
        auto seq = engine_->add_request(req.prompt_ids, req.max_new_tokens);

        // Map seq_id to client slot_id for streaming
        {
            std::lock_guard<std::mutex> lk(map_mtx_);
            seq_to_slot_[seq->seq_id] = req.slot_id;
        }
        return true;
    }

    // Deprecated
    EngineStepResp step(int)
    {
        return {true};
    }
    EngineGetFinishedResp get_finished(int)
    {
        return {};
    }

    bool release(int)
    {
        NANODEPLOY_LOG_INFO("[EngineActor] Received Release Request. Releasing resources...");
        if (engine_) {
            engine_->release_resources();
            engine_->shutdown();
            engine_.reset();
        }
        stop_loop_ = true;
        return true;
    }

private:
    void loop()
    {
        while (!stop_loop_) {
            if (!engine_ || engine_->is_finished()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            // Step engine
            auto res = engine_->step();

            // Push tokens
            for (const auto& pair : res.new_tokens) {
                uint64_t    seq_id = pair.first;
                const auto& tokens = pair.second;

                uint32_t slot_id = 0;
                {
                    std::lock_guard<std::mutex> lk(map_mtx_);
                    if (seq_to_slot_.count(seq_id))
                        slot_id = seq_to_slot_[seq_id];
                }

                if (slot_id > 0) {
                    StreamToken st;
                    st.seq_id   = seq_id;
                    st.tokens   = tokens;
                    st.finished = false;

                    std::string body = spoke::Pack(st);
                    pushToClient(slot_id, body);
                }
            }

            // Push finish signals
            for (uint64_t seq_id : res.finished_seq_ids) {
                uint32_t slot_id = 0;
                {
                    std::lock_guard<std::mutex> lk(map_mtx_);
                    if (seq_to_slot_.count(seq_id)) {
                        slot_id = seq_to_slot_[seq_id];
                        seq_to_slot_.erase(seq_id);  // Cleanup
                    }
                }

                if (slot_id > 0) {
                    StreamToken st;
                    st.seq_id   = seq_id;
                    st.tokens   = {};  // No extra tokens in finish message usually
                    st.finished = true;

                    std::string body = spoke::Pack(st);
                    pushToClient(slot_id, body);
                    NANODEPLOY_LOG_DEBUG("[EngineActor] Pushed finish for seq ", seq_id, " slot ", slot_id);
                }
            }
        }
    }

    std::unique_ptr<SimpleEngine> engine_;
    std::thread                   loop_thread_;
    std::atomic<bool>             stop_loop_{false};
    std::mutex                    map_mtx_;
    std::map<uint64_t, uint32_t>  seq_to_slot_;
};

}  // namespace nanodeploy
