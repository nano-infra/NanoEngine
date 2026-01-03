#pragma once

#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
// Use NCCL as requested
#ifdef USE_C10D_NCCL
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#endif
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/torch.h>

#include "nanodeploy/logging.h"
#include "nanodeploy/sequence/sequence.h"

namespace nanodeploy {

// Request/Response types for DummyRunner
using RunReq  = std::vector<std::shared_ptr<Sequence>>;
using RunResp = torch::Tensor;

struct DistConfig {
    int         rank;
    int         world_size;
    std::string master_addr;
    int         master_port;
};

struct DistResp {
    bool        success;
    std::string message;
};

class DummyRunner {
public:
    DummyRunner() = default;

    // Pure logic: run sequences and return tensor
    RunResp run(RunReq dp_seqs)
    {
        int num_sequence = dp_seqs.size();
        // Log rank if distributed
        if (pg_ptr_ && *pg_ptr_) {
            // Basic distributed check
            // std::cout << "[Rank " << rank_ << "] running batch " << num_sequence << std::endl;
        }
        return torch::zeros({num_sequence, 16}, torch::kFloat32);
    }

    // Real c10d initialization
    DistResp init_distributed(DistConfig config)
    {
        try {
            rank_       = config.rank;
            world_size_ = config.world_size;

            std::cout << "[DummyRunner] Rank " << rank_ << " connecting to store at " << config.master_addr << ":"
                      << config.master_port << std::endl
                      << std::flush;

            // Create TCPStore
            c10d::TCPStoreOptions store_opts;
            store_opts.port       = config.master_port;
            store_opts.isServer   = (config.rank == 0);
            store_opts.numWorkers = config.world_size;
            // store_opts.waitHandlers = true; // Optional

            store_ptr_ = new c10::intrusive_ptr<c10d::Store>(
                c10::make_intrusive<c10d::TCPStore>(config.master_addr, store_opts));

            c10::intrusive_ptr<c10d::Store> store_ref = *store_ptr_;

#ifdef USE_C10D_NCCL
            // Create ProcessGroupNCCL
            auto options = c10d::ProcessGroupNCCL::Options::create();
            pg_ptr_      = new c10::intrusive_ptr<c10d::Backend>(
                c10::make_intrusive<c10d::ProcessGroupNCCL>(store_ref, config.rank, config.world_size, options));

            return DistResp{true, "Initialized ProcessGroupNCCL"};
#else
            return DistResp{false, "USE_C10D_NCCL not defined!"};
#endif
        }
        catch (const std::exception& e) {
            return DistResp{false, std::string("Init failed: ") + e.what()};
        }
    }

    // New method for AllReduce test
    // Returns the result tensor (should be sum of all ranks)
    torch::Tensor run_allreduce()
    {
        NANODEPLOY_ASSERT(pg_ptr_ && *pg_ptr_, "ProcessGroup not initialized! Call init_distributed first.");
        NANODEPLOY_ASSERT(torch::cuda::is_available(), "NCCL requires CUDA!");
        NANODEPLOY_ASSERT(rank_ >= 0, "Invalid rank", rank_);

        // NCCL requires correct device
        // Set device to rank % device_count?
        int           num_gpus  = torch::cuda::device_count();
        int           device_id = rank_ % num_gpus;
        torch::Device device(torch::kCUDA, device_id);

        // Create tensor on GPU: [rank]
        auto tensor = torch::tensor({(float)rank_}, torch::dtype(torch::kFloat32).device(device));

        // AllReduce (SUM)
        std::vector<at::Tensor> tensors = {tensor};
        auto                    work    = (*pg_ptr_)->allreduce(tensors);
        work->wait();

        // Move back to CPU for response serialization
        return tensors[0].cpu();
    }

    // Explicit destructor: Do NOT destroy PG/Store to avoid double-free/crash on exit.
    // The OS will clean up process resources.
    ~DummyRunner() = default;

private:
    int rank_       = -1;
    int world_size_ = -1;
    // Use pointer to intrusive_ptr to avoid automatic destruction
    c10::intrusive_ptr<c10d::Backend>* pg_ptr_    = nullptr;
    c10::intrusive_ptr<c10d::Store>*   store_ptr_ = nullptr;
};

}  // namespace nanodeploy
