#include "model_runner.h"

#include <algorithm>
#include <c10/cuda/CUDAGuard.h>
#include <iostream>
#include <torch/torch.h>
#include <torch/types.h>
#include <unordered_set>

#include "context/deep_ep_context.h"
#include "nanodeploy/csrc/logging.h"

#ifdef DEEPSEEK_MOE
#include "deep_ep.hpp"
#include "nanodeploy/csrc/ops/deep_ep_utils.h"
#include <cuda_runtime.h>
#endif

namespace nanodeploy {

using namespace models;

// =========================================================================
// WeightManager Implementation
// =========================================================================

WeightManager::WeightManager(const std::filesystem::path& model_dir, torch::Device device):
    model_dir_(model_dir), device_(device)
{
    if (!std::filesystem::exists(model_dir_)) {
        throw std::runtime_error("Model directory does not exist: " + model_dir_.string());
    }

    // Check for index file
    auto index_path = model_dir_ / "model.safetensors.index.json";
    if (std::filesystem::exists(index_path)) {
        is_sharded_ = true;
        std::ifstream  f(index_path);
        nlohmann::json j          = nlohmann::json::parse(f);
        auto           weight_map = j["weight_map"];
        for (auto it = weight_map.begin(); it != weight_map.end(); ++it) {
            param_to_file_[it.key()] = it.value();
        }
    }
    else {
        auto single_path = model_dir_ / "model.safetensors";
        if (std::filesystem::exists(single_path)) {
            loaders_.emplace("model.safetensors", SafeTensorLoader(single_path.string()));
        }
        else {
            std::cerr << "[WeightManager] Warning: No index.json or model.safetensors found in " << model_dir_
                      << std::endl;
        }
    }
}

bool WeightManager::has_param(const std::string& param_name)
{
    if (is_sharded_) {
        return param_to_file_.find(param_name) != param_to_file_.end();
    }
    else {
        return !loaders_.empty();
    }
}

torch::Tensor WeightManager::load(const std::string& param_name)
{
    std::string filename;

    if (is_sharded_) {
        auto it = param_to_file_.find(param_name);
        if (it == param_to_file_.end()) {
            throw std::runtime_error("Parameter not found in index: " + param_name);
        }
        filename = it->second;
    }
    else {
        filename = "model.safetensors";
    }

    if (loaders_.find(filename) == loaders_.end()) {
        auto path = model_dir_ / filename;
        NANODEPLOY_LOG_INFO("Opening new loader for: ", path);
        loaders_.emplace(filename, SafeTensorLoader(path.string()));
    }

    auto t = loaders_.at(filename).load(param_name, device_);
    return t;
}

// =========================================================================
// ModelRunner Implementation
// =========================================================================

void ModelRunner::init(const std::string& config_path_str, int rank, int world_size)
{
    DistributedConfig dconf;
    dconf.global_rank = rank;
    dconf.world_size  = world_size;
    init(config_path_str, dconf);
}

void ModelRunner::init(const std::string& config_path_str, const DistributedConfig& dconf)
{
    get_dist_context().init(dconf);
    enable_cuda_graph_ = dconf.enable_cuda_graph;
    NANODEPLOY_LOG_INFO("ModelRunner::init: enable_cuda_graph=", enable_cuda_graph_);
    // enable_cuda_graph_ is now part of DistributedConfig
    init_internal(config_path_str, dconf.global_rank);
}

void ModelRunner::init_internal(const std::string& config_path_str, int rank)
{
    rank_       = rank;
    world_size_ = get_dist_context().world_size();

    torch::set_num_threads(1);

    std::filesystem::path config_path(config_path_str);
    auto                  model_dir = config_path.parent_path();

    // =========================================================
    // Fix for DeepGEMM JIT Compilation (Missing CUTLASS Headers)
    // =========================================================
    // Attempt to locate CUTLASS headers relative to the current working directory or binary location
    // Typical path: ./third_party/DeepGemm/third-party/cutlass/include
    namespace fs              = std::filesystem;
    fs::path cwd              = fs::current_path();
    fs::path deepgemm_cutlass = cwd / "third_party/DeepGemm/third-party/cutlass/include";

    // Check adjacent directory structure (e.g. if running from build dir)
    if (!fs::exists(deepgemm_cutlass)) {
        deepgemm_cutlass = cwd / "../third_party/DeepGemm/third-party/cutlass/include";
    }

    if (fs::exists(deepgemm_cutlass)) {
        std::string new_path      = deepgemm_cutlass.string();
        const char* env_cpath     = std::getenv("CPATH");
        std::string current_cpath = env_cpath ? std::string(env_cpath) : "";

        if (current_cpath.find(new_path) == std::string::npos) {
            std::string updated_cpath = new_path;
            if (!current_cpath.empty()) {
                updated_cpath += ":" + current_cpath;
            }
            // Overwrite CPATH
            setenv("CPATH", updated_cpath.c_str(), 1);
            NANODEPLOY_LOG_INFO("[AutoConfig] Appended CUTLASS to CPATH: ", new_path);
        }
    }
    else {
        NANODEPLOY_LOG_WARN("[AutoConfig] Could not find CUTLASS headers at: ", deepgemm_cutlass.string());
    }

    NANODEPLOY_LOG_INFO("Loading config from ", config_path);
    config_ = std::make_unique<core::ModelConfig>(core::ModelConfig::load_hf(config_path.string()));

    if (config_->is_moe && get_dist_context().ffn_ep() == 1 && world_size_ > 1) {
        NANODEPLOY_LOG_WARN("MoE detected but EP degree is 1. Running in TP/DP only mode?");
    }

    if (torch::cuda::is_available()) {
        int num_devices = 0;
        cudaGetDeviceCount(&num_devices);
        NANODEPLOY_LOG_INFO("CUDA device count: ", num_devices, " rank_: ", rank_);
        if (num_devices > 0) {
            int device_id = rank_ % num_devices;
            NANODEPLOY_LOG_INFO("Setting CUDA device to ", device_id, " for rank ", rank_);
            cudaSetDevice(device_id);
            device_ = torch::Device(torch::kCUDA, device_id);

            // Verify device was set correctly
            int current_device = -1;
            cudaGetDevice(&current_device);
            NANODEPLOY_LOG_INFO("CUDA device after cudaSetDevice: ", current_device, " (expected ", device_id, ")");

            // Initialize capture stream
            capture_stream_ = c10::cuda::getStreamFromPool(true, device_.index());
            NANODEPLOY_LOG_INFO("Initialized capture stream: ", (void*)capture_stream_.stream());
        }
        else {
            device_ = torch::kCPU;
            NANODEPLOY_LOG_INFO("CUDA Not Available (Count=0). Using CPU.");
        }
    }
    else {
        device_ = torch::kCPU;
        NANODEPLOY_LOG_INFO("CUDA Not Available. Using CPU.");
    }

    NANODEPLOY_LOG_INFO("Initializing WeightManager for ", model_dir);
    weight_manager_ = std::make_unique<WeightManager>(model_dir, device_);

    NANODEPLOY_LOG_INFO("init: Allocating Model (FP16)...");

    NANODEPLOY_LOG_INFO("Initializing Contexts...");

    // 1. Attention Context
    attn_ctx_ = std::make_unique<AttentionContext>(device_);
    attn_ctx_->init(*config_);

    // 2. DeepGemm Context
    gemm_ctx_ = std::make_unique<DeepGemmContext>();
    gemm_ctx_->init(*config_);

    // 3. DeepEP Context

    if (config_->is_moe) {
        NANODEPLOY_LOG_INFO("      MoE Model Detected. QuantMethod: ", config_->quant_method);
        NANODEPLOY_LOG_INFO("      Experts: ", config_->num_experts, " TopK: ", config_->num_experts_per_tok);

        if (config_->quant_method == "fp8") {
            NANODEPLOY_LOG_INFO("      Instantiating FP8 MoE Model...");
            moe_model_fp8_ = std::make_unique<Qwen3MoeForCausalLM<QuantType::FP8_E4M3>>(*config_, device_);
        }
        else {
            NANODEPLOY_LOG_INFO("      Instantiating BF16/FP16 MoE Model...");
            moe_model_ = std::make_unique<Qwen3MoeForCausalLM<QuantType::FP16>>(*config_, device_);
        }

        int ep_size = get_dist_context().ffn_ep();
        int ep_rank = get_dist_context().ffn_ep_rank();

        get_deep_ep_context().init(*config_, ep_size, ep_rank);
    }
    else {
        model_ = std::make_unique<Qwen3ForCausalLM<QuantType::FP16>>(*config_, device_);
    }

    load_weights("");
}

ModelRunResp ModelRunner::run(ModelRunReq req)
{
    // Enable Inference Mode for the entire run execution to allow modifying Inference Tensors (static buffers)
    torch::InferenceMode guard(true);

    NANODEPLOY_LOG_DEBUG(
        "ModelRunner::run called. Rank: ", rank_, " Seqs: ", req.seqs.size(), " Prefill: ", req.is_prefill);

    if (req.seqs.empty()) {
        NANODEPLOY_LOG_WARN("ModelRunner::run: Empty seqs for Rank ", rank_, ". Returning empty response.");
        // WARN: This might cause deadlock if other ranks are waiting in collective ops!
        // We should probably proceed with empty tensors if collective ops are expected.
        // For now, just logging.

        return ModelRunResp{};
    }

    std::vector<int64_t> input_ids_vec;
    std::vector<int64_t> positions_vec;
    std::vector<int32_t> slot_mapping_vec;
    std::vector<int32_t> block_tables_vec;
    std::vector<int32_t> seq_lens_vec;
    std::vector<int32_t> q_indptr_vec;        // Added for prefill
    std::vector<int32_t> last_page_lens_vec;  // Added for prefill
    q_indptr_vec.push_back(0);
    int32_t current_len_acc = 0;

    int batch_size     = req.seqs.size();
    int max_num_blocks = 0;
    int block_size     = Sequence::block_size;

    for (const auto& seq : req.seqs) {
        if ((int)seq->block_table().size() > max_num_blocks)
            max_num_blocks = seq->block_table().size();
    }
    if (max_num_blocks == 0)
        max_num_blocks = 1;

    for (const auto& seq : req.seqs) {
        auto& blocks = seq->block_table();
        for (int id : blocks)
            block_tables_vec.push_back(id);
        for (int k = blocks.size(); k < max_num_blocks; ++k)
            block_tables_vec.push_back(0);

        int   seql_len             = seq->num_tokens;
        auto& dist_ctx             = nanodeploy::get_dist_context();
        int   sp_rank              = dist_ctx.attention_sp_rank();
        int   seq_len_in_this_rank = seq->context_len(nanodeploy::BlockContextSlot::ACTIVE, sp_rank);
        seq_lens_vec.push_back(seq_len_in_this_rank);
        current_len_acc += seq_len_in_this_rank;
        q_indptr_vec.push_back(current_len_acc);

        if (req.is_prefill) {
            // Get last page len based on total dispatched tokens (handles append/padding correctly)
            // Note: rank_ is our DP/SP rank. The Sequence tracks context per SP rank.
            // Assumption: ModelRunner rank maps to SP index 0?
            // Actually ModelRunner is one process. It has 'attention_sp_' config?
            // For now assuming sp_idx=0 is safe for simple DP, or we need attention_sp info.
            // We use block_ctx(ACTIVE).master_sp_idx_ or 0?
            // The method `last_block_num_tokens(ACTIVE, 0)` is safest if we don't know SP index.
            // But if we are in SP, we need local SP index.
            // Since we don't have SP logic fully visible here, let's assume index 0
            // OR use the correct one if we can find it.
            // blocks track capacity.
            // SP Rank calculation:
            auto& dist_ctx = nanodeploy::get_dist_context();
            int   sp_rank  = dist_ctx.attention_sp_rank();

            int lp_len = 0;
            if (!blocks.empty()) {
                lp_len = seq->last_block_num_tokens(nanodeploy::BlockContextSlot::ACTIVE, sp_rank);
            }
            last_page_lens_vec.push_back(lp_len);

            if (rank_ == 0) {
                // Debug lp_len calculation
                NANODEPLOY_LOG_INFO("[ModelRunner] Prefill Seq ",
                                    seq->seq_id,
                                    " Tokens: ",
                                    seq->num_tokens,
                                    " Blocks: ",
                                    blocks.size(),
                                    " LastPageLen: ",
                                    lp_len);
            }
        }

        if (req.is_prefill) {
            // Debug: Print input ids
            if (rank_ == 0) {
                std::cout << "[ModelRunner] Prefill Seq " << seq->seq_id << " Tokens: ";
                for (int id : seq->token_ids)
                    std::cout << id << " ";
                std::cout << std::endl;
                std::cout << "[ModelRunner] Prefill Seq " << seq->seq_id << " Blocks: ";
                for (int id : seq->block_table())
                    std::cout << id << " ";
                std::cout << std::endl;
            }
            for (int id : seq->token_ids)
                input_ids_vec.push_back(id);
            for (int i = 0; i < seq->token_ids.size(); ++i)
                positions_vec.push_back(i);
            for (int i = 0; i < seq->token_ids.size(); ++i) {
                slot_mapping_vec.push_back(seq->slot_mapping_idx(nanodeploy::BlockContextSlot::ACTIVE, sp_rank, i));
            }
        }
        else {
            input_ids_vec.push_back(seq->token_ids.back());
            int pos = seq->num_tokens - 1;
            positions_vec.push_back(pos);
            if (blocks.empty())
                slot_mapping_vec.push_back(0);
            else {
                auto& dist_ctx = nanodeploy::get_dist_context();
                int   sp_rank  = dist_ctx.attention_sp_rank();
                slot_mapping_vec.push_back(seq->slot_mapping_idx(nanodeploy::BlockContextSlot::ACTIVE, sp_rank));
            }
        }
    }

    auto input_ids = torch::from_blob(input_ids_vec.data(), {(long)input_ids_vec.size()}, torch::kLong).to(device_);
    auto positions = torch::from_blob(positions_vec.data(), {(long)positions_vec.size()}, torch::kLong).to(device_);
    auto slot_mapping =
        torch::from_blob(slot_mapping_vec.data(), {(long)slot_mapping_vec.size()}, torch::kInt32).to(device_);
    auto block_tables =
        torch::from_blob(block_tables_vec.data(), {batch_size, max_num_blocks}, torch::kInt32).to(device_);
    auto seq_lens = torch::from_blob(seq_lens_vec.data(), {batch_size}, torch::kInt32).to(device_);

    // Check for Graph Execution
    bool use_graph = false;
    int  target_bs = 1;

    if (!req.is_prefill && !decode_graphs_.empty()) {
        while (target_bs < batch_size)
            target_bs *= 2;
        if (target_bs <= max_batch_size_ && decode_graphs_.count(target_bs)) {
            use_graph = true;
        }
    }

    NANODEPLOY_LOG_DEBUG("ModelRunner::run: batch_size=",
                         batch_size,
                         " prefill=",
                         req.is_prefill,
                         " use_graph=",
                         use_graph,
                         " graphs_size=",
                         decode_graphs_.size());

    torch::Tensor logits;

    if (use_graph) {
        if (!static_input_ids_.defined()) {
            NANODEPLOY_LOG_ERROR("CRITICAL: static_input_ids_ is undefined but use_graph=true!");
            return {};
        }

        // 1. Copy Inputs to Static Buffers
        // We use async copy where possible, but from_blob is Host.
        // We copy actual data to start of static buffers.
        // We must ensure padding is safe (0s).

        // Helper to copy and pad
        auto copy_pad = [&](torch::Tensor& dst, int64_t actual_size, void* src_data, int64_t pad_size) {
            // Copy actual
            cudaMemcpyAsync(
                dst.data_ptr(), src_data, actual_size, cudaMemcpyHostToDevice, c10::cuda::getCurrentCUDAStream());
            // Zero pad (launch kernel)
            if (pad_size > 0) {
                // dst is typed. Byte offset? No, pointer arithmetic.
                // We can invoke .slice().fill_(0) but that might be slow/synchronous launch overhead?
                // Actually slice().fill_ is kernel launch.
                // Optimization: Just trust previous 0s if we clean up? Or valid data doesn't matter?
                // Safer: Zero it.
                // dst.slice(0, batch_size, target_bs).fill_(0);
                // For performance, we assume padding 0 is needed for correctness (FlashInfer empty requests).
                // Use cudaMemsetAsync?
                // Simplest PyTorch:
                dst.slice(0, batch_size, target_bs).fill_(0);
            }
        };

        // Input IDs
        static_input_ids_.slice(0, 0, batch_size).copy_(input_ids);  // Is input_ids allocated above?
        // Wait, input_ids above IS dynamic allocation from previous lines:
        // "auto input_ids = torch::from_blob(...).to(device_)" -> Allocation!
        // We wanted to avoid dynamic allocation.
        // Re-write: Copy directly from input_ids_vec (Host) to static_input_ids_ (Device).

        cudaMemcpyAsync(static_input_ids_.data_ptr(),
                        input_ids_vec.data(),
                        input_ids_vec.size() * sizeof(int64_t),
                        cudaMemcpyHostToDevice,
                        c10::cuda::getCurrentCUDAStream());
        if (target_bs > batch_size)
            static_input_ids_.slice(0, batch_size, target_bs).fill_(0);

        cudaMemcpyAsync(static_positions_.data_ptr(),
                        positions_vec.data(),
                        positions_vec.size() * sizeof(int64_t),
                        cudaMemcpyHostToDevice,
                        c10::cuda::getCurrentCUDAStream());  // input was int32 converted to long?
        // positions_vec is int64_t (line 205).
        if (target_bs > batch_size)
            static_positions_.slice(0, batch_size, target_bs).fill_(0);

        cudaMemcpyAsync(static_slot_mapping_.data_ptr(),
                        slot_mapping_vec.data(),
                        slot_mapping_vec.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice,
                        c10::cuda::getCurrentCUDAStream());
        if (target_bs > batch_size)
            static_slot_mapping_.slice(0, batch_size, target_bs).fill_(0);

        cudaMemcpyAsync(static_seq_lens_.data_ptr(),
                        seq_lens_vec.data(),
                        seq_lens_vec.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice,
                        c10::cuda::getCurrentCUDAStream());
        if (target_bs > batch_size)
            static_seq_lens_.slice(0, batch_size, target_bs).fill_(0);

        // Block Tables: vec is flat [batch * max_num_blocks].
        // Static is [max_bs, max_context_blocks].
        // Dynamic max_num_blocks might be != max_context_blocks (likely smaller).
        // We must copy row by row? Or can we copy contiguous?
        // block_tables_vec has stride `max_num_blocks`.
        // static has stride `max_context_blocks`.
        // Must copy row-by-row if strides differ.
        // Optimization: Launch a kernel to scatter? Or loop copy?
        // For now: Using Tensor copy.
        // Construct temporary host view of vec?
        auto block_tables_host = torch::from_blob(block_tables_vec.data(), {batch_size, max_num_blocks}, torch::kInt32);
        // Copy into static slice
        static_block_tables_.slice(0, 0, batch_size).slice(1, 0, max_num_blocks).copy_(block_tables_host);

        // 2. Prepare FlashInfer Metadata (CPU -> Static Device)
        // We need padding seq_lens for FlashInfer setup?
        // Pad `seq_lens_vec` with 0s?
        std::vector<int32_t> padded_seq_lens = seq_lens_vec;
        padded_seq_lens.resize(target_bs, 0);

        // Pad block tables?? `begin_forward` takes `block_tables_host` pointer.
        // It iterates `batch_size`.
        // We call `begin_forward` with `target_bs`.
        // It will read `target_bs` rows.
        // So we need `block_tables_host` to have `target_bs` rows?
        // What stride does `begin_forward` assume? `max_num_blocks` argument.
        // We pass `max_context_blocks` (static stride) to `begin_forward`.
        // So `block_tables_host` must have that stride.
        // `block_tables_vec` has dynamic stride.
        // We must repack block tables on HOST to match static stride if we want to pass it to `begin_forward`.
        // This repack is overhead.

        // WORKAROUND: FlashInfer's `begin_forward` builds `flat_indices`.
        // Typically it iterates:
        // for i in batch: for b in num_blocks: indices.push(block_table[i][b])
        // It accesses block_table via accessor [i][b].
        // Accessor depends on stride.
        // So yes, we must provide data matching stride.

        // We need a persistent host buffer for block tables (padded)?
        // Allocating vector every time is slow.
        // But let's assume stride conversion cost is acceptable for now.
        std::vector<int32_t> padded_block_tables_host(target_bs * static_block_tables_.size(1), 0);
        int                  stride = static_block_tables_.size(1);
        for (int i = 0; i < batch_size; ++i) {
            int num_blks = (seq_lens_vec[i] + block_size - 1) / block_size;
            for (int k = 0; k < num_blks; ++k) {
                padded_block_tables_host[i * stride + k] = block_tables_vec[i * max_num_blocks + k];
            }
        }

        // Log input_ids for verification (even in CUDA Graph path)
        std::cout << "  input_ids (CUDA Graph): ";
        for (size_t i = 0; i < input_ids_vec.size(); ++i) {
            if (seq_lens_vec[i] > 10) {
                std::cout << "[" << i << "](" << seq_lens_vec[i] << "):" << input_ids_vec[i] << " ";
            }
        }
        std::cout << std::endl;

        attn_ctx_->get_handler()->begin_forward(padded_block_tables_host.data(),
                                                padded_seq_lens.data(),
                                                target_bs,
                                                stride,
                                                config_->num_attention_heads,
                                                config_->num_key_value_heads,
                                                config_->hidden_size / config_->num_attention_heads,
                                                block_size);

        // 3. Launch Graph
        auto& instance = decode_graphs_[target_bs];
        cudaGraphLaunch(instance, c10::cuda::getCurrentCUDAStream());

        // 4. Get Result
        // Output is in decode_graph_outputs_[target_bs]
        // Slice to actual batch size and clone (trigger D2H later or copy to resp tensor)
        logits = decode_graph_outputs_[target_bs].slice(0, 0, batch_size).clone();
    }
    else {
        // Dynamic Fallback

        // (Original Logic)
        if (req.is_prefill) {
            std::cout << "[ModelRunner] PREFILL DEBUG:" << std::endl;
            std::cout << "  q_indptr: ";
            for (auto v : q_indptr_vec)
                std::cout << v << " ";
            std::cout << std::endl;
            std::cout << "  last_page_lens: ";
            for (auto v : last_page_lens_vec)
                std::cout << v << " ";
            std::cout << std::endl;
            std::cout << "  block_tables (first " << std::min((size_t)10, block_tables_vec.size()) << "): ";
            for (size_t i = 0; i < std::min((size_t)10, block_tables_vec.size()); ++i)
                std::cout << block_tables_vec[i] << " ";
            std::cout << std::endl;
            std::cout << "  slot_mapping (first " << std::min((size_t)20, slot_mapping_vec.size()) << "): ";
            for (size_t i = 0; i < std::min((size_t)20, slot_mapping_vec.size()); ++i)
                std::cout << slot_mapping_vec[i] << " ";
            std::cout << std::endl;
            std::cout << "  positions (first " << std::min((size_t)20, positions_vec.size()) << "): ";
            for (size_t i = 0; i < std::min((size_t)20, positions_vec.size()); ++i)
                std::cout << positions_vec[i] << " ";
            std::cout << std::endl;
            attn_ctx_->get_handler()->begin_forward_prefill(q_indptr_vec.data(),
                                                            block_tables_vec.data(),
                                                            last_page_lens_vec.data(),
                                                            batch_size,
                                                            max_num_blocks,
                                                            config_->num_attention_heads,
                                                            config_->num_key_value_heads,
                                                            config_->hidden_size / config_->num_attention_heads,
                                                            block_size);
        }
        else {
            std::cout << "[ModelRunner] DECODE DEBUG:" << std::endl;
            std::cout << "  seq_lens: ";
            for (auto v : seq_lens_vec)
                std::cout << v << " ";
            std::cout << std::endl;
            std::cout << "  block_tables (first " << std::min((size_t)10, block_tables_vec.size()) << "): ";
            for (size_t i = 0; i < std::min((size_t)10, block_tables_vec.size()); ++i)
                std::cout << block_tables_vec[i] << " ";
            std::cout << std::endl;
            std::cout << "last_block_num_tokens: "
                      << req.seqs[0]->last_block_num_tokens(nanodeploy::BlockContextSlot::ACTIVE, 0) << std::endl;
            std::cout << "  slot_mapping: ";
            for (auto v : slot_mapping_vec)
                std::cout << v << " ";
            std::cout << std::endl;
            std::cout << "  positions: ";
            for (auto v : positions_vec)
                std::cout << v << " ";
            std::cout << std::endl;
            std::cout << "  input_ids: ";
            for (size_t i = 0; i < input_ids_vec.size(); ++i) {
                if (seq_lens_vec[i] > 10) {
                    std::cout << "[" << i << "](" << seq_lens_vec[i] << "):" << input_ids_vec[i] << " ";
                }
            }
            std::cout << std::endl;
            attn_ctx_->get_handler()->begin_forward(block_tables_vec.data(),
                                                    seq_lens_vec.data(),
                                                    batch_size,
                                                    max_num_blocks,
                                                    config_->num_attention_heads,
                                                    config_->num_key_value_heads,
                                                    config_->hidden_size / config_->num_attention_heads,
                                                    block_size);
        }

        // Run Forward
        torch::Tensor hidden_states;

        if (config_->is_moe) {
#ifdef DEEPSEEK_MOE
            if (moe_model_fp8_) {
                hidden_states = moe_model_fp8_->forward(input_ids,
                                                        positions,
                                                        attn_ctx_->get_kv_cache(),
                                                        attn_ctx_->get_handler(),
                                                        slot_mapping,
                                                        block_tables,
                                                        seq_lens,
                                                        req.is_prefill);
            }
            else {
                hidden_states = moe_model_->forward(input_ids,
                                                    positions,
                                                    attn_ctx_->get_kv_cache(),
                                                    attn_ctx_->get_handler(),
                                                    slot_mapping,
                                                    block_tables,
                                                    seq_lens,
                                                    req.is_prefill);
            }
#else
            // Fallback if DEEPSEEK_MOE not defined (should not happen in this env)
            if (moe_model_fp8_) {
                hidden_states = moe_model_fp8_->forward(input_ids,
                                                        positions,
                                                        attn_ctx_->get_kv_cache(),
                                                        attn_ctx_->get_handler(),
                                                        slot_mapping,
                                                        block_tables,
                                                        seq_lens);
            }
            else {
                hidden_states = moe_model_->forward(input_ids,
                                                    positions,
                                                    attn_ctx_->get_kv_cache(),
                                                    attn_ctx_->get_handler(),
                                                    slot_mapping,
                                                    block_tables,
                                                    seq_lens);
            }
#endif
        }
        else {
            hidden_states = model_->forward(input_ids,
                                            positions,
                                            attn_ctx_->get_kv_cache(),
                                            attn_ctx_->get_handler(),
                                            slot_mapping,
                                            block_tables,
                                            seq_lens);
        }

        // Compute Logits
        if (req.is_prefill) {
            std::vector<int64_t> last_token_indices;
            int64_t              current_offset = 0;
            for (const auto& seq : req.seqs) {
                last_token_indices.push_back(current_offset + (int)seq->token_ids.size() - 1);
                current_offset += seq->token_ids.size();
            }
            auto indices = torch::from_blob(last_token_indices.data(), {(long)last_token_indices.size()}, torch::kLong)
                               .to(device_);

            if (hidden_states.dim() == 3)
                hidden_states = hidden_states.reshape({-1, (long)config_->hidden_size});

            auto last_hidden = hidden_states.index_select(0, indices);

            if (config_->is_moe) {
                if (moe_model_fp8_)
                    logits = moe_model_fp8_->compute_logits(last_hidden);
                else
                    logits = moe_model_->compute_logits(last_hidden);
            }
            else
                logits = model_->compute_logits(last_hidden);
        }
        else {
            if (config_->is_moe) {
                if (moe_model_fp8_)
                    logits = moe_model_fp8_->compute_logits(hidden_states);
                else
                    logits = moe_model_->compute_logits(hidden_states);
            }
            else
                logits = model_->compute_logits(hidden_states);
        }
    }

    // --- Sampling Implementation (Moved from Engine) ---
    // 1. Move logits to CPU for sampling
    auto logits_cpu = logits.cpu().to(torch::kFloat32);  // [batch, vocab]
    if (logits_cpu.dim() == 3) {
        logits_cpu = logits_cpu.squeeze(1);
    }

    std::vector<int> result_token_ids;
    result_token_ids.reserve(batch_size);

    float repetition_penalty = 1.0f;
    float temperature        = 0.0f;
    float top_p              = 1.0f;

    for (int j = 0; j < batch_size; ++j) {
        auto  seq_logits = logits_cpu[j];  // View
        auto& seq        = req.seqs[j];

        NANODEPLOY_LOG_INFO("C++ Logits: [",
                            seq_logits.slice(0, 0, 10),
                            "...], Max: ",
                            seq_logits.max().item<float>(),
                            ", Max Index: ",
                            seq_logits.argmax().item<long>(),
                            ", Sum: ",
                            seq_logits.sum().item<float>());

        // 2. Apply Repetition Penalty
        // req.seqs contains full history now (due to serialization change)
        auto                    generated_tokens = seq->completion_token_ids();
        std::unordered_set<int> seen(generated_tokens.begin(), generated_tokens.end());

        for (int token_id : seen) {
            if (token_id < seq_logits.size(0)) {
                float score = seq_logits[token_id].item<float>();
                if (score < 0) {
                    seq_logits[token_id] = score * repetition_penalty;
                }
                else {
                    seq_logits[token_id] = score / repetition_penalty;
                }
            }
        }

        // 3. Temperature & Top-P Sampling
        int next_token = -1;

        if (temperature > 0.0f) {
            // Temperature Scaling
            seq_logits /= temperature;
            auto probs = torch::softmax(seq_logits, -1);

            // Top-P (Nucleus) Sampling
            if (top_p < 1.0f && top_p > 0.0f) {
                auto sorted_tuple   = torch::sort(probs, -1, true);
                auto sorted_probs   = std::get<0>(sorted_tuple);
                auto sorted_indices = std::get<1>(sorted_tuple);

                auto cumulative_probs = torch::cumsum(sorted_probs, -1);

                // Remove tokens with cumulative probability above the threshold
                auto sorted_indices_to_remove = cumulative_probs > top_p;

                // Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove.slice(0, 1, sorted_indices_to_remove.size(0))
                    .copy_(sorted_indices_to_remove.slice(0, 0, sorted_indices_to_remove.size(0) - 1).clone());
                sorted_indices_to_remove[0] = false;

                // Scatter sorted_indices_to_remove to original indices
                auto indices_to_remove = torch::zeros_like(sorted_indices_to_remove, torch::kBool)
                                             .scatter_(0, sorted_indices, sorted_indices_to_remove);

                probs.masked_fill_(indices_to_remove, 0.0f);
                probs = probs / probs.sum();  // Re-normalize
            }

            next_token = torch::multinomial(probs, 1).item<long>();
        }
        else {
            // Greedy
            next_token = torch::argmax(seq_logits, -1).item<long>();
        }

        result_token_ids.push_back(next_token);
    }

    return ModelRunResp{result_token_ids};
}

void ModelRunner::load_weights(const std::string& /*weight_path*/)
{
    NANODEPLOY_LOG_INFO("Loading Embeddings...");
    if (config_->is_moe) {
        if (moe_model_fp8_)
            moe_model_fp8_->model_->embed_tokens_->weight =
                weight_manager_->load("model.embed_tokens.weight").to(torch::kFloat32);
        else
            moe_model_->model_->embed_tokens_->weight = weight_manager_->load("model.embed_tokens.weight");
    }
    else {
        model_->model_->embed_tokens_->weight = weight_manager_->load("model.embed_tokens.weight");
    }

    int total_layers = config_->num_hidden_layers;
    for (int i = 0; i < total_layers; ++i) {
        if (i % 1 == 0)
            NANODEPLOY_LOG_INFO("Loading Layer ", i, " / ", total_layers);

        if (config_->is_moe) {
            if (moe_model_fp8_)
                load_moe_layer_weights(i, moe_model_fp8_->model_->layers_[i].get());
            else
                load_moe_layer_weights(i, moe_model_->model_->layers_[i].get());
        }
        else {
            load_layer_weights(i, model_->model_->layers_[i].get());
        }
    }

    NANODEPLOY_LOG_INFO("Loading Final Norm...");
    if (config_->is_moe) {
        if (moe_model_fp8_) {
            moe_model_fp8_->model_->norm_->weight = weight_manager_->load("model.norm.weight").to(torch::kFloat32);
            moe_model_fp8_->lm_head_->weight      = weight_manager_->load("lm_head.weight").to(torch::kFloat32);
        }
        else {
            moe_model_->model_->norm_->weight = weight_manager_->load("model.norm.weight");
            moe_model_->lm_head_->weight      = weight_manager_->load("lm_head.weight");
        }
    }
    else {
        model_->model_->norm_->weight = weight_manager_->load("model.norm.weight");
        model_->lm_head_->weight      = weight_manager_->load("lm_head.weight");
    }
}

void ModelRunner::load_layer_weights(int layer_idx, Qwen3DecoderLayer<QuantType::FP16>* layer)
{
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";

    auto q_w = weight_manager_->load(prefix + "self_attn.q_proj.weight");
    auto k_w = weight_manager_->load(prefix + "self_attn.k_proj.weight");
    auto v_w = weight_manager_->load(prefix + "self_attn.v_proj.weight");
    if (q_w.scalar_type() == torch::kFloat8_e4m3fn) {
        q_w = q_w.clone().to(torch::kFloat16);
        k_w = k_w.clone().to(torch::kFloat16);
        v_w = v_w.clone().to(torch::kFloat16);
    }
    layer->self_attn_->qkv_proj_->weight = torch::cat({q_w, k_w, v_w}, 0);

    try {
        auto q_b                           = weight_manager_->load(prefix + "self_attn.q_proj.bias");
        auto k_b                           = weight_manager_->load(prefix + "self_attn.k_proj.bias");
        auto v_b                           = weight_manager_->load(prefix + "self_attn.v_proj.bias");
        layer->self_attn_->qkv_proj_->bias = torch::cat({q_b, k_b, v_b}, 0);
    }
    catch (...) {
        layer->self_attn_->qkv_proj_->bias = torch::zeros({layer->self_attn_->qkv_proj_->weight.size(0)},
                                                          layer->self_attn_->qkv_proj_->weight.options());
    }

    layer->self_attn_->o_proj_->weight = weight_manager_->load(prefix + "self_attn.o_proj.weight");

    try {
        layer->self_attn_->q_norm_->weight = weight_manager_->load(prefix + "self_attn.q_norm.weight");
        layer->self_attn_->k_norm_->weight = weight_manager_->load(prefix + "self_attn.k_norm.weight");
    }
    catch (...) {
    }

    layer->input_layernorm_->weight          = weight_manager_->load(prefix + "input_layernorm.weight");
    layer->post_attention_layernorm_->weight = weight_manager_->load(prefix + "post_attention_layernorm.weight");

    auto gate_w                        = weight_manager_->load(prefix + "mlp.gate_proj.weight");
    auto up_w                          = weight_manager_->load(prefix + "mlp.up_proj.weight");
    layer->mlp_->gate_up_proj_->weight = torch::cat(std::vector<torch::Tensor>{gate_w, up_w}, 0);
    layer->mlp_->down_proj_->weight    = weight_manager_->load(prefix + "mlp.down_proj.weight");
}

void ModelRunner::load_moe_layer_weights(int layer_idx, models::DeepSeekMoeDecoderLayer<QuantType::FP16>* layer)
{
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";

    // A. Attention (same as Qwen3MoeDecoderLayer)
    auto q_w = weight_manager_->load(prefix + "self_attn.q_proj.weight");
    auto k_w = weight_manager_->load(prefix + "self_attn.k_proj.weight");
    auto v_w = weight_manager_->load(prefix + "self_attn.v_proj.weight");

    layer->self_attn_->qkv_proj_->weight = torch::cat({q_w, k_w, v_w}, 0);

    try {
        auto q_b                           = weight_manager_->load(prefix + "self_attn.q_proj.bias");
        auto k_b                           = weight_manager_->load(prefix + "self_attn.k_proj.bias");
        auto v_b                           = weight_manager_->load(prefix + "self_attn.v_proj.bias");
        layer->self_attn_->qkv_proj_->bias = torch::cat({q_b, k_b, v_b}, 0);
    }
    catch (...) {
        auto out_features                  = layer->self_attn_->qkv_proj_->weight.size(0);
        layer->self_attn_->qkv_proj_->bias = torch::zeros({out_features}, q_w.options());
    }

    layer->self_attn_->o_proj_->weight = weight_manager_->load(prefix + "self_attn.o_proj.weight");
    layer->self_attn_->q_norm_->weight = weight_manager_->load(prefix + "self_attn.q_norm.weight");
    layer->self_attn_->k_norm_->weight = weight_manager_->load(prefix + "self_attn.k_norm.weight");

    {
        auto w = layer->self_attn_->q_norm_->weight.to(torch::kFloat32);
        NANODEPLOY_LOG_INFO("[Weight Check] L",
                            layer_idx,
                            " Q_Norm Weight Mean: ",
                            w.mean().item<float>(),
                            " Max: ",
                            w.max().item<float>());
    }

    layer->input_layernorm_->weight          = weight_manager_->load(prefix + "input_layernorm.weight");
    layer->post_attention_layernorm_->weight = weight_manager_->load(prefix + "post_attention_layernorm.weight");

    // B. MLP (Sparse or Dense)
    if (layer->is_sparse_) {
        // DeepSeek MoE Loading
        auto* block = layer->mlp_moe_.get();  // DeepSeekMoeSparseMoeBlock

        // Gate
        block->gate_->weight = weight_manager_->load(prefix + "mlp.gate.weight").to(torch::kFloat32);

        // Experts weights (tensor format: [num_local_experts, ...])
        int num_experts       = config_->num_experts;
        int ep_size           = get_dist_context().ffn_ep();
        int num_local_experts = num_experts / ep_size;

        // Load weights for each local expert
        std::vector<torch::Tensor> gate_up_list, down_list;
        for (int i = 0; i < num_local_experts; ++i) {
            // Calculate global expert index
            int         global_expert_idx = get_dist_context().ffn_ep_rank() * num_local_experts + i;
            std::string exp_prefix        = prefix + "mlp.experts." + std::to_string(global_expert_idx) + ".";

            auto g = weight_manager_->load(exp_prefix + "gate_proj.weight");
            auto u = weight_manager_->load(exp_prefix + "up_proj.weight");
            auto d = weight_manager_->load(exp_prefix + "down_proj.weight");

            // Concatenate gate and up: [moe_inter*2, hidden_size]
            // Storage format: [num_local_experts, moe_inter*2, hidden_size]
            gate_up_list.push_back(torch::cat({g, u}, 0));
            // Down proj: loaded as [hidden_size, moe_inter]
            // Storage format: [num_local_experts, hidden_size, moe_inter] (no transpose needed)
            down_list.push_back(d);
        }

        // Stack into tensors: [num_local_experts, moe_inter*2, hidden_size] and [num_local_experts, hidden_size,
        // moe_inter]
        block->gate_up_proj_ = torch::stack(gate_up_list, 0).to(torch::kBFloat16);
        block->down_proj_    = torch::stack(down_list, 0).to(torch::kBFloat16);
    }
    else {
        // Dense MLP (same as Qwen3MoeDecoderLayer)
        auto* mlp                  = layer->mlp_dense_.get();
        auto  gate_w               = weight_manager_->load(prefix + "mlp.gate_proj.weight");
        auto  up_w                 = weight_manager_->load(prefix + "mlp.up_proj.weight");
        mlp->gate_up_proj_->weight = torch::cat(std::vector<torch::Tensor>{gate_w, up_w}, 0);
        mlp->down_proj_->weight    = weight_manager_->load(prefix + "mlp.down_proj.weight");
    }
}

DeepEpInfoResp ModelRunner::getDeepEpInfo()
{
    return get_deep_ep_context().get_info();
}

bool ModelRunner::syncDeepEp(DeepEpSyncReq req)
{
    return get_deep_ep_context().sync(req);
}

KvCacheInitResp ModelRunner::init_kv_cache(KvCacheInitReq req)
{
    GetAvailableKVBlocksReq avail_req;
    avail_req.max_batch_size         = req.max_batch_size;
    avail_req.gpu_memory_utilization = req.gpu_memory_utilization;

    int available_blocks = getNumAvailableKVBlocks(avail_req);

    int num_blocks = req.num_blocks;
    if (num_blocks <= 0) {
        num_blocks = available_blocks;
    }

    AllocKVBlocksReq alloc_req;
    alloc_req.num_blocks     = num_blocks;
    alloc_req.max_batch_size = req.max_batch_size;

    if (allocKVBlocks(alloc_req)) {
        return num_blocks;
    }
    return 0;
}

GetAvailableKVBlocksResp ModelRunner::getNumAvailableKVBlocks(GetAvailableKVBlocksReq req)
{
    // Ensure calculation runs on correct device
    c10::cuda::CUDAGuard device_guard(device_);

    // Auto-calculate based on available memory using Python's logic:
    // num_blocks = int(total * ratio - used - peak + current) // block_bytes

    // Ensure all pending operations are complete before checking memory
    cudaDeviceSynchronize();

    size_t free_mem  = 0;
    size_t total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);

    // Get PyTorch memory stats using getDeviceStats
    auto   stats   = c10::cuda::CUDACachingAllocator::getDeviceStats(device_.index());
    size_t current = stats.allocated_bytes[0].current;
    size_t peak    = stats.allocated_bytes[0].peak;
    size_t used    = total_mem - free_mem;

    // Python formula: total * ratio - used - peak + current
    int64_t available = static_cast<int64_t>(total_mem * req.gpu_memory_utilization) - static_cast<int64_t>(used)
                        - static_cast<int64_t>(peak) + static_cast<int64_t>(current);

    if (available <= 0) {
        NANODEPLOY_LOG_WARN("[Auto KV] Available memory is negative! Using minimal allocation.");
        available = 512LL * 1024 * 1024;  // 512MB minimum
    }

    // Use config values directly - must match AttentionContext::init exactly
    int num_layers   = config_->num_hidden_layers;
    int num_kv_heads = config_->num_key_value_heads;
    // Match AttentionContext::init: use head_dim from config if set, else calculate
    int head_dim   = config_->head_dim > 0 ? config_->head_dim : (config_->hidden_size / config_->num_attention_heads);
    int block_size = Sequence::block_size;
    int element_size = 2;  // BF16

    // Match Python's block_bytes calculation exactly:
    // block_bytes = num_layers * block_size * num_kv_heads * head_dim * dtype_size * 2 (K+V)
    size_t block_bytes = (size_t)num_layers * block_size * num_kv_heads * head_dim * element_size * 2;

    int num_blocks = available / block_bytes;

    NANODEPLOY_LOG_INFO("[Auto KV] Device: ", device_.index());
    NANODEPLOY_LOG_INFO("[Auto KV] Total: ",
                        total_mem / 1024 / 1024,
                        " MB, Free: ",
                        free_mem / 1024 / 1024,
                        " MB, Used: ",
                        used / 1024 / 1024,
                        " MB");
    NANODEPLOY_LOG_INFO("[Auto KV] PyTorch: Peak=", peak / 1024 / 1024, " MB, Current=", current / 1024 / 1024, " MB");
    NANODEPLOY_LOG_INFO("[Auto KV] KV Params: layers=",
                        num_layers,
                        " kv_heads=",
                        num_kv_heads,
                        " head_dim=",
                        head_dim,
                        " block_size=",
                        block_size);
    NANODEPLOY_LOG_INFO("[Auto KV] Available: ",
                        available / 1024 / 1024,
                        " MB, Block Size: ",
                        block_bytes,
                        " bytes, Blocks: ",
                        num_blocks);

    if (num_blocks <= 0) {
        NANODEPLOY_LOG_WARN("[Auto KV] Calculated blocks is 0! Defaulting to 512.");
        num_blocks = 512;
    }

    return num_blocks;
}

AllocKVBlocksResp ModelRunner::allocKVBlocks(AllocKVBlocksReq req)
{
    // Ensure allocation runs on correct device
    c10::cuda::CUDAGuard device_guard(device_);

    max_batch_size_ = req.max_batch_size;
    int num_blocks  = req.num_blocks;

    NANODEPLOY_LOG_INFO("ModelRunner::allocKVBlocks: Device=",
                        device_.index(),
                        " max_bs=",
                        req.max_batch_size,
                        " num_blocks=",
                        num_blocks);

    if (attn_ctx_) {
        // block_size is currently fixed in Sequence::block_size
        if (attn_ctx_->init_kv_cache(req.max_batch_size, num_blocks, Sequence::block_size)) {
            return true;
        }
    }
    return false;
}

bool ModelRunner::warmup_moe()
{
    NANODEPLOY_LOG_INFO("warmup_moe: Triggering DeepGEMM JIT compilation...");

    if (max_batch_size_ <= 0) {
        NANODEPLOY_LOG_WARN("warmup_moe: max_batch_size not set, skipping.");
        return false;
    }

    c10::cuda::CUDAGuard device_guard(device_);
    torch::InferenceMode guard(true);

    // Use max_batch_size to ensure we compile the kernel needed for the largest input
    int bs = max_batch_size_;
    if (bs <= 0)
        bs = 1;

    auto options_long = torch::TensorOptions().dtype(torch::kLong).device(device_);
    auto options_int  = torch::TensorOptions().dtype(torch::kInt32).device(device_);

    // CRITICAL: For decode, input shape must be [batch_size, 1] (2D) to get seq_len=1 in attention
    // Shape [batch_size] (1D) would be interpreted as batch=1, seq_len=batch_size -> prefill path!
    auto input_ids = torch::randint(1, 1000, {bs, 1}, options_long);  // [bs, 1] for decode
    // auto input_ids    = torch::zeros({bs, 1}, options_long);  // [bs, 1] for decode - Zero init for determinism
    auto positions    = torch::zeros({bs, 1}, options_long);  // [bs, 1] - position 0 for each request
    auto slot_mapping = torch::arange(0, bs, options_int);    // [bs] - one slot per request

    // seq_lens for FlashInfer: context length for each request (seq_len=1 for first token decode)
    auto seq_lens = torch::ones({bs}, options_int);

    // Block tables (dummy, index 0 is valid for capture)
    int  max_context_blocks = 1;
    auto block_tables       = torch::zeros({bs, max_context_blocks}, options_int);

    // Prepare host arrays for FlashInfer begin_forward
    std::vector<int32_t> fake_block_tables_host(bs * max_context_blocks, 0);
    std::vector<int32_t> fake_seq_lens_host(bs, 1);  // seq_len=1 for each request

    // CRITICAL: Initialize FlashInfer handler BEFORE forward!
    // Without this, padded_batch_size=0 causing "Invalid Grid: Batch=0" error
    attn_ctx_->get_handler()->begin_forward(fake_block_tables_host.data(),
                                            fake_seq_lens_host.data(),
                                            bs,
                                            max_context_blocks,
                                            config_->num_attention_heads,
                                            config_->num_key_value_heads,
                                            config_->hidden_size / config_->num_attention_heads,
                                            Sequence::block_size);

    // Run forward to trigger JIT
    // try {
    if (config_->is_moe) {
#ifdef DEEPSEEK_MOE
        if (moe_model_fp8_) {
            moe_model_fp8_->forward(input_ids,
                                    positions,
                                    attn_ctx_->get_kv_cache(),
                                    attn_ctx_->get_handler(),
                                    slot_mapping,
                                    block_tables,
                                    seq_lens,
                                    false);
        }
        else {
            moe_model_->forward(input_ids,
                                positions,
                                attn_ctx_->get_kv_cache(),
                                attn_ctx_->get_handler(),
                                slot_mapping,
                                block_tables,
                                seq_lens,
                                false);
        }
#else
        // Fallback for non-DeepSeek MoE
        if (moe_model_fp8_) {
            moe_model_fp8_->forward(input_ids,
                                    positions,
                                    attn_ctx_->get_kv_cache(),
                                    attn_ctx_->get_handler(),
                                    slot_mapping,
                                    block_tables,
                                    seq_lens);
        }
        else {
            moe_model_->forward(input_ids,
                                positions,
                                attn_ctx_->get_kv_cache(),
                                attn_ctx_->get_handler(),
                                slot_mapping,
                                block_tables,
                                seq_lens);
        }
#endif
    }
    // }
    // catch (const std::exception& e) {
    // NANODEPLOY_LOG_ERROR("warmup_moe failed: ", e.what());
    // return false;
    // }

    cudaDeviceSynchronize();
    NANODEPLOY_LOG_INFO("warmup_moe: DeepGEMM JIT compilation triggered successfully.");
    return true;
}

GraphCaptureResp ModelRunner::capture_decode_graphs(GraphCaptureReq req)
{
    NANODEPLOY_LOG_DEBUG("ModelRunner::capture_decode_graphs: warmup=", req.warm_up_steps);

    // Ensure we are on the correct device
    c10::cuda::CUDAGuard device_guard(device_);
    // Disable autograd to prevent graph capture issues
    torch::InferenceMode guard(true);

    if (max_batch_size_ <= 0) {
        NANODEPLOY_LOG_ERROR("Max batch size not set! Call init_kv_cache first.");
        return false;
    }

    // 1. Allocate Static Buffers with VALID values (not zeros!)\n    // DeepGemm and MoE router can crash with all-zero
    // inputs due to degenerate routing.
    if (!static_input_ids_.defined()) {
        auto options_long = torch::TensorOptions().dtype(torch::kLong).device(device_);
        auto options_int  = torch::TensorOptions().dtype(torch::kInt32).device(device_);

        // Use valid token IDs within vocab range (avoid 0 which may be pad token)
        int64_t max_token_id = std::min(static_cast<int64_t>(config_->vocab_size) - 1, static_cast<int64_t>(10000));
        NANODEPLOY_LOG_INFO("capture_decode_graphs: vocab_size=", config_->vocab_size, " max_token_id=", max_token_id);
        // CRITICAL: For decode, input shape must be [batch_size, 1] (2D) to get seq_len=1 in attention
        // Shape [batch_size] (1D) would be interpreted as batch=1, seq_len=batch_size -> prefill path!
        static_input_ids_ = torch::randint(1, max_token_id, {max_batch_size_, 1}, options_long);
        // static_input_ids_ = torch::zeros({max_batch_size_, 1}, options_long); // Zero init for determinism
        // Positions: [batch_size, 1] - each request at position 0 for decode
        static_positions_ = torch::zeros({max_batch_size_, 1}, options_long);
        // Slot mapping: valid slot indices (0, 1, 2, ... within allocated KV cache)
        static_slot_mapping_ = torch::arange(0, max_batch_size_, options_int);
        // Seq lens: all 0 for capture (no context to read from KV cache)
        // This prevents page table access and potential illegal memory access during capture
        static_seq_lens_ = torch::zeros({max_batch_size_}, options_int);

        // Assume max blocks per request for capture stride
        int max_context_blocks = (config_->max_position_embeddings + Sequence::block_size - 1) / Sequence::block_size;
        // Block tables: valid block indices (0, 1, 2, ... for each block)
        static_block_tables_ = torch::zeros({max_batch_size_, max_context_blocks}, options_int);
        // Fill block tables with valid block indices (row i gets blocks i*max_context_blocks, i*max_context_blocks+1,
        // ...)
        for (int i = 0; i < max_batch_size_; ++i) {
            for (int j = 0; j < max_context_blocks; ++j) {
                // Use modulo to stay within allocated blocks
                static_block_tables_[i][j] = (i * max_context_blocks + j) % (max_batch_size_ * max_context_blocks);
            }
        }
    }
    else {
        NANODEPLOY_LOG_INFO("capture_decode_graphs: reusing static buffers from warmup_moe");
    }

    int max_context_blocks = (config_->max_position_embeddings + Sequence::block_size - 1) / Sequence::block_size;

    // 2. Prepare Fake Inputs (Host) for FlashInfer Setup
    std::vector<int32_t> fake_block_tables(max_batch_size_ * max_context_blocks);
    for (int i = 0; i < max_batch_size_ * max_context_blocks; ++i) {
        fake_block_tables[i] = i % (max_batch_size_ * max_context_blocks);
    }
    std::vector<int32_t> fake_seq_lens(max_batch_size_, 0);  // Length 0 to prevent KV cache access during capture

#ifdef DEEPSEEK_MOE
    // DISABLED: clean_low_latency_buffer may be causing memory issues
    // TODO: Re-investigate after baseline tests are working
    // if (get_deep_ep_context().get_buffer()) {
    //     int num_max_dispatch_tokens_per_rank = 256;
    //     NANODEPLOY_LOG_INFO("Cleaning DeepEP low latency buffer before capture...");
    //     get_deep_ep_context().get_buffer()->clean_low_latency_buffer(
    //         num_max_dispatch_tokens_per_rank, config_->hidden_size, config_->num_experts);
    //     cudaDeviceSynchronize();
    //     NANODEPLOY_LOG_INFO("DeepEP buffer cleaned.");
    // }
#endif

    // Warmup MoE before capture to pre-compile DeepGemm JIT kernels
#ifdef DEEPSEEK_MOE
    if (config_->is_moe) {
        NANODEPLOY_LOG_INFO("Calling warmup_moe before graph capture...");
        warmup_moe();
    }
#endif

    // 3. Capture Loop
    // 3. Capture Loop (Max BS -> 1)
    int bs = max_batch_size_;
    while (bs >= 1) {
        NANODEPLOY_LOG_DEBUG("Preparing Graph for BS=", bs);

        // Skip actual capture if disabled by config
        if (!enable_cuda_graph_) {
            NANODEPLOY_LOG_INFO("CUDA Graph capture disabled by config, skipping BS=", bs);
            if (bs == 1)
                break;
            bs /= 2;
            if (bs < 1)
                bs = 1;
            continue;
        }

        // Capture
        c10::cuda::CUDAStreamGuard stream_guard(capture_stream_);
        cudaStream_t               stream = capture_stream_.stream();

        // Ensure synchronization before we start using the capture stream
        cudaDeviceSynchronize();

        // 1. Attention Setup MUST be OUTSIDE capture region!
        attn_ctx_->get_handler()->begin_forward(fake_block_tables.data(),
                                                fake_seq_lens.data(),
                                                bs,
                                                max_context_blocks,
                                                config_->num_attention_heads,
                                                config_->num_key_value_heads,
                                                config_->hidden_size / config_->num_attention_heads,
                                                Sequence::block_size);

        // Sync to ensure begin_forward completes before warmup
        cudaStreamSynchronize(stream);

        // === WARMUP FORWARD ===
        // Run forward once BEFORE capture to prime all kernel JIT and allocator caches.
        // This is CRITICAL - embedding/index_select allocate output tensor dynamically.
        NANODEPLOY_LOG_DEBUG("Warmup forward for BS=", bs);
        {
            torch::Tensor warmup_hidden;
            if (config_->is_moe) {
#ifdef DEEPSEEK_MOE
                if (moe_model_fp8_) {
                    warmup_hidden = moe_model_fp8_->forward(static_input_ids_.slice(0, 0, bs),
                                                            static_positions_.slice(0, 0, bs),
                                                            attn_ctx_->get_kv_cache(),
                                                            attn_ctx_->get_handler(),
                                                            static_slot_mapping_.slice(0, 0, bs),
                                                            static_block_tables_.slice(0, 0, bs),
                                                            static_seq_lens_.slice(0, 0, bs),
                                                            false);
                }
                else {
                    warmup_hidden = moe_model_->forward(static_input_ids_.slice(0, 0, bs),
                                                        static_positions_.slice(0, 0, bs),
                                                        attn_ctx_->get_kv_cache(),
                                                        attn_ctx_->get_handler(),
                                                        static_slot_mapping_.slice(0, 0, bs),
                                                        static_block_tables_.slice(0, 0, bs),
                                                        static_seq_lens_.slice(0, 0, bs),
                                                        false);
                }
#else
                if (moe_model_fp8_) {
                    warmup_hidden = moe_model_fp8_->forward(static_input_ids_.slice(0, 0, bs),
                                                            static_positions_.slice(0, 0, bs),
                                                            attn_ctx_->get_kv_cache(),
                                                            attn_ctx_->get_handler(),
                                                            static_slot_mapping_.slice(0, 0, bs),
                                                            static_block_tables_.slice(0, 0, bs),
                                                            static_seq_lens_.slice(0, 0, bs));
                }
                else {
                    warmup_hidden = moe_model_->forward(static_input_ids_.slice(0, 0, bs),
                                                        static_positions_.slice(0, 0, bs),
                                                        attn_ctx_->get_kv_cache(),
                                                        attn_ctx_->get_handler(),
                                                        static_slot_mapping_.slice(0, 0, bs),
                                                        static_block_tables_.slice(0, 0, bs),
                                                        static_seq_lens_.slice(0, 0, bs));
                }
#endif
            }
            else {
                warmup_hidden = model_->forward(static_input_ids_.slice(0, 0, bs),
                                                static_positions_.slice(0, 0, bs),
                                                attn_ctx_->get_kv_cache(),
                                                attn_ctx_->get_handler(),
                                                static_slot_mapping_.slice(0, 0, bs),
                                                static_block_tables_.slice(0, 0, bs),
                                                static_seq_lens_.slice(0, 0, bs));
            }

            torch::Tensor warmup_logits;
            if (config_->is_moe) {
                if (moe_model_fp8_)
                    warmup_logits = moe_model_fp8_->compute_logits(warmup_hidden);
                else
                    warmup_logits = moe_model_->compute_logits(warmup_hidden);
            }
            else
                warmup_logits = model_->compute_logits(warmup_hidden);
        }
        cudaDeviceSynchronize();
        NANODEPLOY_LOG_DEBUG("Warmup complete for BS=", bs);

        cudaGraph_t     graph;
        cudaGraphExec_t instance;

        // Now begin capture - ONLY GPU kernel launches should be inside
        // Note: Sampler is intentionally NOT included in capture for future flexibility
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

        // 2. Model Forward (pure GPU kernels)
        torch::Tensor hidden_states;
        if (config_->is_moe) {
#ifdef DEEPSEEK_MOE
            if (moe_model_fp8_) {
                hidden_states = moe_model_fp8_->forward(static_input_ids_.slice(0, 0, bs),
                                                        static_positions_.slice(0, 0, bs),
                                                        attn_ctx_->get_kv_cache(),
                                                        attn_ctx_->get_handler(),
                                                        static_slot_mapping_.slice(0, 0, bs),
                                                        static_block_tables_.slice(0, 0, bs),
                                                        static_seq_lens_.slice(0, 0, bs),
                                                        false);
            }
            else {
                hidden_states = moe_model_->forward(static_input_ids_.slice(0, 0, bs),
                                                    static_positions_.slice(0, 0, bs),
                                                    attn_ctx_->get_kv_cache(),
                                                    attn_ctx_->get_handler(),
                                                    static_slot_mapping_.slice(0, 0, bs),
                                                    static_block_tables_.slice(0, 0, bs),
                                                    static_seq_lens_.slice(0, 0, bs),
                                                    false);
            }
#else
            // Fallback
            if (moe_model_fp8_) {
                hidden_states = moe_model_fp8_->forward(static_input_ids_.slice(0, 0, bs),
                                                        static_positions_.slice(0, 0, bs),
                                                        attn_ctx_->get_kv_cache(),
                                                        attn_ctx_->get_handler(),
                                                        static_slot_mapping_.slice(0, 0, bs),
                                                        static_block_tables_.slice(0, 0, bs),
                                                        static_seq_lens_.slice(0, 0, bs));
            }
            else {
                hidden_states = moe_model_->forward(static_input_ids_.slice(0, 0, bs),
                                                    static_positions_.slice(0, 0, bs),
                                                    attn_ctx_->get_kv_cache(),
                                                    attn_ctx_->get_handler(),
                                                    static_slot_mapping_.slice(0, 0, bs),
                                                    static_block_tables_.slice(0, 0, bs),
                                                    static_seq_lens_.slice(0, 0, bs));
            }
#endif
        }
        else {
            hidden_states = model_->forward(static_input_ids_.slice(0, 0, bs),
                                            static_positions_.slice(0, 0, bs),
                                            attn_ctx_->get_kv_cache(),
                                            attn_ctx_->get_handler(),
                                            static_slot_mapping_.slice(0, 0, bs),
                                            static_block_tables_.slice(0, 0, bs),
                                            static_seq_lens_.slice(0, 0, bs));
        }

        // 3. Logits only - Sampler is OUTSIDE capture for future flexibility
        torch::Tensor logits;
        if (config_->is_moe) {
            if (moe_model_fp8_)
                logits = moe_model_fp8_->compute_logits(hidden_states);
            else
                logits = moe_model_->compute_logits(hidden_states);
        }
        else
            logits = model_->compute_logits(hidden_states);

        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

        decode_graphs_[bs]        = instance;
        decode_graph_outputs_[bs] = logits;

        if (bs == 1)
            break;
        bs /= 2;
        if (bs < 1)
            bs = 1;
    }

    return true;
}

void ModelRunner::load_moe_layer_weights(int layer_idx, models::DeepSeekMoeDecoderLayer<QuantType::FP8_E4M3>* layer)
{
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";

    // A. Attention (FP8 Linear with concat)
    auto q_w = weight_manager_->load(prefix + "self_attn.q_proj.weight");
    auto q_s = weight_manager_->load(prefix + "self_attn.q_proj.weight_scale_inv").to(torch::kFloat32);
    auto k_w = weight_manager_->load(prefix + "self_attn.k_proj.weight");
    auto k_s = weight_manager_->load(prefix + "self_attn.k_proj.weight_scale_inv").to(torch::kFloat32);
    auto v_w = weight_manager_->load(prefix + "self_attn.v_proj.weight");
    auto v_s = weight_manager_->load(prefix + "self_attn.v_proj.weight_scale_inv").to(torch::kFloat32);

    // Concat FP8 weights
    layer->self_attn_->qkv_proj_->weight           = torch::cat({q_w, k_w, v_w}, 0);
    layer->self_attn_->qkv_proj_->weight_scale_inv = torch::cat({q_s, k_s, v_s}, 0);

    try {
        auto q_b                           = weight_manager_->load(prefix + "self_attn.q_proj.bias");
        auto k_b                           = weight_manager_->load(prefix + "self_attn.k_proj.bias");
        auto v_b                           = weight_manager_->load(prefix + "self_attn.v_proj.bias");
        layer->self_attn_->qkv_proj_->bias = torch::cat({q_b, k_b, v_b}, 0);
    }
    catch (...) {
        // Init bias to zeros if not found (BF16, on correct device)
        auto out_features = layer->self_attn_->qkv_proj_->weight.size(0);
        layer->self_attn_->qkv_proj_->bias =
            torch::zeros({out_features}, torch::TensorOptions().dtype(torch::kBFloat16).device(device_));
    }

    layer->self_attn_->o_proj_->weight = weight_manager_->load(prefix + "self_attn.o_proj.weight");
    layer->self_attn_->o_proj_->weight_scale_inv =
        weight_manager_->load(prefix + "self_attn.o_proj.weight_scale_inv").to(torch::kFloat32);

    layer->self_attn_->q_norm_->weight = weight_manager_->load(prefix + "self_attn.q_norm.weight").to(torch::kFloat32);
    layer->self_attn_->k_norm_->weight = weight_manager_->load(prefix + "self_attn.k_norm.weight").to(torch::kFloat32);

    {
        auto w = layer->self_attn_->q_norm_->weight;
        NANODEPLOY_LOG_INFO("[Weight Check] L",
                            layer_idx,
                            " Q_Norm Weight Mean: ",
                            w.mean().item<float>(),
                            " Max: ",
                            w.max().item<float>());
    }

    layer->input_layernorm_->weight = weight_manager_->load(prefix + "input_layernorm.weight").to(torch::kFloat32);
    layer->post_attention_layernorm_->weight =
        weight_manager_->load(prefix + "post_attention_layernorm.weight").to(torch::kFloat32);

    // B. MLP (MoE FP8)
    if (layer->is_sparse_) {
        // DeepSeek MoE Loading
        auto* block = layer->mlp_moe_.get();  // DeepSeekMoeSparseMoeBlock<FP8>

        // Gate (Router) - usually BF16/FP32
        block->gate_->weight = weight_manager_->load(prefix + "mlp.gate.weight").to(torch::kFloat32);
        ;

        // Experts weights
        int num_experts       = config_->num_experts;
        int ep_size           = get_dist_context().ffn_ep();
        int num_local_experts = num_experts / ep_size;

        // Load weights for each local expert
        std::vector<torch::Tensor> gate_up_list, down_list;
        std::vector<torch::Tensor> gate_up_scale_list, down_scale_list;

        for (int i = 0; i < num_local_experts; ++i) {
            // Calculate global expert index
            int         global_expert_idx = get_dist_context().ffn_ep_rank() * num_local_experts + i;
            std::string exp_prefix        = prefix + "mlp.experts." + std::to_string(global_expert_idx) + ".";

            auto g   = weight_manager_->load(exp_prefix + "gate_proj.weight");
            auto g_s = weight_manager_->load(exp_prefix + "gate_proj.weight_scale_inv").to(torch::kFloat32);
            auto u   = weight_manager_->load(exp_prefix + "up_proj.weight");
            auto u_s = weight_manager_->load(exp_prefix + "up_proj.weight_scale_inv").to(torch::kFloat32);
            auto d   = weight_manager_->load(exp_prefix + "down_proj.weight");
            auto d_s = weight_manager_->load(exp_prefix + "down_proj.weight_scale_inv").to(torch::kFloat32);

            // Concatenate gate and up
            gate_up_list.push_back(torch::cat({g, u}, 0));
            gate_up_scale_list.push_back(torch::cat({g_s, u_s}, 0));

            down_list.push_back(d);
            down_scale_list.push_back(d_s);
        }

        // Stack into tensors
        block->gate_up_proj_      = torch::stack(gate_up_list, 0);        // FP8
        block->gate_up_scale_inv_ = torch::stack(gate_up_scale_list, 0);  // FP32
        block->down_proj_         = torch::stack(down_list, 0);           // FP8
        block->down_scale_inv_    = torch::stack(down_scale_list, 0);     // FP32
    }
    else {
        // Dense MLP (FP8) - Assume Qwen3MoeDecoderLayer handles dense fallback via mlp_dense_
        // Qwen3MoeDecoderLayer has mlp_dense_ as MergedColumnParallelLinear
        // But MergedColumnParallelLinear<FP8> IS Linear<FP8> so it works!

        auto* mlp    = layer->mlp_dense_.get();
        auto  gate_w = weight_manager_->load(prefix + "mlp.gate_proj.weight");
        auto  gate_s = weight_manager_->load(prefix + "mlp.gate_proj.weight_scale_inv").to(torch::kFloat32);
        auto  up_w   = weight_manager_->load(prefix + "mlp.up_proj.weight");
        auto  up_s   = weight_manager_->load(prefix + "mlp.up_proj.weight_scale_inv").to(torch::kFloat32);

        mlp->gate_up_proj_->weight           = torch::cat({gate_w, up_w}, 0);
        mlp->gate_up_proj_->weight_scale_inv = torch::cat({gate_s, up_s}, 0);

        mlp->down_proj_->weight           = weight_manager_->load(prefix + "mlp.down_proj.weight");
        mlp->down_proj_->weight_scale_inv = weight_manager_->load(prefix + "mlp.down_proj.weight_scale_inv");
    }
}

}  // namespace nanodeploy
