#include "model_runner.h"
#include "distributed.h"

#include <algorithm>
#include <iostream>
#include <torch/torch.h>

#include "nanodeploy/logging.h"

#ifdef DEEPSEEK_MOE
#include "deep_ep.hpp"
#include "nanodeploy/worker/deep_ep_utils.h"
#include "nanodeploy/worker/deep_gemm_runner.h"
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
    init_internal(config_path_str, dconf.global_rank);
}

void ModelRunner::init_internal(const std::string& config_path_str, int rank)
{
    rank_       = rank;
    world_size_ = get_dist_context().world_size();

    torch::set_num_threads(1);

    std::filesystem::path config_path(config_path_str);
    auto                  model_dir = config_path.parent_path();

    nanodeploy::get_log_level() = 2;  // Set to DEBUG early

    NANODEPLOY_LOG_INFO("Loading config from ", config_path);
    config_ = std::make_unique<core::ModelConfig>(core::ModelConfig::load_hf(config_path.string()));

    if (config_->is_moe && get_dist_context().ep_world_size() == 1 && world_size_ > 1) {
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

    if (config_->is_moe) {
        NANODEPLOY_LOG_INFO("      MoE Model Detected.");
        NANODEPLOY_LOG_INFO("      Experts: ", config_->num_experts, " TopK: ", config_->num_experts_per_tok);

        moe_model_ = std::make_unique<Qwen3MoeForCausalLM<QuantType::FP16>>(*config_);

#ifdef DEEPSEEK_MOE
        // Initialize DeepGemm via runner
        DeepGemmRunner::init_utils();

        // Initialize DeepEP Buffer for DeepSeek MoE (only for multi-GPU)
        int ep_size = get_dist_context().ffn_ep_world_size();

        if (ep_size > 1) {
            // Multi-GPU: Initialize DeepEP for expert parallel communication
            int num_experts = config_->num_experts;
            int hidden_size = config_->hidden_size;

            // Setup NVSHMEM environment variables BEFORE creating buffer
            int num_local_experts = num_experts / ep_size;
            int num_qps_per_rank  = std::max(32, num_local_experts);
            setup_nvshmem_env(num_qps_per_rank, false);  // multi-card mode

            // Calculate buffer sizes using DeepEP Config for Normal Mode
            int num_sms = 0;
            cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_.index());
            if (num_sms <= 0)
                num_sms = 132;  // H100/H200 default

            // Dynamically adjust config based on hidden_size to avoid buffer overflow
            // Buffer size ~ num_channels * num_nvl_ranks * recv_tokens * hidden_bytes
            // For large models (hidden_size > 5000), use smaller token limits
            int nvl_send_tokens = 4096;
            int nvl_recv_tokens = 8192;
            if (hidden_size > 6000) {
                // 235B model: hidden=7168, need ~50% reduction
                nvl_send_tokens = 1024;
                nvl_recv_tokens = 2048;
            }
            else if (hidden_size > 4000) {
                // Medium-large models
                nvl_send_tokens = 2048;
                nvl_recv_tokens = 4096;
            }
            deep_ep::Config config(num_sms, nvl_send_tokens, nvl_recv_tokens, nvl_send_tokens, nvl_recv_tokens);

            int64_t hidden_size_bytes = hidden_size * 2;  // BF16
            int64_t num_nvl_bytes     = config.get_nvl_buffer_size_hint(hidden_size_bytes, ep_size);
            int64_t num_rdma_bytes    = config.get_rdma_buffer_size_hint(hidden_size_bytes, ep_size);

            NANODEPLOY_LOG_INFO("DeepEP Config: nvl_send=" + std::to_string(nvl_send_tokens)
                                + " nvl_recv=" + std::to_string(nvl_recv_tokens)
                                + " buffer_size=" + std::to_string(num_nvl_bytes / (1024 * 1024)) + "MB");

            ep_buffer_ = std::make_unique<deep_ep::Buffer>(rank_,
                                                           ep_size,
                                                           num_nvl_bytes,
                                                           num_rdma_bytes,
                                                           false,  // low_latency_mode = false (Normal Dispatch)
                                                           true,   // explicitly_destroy
                                                           true,   // enable_shrink
                                                           false   // use_fabric
            );

            NANODEPLOY_LOG_INFO("DeepEP Buffer initialized for DeepSeek MoE (ep_size=" + std::to_string(ep_size) + ")");
        }
        else {
            // Single-GPU: No DeepEP needed, use local computation
            // ep_buffer_ remains nullptr
            NANODEPLOY_LOG_INFO("DeepSeek MoE running in single-card mode (no DeepEP, local gather/scatter)");
        }
#else
        // DeepEP initialization removed for Simple MoE
#endif
    }
    else {
        model_ = std::make_unique<Qwen3ForCausalLM<QuantType::FP16>>(*config_, device_);
    }

    // KV Cache
    int head_dim   = config_->head_dim > 0 ? config_->head_dim : (config_->hidden_size / config_->num_attention_heads);
    int block_size = Sequence::block_size;
    kv_cache_      = std::make_unique<KvCache>(
        config_->num_hidden_layers, config_->num_key_value_heads, head_dim, 4096, block_size, device_);

    // FlashInfer
    flashinfer_handler_ = std::make_unique<layers::FlashInferHandler>(config_->num_hidden_layers,
                                                                      config_->num_attention_heads,
                                                                      config_->num_key_value_heads,
                                                                      head_dim,
                                                                      block_size,
                                                                      device_);

    load_weights("");
}

ModelRunResp ModelRunner::run(ModelRunReq req)
{
    NANODEPLOY_LOG_INFO(
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

        int seq_len = seq->num_tokens;
        seq_lens_vec.push_back(seq_len);

        if (req.is_prefill) {
            for (int id : seq->token_ids)
                input_ids_vec.push_back(id);
            for (int i = 0; i < seq->token_ids.size(); ++i)
                positions_vec.push_back(i);
            for (int i = 0; i < seq->token_ids.size(); ++i) {
                if (blocks.empty()) {
                    slot_mapping_vec.push_back(0);
                    continue;
                }
                int block_idx = blocks[i / block_size];
                int offset    = i % block_size;
                slot_mapping_vec.push_back(block_idx * block_size + offset);
            }
        }
        else {
            if (seq->token_ids.empty())
                continue;
            input_ids_vec.push_back(seq->token_ids.back());
            int pos = seq_len - 1;
            positions_vec.push_back(pos);
            if (blocks.empty())
                slot_mapping_vec.push_back(0);
            else {
                int block_in_seq = pos / block_size;
                if (block_in_seq >= (int)blocks.size())
                    block_in_seq = blocks.size() - 1;
                int block_idx = blocks[block_in_seq];
                int offset    = pos % block_size;
                slot_mapping_vec.push_back(block_idx * block_size + offset);
            }
        }
    }

    auto input_ids = torch::from_blob(input_ids_vec.data(), {(long)input_ids_vec.size()}, torch::kLong).to(device_);
    auto positions =
        torch::from_blob(positions_vec.data(), {(long)positions_vec.size()}, torch::kInt).to(device_, torch::kLong);
    auto slot_mapping =
        torch::from_blob(slot_mapping_vec.data(), {(long)slot_mapping_vec.size()}, torch::kInt).to(device_);
    auto block_tables =
        torch::from_blob(block_tables_vec.data(), {batch_size, max_num_blocks}, torch::kInt).to(device_);
    auto seq_lens = torch::from_blob(seq_lens_vec.data(), {batch_size}, torch::kInt).to(device_);

    if (!req.is_prefill) {
        flashinfer_handler_->begin_forward(block_tables_vec.data(),
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
        hidden_states = moe_model_->forward(input_ids,
                                            positions,
                                            kv_cache_.get(),
                                            flashinfer_handler_.get(),
                                            slot_mapping,
                                            block_tables,
                                            seq_lens,
                                            ep_buffer_.get());
#else
        // Simple MoE forward signature doesn't take ep_buffer
        hidden_states = moe_model_->forward(
            input_ids, positions, kv_cache_.get(), flashinfer_handler_.get(), slot_mapping, block_tables, seq_lens);
#endif
    }
    else {
        hidden_states = model_->forward(
            input_ids, positions, kv_cache_.get(), flashinfer_handler_.get(), slot_mapping, block_tables, seq_lens);
    }

    // Compute Logits
    torch::Tensor logits;
    if (req.is_prefill) {
        std::vector<int64_t> last_token_indices;
        int64_t              current_offset = 0;
        for (const auto& seq : req.seqs) {
            last_token_indices.push_back(current_offset + (int)seq->token_ids.size() - 1);
            current_offset += seq->token_ids.size();
        }
        auto indices =
            torch::from_blob(last_token_indices.data(), {(long)last_token_indices.size()}, torch::kLong).to(device_);

        if (hidden_states.dim() == 3)
            hidden_states = hidden_states.reshape({-1, (long)config_->hidden_size});

        auto last_hidden = hidden_states.index_select(0, indices);

        if (config_->is_moe)
            logits = moe_model_->compute_logits(last_hidden);
        else
            logits = model_->compute_logits(last_hidden);
    }
    else {
        if (config_->is_moe)
            logits = moe_model_->compute_logits(hidden_states);
        else
            logits = model_->compute_logits(hidden_states);
    }

    return ModelRunResp{logits};
}

void ModelRunner::load_weights(const std::string& /*weight_path*/)
{
    NANODEPLOY_LOG_INFO("Loading Embeddings...");
    if (config_->is_moe) {
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
            load_moe_layer_weights(i, moe_model_->model_->layers_[i].get());
        }
        else {
            load_layer_weights(i, model_->model_->layers_[i].get());
        }
    }

    NANODEPLOY_LOG_INFO("Loading Final Norm...");
    if (config_->is_moe) {
        moe_model_->model_->norm_->weight = weight_manager_->load("model.norm.weight");
        moe_model_->lm_head_->weight      = weight_manager_->load("lm_head.weight");
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

    layer->input_layernorm_->weight          = weight_manager_->load(prefix + "input_layernorm.weight");
    layer->post_attention_layernorm_->weight = weight_manager_->load(prefix + "post_attention_layernorm.weight");

    // B. MLP (Sparse or Dense)
    if (layer->is_sparse_) {
        // DeepSeek MoE Loading
        auto* block = layer->mlp_moe_.get();  // DeepSeekMoeSparseMoeBlock

        // Gate
        block->gate_->weight = weight_manager_->load(prefix + "mlp.gate.weight");

        // Experts weights (tensor format: [num_local_experts, ...])
        int num_experts       = config_->num_experts;
        int ep_size           = get_dist_context().ffn_ep_world_size();
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
    DeepEpInfoResp resp;
    if (ep_buffer_) {
        resp.device_id      = ep_buffer_->get_local_device_id();
        resp.num_rdma_ranks = ep_buffer_->get_num_rdma_ranks();
        resp.rdma_rank      = ep_buffer_->get_rdma_rank();
        resp.root_rdma_rank = ep_buffer_->get_root_rdma_rank(true);

        // Get IPC handle as binary string
        auto ipc_handle_py = ep_buffer_->get_local_ipc_handle();
        resp.ipc_handle    = ipc_handle_py.cast<std::string>();

        // Get NVSHMEM unique ID if this is the root rank
        if (resp.rdma_rank == resp.root_rdma_rank && resp.num_rdma_ranks > 1) {
            auto nvshmem_id_py     = ep_buffer_->get_local_nvshmem_unique_id();
            resp.nvshmem_unique_id = nvshmem_id_py.cast<std::string>();
        }

        NANODEPLOY_LOG_INFO("DeepEP Info: rank=",
                            rank_,
                            " device_id=",
                            resp.device_id,
                            " num_rdma_ranks=",
                            resp.num_rdma_ranks,
                            " ipc_handle_size=",
                            resp.ipc_handle.size());
    }
    return resp;
}

bool ModelRunner::syncDeepEp(DeepEpSyncReq req)
{
    if (!ep_buffer_) {
        NANODEPLOY_LOG_WARN("No DeepEP buffer to sync");
        return false;
    }

    NANODEPLOY_LOG_INFO("Syncing DeepEP: rank=", rank_, " num_devices=", req.device_ids.size());

    // Debug: print received handle sizes
    for (size_t i = 0; i < req.ipc_handles.size(); ++i) {
        NANODEPLOY_LOG_INFO("  Received handle[", i, "] size=", req.ipc_handles[i].size());
    }

    // Convert to pybind11 types for sync call
    std::vector<std::optional<pybind11::bytearray>> all_handles;
    all_handles.reserve(req.ipc_handles.size());
    for (const auto& h : req.ipc_handles) {
        // Use data() and size() explicitly for binary data
        all_handles.push_back(pybind11::bytearray(h.data(), h.size()));
    }

    std::optional<pybind11::bytearray> root_unique_id;
    if (!req.root_nvshmem_unique_id.empty()) {
        root_unique_id = pybind11::bytearray(req.root_nvshmem_unique_id);
    }

    try {
        ep_buffer_->sync(req.device_ids, all_handles, root_unique_id);
        NANODEPLOY_LOG_INFO("DeepEP sync completed for rank ", rank_);
        return true;
    }
    catch (const std::exception& e) {
        NANODEPLOY_LOG_ERROR("DeepEP sync failed: ", e.what());
        return false;
    }
    return true;
}

}  // namespace nanodeploy
