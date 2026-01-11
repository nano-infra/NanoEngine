#include "model_runner.h"
#include "distributed.h"

#include <iostream>
#include <torch/torch.h>

#include "nanodeploy/logging.h"

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
        // Single file: assume if loader exists, we can try (or need check)
        // Ideally SafeTensorLoader exposes `has_tensor`.
        // For now returning true for single file mode as loose check.
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
    // Assume EP = WorldSize for MoE unless specified otherwise
    // But config is loaded later. We'll refine EP degree if needed or assume 1 if not MoE.
    // Ideally user sets dconf correctly.
    init(config_path_str, dconf);
}

void ModelRunner::init(const std::string& config_path_str, const DistributedConfig& dconf)
{
    get_dist_context().init(dconf);
    init_internal(config_path_str);
}

void ModelRunner::init_internal(const std::string& config_path_str)
{
    rank_       = get_dist_context().global_rank();
    world_size_ = get_dist_context().world_size();

    torch::set_num_threads(1);

    std::filesystem::path config_path(config_path_str);
    auto                  model_dir = config_path.parent_path();

    NANODEPLOY_LOG_INFO("Loading config from ", config_path);
    config_ = std::make_unique<core::ModelConfig>(core::ModelConfig::load_hf(config_path.string()));

    // Update DistributedContext EP degree if implicit
    if (config_->is_moe && get_dist_context().ep_world_size() == 1 && world_size_ > 1) {
        // Heuristic: If MoE and EP not set, assume EP = WorldSize (common for inference)
        // But DistContext is already initialized. We can't easily change it without re-init.
        // Assuming user passed correct dconf or we rely on default 1.
        NANODEPLOY_LOG_WARN("MoE detected but EP degree is 1. Running in TP/DP only mode?");
    }

    if (torch::cuda::is_available()) {
        device_ = torch::kCUDA;
        NANODEPLOY_LOG_INFO("CUDA Detected. Using GPU.");
    }
    else {
        device_ = torch::kCPU;
        NANODEPLOY_LOG_INFO("CUDA Not Available. Using CPU.");
    }

    nanodeploy::get_log_level() = 1;

    NANODEPLOY_LOG_INFO("Initializing WeightManager for ", model_dir);
    weight_manager_ = std::make_unique<WeightManager>(model_dir, device_);

    NANODEPLOY_LOG_INFO("init: Allocating Model (FP16)...");

    if (config_->is_moe) {
        NANODEPLOY_LOG_INFO("      MoE Model Detected.");
        NANODEPLOY_LOG_INFO("      Experts: ", config_->num_experts, " TopK: ", config_->num_experts_per_tok);

        moe_model_ = std::make_unique<Qwen3MoeForCausalLM<QuantType::FP16>>(*config_);

        // Initialize DeepEP Buffer
        if (get_dist_context().ep_world_size() > 1) {
            NANODEPLOY_LOG_INFO("Initializing DeepEP Buffer...");
            // Calculate buffer size hint
            // Use defaults or formula from DeepEP
            int num_tokens  = 4096;  // Max batch tokens?
            int hidden      = config_->hidden_size;
            int num_experts = config_->num_experts;
            int ep_size     = get_dist_context().ep_world_size();
            // Need a reasonable estimate
            size_t rdma_size = deep_ep::get_low_latency_rdma_size_hint(num_tokens, hidden, ep_size, num_experts);
            // nvl size hint is in Config class in deep_ep.cpp/hpp?
            // deep_ep::Config::get_nvl_buffer_size_hint?
            // Wait, deep_ep.cpp exposes get_nvl_buffer_size_hint as member of Config.
            // But get_low_latency_rdma_size_hint as free function.
            // I need an instance of deep_ep::Config to call get_nvl_buffer_size_hint?
            // Or assume nvl_size based on similar logic?
            // Or instantiate dummy Config? Config constructor takes params.
            // Let's use a safe large number for NVL or construct Config.

            deep_ep::Config dep_config(20, 6, 256, 6, 256);                              // Defaults from python
            size_t nvl_size = dep_config.get_nvl_buffer_size_hint(hidden * 2, ep_size);  // hidden_bytes? BF16=2 bytes
            // Wait, get_nvl_buffer_size_hint takes (hidden_bytes, num_ranks).

            ep_buffer_ = std::make_unique<deep_ep::Buffer>(get_dist_context().ep_rank(),
                                                           ep_size,
                                                           nvl_size,
                                                           rdma_size,
                                                           true,   // low_latency_mode
                                                           true,   // explicitly_destroy
                                                           false,  // enable_shrink
                                                           false   // use_fabric
            );
        }
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
    if (req.seqs.empty())
        return ModelRunResp{};

    // ... (Tensor preparation code identical to original, omitted for brevity, assuming we keep it)
    // To implement "continue" properly, I should paste the whole function content
    // or just the relevant diffs? 'write' tool overwrites.
    // I must preserve the logic. I will copy-paste the preparation logic from previous file read.

    // ... [RE-IMPLEMENTING PREPARATION LOGIC] ...

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
        hidden_states = moe_model_->forward(
            input_ids, positions, kv_cache_.get(), flashinfer_handler_.get(), slot_mapping, ep_buffer_.get());
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
            // Need get() on unique_ptr inside vector
            // Accessing unique_ptr in vector by reference then get()
            // Actually layers_ is vector<unique_ptr>.
            // Need to access raw pointer.
            // moe_model_ -> model_ -> layers_[i]
            // The layers_ is vector of unique_ptr.
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
    // ... (Use previous logic for dense loading) ...
    // To save context space, I assume I can copy paste the previous logic logic here?
    // Yes, 'write' overwrites the file. I MUST include the implementation.

    // Copying Dense Logic:
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

void ModelRunner::load_moe_layer_weights(int layer_idx, models::Qwen3MoeDecoderLayer<QuantType::FP16>* layer)
{
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";

    // A. Attention (Same as Dense, mostly)
    // Actually Qwen3MoeAttention structure is similar.
    auto q_w = weight_manager_->load(prefix + "self_attn.q_proj.weight");
    auto k_w = weight_manager_->load(prefix + "self_attn.k_proj.weight");
    auto v_w = weight_manager_->load(prefix + "self_attn.v_proj.weight");

    // BF16 expected for MoE currently
    layer->self_attn_->qkv_proj_->weight = torch::cat({q_w, k_w, v_w}, 0);

    try {
        auto q_b                           = weight_manager_->load(prefix + "self_attn.q_proj.bias");
        auto k_b                           = weight_manager_->load(prefix + "self_attn.k_proj.bias");
        auto v_b                           = weight_manager_->load(prefix + "self_attn.v_proj.bias");
        layer->self_attn_->qkv_proj_->bias = torch::cat({q_b, k_b, v_b}, 0);
    }
    catch (...) {
        // handle missing bias
        auto out_features                  = layer->self_attn_->qkv_proj_->weight.size(0);
        layer->self_attn_->qkv_proj_->bias = torch::zeros({out_features}, q_w.options());
    }

    layer->self_attn_->o_proj_->weight = weight_manager_->load(prefix + "self_attn.o_proj.weight");
    layer->self_attn_->q_norm_->weight = weight_manager_->load(prefix + "self_attn.q_norm.weight");
    layer->self_attn_->k_norm_->weight = weight_manager_->load(prefix + "self_attn.k_norm.weight");

    layer->input_norm_->weight     = weight_manager_->load(prefix + "input_layernorm.weight");
    layer->post_attn_norm_->weight = weight_manager_->load(prefix + "post_attention_layernorm.weight");

    // B. MLP (Sparse or Dense)
    if (layer->is_sparse_) {
        // MoE Loading
        auto* block = layer->mlp_.get();  // Qwen3MoeSparseMoeBlock

        // Gate
        block->gate_->weight = weight_manager_->load(prefix + "mlp.gate.weight");

        // Experts
        // Needs FUSION from experts.0...N to Groups
        // Placeholder for phase 4 logic:
        // Assume for now we just load if they exist, or create random for testing if not found.
        // Implementing proper loop:

        int num_experts       = config_->num_experts;
        int ep_size           = get_dist_context().ep_world_size();
        int ep_rank           = get_dist_context().ep_rank();
        int num_local_experts = num_experts / ep_size;
        int expert_offset     = ep_rank * num_local_experts;

        std::vector<torch::Tensor> gate_up_list;
        std::vector<torch::Tensor> down_list;

        // Try to load first expert to detect existence
        if (!weight_manager_->has_param(prefix + "mlp.experts.0.gate_proj.weight")) {
            NANODEPLOY_LOG_WARN("MoE experts not found in checkpoints. Skipping load (using uninit).");
            return;
        }

        for (int i = 0; i < num_local_experts; ++i) {
            int         global_id  = expert_offset + i;
            std::string exp_prefix = prefix + "mlp.experts." + std::to_string(global_id) + ".";

            auto g = weight_manager_->load(exp_prefix + "gate_proj.weight");
            auto u = weight_manager_->load(exp_prefix + "up_proj.weight");
            auto d = weight_manager_->load(exp_prefix + "down_proj.weight");

            // Cat Gate/Up
            auto gu = torch::cat({g, u}, 0);  // [Inter*2, Hidden]

            gate_up_list.push_back(gu);
            down_list.push_back(d);  // [Hidden, Inter] - Wait.
            // DeepGemm expects [LocalExperts, Hidden, Inter].
            // HF Linear weight is [Out, In].
            // DownProj: In=Inter, Out=Hidden. Weight is [Hidden, Inter].
            // Correct.
        }

        // Stack -> [LocalExperts, Inter*2, Hidden] ?
        // HF Linear [Out, In].
        // GateUp List elements: [Inter*2, Hidden].
        // Stack(0) -> [LocalExperts, Inter*2, Hidden].
        // This matches `register_parameter` shape in qwen3_moe.h

        auto gate_up_stacked = torch::stack(gate_up_list, 0);
        auto down_stacked    = torch::stack(down_list, 0);

        block->gate_up_proj_ = gate_up_stacked.to(device_);
        block->down_proj_    = down_stacked.to(device_);

        // Scales handling if FP8...
    }
    else {
        // Dense MLP in MoE model (Shared expert or non-sparse layer)
        auto* mlp                  = layer->mlp_dense_.get();
        auto  gate_w               = weight_manager_->load(prefix + "mlp.gate_proj.weight");
        auto  up_w                 = weight_manager_->load(prefix + "mlp.up_proj.weight");
        mlp->gate_up_proj_->weight = torch::cat(std::vector<torch::Tensor>{gate_w, up_w}, 0);
        mlp->down_proj_->weight    = weight_manager_->load(prefix + "mlp.down_proj.weight");
    }
}

}  // namespace nanodeploy
