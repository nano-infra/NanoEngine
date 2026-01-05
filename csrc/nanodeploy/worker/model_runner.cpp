#include "model_runner.h"

#include <iostream>
#include <torch/torch.h>

#include "nanodeploy/logging.h"

namespace nanodeploy {

using namespace models;

// =========================================================================
// WeightManager Implementation
// =========================================================================

WeightManager::WeightManager(const std::filesystem::path& model_dir): model_dir_(model_dir)
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
        // Assume single file or iterate
        // For simplicity, look for model.safetensors
        auto single_path = model_dir_ / "model.safetensors";
        if (std::filesystem::exists(single_path)) {
            // If not sharded, we don't have a map.
            // We'll rely on SafeTensorLoader to find the key in the single file.
            // We store a default loader key?
            // Actually, simplest strategy: Just open the single file immediately.
            loaders_.emplace("model.safetensors", SafeTensorLoader(single_path.string()));
        }
        else {
            // Check for *.safetensors?
            // Fallback: Try loading weight_map.json if it exists (legacy path compatibility?)
            // Or just error out for now.
            std::cerr << "[WeightManager] Warning: No index.json or model.safetensors found in " << model_dir_
                      << std::endl;
        }
    }
}

torch::Tensor WeightManager::load(const std::string& param_name)
{
    // std::cout << "  [WeightManager] Request load: " << param_name << std::endl;
    std::string filename;

    if (is_sharded_) {
        // 1. Resolve filename from index
        auto it = param_to_file_.find(param_name);
        if (it == param_to_file_.end()) {
            throw std::runtime_error("Parameter not found in index: " + param_name);
        }
        filename = it->second;
    }
    else {
        // Single file mode
        filename = "model.safetensors";
    }

    std::cout << "  [WeightManager] File: " << filename << std::endl;

    // 2. Get or Open Loader
    if (loaders_.find(filename) == loaders_.end()) {
        auto path = model_dir_ / filename;
        std::cout << "  [WeightManager] Opening new loader for: " << path << std::endl;
        loaders_.emplace(filename, SafeTensorLoader(path.string()));
        std::cout << "  [WeightManager] Loader opened." << std::endl;
    }

    // 3. Load
    std::cout << "  [WeightManager] Loading tensor data..." << std::endl;
    auto t = loaders_.at(filename).load(param_name);
    std::cout << "  [WeightManager] Tensor loaded." << std::endl;
    return t;
}

// =========================================================================
// ModelRunner Implementation
// =========================================================================

void ModelRunner::init(const std::string& config_path_str, int rank, int world_size)
{
    rank_       = rank;
    world_size_ = world_size;

    std::filesystem::path config_path(config_path_str);
    auto                  model_dir = config_path.parent_path();

    // If user passed a directory as config_path (which they might), handle it?
    // User request: "only load config.json". Implies path to config.json.
    // So parent_path() gives the dir.

    std::cout << "[ModelRunner] Loading config from " << config_path << std::endl;
    config_ = std::make_unique<core::ModelConfig>(core::ModelConfig::load_hf(config_path.string()));

    std::cout << "[ModelRunner] Initializing WeightManager for " << model_dir << std::endl;
    weight_manager_ = std::make_unique<WeightManager>(model_dir);

    std::cout << "[ModelRunner] Allocating Model (FP16)..." << std::endl;
    // Initialize model (FP16 default)
    model_ = std::make_unique<Qwen3MoeForCausalLM<QuantType::FP16>>(*config_);

    std::cout << "[ModelRunner] Allocating KV Cache..." << std::endl;
    // Initialize KV Cache (Placeholder sizing)
    int head_dim = config_->hidden_size / config_->num_attention_heads;
    kv_cache_    = std::make_unique<KvCache>(
        config_->num_hidden_layers, config_->num_attention_heads, head_dim, 256, config_->max_position_embeddings);

    // Load Weights
    std::cout << "[ModelRunner] Starting Weight Loading..." << std::endl;
    load_weights("");  // Path unused now
}

ModelRunResp ModelRunner::run(ModelRunReq req)
{
    // ... (keep existing run logic) ...
    // Simplified run logic:
    // ...
    if (req.seqs.empty()) {
        return {torch::empty({0})};
    }
    std::cout << "[ModelRunner] Processing request with " << req.seqs.size() << " sequences." << std::endl;

    // Dummy forward for verification
    int64_t total_len = 0;
    for (const auto& seq : req.seqs) {
        if (req.is_prefill) {
            total_len += seq->token_ids.size();
        }
        else {
            total_len += 1;
        }
    }

    // Create dummy input
    std::vector<int64_t> input_shape = {1, total_len};
    auto                 input_ids   = torch::randint(0, config_->vocab_size, input_shape, torch::kLong);
    auto                 positions   = torch::arange(0, total_len, torch::kLong).unsqueeze(0);

    // Call Model
    auto hidden = model_->forward(input_ids, positions);

    // Logits
    auto logits = model_->compute_logits(hidden);

    // Return last token logits for verification
    return ModelRunResp{logits.index({torch::indexing::Slice(), -1, torch::indexing::Slice()}).clone()};
}

void ModelRunner::load_weights(const std::string& /*weight_path*/)
{
    // 1. Embeddings
    std::cout << "[ModelRunner] Loading Embeddings..." << std::endl;
    model_->model_->embed_tokens_->weight = weight_manager_->load("model.embed_tokens.weight");

    // 2. Layers
    int total_layers = config_->num_hidden_layers;
    for (int i = 0; i < total_layers; ++i) {
        if (i % 1 == 0) {  // Log every layer for now to be verbose as requested
            std::cout << "[ModelRunner] Loading Layer " << i << " / " << total_layers << "\r" << std::flush;
        }
        load_layer_weights(i, model_->model_->layers_[i].get());
    }
    std::cout << std::endl;  // Newline after loop

    // 3. Final Norm
    std::cout << "[ModelRunner] Loading Final Norm..." << std::endl;
    model_->model_->norm_->weight = weight_manager_->load("model.norm.weight");

    // 4. LM Head
    std::cout << "[ModelRunner] Loading LM Head..." << std::endl;
    model_->lm_head_->weight = weight_manager_->load("lm_head.weight");

    std::cout << "[ModelRunner] Weights loaded successfully." << std::endl;
}

void ModelRunner::load_layer_weights(int layer_idx, Qwen3MoeDecoderLayer<QuantType::FP16>* layer)
{
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";

    // A. Attention
    auto& attn = layer->self_attn_;

    // QKV Projection
    auto q_w = weight_manager_->load(prefix + "self_attn.q_proj.weight");
    auto k_w = weight_manager_->load(prefix + "self_attn.k_proj.weight");
    auto v_w = weight_manager_->load(prefix + "self_attn.v_proj.weight");

    // Fix: Cast FP8 to FP16 before cat if needed.
    // torch::cat for FP8 on CPU might be problematic or not implemented efficiently.
    if (q_w.scalar_type() == torch::kFloat8_e4m3fn) {
        std::cout << "  [ModelRunner] Converting FP8 weights to FP16 for concat..." << std::endl;
        // Optimization: Ensure contiguous memory on CPU before casting,
        // as casting from mmap view might trigger bad kernels.
        q_w = q_w.clone().to(torch::kFloat16);
        k_w = k_w.clone().to(torch::kFloat16);
        v_w = v_w.clone().to(torch::kFloat16);
    }

    std::cout << "  [ModelRunner] Concatenating QKV..." << std::endl;
    attn->qkv_proj_->weight = torch::cat({q_w, k_w, v_w}, 0);
    std::cout << "  [ModelRunner] QKV concatenated." << std::endl;

    // Bias (Optional)
    try {
        auto q_b              = weight_manager_->load(prefix + "self_attn.q_proj.bias");
        auto k_b              = weight_manager_->load(prefix + "self_attn.k_proj.bias");
        auto v_b              = weight_manager_->load(prefix + "self_attn.v_proj.bias");
        attn->qkv_proj_->bias = torch::cat({q_b, k_b, v_b}, 0);
    }
    catch (...) {
        // Ignore missing bias
    }

    // O Projection
    attn->o_proj_->weight = weight_manager_->load(prefix + "self_attn.o_proj.weight");

    // Norms
    layer->input_layernorm_->weight          = weight_manager_->load(prefix + "input_layernorm.weight");
    layer->post_attention_layernorm_->weight = weight_manager_->load(prefix + "post_attention_layernorm.weight");

    // B. MLP / MoE
    bool is_moe = (config_->num_experts > 0 && (layer_idx + 1) % config_->decoder_sparse_step == 0);
    if (!is_moe) {
        // Dense MLP
        auto mlp = dynamic_cast<Qwen3MoeMLP<QuantType::FP16>*>(layer->mlp_.get());
        if (mlp) {
            auto gate_w                = weight_manager_->load(prefix + "mlp.gate_proj.weight");
            auto up_w                  = weight_manager_->load(prefix + "mlp.up_proj.weight");
            mlp->gate_up_proj_->weight = torch::cat({gate_w, up_w}, 0);

            mlp->down_proj_->weight = weight_manager_->load(prefix + "mlp.down_proj.weight");
        }
    }
    else {
        // Sparse MoE Logic (Placeholder)
    }
}

}  // namespace nanodeploy
