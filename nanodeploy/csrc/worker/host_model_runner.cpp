#include "host_model_runner.h"
#include "nanodeploy/csrc/context/distributed_context.h"
#include "nanodeploy/csrc/logging.h"

#include <algorithm>
#include <iostream>
#include <torch/torch.h>

namespace nanodeploy {

using namespace models;

// =========================================================================
// HostWeightManager Implementation
// =========================================================================

HostWeightManager::HostWeightManager(const std::filesystem::path& model_dir, torch::Device device):
    model_dir_(model_dir), device_(device)
{
    if (!std::filesystem::exists(model_dir_)) {
        throw std::runtime_error("Model directory does not exist: " + model_dir_.string());
    }

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
    }
}

bool HostWeightManager::has_param(const std::string& param_name)
{
    if (is_sharded_) {
        return param_to_file_.find(param_name) != param_to_file_.end();
    }
    else {
        return !loaders_.empty();
    }
}

torch::Tensor HostWeightManager::load(const std::string& param_name)
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
        loaders_.emplace(filename, SafeTensorLoader(path.string()));
    }

    return loaders_.at(filename).load(param_name, device_);
}

// =========================================================================
// HostModelRunner Implementation
// =========================================================================

void HostModelRunner::init(const std::string& config_path_str, int rank, int world_size)
{
    DistributedConfig dconf;
    dconf.global_rank = rank;
    dconf.world_size  = world_size;
    init(config_path_str, dconf);
}

void HostModelRunner::init(const std::string& config_path_str, const DistributedConfig& dconf)
{
    get_dist_context().init(dconf);
    init_internal(config_path_str, dconf.global_rank);
}

void HostModelRunner::init_internal(const std::string& config_path_str, int rank)
{
    rank_       = rank;
    world_size_ = get_dist_context().world_size();

    std::filesystem::path config_path(config_path_str);
    auto                  model_dir = config_path.parent_path();

    nanodeploy::get_log_level() = 2;

    NANODEPLOY_LOG_INFO("[HostRunner] Loading config from ", config_path);
    config_ = std::make_unique<core::ModelConfig>(core::ModelConfig::load_hf(config_path.string()));

    device_ = torch::kCPU;
    NANODEPLOY_LOG_INFO("[HostRunner] Using CPU Device.");

    NANODEPLOY_LOG_INFO("[HostRunner] Initializing WeightManager for ", model_dir);
    weight_manager_ = std::make_unique<HostWeightManager>(model_dir, device_);

    NANODEPLOY_LOG_INFO("[HostRunner] Allocating Host Model...");
    model_ = std::make_unique<Qwen3HostForCausalLM<QuantType::FP16>>(*config_, device_);

    // KV Cache
    int head_dim   = config_->head_dim > 0 ? config_->head_dim : (config_->hidden_size / config_->num_attention_heads);
    int block_size = Sequence::block_size;
    kv_cache_      = std::make_unique<KvCache>(
        config_->num_hidden_layers, config_->num_key_value_heads, head_dim, 4096, block_size, device_);

    load_weights("");
}

ModelRunResp HostModelRunner::run(ModelRunReq req)
{
    if (req.seqs.empty()) {
        return ModelRunResp{};
    }

    // Input Preparation
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
            if (blocks.empty()) {
                slot_mapping_vec.push_back(0);
            }
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

    // Forward
    auto hidden_states = model_->forward(input_ids, positions, kv_cache_.get(), slot_mapping, block_tables, seq_lens);

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
        logits           = model_->compute_logits(last_hidden);
    }
    else {
        logits = model_->compute_logits(hidden_states);
    }

    return ModelRunResp{logits};
}

void HostModelRunner::load_weights(const std::string&)
{
    NANODEPLOY_LOG_INFO("[HostRunner] Loading Weights...");

    model_->model_->embed_tokens_->weight = weight_manager_->load("model.embed_tokens.weight");

    int total_layers = config_->num_hidden_layers;
    for (int i = 0; i < total_layers; ++i) {
        if (i % 1 == 0)
            NANODEPLOY_LOG_INFO("Loading Layer ", i, " / ", total_layers);
        load_layer_weights(i, model_->model_->layers_[i].get());
    }

    NANODEPLOY_LOG_INFO("[HostRunner] Loading Final Norm...");
    model_->model_->norm_->weight = weight_manager_->load("model.norm.weight");
    model_->lm_head_->weight      = weight_manager_->load("lm_head.weight");
}

void HostModelRunner::load_layer_weights(int layer_idx, Qwen3HostDecoderLayer<QuantType::FP16>* layer)
{
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";

    auto q_w = weight_manager_->load(prefix + "self_attn.q_proj.weight");
    auto k_w = weight_manager_->load(prefix + "self_attn.k_proj.weight");
    auto v_w = weight_manager_->load(prefix + "self_attn.v_proj.weight");

    // Ensure FP16 or Float
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

    layer->self_attn_->o_proj_->weight       = weight_manager_->load(prefix + "self_attn.o_proj.weight");
    layer->input_layernorm_->weight          = weight_manager_->load(prefix + "input_layernorm.weight");
    layer->post_attention_layernorm_->weight = weight_manager_->load(prefix + "post_attention_layernorm.weight");

    auto gate_w                        = weight_manager_->load(prefix + "mlp.gate_proj.weight");
    auto up_w                          = weight_manager_->load(prefix + "mlp.up_proj.weight");
    layer->mlp_->gate_up_proj_->weight = torch::cat(std::vector<torch::Tensor>{gate_w, up_w}, 0);
    layer->mlp_->down_proj_->weight    = weight_manager_->load(prefix + "mlp.down_proj.weight");

    try {
        layer->self_attn_->q_norm_->weight = weight_manager_->load(prefix + "self_attn.q_norm.weight");
        layer->self_attn_->k_norm_->weight = weight_manager_->load(prefix + "self_attn.k_norm.weight");
    }
    catch (...) {
    }
}

}  // namespace nanodeploy
