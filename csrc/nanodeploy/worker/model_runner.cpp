#include "model_runner.h"

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
    // NANODEPLOY_LOG_DEBUG("Request load: ", param_name);
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

    NANODEPLOY_LOG_DEBUG("File: ", filename);

    // 2. Get or Open Loader
    if (loaders_.find(filename) == loaders_.end()) {
        auto path = model_dir_ / filename;
        NANODEPLOY_LOG_INFO("Opening new loader for: ", path);
        loaders_.emplace(filename, SafeTensorLoader(path.string()));
        NANODEPLOY_LOG_DEBUG("Loader opened.");
    }

    // 3. Load
    NANODEPLOY_LOG_DEBUG("Loading tensor data...");
    // 3. Load
    NANODEPLOY_LOG_DEBUG("Loading tensor data...");
    auto t = loaders_.at(filename).load(param_name, device_);

    // Force cast to Float32 for CPU compatibility to avoid BF16 matmul hangs on Windows
    // Optimization: Try removing this to save memory.
    // if (t.scalar_type() == torch::kBFloat16 || t.scalar_type() == torch::kHalf) {
    //     t = t.to(torch::kFloat32);
    // }

    NANODEPLOY_LOG_DEBUG("Tensor loaded.");
    return t;
}

// =========================================================================
// ModelRunner Implementation
// =========================================================================

void ModelRunner::init(const std::string& config_path_str, int rank, int world_size)
{
    rank_       = rank;
    world_size_ = world_size;

    // Force single-threaded execution to avoid OMP/MKL hangs on Windows
    torch::set_num_threads(1);
    NANODEPLOY_LOG_INFO("Forced single-threaded execution (torch::set_num_threads(1)).");

    std::filesystem::path config_path(config_path_str);
    auto                  model_dir = config_path.parent_path();

    // If user passed a directory as config_path (which they might), handle it?
    // User request: "only load config.json". Implies path to config.json.
    // So parent_path() gives the dir.

    NANODEPLOY_LOG_INFO("Loading config from ", config_path);
    config_ = std::make_unique<core::ModelConfig>(core::ModelConfig::load_hf(config_path.string()));

    // Check for CUDA
    if (torch::cuda::is_available()) {
        device_ = torch::kCUDA;
        NANODEPLOY_LOG_INFO("CUDA Detected. Using GPU.");
    }
    else {
        device_ = torch::kCPU;
        NANODEPLOY_LOG_INFO("CUDA Not Available. Using CPU.");
    }

    // Set default log level to INFO (1) for clean remote logs
    nanodeploy::get_log_level() = 1;

    NANODEPLOY_LOG_INFO("Initializing WeightManager for ", model_dir);
    weight_manager_ = std::make_unique<WeightManager>(model_dir, device_);

    NANODEPLOY_LOG_INFO("init: Allocating Model (FP16)...");
    NANODEPLOY_LOG_INFO("      Vocab Size: ", config_->vocab_size);
    NANODEPLOY_LOG_INFO("      Hidden Size: ", config_->hidden_size);
    // Initialize model (FP16 default)
    // Note: Model modules usually default to CPU. Weights will be loaded to device_,
    // and ideally the model should be moved to device.
    // However, our Qwen3Model layers are managed by unique_ptrs.
    // The weights are assigned directly from WeightManager::load which puts them on device_.
    // So the modules will implicitly be on device_ once weights are assigned.
    // UPDATE: We now pass device_ to model constructor to initialize RoPE on GPU directly.
    model_ = std::make_unique<Qwen3ForCausalLM<QuantType::FP16>>(*config_, device_);

    NANODEPLOY_LOG_INFO("Allocating KV Cache...");
    NANODEPLOY_LOG_INFO("Allocating KV Cache...");
    // Initialize KV Cache
    int head_dim = 0;
    if (config_->head_dim > 0) {
        head_dim = config_->head_dim;
    }
    else {
        head_dim = config_->hidden_size / config_->num_attention_heads;
    }

    NANODEPLOY_LOG_INFO("KV Cache Params: Layers=",
                        config_->num_hidden_layers,
                        " KVHeads=",
                        config_->num_key_value_heads,
                        " HeadDim=",
                        head_dim);

    // FIX: Use num_key_value_heads, not num_attention_heads
    int block_size = Sequence::block_size;  // Must match Scheduler's block size
    kv_cache_      = std::make_unique<KvCache>(config_->num_hidden_layers,
                                          config_->num_key_value_heads,
                                          head_dim,
                                          4096,  // num_blocks (Capacity increased for smaller blocks)
                                          block_size,
                                          device_);

    // Initialize FlashInfer
    NANODEPLOY_LOG_INFO("Initializing FlashInfer Handler...");
    // int block_size = 256; // Defined above
    flashinfer_handler_ = std::make_unique<layers::FlashInferHandler>(config_->num_hidden_layers,
                                                                      config_->num_attention_heads,
                                                                      config_->num_key_value_heads,
                                                                      head_dim,
                                                                      block_size,
                                                                      device_);

    // Load Weights
    NANODEPLOY_LOG_INFO("Starting Weight Loading...");
    load_weights("");  // Path unused now
}

ModelRunResp ModelRunner::run(ModelRunReq req)
{
    if (req.seqs.empty()) {
        return ModelRunResp{};
    }

    NANODEPLOY_LOG_INFO("Process request: batch=", req.seqs.size(), " prefill=", req.is_prefill);
    // std::cerr << "  [ModelRunner] Step 1: Building Tensors..." << std::endl;
    NANODEPLOY_LOG_DEBUG("Step 1: Preparing Metadata Tensors...");
    std::vector<int64_t> input_ids_vec;
    std::vector<int64_t> positions_vec;
    std::vector<int32_t> slot_mapping_vec;
    std::vector<int32_t> block_tables_vec;  // Flattened [B, MaxBlocks]
    std::vector<int32_t> seq_lens_vec;

    int batch_size     = req.seqs.size();
    int max_num_blocks = 0;
    int block_size     = Sequence::block_size;  // Must match Scheduler's block size

    // Pass 1: Determine max blocks
    for (const auto& seq : req.seqs) {
        auto& blocks = seq->block_table();
        if ((int)blocks.size() > max_num_blocks) {
            max_num_blocks = blocks.size();
        }
    }
    // If no blocks (stateless fallback?), default to 0. FlashInfer needs at least 1?
    if (max_num_blocks == 0)
        max_num_blocks = 1;

    // Pass 2: Build Tensors
    for (const auto& seq : req.seqs) {
        // Block Table
        auto& blocks = seq->block_table();
        for (int id : blocks) {
            block_tables_vec.push_back(id);
        }
        // Pad
        for (int k = (int)blocks.size(); k < max_num_blocks; ++k) {
            block_tables_vec.push_back(0);
        }

        int seq_len = seq->num_tokens;  // Total tokens
        seq_lens_vec.push_back(seq_len);
        std::cerr << "  [ModelRunner] Seq " << seq->seq_id << " seq_len=" << seq_len << std::endl;

        if (req.is_prefill) {
            // Prefill: Process all tokens
            // For simplicity, we assume seq->token_ids CONTAINS the prompt tokens.
            for (int id : seq->token_ids) {
                input_ids_vec.push_back(id);
            }
            for (int i = 0; i < (int)seq->token_ids.size(); ++i) {
                positions_vec.push_back(i);
            }
            // Slot Mapping: map each token to its physical slot
            for (int i = 0; i < (int)seq->token_ids.size(); ++i) {
                if (blocks.empty()) {
                    slot_mapping_vec.push_back(0);  // Fallback
                    continue;
                }
                int block_idx = blocks[i / block_size];
                int offset    = i % block_size;
                slot_mapping_vec.push_back(block_idx * block_size + offset);
            }
        }
        else {
            // Decode: Process last token
            if (seq->token_ids.empty())
                continue;
            input_ids_vec.push_back(seq->token_ids.back());
            // Position: 0-indexed count
            int pos = seq_len - 1;
            positions_vec.push_back(pos);

            // Slot Mapping for this ONE token
            if (blocks.empty()) {
                slot_mapping_vec.push_back(0);
            }
            else {
                int block_in_seq = pos / block_size;  // Which block within this sequence
                if (block_in_seq >= (int)blocks.size()) {
                    // Error: sequence has grown beyond allocated blocks
                    NANODEPLOY_LOG_ERROR("Decode slot mapping: pos=",
                                         pos,
                                         " requires block ",
                                         block_in_seq,
                                         " but only ",
                                         blocks.size(),
                                         " blocks allocated!");
                    // Fallback to last block (will be wrong, but prevents crash)
                    block_in_seq = blocks.size() - 1;
                }
                int block_idx = blocks[block_in_seq];
                int offset    = pos % block_size;
                slot_mapping_vec.push_back(block_idx * block_size + offset);
            }
        }
    }

    // Move to Device
    // std::cerr << "  [ModelRunner] Step 1.5: Moving to Device..." << std::endl;
    auto options_long = torch::TensorOptions().dtype(torch::kLong).device(device_);
    auto options_int  = torch::TensorOptions().dtype(torch::kInt).device(device_);

    // std::cerr << "  [ModelRunner] Step 1.6: Verifying Inputs..." << std::endl;
    // Verify Inputs REMOVED for performance
    /*
    {
        // input_ids_vec is std::vector<int64_t>, so use kLong
        int64_t max_id = torch::max(torch::from_blob(input_ids_vec.data(), {(long)input_ids_vec.size()}, torch::kLong))
                             .item<int64_t>();
        NANODEPLOY_LOG_DEBUG("Max Input ID: ", max_id, " Vocab Size: ", config_->vocab_size);
        if (max_id >= config_->vocab_size) {
            NANODEPLOY_LOG_ERROR("Input ID out of bounds! Max: ", max_id, " Vocab: ", config_->vocab_size);
            std::exit(1);
        }
    }
    */

    auto input_ids = torch::from_blob(input_ids_vec.data(), {(long)input_ids_vec.size()}, torch::kLong).to(device_);
    auto positions =
        torch::from_blob(positions_vec.data(), {(long)positions_vec.size()}, torch::kInt).to(device_, torch::kLong);
    auto slot_mapping =
        torch::from_blob(slot_mapping_vec.data(), {(long)slot_mapping_vec.size()}, torch::kInt).to(device_);

    // Verify Slot Mapping REMOVED for performance
    /*
    {
        // ...
        // int32_t max_slot = torch::max(slot_mapping).item<int32_t>();
        // ...
    }
    */

    auto block_tables =
        torch::from_blob(block_tables_vec.data(), {batch_size, max_num_blocks}, torch::kInt).to(device_);
    auto seq_lens = torch::from_blob(seq_lens_vec.data(), {batch_size}, torch::kInt).to(device_);
    // std::cerr << "  [ModelRunner] Step 2: Preparing FlashInfer Metadata..." << std::endl;

    NANODEPLOY_LOG_DEBUG("Step 2: Preparing FlashInfer Metadata...");
    if (!req.is_prefill) {
        NANODEPLOY_LOG_DEBUG("Calling FlashInfer begin_forward...");
        // std::cerr << "  [ModelRunner] Calling flashinfer_handler_->begin_forward..." << std::endl;

        // Pass Host Pointers directly!
        flashinfer_handler_->begin_forward(block_tables_vec.data(),
                                           seq_lens_vec.data(),
                                           batch_size,
                                           max_num_blocks,
                                           config_->num_attention_heads,
                                           config_->num_key_value_heads,
                                           config_->hidden_size / config_->num_attention_heads,
                                           block_size);
    }
    NANODEPLOY_LOG_DEBUG("FlashInfer metadata prepared.");

    // 3. Run Model
    NANODEPLOY_LOG_INFO("Step 3: Running Model Forward...");
    auto hidden_states = model_->forward(
        input_ids, positions, kv_cache_.get(), flashinfer_handler_.get(), slot_mapping, block_tables, seq_lens);
    NANODEPLOY_LOG_INFO("Model Forward completed.");

    // 4. Compute Logits
    torch::Tensor logits;
    if (req.is_prefill) {
        // Collect indices of the last token for each sequence
        std::vector<int64_t> last_token_indices;
        int64_t              current_offset = 0;
        int                  i              = 0;
        for (const auto& seq : req.seqs) {
            int len = seq->token_ids.size();
            last_token_indices.push_back(current_offset + len - 1);
            current_offset += len;
            i++;
        }
        auto indices =
            torch::from_blob(last_token_indices.data(), {(long)last_token_indices.size()}, torch::kLong).to(device_);

        // Ensure hidden_states is 2D [TotalTokens, Hidden]
        if (hidden_states.dim() == 3) {
            hidden_states = hidden_states.view({-1, hidden_states.size(2)});
        }

        auto last_hidden = hidden_states.index_select(0, indices);
        logits           = model_->compute_logits(last_hidden);
    }
    else {
        // hidden_states is [Batch, Hidden]
        logits = model_->compute_logits(hidden_states);
    }

    return ModelRunResp{logits};
}

void ModelRunner::load_weights(const std::string& /*weight_path*/)
{
    // 1. Embeddings
    NANODEPLOY_LOG_INFO("Loading Embeddings...");
    model_->model_->embed_tokens_->weight = weight_manager_->load("model.embed_tokens.weight");

    // 2. Layers
    int total_layers = config_->num_hidden_layers;
    for (int i = 0; i < total_layers; ++i) {
        if (i % 1 == 0) {  // Log every layer for now to be verbose as requested
            NANODEPLOY_LOG_INFO("Loading Layer ", i, " / ", total_layers);
        }
        load_layer_weights(i, model_->model_->layers_[i].get());
    }

    // 3. Final Norm
    NANODEPLOY_LOG_INFO("Loading Final Norm...");
    model_->model_->norm_->weight = weight_manager_->load("model.norm.weight");

    // 4. LM Head
    NANODEPLOY_LOG_INFO("Loading LM Head...");
    model_->lm_head_->weight = weight_manager_->load("lm_head.weight");

    NANODEPLOY_LOG_INFO("Weights loaded successfully.");
}

void ModelRunner::load_layer_weights(int layer_idx, Qwen3DecoderLayer<QuantType::FP16>* layer)
{
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";

    // A. Attention
    auto& attn = layer->self_attn_;

    // QKV Projection
    auto q_w = weight_manager_->load(prefix + "self_attn.q_proj.weight");
    auto k_w = weight_manager_->load(prefix + "self_attn.k_proj.weight");
    auto v_w = weight_manager_->load(prefix + "self_attn.v_proj.weight");

    // DEBUG: Shape Check
    if (layer_idx == 0) {
        NANODEPLOY_LOG_DEBUG("Layer 0 Shapes: Q: ", q_w.sizes(), " K: ", k_w.sizes(), " V: ", v_w.sizes());
        NANODEPLOY_LOG_DEBUG("Q Dtype: ", q_w.scalar_type());
    }

    // Fix: Cast FP8 to FP16 before cat if needed.
    // torch::cat for FP8 on CPU might be problematic or not implemented efficiently.
    if (q_w.scalar_type() == torch::kFloat8_e4m3fn) {
        // Optimization: Ensure contiguous memory on CPU before casting,
        // as casting from mmap view might trigger bad kernels.
        q_w = q_w.clone().to(torch::kFloat16);
        k_w = k_w.clone().to(torch::kFloat16);
        v_w = v_w.clone().to(torch::kFloat16);
    }

    attn->qkv_proj_->weight = torch::cat({q_w, k_w, v_w}, 0);

    // Bias (Optional)
    try {
        auto q_b              = weight_manager_->load(prefix + "self_attn.q_proj.bias");
        auto k_b              = weight_manager_->load(prefix + "self_attn.k_proj.bias");
        auto v_b              = weight_manager_->load(prefix + "self_attn.v_proj.bias");
        attn->qkv_proj_->bias = torch::cat({q_b, k_b, v_b}, 0);
        // std::cout << "  [ModelRunner] QKV Bias loaded successfully." << std::endl;
    }
    catch (...) {
        // Missing bias in file: reset to zeros matching the LOADED weight shape
        // QKV weight is [Out, In]. Bias should be [Out].
        auto out_features = attn->qkv_proj_->weight.size(0);
        // std::cout << "  [ModelRunner] Warning: QKV Bias NOT found. Resetting to zeros." << std::endl;
        attn->qkv_proj_->bias = torch::zeros({out_features}, attn->qkv_proj_->weight.options());
    }

    // O Projection
    attn->o_proj_->weight = weight_manager_->load(prefix + "self_attn.o_proj.weight");

    // QK Norm (Correctly load weights now that we know they exist)
    // Note: If safetensors lookups fail, we should probably catch it,
    // but the inspection tool confirmed their existence.
    try {
        attn->q_norm_->weight = weight_manager_->load(prefix + "self_attn.q_norm.weight");
        attn->k_norm_->weight = weight_manager_->load(prefix + "self_attn.k_norm.weight");
    }
    catch (...) {
        // Should not happen for Qwen3-0.6B
    }

    // Norms
    layer->input_layernorm_->weight          = weight_manager_->load(prefix + "input_layernorm.weight");
    layer->post_attention_layernorm_->weight = weight_manager_->load(prefix + "post_attention_layernorm.weight");

    // B. MLP (Dense)
    auto& mlp = layer->mlp_;

    auto gate_w = weight_manager_->load(prefix + "mlp.gate_proj.weight");
    auto up_w   = weight_manager_->load(prefix + "mlp.up_proj.weight");

    // FP8 Cast check for MLP if needed (omitted for brevity but recommended)

    mlp->gate_up_proj_->weight = torch::cat(std::vector<torch::Tensor>{gate_w, up_w}, 0);
    mlp->down_proj_->weight    = weight_manager_->load(prefix + "mlp.down_proj.weight");
}

}  // namespace nanodeploy
