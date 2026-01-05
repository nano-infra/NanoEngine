#pragma once

#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "nanodeploy/json.hpp"

namespace nanodeploy {
namespace core {

struct ModelConfig {
    std::string model_type;
    int         hidden_size;
    int         num_attention_heads;
    int         num_key_value_heads;
    int         num_hidden_layers;
    int         intermediate_size;
    float       rms_norm_eps;
    int         vocab_size;
    int         max_position_embeddings;

    // MoE specific
    int  num_experts         = 0;
    int  num_experts_per_tok = 0;
    bool is_moe              = false;
    int  decoder_sparse_step = 1;

    float rope_theta = 10000.0f;

    // Quantization
    std::string quant_method = "none";  // "none", "fp8", "w8a8"

    static ModelConfig load_hf(const std::string& path)
    {
        std::ifstream f(path);
        if (!f.is_open()) {
            throw std::runtime_error("Failed to open config file: " + path);
        }
        nlohmann::json j = nlohmann::json::parse(f);

        ModelConfig config;
        config.model_type              = j.value("model_type", "");
        config.hidden_size             = j.value("hidden_size", 0);
        config.num_attention_heads     = j.value("num_attention_heads", 0);
        config.num_key_value_heads     = j.value("num_key_value_heads", 0);
        config.num_hidden_layers       = j.value("num_hidden_layers", 0);
        config.intermediate_size       = j.value("intermediate_size", 0);
        config.rms_norm_eps            = j.value("rms_norm_eps", 1e-6f);
        config.vocab_size              = j.value("vocab_size", 0);
        config.max_position_embeddings = j.value("max_position_embeddings", 0);

        if (j.contains("num_experts")) {
            config.num_experts         = j["num_experts"];
            config.num_experts_per_tok = j.value("num_experts_per_tok", 0);
            config.is_moe              = true;
            config.decoder_sparse_step = j.value("decoder_sparse_step", 1);
        }

        config.rope_theta = j.value("rope_theta", 10000.0f);

        return config;
    }
};

}  // namespace core
}  // namespace nanodeploy
