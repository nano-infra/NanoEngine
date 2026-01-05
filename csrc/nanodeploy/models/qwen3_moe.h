#pragma once
#include <memory>
#include <string>
#include <torch/torch.h>
#include <vector>

#include "nanodeploy/core/common.h"
#include "nanodeploy/core/config.h"
#include "nanodeploy/core/module.h"

// Layer Includes
#include "nanodeploy/layers/activation.h"
#include "nanodeploy/layers/attention.h"
#include "nanodeploy/layers/embedding.h"
#include "nanodeploy/layers/linear.h"
#include "nanodeploy/layers/rms_norm.h"
#include "nanodeploy/layers/rotary_embedding.h"

namespace nanodeploy {
namespace models {

// Helper for quantization config (Placeholder)
struct QuantizationConfig {
    QuantType        quant_method = QuantType::FP16;
    std::vector<int> block_size   = {128, 128};  // [m, n]
};

// =========================================================================
// Qwen3MoeAttention
// =========================================================================
template<QuantType Q>
class Qwen3MoeAttention: public core::Module {
public:
    Qwen3MoeAttention(const core::ModelConfig& config): core::Module()
    {
        // Dimensions
        int hidden_size = config.hidden_size;
        int num_heads   = config.num_attention_heads;
        int head_dim    = hidden_size / num_heads;  // Assuming head_dim is derived this way
        // config might need explicit head_dim for some models

        int num_kv_heads = config.num_key_value_heads;

        // Layers
        qkv_proj_ = std::make_unique<layers::QKVParallelLinear<Q>>(
            hidden_size,
            (num_heads + 2 * num_kv_heads) * head_dim,
            /*bias=*/true  // Attention usually has bias in Qwen? Check config. Qwen2 usually true.
        );

        o_proj_ = std::make_unique<layers::RowParallelLinear<Q>>(num_heads * head_dim,
                                                                 hidden_size,
                                                                 /*bias=*/false);

        rotary_emb_ =
            std::make_unique<layers::RotaryEmbedding>(head_dim, config.max_position_embeddings, config.rope_theta);

        attn_ = std::make_unique<layers::Attention>(num_heads,
                                                    head_dim,
                                                    /*scaling=*/1.0 / std::sqrt(head_dim),  // Default scaling
                                                    num_kv_heads,
                                                    head_dim);

        q_norm_ = std::make_unique<layers::RMSNorm>(head_dim, config.rms_norm_eps);
        k_norm_ = std::make_unique<layers::RMSNorm>(head_dim, config.rms_norm_eps);
    }

    torch::Tensor forward(torch::Tensor positions, torch::Tensor hidden_states)
    {
        // hidden_states: [Batch, Seq, Hidden]
        // hidden_states: [Batch, Seq, Hidden]
        auto sizes   = hidden_states.sizes();
        auto batch   = sizes[0];
        auto seq_len = sizes[1];

        // 1. QKV Proj
        auto qkv = qkv_proj_->forward(hidden_states);  // [B, S, (H + 2*nKV) * D]

        // 2. Split and Reshape for GQA
        int head_dim  = rotary_emb_->dim_;
        int num_heads = q_norm_->weight.size(0) / head_dim;  // Indirect check size?
        // Better to store num_heads? Re-calculate from hidden_size_?
        // Let's assume standard config access or store member variables.
        // For now, infer from tensor sizes if possible or pass config.
        // Actually, let's just use the known splits logic.

        int num_kv_heads = (qkv.size(2) / head_dim - num_heads) / 2;  // Infer or assume
        // The qkv_proj output size is (num_heads + 2*num_kv_heads) * head_dim
        // We can just chunk.

        // Reshape to [B, S, (H + 2*nKV), D]
        qkv = qkv.view({batch, seq_len, -1, head_dim});

        // Split
        auto chunks = qkv.split({num_heads, num_kv_heads, num_kv_heads}, 2);
        auto q      = chunks[0];  // [B, S, H, D]
        auto k      = chunks[1];  // [B, S, nKV, D]
        auto v      = chunks[2];  // [B, S, nKV, D]

        // Transpose to [B, H, S, D] for attention
        q = q.transpose(1, 2);
        k = k.transpose(1, 2);
        v = v.transpose(1, 2);

        // 3. Rotary Embedding
        std::tie(q, k) = rotary_emb_->forward(positions, q, k);

        // 4. Repeat KV for GQA if needed (num_heads > num_kv_heads)
        if (num_heads > num_kv_heads) {
            int n_rep = num_heads / num_kv_heads;
            k         = k.repeat_interleave(n_rep, 1);
            v         = v.repeat_interleave(n_rep, 1);
        }

        // 5. Attention
        auto attn_output = attn_->forward(q, k, v);  // [B, H, S, D]

        // 6. Reshape back [B, S, H, D] -> [B, S, Hidden]
        attn_output = attn_output.transpose(1, 2).contiguous().view({batch, seq_len, -1});

        // 7. Output Proj
        return o_proj_->forward(attn_output);
    }

    // private:
    std::unique_ptr<layers::QKVParallelLinear<Q>> qkv_proj_;
    std::unique_ptr<layers::RowParallelLinear<Q>> o_proj_;
    std::unique_ptr<layers::RotaryEmbedding>      rotary_emb_;
    std::unique_ptr<layers::Attention>            attn_;
    std::unique_ptr<layers::RMSNorm>              q_norm_;
    std::unique_ptr<layers::RMSNorm>              k_norm_;
};

// =========================================================================
// Qwen3MoeMLP
// =========================================================================
template<QuantType Q>
class Qwen3MoeMLP: public core::Module {
public:
    Qwen3MoeMLP(const core::ModelConfig& config): core::Module()
    {
        int hidden_size       = config.hidden_size;
        int intermediate_size = config.intermediate_size;

        // Merged Gate + Up projection
        // Output dim = 2 * intermediate_size (gate + up)
        gate_up_proj_ = std::make_unique<layers::MergedColumnParallelLinear<Q>>(hidden_size,
                                                                                2 * intermediate_size,
                                                                                /*bias=*/false);

        down_proj_ = std::make_unique<layers::RowParallelLinear<Q>>(intermediate_size,
                                                                    hidden_size,
                                                                    /*bias=*/false);

        act_fn_ = std::make_unique<layers::SiluAndMul>();
    }

    torch::Tensor forward(torch::Tensor x)
    {
        auto gate_up = gate_up_proj_->forward(x);
        auto act_out = act_fn_->forward(gate_up);
        return down_proj_->forward(act_out);
    }

    // private:
    std::unique_ptr<layers::MergedColumnParallelLinear<Q>> gate_up_proj_;
    std::unique_ptr<layers::RowParallelLinear<Q>>          down_proj_;
    std::unique_ptr<layers::SiluAndMul>                    act_fn_;
    std::unique_ptr<layers::RMSNorm> gate_up_norm_;  // Added placeholder if needed or just expose existing
};

// =========================================================================
// Qwen3MoeSparseMoeBlock
// =========================================================================
template<QuantType Q>
class Qwen3MoeSparseMoeBlock: public core::Module {
public:
    Qwen3MoeSparseMoeBlock(const core::ModelConfig& /*config*/): core::Module()
    {
        // DeepEP logic will go here
    }

    torch::Tensor forward(torch::Tensor hidden_states)
    {
        return hidden_states;
    }

    // private:
    // Gating
    std::unique_ptr<torch::nn::Linear> gate_;

    // Experts (List of MLPs)
    std::vector<std::unique_ptr<Qwen3MoeMLP<Q>>> experts_;

    // DeepEP related (Placeholder)
    // std::unique_ptr<DeepEPMoe> moe_;
};

// =========================================================================
// Qwen3MoeDecoderLayer
// =========================================================================
template<QuantType Q>
class Qwen3MoeDecoderLayer: public core::Module {
public:
    Qwen3MoeDecoderLayer(const core::ModelConfig& config, int layer_idx): core::Module()
    {

        self_attn_ = std::make_unique<Qwen3MoeAttention<Q>>(config);

        // Simplified selection logic: Sparse or Dense
        if (config.num_experts > 0 && (layer_idx + 1) % config.decoder_sparse_step == 0) {
            mlp_ = std::make_unique<Qwen3MoeSparseMoeBlock<Q>>(config);
        }
        else {
            // mlp_ = std::make_unique<Qwen3MoeMLP<Q>>(config);
            // Type mismatch here if using same ptr, might need Variant or Base class
            // keeping simple for now
        }

        input_layernorm_          = std::make_unique<layers::RMSNorm>(config.hidden_size);
        post_attention_layernorm_ = std::make_unique<layers::RMSNorm>(config.hidden_size);
    }

    std::tuple<torch::Tensor, torch::Tensor>
    forward(torch::Tensor /*positions*/, torch::Tensor hidden_states, torch::Tensor residual)
    {
        // Forward logic
        return {hidden_states, residual};
    }

    // private:
    std::unique_ptr<Qwen3MoeAttention<Q>> self_attn_;
    // Using Module* or Variant for MLP/MoE
    std::unique_ptr<core::Module> mlp_;

    std::unique_ptr<layers::RMSNorm> input_layernorm_;
    std::unique_ptr<layers::RMSNorm> post_attention_layernorm_;
};

// =========================================================================
// Qwen3MoeModel
// =========================================================================
template<QuantType Q>
class Qwen3MoeModel: public core::Module {
public:
    Qwen3MoeModel(const core::ModelConfig& config): core::Module()
    {
        embed_tokens_ = std::make_unique<layers::VocabParallelEmbedding>(config.vocab_size, config.hidden_size);

        for (int i = 0; i < config.num_hidden_layers; ++i) {
            layers_.push_back(std::make_unique<Qwen3MoeDecoderLayer<Q>>(config, i));
        }

        norm_ = std::make_unique<layers::RMSNorm>(config.hidden_size);
    }

    torch::Tensor forward(torch::Tensor input_ids, torch::Tensor positions)
    {
        // 1. Embedding
        auto hidden_states = embed_tokens_->forward(input_ids);

        // 2. Decoder Layers
        for (auto& layer : layers_) {
            // In simple non-pipeline case, residual is handled inside?
            // Or we pass implicit residual. My DecoderLayer currently returns {hidden, residual}
            // but effectively modifies hidden_states as main stream.
            // Let's simplify: layer takes (pos, hidden) -> hidden
            // Wait, I designed it to take residual.
            // Let's just pass dummy residual for now or fix DecoderLayer signature if I want pure residual stream.
            // For standard transformers, usually just passing hidden_states is enough if residual is inside.
            // My implementation (hidden = hidden + attn) implies residual is inside.

            auto [out, _] = layer->forward(positions, hidden_states, /*residual=*/{});
            hidden_states = out;
        }

        // 3. Final Norm
        hidden_states = norm_->forward(hidden_states);

        return hidden_states;
    }

    // private:
    std::unique_ptr<layers::VocabParallelEmbedding>       embed_tokens_;
    std::vector<std::unique_ptr<Qwen3MoeDecoderLayer<Q>>> layers_;
    std::unique_ptr<layers::RMSNorm>                      norm_;
};

// =========================================================================
// Qwen3MoeForCausalLM (Main Entry)
// =========================================================================
template<QuantType Q>
class Qwen3MoeForCausalLM: public core::Module {
public:
    Qwen3MoeForCausalLM(const core::ModelConfig& config): core::Module()
    {
        model_   = std::make_unique<Qwen3MoeModel<Q>>(config);
        lm_head_ = std::make_unique<layers::ParallelLMHead>(config.vocab_size, config.hidden_size);
    }

    torch::Tensor forward(torch::Tensor input_ids, torch::Tensor positions)
    {
        return model_->forward(input_ids, positions);
    }

    torch::Tensor compute_logits(torch::Tensor hidden_states)
    {
        // return lm_head_->forward(hidden_states);
        return hidden_states;
    }

    // private:
    std::unique_ptr<Qwen3MoeModel<Q>>       model_;
    std::unique_ptr<layers::ParallelLMHead> lm_head_;
};

}  // namespace models
}  // namespace nanodeploy
