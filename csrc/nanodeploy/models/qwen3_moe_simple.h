#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <torch/nn/functional.h>
#include <torch/nn/modules/linear.h>
#include <torch/torch.h>

#include "nanodeploy/core/common.h"
#include "nanodeploy/core/config.h"
#include "nanodeploy/core/module.h"
#include "nanodeploy/layers/activation.h"
#include "nanodeploy/layers/attention.h"
#include "nanodeploy/layers/embedding.h"
#include "nanodeploy/layers/flashinfer_handler.h"
#include "nanodeploy/layers/linear.h"
#include "nanodeploy/layers/rms_norm.h"
#include "nanodeploy/layers/rotary_embedding.h"
#include "nanodeploy/logging.h"
#include "nanodeploy/worker/kv_cache.h"

namespace nanodeploy {
namespace models {

// =========================================================================
// Qwen3MoeAttention
// =========================================================================
template<QuantType Quant>
class Qwen3MoeAttention: public core::Module {
public:
    Qwen3MoeAttention(const core::ModelConfig& config, std::shared_ptr<layers::RotaryEmbedding> rotary_emb):
        core::Module(), rotary_emb_(rotary_emb)
    {
        int hidden_size = config.hidden_size;
        int num_heads   = config.num_attention_heads;

        // Use explicit head_dim if available
        int head_dim;
        if (config.head_dim > 0) {
            head_dim = config.head_dim;
        }
        else {
            head_dim = hidden_size / num_heads;
        }

        int num_kv_heads = config.num_key_value_heads;

        qkv_proj_ = std::make_unique<layers::QKVParallelLinear<Quant>>(hidden_size,
                                                                       (num_heads + 2 * num_kv_heads) * head_dim,
                                                                       /*bias=*/true);

        o_proj_ = std::make_unique<layers::RowParallelLinear<Quant>>(num_heads * head_dim,
                                                                     hidden_size,
                                                                     /*bias=*/false);

        // rotary_emb_ is properly initialized via shared_ptr

        attn_ = std::make_unique<layers::Attention>(num_heads,
                                                    head_dim,
                                                    /*scaling=*/1.0 / std::sqrt(head_dim),
                                                    num_kv_heads,
                                                    head_dim);

        q_norm_ = std::make_unique<layers::RMSNorm>(head_dim, config.rms_norm_eps);
        k_norm_ = std::make_unique<layers::RMSNorm>(head_dim, config.rms_norm_eps);

        num_heads_    = num_heads;
        num_kv_heads_ = num_kv_heads;
        head_dim_     = head_dim;
    }

    torch::Tensor forward(torch::Tensor              positions,
                          torch::Tensor              hidden_states,
                          KvCache*                   kv_cache     = nullptr,
                          layers::FlashInferHandler* handler      = nullptr,
                          torch::Tensor              slot_mapping = {},
                          int                        layer_idx    = -1,
                          torch::Tensor              block_tables = {},
                          torch::Tensor              seq_lens     = {})
    {
        auto    sizes = hidden_states.sizes();
        int64_t batch, seq_len;
        if (sizes.size() == 2) {
            batch   = 1;
            seq_len = sizes[0];
        }
        else {
            batch   = sizes[0];
            seq_len = sizes[1];
        }

        auto qkv = qkv_proj_->forward(hidden_states);
        qkv      = qkv.view({batch, seq_len, -1, head_dim_});

        // Split Q, K, V
        auto chunks = qkv.split({num_heads_, num_kv_heads_, num_kv_heads_}, 2);
        auto q      = chunks[0];
        auto k      = chunks[1];
        auto v      = chunks[2];

        // Transpose to [B, H, S, D]
        q = q.transpose(1, 2);
        k = k.transpose(1, 2);
        v = v.transpose(1, 2);

        // QK Norm
        q = q_norm_->forward(q);
        k = k_norm_->forward(k);

        // Rotary
        std::tie(q, k) = rotary_emb_->forward(positions, q, k);

        // Attention
        torch::Tensor attn_output;

        if (kv_cache && handler && slot_mapping.defined() && layer_idx >= 0) {
            // Flatten k, v for set_kv
            auto k_cont = k.transpose(1, 2).contiguous().view({-1, num_kv_heads_, head_dim_});
            auto v_cont = v.transpose(1, 2).contiguous().view({-1, num_kv_heads_, head_dim_});

            kv_cache->set_kv(layer_idx, slot_mapping, k_cont, v_cont);

            bool use_flashinfer_decode = (seq_len == 1);

            if (use_flashinfer_decode) {
                // Decode
                if (!q.is_contiguous())
                    q = q.contiguous();
                auto q_fi = q.squeeze(2);  // [B, H, D]

                if (q_fi.scalar_type() != torch::kBFloat16) {
                    q_fi = q_fi.to(torch::kBFloat16);
                }

                auto k_cache_tensor = kv_cache->k_caches[layer_idx];
                auto v_cache_tensor = kv_cache->v_caches[layer_idx];

                auto output_fi = handler->attention(q_fi.data_ptr(),
                                                    k_cache_tensor.data_ptr(),
                                                    v_cache_tensor.data_ptr(),
                                                    batch,
                                                    seq_len,
                                                    num_heads_,
                                                    head_dim_,
                                                    layer_idx);
                attn_output    = output_fi.unsqueeze(2);
            }
            else {
                // Prefill
                auto k_sdpa = k;
                auto v_sdpa = v;
                if (num_heads_ > num_kv_heads_) {
                    int n_rep = num_heads_ / num_kv_heads_;
                    k_sdpa    = k_sdpa.repeat_interleave(n_rep, 1);
                    v_sdpa    = v_sdpa.repeat_interleave(n_rep, 1);
                }
                bool is_causal = true;
                attn_output    = torch::scaled_dot_product_attention(q, k_sdpa, v_sdpa, {}, 0.0, is_causal);
            }
        }
        else {
            // Stateless
            if (num_heads_ > num_kv_heads_) {
                int n_rep = num_heads_ / num_kv_heads_;
                k         = k.repeat_interleave(n_rep, 1);
                v         = v.repeat_interleave(n_rep, 1);
            }
            attn_output = attn_->forward(q, k, v);
        }

        attn_output = attn_output.transpose(1, 2).contiguous().view({batch, seq_len, -1});

        auto& w = o_proj_->weight;
        if (attn_output.scalar_type() != w.scalar_type()) {
            attn_output = attn_output.to(w.scalar_type());
        }

        return o_proj_->forward(attn_output);
    }

    std::unique_ptr<layers::QKVParallelLinear<Quant>> qkv_proj_;
    std::unique_ptr<layers::RowParallelLinear<Quant>> o_proj_;
    std::shared_ptr<layers::RotaryEmbedding>          rotary_emb_;
    std::unique_ptr<layers::Attention>                attn_;
    std::unique_ptr<layers::RMSNorm>                  q_norm_;
    std::unique_ptr<layers::RMSNorm>                  k_norm_;

    int num_heads_;
    int num_kv_heads_;
    int head_dim_;
};

// =========================================================================
// Qwen3MoeMLP (Standard MLP)
// =========================================================================
template<QuantType Quant>
class Qwen3MoeMLP: public core::Module {
public:
    Qwen3MoeMLP(int hidden_size, int intermediate_size): core::Module()
    {
        gate_up_proj_ = std::make_unique<layers::MergedColumnParallelLinear<Quant>>(hidden_size,
                                                                                    2 * intermediate_size,
                                                                                    /*bias=*/false);

        down_proj_ = std::make_unique<layers::RowParallelLinear<Quant>>(intermediate_size,
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

    std::unique_ptr<layers::MergedColumnParallelLinear<Quant>> gate_up_proj_;
    std::unique_ptr<layers::RowParallelLinear<Quant>>          down_proj_;
    std::unique_ptr<layers::SiluAndMul>                        act_fn_;
};

// =========================================================================
// Qwen3MoeSparseMoeBlock (Simplified MoE)
// =========================================================================
template<QuantType Quant>
class Qwen3MoeSparseMoeBlock: public core::Module {
public:
    Qwen3MoeSparseMoeBlock(const core::ModelConfig& config): core::Module()
    {
        hidden_size_ = config.hidden_size;
        num_experts_ = config.num_experts;
        top_k_       = config.num_experts_per_tok;

        // Gating
        gate_ = register_module("gate",
                                torch::nn::Linear(torch::nn::LinearOptions(hidden_size_, num_experts_).bias(false)));

        // Experts
        for (int i = 0; i < num_experts_; ++i) {
            experts_.push_back(std::make_shared<Qwen3MoeMLP<Quant>>(config.hidden_size, config.moe_intermediate_size));
            register_module("experts." + std::to_string(i), experts_.back());
        }
    }

    torch::Tensor forward(torch::Tensor hidden_states)
    {
        // hidden_states: [Batch, Seq, Hidden] or [Tokens, Hidden]
        auto original_shape = hidden_states.sizes();
        auto hidden_flat    = hidden_states.view({-1, hidden_size_});
        int  num_tokens     = hidden_flat.size(0);

        auto router_logits = gate_->forward(hidden_flat);

        auto routing_weights = torch::softmax(router_logits, /*dim=*/1);

        auto topk_out         = torch::topk(routing_weights, top_k_, /*dim=*/-1);
        auto selected_weights = std::get<0>(topk_out);  // [Tokens, TopK]
        auto selected_experts = std::get<1>(topk_out);  // [Tokens, TopK]

        // Normalize weights
        selected_weights = selected_weights / selected_weights.sum(/*dim=*/-1, /*keepdim=*/true);
        selected_weights = selected_weights.to(hidden_flat.dtype());

        auto final_hidden_states = torch::zeros_like(hidden_flat);

        // Find unique experts selected
        // Using at::_unique to be robust
        auto unique_experts_tuple = at::_unique(selected_experts.view(-1));
        auto unique_experts       = std::get<0>(unique_experts_tuple);

        // Iterate over active experts
        for (int64_t i = 0; i < unique_experts.size(0); ++i) {
            int expert_idx = unique_experts[i].item<int>();

            // Mask: [Tokens, TopK] boolean
            auto mask = (selected_experts == expert_idx);

            // Indices where this expert is selected
            auto where_out     = torch::where(mask);
            auto token_indices = where_out[0];  // [NumHits]
            auto k_indices     = where_out[1];  // [NumHits] -> 0 or 1 etc.

            if (token_indices.size(0) == 0)
                continue;

            // Gather inputs: hidden_flat[token_indices]
            auto current_state = hidden_flat.index({token_indices});

            // Run expert
            auto expert_out = experts_[expert_idx]->forward(current_state);

            // Scale by weights
            auto weights = selected_weights.index({token_indices, k_indices});  // [NumHits]

            expert_out = expert_out * weights.unsqueeze(1);

            // Scatter add back
            final_hidden_states.index_add_(0, token_indices, expert_out.to(final_hidden_states.dtype()));
        }

        return final_hidden_states.view(original_shape);
    }

    int hidden_size_;
    int num_experts_;
    int top_k_;

    torch::nn::Linear                                gate_ = nullptr;
    std::vector<std::shared_ptr<Qwen3MoeMLP<Quant>>> experts_;
};

// =========================================================================
// Qwen3MoeDecoderLayer
// =========================================================================
template<QuantType Quant>
class Qwen3MoeDecoderLayer: public core::Module {
public:
    Qwen3MoeDecoderLayer(const core::ModelConfig&                 config,
                         int                                      layer_idx,
                         std::shared_ptr<layers::RotaryEmbedding> rotary_emb):
        core::Module()
    {
        self_attn_ = std::make_unique<Qwen3MoeAttention<Quant>>(config, rotary_emb);

        bool is_moe_layer = (config.num_experts > 0) && ((layer_idx + 1) % config.decoder_sparse_step == 0);

        // Check if excluded from MoE (mlp_only_layers in config? Not in C++ config yet, assuming none)

        if (is_moe_layer) {
            mlp_moe_   = std::make_unique<Qwen3MoeSparseMoeBlock<Quant>>(config);
            is_sparse_ = true;
        }
        else {
            mlp_dense_ = std::make_unique<Qwen3MoeMLP<Quant>>(config.hidden_size, config.intermediate_size);
            is_sparse_ = false;
        }

        input_layernorm_          = std::make_unique<layers::RMSNorm>(config.hidden_size, config.rms_norm_eps);
        post_attention_layernorm_ = std::make_unique<layers::RMSNorm>(config.hidden_size, config.rms_norm_eps);

        layer_idx_ = layer_idx;
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor              positions,
                                                     torch::Tensor              hidden_states,
                                                     torch::Tensor              residual,
                                                     KvCache*                   kv_cache     = nullptr,
                                                     layers::FlashInferHandler* handler      = nullptr,
                                                     torch::Tensor              slot_mapping = {},
                                                     torch::Tensor              block_tables = {},
                                                     torch::Tensor              seq_lens     = {})
    {
        // Simple Pre-Norm implementation (ignoring fused residual logic for now to ensure correctness)
        // hidden_states is the residual stream.

        // 1. Input Norm
        auto normed = input_layernorm_->forward(hidden_states);

        // 2. Attention
        auto attn_out =
            self_attn_->forward(positions, normed, kv_cache, handler, slot_mapping, layer_idx_, block_tables, seq_lens);

        // Residual Add
        hidden_states = hidden_states + attn_out;

        // 3. Post Norm
        normed = post_attention_layernorm_->forward(hidden_states);

        // 4. MLP
        torch::Tensor mlp_out;
        if (is_sparse_) {
            mlp_out = mlp_moe_->forward(normed);
        }
        else {
            mlp_out = mlp_dense_->forward(normed);
        }

        // Residual Add
        hidden_states = hidden_states + mlp_out;

        return {hidden_states, hidden_states};
    }

    std::unique_ptr<Qwen3MoeAttention<Quant>>      self_attn_;
    std::unique_ptr<Qwen3MoeSparseMoeBlock<Quant>> mlp_moe_;
    std::unique_ptr<Qwen3MoeMLP<Quant>>            mlp_dense_;
    std::unique_ptr<layers::RMSNorm>               input_layernorm_;
    std::unique_ptr<layers::RMSNorm>               post_attention_layernorm_;

    bool is_sparse_;
    int  layer_idx_;
};

// =========================================================================
// Qwen3MoeModel
// =========================================================================
template<QuantType Quant>
class Qwen3MoeModel: public core::Module {
public:
    Qwen3MoeModel(const core::ModelConfig& config, torch::Device device = torch::kCPU): core::Module()
    {
        embed_tokens_ = std::make_unique<layers::VocabParallelEmbedding>(config.vocab_size, config.hidden_size);

        int head_dim = config.head_dim > 0 ? config.head_dim : (config.hidden_size / config.num_attention_heads);

        rotary_emb_ = std::make_shared<layers::RotaryEmbedding>(
            head_dim, config.max_position_embeddings, config.rope_theta, device);

        for (int i = 0; i < config.num_hidden_layers; ++i) {
            layers_.push_back(std::make_unique<Qwen3MoeDecoderLayer<Quant>>(config, i, rotary_emb_));
        }

        norm_ = std::make_unique<layers::RMSNorm>(config.hidden_size, config.rms_norm_eps);
    }

    torch::Tensor forward(torch::Tensor              input_ids,
                          torch::Tensor              positions,
                          KvCache*                   kv_cache     = nullptr,
                          layers::FlashInferHandler* handler      = nullptr,
                          torch::Tensor              slot_mapping = {},
                          torch::Tensor              block_tables = {},
                          torch::Tensor              seq_lens     = {})
    {
        auto          hidden_states = embed_tokens_->forward(input_ids);
        torch::Tensor residual;

        for (int i = 0; i < layers_.size(); ++i) {
            auto out_tuple = layers_[i]->forward(
                positions, hidden_states, residual, kv_cache, handler, slot_mapping, block_tables, seq_lens);
            hidden_states = std::get<0>(out_tuple);
            // residual = std::get<1>(out_tuple); // logic above merges it into hidden
        }

        return norm_->forward(hidden_states);
    }

    std::unique_ptr<layers::VocabParallelEmbedding>           embed_tokens_;
    std::vector<std::unique_ptr<Qwen3MoeDecoderLayer<Quant>>> layers_;
    std::unique_ptr<layers::RMSNorm>                          norm_;
    std::shared_ptr<layers::RotaryEmbedding>                  rotary_emb_;
};

// =========================================================================
// Qwen3MoeForCausalLM
// =========================================================================
template<QuantType Quant>
class Qwen3MoeForCausalLM: public core::Module {
public:
    Qwen3MoeForCausalLM(const core::ModelConfig& config, torch::Device device = torch::kCPU): core::Module()
    {
        model_   = std::make_unique<Qwen3MoeModel<Quant>>(config, device);
        lm_head_ = std::make_unique<layers::ParallelLMHead>(config.vocab_size, config.hidden_size);
    }

    torch::Tensor forward(torch::Tensor              input_ids,
                          torch::Tensor              positions,
                          KvCache*                   kv_cache     = nullptr,
                          layers::FlashInferHandler* handler      = nullptr,
                          torch::Tensor              slot_mapping = {},
                          torch::Tensor              block_tables = {},
                          torch::Tensor              seq_lens     = {})
    {
        return model_->forward(input_ids, positions, kv_cache, handler, slot_mapping, block_tables, seq_lens);
    }

    torch::Tensor compute_logits(torch::Tensor hidden_states)
    {
        return lm_head_->forward(hidden_states);
    }

    std::unique_ptr<Qwen3MoeModel<Quant>>   model_;
    std::unique_ptr<layers::ParallelLMHead> lm_head_;
};

}  // namespace models
}  // namespace nanodeploy
