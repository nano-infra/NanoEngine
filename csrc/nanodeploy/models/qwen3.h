#pragma once
#include <cstdio>
#include <memory>
#include <string>
#include <torch/torch.h>
#include <vector>

#include "nanodeploy/core/common.h"
#include "nanodeploy/core/config.h"
#include "nanodeploy/core/module.h"
#include "nanodeploy/logging.h"

// Layer Includes
#include "nanodeploy/layers/activation.h"
#include "nanodeploy/layers/attention.h"
#include "nanodeploy/layers/embedding.h"
#include "nanodeploy/layers/flashinfer_handler.h"
#include "nanodeploy/layers/linear.h"
#include "nanodeploy/layers/rms_norm.h"
#include "nanodeploy/layers/rotary_embedding.h"
#include "nanodeploy/logging.h"
#include "nanodeploy/worker/kv_cache.h"
#include <cuda_runtime.h>
#include <iostream>

namespace nanodeploy {
namespace models {

// =========================================================================
// Qwen3Attention
// =========================================================================
template<QuantType Quant>
class Qwen3Attention: public core::Module {
public:
    Qwen3Attention(const core::ModelConfig& config, std::shared_ptr<layers::RotaryEmbedding> rotary_emb):
        core::Module(), rotary_emb_(rotary_emb)
    {
        int hidden_size = config.hidden_size;
        int num_heads   = config.num_attention_heads;

        // Use explicit head_dim if available (critical for Qwen3 where head_dim != hidden/num_heads)
        int head_dim;
        if (config.head_dim > 0) {
            head_dim = config.head_dim;
        }
        else {
            head_dim = hidden_size / num_heads;
        }

        int num_kv_heads = config.num_key_value_heads;

        // NANODEPLOY_LOG_DEBUG("Hidden: ", hidden_size, " Heads: ", num_heads, " HeadDim: ", head_dim);

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
        // ... (existing code up to Attention selection) ...
        // Dimensions check
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
        // 1. QKV Proj
        NANODEPLOY_LOG_DEBUG("    [Attn] Start forward. Calling QKV Proj...");
        auto qkv = qkv_proj_->forward(hidden_states);
        NANODEPLOY_LOG_DEBUG("    [Attn] QKV Proj Done.");

        int head_dim  = rotary_emb_->dim_;
        int num_heads = num_heads_;  // Use stored member

        // Reshape to [B, S, (H + 2*nKV), D]
        qkv = qkv.view({batch, seq_len, -1, head_dim});

        // Split
        // Note: split logic depends on QKV layout.
        // Assuming packed [Q, K, V] where Q has H heads, K has nKV heads, V has nKV heads
        int  num_kv_heads = (qkv.size(2) - num_heads) / 2;
        auto chunks       = qkv.split({num_heads, num_kv_heads, num_kv_heads}, 2);
        auto q            = chunks[0];
        auto k            = chunks[1];
        auto v            = chunks[2];

        // Transpose to [B, H, S, D]
        q = q.transpose(1, 2);
        k = k.transpose(1, 2);
        v = v.transpose(1, 2);

        // QK Norm (Enable as weights exist)
        q = q_norm_->forward(q);
        k = k_norm_->forward(k);

        // Rotary
        std::tie(q, k) = rotary_emb_->forward(positions, q, k);

        // Attention Mechanism
        torch::Tensor attn_output;

        if (kv_cache && handler && slot_mapping.defined() && layer_idx >= 0) {
            // STATEFUL GENERATION (FlashInfer)

            // 1. Append to KV Cache
            // k, v are [B, H, S, D].
            // We need to flatten k, v to [TotalTokens, H, D] for set_kv?
            // set_kv implementation uses view({-1, ...}) so passing [B, H, S, D] is fine if slot_mapping is flat.
            auto k_cont = k.transpose(1, 2).contiguous().view({-1, num_kv_heads, head_dim});
            auto v_cont = v.transpose(1, 2).contiguous().view({-1, num_kv_heads, head_dim});

            kv_cache->set_kv(layer_idx, slot_mapping, k_cont, v_cont);

            // 2. Compute Attention (Hybrid: SDPA for Prefill, FlashInfer for Decode)
            // FlashInfer BatchDecode kernel expects seq_len=1 per request.
            bool use_flashinfer_decode = (seq_len == 1);

            if (use_flashinfer_decode) {
                // --- FlashInfer Path (Decode) ---
                int batch = q.size(0);
                int heads = q.size(1);
                int seq   = q.size(2);
                int dim   = q.size(3);

                if (!q.is_contiguous())
                    q = q.contiguous();
                auto q_fi = q.squeeze(2);  // [B, H, D]

                // Ensure q is BF16 for FlashInfer
                if (q_fi.scalar_type() != torch::kBFloat16) {
                    q_fi = q_fi.to(torch::kBFloat16);
                }

                auto k_cache_tensor = kv_cache->k_caches[layer_idx];
                auto v_cache_tensor = kv_cache->v_caches[layer_idx];

                auto output_fi = handler->attention(q_fi.data_ptr(),
                                                    k_cache_tensor.data_ptr(),
                                                    v_cache_tensor.data_ptr(),
                                                    batch,
                                                    seq,
                                                    heads,
                                                    dim,
                                                    layer_idx);

                // FlashInfer returns [B, H, D], reshape to [B, H, 1, D] for consistency
                attn_output = output_fi.unsqueeze(2);
            }
            else {
                // --- SDPA Path (Prefill) ---

                // For prefill, K/V are simply the current input's K/V (full sequence)
                auto k_sdpa = k;
                auto v_sdpa = v;

                // GQA Repeat
                if (num_heads > num_kv_heads) {
                    int n_rep = num_heads / num_kv_heads;
                    k_sdpa    = k_sdpa.repeat_interleave(n_rep, 1);
                    v_sdpa    = v_sdpa.repeat_interleave(n_rep, 1);
                }

                // Causal Masking for Prefill
                bool is_causal = true;
                attn_output    = torch::scaled_dot_product_attention(q, k_sdpa, v_sdpa, {}, 0.0, is_causal);
            }
        }
        else {
            // STATELESS / LEGACY
            if (num_heads > num_kv_heads) {
                int n_rep = num_heads / num_kv_heads;
                k         = k.repeat_interleave(n_rep, 1);
                v         = v.repeat_interleave(n_rep, 1);
            }
            attn_output = attn_->forward(q, k, v);
        }

        // Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view({batch, seq_len, -1});

        // Debug: Check o_proj weight (member variable, not method)
        auto& w = o_proj_->weight;

        // FIX: Cast attn_output to match weight dtype (FlashInfer outputs FP16, weights may be BF16)
        if (attn_output.scalar_type() != w.scalar_type()) {
            attn_output = attn_output.to(w.scalar_type());
        }

        // Output Proj
        auto out = o_proj_->forward(attn_output);
        return out;
    }

    std::unique_ptr<layers::QKVParallelLinear<Quant>> qkv_proj_;
    std::unique_ptr<layers::RowParallelLinear<Quant>> o_proj_;
    std::shared_ptr<layers::RotaryEmbedding>          rotary_emb_;
    std::unique_ptr<layers::Attention>                attn_;
    std::unique_ptr<layers::RMSNorm>                  q_norm_;
    std::unique_ptr<layers::RMSNorm>                  k_norm_;

    int num_heads_;
    int num_kv_heads_;
};

// =========================================================================
// Qwen3MLP
// =========================================================================
template<QuantType Quant>
class Qwen3MLP: public core::Module {
public:
    Qwen3MLP(const core::ModelConfig& config): core::Module()
    {
        int hidden_size       = config.hidden_size;
        int intermediate_size = config.intermediate_size;

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
// Qwen3DecoderLayer
// =========================================================================
template<QuantType Quant>
class Qwen3DecoderLayer: public core::Module {
public:
    Qwen3DecoderLayer(const core::ModelConfig& config, std::shared_ptr<layers::RotaryEmbedding> rotary_emb):
        core::Module()
    {
        self_attn_ = std::make_unique<Qwen3Attention<Quant>>(config, rotary_emb);
        mlp_       = std::make_unique<Qwen3MLP<Quant>>(config);

        input_layernorm_          = std::make_unique<layers::RMSNorm>(config.hidden_size, config.rms_norm_eps);
        post_attention_layernorm_ = std::make_unique<layers::RMSNorm>(config.hidden_size, config.rms_norm_eps);
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor              positions,
                                                     torch::Tensor              hidden_states,
                                                     KvCache*                   kv_cache     = nullptr,
                                                     layers::FlashInferHandler* handler      = nullptr,
                                                     torch::Tensor              slot_mapping = {},
                                                     int                        layer_idx    = -1,
                                                     torch::Tensor              block_tables = {},
                                                     torch::Tensor              seq_lens     = {},
                                                     torch::Tensor /*residual*/              = {})
    {
        // Standard Pre-Norm Logic: x = x + attn(norm(x))
        // ...

        // 1. Input Norm
        auto normed_hidden = input_layernorm_->forward(hidden_states);

        // DEBUG
        static bool s_logged = false;
        if (!s_logged) {
            NANODEPLOY_LOG_DEBUG("Pre-Norm Done. Entering Attn...");
        }

        auto attn_out = self_attn_->forward(
            positions, normed_hidden, kv_cache, handler, slot_mapping, layer_idx, block_tables, seq_lens);

        if (!s_logged) {
            NANODEPLOY_LOG_DEBUG("Attn Done. Entering Post-Norm...");
        }

        // ...
        hidden_states = hidden_states + attn_out;
        normed_hidden = post_attention_layernorm_->forward(hidden_states);

        if (!s_logged) {
            NANODEPLOY_LOG_DEBUG("Post-Norm Done. Entering MLP...");
            s_logged = true;
        }

        auto mlp_out  = mlp_->forward(normed_hidden);
        hidden_states = hidden_states + mlp_out;

        return {hidden_states, hidden_states};
    }
    // Members
    std::unique_ptr<Qwen3Attention<Quant>> self_attn_;
    std::unique_ptr<Qwen3MLP<Quant>>       mlp_;
    std::unique_ptr<layers::RMSNorm>       input_layernorm_;
    std::unique_ptr<layers::RMSNorm>       post_attention_layernorm_;
};

// =========================================================================
// Qwen3Model
// =========================================================================
template<QuantType Quant>
class Qwen3Model: public core::Module {
public:
    Qwen3Model(const core::ModelConfig& config, torch::Device device = torch::kCPU): core::Module()
    {
        embed_tokens_ = std::make_unique<layers::VocabParallelEmbedding>(config.vocab_size, config.hidden_size);

        int head_dim = config.hidden_size / config.num_attention_heads;
        if (config.head_dim > 0)
            head_dim = config.head_dim;

        rotary_emb_ = std::make_shared<layers::RotaryEmbedding>(
            head_dim, config.max_position_embeddings, config.rope_theta, device);

        for (int i = 0; i < config.num_hidden_layers; ++i) {
            layers_.push_back(std::make_unique<Qwen3DecoderLayer<Quant>>(config, rotary_emb_));
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
        // 1. Embedding
        auto hidden_states = embed_tokens_->forward(input_ids);

        // 2. Layers
        for (int i = 0; i < layers_.size(); ++i) {
            auto out_tuple = layers_[i]->forward(
                positions, hidden_states, kv_cache, handler, slot_mapping, i, block_tables, seq_lens);
            hidden_states = std::get<0>(out_tuple);
        }

        // 3. Final Norm
        return norm_->forward(hidden_states);
    }
    // ...

    // (Garbage removed)

public:
    std::unique_ptr<layers::VocabParallelEmbedding>        embed_tokens_;
    std::vector<std::unique_ptr<Qwen3DecoderLayer<Quant>>> layers_;
    std::unique_ptr<layers::RMSNorm>                       norm_;
    std::shared_ptr<layers::RotaryEmbedding>               rotary_emb_;
};

// =========================================================================
// Qwen3ForCausalLM
// =========================================================================
template<QuantType Quant>
class Qwen3ForCausalLM: public core::Module {
public:
    Qwen3ForCausalLM(const core::ModelConfig& config, torch::Device device = torch::kCPU): core::Module()
    {
        model_   = std::make_unique<Qwen3Model<Quant>>(config, device);
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

    // Public exposure for loading
    std::unique_ptr<Qwen3Model<Quant>>      model_;
    std::unique_ptr<layers::ParallelLMHead> lm_head_;
};

}  // namespace models
}  // namespace nanodeploy
