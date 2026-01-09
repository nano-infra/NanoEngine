#pragma once
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
template<QuantType Q>
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

        qkv_proj_ = std::make_unique<layers::QKVParallelLinear<Q>>(hidden_size,
                                                                   (num_heads + 2 * num_kv_heads) * head_dim,
                                                                   /*bias=*/true);

        o_proj_ = std::make_unique<layers::RowParallelLinear<Q>>(num_heads * head_dim,
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

        {
            cudaDeviceSynchronize();
            auto err = cudaGetLastError();
            if (err != cudaSuccess) {
                NANODEPLOY_LOG_DEBUG("CUDA Error after QKV Proj: ", cudaGetErrorString(err));
                std::exit(1);
            }
        }
        NANODEPLOY_LOG_DEBUG("    [Attn] QKV Proj Done.");

        std::cerr << "    [AttnDebug] QKV Shape: " << qkv.sizes() << std::endl;
        std::cerr << "    [AttnDebug] Head Dim: " << rotary_emb_->dim_ << std::endl;
        std::cerr << "    [AttnDebug] Num Heads: " << num_heads_ << std::endl;

        int head_dim  = rotary_emb_->dim_;
        int num_heads = num_heads_;  // Use stored member

        std::cerr << "    [AttnDebug] View..." << std::endl;
        // Reshape to [B, S, (H + 2*nKV), D]
        qkv = qkv.view({batch, seq_len, -1, head_dim});

        std::cerr << "    [AttnDebug] Split..." << std::endl;
        // Split
        // Note: split logic depends on QKV layout.
        // Assuming packed [Q, K, V] where Q has H heads, K has nKV heads, V has nKV heads
        int num_kv_heads = (qkv.size(2) - num_heads) / 2;
        std::cerr << "    [AttnDebug] Num KV Heads derived: " << num_kv_heads << std::endl;
        auto chunks = qkv.split({num_heads, num_kv_heads, num_kv_heads}, 2);
        auto q      = chunks[0];
        auto k      = chunks[1];
        auto v      = chunks[2];

        std::cerr << "    [AttnDebug] Transpose..." << std::endl;
        // Transpose to [B, H, S, D]
        q = q.transpose(1, 2);
        k = k.transpose(1, 2);
        v = v.transpose(1, 2);

        // Debug Stats
        {
            auto q_f = q.to(torch::kFloat32);
            fprintf(stderr,
                    "[AttnDebug] Q Stats: Mean=%f Std=%f Min=%f Max=%f\n",
                    q_f.mean().template item<float>(),
                    q_f.std().template item<float>(),
                    q_f.min().template item<float>(),
                    q_f.max().template item<float>());
            auto k_f = k.to(torch::kFloat32);
            fprintf(stderr,
                    "[AttnDebug] K Stats: Mean=%f Std=%f Min=%f Max=%f\n",
                    k_f.mean().template item<float>(),
                    k_f.std().template item<float>(),
                    k_f.min().template item<float>(),
                    k_f.max().template item<float>());
            auto v_f = v.to(torch::kFloat32);
            fprintf(stderr,
                    "[AttnDebug] V Stats: Mean=%f Std=%f Min=%f Max=%f\n",
                    v_f.mean().template item<float>(),
                    v_f.std().template item<float>(),
                    v_f.min().template item<float>(),
                    v_f.max().template item<float>());
            fflush(stderr);
        }

        std::cerr << "    [AttnDebug] Norm..." << std::endl;
        // QK Norm (Enable as weights exist)
        NANODEPLOY_LOG_DEBUG("    [Attn] QK Norm...");
        q = q_norm_->forward(q);
        k = k_norm_->forward(k);
        std::cerr << "    [AttnDebug] Norm Done." << std::endl;
        NANODEPLOY_LOG_DEBUG("    [Attn] QK Norm Done.");

        // Rotary
        fprintf(stderr, "    [Attn] Calling RoPE...\n");
        // Log Positions
        {
            auto pos_cpu = positions.to(torch::kCPU);
            auto pos_acc = pos_cpu.accessor<long, 1>();  // Assuming 1D [Total] or 2D [B, S] flattened
            // Check dim
            if (pos_cpu.dim() == 1) {
                fprintf(stderr, "    [AttnDebug] Positions (First 10):");
                for (int i = 0; i < std::min((int)pos_cpu.size(0), 10); ++i)
                    fprintf(stderr, " %ld", pos_acc[i]);
                fprintf(stderr, "\n");
            }
            else if (pos_cpu.dim() == 2) {
                fprintf(stderr, "    [AttnDebug] Positions (Batch 0, First 10):");
                auto pos_acc_2d = pos_cpu.accessor<long, 2>();
                for (int i = 0; i < std::min((int)pos_cpu.size(1), 10); ++i)
                    fprintf(stderr, " %ld", pos_acc_2d[0][i]);
                fprintf(stderr, "\n");
            }
            fflush(stderr);
        }
        std::tie(q, k) = rotary_emb_->forward(positions, q, k);

        // --- Added Debug Logic ---
        fprintf(stderr, "    [AttnDebug] RoPE returned. Syncing CUDA...\n");
        fflush(stderr);
        cudaDeviceSynchronize();
        fprintf(stderr, "    [AttnDebug] CUDA Sync Done. Preparing KV...\n");
        fflush(stderr);
        // -----------------------

        fprintf(stderr, "    [Attn] RoPE Done.\n");
        fflush(stderr);

        // Attention Mechanism
        torch::Tensor attn_output;

        if (kv_cache && handler && slot_mapping.defined() && layer_idx >= 0) {
            // STATEFUL GENERATION (FlashInfer)

            // 1. Append to KV Cache
            // k, v are [B, H, S, D].
            // set_kv expects [B, H, D] for new tokens (assuming S=1 for decode) or flattened S.
            // slot_mapping should match total tokens in k, v.
            // If prefill (S > 1), slot_mapping is [TotalTokens].
            // We need to flatten k, v to [TotalTokens, H, D] for set_kv?
            // set_kv implementation uses view({-1, ...}) so passing [B, H, S, D] is fine if slot_mapping is flat.
            // But we need to make sure k, v are permuted to [B, S, H, D] or [Total, H, D] first?
            // Current k, v are [B, H, S, D].
            // set_kv params `k` `v` expected layout?
            // In set_kv: `k_flat.index_copy_(0, slot_mapping, k)`.
            // k needs to have 0-dim size equal to slot_mapping size.
            // slot_mapping is 1D.
            // So k must be [TotalTokens, H, D].
            // We need to transpose/reshape k, v from [B, H, S, D] -> [B, S, H, D] -> [Total, H, D].

            fprintf(stderr, "    [AttnDebug] Transposing K/V for Cache...\n");
            fflush(stderr);
            auto k_cont = k.transpose(1, 2).contiguous().view({-1, num_kv_heads, head_dim});
            auto v_cont = v.transpose(1, 2).contiguous().view({-1, num_kv_heads, head_dim});

            fprintf(stderr, "    [AttnDebug] Calling set_kv...\n");
            fflush(stderr);
            kv_cache->set_kv(layer_idx, slot_mapping, k_cont, v_cont);
            fprintf(stderr, "    [AttnDebug] set_kv done.\n");
            fflush(stderr);

            // 2. Compute Attention (Hybrid: SDPA for Prefill, FlashInfer for Decode)
            // FlashInfer BatchDecode kernel expects seq_len=1 per request.
            // Passing seq_len > 1 to it is undefined/wrong without using the specialized Prefill kernels.
            // For Simplicity, we use native SDPA for Prefill.

            bool use_flashinfer_decode = (seq_len == 1);

            // USER REQUEST: Validating Slow Path First (Force SDPA for Decode too)
            bool force_slow_path = true;

            if (!use_flashinfer_decode || force_slow_path) {
                // --- SLOW PATH (Prefill OR Force Decode via SDPA) ---
                if (force_slow_path && use_flashinfer_decode) {
                    fprintf(stderr, "    [Attn] Mode: DECODE (SeqLen=1). FORCING SLOW PATH (SDPA).\n");
                }
                else {
                    fprintf(stderr, "    [Attn] Mode: PREFILL (SeqLen=%ld). Using SDPA.\n", seq_len);
                }
                fflush(stderr);

                torch::Tensor k_sdpa, v_sdpa;

                if (use_flashinfer_decode) {
                    // Decode Check
                    if (!block_tables.defined() || !seq_lens.defined()) {
                        fprintf(
                            stderr,
                            "    [Attn] ERROR: force_slow_path requires block_tables and seq_lens. Fallback to FlashInfer.\n");
                        goto run_flashinfer;
                    }

                    // Gather full history
                    auto tuple_kv = kv_cache->gather_kv(layer_idx, block_tables, seq_lens);
                    k_sdpa        = std::get<0>(tuple_kv).to(q.device()).to(q.dtype());
                    v_sdpa        = std::get<1>(tuple_kv).to(q.device()).to(q.dtype());

                    // k_sdpa is [B, KV, TotalSeq, D].
                    // We need to match Q [B, H, 1, D].
                    // SDPA expects [B, H, S, D]. S here is TotalSeq.
                    // Q has S=1. Masking?
                    // SDPA handles broadcasting S_q vs S_kv? Yes.
                    // But we need to repeat KV heads to match Q heads.
                }
                else {
                    // Prefill (existing logic)
                    k_sdpa = k;
                    v_sdpa = v;
                }

                // GQA Repeat for SDPA
                if (num_heads > num_kv_heads) {
                    int n_rep = num_heads / num_kv_heads;
                    // k_sdpa is [B, KV, S, D] or [B, H, S, D] (if already prefill logic).
                    // Prefill logic k was [B, H, S, D]... wait.
                    // In prefill block above (original code), k was transposed to [B, H, S, D] already?
                    // Line 175: k = k.transpose(1, 2); -> [B, KV, S, D]. (Wait, chunks[1] size(2) is num_kv_heads).
                    // So k is [B, KV, S, D].
                    // We need to repeat.
                    k_sdpa = k_sdpa.repeat_interleave(n_rep, 1);
                    v_sdpa = v_sdpa.repeat_interleave(n_rep, 1);
                }

                // SDPA
                attn_output = torch::scaled_dot_product_attention(q, k_sdpa, v_sdpa, {}, 0.0, true);
                fprintf(stderr, "    [Attn] SDPA Done.\n");
                fflush(stderr);
            }
            else {
                // FlashInfer Path (Decode default or Fallback)
            run_flashinfer:
                // --- DECODE PHASE (SeqLen=1) ---
                fprintf(stderr, "    [Attn] Mode: DECODE (SeqLen=1). Using FlashInfer.\n");
                fflush(stderr);

                // Decode Phase (S=1) -> Use FlashInfer
                int batch = q.size(0);
                int heads = q.size(1);
                int seq   = q.size(2);
                int dim   = q.size(3);

                // Ensure q is contiguous
                if (!q.is_contiguous()) {
                    q = q.contiguous();
                }
                // Squeeze seq dimension for decode (seq=1)
                q = q.squeeze(2);  // [B, H, 1, D] -> [B, H, D]

                auto k_cache_tensor = kv_cache->k_caches[layer_idx];
                auto v_cache_tensor = kv_cache->v_caches[layer_idx];

                attn_output = handler->attention(q.data_ptr(),
                                                 k_cache_tensor.data_ptr(),
                                                 v_cache_tensor.data_ptr(),
                                                 batch,
                                                 seq,
                                                 heads,
                                                 dim,
                                                 layer_idx);
                fprintf(stderr, "    [AttnDebug] FlashInfer attention returned (Layer %d).\n", layer_idx);
                fflush(stderr);
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

        // Post-Attention Debug Stats
        {
            auto a_f = attn_output.to(torch::kFloat32);
            fprintf(stderr,
                    "[AttnDebug] Output Stats: Mean=%f Std=%f Min=%f Max=%f\n",
                    a_f.mean().template item<float>(),
                    a_f.std().template item<float>(),
                    a_f.min().template item<float>(),
                    a_f.max().template item<float>());
            fflush(stderr);
        }

        // Reshape back
        fprintf(stderr, "    [AttnDebug] Reshaping attn_output... Shape before: [%ld", attn_output.size(0));
        for (int i = 1; i < attn_output.dim(); ++i)
            fprintf(stderr, ", %ld", attn_output.size(i));
        fprintf(stderr, "]\n");
        fflush(stderr);
        attn_output = attn_output.transpose(1, 2).contiguous().view({batch, seq_len, -1});
        fprintf(stderr,
                "    [AttnDebug] Reshape done. Shape: [%ld, %ld, %ld]. Syncing CUDA...\n",
                attn_output.size(0),
                attn_output.size(1),
                attn_output.size(2));
        fflush(stderr);
        cudaDeviceSynchronize();
        fprintf(stderr, "    [AttnDebug] CUDA synced. Checking o_proj weights...\n");
        fflush(stderr);

        // Debug: Check o_proj weight (member variable, not method)
        auto& w = o_proj_->weight;
        fprintf(stderr,
                "    [AttnDebug] o_proj weight: Shape=[%ld, %ld], Device=%s, Dtype=%s\n",
                w.size(0),
                w.size(1),
                w.device().str().c_str(),
                c10::toString(w.scalar_type()));
        fflush(stderr);

        fprintf(stderr, "    [AttnDebug] Calling o_proj->forward...\n");
        fprintf(stderr, "    [AttnDebug] attn_output Dtype: %s\n", c10::toString(attn_output.scalar_type()));
        fflush(stderr);

        // FIX: Cast attn_output to match weight dtype (FlashInfer outputs FP16, weights may be BF16)
        if (attn_output.scalar_type() != w.scalar_type()) {
            fprintf(stderr,
                    "    [AttnDebug] Dtype mismatch! Casting attn_output from %s to %s\n",
                    c10::toString(attn_output.scalar_type()),
                    c10::toString(w.scalar_type()));
            fflush(stderr);
            attn_output = attn_output.to(w.scalar_type());
        }

        // Output Proj
        auto out = o_proj_->forward(attn_output);
        fprintf(stderr, "    [AttnDebug] o_proj done. Returning.\n");
        fflush(stderr);
        return out;
    }

    std::unique_ptr<layers::QKVParallelLinear<Q>> qkv_proj_;
    std::unique_ptr<layers::RowParallelLinear<Q>> o_proj_;
    std::shared_ptr<layers::RotaryEmbedding>      rotary_emb_;
    std::unique_ptr<layers::Attention>            attn_;
    std::unique_ptr<layers::RMSNorm>              q_norm_;
    std::unique_ptr<layers::RMSNorm>              k_norm_;

    int num_heads_;
    int num_kv_heads_;
};

// =========================================================================
// Qwen3MLP
// =========================================================================
template<QuantType Q>
class Qwen3MLP: public core::Module {
public:
    Qwen3MLP(const core::ModelConfig& config): core::Module()
    {
        int hidden_size       = config.hidden_size;
        int intermediate_size = config.intermediate_size;

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

    std::unique_ptr<layers::MergedColumnParallelLinear<Q>> gate_up_proj_;
    std::unique_ptr<layers::RowParallelLinear<Q>>          down_proj_;
    std::unique_ptr<layers::SiluAndMul>                    act_fn_;
};

// =========================================================================
// Qwen3DecoderLayer
// =========================================================================
template<QuantType Q>
class Qwen3DecoderLayer: public core::Module {
public:
    Qwen3DecoderLayer(const core::ModelConfig& config, std::shared_ptr<layers::RotaryEmbedding> rotary_emb):
        core::Module()
    {
        self_attn_ = std::make_unique<Qwen3Attention<Q>>(config, rotary_emb);
        mlp_       = std::make_unique<Qwen3MLP<Q>>(config);

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
    std::unique_ptr<Qwen3Attention<Q>> self_attn_;
    std::unique_ptr<Qwen3MLP<Q>>       mlp_;
    std::unique_ptr<layers::RMSNorm>   input_layernorm_;
    std::unique_ptr<layers::RMSNorm>   post_attention_layernorm_;
};

// =========================================================================
// Qwen3Model
// =========================================================================
template<QuantType Q>
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
            layers_.push_back(std::make_unique<Qwen3DecoderLayer<Q>>(config, rotary_emb_));
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
    std::unique_ptr<layers::VocabParallelEmbedding>    embed_tokens_;
    std::vector<std::unique_ptr<Qwen3DecoderLayer<Q>>> layers_;
    std::unique_ptr<layers::RMSNorm>                   norm_;
    std::shared_ptr<layers::RotaryEmbedding>           rotary_emb_;
};

// =========================================================================
// Qwen3ForCausalLM
// =========================================================================
template<QuantType Q>
class Qwen3ForCausalLM: public core::Module {
public:
    Qwen3ForCausalLM(const core::ModelConfig& config, torch::Device device = torch::kCPU): core::Module()
    {
        model_   = std::make_unique<Qwen3Model<Q>>(config, device);
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
    std::unique_ptr<Qwen3Model<Q>>          model_;
    std::unique_ptr<layers::ParallelLMHead> lm_head_;
};

}  // namespace models
}  // namespace nanodeploy
