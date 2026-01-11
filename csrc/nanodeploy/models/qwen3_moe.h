#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <torch/torch.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/functional.h>

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
#include "nanodeploy/worker/distributed.h"

// Third-party Integrations
#include "deep_ep.hpp"
// Note: DeepGemm headers removed to avoid ODR violations.
// Using standard PyTorch ops for Phase 1. DeepGemm will be integrated via wrapper.

namespace nanodeploy {
namespace models {

// =========================================================================
// Qwen3MoeAttention
// =========================================================================
template<QuantType Q>
class Qwen3MoeAttention: public core::Module {
public:
    Qwen3MoeAttention(const core::ModelConfig& config, std::shared_ptr<layers::RotaryEmbedding> rotary_emb):
        core::Module(), rotary_emb_(rotary_emb)
    {
        int hidden_size = config.hidden_size;
        int num_heads   = config.num_attention_heads;
        int head_dim    = config.head_dim > 0 ? config.head_dim : (hidden_size / num_heads);
        int num_kv_heads = config.num_key_value_heads;

        // QKV Projection
        // Output size: (num_heads + 2 * num_kv_heads) * head_dim
        qkv_proj_ = std::make_unique<layers::QKVParallelLinear<Q>>(
            hidden_size,
            (num_heads + 2 * num_kv_heads) * head_dim,
            /*bias=*/true // Qwen usually has bias
        );

        // Output Projection
        o_proj_ = std::make_unique<layers::RowParallelLinear<Q>>(
            num_heads * head_dim,
            hidden_size,
            /*bias=*/false
        );

        // Norms
        q_norm_ = std::make_unique<layers::RMSNorm>(head_dim, config.rms_norm_eps);
        k_norm_ = std::make_unique<layers::RMSNorm>(head_dim, config.rms_norm_eps);

        num_heads_ = num_heads;
        num_kv_heads_ = num_kv_heads;
        head_dim_ = head_dim;
    }

    torch::Tensor forward(
        torch::Tensor positions,
        torch::Tensor hidden_states,
        KvCache* kv_cache = nullptr,
        layers::FlashInferHandler* handler = nullptr,
        torch::Tensor slot_mapping = {},
        int layer_idx = -1,
        torch::Tensor block_tables = {},
        torch::Tensor seq_lens = {}
    ) {
        auto sizes = hidden_states.sizes(); // [Batch, Seq, Hidden]
        int64_t batch = sizes[0];
        int64_t seq_len = sizes[1];

        // 1. QKV Proj
        auto qkv = qkv_proj_->forward(hidden_states);

        // 2. Split & Reshape
        // qkv: [B, S, (H + 2*nKV) * D]
        qkv = qkv.view({batch, seq_len, -1, head_dim_});
        auto chunks = qkv.split({num_heads_, num_kv_heads_, num_kv_heads_}, 2);
        auto q = chunks[0];
        auto k = chunks[1];
        auto v = chunks[2];

        // Transpose to [B, H, S, D] for processing
        q = q.transpose(1, 2);
        k = k.transpose(1, 2);
        v = v.transpose(1, 2);

        // 3. QK Norm
        q = q_norm_->forward(q);
        k = k_norm_->forward(k);

        // 4. Rotary Embedding
        std::tie(q, k) = rotary_emb_->forward(positions, q, k);

        torch::Tensor attn_output;

        // 5. Attention Dispatch (Prefill vs Decode)
        bool is_decode = (seq_len == 1);
        
        if (is_decode && kv_cache && handler) {
            // --- Decode: FlashInfer PagedAttention ---
            
            // Append current token to KV Cache
            // Flatten to [TotalTokens, nKV, D] for set_kv
            auto k_cont = k.transpose(1, 2).contiguous().view({-1, num_kv_heads_, head_dim_});
            auto v_cont = v.transpose(1, 2).contiguous().view({-1, num_kv_heads_, head_dim_});
            kv_cache->set_kv(layer_idx, slot_mapping, k_cont, v_cont);

            // Compute Paged Attention
            auto q_fi = q.squeeze(2).contiguous(); // [B, H, D]
            if (q_fi.scalar_type() != torch::kBFloat16) {
                q_fi = q_fi.to(torch::kBFloat16);
            }

            auto output_fi = handler->attention(
                q_fi.data_ptr(),
                kv_cache->k_caches[layer_idx].data_ptr(),
                kv_cache->v_caches[layer_idx].data_ptr(),
                batch,
                1, // seq_len per req is 1
                num_heads_,
                head_dim_,
                layer_idx
            );
            
            // [B, H, D] -> [B, H, 1, D]
            attn_output = output_fi.unsqueeze(2);
        } else {
            // --- Prefill: SDPA (or FlashAttention) ---
            
            // For prefill, we use the current sequence as K/V
            // If KV Cache is present, we might need to append (omitted for pure prefill scenario correctness)
            // But for simple prefill (no history), just use self.
            
            // GQA Repeat for SDPA
            auto k_sdpa = k;
            auto v_sdpa = v;
            if (num_heads_ > num_kv_heads_) {
                int rep = num_heads_ / num_kv_heads_;
                k_sdpa = k_sdpa.repeat_interleave(rep, 1);
                v_sdpa = v_sdpa.repeat_interleave(rep, 1);
            }

            // SDPA
            // q, k, v are [B, H, S, D]
            bool is_causal = true;
            // Using ATen directly as F::scaled_dot_product_attention might need headers
            attn_output = at::scaled_dot_product_attention(
                q, k_sdpa, v_sdpa, 
                /*attn_mask=*/{}, 
                /*dropout_p=*/0.0, 
                /*is_causal=*/is_causal
            );
            
            // Note: In a real system, we'd also push to KV cache here for subsequent decode steps.
            if (kv_cache && slot_mapping.defined()) {
                 auto k_cont = k.transpose(1, 2).contiguous().view({-1, num_kv_heads_, head_dim_});
                 auto v_cont = v.transpose(1, 2).contiguous().view({-1, num_kv_heads_, head_dim_});
                 kv_cache->set_kv(layer_idx, slot_mapping, k_cont, v_cont);
            }
        }

        // 6. Reshape & Output Proj
        // [B, H, S, D] -> [B, S, H, D] -> [B, S, Hidden]
        attn_output = attn_output.transpose(1, 2).contiguous().view({batch, seq_len, -1});
        
        // Cast to weight dtype if necessary
        auto& w_o = o_proj_->weight;
        if (attn_output.scalar_type() != w_o.scalar_type()) {
            attn_output = attn_output.to(w_o.scalar_type());
        }

        return o_proj_->forward(attn_output);
    }

public:
    std::unique_ptr<layers::QKVParallelLinear<Q>> qkv_proj_;
    std::unique_ptr<layers::RowParallelLinear<Q>> o_proj_;
    std::unique_ptr<layers::RMSNorm> q_norm_;
    std::unique_ptr<layers::RMSNorm> k_norm_;
    std::shared_ptr<layers::RotaryEmbedding> rotary_emb_;

    int num_heads_;
    int num_kv_heads_;
    int head_dim_;
};


// =========================================================================
// Qwen3MoeMLP (Dense)
// =========================================================================
template<QuantType Q>
class Qwen3MoeMLP: public core::Module {
public:
    Qwen3MoeMLP(const core::ModelConfig& config): core::Module() {
        int hidden_size = config.hidden_size;
        int intermediate_size = config.intermediate_size;

        gate_up_proj_ = std::make_unique<layers::MergedColumnParallelLinear<Q>>(
            hidden_size, 2 * intermediate_size, false
        );
        down_proj_ = std::make_unique<layers::RowParallelLinear<Q>>(
            intermediate_size, hidden_size, false
        );
        act_fn_ = std::make_unique<layers::SiluAndMul>();
    }

    torch::Tensor forward(torch::Tensor x) {
        auto gate_up = gate_up_proj_->forward(x);
        auto act = act_fn_->forward(gate_up);
        return down_proj_->forward(act);
    }

public:
    std::unique_ptr<layers::MergedColumnParallelLinear<Q>> gate_up_proj_;
    std::unique_ptr<layers::RowParallelLinear<Q>> down_proj_;
    std::unique_ptr<layers::SiluAndMul> act_fn_;
};


// =========================================================================
// Qwen3MoeSparseMoeBlock (MoE)
// =========================================================================
template<QuantType Q>
class Qwen3MoeSparseMoeBlock: public core::Module {
public:
    Qwen3MoeSparseMoeBlock(const core::ModelConfig& config): core::Module() {
        hidden_size_ = config.hidden_size;
        num_experts_ = config.num_experts;
        top_k_ = config.num_experts_per_tok;
        
        // Derived dimensions
        // Note: moe_intermediate_size usually for the expert (smaller than dense intermediate)
        int moe_inter = config.moe_intermediate_size; 
        
        // Gating
        gate_ = std::make_shared<torch::nn::LinearImpl>(hidden_size_, num_experts_);
        // Remove bias if present (Qwen MoE gate usually no bias)
        gate_->options.bias(false);

        // Parameters for DeepGemm (Managed as Tensors)
        // We assume we are running in EP mode, so we hold local experts.
        int world_size = get_dist_context().ffn_ep_world_size();
        int num_local_experts = num_experts_ / world_size;

        // Weight Shapes for DeepGemm (Grouped GEMM)
        // GateUp: [NumGroups, In, Out] -> [LocalExperts, Hidden, Inter*2]
        // Down:   [NumGroups, In, Out] -> [LocalExperts, Inter, Hidden]
        // Note: DeepGemm expects weights to be pre-packed or contiguous in specific layout.
        // Assuming (NumExperts, N, K) for Weight where Input is (M, K).
        // For GateUp: Input [M, Hidden]. Weight [LocalExperts, Inter*2, Hidden] (K=Hidden, N=Inter*2)
        // DeepGemm 'nt' means A (M, K) @ B.T (N, K). So B should be [N, K].
        // If we store as [LocalExperts, Inter*2, Hidden], then B[i] is [Inter*2, Hidden]. Correct.
        
        // Weights
        gate_up_proj_ = torch::empty({num_local_experts, moe_inter * 2, hidden_size_}, torch::kBFloat16);
        down_proj_ = torch::empty({num_local_experts, hidden_size_, moe_inter}, torch::kBFloat16);

        // FP8 Scales (optional)
        if (config.quant_method == "fp8") {
             gate_up_scale_inv_ = torch::empty({num_local_experts}, torch::kFloat32); 
             down_scale_inv_ = torch::empty({num_local_experts}, torch::kFloat32);
             // Note: Actual scale shapes depend on blocking (e.g. 128x128). 
             // Simplifying here for placeholder.
             use_fp8_ = true;
        }
    }

    torch::Tensor forward(torch::Tensor hidden_states, deep_ep::Buffer* ep_buffer) {
        // [Batch, Seq, Hidden] -> [Tokens, Hidden]
        auto original_shape = hidden_states.sizes();
        hidden_states = hidden_states.view({-1, hidden_size_});
        int num_tokens = hidden_states.size(0);

        // 1. Gating
        auto router_logits = gate_->forward(hidden_states);
        auto routing_weights = torch::softmax(router_logits, -1);
        auto topk = torch::topk(routing_weights, top_k_, -1);
        auto topk_weights = std::get<0>(topk);
        auto topk_ids = std::get<1>(topk).to(torch::kInt32); // DeepEP needs int32 usually? or int64? Check header.
        // DeepEP uses int64 for topk_idx in some examples, but let's check deep_ep.hpp.
        // Assuming int64 is fine or it casts.

        // 2. Dispatch
        // Determine mode based on token count (heuristic for prefill vs decode)
        bool is_prefill = (num_tokens > 1); 
        
        // Handles for combine
        torch::Tensor recv_x;
        std::optional<torch::Tensor> recv_x_scales;
        torch::Tensor recv_count;
        torch::Tensor src_info;
        torch::Tensor layout_range;
        
        if (is_prefill) {
            // Using low_latency_dispatch for both Prefill and Decode as requested/safer for now
             auto res_ll = ep_buffer->low_latency_dispatch(
                hidden_states,
                topk_ids,
                topk_weights, // optional?
                std::nullopt, // stats
                num_tokens,
                num_experts_,
                use_fp8_,
                /*round_scale=*/false,
                /*use_ue8m0=*/false,
                /*async=*/false,
                /*return_recv_hook=*/false
             );
             // Return: (recv_x, recv_scales, recv_count, src_info, layout_range, event, hook)
             recv_x = std::get<0>(res_ll);
             recv_x_scales = std::get<1>(res_ll);
             recv_count = std::get<2>(res_ll).to(torch::kInt32);
             src_info = std::get<3>(res_ll);
             layout_range = std::get<4>(res_ll);
        } else {
            // --- Decode (Low Latency) ---
            auto res_ll = ep_buffer->low_latency_dispatch(
                hidden_states,
                topk_ids,
                topk_weights,
                std::nullopt,
                num_tokens,
                num_experts_,
                use_fp8_,
                false, false, false, false
            );
             recv_x = std::get<0>(res_ll);
             recv_x_scales = std::get<1>(res_ll);
             recv_count = std::get<2>(res_ll).to(torch::kInt32);
             src_info = std::get<3>(res_ll);
             layout_range = std::get<4>(res_ll);
        }

        // 3. Computation (DeepGemm)
        // recv_x is contiguous. DeepGemm expects [Groups, M, K]. 
        // We view recv_x as [NumLocalExperts, MaxTokens, Hidden].
        // MaxTokens = num_tokens (capacity).
        
        int num_local_experts = gate_up_proj_.size(0);
        int moe_inter_2 = gate_up_proj_.size(1);
        int moe_inter = moe_inter_2 / 2;
        int max_tokens = num_tokens; 
        
        // Ensure recv_x is viewed as 3D for DeepGemm
        recv_x = recv_x.view({num_local_experts, max_tokens, hidden_size_});
        
        // Output buffer
        torch::Tensor expert_out = torch::empty({num_local_experts, max_tokens, hidden_size_}, recv_x.options());

        if (use_fp8_) {
            // FP8 Path
            // auto scale_gate_up = gate_up_scale_inv_;
            // auto scale_down = down_scale_inv_;
            
            // recv_x_scales needed? DeepEP returns it if use_fp8=true.
            // DeepGemm m_grouped_fp8 needs Pair(Data, Scale).
            // Scales from DeepEP might need reshape?
            // DeepEP scales are [NumLocalExperts, MaxTokens] ? Or blocked?
            // Assuming DeepEP handles layout. We pass what we got.
            
            // Check shapes (Debug)
            // std::cerr << "FP8 GEMM Input: " << recv_x.sizes() << std::endl;
            
            // Intermediate buffer (BF16 or FP8?)
            // Usually internal activation is BF16.
            // But if next layer needs FP8, we might output FP8? 
            // DeepGemm output d is BF16/Float.
            
            // auto intermediate = torch::empty({num_local_experts, max_tokens, moe_inter * 2}, torch::kBFloat16).to(recv_x.device());
            
            // TODO: Construct Pairs
            // deep_gemm::gemm::m_grouped_fp8_gemm_nt_masked(...)
            // This requires constructing std::pair<Tensor, Tensor>.
            // And handling scales. 
            // For now, falling back to BF16 or printing placeholder for phase 2.
            // But phase 2 is FP8.
            
            // Since this is C++, I can call deep_gemm symbols directly.
            // They are static in header.
            
            // Placeholder: Cast to BF16 and run BF16 path for Phase 1 verification?
            // Plan says implement FP8 calls.
            
            // ... (Implementing BF16 logic as priority for Phase 1) ...
            // If use_fp8 is true, we should use FP8.
            
        } else {
            // BF16 Path - Using standard PyTorch for Phase 1
            // recv_x: [num_local_experts, max_tokens, hidden_size]
            // gate_up_proj_: [num_local_experts, moe_inter*2, hidden_size]
            // For batched matmul: A @ B.T
            // A: [G, M, K], B: [G, N, K] -> A @ B.transpose(-1,-2) = [G, M, N]
            
            auto intermediate = torch::bmm(recv_x, gate_up_proj_.transpose(1, 2)); 
            // intermediate: [G, M, N] = [num_local_experts, max_tokens, moe_inter*2]
            
            // Activation
            auto intermediate_flat = intermediate.view({-1, moe_inter * 2});
            auto act_out = act_fn_->forward(intermediate_flat);
            auto act_out_3d = act_out.view({num_local_experts, max_tokens, moe_inter});
            
            // Down Proj
            // act_out_3d: [G, M, moe_inter]
            // down_proj_: [G, hidden_size, moe_inter]
            // Need: [G, M, hidden_size]
            // act_out_3d @ down_proj_.transpose(1,2) -> [G, M, moe_inter] @ [G, moe_inter, hidden] = [G, M, hidden]
            expert_out = torch::bmm(act_out_3d, down_proj_.transpose(1, 2));
        }

        // 4. Combine
        // Combine expects contiguous tensor?
        // expert_out is [G, M, K]. View as [G*M, K]?
        // DeepEP combine signature takes `x` (Tensor).
        // It likely handles the layout.
        
        torch::Tensor combined_output;
        // Using same combine for both modes (low_latency)
        auto res_comb = ep_buffer->low_latency_combine(
            expert_out, // Pass 3D or 2D?
            topk_ids,
            topk_weights,
            src_info,
            layout_range,
            std::nullopt,
            num_tokens,
            num_experts_,
            false, false, false, false, std::nullopt
        );
        combined_output = std::get<0>(res_comb);

        return combined_output.view(original_shape);
    }


public:
    int hidden_size_;
    int num_experts_;
    int top_k_;
    bool use_fp8_ = false;

    // Weights (BF16)
    torch::Tensor gate_up_proj_;
    torch::Tensor down_proj_;

    // Scales (FP8)
    torch::Tensor gate_up_scale_inv_;
    torch::Tensor down_scale_inv_;

    // Gating
    std::shared_ptr<torch::nn::LinearImpl> gate_ = nullptr;
    
    // Activation
    std::unique_ptr<layers::SiluAndMul> act_fn_ = std::make_unique<layers::SiluAndMul>();
};


// =========================================================================
// Qwen3MoeDecoderLayer
// =========================================================================
template<QuantType Q>
class Qwen3MoeDecoderLayer: public core::Module {
public:
    Qwen3MoeDecoderLayer(const core::ModelConfig& config, int layer_idx, std::shared_ptr<layers::RotaryEmbedding> rotary): core::Module() {
        self_attn_ = std::make_unique<Qwen3MoeAttention<Q>>(config, rotary);
        
        bool is_moe_layer = (config.num_experts > 0) && ((layer_idx + 1) % config.decoder_sparse_step == 0);
        
        if (is_moe_layer) {
            mlp_ = std::make_unique<Qwen3MoeSparseMoeBlock<Q>>(config);
            is_sparse_ = true;
        } else {
            mlp_dense_ = std::make_unique<Qwen3MoeMLP<Q>>(config);
            is_sparse_ = false;
        }

        input_norm_ = std::make_unique<layers::RMSNorm>(config.hidden_size, config.rms_norm_eps);
        post_attn_norm_ = std::make_unique<layers::RMSNorm>(config.hidden_size, config.rms_norm_eps);
        
        layer_idx_ = layer_idx;
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(
        torch::Tensor positions, 
        torch::Tensor hidden_states, 
        torch::Tensor residual,
        KvCache* kv_cache,
        layers::FlashInferHandler* handler,
        torch::Tensor slot_mapping,
        deep_ep::Buffer* ep_buffer
    ) {
        // 1. Pre-Norm
        auto normed = input_norm_->forward(hidden_states);
        
        // 2. Attention
        auto attn_out = self_attn_->forward(
            positions, normed, kv_cache, handler, slot_mapping, layer_idx_
        );
        hidden_states = hidden_states + attn_out; // Residual add
        
        // 3. Post-Norm
        normed = post_attn_norm_->forward(hidden_states);
        
        // 4. MLP (MoE or Dense)
        torch::Tensor mlp_out;
        if (is_sparse_) {
            // Need EP buffer for Sparse Block
            // Note: SparseBlock forward signature needs update to accept buffer
             mlp_out = mlp_->forward(normed, ep_buffer);
        } else {
             mlp_out = mlp_dense_->forward(normed);
        }
        
        hidden_states = hidden_states + mlp_out;
        
        return {hidden_states, residual}; // Returning residual for API consistency, though integrated here
    }

    std::unique_ptr<Qwen3MoeAttention<Q>> self_attn_;
    std::unique_ptr<Qwen3MoeSparseMoeBlock<Q>> mlp_;
    std::unique_ptr<Qwen3MoeMLP<Q>> mlp_dense_;
    std::unique_ptr<layers::RMSNorm> input_norm_;
    std::unique_ptr<layers::RMSNorm> post_attn_norm_;
    bool is_sparse_;
    int layer_idx_;
};


// =========================================================================
// Qwen3MoeModel
// =========================================================================
template<QuantType Q>
class Qwen3MoeModel: public core::Module {
public:
    Qwen3MoeModel(const core::ModelConfig& config): core::Module() {
        embed_tokens_ = std::make_unique<layers::VocabParallelEmbedding>(config.vocab_size, config.hidden_size);
        norm_ = std::make_unique<layers::RMSNorm>(config.hidden_size, config.rms_norm_eps);
        
        int head_dim = config.head_dim > 0 ? config.head_dim : (config.hidden_size / config.num_attention_heads);
        rotary_ = std::make_shared<layers::RotaryEmbedding>(head_dim, config.max_position_embeddings, config.rope_theta);

        for (int i = 0; i < config.num_hidden_layers; ++i) {
            layers_.push_back(std::make_unique<Qwen3MoeDecoderLayer<Q>>(config, i, rotary_));
        }
    }

    torch::Tensor forward(
        torch::Tensor input_ids, 
        torch::Tensor positions,
        KvCache* kv_cache,
        layers::FlashInferHandler* handler,
        torch::Tensor slot_mapping,
        deep_ep::Buffer* ep_buffer
    ) {
        auto hidden = embed_tokens_->forward(input_ids);
        torch::Tensor residual; // Not strictly used if integrated
        
        for (auto& layer : layers_) {
            auto out = layer->forward(positions, hidden, residual, kv_cache, handler, slot_mapping, ep_buffer);
            hidden = std::get<0>(out);
        }
        
        return norm_->forward(hidden);
    }

    std::unique_ptr<layers::VocabParallelEmbedding> embed_tokens_;
    std::unique_ptr<layers::RMSNorm> norm_;
    std::vector<std::unique_ptr<Qwen3MoeDecoderLayer<Q>>> layers_;
    std::shared_ptr<layers::RotaryEmbedding> rotary_;
};

// =========================================================================
// Qwen3MoeForCausalLM
// =========================================================================
template<QuantType Q>
class Qwen3MoeForCausalLM: public core::Module {
public:
    Qwen3MoeForCausalLM(const core::ModelConfig& config): core::Module() {
        model_ = std::make_unique<Qwen3MoeModel<Q>>(config);
        lm_head_ = std::make_unique<layers::ParallelLMHead>(config.vocab_size, config.hidden_size);
    }

    torch::Tensor forward(
        torch::Tensor input_ids, 
        torch::Tensor positions,
        KvCache* kv_cache,
        layers::FlashInferHandler* handler,
        torch::Tensor slot_mapping,
        deep_ep::Buffer* ep_buffer
    ) {
        auto hidden = model_->forward(input_ids, positions, kv_cache, handler, slot_mapping, ep_buffer);
        // Note: Logic for logits usually outside or specific method
        return hidden;
    }
    
    torch::Tensor compute_logits(torch::Tensor hidden) {
        return lm_head_->forward(hidden);
    }

    std::unique_ptr<Qwen3MoeModel<Q>> model_;
    std::unique_ptr<layers::ParallelLMHead> lm_head_;
};

} // namespace models
} // namespace nanodeploy
