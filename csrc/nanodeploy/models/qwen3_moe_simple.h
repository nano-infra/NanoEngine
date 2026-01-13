#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
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
#include "nanodeploy/worker/distributed.h"
#include "nanodeploy/worker/kv_cache.h"

// DeepEP integration (only when DEEPSEEK_MOE is enabled)
#ifdef DEEPSEEK_MOE
#include "deep_ep.hpp"
#include "nanodeploy/worker/deep_ep_runner.h"
#include "nanodeploy/worker/deep_gemm_runner.h"
#include <cuda_runtime.h>
#endif

namespace nanodeploy {
namespace models {

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
// DeepSeekMoeSparseMoeBlock (Stage 1: Non-FP8, using DeepEP + PyTorch bmm)
// =========================================================================
template<QuantType Quant>
class DeepSeekMoeSparseMoeBlock: public core::Module {
public:
    DeepSeekMoeSparseMoeBlock(const core::ModelConfig& config): core::Module()
    {
        hidden_size_ = config.hidden_size;
        num_experts_ = config.num_experts;
        top_k_       = config.num_experts_per_tok;

        // Derived dimensions
        int moe_inter = config.moe_intermediate_size;

        // Gating
        gate_ = register_module("gate",
                                torch::nn::Linear(torch::nn::LinearOptions(hidden_size_, num_experts_).bias(false)));

        // Parameters for EP mode
        int world_size        = get_dist_context().ffn_ep_world_size();
        int num_local_experts = num_experts_ / world_size;

        // Weight Shapes (BF16 for Stage 1)
        // GateUp: [num_local_experts, moe_inter*2, hidden_size]
        // Down:   [num_local_experts, hidden_size, moe_inter]
        gate_up_proj_ = torch::empty({num_local_experts, moe_inter * 2, hidden_size_}, torch::kBFloat16);
        down_proj_    = torch::empty({num_local_experts, hidden_size_, moe_inter}, torch::kBFloat16);

        // Activation
        act_fn_ = std::make_unique<layers::SiluAndMul>();
    }

    torch::Tensor compute_experts(torch::Tensor input,         // [num_tokens, hidden]
                                  torch::Tensor topk_idx,      // [num_tokens, top_k] - local expert IDs
                                  torch::Tensor topk_weights,  // [num_tokens, top_k] - weights
                                  torch::Tensor gate_up_w,     // [num_local_experts, moe_inter*2, hidden]
                                  torch::Tensor down_w         // [num_local_experts, hidden, moe_inter]
    )
    {
        int  num_tokens        = input.size(0);
        auto device            = input.device();
        int  num_local_experts = gate_up_w.size(0);
        int  moe_inter_2       = gate_up_w.size(1);
        int  top_k             = topk_idx.size(1);

        if (num_tokens == 0) {
            return torch::zeros({0, hidden_size_}, input.options());
        }

        // Convert to CPU for routing logic
        auto topk_idx_cpu = topk_idx.to(torch::kCPU);
        auto topk_idx_acc = topk_idx_cpu.accessor<int64_t, 2>();

        // 1. Count tokens per expert and build scatter indices
        std::vector<std::vector<int64_t>> token_indices_per_expert(num_local_experts);
        std::vector<std::vector<int>>     token_k_per_expert(num_local_experts);

        for (int64_t t = 0; t < num_tokens; ++t) {
            for (int k = 0; k < top_k; ++k) {
                int64_t expert_id = topk_idx_acc[t][k];
                // expert_id < 0 means not assigned (e.g., -1 from DeepEP dispatch)
                if (expert_id >= 0 && expert_id < num_local_experts) {
                    token_indices_per_expert[expert_id].push_back(t);
                    token_k_per_expert[expert_id].push_back(k);
                }
            }
        }

        // 2. Compute aligned counts (128 alignment for DeepGemm)
        constexpr int64_t    ALIGNMENT = 128;
        std::vector<int64_t> actual_counts(num_local_experts);
        std::vector<int64_t> aligned_counts(num_local_experts);
        std::vector<int64_t> used_experts;
        int64_t              total_aligned_rows = 0;

        for (int e = 0; e < num_local_experts; ++e) {
            int64_t count     = token_indices_per_expert[e].size();
            actual_counts[e]  = count;
            aligned_counts[e] = (count + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
            total_aligned_rows += aligned_counts[e];
            if (count > 0) {
                used_experts.push_back(e);
            }
        }

        if (used_experts.empty() || total_aligned_rows == 0) {
            return torch::zeros({num_tokens, hidden_size_}, input.options());
        }

        // 3. Scatter: build aligned input tensor and m_indices
        auto input_bf16 = input.to(torch::kBFloat16);
        auto aligned_x  = torch::zeros({total_aligned_rows, input.size(1)}, input_bf16.options());
        auto m_indices =
            torch::full({total_aligned_rows}, -1, torch::TensorOptions().dtype(torch::kInt32).device(device));

        int64_t aligned_offset = 0;
        int     group_idx      = 0;

        for (int e = 0; e < num_local_experts; ++e) {
            int64_t count         = actual_counts[e];
            int64_t aligned_count = aligned_counts[e];

            if (count > 0) {
                // Copy tokens for this expert to aligned positions
                auto token_indices_tensor =
                    torch::from_blob(token_indices_per_expert[e].data(), {count}, torch::kLong).clone().to(device);

                aligned_x.slice(0, aligned_offset, aligned_offset + count) =
                    input_bf16.index_select(0, token_indices_tensor);

                // Set m_indices for valid rows (group index)
                m_indices.slice(0, aligned_offset, aligned_offset + count).fill_(group_idx);
                group_idx++;
            }
            aligned_offset += aligned_count;
        }

        // 4. Select weights for used experts only
        auto used_experts_tensor =
            torch::from_blob(used_experts.data(), {(long)used_experts.size()}, torch::kLong).clone().to(device);
        auto gate_up_selected = gate_up_w.index_select(0, used_experts_tensor).contiguous();
        auto down_selected    = down_w.index_select(0, used_experts_tensor).contiguous();

        // 5. GateUp GEMM
        auto gateup_out = torch::empty({total_aligned_rows, moe_inter_2}, input_bf16.options());
        DeepGemmRunner::m_grouped_bf16_gemm_nt_contiguous(aligned_x, gate_up_selected, gateup_out, m_indices);

        // 6. Activation (SiLU and Mul)
        auto act_out = act_fn_->forward(gateup_out);

        // 7. Down GEMM
        auto aligned_down_out = torch::empty({total_aligned_rows, hidden_size_}, input_bf16.options());
        DeepGemmRunner::m_grouped_bf16_gemm_nt_contiguous(act_out, down_selected, aligned_down_out, m_indices);

        // 8. Gather with weighted sum (scatter-add)
        auto output      = torch::zeros({num_tokens, hidden_size_}, input_bf16.options());
        auto weights_gpu = topk_weights.to(torch::kFloat32).to(device);

        aligned_offset = 0;
        for (int e = 0; e < num_local_experts; ++e) {
            int64_t count         = actual_counts[e];
            int64_t aligned_count = aligned_counts[e];

            if (count > 0) {
                // Get token indices and k indices for this expert
                auto token_indices =
                    torch::from_blob(token_indices_per_expert[e].data(), {count}, torch::kLong).clone().to(device);

                auto k_indices = torch::from_blob(token_k_per_expert[e].data(), {count}, torch::kInt32)
                                     .clone()
                                     .to(torch::kLong)
                                     .to(device);

                // Get weights for these token-expert pairs
                auto expert_weights = weights_gpu.index({token_indices, k_indices});  // [count]

                // Get expert outputs
                auto expert_outputs = aligned_down_out.slice(0, aligned_offset, aligned_offset + count);

                // Weight and scatter-add
                auto weighted_outputs = expert_outputs * expert_weights.unsqueeze(1).to(expert_outputs.dtype());
                output.index_add_(0, token_indices, weighted_outputs);
            }
            aligned_offset += aligned_count;
        }

        return output.to(input.dtype());
    }

    // Single-card forward: gating + compute_experts
    torch::Tensor expert(torch::Tensor hidden_states)
    {
        auto original_shape = hidden_states.sizes();
        auto hidden_flat    = hidden_states.view({-1, hidden_size_});
        auto device         = hidden_flat.device();

        // 1. Gating
        auto router_logits   = gate_->forward(hidden_flat);
        auto routing_weights = torch::softmax(router_logits, -1);
        auto topk            = torch::topk(routing_weights, top_k_, -1);
        auto topk_weights    = std::get<0>(topk);  // [num_tokens, top_k]
        auto topk_ids        = std::get<1>(topk);  // [num_tokens, top_k]

        // Normalize weights
        topk_weights = topk_weights / topk_weights.sum(-1, true);

        // Get weights on device
        auto gate_up_w = gate_up_proj_.to(device).contiguous();
        auto down_w    = down_proj_.to(device).contiguous();

        // 2. Call unified compute_experts
        auto output = compute_experts(hidden_flat, topk_ids, topk_weights, gate_up_w, down_w);

        return output.view(original_shape);
    }

    // Multi-card forward: DeepEP Normal Dispatch + compute_experts + Combine
    torch::Tensor forward_multi_card(torch::Tensor hidden_states, deep_ep::Buffer* ep_buffer)
    {
        auto original_shape = hidden_states.sizes();
        auto hidden_flat    = hidden_states.view({-1, hidden_size_});
        auto device         = hidden_flat.device();

        // 1. Gating
        auto router_logits    = gate_->forward(hidden_flat);
        auto routing_weights  = torch::softmax(router_logits, -1);
        auto topk             = torch::topk(routing_weights, top_k_, -1);
        auto topk_weights_f32 = std::get<0>(topk).to(torch::kFloat32);
        auto topk_ids         = std::get<1>(topk).to(torch::kLong);

        // Normalize weights
        topk_weights_f32 = topk_weights_f32 / topk_weights_f32.sum(-1, true);

        // 2. Dispatch (automatically selects intranode/internode)
        auto dispatch_result =
            DeepEPRunner::dispatch_normal(ep_buffer, hidden_flat, topk_ids, topk_weights_f32, num_experts_);

        // 3. Compute experts
        torch::Tensor expert_output;
        if (!dispatch_result.recv_topk_idx.has_value() || !dispatch_result.recv_topk_weights.has_value()) {
            NANODEPLOY_LOG_INFO("DeepSeekMoe: ERROR! recv_topk_idx/weights not available");
            expert_output =
                torch::zeros({dispatch_result.recv_x.size(0), hidden_size_}, dispatch_result.recv_x.options());
        }
        else {
            auto gate_up_w = gate_up_proj_.to(device).contiguous();
            auto down_w    = down_proj_.to(device).contiguous();

            expert_output = compute_experts(dispatch_result.recv_x,
                                            dispatch_result.recv_topk_idx.value(),
                                            dispatch_result.recv_topk_weights.value(),
                                            gate_up_w,
                                            down_w);
        }

        // 4. Combine (sends results back to original ranks)
        auto combined_x = DeepEPRunner::combine_normal(ep_buffer, expert_output, dispatch_result.handle);

        return combined_x.view(original_shape);
    }

    // Compute experts using masked GEMM (for Low Latency mode)
    // Input: recv_x [num_local_experts, max_m, hidden_size]
    // Output: [num_local_experts, max_m, hidden_size]
    torch::Tensor compute_experts_masked(torch::Tensor recv_x,    // [G, M, K]
                                         torch::Tensor masked_m,  // [G] - actual token counts
                                         int           expected_m,
                                         torch::Tensor gate_up_w,  // [G, intermediate*2, hidden]
                                         torch::Tensor down_w      // [G, hidden, intermediate]
    )
    {
        int  num_groups        = recv_x.size(0);
        int  max_m             = recv_x.size(1);
        int  hidden_dim        = recv_x.size(2);
        int  intermediate_size = gate_up_w.size(1) / 2;
        auto device            = recv_x.device();

        // Allocate output tensors
        auto gateup_output = torch::empty({num_groups, max_m, intermediate_size * 2},
                                          torch::TensorOptions().dtype(torch::kBFloat16).device(device));

        DeepGemmRunner::m_grouped_bf16_gemm_nt_masked(recv_x.contiguous(),
                                                      gate_up_w.contiguous(),
                                                      gateup_output,
                                                      masked_m.to(torch::kInt32).contiguous(),
                                                      expected_m,
                                                      "nk");

        auto gate_out = gateup_output.slice(2, 0, intermediate_size);
        auto up_out   = gateup_output.slice(2, intermediate_size, intermediate_size * 2);
        auto act_out  = torch::silu(gate_out) * up_out;

        auto down_output = torch::empty({num_groups, max_m, hidden_dim},
                                        torch::TensorOptions().dtype(torch::kBFloat16).device(device));

        DeepGemmRunner::m_grouped_bf16_gemm_nt_masked(act_out.contiguous(),
                                                      down_w.contiguous(),
                                                      down_output,
                                                      masked_m.to(torch::kInt32).contiguous(),
                                                      expected_m,
                                                      "nk");

        return down_output;
    }

    torch::Tensor forward_multi_card_low_latency(torch::Tensor    hidden_states,
                                                 deep_ep::Buffer* ep_buffer,
                                                 int              num_max_dispatch_tokens_per_rank)
    {
        auto original_shape = hidden_states.sizes();
        auto hidden_flat    = hidden_states.view({-1, hidden_size_});
        auto device         = hidden_flat.device();

        auto& dist_ctx   = get_dist_context();
        int   world_size = dist_ctx.ffn_ep_world_size();

        auto router_logits    = gate_->forward(hidden_flat);
        auto routing_weights  = torch::softmax(router_logits, -1);
        auto topk             = torch::topk(routing_weights, top_k_, -1);
        auto topk_weights_f32 = std::get<0>(topk).to(torch::kFloat32);
        auto topk_ids         = std::get<1>(topk).to(torch::kLong);

        // Normalize weights
        topk_weights_f32 = topk_weights_f32 / topk_weights_f32.sum(-1, true);

        // 2. Low Latency Dispatch
        auto dispatch_result = DeepEPRunner::dispatch_low_latency(
            ep_buffer, hidden_flat, topk_ids, topk_weights_f32, num_max_dispatch_tokens_per_rank, num_experts_);

        auto gate_up_w = gate_up_proj_.to(device).contiguous();
        auto down_w    = down_proj_.to(device).contiguous();

        auto expert_output = compute_experts_masked(
            dispatch_result.recv_x, dispatch_result.masked_m, dispatch_result.expected_m, gate_up_w, down_w);

        // 4. Combine (sends results back with weighted sum)
        auto combined_x = DeepEPRunner::combine_low_latency(ep_buffer, expert_output, dispatch_result.handle);

        return combined_x.view(original_shape);
    }

    torch::Tensor forward(torch::Tensor    hidden_states,
                          deep_ep::Buffer* ep_buffer               = nullptr,
                          bool             use_low_latency         = true,
                          int              num_max_dispatch_tokens = 256)
    {
        if (ep_buffer == nullptr) {
            return expert(hidden_states);
        }
        else if (use_low_latency) {
            return forward_multi_card_low_latency(hidden_states, ep_buffer, num_max_dispatch_tokens);
        }
        else {
            return forward_multi_card(hidden_states, ep_buffer);
        }
    }

    int hidden_size_;
    int num_experts_;
    int top_k_;

    // Weights (BF16 for Stage 1)
    torch::Tensor gate_up_proj_;
    torch::Tensor down_proj_;

    // Gating
    torch::nn::Linear gate_ = nullptr;

    // Activation
    std::unique_ptr<layers::SiluAndMul> act_fn_;
};

template<QuantType Quant>
class DeepSeekMoeDecoderLayer: public core::Module {
public:
    DeepSeekMoeDecoderLayer(const core::ModelConfig&                 config,
                            int                                      layer_idx,
                            std::shared_ptr<layers::RotaryEmbedding> rotary_emb):
        core::Module()
    {
        self_attn_ = std::make_unique<Qwen3MoeAttention<Quant>>(config, rotary_emb);

        bool is_moe_layer = (config.num_experts > 0) && ((layer_idx + 1) % config.decoder_sparse_step == 0);

        if (is_moe_layer) {
            mlp_moe_   = std::make_unique<DeepSeekMoeSparseMoeBlock<Quant>>(config);
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
                                                     torch::Tensor              seq_lens     = {},
                                                     deep_ep::Buffer*           ep_buffer    = nullptr)
    {
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
            // ep_buffer can be nullptr for single-card mode
            mlp_out = mlp_moe_->forward(normed, ep_buffer);
        }
        else {
            mlp_out = mlp_dense_->forward(normed);
        }

        // Residual Add
        hidden_states = hidden_states + mlp_out;

        return {hidden_states, hidden_states};
    }

    std::unique_ptr<Qwen3MoeAttention<Quant>>         self_attn_;
    std::unique_ptr<DeepSeekMoeSparseMoeBlock<Quant>> mlp_moe_;
    std::unique_ptr<Qwen3MoeMLP<Quant>>               mlp_dense_;
    std::unique_ptr<layers::RMSNorm>                  input_layernorm_;
    std::unique_ptr<layers::RMSNorm>                  post_attention_layernorm_;

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
            layers_.push_back(std::make_unique<DeepSeekMoeDecoderLayer<Quant>>(config, i, rotary_emb_));
        }

        norm_ = std::make_unique<layers::RMSNorm>(config.hidden_size, config.rms_norm_eps);
    }

    torch::Tensor forward(torch::Tensor              input_ids,
                          torch::Tensor              positions,
                          KvCache*                   kv_cache     = nullptr,
                          layers::FlashInferHandler* handler      = nullptr,
                          torch::Tensor              slot_mapping = {},
                          torch::Tensor              block_tables = {},
                          torch::Tensor              seq_lens     = {},
                          deep_ep::Buffer*           ep_buffer    = nullptr)
    {
        auto          hidden_states = embed_tokens_->forward(input_ids);
        torch::Tensor residual;

        for (size_t i = 0; i < layers_.size(); ++i) {
            auto out_tuple = layers_[i]->forward(
                positions, hidden_states, residual, kv_cache, handler, slot_mapping, block_tables, seq_lens, ep_buffer);
            hidden_states = std::get<0>(out_tuple);
            // residual = std::get<1>(out_tuple); // logic above merges it into hidden
        }

        return norm_->forward(hidden_states);
    }

    std::unique_ptr<layers::VocabParallelEmbedding>              embed_tokens_;
    std::vector<std::unique_ptr<DeepSeekMoeDecoderLayer<Quant>>> layers_;
    std::unique_ptr<layers::RMSNorm>                             norm_;
    std::shared_ptr<layers::RotaryEmbedding>                     rotary_emb_;
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
                          torch::Tensor              seq_lens     = {},
                          deep_ep::Buffer*           ep_buffer    = nullptr)
    {
        return model_->forward(
            input_ids, positions, kv_cache, handler, slot_mapping, block_tables, seq_lens, ep_buffer);
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
