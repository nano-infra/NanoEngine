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

#include "context/deep_ep_context.h"
#include "nanodeploy/csrc/context/attention_context.h"
#include "nanodeploy/csrc/context/distributed_context.h"
#include "nanodeploy/csrc/core/common.h"
#include "nanodeploy/csrc/core/config.h"
#include "nanodeploy/csrc/core/module.h"
#include "nanodeploy/csrc/layers/activation.h"
#include "nanodeploy/csrc/layers/attention.h"
#include "nanodeploy/csrc/layers/embedding.h"
#include "nanodeploy/csrc/layers/linear.h"
#include "nanodeploy/csrc/layers/rms_norm.h"
#include "nanodeploy/csrc/layers/rotary_embedding.h"
#include "nanodeploy/csrc/logging.h"
#include "nanodeploy/csrc/ops/flashinfer_ops.h"

#include "deep_ep.hpp"
#include "nanodeploy/csrc/ops/deep_ep_ops.h"
#include "nanodeploy/csrc/ops/moe_expert_ops.h"
#include <cuda_runtime.h>

namespace nanodeploy {
namespace models {

// Helper function for FP8 quantization (for dispatch optimization)
// NOTE: This does NOT pad M dimension - DeepEP handles routing with original token count.
// Follows DLBlas per_token_group_quant_fp8 behavior.
inline std::pair<torch::Tensor, torch::Tensor> quant_fp8_for_dispatch(torch::Tensor input)
{
    constexpr int BLOCK_SIZE = 128;

    // Ensure BF16
    if (input.scalar_type() != torch::kBFloat16) {
        input = input.to(torch::kBFloat16);
    }

    int m = input.size(0);  // Original token count, NOT padded
    int k = input.size(1);

    int num_groups = k / BLOCK_SIZE;  // K_tiles

    // Allocate output - NO M padding
    auto quantized = torch::empty({m, k}, input.options().dtype(torch::kFloat8_e4m3fn));

    // Scale: [num_groups, m] then transpose VIEW (TMA layout) -> [m, num_groups]
    auto scale_base = torch::empty({num_groups, m}, input.options().dtype(torch::kFloat32));
    auto scale      = scale_base.t();  // Transpose VIEW!

    // FP8 E4M3 max value
    constexpr float fp8_max  = 448.0f;
    float           rfp8_max = 1.0f / fp8_max;

    // Vectorized processing
    auto input_reshaped = input.view({m, num_groups, BLOCK_SIZE});
    auto abs_max        = input_reshaped.abs().amax(-1);
    abs_max             = abs_max.clamp_min(1e-6f);

    scale.copy_(abs_max * rfp8_max);

    auto scale_expanded = abs_max.unsqueeze(-1);
    auto scaled         = input_reshaped / scale_expanded * fp8_max;
    auto clamped        = scaled.clamp(-fp8_max, fp8_max);
    quantized.copy_(clamped.view({m, k}));

    return {quantized.to(torch::kFloat8_e4m3fn), scale};
}

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
                                                                       /*bias=*/false);

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

    torch::Tensor forward(torch::Tensor       positions,
                          torch::Tensor       hidden_states,
                          KvCache*            kv_cache     = nullptr,
                          ops::FlashInferOps* handler      = nullptr,
                          torch::Tensor       slot_mapping = {},
                          int                 layer_idx    = -1,
                          torch::Tensor       block_tables = {},
                          torch::Tensor       seq_lens     = {})
    {
        NANODEPLOY_LOG_DEBUG("Qwen3MoeAttention forward");
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
                NANODEPLOY_LOG_DEBUG("Qwen3MoeAttention use_flashinfer_decode");
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
                // FlashInfer Prefill Path - Use Ragged (direct K/V) instead of PagedKV
                NANODEPLOY_LOG_DEBUG("Qwen3MoeAttention prefill using FlashInfer Ragged");

                // q is [batch, heads, seq, dim]. Flatten to [total_tokens, heads, dim]
                auto q_fi = q.transpose(1, 2).contiguous().view({-1, num_heads_, head_dim_});
                // k, v are [batch, kv_heads, seq, dim]. Flatten to [total_tokens, kv_heads, dim]
                auto k_fi = k.transpose(1, 2).contiguous().view({-1, num_kv_heads_, head_dim_});
                auto v_fi = v.transpose(1, 2).contiguous().view({-1, num_kv_heads_, head_dim_});

                // FlashInfer expects BF16
                if (q_fi.scalar_type() != torch::kBFloat16) {
                    q_fi = q_fi.to(torch::kBFloat16);
                }
                if (k_fi.scalar_type() != torch::kBFloat16) {
                    k_fi = k_fi.to(torch::kBFloat16);
                }
                if (v_fi.scalar_type() != torch::kBFloat16) {
                    v_fi = v_fi.to(torch::kBFloat16);
                }

                // Build q_indptr and kv_indptr for variable length support
                // For batch=1: [0, seq_len]
                auto q_indptr  = torch::tensor({0, static_cast<int32_t>(seq_len)},
                                              torch::TensorOptions().dtype(torch::kInt32).device(q.device()));
                auto kv_indptr = torch::tensor({0, static_cast<int32_t>(seq_len)},
                                               torch::TensorOptions().dtype(torch::kInt32).device(k.device()));

                int total_q_tokens  = q_fi.size(0);
                int total_kv_tokens = k_fi.size(0);

                // Call ragged prefill - directly use K/V tensors
                auto output_fi = handler->prefill_ragged(q_fi.data_ptr(),
                                                         k_fi.data_ptr(),
                                                         v_fi.data_ptr(),
                                                         total_q_tokens,
                                                         total_kv_tokens,
                                                         static_cast<int32_t*>(q_indptr.data_ptr()),
                                                         static_cast<int32_t*>(kv_indptr.data_ptr()),
                                                         batch  // batch_size
                );

                // FlashInfer returns [total_tokens, heads, dim].
                // Reshape back to [batch, heads, seq, dim] consistent with model layout
                attn_output = output_fi.view({batch, seq_len, num_heads_, head_dim_}).transpose(1, 2);
            }
        }
        else {
            NANODEPLOY_ASSERT("false", "Stateless attention is not supported");
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

    // Compute experts using contiguous grouped GEMM (for normal dispatch)
    torch::Tensor compute_experts(torch::Tensor input,         // [num_tokens, hidden]
                                  torch::Tensor topk_idx,      // [num_tokens, top_k] - local expert IDs
                                  torch::Tensor topk_weights,  // [num_tokens, top_k] - weights
                                  torch::Tensor gate_up_w,     // [num_local_experts, moe_inter*2, hidden]
                                  torch::Tensor down_w         // [num_local_experts, hidden, moe_inter]
    )
    {
        return ops::MoeExpertOps::compute_contiguous(input, topk_idx, topk_weights, gate_up_w, down_w, hidden_size_);
    }

    // Single-card forward: gating + compute_experts
    torch::Tensor expert(torch::Tensor hidden_states)
    {
        auto original_shape = hidden_states.sizes();
        auto hidden_flat    = hidden_states.view({-1, hidden_size_});
        auto device         = hidden_flat.device();

        // 1. Gating
        auto router_logits   = gate_->forward(hidden_flat.to(torch::kFloat32));
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
    torch::Tensor forward_multi_card(torch::Tensor hidden_states)
    {
        auto original_shape = hidden_states.sizes();
        auto hidden_flat    = hidden_states.view({-1, hidden_size_});
        auto device         = hidden_flat.device();

        // 1. Gating
        auto router_logits    = gate_->forward(hidden_flat.to(torch::kFloat32));
        auto routing_weights  = torch::softmax(router_logits, -1);
        auto topk             = torch::topk(routing_weights, top_k_, -1);
        auto topk_weights_f32 = std::get<0>(topk).to(torch::kFloat32);
        auto topk_ids         = std::get<1>(topk).to(torch::kLong);

        // Normalize weights
        topk_weights_f32 = topk_weights_f32 / topk_weights_f32.sum(-1, true);

        // 2. Dispatch (automatically selects intranode/internode)
        auto dispatch_result = ops::DeepEpOps::dispatch_normal(hidden_flat, topk_ids, topk_weights_f32, num_experts_);

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
        auto combined_x = ops::DeepEpOps::combine_normal(expert_output, dispatch_result.handle);

        return combined_x.view(original_shape);
    }

    // Compute experts using masked GEMM (for Low Latency mode)
    torch::Tensor compute_experts_masked(torch::Tensor recv_x,    // [G, M, K]
                                         torch::Tensor masked_m,  // [G] - actual token counts
                                         int           expected_m,
                                         torch::Tensor gate_up_w,  // [G, intermediate*2, hidden]
                                         torch::Tensor down_w      // [G, hidden, intermediate]
    )
    {
        return ops::MoeExpertOps::compute_masked(recv_x, masked_m, expected_m, gate_up_w, down_w);
    }

    torch::Tensor forward_multi_card_low_latency(torch::Tensor hidden_states, int num_max_dispatch_tokens_per_rank)
    {
        auto original_shape = hidden_states.sizes();
        auto hidden_flat    = hidden_states.view({-1, hidden_size_});
        auto device         = hidden_flat.device();

        auto& dist_ctx   = get_dist_context();
        int   world_size = dist_ctx.ffn_ep_world_size();

        auto router_logits    = gate_->forward(hidden_flat.to(torch::kFloat32));
        auto routing_weights  = torch::softmax(router_logits, -1);
        auto topk             = torch::topk(routing_weights, top_k_, -1);
        auto topk_weights_f32 = std::get<0>(topk).to(torch::kFloat32);
        auto topk_ids         = std::get<1>(topk).to(torch::kLong);

        // Normalize weights
        topk_weights_f32 = topk_weights_f32 / topk_weights_f32.sum(-1, true);

        // 2. Low Latency Dispatch
        auto dispatch_result = ops::DeepEpOps::dispatch_low_latency(
            hidden_flat, topk_ids, topk_weights_f32, num_max_dispatch_tokens_per_rank, num_experts_);

        auto gate_up_w = gate_up_proj_.to(device).contiguous();
        auto down_w    = down_proj_.to(device).contiguous();

        auto expert_output = compute_experts_masked(
            dispatch_result.recv_x, dispatch_result.masked_m, dispatch_result.expected_m, gate_up_w, down_w);

        // 4. Combine (sends results back with weighted sum)
        auto combined_x = ops::DeepEpOps::combine_low_latency(expert_output, dispatch_result.handle);

        return combined_x.view(original_shape);
    }

    torch::Tensor forward(torch::Tensor hidden_states, bool use_low_latency = true, int num_max_dispatch_tokens = 256)
    {
        if (!get_deep_ep_context().get_buffer()) {
            return expert(hidden_states);
        }
        else if (use_low_latency) {
            return forward_multi_card_low_latency(hidden_states, num_max_dispatch_tokens);
        }
        else {
            return forward_multi_card(hidden_states);
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

// =========================================================================
// DeepSeekMoeSparseMoeBlock (Stage 2: FP8 specialization)
// =========================================================================
template<>
class DeepSeekMoeSparseMoeBlock<QuantType::FP8_E4M3>: public core::Module {
public:
    static constexpr int BLOCK_SIZE = 128;

    DeepSeekMoeSparseMoeBlock(const core::ModelConfig& config): core::Module()
    {
        hidden_size_ = config.hidden_size;
        num_experts_ = config.num_experts;
        top_k_       = config.num_experts_per_tok;

        int moe_inter = config.moe_intermediate_size;

        gate_ = register_module("gate",
                                torch::nn::Linear(torch::nn::LinearOptions(hidden_size_, num_experts_).bias(false)));

        int world_size        = get_dist_context().ffn_ep_world_size();
        int num_local_experts = num_experts_ / world_size;

        gate_up_proj_ = torch::empty({num_local_experts, moe_inter * 2, hidden_size_}, torch::kFloat8_e4m3fn);
        down_proj_    = torch::empty({num_local_experts, hidden_size_, moe_inter}, torch::kFloat8_e4m3fn);

        int gu_out_blocks  = (moe_inter * 2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int gu_in_blocks   = (hidden_size_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
        gate_up_scale_inv_ = torch::empty({num_local_experts, gu_out_blocks, gu_in_blocks}, torch::kFloat32);

        int d_out_blocks = (hidden_size_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int d_in_blocks  = (moe_inter + BLOCK_SIZE - 1) / BLOCK_SIZE;
        down_scale_inv_  = torch::empty({num_local_experts, d_out_blocks, d_in_blocks}, torch::kFloat32);

        register_parameter("gate_up_proj", gate_up_proj_);
        register_parameter("down_proj", down_proj_);
        register_parameter("gate_up_scale_inv", gate_up_scale_inv_);
        register_parameter("down_scale_inv", down_scale_inv_);
    }

    torch::Tensor compute_experts(torch::Tensor input, torch::Tensor topk_idx, torch::Tensor topk_weights)
    {
        return ops::MoeExpertOps::compute_contiguous(input,
                                                     topk_idx,
                                                     topk_weights,
                                                     {gate_up_proj_, gate_up_scale_inv_},
                                                     {down_proj_, down_scale_inv_},
                                                     hidden_size_);
    }

    // FP8 version for pre-quantized input from dispatch
    torch::Tensor compute_experts_fp8(torch::Tensor input_fp8,
                                      torch::Tensor input_scales,
                                      torch::Tensor topk_idx,
                                      torch::Tensor topk_weights)
    {
        return ops::MoeExpertOps::compute_contiguous_fp8(input_fp8,
                                                         input_scales,
                                                         topk_idx,
                                                         topk_weights,
                                                         {gate_up_proj_, gate_up_scale_inv_},
                                                         {down_proj_, down_scale_inv_},
                                                         hidden_size_);
    }

    torch::Tensor compute_experts_masked(torch::Tensor recv_x, torch::Tensor masked_m, int expected_m)
    {
        return ops::MoeExpertOps::compute_masked(
            recv_x, masked_m, expected_m, {gate_up_proj_, gate_up_scale_inv_}, {down_proj_, down_scale_inv_});
    }

    torch::Tensor expert(torch::Tensor hidden_states)
    {
        auto original_shape = hidden_states.sizes();
        auto hidden_flat    = hidden_states.view({-1, hidden_size_});

        auto router_logits   = gate_->forward(hidden_flat.to(torch::kFloat32));
        auto routing_weights = torch::softmax(router_logits, -1);
        auto topk            = torch::topk(routing_weights, top_k_, -1);
        auto topk_weights    = std::get<0>(topk);
        auto topk_ids        = std::get<1>(topk);

        topk_weights = topk_weights / topk_weights.sum(-1, true);

        auto output = compute_experts(hidden_flat, topk_ids, topk_weights);

        return output.view(original_shape);
    }

    torch::Tensor forward_multi_card(torch::Tensor hidden_states)
    {
        auto original_shape = hidden_states.sizes();
        auto hidden_flat    = hidden_states.view({-1, hidden_size_});

        auto router_logits    = gate_->forward(hidden_flat.to(torch::kFloat32));
        auto routing_weights  = torch::softmax(router_logits, -1);
        auto topk             = torch::topk(routing_weights, top_k_, -1);
        auto topk_weights_f32 = std::get<0>(topk).to(torch::kFloat32);
        auto topk_ids         = std::get<1>(topk).to(torch::kLong);

        topk_weights_f32 = topk_weights_f32 / topk_weights_f32.sum(-1, true);

        // Quantize to FP8 before dispatch (reduces bandwidth by 50%)
        auto [hidden_fp8, hidden_scales] = quant_fp8_for_dispatch(hidden_flat);

        // FP8 dispatch
        auto dispatch_result =
            ops::DeepEpOps::dispatch_normal_fp8(hidden_fp8, hidden_scales, topk_ids, topk_weights_f32, num_experts_);

        torch::Tensor expert_output;
        if (!dispatch_result.recv_topk_idx.has_value() || !dispatch_result.recv_topk_weights.has_value()) {
            expert_output =
                torch::zeros({dispatch_result.recv_x.size(0), hidden_size_}, dispatch_result.recv_x.options());
        }
        else {
            // Use FP8 expert compute (skips internal quantization)
            expert_output = compute_experts_fp8(dispatch_result.recv_x,
                                                dispatch_result.recv_x_scales,
                                                dispatch_result.recv_topk_idx.value(),
                                                dispatch_result.recv_topk_weights.value());
        }

        auto combined_x = ops::DeepEpOps::combine_normal(expert_output, dispatch_result.handle);
        return combined_x.view(original_shape);
    }

    torch::Tensor forward_multi_card_low_latency(torch::Tensor hidden_states, int num_max_dispatch_tokens_per_rank)
    {
        auto original_shape = hidden_states.sizes();
        auto hidden_flat    = hidden_states.view({-1, hidden_size_});
        auto device         = hidden_flat.device();

        auto router_logits    = gate_->forward(hidden_flat.to(torch::kFloat32));
        auto routing_weights  = torch::softmax(router_logits, -1);
        auto topk             = torch::topk(routing_weights, top_k_, -1);
        auto topk_weights_f32 = std::get<0>(topk).to(torch::kFloat32);
        auto topk_ids         = std::get<1>(topk).to(torch::kLong);

        topk_weights_f32 = topk_weights_f32 / topk_weights_f32.sum(-1, true);

        auto dispatch_result = ops::DeepEpOps::dispatch_low_latency(
            hidden_flat, topk_ids, topk_weights_f32, num_max_dispatch_tokens_per_rank, num_experts_);

        auto expert_output =
            compute_experts_masked(dispatch_result.recv_x, dispatch_result.masked_m, dispatch_result.expected_m);

        auto combined_x = ops::DeepEpOps::combine_low_latency(expert_output, dispatch_result.handle);
        return combined_x.view(original_shape);
    }

    torch::Tensor forward(torch::Tensor hidden_states, bool use_low_latency = true, int num_max_dispatch_tokens = 256)
    {
        if (!get_deep_ep_context().get_buffer()) {
            return expert(hidden_states);
        }
        else if (use_low_latency) {
            return forward_multi_card_low_latency(hidden_states, num_max_dispatch_tokens);
        }
        else {
            return forward_multi_card(hidden_states);
        }
    }

    int               hidden_size_;
    int               num_experts_;
    int               top_k_;
    torch::nn::Linear gate_ = nullptr;

    torch::Tensor gate_up_proj_;
    torch::Tensor down_proj_;
    torch::Tensor gate_up_scale_inv_;
    torch::Tensor down_scale_inv_;
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

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor       positions,
                                                     torch::Tensor       hidden_states,
                                                     torch::Tensor       residual,
                                                     KvCache*            kv_cache     = nullptr,
                                                     ops::FlashInferOps* handler      = nullptr,
                                                     torch::Tensor       slot_mapping = {},
                                                     torch::Tensor       block_tables = {},
                                                     torch::Tensor       seq_lens     = {},
                                                     bool                is_prefill   = false)
    {
        // === Align with Python's Fused AddRMSNorm pattern ===
        // Python logic:
        //   if residual is None:
        //       residual = hidden_states
        //       hidden_states = input_layernorm(hidden_states)
        //   else:
        //       hidden_states, residual = input_layernorm(hidden_states, residual)
        //       where add_rms_forward does: x = hidden_states + residual, then norm

        torch::Tensor normed;
        if (!residual.defined()) {
            // First layer: residual = original input
            residual = hidden_states;
            normed   = input_layernorm_->forward(hidden_states);
        }
        else {
            // Subsequent layers: fused add + norm
            // hidden_states here is the MLP output from previous layer
            // residual is the accumulated residual from previous layer
            auto fused = hidden_states + residual;
            residual   = fused;  // Update residual for next layer
            normed     = input_layernorm_->forward(fused);
        }

        // 2. Attention
        auto attn_out =
            self_attn_->forward(positions, normed, kv_cache, handler, slot_mapping, layer_idx_, block_tables, seq_lens);

        // 3. Post-Attention Fused Add + Norm (like Python's add_rms_forward)
        // Python: hidden_states, residual = post_attention_layernorm(attn_out, residual)
        // where add_rms_forward does: x = attn_out + residual, then norm
        auto fused_post = attn_out + residual;
        residual        = fused_post;  // Update residual for MLP residual add

        normed = post_attention_layernorm_->forward(fused_post);

        // 4. MLP
        torch::Tensor mlp_out;
        if (is_sparse_) {
            bool use_low_latency = !is_prefill;
            mlp_out              = mlp_moe_->forward(normed, use_low_latency);
        }
        else {
            mlp_out = mlp_dense_->forward(normed);
        }

        // Return MLP output and residual for next layer
        // Note: The final residual add (mlp_out + residual) happens in the NEXT layer's input_layernorm
        return {mlp_out, residual};
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
        embed_tokens_ = std::make_unique<layers::VocabParallelEmbedding>(config.vocab_size, config.hidden_size, device);

        int head_dim = config.head_dim > 0 ? config.head_dim : (config.hidden_size / config.num_attention_heads);

        rotary_emb_ = std::make_shared<layers::RotaryEmbedding>(
            head_dim, config.max_position_embeddings, config.rope_theta, device);

        for (int i = 0; i < config.num_hidden_layers; ++i) {
            layers_.push_back(std::make_unique<DeepSeekMoeDecoderLayer<Quant>>(config, i, rotary_emb_));
        }

        norm_ = std::make_unique<layers::RMSNorm>(config.hidden_size, config.rms_norm_eps);
    }

    torch::Tensor forward(torch::Tensor       input_ids,
                          torch::Tensor       positions,
                          KvCache*            kv_cache     = nullptr,
                          ops::FlashInferOps* handler      = nullptr,
                          torch::Tensor       slot_mapping = {},
                          torch::Tensor       block_tables = {},
                          torch::Tensor       seq_lens     = {},
                          bool                is_prefill   = false)
    {
        auto hidden_states = embed_tokens_->forward(input_ids);

        // Note: Attention layer handles both 2D [seq_len, hidden] and 3D [batch, seq_len, hidden] inputs
        // For 2D input, it infers batch=1, seq_len=sizes[0]
        torch::Tensor residual;

        NANODEPLOY_LOG_DEBUG("Qwen3MoeModel layers_.size(): ", layers_.size());
        for (size_t i = 0; i < layers_.size(); ++i) {
            NANODEPLOY_LOG_DEBUG("Qwen3MoeModel layer ", i, " forward");
            std::flush(std::cout);
            std::flush(std::cerr);
            std::flush(std::clog);
            auto out_tuple = layers_[i]->forward(positions,
                                                 hidden_states,
                                                 residual,
                                                 kv_cache,
                                                 handler,
                                                 slot_mapping,
                                                 block_tables,
                                                 seq_lens,
                                                 is_prefill);
            hidden_states  = std::get<0>(out_tuple);
            residual       = std::get<1>(out_tuple);  // Track residual for final norm
        }

        // Final Fused Add + Norm (like Python's self.norm(hidden_states, residual))
        auto final_fused = hidden_states + residual;
        return norm_->forward(final_fused);
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
        lm_head_ = std::make_unique<layers::ParallelLMHead>(config.vocab_size, config.hidden_size, device);
    }

    torch::Tensor forward(torch::Tensor       input_ids,
                          torch::Tensor       positions,
                          KvCache*            kv_cache     = nullptr,
                          ops::FlashInferOps* handler      = nullptr,
                          torch::Tensor       slot_mapping = {},
                          torch::Tensor       block_tables = {},
                          torch::Tensor       seq_lens     = {},
                          bool                is_prefill   = false)
    {
        return model_->forward(
            input_ids, positions, kv_cache, handler, slot_mapping, block_tables, seq_lens, is_prefill);
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
