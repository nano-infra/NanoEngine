"""DeepSeek V2/V3 Multi-Token Prediction (MTP) model.

MTP adds prediction layers on top of the base transformer that produce
draft tokens for speculative decoding.  Each MTP layer fuses the previous
hidden states with the embedding of the predicted token, then passes
through a full DeepseekV2DecoderLayer.

Reference: vLLM deepseek_mtp.py
"""

import torch
import torch.nn as nn

from nanodeploy.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanodeploy.layers.layernorm import RMSNorm
from nanodeploy.models.deepseek_v2.deepseek_v2 import DeepseekV2DecoderLayer
from nanodeploy.models.quant_config import QuantizationConfig


class DeepSeekMTPSharedHead(nn.Module):
    """Per-layer head: RMSNorm → ParallelLMHead."""

    def __init__(self, config):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = ParallelLMHead(config.vocab_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply norm (head projection is called separately)."""
        return self.norm(hidden_states)


class DeepSeekMTPLayer(nn.Module):
    """Single MTP prediction layer.

    Architecture per vLLM:
      1. enorm(embedding) ⊕ hnorm(hidden_states) → eh_proj → fused hidden
      2. fused hidden → DeepseekV2DecoderLayer (attention + MoE) → output
      3. shared_head for logits (norm + LMHead)
    """

    def __init__(
        self,
        config,
        quantization_config: QuantizationConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.shared_head = DeepSeekMTPSharedHead(config)
        self.mtp_block = DeepseekV2DecoderLayer(config, quantization_config, layer_idx)

    def forward(
        self,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        # Normalize both inputs
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)

        # Fuse: concat + project (2H → H)
        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )

        # Run through full decoder layer
        hidden_states, residual = self.mtp_block(
            hidden_states, positions, residual=None
        )
        hidden_states = residual + hidden_states
        return hidden_states


class DeepSeekMTP(nn.Module):
    """MTP container: shared embedding + N MTP layers.

    Each MTP layer has its own shared_head (RMSNorm + LMHead) for logits.
    The embedding layer is shared across all MTP layers.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.quantization_config = QuantizationConfig(
            **getattr(config, "quantization_config", dict())
        )
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.mtp_start_layer_idx = config.num_hidden_layers

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )

        self.layers = nn.ModuleDict(
            {
                str(idx): DeepSeekMTPLayer(config, self.quantization_config, idx)
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        layer_idx = self.mtp_start_layer_idx + (spec_step_idx % self.num_mtp_layers)
        return self.layers[str(layer_idx)](positions, hidden_states, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        layer_idx = self.mtp_start_layer_idx + (spec_step_idx % self.num_mtp_layers)
        layer = self.layers[str(layer_idx)]
        normed = layer.shared_head(hidden_states)  # RMSNorm
        return layer.shared_head.head(normed)  # ParallelLMHead

    def load_weights(self, weights):
        from .deepseek_v2_mtp_loader import load_weights

        load_weights(self, weights)
