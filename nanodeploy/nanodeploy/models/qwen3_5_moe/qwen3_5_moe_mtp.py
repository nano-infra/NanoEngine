"""Qwen3.5-MoE Multi-Token Prediction (MTP) model.

MTP adds prediction layers on top of the base transformer that produce
draft tokens for speculative decoding.  The MTP predictor fuses token
embeddings with hidden states through a shared FC projection, then passes
through full_attention decoder layers (GQA only, no GDN).

Reference: vLLM qwen3_5_mtp.py
"""

import torch
import torch.nn as nn

from nanodeploy.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanodeploy.layers.layernorm import RMSNorm
from nanodeploy.models.quant_config import QuantizationConfig
from nanodeploy.models.qwen3_5_moe.qwen3_5_moe import Qwen3_5MoeDecoderLayer


class Qwen3_5MTP(nn.Module):
    """Qwen3.5-MoE MTP predictor.

    Architecture per vLLM:
      1. embed(token) → pre_fc_norm_embedding
      2. hidden_states → pre_fc_norm_hidden
      3. concat → fc (2H → H)
      4. decoder layer (full_attention + MoE)
      5. final norm
      6. shared lm_head for logits
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.quantization_config = QuantizationConfig(
            **getattr(config, "quantization_config", dict())
        )
        self.num_mtp_layers = getattr(config, "mtp_num_hidden_layers", 1)

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )

        self.fc = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)

        self.pre_fc_norm_embedding = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, add_unit_offset=True
        )
        self.pre_fc_norm_hidden = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, add_unit_offset=True
        )

        # MTP decoder layers use full_attention type (GQA, not GDN).
        # Pass layer_idx beyond config.layer_types length so
        # Qwen3_5MoeDecoderLayer defaults to "full_attention".
        num_base_layers = getattr(config, "num_hidden_layers", 0)
        self.layers = nn.ModuleList(
            [
                Qwen3_5MoeDecoderLayer(
                    config,
                    self.quantization_config,
                    layer_idx=num_base_layers + i,
                )
                for i in range(self.num_mtp_layers)
            ]
        )

        self.norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, add_unit_offset=True
        )

        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight.data = self.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)

        # Normalize both branches
        inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
        hidden_states = self.pre_fc_norm_hidden(hidden_states)

        # Fuse: concat + project (2H → H)
        hidden_states = self.fc(torch.cat([inputs_embeds, hidden_states], dim=-1))

        # Decoder layer (cycling if multiple)
        residual = None
        layer_idx = spec_step_idx % self.num_mtp_layers
        hidden_states, residual = self.layers[layer_idx](
            positions, hidden_states, residual
        )

        # Final norm
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def load_weights(self, weights):
        from .qwen3_5_moe_mtp_loader import load_weights

        load_weights(self, weights)
