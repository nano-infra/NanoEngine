"""DeepSeek V2/V3 Multi-Token Prediction (MTP) model.

MTP adds prediction layers on top of the base transformer that produce
draft tokens for speculative decoding.  Each MTP layer fuses the previous
hidden states with the embedding of the predicted token, then passes
through a full DeepseekV2DecoderLayer.

Reference: vLLM deepseek_mtp.py
"""

import fnmatch

import torch
import torch.nn as nn

from dlengine.runtime.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from dlengine.runtime.layers.layernorm import RMSNorm
from dlengine.runtime.models.deepseek_v2.deepseek_v2 import (
    _IndexerTopKState,
    DeepseekV2DecoderLayer,
)
from dlengine.runtime.models.quant_config import QuantizationConfig
from dlengine.runtime.models.pp_utils import get_pp_layer_range


def _mtp_layer_quantization_config(config, layer_idx: int) -> QuantizationConfig:
    """Resolve predictor quantization using the checkpoint's ignore patterns."""
    raw_config = getattr(config, "quantization_config", None) or {}
    layer_name = f"model.layers.{layer_idx}"
    ignored = any(
        fnmatch.fnmatchcase(layer_name, pattern)
        or fnmatch.fnmatchcase(f"{layer_name}.", pattern)
        for pattern in raw_config.get("ignore", ())
    )
    return QuantizationConfig() if ignored else QuantizationConfig(**raw_config)


class DeepSeekMTPSharedHead(nn.Module):
    """Per-layer head: RMSNorm → ParallelLMHead."""

    def __init__(self, config):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = ParallelLMHead(config.vocab_size, config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Produce the normalized hidden state consumed by logits and recurrence."""
        if residual is None:
            return self.norm(hidden_states)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


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
        cache_layer_idx: int | None = None,
    ):
        super().__init__()
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.shared_head = DeepSeekMTPSharedHead(config)
        self.mtp_block = DeepseekV2DecoderLayer(
            config,
            quantization_config,
            layer_idx,
            cache_layer_idx=cache_layer_idx,
        )

    def forward(
        self,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        indexer_state: _IndexerTopKState | None = None,
        reuse_indexer_topk: bool = False,
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
            hidden_states,
            positions,
            residual=None,
            indexer_state=indexer_state,
            reuse_indexer_topk=reuse_indexer_topk,
        )
        return self.shared_head(hidden_states, residual)


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
        target_start, target_end = get_pp_layer_range(config.num_hidden_layers)
        # Predictor cache follows the target layers local to its PP owner.
        self.mtp_cache_start_idx = target_end - target_start

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )

        self.layers = nn.ModuleDict(
            {
                str(idx): DeepSeekMTPLayer(
                    config,
                    _mtp_layer_quantization_config(config, idx),
                    idx,
                    cache_layer_idx=self.mtp_cache_start_idx
                    + idx
                    - self.mtp_start_layer_idx,
                )
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
        indexer_state: _IndexerTopKState | None = None,
        reuse_indexer_topk: bool = False,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        layer_idx = self.mtp_start_layer_idx + (spec_step_idx % self.num_mtp_layers)
        return self.layers[str(layer_idx)](
            positions,
            hidden_states,
            inputs_embeds,
            indexer_state=indexer_state,
            reuse_indexer_topk=reuse_indexer_topk,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        layer_idx = self.mtp_start_layer_idx + (spec_step_idx % self.num_mtp_layers)
        layer = self.layers[str(layer_idx)]
        # forward already returned shared_head.norm output. Reapplying the
        # learned RMSNorm here would distort both logits and recurrent hidden.
        head = layer.shared_head.head
        forward_all_rows = getattr(head, "forward_all_rows", None)
        if forward_all_rows is not None:
            return forward_all_rows(hidden_states)
        return head(hidden_states)

    def load_weights(self, weights):
        from .deepseek_v2_mtp_loader import load_weights

        load_weights(self, weights)
