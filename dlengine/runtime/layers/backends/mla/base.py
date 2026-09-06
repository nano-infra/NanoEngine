"""MLA (Multi-head Latent Attention) family base.

MLA is a dense attention type with a compressed KV representation and absorbed
projections; it is distinct from GQA. This family owns the MLA *decode* kernels
(FlashMLA on Hopper, TRTLLM-GEN on Blackwell, and a reference path). MLA prefill
currently lives in ``DeepseekV2Attention.forward`` and is pulled into the family
in the DSA stage, which reuses the same prefill path.

``MlaAttentionBase`` is the shared contract; concrete backends set ``k_cache`` /
``v_cache`` (injected by ``ModelRunner``) and implement ``forward``.
"""

import torch

from dlengine.runtime.layers.base_backend import AttentionBase


class MlaAttentionBase(AttentionBase):
    """Contract for dense MLA attention backends."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        v_head_dim: int,
        attention_type: str = "MLA",
        nsa_index_topk: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        if attention_type != "MLA":
            raise ValueError(
                f"{type(self).__name__} requires MLA attention, got {attention_type!r}."
            )
        if num_kv_heads != 1:
            raise ValueError(
                f"MLA requires one compressed KV head, got {num_kv_heads}."
            )
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.v_head_dim = v_head_dim
        self.nsa_index_topk = nsa_index_topk
        # Cache tensors are injected by ModelRunner after construction.
        self.k_cache = self.v_cache = torch.tensor([])
        self.hisparse_k_cache = self.hisparse_v_cache = torch.tensor([])


__all__ = ["MlaAttentionBase"]
