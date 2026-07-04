from dataclasses import dataclass, field

import torch

from dlengine.context_v2 import BaseContext
from dlengine.context_v2.cache.hca import DSV4_BYTES_PER_TOKEN
from dlengine.logging import get_logger

logger = get_logger("dlengine")

DSV4_COMPRESSED_PAGE_SIZE = 2


@dataclass
class CSAContext(BaseContext):
    dsv4_compress_ratios: list[int] | None = None
    dsv4_compressed_caches: dict[int, torch.Tensor] | None = None
    dsv4_compressed_caches_flat: dict[int, torch.Tensor] = field(default_factory=dict)
    dsv4_layers_per_ratio: dict[int, list[int]] = field(default_factory=dict)
    dsv4_layer_to_ratio_idx: dict[int, int] = field(default_factory=dict)
    dsv4_compressed_pool_config: dict[int, tuple[int, int, int]] = field(
        default_factory=dict
    )
    dsv4_compressed_dummy_page: dict[int, int] = field(default_factory=dict)
    dsv4_compressor_kv_flat: dict[int, torch.Tensor] = field(default_factory=dict)
    dsv4_compressor_score_flat: dict[int, torch.Tensor] = field(default_factory=dict)
    dsv4_compressor_counts_flat: dict[int, torch.Tensor] = field(default_factory=dict)

    @classmethod
    def get_context_type(cls) -> str:
        return "csa"

    @classmethod
    def get_context_name(cls) -> str:
        return "CSAContext"

    def clear_context(self) -> None:
        self.dsv4_compress_ratios = None
        self.dsv4_compressed_caches = None
        self.dsv4_compressed_caches_flat.clear()
        self.dsv4_layers_per_ratio.clear()
        self.dsv4_layer_to_ratio_idx.clear()
        self.dsv4_compressed_pool_config.clear()
        self.dsv4_compressed_dummy_page.clear()
        self.dsv4_compressor_kv_flat.clear()
        self.dsv4_compressor_score_flat.clear()
        self.dsv4_compressor_counts_flat.clear()

    def reset_context(self) -> None:
        self.clear_context()


_CSA_CONTEXT = CSAContext()


def get_csa_context() -> CSAContext:
    return _CSA_CONTEXT


def reset_csa_context() -> None:
    global _CSA_CONTEXT
    _CSA_CONTEXT = CSAContext()


def allocate_dsv4_compressed_caches(
    context,
    compress_ratios: list[int],
    max_num_seqs: int,
    max_model_len: int,
    pool_pages_per_ratio: dict[int, int] | None = None,
) -> None:
    """Allocate DSv4 CSA compressed KV caches (paged shared pool)."""
    context.dsv4_compress_ratios = compress_ratios
    context.dsv4_compressed_caches = {}
    context.dsv4_compressed_caches_flat = {}
    context.dsv4_layers_per_ratio = {}
    context.dsv4_layer_to_ratio_idx = {}
    context.dsv4_compressed_pool_config = {}
    context.dsv4_compressed_dummy_page = {}
    pool_pages_per_ratio = pool_pages_per_ratio or {}

    total_bytes = 0
    unique_ratios = sorted({r for r in compress_ratios if r > 0})
    for ratio in unique_ratios:
        max_compressed = (max_model_len // ratio + 63) // 64 * 64
        max_blocks_per_seq = (
            max_compressed + DSV4_COMPRESSED_PAGE_SIZE - 1
        ) // DSV4_COMPRESSED_PAGE_SIZE
        worst_case_pages = max_num_seqs * max_blocks_per_seq
        override = pool_pages_per_ratio.get(ratio, 0)
        num_pages = override if override > 0 else worst_case_pages
        context.dsv4_compressed_pool_config[ratio] = (
            num_pages,
            DSV4_COMPRESSED_PAGE_SIZE,
            max_blocks_per_seq,
        )
        context.dsv4_compressed_dummy_page[ratio] = num_pages

        layers_for_ratio = [
            i for i, layer_ratio in enumerate(compress_ratios) if layer_ratio == ratio
        ]
        context.dsv4_layers_per_ratio[ratio] = layers_for_ratio
        n_layers = len(layers_for_ratio)
        flat = torch.zeros(
            n_layers,
            num_pages + 1,
            DSV4_COMPRESSED_PAGE_SIZE,
            1,
            DSV4_BYTES_PER_TOKEN,
            dtype=torch.uint8,
            device=context.device,
        )
        context.dsv4_compressed_caches_flat[ratio] = flat
        total_bytes += flat.nelement()

        for ratio_layer_idx, layer_idx in enumerate(layers_for_ratio):
            context.dsv4_compressed_caches[layer_idx] = flat[ratio_layer_idx]
            context.dsv4_layer_to_ratio_idx[layer_idx] = ratio_layer_idx

    if total_bytes > 0:
        sizes_str = ", ".join(
            f"ratio={r}: {p[0]} pages × {p[1]} tok × "
            f"{len(context.dsv4_layers_per_ratio[r])} layers"
            for r, p in context.dsv4_compressed_pool_config.items()
        )
        logger.info(
            f"DSv4 CSA compressed caches (flat per ratio): "
            f"{len(context.dsv4_compressed_caches)} layer views, "
            f"total {total_bytes / 1e9:.2f} GB ({sizes_str})"
        )


def allocate_dsv4_compressor_state(
    context,
    compress_ratios: list[int],
    head_dim: int,
    max_num_seqs: int,
) -> None:
    context.dsv4_compressor_kv_flat = {}
    context.dsv4_compressor_score_flat = {}
    context.dsv4_compressor_counts_flat = {}
    if not getattr(context, "dsv4_layers_per_ratio", None):
        return
    total_bytes = 0
    for ratio, layers in context.dsv4_layers_per_ratio.items():
        n_layers = len(layers)
        coeff = 2 if ratio == 4 else 1
        kv_buf = torch.zeros(
            n_layers,
            max_num_seqs + 1,
            coeff * ratio,
            coeff * head_dim,
            dtype=torch.float32,
            device=context.device,
        )
        score_buf = torch.full(
            (n_layers, max_num_seqs + 1, coeff * ratio, coeff * head_dim),
            float("-inf"),
            dtype=torch.float32,
            device=context.device,
        )
        counts_buf = torch.zeros(
            n_layers,
            max_num_seqs + 1,
            dtype=torch.int32,
            device=context.device,
        )
        context.dsv4_compressor_kv_flat[ratio] = kv_buf
        context.dsv4_compressor_score_flat[ratio] = score_buf
        context.dsv4_compressor_counts_flat[ratio] = counts_buf
        total_bytes += (
            kv_buf.nelement() * kv_buf.element_size()
            + score_buf.nelement() * score_buf.element_size()
            + counts_buf.nelement() * counts_buf.element_size()
        )
    if total_bytes > 0:
        logger.info(
            f"DSv4 CSA compressor scratch state (flat per ratio): "
            f"{total_bytes / 1e9:.3f} GB"
        )


__all__ = [
    "CSAContext",
    "DSV4_COMPRESSED_PAGE_SIZE",
    "allocate_dsv4_compressed_caches",
    "allocate_dsv4_compressor_state",
    "get_csa_context",
    "reset_csa_context",
]
