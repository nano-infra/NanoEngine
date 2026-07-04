"""Model/cache traits inferred from HuggingFace config objects."""

from __future__ import annotations

from typing import Any


def has_gdn_component(hf_config: Any) -> bool:
    """Whether a config or nested sub-config declares a GDN layer."""
    return has_layer_type(hf_config, "linear_attention")


def has_hca_csa_cache(hf_config: Any) -> bool:
    """Whether a config declares DSv4-style HCA/CSA compressed cache."""
    compress_ratios = getattr(hf_config, "compress_ratios", None) or []
    ratios = {ratio for ratio in compress_ratios if ratio > 0}
    return {4, 128}.issubset(ratios)


def has_layer_type(hf_config: Any, layer_type: str) -> bool:
    seen: set[int] = set()

    def visit(cfg: Any) -> bool:
        if cfg is None or id(cfg) in seen:
            return False
        seen.add(id(cfg))

        layer_types = getattr(cfg, "layer_types", None)
        if layer_types and any(lt == layer_type for lt in layer_types):
            return True

        sub = getattr(cfg, "sub_configs", None)
        names = list(sub.keys()) if isinstance(sub, dict) else []
        for name in ("text_config", "thinker_config", "decoder_config"):
            if name not in names:
                names.append(name)

        for name in names:
            child = getattr(cfg, name, None)
            if hasattr(child, "to_dict") or getattr(child, "layer_types", None):
                if visit(child):
                    return True
        return False

    return visit(hf_config)


__all__ = [
    "has_gdn_component",
    "has_hca_csa_cache",
    "has_layer_type",
]
