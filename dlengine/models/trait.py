"""Model/cache traits inferred from HuggingFace config objects."""

from __future__ import annotations

from typing import Any

from dlengine.logging import get_logger

logger = get_logger("dlengine")


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


def apply_hf_config_compatibility_fixes(hf_config: Any, raw_config: dict) -> None:
    """Repair model dimensions clobbered by Hugging Face config aliases."""
    if raw_config.get("model_type") != "glm_moe_dsa":
        return

    raw_rope_dim = raw_config.get("qk_rope_head_dim")
    if raw_rope_dim is None:
        return

    # Transformers 5.12 maps generic ``head_dim`` to ``qk_rope_head_dim``.
    # GLM-5.2 carries both (192 and 64), so the alias overwrites the latter and
    # constructs a 512+192=704 projection for a 512+64=576 checkpoint weight.
    raw_rope_dim = int(raw_rope_dim)
    parsed_rope_dim = int(getattr(hf_config, "qk_rope_head_dim", raw_rope_dim))
    if parsed_rope_dim != raw_rope_dim:
        logger.warning(
            "Restoring GLM DSA qk_rope_head_dim=%s from config.json "
            "(Transformers parsed %s via the head_dim alias)",
            raw_rope_dim,
            parsed_rope_dim,
        )
    hf_config.qk_rope_head_dim = raw_rope_dim

    qk_nope_dim = getattr(hf_config, "qk_nope_head_dim", None)
    if qk_nope_dim is not None:
        hf_config.qk_head_dim = int(qk_nope_dim) + raw_rope_dim


def resolve_eos_token_ids(model: str, tokenizer: Any) -> list[int]:
    """Resolve all EOS token ids declared by tokenizer and generation config."""
    eos_ids: set[int] = set()
    tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
    if tokenizer_eos is not None:
        eos_ids.add(int(tokenizer_eos))

    try:
        from transformers import GenerationConfig

        gen_config = GenerationConfig.from_pretrained(model)
    except Exception:
        gen_config = None

    if gen_config is not None:
        gen_eos = getattr(gen_config, "eos_token_id", None)
        if isinstance(gen_eos, list):
            eos_ids.update(int(eos) for eos in gen_eos if eos is not None)
        elif gen_eos is not None:
            eos_ids.add(int(gen_eos))

    return sorted(eos_ids)


def load_tokenizer_and_eos(model: str) -> tuple[Any, list[int]]:
    """Load the fast tokenizer and resolve model EOS token ids together."""
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(model)
    return tokenizer, resolve_eos_token_ids(model, tokenizer)


__all__ = [
    "apply_hf_config_compatibility_fixes",
    "has_gdn_component",
    "has_hca_csa_cache",
    "has_layer_type",
    "load_tokenizer_and_eos",
    "resolve_eos_token_ids",
]
