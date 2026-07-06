"""Lazy model-class registry used by workers.

Keep imports inside loader functions so importing ModelRunner does not eagerly
import every model implementation.
"""


def _qwen3_cls():
    from dlengine.models.qwen3.qwen3 import Qwen3ForCausalLM

    return Qwen3ForCausalLM


def _qwen3_moe_cls():
    from dlengine.models.qwen3_moe.qwen3_moe import Qwen3MoeForCausalLM

    return Qwen3MoeForCausalLM


def _qwen3_5_cls():
    from dlengine.models.qwen3_5.qwen3_5 import Qwen3_5ForConditionalGeneration

    return Qwen3_5ForConditionalGeneration


def _qwen3_5_moe_cls():
    from dlengine.models.qwen3_5_moe.qwen3_5_moe import (
        Qwen3_5MoeForConditionalGeneration,
    )

    return Qwen3_5MoeForConditionalGeneration


def _deepseek_v2_cls():
    from dlengine.models.deepseek_v2.deepseek_v2 import DeepseekV2ForCausalLM

    return DeepseekV2ForCausalLM


def _deepseek_v4_cls():
    from dlengine.models.deepseek_v4.deepseek_v4 import DeepseekV4ForCausalLM

    return DeepseekV4ForCausalLM


def _deepseek_mtp_cls():
    from dlengine.models.deepseek_v2.deepseek_v2_mtp import DeepSeekMTP

    return DeepSeekMTP


def _gemma4_cls():
    from dlengine.models.gemma4.gemma4 import Gemma4ForCausalLM

    return Gemma4ForCausalLM


def _qwen3_5_mtp_cls():
    from dlengine.models.qwen3_5_moe.qwen3_5_moe_mtp import Qwen3_5MTP

    return Qwen3_5MTP


architecture_loaders = {
    "Qwen3ForCausalLM": _qwen3_cls,
    "Qwen3MoeForCausalLM": _qwen3_moe_cls,
    "Qwen3_5ForConditionalGeneration": _qwen3_5_cls,
    "DeepseekV3ForCausalLM": _deepseek_v2_cls,
    "DeepseekV32ForCausalLM": _deepseek_v2_cls,
    "DeepseekV4ForCausalLM": _deepseek_v4_cls,
    "GlmMoeDsaForCausalLM": _deepseek_v2_cls,
    "Qwen3_5MoeForConditionalGeneration": _qwen3_5_moe_cls,
    "Gemma4ForCausalLM": _gemma4_cls,
    "Gemma4ForConditionalGeneration": _gemma4_cls,
}

architecture_mtp_loaders = {
    "DeepseekV3ForCausalLM": _deepseek_mtp_cls,
    "DeepseekV32ForCausalLM": _deepseek_mtp_cls,
    "GlmMoeDsaForCausalLM": _deepseek_mtp_cls,
    "Qwen3_5MoeForConditionalGeneration": _qwen3_5_mtp_cls,
}


__all__ = ["architecture_loaders", "architecture_mtp_loaders"]
