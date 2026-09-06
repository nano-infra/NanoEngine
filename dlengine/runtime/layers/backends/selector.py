"""Capability-aware selection of composable layer backends."""

from dataclasses import dataclass

import torch

from dlengine.logging import get_logger

logger = get_logger()


@dataclass(frozen=True)
class AttentionBackendPlan:
    prefill: str
    decode: str


@dataclass(frozen=True)
class GDNBackendPlan:
    prefill: str
    decode: str


@dataclass(frozen=True)
class BackendPlan:
    attention: AttentionBackendPlan
    gdn: GDNBackendPlan
    linear: str
    experts: str


def resolve_attention_plan(requested="auto", capability=None):
    capability = capability or torch.cuda.get_device_capability()
    major = capability[0]
    # ``torch`` is a deprecated alias for the pure-SDPA ``generic`` backend.
    if requested == "torch":
        requested = "generic"
    if requested == "auto":
        if major >= 10:
            return AttentionBackendPlan("fa4", "flashinfer")
        if major >= 9:
            return AttentionBackendPlan("fa3", "fa3")
        return AttentionBackendPlan("fa2", "flashinfer")
    if requested == "fa4":
        if major < 10:
            raise RuntimeError("FA4 attention requires a Blackwell-class GPU (SM100+).")
        return AttentionBackendPlan("fa4", "flashinfer")
    if requested == "fa3":
        if major < 9:
            raise RuntimeError("FA3 attention requires a Hopper-class GPU (SM90+).")
        return AttentionBackendPlan("fa3", "fa3")
    if requested in ("fa2", "flashinfer", "generic"):
        return AttentionBackendPlan(requested, requested)
    raise ValueError(f"Unknown attention backend: {requested!r}")


def resolve_gdn_plan(requested="auto", capability=None):
    capability = capability or torch.cuda.get_device_capability()
    if requested == "auto":
        name = "flashinfer" if capability[0] >= 9 else "fla"
        return GDNBackendPlan(name, name)
    if requested in ("flashinfer", "fla", "torch"):
        return GDNBackendPlan(requested, requested)
    raise ValueError(f"Unknown GDN backend: {requested!r}")


def resolve_backend_plan(
    attention="auto", gdn="auto", linear="auto", experts="auto", capability=None
):
    capability = capability or torch.cuda.get_device_capability()
    optimized = capability[0] >= 9
    return BackendPlan(
        attention=resolve_attention_plan(attention, capability),
        gdn=resolve_gdn_plan(gdn, capability),
        linear=("deepseek" if optimized else "generic") if linear == "auto" else linear,
        experts=(
            ("deepseek" if optimized else "generic") if experts == "auto" else experts
        ),
    )


def create_linear(
    kind,
    *,
    family,
    quantization_config=None,
    scale_tensor=None,
    **kwargs,
):
    """Instantiate a linear layer of ``kind`` from the given implementation family.

    ``kind`` is one of ``row``/``column``/``merged``/``qkv``/``replicated``.
    ``family`` is ``"generic"`` (BF16/ref) or ``"deepseek"`` (FP8/DeepGEMM).

    The generic family does not accept ``quantization_config``/``scale_tensor``;
    the deepseek family always receives both. Remaining constructor arguments
    (input_size/output_size/bias/meta/weight_tensor/bias_tensor/tp_group, or the
    QKV/merged variants) are passed through ``kwargs`` verbatim.
    """
    if family == "deepseek":
        from dlengine.runtime.layers.backends.deepseek import linear as _mod

        classes = {
            "row": _mod.HopperRowParallelLinear,
            "column": _mod.HopperColumnParallelLinear,
            "merged": _mod.HopperMergedColumnParallelLinear,
            "qkv": _mod.HopperQKVParallelLinear,
            "replicated": _mod.HopperReplicatedLinear,
        }
        kwargs["quantization_config"] = quantization_config
        kwargs["scale_tensor"] = scale_tensor
    elif family == "generic":
        from dlengine.runtime.layers.backends.generic import linear as _mod

        classes = {
            "row": _mod.GenericRowParallelLinear,
            "column": _mod.GenericColumnParallelLinear,
            "merged": _mod.GenericMergedColumnParallelLinear,
            "qkv": _mod.GenericQKVParallelLinear,
            "replicated": _mod.GenericReplicatedLinear,
        }
    else:
        raise ValueError(f"Unknown linear backend family: {family!r}")

    if kind not in classes:
        raise ValueError(f"Unknown linear kind: {kind!r}")
    return classes[kind](**kwargs)


def _create_generic_experts(**kwargs):
    from dlengine.runtime.layers.backends.generic.experts import (
        GenericDistributedRoutedExperts,
    )

    kwargs.pop("quantization_config", None)
    return GenericDistributedRoutedExperts(**kwargs)


def create_experts(
    *,
    family,
    quantization_config=None,
    experts_quant_override=False,
    ref_fallback_allowed=False,
    **kwargs,
):
    """Instantiate routed experts from the given implementation family.

    When ``experts_quant_override`` is set (Blackwell tier), the checkpoint's
    quantization format takes precedence: NVFP4 -> ModelOptNvFp4Experts,
    MXFP4 -> MegaMoEExperts, otherwise the base ``family`` is used.

    When ``ref_fallback_allowed`` is set and the preferred family fails to
    construct (e.g. the DeepGEMM/DeepEP path is unavailable in this build), the
    portable generic experts are used instead. When it is not set, the failure
    propagates so performance expectations stay deterministic.
    """
    if experts_quant_override:
        if bool(getattr(quantization_config, "is_modelopt_nvfp4", False)):
            from dlengine.runtime.layers.backends.nvfp4 import ModelOptNvFp4Experts

            return ModelOptNvFp4Experts(
                quantization_config=quantization_config, **kwargs
            )
        if bool(getattr(quantization_config, "is_mxfp4", False)):
            from dlengine.runtime.layers.backends.megamoe import MegaMoEExperts

            return MegaMoEExperts(quantization_config=quantization_config, **kwargs)

    if family == "deepseek":
        try:
            from dlengine.runtime.layers.backends.deepseek.experts import (
                HopperDistributedRoutedExperts,
            )

            return HopperDistributedRoutedExperts(
                quantization_config=quantization_config, **kwargs
            )
        except Exception:
            if not ref_fallback_allowed:
                raise
            logger.warning(
                "deepseek experts unavailable; falling back to generic experts "
                "(ref_fallback_allowed=True)."
            )
            return _create_generic_experts(**kwargs)
    if family == "generic":
        return _create_generic_experts(quantization_config=quantization_config, **kwargs)
    raise ValueError(f"Unknown experts backend family: {family!r}")


def create_attention(*, requested="auto", hardware_backend="gpu_generic", **kwargs):
    attention_type = kwargs.get("attention_type", "MLA")
    if attention_type != "GQA":
        if requested not in (None, "auto"):
            raise RuntimeError(
                f"Explicit attention backend {requested!r} only supports GQA; "
                f"{attention_type} uses its dedicated implementation."
            )
        if hardware_backend == "blackwell":
            from .attention.fa4 import Fa4MlaAttention

            return Fa4MlaAttention(**kwargs)
        if hardware_backend == "hopper":
            from .attention.fa3 import Fa3Attention

            kwargs.pop("mla_qk_nope_head_dim", None)
            return Fa3Attention(**kwargs)
        from .attention.generic import GenericAttention

        return GenericAttention(**kwargs)

    plan = resolve_attention_plan(requested or "auto")
    if plan.prefill == "fa4":
        from .attention.fa4 import Fa4Attention

        return Fa4Attention(**kwargs)
    if plan.prefill == "fa3":
        from .attention.fa3 import Fa3Attention

        return Fa3Attention(**kwargs)
    kwargs.pop("nsa_index_topk", None)
    if plan.prefill == "fa2":
        from .attention.fa2 import Fa2Attention

        return Fa2Attention(
            force_flashinfer_decode=plan.decode == "flashinfer", **kwargs
        )
    if plan.prefill == "flashinfer":
        from .attention.flashinfer import FlashInferAttention

        return FlashInferAttention(**kwargs)
    from .attention.generic import GenericAttention

    return GenericAttention(**kwargs)


def create_gdn(*, requested="auto", **kwargs):
    plan = resolve_gdn_plan(requested or "auto")
    if plan.prefill == "flashinfer":
        from .flashinfer.gdn import FlashInferGatedDeltaNet

        return FlashInferGatedDeltaNet(**kwargs)
    if plan.prefill == "fla":
        from .fla.gdn import FLAGatedDeltaNet

        return FLAGatedDeltaNet(**kwargs)
    from .torch.gdn import TorchGatedDeltaNet

    return TorchGatedDeltaNet(**kwargs)


def create_kda(*, layer_idx, state_layer_idx, config, **kwargs):
    """Instantiate the Kimi Delta Attention (KDA) linear-attention layer.

    KDA has a single FlashInfer-backed implementation with no GDN/torch
    fallback; it is routed through the selector so model topologies depend only
    on the factory contract rather than importing the implementation directly.
    """
    from .kda import FlashInferKDA

    return FlashInferKDA(layer_idx, state_layer_idx, config)
