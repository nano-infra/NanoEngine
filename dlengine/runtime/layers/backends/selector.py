"""Capability-aware selection of composable layer backends."""

from dataclasses import dataclass

import torch


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
    if requested in ("fa2", "flashinfer", "torch"):
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
        linear=("deepseek" if optimized else "torch") if linear == "auto" else linear,
        experts=(
            ("deepseek" if optimized else "torch") if experts == "auto" else experts
        ),
    )


def create_attention(*, requested="auto", hardware_backend="gpu_generic", **kwargs):
    attention_type = kwargs.get("attention_type", "MLA")
    if attention_type != "GQA":
        if requested not in (None, "auto"):
            raise RuntimeError(
                f"Explicit attention backend {requested!r} only supports GQA; "
                f"{attention_type} uses its dedicated implementation."
            )
        if hardware_backend == "blackwell":
            from dlengine.runtime.layers.blackwell.attention import (
                BlackwellMLAAttention,
            )

            return BlackwellMLAAttention(**kwargs)
        if hardware_backend == "hopper":
            from dlengine.runtime.layers.hopper.attention import HopperAttention

            kwargs.pop("mla_qk_nope_head_dim", None)
            return HopperAttention(**kwargs)
        from dlengine.runtime.layers.generic.attention import GenericAttention

        return GenericAttention(**kwargs)

    plan = resolve_attention_plan(requested or "auto")
    if plan.prefill == "fa4":
        from .fa.fa4 import FA4Attention

        return FA4Attention(**kwargs)
    if plan.prefill == "fa3":
        from .fa.fa3 import FA3Attention

        return FA3Attention(**kwargs)
    kwargs.pop("nsa_index_topk", None)
    if plan.prefill == "fa2":
        from .fa.fa2 import FA2Attention

        return FA2Attention(
            force_flashinfer_decode=plan.decode == "flashinfer", **kwargs
        )
    if plan.prefill == "flashinfer":
        from .flashinfer.attention import FlashInferAttention

        return FlashInferAttention(**kwargs)
    from .torch.attention import TorchAttention

    return TorchAttention(**kwargs)


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
