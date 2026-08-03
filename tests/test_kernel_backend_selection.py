import pytest

from dlengine.layers.backends.selector import (
    AttentionBackendPlan,
    GDNBackendPlan,
    resolve_attention_plan,
    resolve_gdn_plan,
)


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ((10, 3), AttentionBackendPlan("fa4", "flashinfer")),
        ((9, 0), AttentionBackendPlan("fa3", "fa3")),
        ((8, 0), AttentionBackendPlan("fa2", "flashinfer")),
    ],
)
def test_attention_auto_plan(capability, expected):
    assert resolve_attention_plan("auto", capability) == expected


def test_attention_explicit_backend_validates_architecture():
    with pytest.raises(RuntimeError, match="SM100"):
        resolve_attention_plan("fa4", (9, 0))
    with pytest.raises(RuntimeError, match="SM90"):
        resolve_attention_plan("fa3", (8, 0))


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ((10, 3), GDNBackendPlan("flashinfer", "flashinfer")),
        ((9, 0), GDNBackendPlan("flashinfer", "flashinfer")),
        ((8, 0), GDNBackendPlan("fla", "fla")),
    ],
)
def test_gdn_auto_plan(capability, expected):
    assert resolve_gdn_plan("auto", capability) == expected


def test_explicit_debug_backends_are_never_selected_by_auto():
    assert resolve_attention_plan("torch", (10, 3)) == AttentionBackendPlan(
        "torch", "torch"
    )
    assert resolve_gdn_plan("torch", (10, 3)) == GDNBackendPlan("torch", "torch")
