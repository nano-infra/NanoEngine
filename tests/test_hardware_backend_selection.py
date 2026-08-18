import pytest
from dlengine.layers.backend_selection import (
    BackendSelection,
    resolve_backend_selection,
)


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ((10, 0), "blackwell"),
        ((9, 0), "hopper"),
        ((8, 0), "gpu_generic"),
        (None, "gpu_generic"),
    ],
)
def test_auto_hardware_backend_selection(capability, expected):
    selection = resolve_backend_selection(cuda_capability=capability)

    assert selection.hardware == expected
    assert selection.hardware_source == "auto"


def test_explicit_config_wins_over_legacy_environment():
    selection = resolve_backend_selection(
        requested_hardware="hopper",
        requested_attention="fa3",
        requested_gdn="flashinfer",
        cuda_capability=(10, 0),
        legacy_hardware_backend="blackwell",
    )

    assert selection == BackendSelection(
        hardware="hopper",
        attention="fa3",
        gdn="flashinfer",
        hardware_source="config",
        hardware_reason="explicit hardware_backend=hopper",
    )


def test_legacy_environment_is_only_used_for_auto():
    selection = resolve_backend_selection(
        cuda_capability=(10, 0),
        legacy_hardware_backend="hopper",
    )

    assert selection.hardware == "hopper"
    assert selection.hardware_source == "environment"
    assert selection.hardware_reason == "NANO_BACKEND=hopper"


def test_invalid_legacy_environment_fails_at_selection_boundary():
    with pytest.raises(ValueError, match="Unknown NANO_BACKEND"):
        resolve_backend_selection(
            cuda_capability=(9, 0),
            legacy_hardware_backend="typo",
        )
