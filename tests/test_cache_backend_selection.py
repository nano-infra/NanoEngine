import pytest
from dlengine.runtime.context.cache._backend import (
    CacheKind,
    resolve_cache_backend,
    supported_cache_kinds,
)


@pytest.mark.parametrize("kind", list(CacheKind))
def test_resolve_cache_backend_is_explicit_and_import_order_independent(kind):
    backend = resolve_cache_backend(kind)

    assert backend.kind is kind
    assert callable(backend.configure)
    assert callable(backend.get_block_bytes)
    assert callable(backend.allocate)


def test_supported_cache_kinds_are_a_closed_set():
    assert supported_cache_kinds() == (
        CacheKind.GQA,
        CacheKind.MLA,
        CacheKind.DSV4,
    )


def test_unknown_cache_backend_fails_with_supported_modes():
    with pytest.raises(ValueError, match="Supported modes: gqa, mla, dsv4"):
        resolve_cache_backend("unknown")
