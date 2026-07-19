import pytest
from dlengine.context_v2.cache.hca import DSV4_BYTES_PER_TOKEN
from dlengine.disagg.p2p.cache_layout import CacheTensorLayout


def _layout(**overrides):
    values = dict(
        num_blocks=10,
        block_size=64,
        num_local_kv_heads=1,
        head_dim=128,
        dtype_itemsize=2,
        num_hidden_layers=3,
        mode="gqa",
    )
    values.update(overrides)
    return CacheTensorLayout(**values)


def test_gqa_remote_stage_uses_stage_local_layer_count_for_v_plane():
    layout = _layout(num_hidden_layers=2)
    block_bytes = 64 * 1 * 128 * 2

    assert layout.kv_stride(1, 1, 3) == (2 * 10 + 1 * 10 + 3) * block_bytes


def test_dsv4_hca_stride_includes_dummy_page_between_layers():
    layout = _layout(
        mode="dsv4",
        head_dim=512,
        dtype_itemsize=2,
        num_hidden_layers=2,
    )
    page_bytes = 64 * DSV4_BYTES_PER_TOKEN

    assert layout.kv_stride(0, 1, 3) == (11 + 3) * page_bytes
    assert layout.block_stride(1) == page_bytes
    with pytest.raises(ValueError, match="single cache plane"):
        layout.kv_stride(1, 0, 0)
