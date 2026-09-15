from dlengine.runtime.kernel.triton.hopper import block_gemm_fp8


def test_packed_small_m_quant_dispatch_boundaries(monkeypatch):
    monkeypatch.setattr(block_gemm_fp8, "_USE_PACKED_SMALL_M_QUANT", True)

    should_pack = block_gemm_fp8._should_use_packed_small_m_quant
    assert should_pack(1, 128, 1, 1)
    assert should_pack(6, 128, 1, 1)
    assert should_pack(64, 128, 1, 1)
    assert not should_pack(0, 128, 1, 1)
    assert not should_pack(65, 128, 1, 1)
    assert not should_pack(6, 64, 1, 1)
    assert not should_pack(6, 128, 2, 1)
    assert not should_pack(6, 128, 1, 2)


def test_packed_small_m_quant_can_be_disabled(monkeypatch):
    monkeypatch.setattr(block_gemm_fp8, "_USE_PACKED_SMALL_M_QUANT", False)
    assert not block_gemm_fp8._should_use_packed_small_m_quant(6, 128, 1, 1)
