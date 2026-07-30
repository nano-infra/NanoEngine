import torch
from dlengine.context_v2.graph import (
    FlashInferDecodeGraphConfig,
    FlashInferDecodeGraphState,
)


class _FakeWrapper:
    def __init__(self):
        self.plan_calls = 0

    def plan(self, *args, **kwargs):
        self.plan_calls += 1


def test_reused_flashinfer_plan_refreshes_physical_page_indices():
    config = FlashInferDecodeGraphConfig(
        enabled=False,
        max_num_blocks=4,
        block_size=16,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        reuse_page_plan=True,
    )
    state = FlashInferDecodeGraphState(config)
    config.enabled = True
    state.flashinfer = object()

    master_bs = 1
    wrapper = _FakeWrapper()
    state.wrappers[master_bs] = wrapper
    state.indptr[master_bs] = torch.empty(2, dtype=torch.int32)
    state.indices[master_bs] = torch.empty(4, dtype=torch.int32)
    state.last_page_len[master_bs] = torch.empty(1, dtype=torch.int32)

    state.plan(
        master_bs,
        1,
        torch.tensor([[3, 0, 0, 0]], dtype=torch.int32),
        torch.tensor([10], dtype=torch.int32),
        page_plan_key=(1,),
    )
    assert state.indices[master_bs][0].item() == 3
    assert state.last_page_len[master_bs][0].item() == 10
    assert wrapper.plan_calls == 1

    state.plan(
        master_bs,
        1,
        torch.tensor([[7, 0, 0, 0]], dtype=torch.int32),
        torch.tensor([11], dtype=torch.int32),
        page_plan_key=(1,),
    )
    assert state.indices[master_bs][0].item() == 7
    assert state.last_page_len[master_bs][0].item() == 11
    assert wrapper.plan_calls == 1
