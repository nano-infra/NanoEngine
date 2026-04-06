from types import SimpleNamespace

import torch

from nanodeploy.worker import sp_backend, sp_context


class _FakeKernelImpl:
    Basic = "basic"


class _FakeCompatHaoBuffer:
    def __init__(self, rank, world_size, max_bs, buffer_size_bytes):
        self.rank = rank
        self.world_size = world_size
        self.max_bs = max_bs
        self.buffer_size_bytes = buffer_size_bytes
        self.last_call = None

    def get_ipc_handle_info(self):
        return {"rank": self.rank}

    def connect_full_mesh(self, all_handles):
        self.all_handles = all_handles

    def all_to_all(self, x, impl, is_transpose, mask, offsets=None):
        self.last_call = {
            "x": x.clone(),
            "impl": impl,
            "is_transpose": is_transpose,
            "mask": None if mask is None else mask.clone(),
            "offsets": None if offsets is None else offsets.clone(),
        }
        return torch.zeros(
            self.world_size,
            self.max_bs,
            x.size(1),
            dtype=x.dtype,
            device=x.device,
        )


class _FakeNativeHaoBuffer(_FakeCompatHaoBuffer):
    def __init__(self, rank, world_size, max_bs, buffer_size_bytes):
        super().__init__(rank, world_size, max_bs, buffer_size_bytes)
        self._native_local_buffer = torch.empty(buffer_size_bytes, dtype=torch.uint8)

    def get_local_buffer(self):
        return self._native_local_buffer


class _FakeCreatedBuffer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.connected_group = None

    @property
    def local_buffer(self):
        return torch.empty(1, dtype=torch.uint8)

    def connect_full_mesh(self, group):
        self.connected_group = group

    def all_to_all_ll(self, x, is_transpose=False, mask=None, offsets=None):
        return x


class _FakeFactory:
    def __init__(self):
        self.size_args = None
        self.created = []

    def get_buffer_size_hint(
        self, max_dispatch_per_msg: int, max_bs: int, max_msg_size: int, itemsize: int
    ) -> int:
        self.size_args = (max_dispatch_per_msg, max_bs, max_msg_size, itemsize)
        return 4096

    def create_buffer(self, **kwargs):
        buffer = _FakeCreatedBuffer(**kwargs)
        self.created.append(buffer)
        return buffer


def test_hao_adapter_compat_mode_translates_mask_and_transpose(monkeypatch):
    monkeypatch.setattr(
        sp_backend, "_resolve_hao_symbols", lambda: (_FakeCompatHaoBuffer, _FakeKernelImpl)
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    adapter = sp_backend.HaoAllToAllBufferAdapter(
        max_dispatch_per_msg=1,
        max_bs=3,
        rank=1,
        world_size=2,
        buffer_size_bytes=2 * 3 * 2 * torch.tensor([], dtype=torch.float32).element_size(),
    )

    x = torch.arange(12, dtype=torch.float32).view(6, 2)
    mask = torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=torch.int32)

    staging = adapter.local_buffer.view(torch.float32)[: 2 * 3 * 2].view(2, 3, 2)
    staging[1].copy_(
        torch.tensor(
            [[101.0, 102.0], [103.0, 104.0], [105.0, 106.0]], dtype=torch.float32
        )
    )

    output = adapter.all_to_all_ll(x, is_transpose=True, mask=mask)
    call = adapter._buffer.last_call

    expected_x = torch.tensor(
        [[6.0, 7.0], [2.0, 3.0], [4.0, 5.0]], dtype=torch.float32
    )

    assert call is not None
    assert call["impl"] == _FakeKernelImpl.Basic
    assert call["is_transpose"] is False
    assert torch.equal(call["mask"], mask.transpose(0, 1).contiguous())
    assert call["offsets"] is None
    assert torch.equal(call["x"], expected_x)
    assert torch.equal(output[1], staging[1])


def test_hao_adapter_native_mode_passes_through_semantics(monkeypatch):
    monkeypatch.setattr(
        sp_backend, "_resolve_hao_symbols", lambda: (_FakeNativeHaoBuffer, _FakeKernelImpl)
    )

    adapter = sp_backend.HaoAllToAllBufferAdapter(
        max_dispatch_per_msg=2,
        max_bs=3,
        rank=0,
        world_size=2,
        buffer_size_bytes=64,
    )

    x = torch.arange(6, dtype=torch.float32).view(3, 2)
    mask = torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=torch.int32)

    output = adapter.all_to_all_ll(x, is_transpose=True, mask=mask)
    call = adapter._buffer.last_call

    assert adapter.local_buffer is adapter._buffer.get_local_buffer()
    assert call is not None
    assert call["is_transpose"] is True
    assert torch.equal(call["x"], x)
    assert torch.equal(call["mask"], mask)
    assert call["offsets"] is None
    assert output.shape == (2, 3, 2)


def test_hao_adapter_native_mode_pads_masked_non_transpose_input(monkeypatch):
    monkeypatch.setattr(
        sp_backend, "_resolve_hao_symbols", lambda: (_FakeNativeHaoBuffer, _FakeKernelImpl)
    )

    adapter = sp_backend.HaoAllToAllBufferAdapter(
        max_dispatch_per_msg=2,
        max_bs=4,
        rank=0,
        world_size=2,
        buffer_size_bytes=64,
    )

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
    mask = torch.tensor([[0, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.int32)

    adapter.all_to_all_ll(x, is_transpose=False, mask=mask)
    call = adapter._buffer.last_call

    expected_x = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [0.0, 0.0]], dtype=torch.float32
    )

    assert call is not None
    assert call["is_transpose"] is False
    assert torch.equal(call["mask"], mask)
    assert call["offsets"] is None
    assert torch.equal(call["x"], expected_x)


def test_hao_adapter_native_mode_passes_offsets_without_padding(monkeypatch):
    monkeypatch.setattr(
        sp_backend, "_resolve_hao_symbols", lambda: (_FakeNativeHaoBuffer, _FakeKernelImpl)
    )

    adapter = sp_backend.HaoAllToAllBufferAdapter(
        max_dispatch_per_msg=2,
        max_bs=4,
        rank=0,
        world_size=2,
        buffer_size_bytes=64,
    )

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
    mask = torch.tensor([[0, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.int32)
    offsets = torch.tensor([0, 3, 3], dtype=torch.int32)

    adapter.all_to_all_ll(x, is_transpose=False, mask=mask, offsets=offsets)
    call = adapter._buffer.last_call

    assert call is not None
    assert call["is_transpose"] is False
    assert torch.equal(call["x"], x)
    assert torch.equal(call["mask"], mask)
    assert torch.equal(call["offsets"], offsets)


def test_hao_adapter_compat_mode_rejects_offsets(monkeypatch):
    monkeypatch.setattr(
        sp_backend, "_resolve_hao_symbols", lambda: (_FakeCompatHaoBuffer, _FakeKernelImpl)
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    adapter = sp_backend.HaoAllToAllBufferAdapter(
        max_dispatch_per_msg=1,
        max_bs=3,
        rank=0,
        world_size=2,
        buffer_size_bytes=64,
    )

    x = torch.arange(6, dtype=torch.float32).view(3, 2)
    offsets = torch.tensor([0, 3, 3], dtype=torch.int32)

    try:
        adapter.all_to_all_ll(x, is_transpose=False, offsets=offsets)
    except NotImplementedError as exc:
        assert "compat mode" in str(exc)
    else:
        raise AssertionError("Expected compat-mode hao_basic offsets to raise NotImplementedError")


def test_hao_adapter_rejects_transpose_offsets(monkeypatch):
    monkeypatch.setattr(
        sp_backend, "_resolve_hao_symbols", lambda: (_FakeNativeHaoBuffer, _FakeKernelImpl)
    )

    adapter = sp_backend.HaoAllToAllBufferAdapter(
        max_dispatch_per_msg=1,
        max_bs=3,
        rank=0,
        world_size=2,
        buffer_size_bytes=64,
    )

    x = torch.arange(12, dtype=torch.float32).view(6, 2)
    offsets = torch.tensor([0, 3, 3], dtype=torch.int32)

    try:
        adapter.all_to_all_ll(x, is_transpose=True, offsets=offsets)
    except NotImplementedError as exc:
        assert "non-transpose" in str(exc)
    else:
        raise AssertionError("Expected transpose hao_basic offsets to raise NotImplementedError")


def test_set_sp_context_uses_backend_factory_and_keyword_args(monkeypatch):
    fake_factory = _FakeFactory()
    monkeypatch.setattr(sp_context, "create_sp_backend_factory", lambda backend: fake_factory)
    monkeypatch.setattr(
        sp_context,
        "get_dist_context",
        lambda: SimpleNamespace(attn_sp_group="fake-sp-group"),
    )

    sp_context.set_sp_context(
        max_num_seqs=16,
        head_size=8,
        num_attention_heads=4,
        dtype=torch.float16,
        rank=7,
        sp_size=11,
        backend="hao_basic",
    )
    ctx = sp_context.get_sp_context()

    assert ctx.backend == "hao_basic"
    assert fake_factory.size_args == (11, 16, 36, torch.float16.itemsize)
    assert [buffer.kwargs["max_dispatch_per_msg"] for buffer in fake_factory.created] == [
        11,
        1,
        1,
    ]
    assert [buffer.kwargs["rank"] for buffer in fake_factory.created] == [7, 7, 7]
    assert [buffer.kwargs["world_size"] for buffer in fake_factory.created] == [
        11,
        11,
        11,
    ]
    assert [buffer.connected_group for buffer in fake_factory.created] == [
        "fake-sp-group",
        "fake-sp-group",
        "fake-sp-group",
    ]
