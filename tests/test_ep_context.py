import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from nanodeploy.worker import ep_context


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class _FakeConfig:
    def __init__(self, nvl_bytes: int, rdma_bytes: int):
        self.nvl_bytes = nvl_bytes
        self.rdma_bytes = rdma_bytes
        self.hint_calls: list[tuple[str, int, int]] = []

    def get_nvl_buffer_size_hint(
        self, hidden_bytes: int, ep_size: int
    ) -> int:
        self.hint_calls.append(("nvl", hidden_bytes, ep_size))
        return self.nvl_bytes

    def get_rdma_buffer_size_hint(
        self, hidden_bytes: int, ep_size: int
    ) -> int:
        self.hint_calls.append(("rdma", hidden_bytes, ep_size))
        return self.rdma_bytes


def _fake_deep_ep(*, low_latency_bytes: int = 600):
    dispatch_config = _FakeConfig(100, 500)
    combine_config = _FakeConfig(200, 400)

    class FakeBuffer:
        set_num_sms_calls: list[int] = []
        constructor_calls: list[tuple[object, dict[str, object]]] = []
        low_latency_hint_calls: list[tuple[int, int, int, int]] = []

        @classmethod
        def set_num_sms(cls, num_sms: int) -> None:
            cls.set_num_sms_calls.append(num_sms)

        @staticmethod
        def get_dispatch_config(ep_size: int) -> _FakeConfig:
            assert ep_size == 16
            return dispatch_config

        @staticmethod
        def get_combine_config(ep_size: int) -> _FakeConfig:
            assert ep_size == 16
            return combine_config

        @classmethod
        def get_low_latency_rdma_size_hint(
            cls,
            num_max_dispatch_tokens_per_rank: int,
            hidden: int,
            num_ranks: int,
            num_experts: int,
        ) -> int:
            cls.low_latency_hint_calls.append(
                (
                    num_max_dispatch_tokens_per_rank,
                    hidden,
                    num_ranks,
                    num_experts,
                )
            )
            return low_latency_bytes

        def __init__(self, group: object, **kwargs: object):
            type(self).constructor_calls.append((group, kwargs))
            self.destroy_calls = 0
            self.clean_calls: list[tuple[int, int, int]] = []

        def clean_low_latency_buffer(
            self,
            max_tokens_per_rank: int,
            hidden_size: int,
            num_experts: int,
        ) -> None:
            self.clean_calls.append(
                (max_tokens_per_rank, hidden_size, num_experts)
            )

        def destroy(self) -> None:
            self.destroy_calls += 1

    topk_idx_t = object()
    module = SimpleNamespace(Buffer=FakeBuffer, topk_idx_t=topk_idx_t)
    return module, dispatch_config, combine_config


@pytest.fixture(autouse=True)
def _reset_process_ep_context(monkeypatch):
    monkeypatch.setattr(ep_context, "_EP_CONTEXT", None)
    monkeypatch.delenv("NVSHMEM_QP_DEPTH", raising=False)
    yield
    monkeypatch.setattr(ep_context, "_EP_CONTEXT", None)


def _initialize_context(
    monkeypatch, *, low_latency_bytes: int = 600, qp_depth: int = 2048
):
    fake_deep_ep, dispatch_config, combine_config = _fake_deep_ep(
        low_latency_bytes=low_latency_bytes
    )
    monkeypatch.setattr(ep_context, "deep_ep", fake_deep_ep)
    group = object()
    context = ep_context.EPContext()
    context.initialize(
        ep_group=group,
        ep_size=16,
        num_experts=384,
        hidden_size=7168,
        max_tokens_per_rank=512,
        num_sms=16,
        allow_mnnvl=True,
        nvshmem_qp_depth=qp_depth,
    )
    return context, fake_deep_ep, group, dispatch_config, combine_config


def test_initialize_combines_normal_and_low_latency_size_hints(monkeypatch):
    (
        context,
        fake_deep_ep,
        group,
        dispatch_config,
        combine_config,
    ) = _initialize_context(monkeypatch)

    assert dispatch_config.hint_calls == [
        ("nvl", 7168 * 2, 16),
        ("rdma", 7168 * 2, 16),
    ]
    assert combine_config.hint_calls == [
        ("nvl", 7168 * 2, 16),
        ("rdma", 7168 * 2, 16),
    ]
    assert fake_deep_ep.Buffer.low_latency_hint_calls == [
        (512, 7168, 16, 384)
    ]
    assert context.num_nvl_bytes == 200
    assert context.num_rdma_bytes == 600

    [(constructed_group, kwargs)] = fake_deep_ep.Buffer.constructor_calls
    assert constructed_group is group
    assert kwargs == {
        "num_nvl_bytes": 200,
        "num_rdma_bytes": 600,
        "low_latency_mode": True,
        "num_qps_per_rank": 24,
        "allow_nvlink_for_low_latency_mode": True,
        "allow_mnnvl": True,
        "explicitly_destroy": True,
    }


def test_num_sms_topk_dtype_and_qps_are_recorded_separately(monkeypatch):
    context, fake_deep_ep, _, _, _ = _initialize_context(monkeypatch)

    assert fake_deep_ep.Buffer.set_num_sms_calls == [16]
    assert context.num_sms == 16
    assert context.num_local_experts == 24
    assert context.num_qps_per_rank == 24
    assert context.topk_idx_t is fake_deep_ep.topk_idx_t
    assert context.topk_idx_dtype is fake_deep_ep.topk_idx_t


def test_normal_rdma_hint_can_dominate_low_latency_hint(monkeypatch):
    context, _, _, _, _ = _initialize_context(
        monkeypatch, low_latency_bytes=300
    )

    assert context.num_rdma_bytes == 500


def test_qp_depth_is_validated_and_set_before_buffer_construction(monkeypatch):
    context, fake_deep_ep, _, _, _ = _initialize_context(monkeypatch)

    assert context.nvshmem_qp_depth == 2048
    assert os.environ["NVSHMEM_QP_DEPTH"] == "2048"
    assert fake_deep_ep.Buffer.constructor_calls


def test_qp_depth_below_twice_max_tokens_plus_one_is_rejected(monkeypatch):
    fake_deep_ep, _, _ = _fake_deep_ep()
    monkeypatch.setattr(ep_context, "deep_ep", fake_deep_ep)

    with pytest.raises(ValueError, match=r"1024 < 1026"):
        ep_context.EPContext().initialize(
            ep_group=object(),
            ep_size=16,
            num_experts=384,
            hidden_size=7168,
            max_tokens_per_rank=512,
            num_sms=16,
            allow_mnnvl=False,
            nvshmem_qp_depth=1024,
        )

    assert fake_deep_ep.Buffer.constructor_calls == []


def test_destroy_is_idempotent(monkeypatch):
    context, _, _, _, _ = _initialize_context(monkeypatch)
    buffer = context.get_buffer()

    assert context.destroy() is True
    assert context.destroy() is False
    assert buffer.destroy_calls == 1
    with pytest.raises(RuntimeError, match="not initialized"):
        context.get_buffer()


def test_normal_to_low_latency_transition_cleans_shared_buffer_once(
    monkeypatch,
):
    context, _, _, _, _ = _initialize_context(monkeypatch)
    buffer = context.get_buffer()

    # Starting directly in low-latency mode and staying there needs no clean.
    context.prepare_low_latency()
    context.prepare_low_latency()
    assert buffer.clean_calls == []

    # Multiple short-lived normal facades only mark one process-wide mode.
    context.mark_normal()
    context.mark_normal()
    context.prepare_low_latency()
    context.prepare_low_latency()
    assert buffer.clean_calls == [(512, 7168, 384)]

    # A later normal phase creates one new transition to clean.
    context.mark_normal()
    context.prepare_low_latency()
    assert buffer.clean_calls == [
        (512, 7168, 384),
        (512, 7168, 384),
    ]


def test_process_owner_destroy_is_idempotent(monkeypatch):
    fake_deep_ep, _, _ = _fake_deep_ep()
    monkeypatch.setattr(ep_context, "deep_ep", fake_deep_ep)
    context = ep_context.set_ep_context(
        ep_group=object(),
        ep_size=16,
        num_experts=384,
        hidden_size=7168,
        max_tokens_per_rank=128,
        num_sms=16,
        allow_mnnvl=False,
        nvshmem_qp_depth=1024,
    )
    buffer = context.get_buffer()

    assert ep_context.get_ep_context() is context
    assert ep_context.destroy_ep_context() is True
    assert ep_context.destroy_ep_context() is False
    assert buffer.destroy_calls == 1


def test_model_runner_orders_deepep_lifecycle_around_model_and_groups():
    path = REPOSITORY_ROOT / "nanodeploy/worker/model_runner.py"
    tree = ast.parse(path.read_text())
    model_runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelRunner"
    )
    init = next(
        node
        for node in model_runner.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    exit_method = next(
        node
        for node in model_runner.body
        if isinstance(node, ast.FunctionDef) and node.name == "exit"
    )

    configure_call = next(
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_configure_decode_deepep"
    )
    model_assignment = next(
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "model"
            for target in node.targets
        )
    )
    destroy_call = next(
        node
        for node in ast.walk(exit_method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "destroy_ep_context"
    )
    group_destroy_call = next(
        node
        for node in ast.walk(exit_method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "destroy_process_group"
    )

    assert configure_call.lineno < model_assignment.lineno
    assert destroy_call.lineno < group_destroy_call.lineno
