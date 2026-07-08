import pytest

from nanodeploy.config import Config


def test_loongserve_decode_scheduler_rejects_piecewise_cuda_graph(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    with pytest.raises(ValueError, match="cuda_graph_mode='full'"):
        Config(
            model=str(model_dir),
            loongserve_decode_scheduler=True,
            loop_count=1,
            cuda_graph_mode="piecewise",
        )


def test_loongserve_decode_scheduler_rejects_kv_migration_flag(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    with pytest.raises(ValueError, match="no-migration"):
        Config(
            model=str(model_dir),
            loongserve_decode_scheduler=True,
            loongserve_enable_kv_migration=True,
            loop_count=1,
            cuda_graph_mode="full",
        )


def test_loongserve_decode_scheduler_rejects_dlslime_sp_backend(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    with pytest.raises(ValueError, match="sp_backend='nccl'"):
        Config(
            model=str(model_dir),
            loongserve_decode_scheduler=True,
            loop_count=1,
            cuda_graph_mode="full",
            sp_backend="legacy_ll",
        )
