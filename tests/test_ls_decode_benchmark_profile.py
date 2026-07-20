import argparse
from types import SimpleNamespace

import pytest

from nanodeploy.config import Config
from scripts import bench_ls_decode_longrun_8gpu as longrun_script
from scripts import bench_ls_decode_serving as serving_script
from scripts.ls_decode_issue001_profile import (
    add_formal_profile_arguments,
    clear_ray_proxy_env,
    formal_engine_kwargs,
    resolved_manifest,
)


@pytest.fixture(autouse=True)
def mock_auto_config(monkeypatch):
    hf_config = SimpleNamespace(
        architectures=["DeepseekV3ForCausalLM"],
        max_position_embeddings=16_384,
        num_key_value_heads=1,
        num_attention_heads=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        torch_dtype="bfloat16",
    )
    monkeypatch.setattr(
        "nanodeploy.config.AutoConfig.from_pretrained",
        lambda *args, **kwargs: hf_config,
    )


def test_formal_cli_requires_explicit_ooe_and_disables_request_warmup():
    parser = argparse.ArgumentParser()
    add_formal_profile_arguments(parser)

    args = parser.parse_args(["--ls-max-num-ooe", "7"])
    assert args.ls_max_num_ooe == 7
    assert args.routing_strategy == "RoundRobin"
    assert args.warmup_requests == 0

    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--ls-max-num-ooe", "7", "--routing-strategy", "LeastBatch"]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(["--ls-max-num-ooe", "7", "--warmup-requests", "1"])


def test_formal_engine_kwargs_are_frozen_and_fresh():
    first = formal_engine_kwargs(9, ls_decode_batch_per_master=64)
    second = formal_engine_kwargs(9, ls_decode_batch_per_master=64)

    assert first == second
    assert first is not second
    assert first["ls_decode_profile"] == "loong_decode_issue001"
    assert first["ls_max_num_ooe"] == 9
    assert first["routing_strategy"] == "RoundRobin"
    assert first["ls_running_max_req_size"] == 1000
    assert first["ls_admission_max_tokens_per_pool"] == "auto"
    assert first["ls_min_comp_bound_decoding_batch_size"] == 128
    assert first["ls_decode_initial_kv_dop"] == 0
    assert first["ls_decode_enable_future_kv_admission"] is True
    assert first["ls_decode_enable_memory_scale_up"] is True
    assert first["ls_disable_scale_up"] is False
    assert first["ls_kv_consolidation_mode"] == "execute"
    assert first["ls_kv_consolidation_stable_steps"] == 2
    assert first["ls_kv_consolidation_cooldown_steps"] == 2
    assert first["ls_kv_consolidation_check_interval_steps"] == 1
    assert first["ls_kv_consolidation_max_source_blocks_per_event"] == 128
    assert first["ls_kv_consolidation_migration_chunk_tokens"] == 64
    assert first["pause_mode"] == "offload"
    assert first["dp_assignment"] == "arrival_round_robin"
    assert first["cross_dp_scale_up"] is False


def test_formal_entrypoints_resolve_shared_cli_contract(tmp_path, monkeypatch):
    csv_path = tmp_path / "issue001.csv"
    csv_path.write_text("prompt_len,output_len\n200,4\n", encoding="utf-8")
    output_jsonl = tmp_path / "result.jsonl"

    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_ls_decode_serving.py",
            "--csv-path",
            str(csv_path),
            "--output-jsonl",
            str(output_jsonl),
            "--ls-max-num-ooe",
            "6",
            "--loop-count",
            "16",
        ],
    )
    serving_args = serving_script.parse_args()
    assert serving_args.ls_max_num_ooe == 6
    assert serving_args.routing_strategy == "RoundRobin"
    assert serving_args.warmup_requests == 0
    assert serving_args.loop_count == 16

    monkeypatch.setattr(
        "sys.argv",
        ["bench_ls_decode_longrun_8gpu.py", "--ls-max-num-ooe", "8"],
    )
    longrun_args = longrun_script.parse_args()
    assert longrun_args.ls_max_num_ooe == 8
    assert longrun_args.routing_strategy == "RoundRobin"
    assert longrun_args.warmup_requests == 0
    assert longrun_args.ls_kv_consolidation_mode == "execute"
    assert longrun_args.ls_kv_consolidation_stable_steps == 2
    assert longrun_args.ls_kv_consolidation_cooldown_steps == 2
    assert longrun_args.ls_kv_consolidation_check_interval_steps == 1


def test_resolved_manifest_uses_scheduler_resolved_values(tmp_path):
    config = Config(
        model=str(tmp_path),
        mode="decode",
        dummy_prefill=True,
        scheduler_mode="centralized",
        loop_count=1,
        attention_dp=1,
        attention_sp=8,
        attention_tp=1,
        ffn_dp=1,
        ffn_ep=8,
        ffn_tp=1,
        use_dlslime_rpc=True,
        sp_backend="hao_basic",
        fixed_sp_size=0,
        enable_dynamic_sp_size=False,
        enable_non_uniform_split=False,
        kvcache_block_size=64,
        **formal_engine_kwargs(7, ls_decode_batch_per_master=64),
    )
    config.ls_resolved_admission_max_tokens_per_pool = (
        config.resolve_ls_admission_max_tokens(8 * 1000 * 64)
    )

    manifest = resolved_manifest(config, 7)

    assert manifest["profile"] == "loong_decode_issue001"
    assert manifest["ls_max_num_ooe"] == 7
    assert manifest["ls_admission_max_tokens_per_pool"] > 0
    assert manifest["ls_admission_max_tokens_per_pool_configured"] == "auto"
    assert manifest["ls_decode_enable_memory_scale_up"] is True
    assert manifest["ls_decode_batch_per_master"] == 64

    config.loop_count = 16
    chunked_manifest = resolved_manifest(config, 7, expected_loop_count=16)
    assert chunked_manifest["loop_count"] == 16
    with pytest.raises(RuntimeError, match="loop_count"):
        resolved_manifest(config, 7)

    config.routing_strategy = "LeastBatch"
    with pytest.raises(RuntimeError, match="routing_strategy"):
        resolved_manifest(config, 7, expected_loop_count=16)

    with pytest.raises(ValueError, match="expected_loop_count"):
        resolved_manifest(config, 7, expected_loop_count=17)


def test_clear_ray_proxy_env(monkeypatch):
    for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        monkeypatch.setenv(name, "http://127.0.0.1:15409")
    monkeypatch.setenv("NO_PROXY", "localhost")

    assert clear_ray_proxy_env() == [
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
    ]
    assert clear_ray_proxy_env() == []
