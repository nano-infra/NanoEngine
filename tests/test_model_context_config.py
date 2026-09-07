from types import SimpleNamespace

import pytest
from jsonargparse import ArgumentParser

import dlengine.config as config_module
from dlengine.config import Config
from dlengine.engine.scheduler import build_scheduler_config


@pytest.fixture
def hf_config(monkeypatch):
    hf = SimpleNamespace(
        architectures=["Qwen3ForCausalLM"],
        dtype="bfloat16",
        max_position_embeddings=131072,
        num_hidden_layers=2,
        num_key_value_heads=8,
    )
    monkeypatch.setattr(config_module.AutoConfig, "from_pretrained", lambda *a, **kw: hf)
    monkeypatch.setattr(
        config_module.PretrainedConfig, "get_config_dict", lambda *a, **kw: ({}, {})
    )
    monkeypatch.setattr(config_module.torch.cuda, "is_available", lambda: False)
    return hf


@pytest.mark.parametrize("length", [4096, 131072, 1048576])
@pytest.mark.parametrize("kwargs", [{}, {"max_model_len": None}])
def test_omitted_length_follows_model(hf_config, length, kwargs):
    hf_config.max_position_embeddings = length
    config = Config(model="unused", **kwargs)
    assert config.max_model_len == length
    assert hf_config.max_position_embeddings == length
    assert build_scheduler_config(config).max_model_len == length


@pytest.mark.parametrize("as_dict", [False, True])
@pytest.mark.parametrize("outer", [None, 4096, 1048576])
def test_nested_text_length_takes_precedence(hf_config, as_dict, outer):
    hf_config.max_position_embeddings = outer
    text = {"max_position_embeddings": 262144}
    hf_config.text_config = text if as_dict else SimpleNamespace(**text)
    config = Config(model="unused")
    assert config.max_model_len == 262144
    assert hf_config.max_position_embeddings == 262144


@pytest.mark.parametrize("override", [8192, 262144])
def test_explicit_length_takes_precedence(hf_config, override):
    config = Config(model="unused", max_model_len=override)
    assert config.max_model_len == override
    assert hf_config.max_position_embeddings == max(131072, override)


@pytest.mark.parametrize("override", [8192, 524288])
def test_explicit_length_with_nested_model(hf_config, override):
    hf_config.text_config = SimpleNamespace(max_position_embeddings=262144)
    config = Config(model="unused", max_model_len=override)
    assert config.max_model_len == override
    assert hf_config.max_position_embeddings == max(262144, override)


@pytest.mark.parametrize("missing", [False, True])
def test_missing_model_length_uses_legacy_fallback(hf_config, missing):
    if missing:
        del hf_config.max_position_embeddings
    else:
        hf_config.max_position_embeddings = None
    config = Config(model="unused")
    assert config.max_model_len == 16384
    assert hf_config.max_position_embeddings == 16384


def test_explicit_length_without_model_metadata(hf_config):
    del hf_config.max_position_embeddings
    config = Config(model="unused", max_model_len=32768)
    assert config.max_model_len == hf_config.max_position_embeddings == 32768


def test_missing_nested_length_uses_outer(hf_config):
    hf_config.text_config = {"max_position_embeddings": None}
    assert Config(model="unused").max_model_len == 131072


@pytest.mark.parametrize("length", [0, -1])
def test_rejects_nonpositive_explicit_length(hf_config, length):
    with pytest.raises(ValueError, match="max_model_len"):
        Config(model="unused", max_model_len=length)


@pytest.mark.parametrize("length", [0, -1, "invalid"])
def test_rejects_invalid_model_length(hf_config, length):
    hf_config.max_position_embeddings = length
    with pytest.raises(ValueError, match="max_position_embeddings"):
        Config(model="unused")


@pytest.mark.parametrize("args,expected", [([], 131072), (["--max_model_len", "8192"], 8192)])
def test_cli_preserves_automatic_or_explicit_length(hf_config, args, expected):
    parser = ArgumentParser()
    parser.add_class_arguments(Config, fail_untyped=False)
    parsed = parser.parse_args(["--model", "unused", *args])
    if not args:
        assert parsed.max_model_len is None
    config = Config(**vars(parsed))
    assert config.max_model_len == expected
