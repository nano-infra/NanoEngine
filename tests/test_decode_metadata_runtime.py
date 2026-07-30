from types import SimpleNamespace

import pytest
from dlengine._rust.config import CachePlan, SchedulerConfig
from dlengine._rust.core import Scheduler
from dlengine._rust.proto import decode_flat_control, RequestIn, SamplingParams
from dlengine.worker.decode_metadata import (
    decode_metadata_enabled,
    DecodeMetadataRuntime,
)


def _flat_decode_batch() -> bytes:
    scheduler = Scheduler(
        SchedulerConfig(
            engine_id="decode-metadata-test",
            max_num_seqs=4,
            max_num_batched_tokens=16,
            max_model_len=64,
            attention_dp=1,
            group_size=1,
            num_kvcache_blocks=16,
            kvcache_block_size=4,
            cache_plan=CachePlan(1),
            use_decode_metadata_kernel=True,
        )
    )
    scheduler.add_request_bytes(
        RequestIn(
            9,
            [1, 2, 3],
            SamplingParams(temperature=0.0, max_tokens=8),
            0,
            None,
        ).to_bytes()
    )
    prefill = scheduler.schedule()
    scheduler.postprocess(prefill.filtered_dp_group_seq_ids, [[[4]]], None, None)
    decode = scheduler.schedule()
    return scheduler.serialize_run_batches_for_result(decode, 1)[0]


def _runtime_config(**overrides):
    values = {
        "max_num_seqs": 4,
        "kvcache_block_size": 4,
        "max_model_len": 64,
        "attention_sp": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_flat_decode_control_and_validation():
    data = _flat_decode_batch()
    control = decode_flat_control(data)

    assert control.num_group_seqs == 1
    assert control.page_plan_key == [1]
    assert control.max_num_seqs == 4
    assert control.max_num_blocks == 16
    assert control.block_size == 4
    assert control.all_greedy is True

    damaged = bytearray(data)
    damaged[0] ^= 0xFF
    with pytest.raises(ValueError, match="magic"):
        decode_flat_control(bytes(damaged))


def test_decode_metadata_enabled_requires_qwen_graph_decode():
    config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["Qwen3_5ForConditionalGeneration"]),
        use_decode_metadata_kernel=True,
        enforce_eager=False,
        num_speculative_tokens=0,
        enable_hisparse=False,
    )
    assert decode_metadata_enabled(config)

    config.enforce_eager = True
    assert not decode_metadata_enabled(config)


def test_mapped_decode_graph_outputs_and_addresses():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    data = _flat_decode_batch()
    runtime = DecodeMetadataRuntime(_runtime_config())
    plan_buffers = (
        torch.empty(5, dtype=torch.int32, device="cuda"),
        torch.empty(64, dtype=torch.int32, device="cuda"),
        torch.empty(4, dtype=torch.int32, device="cuda"),
    )
    try:
        addresses = runtime.views.addresses()
        control = runtime.stage(data)
        runtime.launch_graph(4, plan_buffers)
        torch.cuda.synchronize()

        assert runtime.views.addresses() == addresses
        assert runtime.views.input_ids.tolist() == [4, 0, 0, 0]
        assert runtime.views.positions.tolist() == [3, 0, 0, 0]
        assert runtime.views.slot_mapping.tolist() == [3, -1, -1, -1]
        assert runtime.views.context_lens.tolist() == [[4, 1, 1, 1]]
        assert runtime.views.temperatures.tolist() == [0.0, 1.0, 1.0, 1.0]
        assert plan_buffers[0].tolist() == [0, 1, 2, 3, 4]
        assert plan_buffers[2].tolist() == [4, 1, 1, 1]
        assert plan_buffers[1][: sum(control.page_plan_key)].tolist() == [0]
    finally:
        runtime.close()
