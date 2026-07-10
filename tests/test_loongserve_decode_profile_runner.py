import importlib.util
import random
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_runner():
    path = Path(__file__).resolve().parents[1] / "docs-dev" / "loongserve_decode_profile_runner.py"
    spec = importlib.util.spec_from_file_location("loongserve_decode_profile_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _args(**overrides):
    values = {
        "sp": 8,
        "ep": 8,
        "warmup": 0,
        "steps": 1,
        "max_model_len": 131072,
        "short_context_len": 1024,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_parse_token_count_list_accepts_k_and_m_suffixes():
    assert runner.parse_token_count_list("64K,1M,1536K,4096") == [
        65536,
        1048576,
        1572864,
        4096,
    ]


def test_w_attn_shape_variants_preserve_requested_total_for_mixed_shapes():
    one_long = runner.context_lens_for_w_attn(
        batch_size=4,
        w_attn=10000,
        shape="one_long",
        short_context_len=1000,
    )
    assert one_long == [7000, 1000, 1000, 1000]
    assert sum(one_long) == 10000

    two_long = runner.context_lens_for_w_attn(
        batch_size=5,
        w_attn=13001,
        shape="two_long",
        short_context_len=1000,
    )
    assert two_long == [5001, 5000, 1000, 1000, 1000]
    assert sum(two_long) == 13001


def test_dataset_one_long_uses_active_rows_at_decode_step_offset():
    requests = [
        runner.DatasetRequest(prompt_len=100, output_len=2048),
        runner.DatasetRequest(prompt_len=200, output_len=2048),
        runner.DatasetRequest(prompt_len=70000, output_len=2048),
        runner.DatasetRequest(prompt_len=90000, output_len=64),
    ]

    lens = runner.sample_dataset_batch(
        requests,
        mode="one_long",
        batch_size=3,
        decode_step_offset=128,
        rng=random.Random(0),
    )

    assert len(lens) == 3
    assert sum(length >= 64000 for length in lens) == 1
    assert sorted(length for length in lens if length < 8192) == [228, 328]


def test_dataset_sampling_rejects_inactive_rows_for_offset():
    requests = [runner.DatasetRequest(prompt_len=70000, output_len=64)]

    try:
        runner.sample_dataset_batch(
            requests,
            mode="long_heavy",
            batch_size=1,
            decode_step_offset=128,
            rng=random.Random(0),
        )
    except ValueError as exc:
        assert "not enough active long_heavy rows" in str(exc)
    else:
        raise AssertionError("expected inactive dataset row to be rejected")


def test_validate_case_allows_d_greater_than_batch_when_context_can_split():
    args = _args()
    case = runner.ProfileCase(
        batch_size=1,
        context_lens=[65536],
        dop=8,
        profile_kind="uniform",
    )
    assert runner.validate_case(args, case) == ""

    too_short = runner.ProfileCase(
        batch_size=1,
        context_lens=[4],
        dop=8,
        profile_kind="uniform",
    )
    assert runner.validate_case(args, too_short) == "L_i must be >= d=8"


def test_row_base_writes_plan_fields_and_legacy_aliases():
    args = _args()
    case = runner.ProfileCase(
        batch_size=2,
        context_lens=[1024, 2048],
        dop=4,
        profile_kind="dataset_replay",
        dataset_name="dpsk",
        dataset_mode="natural",
        sample_seed=123,
        sample_repeat=2,
        decode_step_offset=128,
    )

    row = runner.row_base(args, case, viable=True, node_local=False)

    assert row["profile_kind"] == "dataset_replay"
    assert row["dataset_name"] == "dpsk"
    assert row["B"] == 2
    assert row["L_avg"] == 1536
    assert row["L_p90"] == 2048
    assert row["W_attn"] == 3072
    assert row["d_attn"] == 4
    assert row["node_local"] is False
    assert row["batch_size"] == 2
    assert row["context_len"] == ""
    assert row["w_attn"] == 3072
    assert row["dop"] == 4


def test_infer_node_local_rejects_cross_group_and_outside_comm():
    assert runner.infer_node_local(
        occupied=list(range(8)),
        append=list(range(8)),
        sp_send_counts=[0] * 16,
        sp_recv_counts=[0] * 16,
        local_group_size=8,
    )

    assert not runner.infer_node_local(
        occupied=[7, 8],
        append=[7, 8],
        sp_send_counts=[0] * 16,
        sp_recv_counts=[0] * 16,
        local_group_size=8,
    )

    send_counts = [0] * 16
    send_counts[9] = 1
    assert not runner.infer_node_local(
        occupied=[0, 1],
        append=[0, 1],
        sp_send_counts=send_counts,
        sp_recv_counts=[0] * 16,
        local_group_size=8,
    )


def test_write_results_includes_new_profile_columns(tmp_path):
    args = _args()
    case = runner.ProfileCase(
        batch_size=1,
        context_lens=[65536],
        dop=8,
        profile_kind="uniform",
        w_attn_bucket=65536,
    )
    row = runner.row_base(args, case, viable=False, skip_reason="allocator failed")
    out = tmp_path / "profile.jsonl"

    runner.write_results(out, [row])

    assert '"d_attn": 8' in out.read_text()
    csv_text = out.with_suffix(".csv").read_text()
    header = csv_text.splitlines()[0].split(",")
    assert "profile_kind" in header
    assert "L_p90" in header
    assert "W_attn" in header
    assert "viable" in header
    assert "skip_reason" in header
    assert "node_local" in header
