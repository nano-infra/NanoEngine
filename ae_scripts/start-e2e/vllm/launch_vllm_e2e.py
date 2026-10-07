#!/usr/bin/env python3
"""Launch reproducible four-node vLLM E2E cases.

This entry point owns only the service experiment.  Figure-specific snapshot
selection and token/batch conversion intentionally live under each figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
AE_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(AE_ROOT))

from ae_utils.paths import require_path

RUNNER_PATH = SCRIPT_DIR / "manual_multinode_poisson_runner.py"
DEFAULT_HARNESS = Path(
    os.environ.get(
        "VLLM_E2E_HARNESS",
        str(SCRIPT_DIR / "offline_poisson_harness.py"),
    )
)
DEFAULT_ARTIFACT_ROOT = Path(
    os.environ.get(
        "VLLM_E2E_ARTIFACT_ROOT",
        "/vllm/offline_bench/manual_multinode",
    )
)
DEFAULT_MODEL_PATH = Path(
    os.environ.get("VLLM_E2E_MODEL_PATH") or require_path("AE_DPSK_MODEL")
)
DEFAULT_DATASET_PATH = Path(
    os.environ.get("VLLM_E2E_DATASET_PATH")
    or (
        Path(require_path("AE_DATASET_MIXLONG_0326"))
        / "sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv"
    )
)
DEFAULT_VLLM_WORKDIR = os.environ.get("VLLM_WORKDIR", "/vllm")
DEFAULT_MODEL_NAME = os.environ.get(
    "VLLM_E2E_MODEL_NAME", "deepseek_v3_1024k"
)
DEFAULT_DATASET_NAME = "issue01_random"
DEFAULT_CLUSTER = "4node_h200"
DEFAULT_DURATION_SEC = 600.0
DEFAULT_MAX_MODEL_LEN = 1_000_000
DEFAULT_WARMUP_REQUESTS = 32
DEFAULT_DATA_PARALLEL_RPC_PORT = 29550
DEFAULT_REMOTE_HOSTS = (
    tuple(os.environ["VLLM_4NODE_REMOTE_HOSTS"].replace(",", " ").split())
    if os.environ.get("VLLM_4NODE_REMOTE_HOSTS")
    else None
)

CSV_FIELDNAMES = (
    "enabled",
    "name",
    "cluster",
    "model",
    "dataset",
    "strategy",
    "dispatch_policy",
    "request_rate",
    "rate_phase",
    "max_num_seqs",
    "gpu_memory_utilization",
    "max_requests",
    "warmup_requests",
    "max_model_len",
    "data_parallel_rpc_port",
    "reason",
    "historical_reference",
)


@dataclass(frozen=True)
class CaseRow:
    enabled: int
    name: str
    cluster: str
    model: str
    dataset: str
    strategy: str
    dispatch_policy: str
    request_rate: float
    rate_phase: str
    max_num_seqs: int
    gpu_memory_utilization: float
    max_requests: int
    warmup_requests: int
    max_model_len: int
    data_parallel_rpc_port: int
    reason: str = ""
    historical_reference: str = ""


@dataclass(frozen=True)
class Fig5Case:
    part: str
    dispatch_policy: str
    request_rate: float
    gpu_memory_utilization: float
    purpose: str


FIG5_CASES = (
    Fig5Case(
        part="attention",
        dispatch_policy="waiting_x4_plus_running",
        request_rate=30.0,
        gpu_memory_utilization=0.90,
        purpose="raw per-rank KV usage for the Attention panel",
    ),
    Fig5Case(
        part="deepep",
        dispatch_policy="least_cache",
        request_rate=30.0,
        gpu_memory_utilization=0.87,
        purpose="raw per-rank running-request counts for the DeepEP panel",
    ),
    Fig5Case(
        part="hol",
        dispatch_policy="waiting_x4_plus_running",
        request_rate=45.0,
        gpu_memory_utilization=0.90,
        purpose="raw queue and KV time series for the HoL panel",
    ),
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("expected a positive number")
    return parsed


def utilization(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("expected a value in (0, 1]")
    return parsed


def parse_rates(value: str) -> tuple[float, ...]:
    rates = tuple(positive_float(part) for part in value.replace(",", " ").split())
    if not rates:
        raise argparse.ArgumentTypeError("expected at least one request rate")
    return rates


def parse_remote_hosts(value: str) -> tuple[str, ...]:
    hosts = tuple(value.replace(",", " ").split())
    if not hosts:
        raise argparse.ArgumentTypeError("expected at least one remote host")
    return hosts


def stringify_number(value: float) -> str:
    return f"{value:g}"


def max_requests(rate: float, duration_sec: float) -> int:
    return max(1, int(round(rate * duration_sec)))


def make_case(
    *,
    name: str,
    model_name: str,
    dataset_name: str,
    strategy: str,
    dispatch_policy: str,
    request_rate: float,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    duration_sec: float,
    max_model_len: int,
    warmup_requests: int,
    purpose: str,
) -> CaseRow:
    return CaseRow(
        enabled=1,
        name=name,
        cluster=DEFAULT_CLUSTER,
        model=model_name,
        dataset=dataset_name,
        strategy=strategy,
        dispatch_policy=dispatch_policy,
        request_rate=request_rate,
        rate_phase=name,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        max_requests=max_requests(request_rate, duration_sec),
        warmup_requests=warmup_requests,
        max_model_len=max_model_len,
        data_parallel_rpc_port=DEFAULT_DATA_PARALLEL_RPC_PORT,
        reason=purpose,
    )


def build_fig5_cases(args: argparse.Namespace) -> list[CaseRow]:
    selected_parts = set(args.fig5_case)
    if "all" in selected_parts:
        selected_parts = {case.part for case in FIG5_CASES}

    rows: list[CaseRow] = []
    for case in FIG5_CASES:
        if case.part not in selected_parts:
            continue
        rows.append(
            make_case(
                name=f"fig5_vllm_{case.part}_dp32_rate{stringify_number(case.request_rate)}",
                model_name="deepseek_v3_1024k",
                dataset_name=args.dataset_name,
                strategy="dp32",
                dispatch_policy=case.dispatch_policy,
                request_rate=case.request_rate,
                gpu_memory_utilization=case.gpu_memory_utilization,
                max_num_seqs=256,
                duration_sec=600.0,
                max_model_len=1_000_000,
                warmup_requests=32,
                purpose=case.purpose,
            )
        )
    return rows


def build_custom_cases(args: argparse.Namespace) -> list[CaseRow]:
    return [
        make_case(
            name=(
                f"e2e_{args.dataset_name}_{args.strategy}_"
                f"{args.dispatch_policy}_rate{stringify_number(rate)}"
            ),
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            strategy=args.strategy,
            dispatch_policy=args.dispatch_policy,
            request_rate=rate,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs,
            duration_sec=args.bench_duration_sec,
            max_model_len=args.max_model_len,
            warmup_requests=args.warmup_requests,
            purpose="custom reusable E2E service case",
        )
        for rate in args.request_rates
    ]


def write_case_csv(path: Path, rows: list[CaseRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            payload = asdict(row)
            payload["request_rate"] = stringify_number(row.request_rate)
            payload["gpu_memory_utilization"] = stringify_number(
                row.gpu_memory_utilization
            )
            writer.writerow(payload)


def apply_runtime_overrides(args: argparse.Namespace) -> object:
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    import manual_multinode_poisson_runner as runner

    runner.MODELS[args.model_name] = str(args.model_path.expanduser().resolve())
    runner.DATASETS[args.dataset_name] = str(
        args.dataset_path.expanduser().resolve()
    )
    runner.HARNESS_ENTRYPOINT = str(args.harness.expanduser().resolve())

    cluster = runner.CLUSTERS[DEFAULT_CLUSTER]
    overrides: dict[str, object] = {"workdir": args.vllm_workdir}
    if args.master_addr is not None:
        overrides["master_addr"] = args.master_addr
    if args.remote_hosts is not None:
        overrides["remote_hosts"] = args.remote_hosts
    if args.shell is not None:
        overrides["local_shell"] = args.shell
        overrides["remote_shell"] = args.shell
    if args.no_env_script:
        overrides["local_env_script"] = None
        overrides["remote_env_script"] = None
    elif args.env_script is not None:
        env_script = str(args.env_script.expanduser().resolve())
        overrides["local_env_script"] = env_script
        overrides["remote_env_script"] = env_script
    if args.ssh_config is not None:
        ssh_config = str(args.ssh_config.expanduser().resolve())
        ssh_opts = list(cluster.ssh_opts)
        try:
            config_index = ssh_opts.index("-F") + 1
        except ValueError:
            ssh_opts[:0] = ["-F", ssh_config]
        else:
            ssh_opts[config_index] = ssh_config
        overrides["ssh_opts"] = tuple(ssh_opts)
    runner.CLUSTERS[DEFAULT_CLUSTER] = replace(cluster, **overrides)
    return runner


def write_fig5_run_environment(
    artifact_root: Path,
    run_label: str,
    run_id: str,
    master_addr: str,
    runner: object,
) -> Path | None:
    run_root = artifact_root / "_runs"
    if not run_root.is_dir():
        return None
    run_suffix = f"__{runner.sanitize_tag(run_label)}"
    manifests = [
        candidate / "run_manifest.json"
        for candidate in run_root.iterdir()
        if candidate.is_dir() and candidate.name.endswith(run_suffix)
    ]
    if not manifests:
        return None
    manifest_path = max(manifests, key=lambda path: path.stat().st_mtime_ns)
    with manifest_path.open(encoding="utf-8") as input_file:
        manifest = json.load(input_file)

    values = {
        "RUN_ID": run_id,
        "MASTER_ADDR": master_addr,
        "FIG5_RUN_MANIFEST": str(manifest_path.resolve()),
    }
    for result in manifest.get("results", []):
        if result.get("status") != "ok":
            continue
        case_name = str(result.get("case_name", ""))
        case_dir = result.get("case_dir")
        if not case_dir:
            continue
        for part in ("attention", "deepep", "hol"):
            if f"fig5_vllm_{part}_" in case_name:
                values[f"{part.upper()}_LOG"] = str(
                    (Path(case_dir) / "frontend.log").resolve()
                )
                break

    required = {"ATTENTION_LOG", "DEEPEP_LOG", "HOL_LOG"}
    if not required.issubset(values):
        return None
    environment_path = artifact_root / f"fig5_4node_{run_id}_env.sh"
    lines = [
        "# Generated by start-e2e/vllm/launch_vllm_e2e.py.",
        *(f"export {name}={shlex.quote(value)}" for name, value in values.items()),
        "",
    ]
    temporary_path = environment_path.with_name(
        f".{environment_path.name}.{os.getpid()}.tmp"
    )
    temporary_path.write_text("\n".join(lines), encoding="utf-8")
    temporary_path.replace(environment_path)
    return environment_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch four-node vLLM E2E service cases and retain raw artifacts. "
            "No figure-specific snapshot extraction is performed."
        )
    )
    parser.add_argument("--preset", choices=("fig5",), help="Named paper case set.")
    parser.add_argument(
        "--fig5-case",
        action="append",
        choices=("all", "attention", "deepep", "hol"),
        default=[],
        help="Fig. 5 source case to run; repeat as needed. Defaults to all.",
    )
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=(
            "Logical model key used in manifests/artifact layout, for example "
            "deepseek_v3_1024k or kimi_k2_instruct_0905."
        ),
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--harness", type=Path, default=DEFAULT_HARNESS)
    parser.add_argument("--vllm-workdir", default=DEFAULT_VLLM_WORKDIR)
    parser.add_argument(
        "--master-addr",
        default=None,
        help="Head-container address reachable by all worker containers.",
    )
    parser.add_argument(
        "--remote-hosts",
        type=parse_remote_hosts,
        default=DEFAULT_REMOTE_HOSTS,
        help=(
            "Comma- or space-separated worker SSH aliases. Can also be set "
            "with VLLM_4NODE_REMOTE_HOSTS."
        ),
    )
    parser.add_argument("--ssh-config", type=Path, default=None)
    parser.add_argument(
        "--shell",
        choices=("bash", "zsh"),
        default=os.environ.get("VLLM_MULTINODE_SHELL"),
    )
    env_group = parser.add_mutually_exclusive_group()
    env_group.add_argument(
        "--env-script",
        type=Path,
        default=(
            Path(os.environ["VLLM_MULTINODE_ENV_SCRIPT"])
            if os.environ.get("VLLM_MULTINODE_ENV_SCRIPT")
            else None
        ),
    )
    env_group.add_argument("--no-env-script", action="store_true")

    custom = parser.add_argument_group("custom case settings")
    custom.add_argument("--strategy", default="dp32")
    custom.add_argument(
        "--dispatch-policy",
        choices=("waiting_x4_plus_running", "least_batch", "least_cache"),
        default="waiting_x4_plus_running",
    )
    custom.add_argument(
        "--request-rates", type=parse_rates, default=(30.0,), metavar="RATES"
    )
    custom.add_argument("--gpu-memory-utilization", type=utilization, default=0.90)
    custom.add_argument("--max-num-seqs", type=positive_int, default=256)
    custom.add_argument(
        "--bench-duration-sec", type=positive_float, default=DEFAULT_DURATION_SEC
    )
    custom.add_argument(
        "--max-model-len", type=positive_int, default=DEFAULT_MAX_MODEL_LEN
    )
    custom.add_argument(
        "--warmup-requests",
        type=non_negative_int,
        default=DEFAULT_WARMUP_REQUESTS,
    )

    parser.add_argument("--run-label", default=None)
    parser.add_argument(
        "--run-id",
        default="ae_4node1",
        help="Figure 5 run name used by the generated environment file.",
    )
    parser.add_argument("--case-csv-output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ignore-historical-skips", action="store_true")
    parser.add_argument("--no-keep-going", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.fig5_case and args.preset != "fig5":
        parser.error("--fig5-case requires --preset fig5")
    if args.preset == "fig5" and not args.fig5_case:
        args.fig5_case = ["all"]
    if not args.run_id or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in args.run_id
    ):
        parser.error("--run-id contains unsupported characters")
    rows = build_fig5_cases(args) if args.preset == "fig5" else build_custom_cases(args)
    if not rows:
        parser.error("no E2E cases were selected")

    run_label = args.run_label or (
        f"fig5-4node-{args.run_id}"
        if args.preset == "fig5"
        else "custom-vllm-e2e"
    )
    case_csv = args.case_csv_output
    if case_csv is None:
        case_csv = args.artifact_root / "_case_csv" / f"{run_label}_cases.csv"
    case_csv = case_csv.expanduser().resolve()
    write_case_csv(case_csv, rows)

    runner = apply_runtime_overrides(args)
    runner_argv = [
        "--case-csv",
        str(case_csv),
        "--artifact-root",
        str(args.artifact_root.expanduser().resolve()),
        "--run-label",
        run_label,
    ]
    if args.dry_run:
        runner_argv.append("--dry-run")
    if args.ignore_historical_skips:
        runner_argv.append("--ignore-historical-skips")
    if args.no_keep_going:
        runner_argv.append("--no-keep-going")

    print(f"Case CSV: {case_csv}", flush=True)
    for row in rows:
        print(
            f"  {row.name}: policy={row.dispatch_policy}, "
            f"rate={row.request_rate:g}, mem={row.gpu_memory_utilization:g}, "
            f"max_num_seqs={row.max_num_seqs}",
            flush=True,
        )
    exit_code = 0
    try:
        runner.main(runner_argv)
    except SystemExit as exc:
        if exc.code is None:
            exit_code = 0
        elif isinstance(exc.code, int):
            exit_code = exc.code
        else:
            print(exc.code, file=sys.stderr)
            exit_code = 1
    if exit_code != 0:
        return exit_code

    complete_fig5_run = "all" in args.fig5_case or set(args.fig5_case) == {
        "attention",
        "deepep",
        "hol",
    }
    if args.preset == "fig5" and not args.dry_run and complete_fig5_run:
        cluster = runner.CLUSTERS[DEFAULT_CLUSTER]
        environment_path = write_fig5_run_environment(
            args.artifact_root.expanduser().resolve(),
            run_label,
            args.run_id,
            cluster.master_addr,
            runner,
        )
        if environment_path is None:
            print(
                "The complete Figure 5 run environment was not written.",
                file=sys.stderr,
            )
            return 1
        try:
            display_path = environment_path.relative_to(
                SCRIPT_DIR.parent.parent
            )
        except ValueError:
            display_path = environment_path
        print(f"Run environment: {display_path}", flush=True)
        print(f"Next: source {shlex.quote(str(display_path))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
