#!/usr/bin/env python3
"""Run the five service cases needed to reproduce Figure 14."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
AE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AE_ROOT))

from ae_utils.paths import require_path

NANO_LAUNCHER = SCRIPT_DIR / "run_nano_e2e.py"
VLLM_LAUNCHER = AE_ROOT / "start-e2e" / "vllm" / "launch_vllm_e2e.py"
CASE_CSV = SCRIPT_DIR / "fig14_cases.csv"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "results"
GPUS_PER_NODE = 8
PAPER_NUM_NODES = 4
DEFAULT_MODEL_PATH = Path(
    os.environ.get("FIG14_MODEL_PATH") or require_path("AE_DPSK_MODEL")
)

VLLM_ARTIFACT_NAMES = {
    "lb_vllm": "lb_least_batch",
    "lb_vllm_least_cache": "lb_least_cache",
    "hol_vllm": "hol_least_batch",
}


@dataclass(frozen=True)
class VllmCase:
    case_name: str
    artifact_name: str
    dataset_name: str
    dataset_path: Path
    dispatch_policy: str
    request_rate: str
    duration_sec: str
    max_num_seqs: str
    gpu_memory_utilization: str


def parse_remote_hosts(value: str) -> tuple[str, ...]:
    hosts = tuple(value.replace(",", " ").split())
    if not hosts:
        raise argparse.ArgumentTypeError("expected at least one remote host")
    return hosts


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        required=True,
        help="Fresh run ID, for example ae_fig14_1.",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=("nano", "vllm"),
        default=("nano", "vllm"),
        help="Systems to run sequentially (default: nano vllm).",
    )
    parser.add_argument(
        "--nano-cases",
        nargs="+",
        choices=("lb", "hol"),
        default=("lb", "hol"),
        help="NanoDeploy cases to run (default: lb hol).",
    )
    parser.add_argument(
        "--vllm-cases",
        nargs="+",
        choices=tuple(VLLM_ARTIFACT_NAMES),
        default=tuple(VLLM_ARTIFACT_NAMES),
        help="vLLM cases to run (default: all three).",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--num-nodes",
        type=int,
        choices=(2, 4),
        default=PAPER_NUM_NODES,
        help="Number of 8-GPU nodes to use (default: %(default)s).",
    )
    parser.add_argument(
        "--max-request-tokens",
        type=positive_int,
        help=(
            "Maximum prompt_len + output_len for an optional reduced run; "
            "longer CSV rows are removed. Disabled by default."
        ),
    )
    parser.add_argument(
        "--nano-ray-address",
        default=os.environ.get(
            "FIG14_NANO_RAY_ADDRESS", "10.102.252.174:6380"
        ),
    )
    parser.add_argument(
        "--nano-master-address",
        default=os.environ.get(
            "FIG14_NANO_MASTER_ADDRESS", "10.102.252.174:29500"
        ),
    )
    parser.add_argument(
        "--vllm-master-address",
        default=os.environ.get(
            "VLLM_4NODE_H200_MASTER_ADDR", "10.102.252.174"
        ),
    )
    parser.add_argument(
        "--remote-hosts",
        type=parse_remote_hosts,
        default=parse_remote_hosts(
            os.environ.get(
                "FIG14_VLLM_REMOTE_HOSTS",
                "h200-rjob1,h200-rjob3,h200-rjob4",
            )
        ),
    )
    parser.add_argument(
        "--vllm-workdir",
        default=os.environ.get("VLLM_WORKDIR", "/vllm"),
    )
    parser.add_argument(
        "--shell",
        choices=("bash", "zsh"),
        default="zsh",
        help="Local and remote shell for vLLM launch (default: zsh).",
    )
    args = parser.parse_args()
    if any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in args.run_id
    ):
        parser.error("--run-id contains unsupported characters")
    args.systems = tuple(dict.fromkeys(args.systems))
    args.nano_cases = tuple(dict.fromkeys(args.nano_cases))
    args.vllm_cases = tuple(dict.fromkeys(args.vllm_cases))
    remote_count = args.num_nodes - 1
    if "vllm" in args.systems and len(args.remote_hosts) < remote_count:
        parser.error(
            f"--num-nodes {args.num_nodes} requires at least {remote_count} "
            "--remote-hosts entries"
        )
    args.remote_hosts = args.remote_hosts[:remote_count]
    return args


def load_vllm_cases() -> dict[str, VllmCase]:
    with CASE_CSV.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))

    cases: dict[str, VllmCase] = {}
    for row in rows:
        case_name = row["case"]
        if row["system"] != "vLLM":
            continue
        if case_name not in VLLM_ARTIFACT_NAMES:
            raise ValueError(f"Unexpected vLLM case in {CASE_CSV}: {case_name}")
        dataset = row["dataset"]
        cases[case_name] = VllmCase(
            case_name=case_name,
            artifact_name=VLLM_ARTIFACT_NAMES[case_name],
            dataset_name=f"{dataset}_random",
            dataset_path=SCRIPT_DIR / "inputs" / f"{dataset}.csv",
            dispatch_policy=row["dispatch_policy"],
            request_rate=row["request_rate"],
            duration_sec=row["duration_sec"],
            max_num_seqs=row["max_num_seqs"],
            gpu_memory_utilization=row["gpu_memory_utilization"],
        )

    missing = set(VLLM_ARTIFACT_NAMES) - set(cases)
    if missing:
        raise ValueError(
            f"Missing vLLM cases in {CASE_CSV}: {', '.join(sorted(missing))}"
        )
    return cases


def validate_inputs(args: argparse.Namespace, vllm_cases: dict[str, VllmCase]) -> None:
    required_files = [CASE_CSV]
    if "nano" in args.systems:
        required_files.append(NANO_LAUNCHER)
    if "vllm" in args.systems:
        required_files.append(VLLM_LAUNCHER)
        required_files.extend(
            vllm_cases[name].dataset_path for name in args.vllm_cases
        )
    missing_files = [str(path) for path in required_files if not path.is_file()]
    if missing_files:
        raise SystemExit("Required file not found:\n  " + "\n  ".join(missing_files))
    if not args.model_path.is_dir():
        raise SystemExit(f"Model directory not found: {args.model_path}")


def filter_dataset_by_request_tokens(
    source: Path,
    destination: Path,
    max_request_tokens: int,
) -> tuple[int, int]:
    """Keep rows whose prompt and requested output fit the token limit."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    total_rows = 0
    kept_rows = 0
    try:
        with source.open(newline="", encoding="utf-8") as input_file:
            reader = csv.DictReader(input_file)
            if reader.fieldnames is None:
                raise ValueError(f"CSV header not found: {source}")
            with temporary.open("w", newline="", encoding="utf-8") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=reader.fieldnames)
                writer.writeheader()
                for row_number, row in enumerate(reader, start=2):
                    total_rows += 1
                    try:
                        request_tokens = int(row["prompt_len"]) + int(
                            row["output_len"]
                        )
                    except (KeyError, TypeError, ValueError) as error:
                        raise ValueError(
                            f"invalid request lengths at {source}:{row_number}"
                        ) from error
                    if request_tokens <= max_request_tokens:
                        writer.writerow(row)
                        kept_rows += 1
        if kept_rows == 0:
            raise ValueError(
                f"no requests fit max_request_tokens={max_request_tokens}: "
                f"{source}"
            )
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return total_rows, kept_rows


def prepare_vllm_cases(
    args: argparse.Namespace,
    run_root: Path,
    vllm_cases: dict[str, VllmCase],
) -> dict[str, VllmCase]:
    if args.max_request_tokens is None or "vllm" not in args.systems:
        return vllm_cases

    prepared_paths: dict[Path, Path] = {}
    for name in args.vllm_cases:
        source = vllm_cases[name].dataset_path.resolve()
        if source in prepared_paths:
            continue
        destination = (
            run_root
            / "_filtered_csv"
            / f"{source.stem}_max{args.max_request_tokens}.csv"
        )
        total_rows, kept_rows = filter_dataset_by_request_tokens(
            source,
            destination,
            args.max_request_tokens,
        )
        prepared_paths[source] = destination.resolve()
        print(
            f"[dataset] {source.stem}: kept {kept_rows}/{total_rows}, "
            f"removed {total_rows - kept_rows} above "
            f"{args.max_request_tokens} tokens",
            flush=True,
        )

    return {
        name: replace(
            case,
            dataset_path=prepared_paths.get(
                case.dataset_path.resolve(), case.dataset_path
            ),
        )
        for name, case in vllm_cases.items()
    }


def nano_command(
    args: argparse.Namespace,
    output_root: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(NANO_LAUNCHER),
        "--run-id",
        args.run_id,
        "--output-root",
        str(output_root),
        "--model-path",
        str(args.model_path),
        "--ray-address",
        args.nano_ray_address,
        "--master-address",
        args.nano_master_address,
        "--num-nodes",
        str(args.num_nodes),
    ]
    if args.max_request_tokens is not None:
        command.extend(
            ["--max-request-tokens", str(args.max_request_tokens)]
        )
    command.extend(["--cases", *args.nano_cases])
    return command


def vllm_command(
    args: argparse.Namespace,
    run_root: Path,
    case: VllmCase,
) -> list[str]:
    command = [
        sys.executable,
        str(VLLM_LAUNCHER),
        "--artifact-root",
        str(run_root / "vllm" / case.artifact_name),
        "--model-name",
        "deepseek_v3_1024k",
        "--model-path",
        str(args.model_path),
        "--dataset-name",
        case.dataset_name,
        "--dataset-path",
        str(case.dataset_path),
        "--vllm-workdir",
        args.vllm_workdir,
        "--master-addr",
        args.vllm_master_address,
        "--remote-hosts",
        ",".join(args.remote_hosts),
        "--shell",
        args.shell,
        "--strategy",
        f"dp{args.num_nodes * GPUS_PER_NODE}",
        "--dispatch-policy",
        case.dispatch_policy,
        "--request-rates",
        case.request_rate,
        "--gpu-memory-utilization",
        case.gpu_memory_utilization,
        "--max-num-seqs",
        case.max_num_seqs,
        "--bench-duration-sec",
        case.duration_sec,
        "--max-model-len",
        str(args.max_request_tokens or 1_000_000),
        "--warmup-requests",
        "32",
        "--run-label",
        f"fig14-{args.run_id}-{case.artifact_name}",
        "--ignore-historical-skips",
        "--no-keep-going",
    ]
    return command


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot resume from manifest {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid manifest object: {path}")
    return payload


def normalize_command(command: object) -> list[str] | None:
    if not isinstance(command, list) or not all(
        isinstance(item, str) for item in command
    ):
        return None
    normalized = ["--run-id" if item == "--run-name" else item for item in command]
    index = 0
    while index + 1 < len(normalized):
        if normalized[index : index + 2] == ["--num-nodes", "4"]:
            del normalized[index : index + 2]
            continue
        index += 1
    return normalized


def benchmark_summary_is_complete(summary_path: Path) -> bool:
    requests_path = summary_path.with_name("requests.jsonl")
    frontend_path = summary_path.parent.parent / "frontend.log"
    if any(
        not path.is_file() or path.stat().st_size == 0
        for path in (requests_path, frontend_path)
    ):
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        total = int(summary["total_requests"])
        successful = int(summary["successful_requests"])
        failed = int(summary.get("failed_requests", 0))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False
    return total > 0 and successful == total and failed == 0


def vllm_component_is_complete(
    component: dict[str, Any],
    command: list[str],
    artifact_root: Path,
) -> bool:
    if component.get("status") != "completed" or component.get("exit_code") != 0:
        return False
    if normalize_command(component.get("command")) != normalize_command(command):
        return False
    return any(
        benchmark_summary_is_complete(summary_path)
        for summary_path in artifact_root.rglob("summary.json")
        if summary_path.parent.name == "benchmark"
    )


def validate_resume_components(
    manifest: dict[str, Any],
    run_id: str,
    num_nodes: int,
    max_request_tokens: int | None,
    commands: list[tuple[str, list[str]]],
) -> None:
    recorded_run_id = manifest.get("run_id", manifest.get("run_name"))
    if recorded_run_id != run_id:
        raise RuntimeError(
            f"manifest run ID is {recorded_run_id!r}, expected {run_id!r}"
        )
    recorded_num_nodes = int(manifest.get("num_nodes", PAPER_NUM_NODES))
    if recorded_num_nodes != num_nodes:
        raise RuntimeError(
            f"manifest uses {recorded_num_nodes} nodes, requested {num_nodes}; "
            "choose a fresh --run-id"
        )
    components = manifest.get("components")
    if not isinstance(components, dict):
        raise RuntimeError("cannot resume: manifest components are missing")
    recorded_max_request_tokens = manifest.get("max_request_tokens")
    if recorded_max_request_tokens != max_request_tokens:
        completed_vllm = [
            name
            for name, component in components.items()
            if name.startswith("vllm/")
            and isinstance(component, dict)
            and component.get("status") == "completed"
        ]
        if completed_vllm or max_request_tokens is None:
            raise RuntimeError(
                "manifest uses max_request_tokens="
                f"{recorded_max_request_tokens!r}, requested "
                f"{max_request_tokens!r}; choose a fresh --run-id"
            )
    for name, command in commands:
        existing = components.get(name)
        if existing is None:
            continue
        if not isinstance(existing, dict):
            raise RuntimeError(
                f"cannot resume component {name} with different settings; "
                "choose a fresh --run-id"
            )
        if normalize_command(existing.get("command")) == normalize_command(command):
            continue
        if (
            name.startswith("vllm/")
            and existing.get("status") != "completed"
            and max_request_tokens is not None
        ):
            continue
        raise RuntimeError(
            f"cannot resume component {name} with different settings; "
            "choose a fresh --run-id"
        )


def run_component(name: str, command: list[str]) -> int:
    print(f"\n===== START {name} =====", flush=True)
    print(f"$ {shlex.join(command)}", flush=True)
    return subprocess.run(command, cwd=AE_ROOT, check=False).returncode


def run_all(
    args: argparse.Namespace,
    output_root: Path,
    vllm_cases: dict[str, VllmCase],
) -> int:
    run_root = output_root.resolve() / args.run_id
    manifest_path = run_root / "manifest.json"
    resuming = run_root.exists()
    if resuming and not manifest_path.is_file():
        raise RuntimeError(f"cannot resume without manifest: {manifest_path}")
    if not resuming:
        run_root.mkdir(parents=True)

    vllm_cases = prepare_vllm_cases(args, run_root, vllm_cases)
    commands: list[tuple[str, list[str]]] = []
    if "nano" in args.systems:
        commands.append(("nano", nano_command(args, output_root)))
    if "vllm" in args.systems:
        commands.extend(
            (
                f"vllm/{vllm_cases[name].artifact_name}",
                vllm_command(args, run_root, vllm_cases[name]),
            )
            for name in args.vllm_cases
        )

    if resuming:
        manifest = load_manifest(manifest_path)
        validate_resume_components(
            manifest,
            args.run_id,
            args.num_nodes,
            args.max_request_tokens,
            commands,
        )
        manifest["run_id"] = args.run_id
        manifest["num_nodes"] = args.num_nodes
        manifest["gpu_count"] = args.num_nodes * GPUS_PER_NODE
        if args.max_request_tokens is None:
            manifest.pop("max_request_tokens", None)
        else:
            manifest["max_request_tokens"] = args.max_request_tokens
        manifest.pop("run_name", None)
        manifest["status"] = "running"
        manifest["last_resumed_at"] = utc_now()
        manifest.pop("failed_component", None)
        manifest.pop("finished_at", None)
        manifest["systems"] = list(
            dict.fromkeys([*manifest.get("systems", []), *args.systems])
        )
        print(f"Resuming existing Fig. 14 run ID: {args.run_id}", flush=True)
    else:
        manifest = {
            "status": "running",
            "run_id": args.run_id,
            "num_nodes": args.num_nodes,
            "gpu_count": args.num_nodes * GPUS_PER_NODE,
            "started_at": utc_now(),
            "systems": list(args.systems),
            "components": {},
        }
        if args.max_request_tokens is not None:
            manifest["max_request_tokens"] = args.max_request_tokens
    write_manifest(manifest_path, manifest)

    try:
        for name, command in commands:
            existing = manifest["components"].get(name)
            if (
                name.startswith("vllm/")
                and isinstance(existing, dict)
                and vllm_component_is_complete(
                    existing,
                    command,
                    run_root / name,
                )
            ):
                print(f"{name} already complete; skipping", flush=True)
                continue
            component = {
                "status": "running",
                "started_at": utc_now(),
                "command": command,
            }
            manifest["components"][name] = component
            write_manifest(manifest_path, manifest)
            return_code = run_component(name, command)
            component["finished_at"] = utc_now()
            component["exit_code"] = return_code
            component["status"] = "completed" if return_code == 0 else "failed"
            write_manifest(manifest_path, manifest)
            if return_code != 0:
                manifest["status"] = "failed"
                manifest["failed_component"] = name
                manifest["finished_at"] = utc_now()
                write_manifest(manifest_path, manifest)
                return return_code
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["finished_at"] = utc_now()
        write_manifest(manifest_path, manifest)
        return 130

    manifest["status"] = "completed"
    manifest["finished_at"] = utc_now()
    write_manifest(manifest_path, manifest)
    try:
        display_root = run_root.relative_to(AE_ROOT)
    except ValueError:
        display_root = run_root
    print(f"\nFig. 14 service runs completed: {display_root}", flush=True)
    print(
        "Next: python3 fig14/plot_fig14_from_service_logs.py "
        f"--run-root {shlex.quote(str(display_root))} --output fig14/fig14",
        flush=True,
    )
    return 0


def main() -> int:
    args = parse_args()
    try:
        vllm_cases = load_vllm_cases()
    except (KeyError, OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    validate_inputs(args, vllm_cases)

    return run_all(args, args.output_root, vllm_cases)


if __name__ == "__main__":
    raise SystemExit(main())
