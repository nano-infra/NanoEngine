"""Figure 12 source-experiment configuration."""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path


FIG12_DIR = Path(__file__).resolve().parent
AE_ROOT = FIG12_DIR.parent
sys.path.insert(0, str(AE_ROOT))

from ae_utils.paths import require_path

NANODEPLOY_WORKDIR = Path(os.environ.get("NANODEPLOY_WORKDIR") or AE_ROOT.parent)
NANO_RAY_ADDR = os.environ.get("NANO_RAY_ADDR", "10.102.252.174:6380")
NANO_MASTER_ADDR = os.environ.get("NANO_MASTER_ADDR", "10.102.252.174:29500")
VLLM_MASTER_ADDR = os.environ.get("FIG12_VLLM_MASTER_ADDR", "10.102.252.174")
VLLM_REMOTE_HOSTS = os.environ.get(
    "VLLM_4NODE_REMOTE_HOSTS",
    "h200-rjob1,h200-rjob2,h200-rjob3",
)
# Keep workload metadata importable without configuring unused experiments.
MODEL_PATH_KEYS = {
    "deepseek_v3_1024k": ("FIG12_DPSK_MODEL", "AE_DPSK_MODEL"),
    "kimi_k2_instruct_0905": ("FIG12_KIMI_MODEL", "AE_KIMI_MODEL"),
}
DATASET_PATH_KEYS = {
    "short_random": (
        "FIG12_SHAREGPT4O_DATASET",
        "sharegpt-4o/sharegpt4o-mixed-random-60k.csv",
    ),
    "issue01_random": (
        "FIG12_ISSUE1_DATASET",
        "sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv",
    ),
    "issue05_random": (
        "FIG12_ISSUE5_DATASET",
        "sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.05_n60000_60k.csv",
    ),
    "long_full": (
        "FIG12_GEMINI_ISSUES_DATASET",
        "madha/Gemini_Issues_Stats_rename-shuffle.csv",
    ),
}


def get_vllm_workdir() -> Path:
    return Path(os.environ.get("VLLM_WORKDIR") or require_path("AE_VLLM_ROOT"))


@dataclass(frozen=True)
class Workload:
    slug: str
    label: str
    model_name: str
    dataset_name: str
    nano_rates: tuple[float, ...]
    vllm_rates: tuple[float, ...]
    least_cache_memory: float

    @property
    def model_path(self) -> Path:
        override_key, path_key = MODEL_PATH_KEYS[self.model_name]
        return Path(os.environ.get(override_key) or require_path(path_key))

    @property
    def dataset_path(self) -> Path:
        override_key, relative_path = DATASET_PATH_KEYS[self.dataset_name]
        override = os.environ.get(override_key)
        if override:
            return Path(override)
        return require_path("AE_DATASET_ROOT") / relative_path


WORKLOADS = (
    Workload(
        slug="dpsk_sharegpt4o",
        label="DPSK-ShareGPT4o",
        model_name="deepseek_v3_1024k",
        dataset_name="short_random",
        nano_rates=(20, 40, 60, 80, 90, 100, 110, 115, 120, 125, 130, 140),
        vllm_rates=(20, 40, 60, 80, 100, 120, 140, 160),
        least_cache_memory=0.87,
    ),
    Workload(
        slug="dpsk_issue1",
        label="DPSK-Issue1%",
        model_name="deepseek_v3_1024k",
        dataset_name="issue01_random",
        nano_rates=(10, 20, 30, 35, 40, 50, 60, 70, 80, 90, 100),
        vllm_rates=(10, 15, 20, 25, 30, 35, 40, 45, 50, 60),
        least_cache_memory=0.87,
    ),
    Workload(
        slug="dpsk_issue5",
        label="DPSK-Issue5%",
        model_name="deepseek_v3_1024k",
        dataset_name="issue05_random",
        nano_rates=(2.5, 5, 10, 15, 17.5, 20, 25, 30, 35, 40, 45, 50, 60, 80),
        vllm_rates=(2.5, 5, 7.5, 10, 12.5, 15, 16, 17, 17.5, 20, 22.5, 25),
        least_cache_memory=0.87,
    ),
    Workload(
        slug="dpsk_gemini_issues",
        label="DPSK-Gemini Issues",
        model_name="deepseek_v3_1024k",
        dataset_name="long_full",
        nano_rates=(0.25, 0.5, 1, 1.5, 1.75, 2, 2.5, 4, 10),
        vllm_rates=(0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3),
        least_cache_memory=0.87,
    ),
    Workload(
        slug="kimi_issue1",
        label="KIMI-Issue1%",
        model_name="kimi_k2_instruct_0905",
        dataset_name="issue01_random",
        nano_rates=(10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110),
        vllm_rates=(10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 57, 58, 60, 65),
        least_cache_memory=0.90,
    ),
    Workload(
        slug="kimi_issue5",
        label="KIMI-Issue5%",
        model_name="kimi_k2_instruct_0905",
        dataset_name="issue05_random",
        nano_rates=(2.5, 5, 10, 20, 25, 30, 35, 40, 45, 50),
        vllm_rates=(2.5, 3, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30),
        least_cache_memory=0.90,
    ),
)


@dataclass(frozen=True)
class VllmBaseline:
    slug: str
    strategy: str
    dispatch_policy: str
    max_num_seqs: int
    gpu_memory_utilization: float | None


VLLM_BASELINES = (
    VllmBaseline("dp_least_batch", "dp32", "waiting_x4_plus_running", 256, 0.90),
    VllmBaseline("dp_least_cache", "dp32", "least_cache", 256, None),
    VllmBaseline("cp2", "dp16cp2", "waiting_x4_plus_running", 384, 0.85),
    VllmBaseline("cp4", "dp8dcp4", "waiting_x4_plus_running", 768, 0.85),
    VllmBaseline("cp8", "dp4dcp8", "waiting_x4_plus_running", 1024, 0.85),
)


def selected_workloads(values: list[str]) -> tuple[Workload, ...]:
    if not values or "all" in values:
        return WORKLOADS
    selected = set(values)
    return tuple(workload for workload in WORKLOADS if workload.slug in selected)


def format_rates(rates: tuple[float, ...]) -> str:
    return " ".join(f"{rate:g}" for rate in rates)


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
    with source.open("r", encoding="utf-8", newline="") as source_file:
        reader = csv.DictReader(source_file)
        fieldnames = reader.fieldnames
        if not fieldnames or not {"prompt_len", "output_len"}.issubset(fieldnames):
            raise ValueError(
                f"dataset must contain prompt_len and output_len: {source}"
            )
        with temporary.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            for row_number, row in enumerate(reader, start=2):
                total_rows += 1
                try:
                    request_tokens = int(row["prompt_len"]) + int(row["output_len"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"invalid request lengths at {source}:{row_number}"
                    ) from error
                if request_tokens <= max_request_tokens:
                    writer.writerow(row)
                    kept_rows += 1
    if kept_rows == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"no requests fit max_request_tokens={max_request_tokens}: {source}"
        )
    temporary.replace(destination)
    return total_rows, kept_rows
