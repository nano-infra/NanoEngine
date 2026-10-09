"""CPU-only regression checks for experiment-specific AE path configuration."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
AE_ROOT = ROOT / "ae_scripts"


class AEPathConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ae-path-config-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "custom-checkpoint"
        self.model.mkdir()
        self.vllm = self.root / "vllm"
        self.vllm.mkdir()
        self.dataset_root = self.root / "dataset"
        self.mixlong = self.dataset_root / "sharegpt-4o-mixlong-0326"
        self.mixlong.mkdir(parents=True)
        self.dataset = self.mixlong / "sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv"
        self.dataset.write_text("prompt_len,output_len\n10,2\n")
        self.env_file = self.root / "paths.env"

    def run_code(self, code, settings=None, overrides=None):
        self.env_file.write_text("".join(
            f"{key}={value}\n" for key, value in (settings or {}).items()
        ))
        env = {
            key: value for key, value in os.environ.items()
            if not key.startswith(("AE_", "FIG12_", "VLLM_"))
        }
        env["AE_PATHS_ENV"] = str(self.env_file)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.update({key: str(value) for key, value in (overrides or {}).items()})
        setup = (
            "import sys\nfrom pathlib import Path\n"
            f"sys.path.insert(0, {str(AE_ROOT / 'start-e2e' / 'vllm')!r})\n"
            f"sys.path.insert(0, {str(AE_ROOT / 'fig12')!r})\n"
            f"model_path = Path({str(self.model)!r})\n"
            f"dataset_path = Path({str(self.dataset)!r})\n"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", setup + textwrap.dedent(code)],
            cwd=ROOT, env=env, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_deepseek_case_needs_only_its_model_and_dataset(self):
        self.run_code("""
            import manual_multinode_poisson_runner as runner
            case = runner.ExperimentCase(
                name="minimal_deepseek", cluster="2node_h200",
                strategy="dp2dcp8", model="deepseek_v3_1024k",
                dataset="issue01_random",
            )
            resolved = runner.resolve_case(case)
            assert resolved.model_path == model_path
            assert resolved.dataset_path == dataset_path
        """, {"AE_DPSK_MODEL": self.model, "AE_DATASET_MIXLONG_0326": self.mixlong})

    def test_kimi_uses_the_configured_checkpoint_directly(self):
        self.run_code("""
            import manual_multinode_poisson_runner as runner
            resolved = runner.resolve_alias_path(
                "kimi_k2_instruct_0905", runner.MODELS,
                label="Kimi", expect_file=False,
            )
            assert resolved == model_path
        """, {"AE_KIMI_MODEL": self.model}, {"AE_KIMI_MODEL_HF": "/unused/legacy-cache"})

    def test_unconfigured_optional_aliases_fail_only_when_selected(self):
        self.run_code("""
            import manual_multinode_poisson_runner as runner
            for alias, aliases, key, expect_file in (
                ("qwen3_235b_fp8", runner.MODELS, "AE_QWEN3_MODEL", False),
                ("qwen3_235b_fp8_1024k", runner.MODELS, "AE_QWEN3_MODEL_1024K", False),
                ("1k1k", runner.DATASETS, "AE_DATASET_0110", True),
                ("kimi_k2_instruct_0905", runner.MODELS, "AE_KIMI_MODEL", False),
            ):
                try:
                    runner.resolve_alias_path(alias, aliases, label=alias, expect_file=expect_file)
                except SystemExit as error:
                    assert f"{key} is not configured" in str(error), str(error)
                else:
                    raise AssertionError(f"missing {key} was accepted")
        """)

    def test_explicit_paths_and_launcher_overrides_need_no_default_paths(self):
        self.run_code("""
            import manual_multinode_poisson_runner as runner
            for model_name, dataset_name in (
                (str(model_path), str(dataset_path)),
                ("basic_test_deepseek", "basic_test_issue01"),
            ):
                runner.MODELS["basic_test_deepseek"] = str(model_path)
                runner.DATASETS["basic_test_issue01"] = str(dataset_path)
                resolved = runner.resolve_case(runner.ExperimentCase(
                    name="custom", cluster="2node_h200", strategy="dp2dcp8",
                    model=model_name, dataset=dataset_name,
                ))
                assert resolved.model_path == model_path
                assert resolved.dataset_path == dataset_path
        """)

    def test_fig12_metadata_does_not_require_model_or_dataset_paths(self):
        self.run_code("""
            import e2e_config as config
            assert len(config.WORKLOADS) == 6
            assert len(config.selected_workloads(["dpsk_issue1"])) == 1
        """)

    def test_fig12_deepseek_selection_does_not_read_kimi_paths(self):
        self.run_code("""
            import e2e_config as config
            workload, = config.selected_workloads(["dpsk_issue1"])
            assert workload.model_path == model_path
            assert workload.dataset_path == dataset_path
            import launch_vllm_e2e as launcher
            sys.argv = ["launch_vllm_e2e.py", "--workload", "dpsk_issue1", "--nodes", "2"]
            args = launcher.parse_args()
            runner = launcher.load_runner()
            launcher.configure_runner(runner, args, {})
            assert runner.MODELS[workload.model_name] == str(model_path)
        """, {
            "AE_DPSK_MODEL": self.model,
            "AE_DATASET_ROOT": self.dataset_root,
            "AE_VLLM_ROOT": self.vllm,
        })

    def test_fig12_explicit_overrides_need_no_default_paths(self):
        self.run_code("""
            import e2e_config as config
            workload, = config.selected_workloads(["kimi_issue1"])
            assert workload.model_path == model_path
            assert workload.dataset_path == dataset_path
        """, overrides={
            "FIG12_KIMI_MODEL": self.model,
            "FIG12_ISSUE1_DATASET": self.dataset,
        })

    def test_fig12_missing_selected_path_reports_the_required_key(self):
        self.run_code("""
            import e2e_config as config
            workload, = config.selected_workloads(["kimi_issue1"])
            try:
                workload.model_path
            except SystemExit as error:
                assert "AE_KIMI_MODEL is not configured" in str(error), str(error)
            else:
                raise AssertionError("missing Kimi checkpoint was accepted")
        """)


if __name__ == "__main__":
    unittest.main()
