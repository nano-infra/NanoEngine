"""
ProtenixRunner — loads the Protenix model in-process (GPU-resident) and exposes
two split entry points:

  run_trunk(request)        → embed_id  (pairformer + prepare_cache, saves to /dev/shm)
  run_diffusion(embed_id, …)→ (structures, confidence)
  run_predict(request)      → (structures, confidence)  [convenience: trunk + diffusion]
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import tempfile
import time
import traceback
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Keys in input_feature_dict that are NOT needed for diffusion (bulky MSA matrices)
_MSA_KEYS_TO_DROP = [
    "msa_profile",
    "msa_deletion_mean",
    "msa_mask",
    "msa",
    "has_deletion",
    "deletion_value",
]


class ProtenixRunner:
    def __init__(
        self,
        model_name: str = "protenix_base_default_v1.0.0",
        checkpoint_dir: str = "/models/fold/checkpoint",
        dtype: str = "bf16",
        n_cycle: int = 10,
        trimul_kernel: str = "cuequivariance",
        triatt_kernel: str = "cuequivariance",
        enable_cache: bool = True,
        enable_fusion: bool = True,
        enable_tf32: bool = True,
        shm_dir: str = "/dev/shm/nanofold",
        gpu: int = 0,
    ) -> None:
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.dtype = dtype
        self.n_cycle = n_cycle
        self.trimul_kernel = trimul_kernel
        self.triatt_kernel = triatt_kernel
        self.enable_cache = enable_cache
        self.enable_fusion = enable_fusion
        self.enable_tf32 = enable_tf32
        self.shm_dir = Path(shm_dir)
        self.shm_dir.mkdir(parents=True, exist_ok=True)
        self.gpu = gpu

        self._runner = None  # lazy-loaded InferenceRunner
        self._configs = None

    # ── Lazy model load ───────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._runner is not None:
            return
        from runner.batch_inference import get_default_runner

        logger.info("Loading Protenix model %s on GPU %d …", self.model_name, self.gpu)
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(self.gpu))
        if self.enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self._runner = get_default_runner(
            model_name=self.model_name,
            n_cycle=self.n_cycle,
            dtype=self.dtype,
            trimul_kernel=self.trimul_kernel,
            triatt_kernel=self.triatt_kernel,
            enable_cache=self.enable_cache,
            enable_fusion=self.enable_fusion,
            enable_tf32=self.enable_tf32,
            use_msa=False,  # overridden per-request
        )
        self._runner.dumper.base_dir = str(self.shm_dir)  # write outputs to shm
        logger.info("Model loaded. Parameters: ready.")

    # ── Helpers ───────────────────────────────────────────────────────

    @property
    def _device(self) -> torch.device:
        return self._runner.device

    @property
    def _dtype_torch(self) -> torch.dtype:
        return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
            self.dtype
        ]

    def _amp(self):
        if torch.cuda.is_available():
            return torch.autocast(device_type="cuda", dtype=self._dtype_torch)
        return nullcontext()

    def _featurize(self, input_json: list[dict], use_msa: bool, use_template: bool):
        """Write JSON to temp file → run InferDataloader → return (data, atom_array)."""
        from runner.inference import get_inference_dataloader, update_inference_configs

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(input_json, fh)
            tmp_path = fh.name

        try:
            configs = self._runner.configs
            configs.input_json_path = tmp_path
            configs.use_msa = use_msa
            configs.use_template = use_template

            dataloader = get_inference_dataloader(configs=configs)
            batch = next(iter(dataloader))
            data, atom_array, err = batch[0]
            if err:
                raise RuntimeError(f"Featurization error: {err}")

            n_token = data["N_token"].item()
            configs = update_inference_configs(configs, n_token)
            self._runner.update_model_configs(configs)
            return data, atom_array, n_token
        finally:
            os.unlink(tmp_path)

    # ── Public API ────────────────────────────────────────────────────

    def run_trunk(
        self,
        name: str,
        sequences: list[dict],
        covalent_bonds: list[dict],
        model_name: str,
        use_msa: bool,
        use_template: bool,
        dtype: str,
        n_cycle: int,
        trimul_kernel: str,
        triatt_kernel: str,
        enable_cache: bool,
    ) -> tuple[str, int]:
        """
        Run pairformer trunk and cache tensors to /dev/shm.
        Returns (embed_id, n_token).
        """
        self._ensure_loaded()

        input_json = [
            {
                "name": name,
                "sequences": sequences,
                "covalent_bonds": covalent_bonds,
            }
        ]

        # Override per-request settings if they differ from loaded defaults
        configs = self._runner.configs
        configs.model.N_cycle = n_cycle
        configs.dtype = dtype
        configs.triangle_multiplicative = trimul_kernel
        configs.triangle_attention = triatt_kernel
        configs.enable_diffusion_shared_vars_cache = enable_cache

        data, atom_array, n_token = self._featurize(input_json, use_msa, use_template)
        model = self._runner.model

        from protenix.model.protenix import update_input_feature_dict
        from protenix.utils.torch_utils import to_device

        data = to_device(data, self._device)
        feat = data["input_feature_dict"]
        feat = update_input_feature_dict(feat)  # adds d_lm, v_lm, pad_info

        with torch.no_grad(), self._amp():
            s_inputs, s, z = model.get_pairformer_output(feat, N_cycle=n_cycle)

            # prepare_cache computes relp and returns pair_z used by diffusion
            relp = model.relative_position_encoding.generate_relp(feat)
            pair_z, p_lm, c_l = (
                model.diffusion_module.diffusion_conditioning.prepare_cache(
                    relp, z, False
                )
            )

        # Save to /dev/shm
        embed_id = uuid.uuid4().hex
        embed_dir = self.shm_dir / embed_id
        embed_dir.mkdir(parents=True, exist_ok=True)

        torch.save(s_inputs.cpu(), embed_dir / "s_inputs.pt")
        torch.save(s.cpu(), embed_dir / "s.pt")
        torch.save(pair_z.cpu(), embed_dir / "pair_z.pt")
        torch.save(p_lm.cpu(), embed_dir / "p_lm.pt")
        torch.save(c_l.cpu(), embed_dir / "c_l.pt")

        # Save feature dict (drop heavy MSA matrices)
        feat_save = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in feat.items()
            if k not in _MSA_KEYS_TO_DROP
        }
        torch.save(feat_save, embed_dir / "feat.pt")
        # Save atom_array for CIF dumping
        torch.save(atom_array, embed_dir / "atom_array.pt")
        # Save entity_poly_type
        torch.save(
            {k: v for k, v in data["entity_poly_type"].items() if v != "non-polymer"},
            embed_dir / "entity_poly_type.pt",
        )

        torch.cuda.empty_cache()
        logger.info(
            "Trunk done for %s: embed_id=%s n_token=%d", name, embed_id, n_token
        )
        return embed_id, n_token

    def run_diffusion(
        self,
        embed_id: str,
        seeds: list[int],
        n_sample: int,
        n_step: int,
    ) -> tuple[list[dict], list[dict]]:
        """
        Run diffusion from a cached embedding.
        Returns (structures, confidence) as plain dicts (serialisable).
        """
        self._ensure_loaded()

        embed_dir = self.shm_dir / embed_id
        if not embed_dir.exists():
            raise FileNotFoundError(f"Embed cache not found: {embed_id}")

        s_inputs = torch.load(embed_dir / "s_inputs.pt", weights_only=True).to(
            self._device
        )
        s = torch.load(embed_dir / "s.pt", weights_only=True).to(self._device)
        pair_z = torch.load(embed_dir / "pair_z.pt", weights_only=True).to(self._device)
        p_lm = torch.load(embed_dir / "p_lm.pt", weights_only=True).to(self._device)
        c_l = torch.load(embed_dir / "c_l.pt", weights_only=True).to(self._device)
        feat = torch.load(embed_dir / "feat.pt", weights_only=False)
        feat = {
            k: v.to(self._device) if isinstance(v, torch.Tensor) else v
            for k, v in feat.items()
        }
        atom_array = torch.load(embed_dir / "atom_array.pt", weights_only=False)
        entity_poly_type = torch.load(
            embed_dir / "entity_poly_type.pt", weights_only=False
        )

        model = self._runner.model
        configs = self._runner.configs
        configs.sample_diffusion.N_sample = n_sample
        configs.sample_diffusion.N_step = n_step

        from protenix.model.generator import DiffusionScheduler

        structures_out: list[dict] = []
        confidence_out: list[dict] = []

        for seed in seeds:
            from protenix.utils.seed import seed_everything

            seed_everything(seed=seed)

            scheduler = DiffusionScheduler(
                sigma_data=model.configs.sample_diffusion.sigma_data,
                s_max=model.configs.sample_diffusion.s_max,
                s_min=model.configs.sample_diffusion.s_min,
                rho=model.configs.sample_diffusion.rho,
            )
            noise_schedule = scheduler.get_schedule(
                N_step=n_step,
                device=self._device,
                dtype=self._dtype_torch,
            )

            with torch.no_grad(), self._amp():
                coords = model.sample_diffusion(
                    denoise_net=model.diffusion_module.denoise_net,
                    input_feature_dict=feat,
                    s_inputs=s_inputs,
                    s_trunk=s,
                    z_trunk=None,  # not needed when pair_z is provided
                    pair_z=pair_z,
                    p_lm=p_lm,
                    c_l=c_l,
                    noise_schedule=noise_schedule,
                    N_sample=n_sample,
                    inplace_safe=True,
                    enable_efficient_fusion=configs.enable_efficient_fusion,
                )

                confidence_raw = model.run_confidence_head(
                    input_feature_dict=feat,
                    s_inputs=s_inputs,
                    s=s,
                    z=None,
                    pred_dict={"coordinate": coords},
                )

            # Dump CIF to temp dir, read back, encode to base64
            dump_dir = self.shm_dir / "dumps" / embed_id / str(seed)
            dump_dir.mkdir(parents=True, exist_ok=True)

            pred_dict = {"coordinate": coords, **confidence_raw}
            sample_name = feat.get("sample_name", embed_id)

            self._runner.dumper.base_dir = str(dump_dir)
            self._runner.dumper.dump(
                dataset_name="",
                pdb_id=sample_name,
                seed=seed,
                pred_dict=pred_dict,
                atom_array=atom_array,
                entity_poly_type=entity_poly_type,
            )

            # Collect CIF files
            cif_paths = sorted(dump_dir.glob("**/*.cif"))
            for idx, cif_path in enumerate(cif_paths):
                content_b64 = base64.b64encode(cif_path.read_bytes()).decode()
                structures_out.append(
                    {
                        "seed": seed,
                        "sample_index": idx,
                        "format": "cif",
                        "content": content_b64,
                    }
                )

            # Collect confidence JSONs
            conf_paths = sorted(dump_dir.glob("**/*summary_confidence*.json"))
            for idx, conf_path in enumerate(conf_paths):
                conf_data = json.loads(conf_path.read_text())
                confidence_out.append(
                    {
                        "seed": seed,
                        "sample_index": idx,
                        "plddt": conf_data.get("plddt", 0.0),
                        "ptm": conf_data.get("ptm", 0.0),
                        "iptm": conf_data.get("iptm", 0.0),
                        "gpde": conf_data.get("gpde", 0.0),
                        "ranking_score": conf_data.get("ranking_score", 0.0),
                        "has_clash": conf_data.get("has_clash", False),
                    }
                )

            shutil.rmtree(dump_dir, ignore_errors=True)

        torch.cuda.empty_cache()
        return structures_out, confidence_out

    def run_predict(
        self,
        name: str,
        sequences: list[dict],
        covalent_bonds: list[dict],
        model_name: str,
        use_msa: bool,
        use_template: bool,
        dtype: str,
        n_cycle: int,
        trimul_kernel: str,
        triatt_kernel: str,
        enable_cache: bool,
        seeds: list[int],
        n_sample: int,
        n_step: int,
    ) -> tuple[str, list[dict], list[dict]]:
        """
        Convenience: trunk + diffusion in one call.
        Returns (embed_id, structures, confidence).
        """
        embed_id, _ = self.run_trunk(
            name=name,
            sequences=sequences,
            covalent_bonds=covalent_bonds,
            model_name=model_name,
            use_msa=use_msa,
            use_template=use_template,
            dtype=dtype,
            n_cycle=n_cycle,
            trimul_kernel=trimul_kernel,
            triatt_kernel=triatt_kernel,
            enable_cache=enable_cache,
        )
        structures, confidence = self.run_diffusion(
            embed_id=embed_id,
            seeds=seeds,
            n_sample=n_sample,
            n_step=n_step,
        )
        return embed_id, structures, confidence

    def cleanup_embed(self, embed_id: str) -> None:
        """Remove cached tensors for an embed_id."""
        embed_dir = self.shm_dir / embed_id
        shutil.rmtree(embed_dir, ignore_errors=True)
