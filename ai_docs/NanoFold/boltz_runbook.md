# boltz-cp Operational Runbook

## 1. Overview

**boltz-cp** (Fold-CP) is an NVIDIA fork of the [Boltz-2](https://github.com/jwohlwend/boltz)
biomolecular structure prediction model that adds **2D DTensor context parallelism (CP)**
for distributed inference and training across multiple GPUs.

| Property       | Value                                                                    |
| -------------- | ------------------------------------------------------------------------ |
| Repo           | `/mnt/nvme1n1/ml_research/majinming/src/boltz-cp/`                       |
| Upstream       | https://github.com/jwohlwend/boltz                                       |
| CP upstream PR | https://github.com/jwohlwend/boltz/pull/658                              |
| Paper          | https://research.nvidia.com/labs/dbr/assets/data/manuscripts/fold_cp.pdf |
| Supports       | Boltz-2 only (Boltz-1 not supported via distributed path)                |

### Key Capabilities

- Distributed inference with DTensor CP + DP across many GPUs
- Distributed training (separate runbook path)
- Attention backends: `cueq` (cuEquivariance), `trifast`, `reference` (FlexAttention)
- Precision modes: `BF16`, `BF16_MIXED`, `TF32`, `FP32`

### Requirements

- Python 3.10+
- PyTorch 2.9+ with CUDA
- Multiple NVIDIA GPUs (CP path requires ≥4 GPUs; `size_cp` must be a perfect square)
- `torchrun` or SLURM `srun` for multi-process launch

______________________________________________________________________

## 2. Installation

```bash
pip install -e /mnt/nvme1n1/ml_research/majinming/src/boltz-cp
```

This installs the `boltz` Python package in editable mode, making the `boltz` CLI
and `src/boltz/distributed/main.py` available.

______________________________________________________________________

## 3. Weight Download

### Automatic (triggered on first `boltz predict` run)

Weights are downloaded automatically to `~/.boltz` (or `$BOLTZ_CACHE`) on first use.
No manual action is needed for the serial path.

### Cache Location

```bash
# Default
~/.boltz/

# Override via env var (must be absolute path)
export BOLTZ_CACHE=/path/to/cache
```

### Artifacts Downloaded

| File                 | Source URL (primary)                                                   | HuggingFace fallback                                                           |
| -------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `boltz2_conf.ckpt`   | `https://model-gateway.boltz.bio/boltz2_conf.ckpt`                     | `https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt` |
| `boltz2_aff.ckpt`    | `https://model-gateway.boltz.bio/boltz2_aff.ckpt`                      | `https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_aff.ckpt`  |
| `mols.tar` → `mols/` | `https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar` | (same)                                                                         |

`mols.tar` contains per-residue CCD molecule pickle files. It is extracted to `~/.boltz/mols/`.

### Manual Download (Python snippet)

```python
from pathlib import Path
from boltz.main import download_boltz2

cache = Path("~/.boltz").expanduser()
cache.mkdir(parents=True, exist_ok=True)
download_boltz2(cache)
```

______________________________________________________________________

## 4. Input Preparation

Preparing input is a **two-stage process**:

1. Write a YAML (or FASTA) spec file describing your molecule(s).
2. Run `boltz predict` (serial path) which preprocesses to a data directory that
   the distributed path can consume.

### 4a. YAML Spec Format

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MKTAYIAKQRQISFVKSHFSRQ...
      msa: path/to/chain_A.a3m   # or omit + use --use_msa_server
  - protein:
      id: B
      sequence: GSHMTKKGHVNQVSIQVKGSSL...
  - ligand:
      id: C
      smiles: "CC(=O)Nc1ccc(O)cc1"  # paracetamol
  # Alternative: CCD code
  # - ligand:
  #     id: C
  #     ccd: ATP
```

**Optional top-level keys:**

```yaml
# Affinity prediction (requires boltz2_aff.ckpt)
affinity:
  binder: C      # chain ID of the ligand

# Distance/contact constraints
constraints:
  - contact:
      token1: [A, 10]
      token2: [B, 20]
      max_distance_angstrom: 8.0

# Structural templates
templates:
  - cif: path/to/template.cif
    chain_id: A
```

Accepted chain mol_types: `protein`, `dna`, `rna`, `ligand`.

### 4b. Preprocessing for Distributed Inference

The distributed path requires **preprocessed** input only. Use the serial `boltz predict`
command with `--input_format config_files` (default) to preprocess:

```bash
# This produces ./output/boltz_results_<name>/
boltz predict input.yaml \
  --out_dir ./output \
  --model boltz2

# The preprocessed data lives at:
# ./output/<record_id>/  (contains manifest.json, structures/, msa/, etc.)
```

The data directory passed to `distributed/main.py predict` must contain:

| Path            | Required | Description                                               |
| --------------- | -------- | --------------------------------------------------------- |
| `manifest.json` | Yes      | Sample manifest; loaded by rank 0, broadcast to all ranks |
| `structures/`   | Yes      | Preprocessed structure `.npz` files                       |
| `msa/`          | Yes      | MSA `.npz` files per chain entity                         |
| `templates/`    | No       | Template structure files                                  |
| `extra_mols/`   | No       | Additional molecule definitions                           |

______________________________________________________________________

## 5. Serial Inference (`boltz predict`)

The serial path runs on a single node with Lightning DDP:

```bash
boltz predict input.yaml \
  --out_dir ./predictions \
  --model boltz2 \
  --recycling_steps 3 \
  --sampling_steps 200 \
  --diffusion_samples 1 \
  --devices 1 \
  --accelerator gpu
```

### Serial CLI Options (selected)

| Option                   | Type   | Default                     | Description                                 |
| ------------------------ | ------ | --------------------------- | ------------------------------------------- |
| `DATA`                   | path   | —                           | Input `.yaml`/`.fasta` file or directory    |
| `--out_dir`              | path   | `./`                        | Output directory                            |
| `--cache`                | path   | `~/.boltz`                  | Model weight cache; respects `$BOLTZ_CACHE` |
| `--checkpoint`           | path   | auto                        | Explicit path to `boltz2_conf.ckpt`         |
| `--model`                | choice | `boltz2`                    | `boltz1` or `boltz2`                        |
| `--devices`              | int    | `1`                         | Number of GPUs for Lightning DDP            |
| `--accelerator`          | choice | `gpu`                       | `gpu`, `cpu`, or `tpu`                      |
| `--recycling_steps`      | int    | `3`                         | Trunk recycling iterations                  |
| `--sampling_steps`       | int    | `200`                       | Diffusion denoising steps                   |
| `--diffusion_samples`    | int    | `1`                         | Independent diffusion samples per input     |
| `--max_parallel_samples` | int    | `5`                         | Max parallel samples                        |
| `--step_scale`           | float  | `1.638`                     | Diffusion step scale                        |
| `--output_format`        | choice | `mmcif`                     | `pdb` or `mmcif`                            |
| `--write_full_pae`       | bool   | —                           | Write full PAE matrix to `.npz`             |
| `--seed`                 | int    | `None`                      | Random seed                                 |
| `--use_msa_server`       | flag   | False                       | Use MMSeqs2 server for MSA generation       |
| `--msa_server_url`       | str    | `https://api.colabfold.com` | MSA server URL                              |
| `--max_msa_seqs`         | int    | `8192`                      | Max MSA sequences                           |
| `--no_kernels`           | flag   | False                       | Disable custom attention kernels            |
| `--input_format`         | choice | `config_files`              | `config_files` or `preprocessed`            |
| `--override`             | flag   | False                       | Override existing predictions               |

______________________________________________________________________

## 6. Distributed Inference

The distributed path uses `torchrun` (or SLURM `srun`) to launch
`src/boltz/distributed/main.py predict`.

### Parallelism Constraint

```
size_dp * size_cp == world_size
```

`size_cp` **must be a perfect square**: 1, 4, 9, 16, ...

The CP mesh is 2D with shape `(sqrt(size_cp), sqrt(size_cp))`.

### Single-Node, 4-GPU Example (dp=1, cp=4)

```bash
BOLTZ_CP=/mnt/nvme1n1/ml_research/majinming/src/boltz-cp

torchrun \
  --nnodes 1 \
  --nproc_per_node 4 \
  ${BOLTZ_CP}/src/boltz/distributed/main.py predict \
  /path/to/preprocessed_data \
  --out_dir ./predictions \
  --size_dp 1 \
  --size_cp 4 \
  --recycling_steps 3 \
  --sampling_steps 200 \
  --diffusion_samples 5 \
  --triattn_backend cueq
```

### Two-Node, 8-GPU SLURM Example (dp=2, cp=4)

```bash
# In your SLURM job script:
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4

BOLTZ_CP=/mnt/nvme1n1/ml_research/majinming/src/boltz-cp

srun --ntasks-per-node=4 --nodes=2 \
  python ${BOLTZ_CP}/src/boltz/distributed/main.py predict \
  /path/to/preprocessed_data \
  --out_dir ./predictions \
  --size_dp 2 \
  --size_cp 4 \
  --checkpoint ${BOLTZ_CP}/~/.boltz/boltz2_conf.ckpt \
  --mol_dir ~/.boltz/mols \
  --recycling_steps 3 \
  --sampling_steps 200 \
  --diffusion_samples 5
```

### Distributed CLI Options

#### Required

| Argument | Description                         |
| -------- | ----------------------------------- |
| `DATA`   | Path to preprocessed data directory |

#### Common

| Option         | Type | Default    | Description                           |
| -------------- | ---- | ---------- | ------------------------------------- |
| `--out_dir`    | path | `./`       | Output directory                      |
| `--cache`      | path | `~/.boltz` | Weight cache; respects `$BOLTZ_CACHE` |
| `--checkpoint` | path | auto       | Explicit checkpoint path              |
| `--mol_dir`    | path | auto       | CCD molecule pickle directory         |

#### Parallelism

| Option      | Type | Default | Description                                          |
| ----------- | ---- | ------- | ---------------------------------------------------- |
| `--size_dp` | int  | `1`     | Data-parallel group size                             |
| `--size_cp` | int  | `1`     | Context-parallel group size (must be perfect square) |

#### Diffusion Sampling

| Option                   | Type  | Default | Description                                    |
| ------------------------ | ----- | ------- | ---------------------------------------------- |
| `--recycling_steps`      | int   | `3`     | Trunk recycling iterations                     |
| `--sampling_steps`       | int   | `200`   | Diffusion denoising steps                      |
| `--diffusion_samples`    | int   | `1`     | Independent diffusion samples per input        |
| `--max_parallel_samples` | int   | `None`  | Max samples in parallel (`None` = all at once) |
| `--step_scale`           | float | `1.5`   | Diffusion step scale (recommended 1–2)         |

#### Model and Precision

| Option          | Type   | Default      | Description                          |
| --------------- | ------ | ------------ | ------------------------------------ |
| `--precision`   | enum   | `BF16_MIXED` | `BF16`, `BF16_MIXED`, `TF32`, `FP32` |
| `--accelerator` | choice | `gpu`        | `gpu` or `cpu`                       |
| `--seed`        | int    | `None`       | Random seed                          |

#### Data Processing

| Option                  | Type   | Default        | Description                   |
| ----------------------- | ------ | -------------- | ----------------------------- |
| `--input_format`        | choice | `preprocessed` | Only `preprocessed` supported |
| `--max_msa_seqs`        | int    | `4096`         | Max MSA sequences             |
| `--msa_pad_to_max_seqs` | flag   | False          | Pad MSA to `max_msa_seqs`     |

#### Output

| Option               | Type   | Default | Description             |
| -------------------- | ------ | ------- | ----------------------- |
| `--output_format`    | choice | `mmcif` | `pdb` or `mmcif`        |
| `--write_full_pae`   | flag   | False   | Write full PAE matrices |
| `--local_batch_size` | int    | `1`     | Per-rank batch size     |
| `--num_ensembles`    | int    | `1`     | Ensemble members        |

#### Timeouts

| Option                  | Type  | Default | Description                        |
| ----------------------- | ----- | ------- | ---------------------------------- |
| `--timeout_nccl`        | float | `30`    | NCCL timeout (minutes)             |
| `--timeout_gloo`        | float | `30`    | Gloo timeout (minutes)             |
| `--cuda_memory_profile` | flag  | False   | Dump CUDA memory snapshot per rank |

______________________________________________________________________

## 7. Output Format

### Directory Structure

```
./predictions/
  boltz_results_<record_id>/
    predictions_dp{dp_rank}_cp0/   # distributed path writes here
      <record_id>_model_0.mmcif
      confidence_<record_id>_model_0.json
      ...
```

For the serial path, outputs are under `./predictions/<record_id>/`.

### mmCIF Structure File

Each sample produces `{record_id}_model_{n}.mmcif` (or `.pdb` if `--output_format pdb`).
Contains the predicted 3D structure in standard mmCIF/PDB format.

### Confidence JSON

```json
{
  "plddt": 0.87,
  "ptm": 0.72,
  "iptm": 0.65,
  "ligand_iptm": 0.55,
  "complex_plddt": 0.84,
  "chains_ptm": [0.78, 0.69],
  "pair_chains_iptm": [[1.0, 0.65], [0.65, 1.0]]
}
```

Full PAE matrices (if `--write_full_pae`) are written as `.npz` files.

> **Note:** The distributed path currently does **not** produce confidence summaries
> (`write_confidence_summary=False`). Only the serial path writes full confidence JSON.

______________________________________________________________________

## 8. Attention Backends

| Backend                   | Flag value       | Notes                                                        |
| ------------------------- | ---------------- | ------------------------------------------------------------ |
| cuEquivariance            | `cueq` (default) | Fastest; requires CUDA + cuequivariance install; no FP32/CPU |
| trifast                   | `trifast`        | Good GPU alternative                                         |
| Reference (FlexAttention) | `reference`      | Portable; CPU-compatible                                     |

```bash
# Set triangle attention backend
--triattn_backend cueq|trifast|reference

# SDPA backends for ring-attention layers
--sdpa_with_bias_backend reference|torch_flex_attn

# SDPA backends for window-batched attention
--sdpa_with_bias_shardwise_backend reference|torch_sdpa_efficient_attention|torch_flex_attn
```

______________________________________________________________________

## 9. Serial vs. Distributed Comparison

| Aspect              | Serial (`main.py predict`)     | Distributed (`distributed/main.py predict`)            |
| ------------------- | ------------------------------ | ------------------------------------------------------ |
| Launch              | `python` / `boltz predict`     | `torchrun` or `srun`                                   |
| Multi-GPU strategy  | Lightning DDP                  | SingleDeviceStrategy + DTensor CP mesh                 |
| Device options      | `--devices`, `--num_nodes`     | `--size_dp`, `--size_cp`                               |
| Input formats       | `config_files`, `preprocessed` | `preprocessed` only                                    |
| Confidence output   | Full JSON + optional PAE       | Not yet supported                                      |
| Affinity prediction | Supported                      | Not yet supported                                      |
| Constraint features | Supported                      | Not yet supported                                      |
| Steering potentials | Supported                      | Not yet supported                                      |
| Template features   | Supported                      | Weights loaded; distributed module not yet implemented |

______________________________________________________________________

## 10. Common Troubleshooting

### `size_dp * size_cp != world_size`

```
AssertionError: size_dp * size_cp must equal world_size
```

Fix: Ensure `--nnodes * --nproc_per_node == size_dp * size_cp`.
Example: 8 GPUs with `--size_dp 2 --size_cp 4` → `2 * 4 = 8`. ✓

### `size_cp` is not a perfect square

```
AssertionError: size_cp must be a perfect square
```

Valid values: 1, 4, 9, 16. `size_cp=2` or `size_cp=3` are not valid.

### NCCL timeout during all-reduce

Increase the NCCL timeout:

```bash
--timeout_nccl 60   # 60 minutes
```

Also check network MTU and verify all nodes can reach each other on the NCCL port.

### `cueq` backend fails

Switch to `trifast` or `reference`:

```bash
--triattn_backend trifast
```

Note: `cueq` does not support `FP32` precision or CPU accelerator.

### Missing MSA files

For the distributed path, MSA files must be pre-generated (the distributed runner does not
call the MSA server). Use the serial path with `--use_msa_server` first to generate
preprocessed data including MSA files.

### Weights not found at cache path

Check `$BOLTZ_CACHE` env var, or pass `--cache` explicitly:

```bash
--cache /path/to/.boltz
```
