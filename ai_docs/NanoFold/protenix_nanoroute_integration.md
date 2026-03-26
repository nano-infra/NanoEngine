# Protenix Integration with NanoRoute — Design Document

## 1. Motivation

Protenix provides state-of-the-art open-source biomolecular structure prediction
(outperforming AlphaFold3 on standard benchmarks, Apache 2.0, live web server with
real DAU). Exposing it through NanoRoute gives clients a single endpoint fabric for
both LLM inference and structure prediction, with unified auth, rate-limiting, and
service discovery.

Protenix is the **primary** NanoFold backend: simpler deployment than boltz-cp
(single-GPU per job, no preprocessing pipeline, JSON in → CIF out), better accuracy
for typical drug-discovery targets, and a broader model menu (base / mini / tiny,
constraint-capable, ESM-augmented).

______________________________________________________________________

## 2. Architecture Overview

```
Client
  │
  │  POST /v1/structure/predict
  ▼
NanoRoute (Rust, Axum)
  │  HTTP reverse-proxy (no FlatBuffers)
  │  New route handler in http_server.rs
  │
  ▼
ProtenixServer (Python, FastAPI)       ← new standalone service
  │  async job queue
  │  protenix pred subprocess per request
  │
  ▼
protenix pred (single-GPU inference)
  │  writes CIF + confidence JSON to temp dir
  ▼
Results returned via polling endpoint
```

### Key Design Choices

| Decision                             | Rationale                                                                                               |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| HTTP reverse proxy (not FlatBuffers) | Structure prediction is coarse-grained; no token-level streaming; JSON is natural for bioinformatics    |
| Async job queue (submit + poll)      | Inference takes seconds–minutes; synchronous HTTP would time out                                        |
| Standalone ProtenixServer            | Keeps Python environment isolated from Rust NanoRoute; mirrors BoltzServer design                       |
| Single-GPU subprocess                | Protenix has no CP; `protenix pred` handles one structure per GPU; multi-GPU = multiple concurrent jobs |

______________________________________________________________________

## 3. ProtenixServer Design

ProtenixServer is a standalone **FastAPI** service managing Protenix prediction jobs.

### 3a. Endpoints

#### `POST /v1/structure/predict`

Submit a structure prediction job.

**Request body** (`application/json`):

```json
{
  "name": "my_complex",
  "sequences": [
    {
      "proteinChain": {
        "sequence": "MKTAYIAKQRQISFVK",
        "count": 1
      }
    },
    {
      "ligand": {
        "ligand": "CC(=O)Nc1ccc(O)cc1",
        "count": 1
      }
    }
  ],
  "covalent_bonds": [],
  "model_name": "protenix_base_default_v1.0.0",
  "seeds": [101],
  "n_sample": 1,
  "use_msa": true,
  "use_template": false,
  "use_rna_msa": false,
  "dtype": "bf16",
  "enable_cache": true
}
```

**Response** (`202 Accepted`):

```json
{
  "job_id": "c4e7a2b1-1f3d-4e9a-8c2b-7f0d1e5a3c4b",
  "status": "queued"
}
```

#### `GET /v1/structure/jobs/{job_id}`

Poll job status and retrieve results.

**Response** (while running):

```json
{
  "job_id": "c4e7a2b1-1f3d-4e9a-8c2b-7f0d1e5a3c4b",
  "status": "running",
  "structures": null,
  "confidence": null,
  "error": null
}
```

**Response** (on completion):

```json
{
  "job_id": "c4e7a2b1-1f3d-4e9a-8c2b-7f0d1e5a3c4b",
  "status": "done",
  "structures": [
    {
      "seed": 101,
      "sample_index": 0,
      "format": "cif",
      "content": "<base64-encoded CIF string>"
    }
  ],
  "confidence": [
    {
      "seed": 101,
      "sample_index": 0,
      "plddt": 0.87,
      "gpde": 0.12,
      "ptm": 0.72,
      "iptm": 0.65,
      "ranking_score": 0.81,
      "has_clash": false
    }
  ],
  "error": null
}
```

**Response** (on error):

```json
{
  "job_id": "c4e7a2b1-1f3d-4e9a-8c2b-7f0d1e5a3c4b",
  "status": "error",
  "structures": null,
  "confidence": null,
  "error": "protenix pred exited with code 1: ..."
}
```

### 3b. Job Lifecycle

```
POST /predict
  → generate UUID job_id
  → write input JSON to tmp/<job_id>/input.json
  → enqueue job to async worker
  → return {job_id, status: "queued"}

Worker picks up job:
  → status = "running"
  → spawn: protenix pred -i tmp/<job_id>/input.json
                         -o tmp/<job_id>/output
                         -n <model_name> -s <seeds>
                         [--use_msa/template/rna_msa]
                         [--enable_cache true]
  → wait for process exit

On success:
  → scan output/<name>/<seed>/ for .cif and _summary_confidence_*.json
  → base64-encode CIF files
  → status = "done", store results

On failure:
  → status = "error", store stderr
```

### 3c. ProtenixServer Implementation Sketch

```python
# protenix_server/server.py
import asyncio, base64, json, tempfile, uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

PROTENIX_SRC = Path("/mnt/nvme1n1/ml_research/majinming/src/Protenix")

app = FastAPI()
jobs: dict[str, dict] = {}  # In production: use Redis

class PredictRequest(BaseModel):
    name: str = "job"
    sequences: list[dict]
    covalent_bonds: list[dict] = []
    model_name: str = "protenix_base_default_v1.0.0"
    seeds: list[int] = [101]
    n_sample: int = 1
    use_msa: bool = True
    use_template: bool = False
    use_rna_msa: bool = False
    dtype: str = "bf16"
    enable_cache: bool = True

@app.post("/v1/structure/predict")
async def predict(req: PredictRequest):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "structures": None,
                    "confidence": None, "error": None}
    asyncio.create_task(_run_job(job_id, req))
    return {"job_id": job_id, "status": "queued"}

@app.get("/v1/structure/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, **jobs[job_id]}

async def _run_job(job_id: str, req: PredictRequest):
    jobs[job_id]["status"] = "running"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        input_json = tmp / "input.json"
        out_dir = tmp / "output"
        # Write Protenix JSON (must be a list)
        input_json.write_text(json.dumps([{
            "name": req.name,
            "sequences": req.sequences,
            "covalent_bonds": req.covalent_bonds,
        }]))
        seeds_str = ",".join(str(s) for s in req.seeds)
        cmd = [
            "protenix", "pred",
            "-i", str(input_json),
            "-o", str(out_dir),
            "-n", req.model_name,
            "-s", seeds_str,
            f"--use_msa={str(req.use_msa).lower()}",
            f"--use_template={str(req.use_template).lower()}",
            f"--use_rna_msa={str(req.use_rna_msa).lower()}",
            f"--dtype={req.dtype}",
            f"--enable_cache={str(req.enable_cache).lower()}",
            f"--sample_diffusion.N_sample={req.n_sample}",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stderr=asyncio.subprocess.PIPE)
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            jobs[job_id].update(status="error", error=stderr.decode())
            return
        jobs[job_id]["structures"] = _collect_structures(out_dir, req.name, req.seeds)
        jobs[job_id]["confidence"] = _collect_confidence(out_dir, req.name, req.seeds)
        jobs[job_id]["status"] = "done"
```

### 3d. Starting ProtenixServer

```bash
pip install fastapi uvicorn
pip install -e /mnt/nvme1n1/ml_research/majinming/src/Protenix

# Single-GPU server (one job at a time)
uvicorn protenix_server.server:app --host 0.0.0.0 --port 8201 --workers 1
```

For multi-GPU throughput, run one ProtenixServer instance per GPU, each on a different
port, with `CUDA_VISIBLE_DEVICES` scoping each instance to one GPU. NanoRoute
load-balances across instances:

```bash
CUDA_VISIBLE_DEVICES=0 uvicorn protenix_server.server:app --port 8201 &
CUDA_VISIBLE_DEVICES=1 uvicorn protenix_server.server:app --port 8202 &
CUDA_VISIBLE_DEVICES=2 uvicorn protenix_server.server:app --port 8203 &
CUDA_VISIBLE_DEVICES=3 uvicorn protenix_server.server:app --port 8204 &
```

______________________________________________________________________

## 4. NanoRoute Changes

### 4a. New `ProtenixConfig` in `config.rs`

```rust
// NanoRoute/src/config.rs

#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)]
pub struct ProtenixConfig {
    /// One or more ProtenixServer URLs for load-balancing.
    /// e.g. ["http://127.0.0.1:8201", "http://127.0.0.1:8202"]
    pub server_urls: Vec<String>,
    pub timeout_s: u64,            // per-request timeout (default: 300)
}

#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)]
pub struct AppConfig {
    pub server: ServerConfig,
    #[serde(default)]
    pub tokenizer: Option<TokenizerConfig>,
    pub engine: EngineConfig,
    pub scheduler: SchedulerConfig,
    #[serde(default)]
    pub protenix: Option<ProtenixConfig>,  // new optional section
    #[serde(default)]
    pub boltz: Option<BoltzConfig>,        // boltz-cp backend (large structures)
}
```

Example TOML:

```toml
[server]
port = 8080
model_name = "Qwen3-235B-A22B"

[engine]
mode = "Unified"
host = "127.0.0.1"
port = 5000

[scheduler]
queue_size = 256
timeout_ms = 30000

# Protenix: primary structure prediction backend
[protenix]
server_urls = [
  "http://127.0.0.1:8201",
  "http://127.0.0.1:8202",
  "http://127.0.0.1:8203",
  "http://127.0.0.1:8204",
]
timeout_s = 300

# boltz-cp: fallback for very large assemblies requiring CP
[boltz]
server_url = "http://127.0.0.1:8200"
timeout_s = 600
```

### 4b. New Route in `http_server.rs`

```rust
// Register routes if protenix config is present
if config.protenix.is_some() {
    router = router
        .route("/v1/structure/predict", post(structure_predict_protenix))
        .route("/v1/structure/jobs/:job_id", get(structure_job_status_protenix));
}
```

Round-robin load-balancing across `server_urls`:

```rust
async fn structure_predict_protenix(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<impl IntoResponse, StatusCode> {
    let cfg = state.protenix.as_ref().ok_or(StatusCode::NOT_FOUND)?;
    // Round-robin via atomic counter
    let idx = state.protenix_idx.fetch_add(1, Ordering::Relaxed)
              % cfg.server_urls.len();
    let url = &cfg.server_urls[idx];
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/structure/predict", url))
        .json(&body)
        .timeout(Duration::from_secs(cfg.timeout_s))
        .send()
        .await
        .map_err(|_| StatusCode::BAD_GATEWAY)?;
    let status = resp.status();
    let body = resp.bytes().await.map_err(|_| StatusCode::BAD_GATEWAY)?;
    Ok((StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY), body))
}
```

### 4c. NanoCtrl Service Registration

```json
{
  "service_type": "protenix",
  "endpoint": "http://127.0.0.1:8201",
  "scope": "gpu-cluster-01"
}
```

NanoCtrl populates `protenix.server_urls` via the same `nanoctrl_address` discovery
mechanism used for LLM engines.

______________________________________________________________________

## 5. Protocol Design

Same rationale as the boltz-cp integration: HTTP + JSON, no FlatBuffers.

### Routing: Protenix vs. boltz-cp

When both backends are registered, NanoRoute selects the backend based on the request:

| Condition                              | Backend  |
| -------------------------------------- | -------- |
| Default / `size_cp` absent or `1`      | Protenix |
| `size_cp` > 1 explicitly requested     | boltz-cp |
| `model_name` starts with `"protenix_"` | Protenix |
| `model_name` ends with `".ckpt"`       | boltz-cp |

This can be implemented as a simple field check in the `/v1/structure/predict` handler
before forwarding.

### Polling vs. SSE

Primary interface is polling (`GET /v1/structure/jobs/{job_id}`). Optional SSE stream
can be added:

```
GET /v1/structure/jobs/{job_id}/stream
→ text/event-stream
  data: {"status": "running", "elapsed_s": 12}
  data: {"status": "done", "structures": [...]}
```

______________________________________________________________________

## 6. Input/Output Contract

### Full Request Schema

```json
{
  "name": "<job_name>",
  "sequences": [
    {
      "proteinChain": {
        "sequence": "<AA_string>",
        "count": 1,
        "pairedMsaPath": "<abs_path_or_null>",
        "unpairedMsaPath": "<abs_path_or_null>",
        "templatesPath": "<abs_path_or_null>",
        "modifications": []
      }
    },
    {
      "dnaSequence": { "sequence": "<ATGC>", "count": 1 }
    },
    {
      "rnaSequence": { "sequence": "<AUGC>", "count": 1,
                       "unpairedMsaPath": "<abs_path_or_null>" }
    },
    {
      "ligand": { "ligand": "CCD_ATP | <SMILES> | FILE_<path>", "count": 1 }
    },
    {
      "ion": { "ion": "MG", "count": 2 }
    }
  ],
  "covalent_bonds": [],
  "contact": [],
  "pocket": null,
  "model_name": "protenix_base_default_v1.0.0",
  "seeds": [101],
  "n_sample": 1,
  "use_msa": true,
  "use_template": false,
  "use_rna_msa": false,
  "dtype": "bf16",
  "enable_cache": true
}
```

### Full Response Schema (terminal states)

```json
{
  "job_id": "<uuid>",
  "status": "queued | running | done | error",
  "structures": [
    {
      "seed": 101,
      "sample_index": 0,
      "format": "cif",
      "content": "<base64-encoded CIF>"
    }
  ],
  "confidence": [
    {
      "seed": 101,
      "sample_index": 0,
      "plddt": 0.87,
      "gpde": 0.12,
      "ptm": 0.72,
      "iptm": 0.65,
      "ranking_score": 0.81,
      "has_clash": false,
      "chain_ptm": [0.78],
      "chain_pair_iptm": [[1.0]]
    }
  ],
  "error": null
}
```

______________________________________________________________________

## 7. Deployment

### Step 1: Start ProtenixServer instances (one per GPU)

```bash
pip install fastapi uvicorn
pip install -e /mnt/nvme1n1/ml_research/majinming/src/Protenix

for GPU in 0 1 2 3; do
  PORT=$((8201 + GPU))
  CUDA_VISIBLE_DEVICES=${GPU} uvicorn protenix_server.server:app \
    --host 0.0.0.0 --port ${PORT} --workers 1 &
done
```

### Step 2: Register with NanoCtrl

```bash
for PORT in 8201 8202 8203 8204; do
  curl -X POST http://localhost:3000/register \
    -H "Content-Type: application/json" \
    -d "{\"service_type\": \"protenix\",
         \"endpoint\": \"http://127.0.0.1:${PORT}\",
         \"scope\": \"default\"}"
done
```

### Step 3: Configure NanoRoute

```toml
[protenix]
server_urls = [
  "http://127.0.0.1:8201",
  "http://127.0.0.1:8202",
  "http://127.0.0.1:8203",
  "http://127.0.0.1:8204",
]
timeout_s = 300
```

### Step 4: Verify

```bash
# Submit a job
curl -X POST http://localhost:8080/v1/structure/predict \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test",
    "sequences": [{"proteinChain": {"sequence": "MKTAYIAKQRQISFVK", "count": 1}}],
    "model_name": "protenix_mini_default_v0.5.0",
    "seeds": [101],
    "use_msa": false
  }'

# Poll result
JOB_ID=<from above>
curl http://localhost:8080/v1/structure/jobs/${JOB_ID}
```

______________________________________________________________________

## 8. Non-Goals

- **Context parallelism** — Protenix has no CP; large-structure sharding requires boltz-cp
- **Token streaming** — structure prediction is not autoregressive
- **KV cache** — diffusion model; no KV cache concept
- **FlatBuffers protocol** — not needed; HTTP+JSON is sufficient
- **NanoDeploy changes** — ProtenixServer is standalone; no C++ engine changes required
- **MSA server integration** — ProtenixServer accepts pre-computed MSA paths; running
  `protenix prep` is a client-side or pre-processing concern
- **Affinity prediction** — not currently in Protenix v1 scope (unlike boltz-cp's
  `boltz2_aff.ckpt`); add as `POST /v1/affinity/predict` if/when Protenix adds it
