# boltz-cp Integration with NanoRoute — Design Document

## 1. Motivation

boltz-cp provides state-of-the-art biomolecular structure prediction (Boltz-2 + context
parallelism) that is complementary to the LLM inference already served by NanoInfra.
Many drug-discovery and bioinformatics workflows need both:

- LLM-based reasoning / generation (NanoDeploy + NanoRoute today)
- Structure prediction (boltz-cp, potentially on the same GPU cluster)

Exposing boltz-cp through a REST API that is co-registered with the NanoInfra service
mesh lets clients use a single endpoint fabric, apply uniform auth/rate-limiting, and
compose LLM + structure prediction in one pipeline.

______________________________________________________________________

## 2. Architecture Overview

```
Client
  │
  │  POST /v1/structure/predict
  ▼
NanoRoute (Rust, Axum)
  │  HTTP reverse-proxy (no FlatBuffers needed)
  │  New route handler in http_server.rs
  │
  ▼
BoltzServer (Python, FastAPI)         ← new standalone service
  │  async job queue
  │  torchrun subprocess per request
  │
  ▼
boltz-cp (distributed/main.py predict)
  │  writes mmCIF + confidence JSON to temp dir
  ▼
Results returned via polling endpoint
```

### Key Design Choices

| Decision                             | Rationale                                                                                                                               |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| HTTP reverse proxy (not FlatBuffers) | Structure prediction is coarse-grained (1 request ≈ 1 job); no token-level streaming needed; JSON is natural for bioinformatics tooling |
| Async job queue (submit + poll)      | Inference can take minutes; synchronous HTTP would time out; SSE or polling gives clients visibility into progress                      |
| Standalone BoltzServer               | Keeps boltz-cp Python environment isolated from Rust NanoRoute; same pattern NanoCtrl uses for engine registration                      |
| NanoRoute as proxy                   | Clients get a single hostname; NanoRoute handles service discovery, load-balancing across multiple BoltzServer instances                |

______________________________________________________________________

## 3. BoltzServer Design

BoltzServer is a standalone **FastAPI** service that manages boltz-cp jobs.

### 3a. Endpoints

#### `POST /v1/structure/predict`

Submit a structure prediction job.

**Request body** (`application/json`):

```json
{
  "sequences": [
    {
      "type": "protein",
      "id": "A",
      "sequence": "MKTAYIAKQRQISFVKSHFSRQ..."
    },
    {
      "type": "ligand",
      "id": "B",
      "smiles": "CC(=O)Nc1ccc(O)cc1"
    }
  ],
  "recycling_steps": 3,
  "sampling_steps": 200,
  "diffusion_samples": 1,
  "output_format": "mmcif",
  "size_dp": 1,
  "size_cp": 4,
  "triattn_backend": "cueq",
  "precision": "BF16_MIXED",
  "seed": null
}
```

**Response** (`202 Accepted`):

```json
{
  "job_id": "b2f3a1c9-8e47-4d2b-9f1a-3c5d7e890abc",
  "status": "queued"
}
```

#### `GET /v1/structure/jobs/{job_id}`

Poll job status and retrieve results.

**Response** (while running):

```json
{
  "job_id": "b2f3a1c9-8e47-4d2b-9f1a-3c5d7e890abc",
  "status": "running",
  "structures": null,
  "confidence": null,
  "error": null
}
```

**Response** (on completion):

```json
{
  "job_id": "b2f3a1c9-8e47-4d2b-9f1a-3c5d7e890abc",
  "status": "done",
  "structures": [
    {
      "model_index": 0,
      "format": "mmcif",
      "content": "<base64-encoded mmCIF string>"
    }
  ],
  "confidence": {
    "plddt": 0.87,
    "ptm": 0.72,
    "iptm": 0.65,
    "complex_plddt": 0.84
  },
  "error": null
}
```

**Response** (on error):

```json
{
  "job_id": "b2f3a1c9-8e47-4d2b-9f1a-3c5d7e890abc",
  "status": "error",
  "structures": null,
  "confidence": null,
  "error": "torchrun exited with code 1: ..."
}
```

### 3b. Job Lifecycle

```
POST /predict
  → generate UUID job_id
  → write YAML spec to tmp/<job_id>/input.yaml
  → run serial "boltz predict" for preprocessing (blocking, fast)
  → enqueue torchrun job to async worker pool
  → return {job_id, status: "queued"}

Worker picks up job:
  → status = "running"
  → spawn torchrun subprocess
  → wait for subprocess exit

On success:
  → read mmCIF + confidence JSON from output dir
  → base64-encode structures
  → status = "done", store results in memory (or Redis)

On failure:
  → status = "error", store stderr
```

### 3c. BoltzServer Implementation Sketch

```python
# boltz_server/server.py
import asyncio, base64, json, subprocess, tempfile, uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BOLTZ_CP = Path("/mnt/nvme1n1/ml_research/majinming/src/boltz-cp")
CACHE_DIR = Path("~/.boltz").expanduser()

app = FastAPI()
jobs: dict[str, dict] = {}  # In production: use Redis

class PredictRequest(BaseModel):
    sequences: list[dict]
    recycling_steps: int = 3
    sampling_steps: int = 200
    diffusion_samples: int = 1
    output_format: str = "mmcif"
    size_dp: int = 1
    size_cp: int = 4
    triattn_backend: str = "cueq"
    precision: str = "BF16_MIXED"
    seed: int | None = None

@app.post("/v1/structure/predict")
async def predict(req: PredictRequest):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "structures": None, "confidence": None, "error": None}
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
        # 1. Write YAML spec
        _write_yaml(tmp / "input.yaml", req.sequences)
        # 2. Preprocess (serial)
        _preprocess(tmp, tmp / "preprocessed")
        # 3. Run distributed inference
        cmd = _build_torchrun_cmd(tmp / "preprocessed", tmp / "output", req)
        proc = await asyncio.create_subprocess_exec(*cmd, stderr=asyncio.subprocess.PIPE)
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = stderr.decode()
            return
        # 4. Collect results
        jobs[job_id]["structures"] = _collect_structures(tmp / "output", req.output_format)
        jobs[job_id]["confidence"] = _collect_confidence(tmp / "output")
        jobs[job_id]["status"] = "done"
```

### 3d. Starting BoltzServer

```bash
cd /mnt/nvme1n1/ml_research/majinming/src/NanoInfra
uvicorn boltz_server.server:app --host 0.0.0.0 --port 8200
```

______________________________________________________________________

## 4. NanoRoute Changes

### 4a. New `BoltzConfig` in `config.rs`

Add a new optional section to `AppConfig`:

```rust
// NanoRoute/src/config.rs

#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)]
pub struct BoltzConfig {
    pub server_url: String,        // e.g., "http://127.0.0.1:8200"
    pub timeout_s: u64,            // request timeout in seconds (default: 600)
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
    pub boltz: Option<BoltzConfig>,  // new optional section
}
```

Example TOML configuration:

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

# Optional: enable boltz structure prediction proxy
[boltz]
server_url = "http://127.0.0.1:8200"
timeout_s = 600
```

### 4b. New Route in `http_server.rs`

Add a reverse-proxy route for structure prediction:

```rust
// In create_router() or equivalent in http_server.rs

// Only register the route if boltz config is present
if let Some(boltz_cfg) = &config.boltz {
    router = router
        .route("/v1/structure/predict", post(structure_predict))
        .route("/v1/structure/jobs/:job_id", get(structure_job_status));
}
```

The handler forwards the request body verbatim to BoltzServer using `reqwest`:

```rust
async fn structure_predict(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<impl IntoResponse, StatusCode> {
    let boltz_url = state.boltz_url.as_ref().ok_or(StatusCode::NOT_FOUND)?;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/structure/predict", boltz_url))
        .json(&body)
        .timeout(Duration::from_secs(state.boltz_timeout_s))
        .send()
        .await
        .map_err(|_| StatusCode::BAD_GATEWAY)?;
    let status = resp.status();
    let body = resp.bytes().await.map_err(|_| StatusCode::BAD_GATEWAY)?;
    Ok((StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY), body))
}
```

### 4c. NanoCtrl Engine Registration

Register a `boltz` service type in NanoCtrl alongside the existing LLM prefill/decode engines:

```json
{
  "service_type": "boltz",
  "endpoint": "http://127.0.0.1:8200",
  "scope": "gpu-cluster-01"
}
```

NanoCtrl can then populate the `boltz.server_url` field in NanoRoute's config at runtime
via the same `nanoctrl_address` mechanism used for LLM engine discovery.

______________________________________________________________________

## 5. Protocol Design

### Why Not FlatBuffers

The LLM path uses FlatBuffers for high-throughput token-level communication where
serialization overhead matters at the microsecond level.

Structure prediction is fundamentally different:

- One request ≈ one multi-minute job
- Input: ~1KB JSON; output: ~100KB mmCIF
- No token streaming; no KV cache; no batching across sequences
- JSON over HTTP is sufficient and standard in bioinformatics tooling

### Polling vs. SSE

BoltzServer supports a **polling** model (`GET /v1/structure/jobs/{job_id}`) as the primary
interface because:

- Simple to implement and debug
- Works through proxies/load-balancers without special config
- Clients can control poll frequency

An optional **SSE** stream can be added later:

```
GET /v1/structure/jobs/{job_id}/stream
→ text/event-stream
  data: {"status": "running", "elapsed_s": 45}
  data: {"status": "done", "structures": [...]}
```

______________________________________________________________________

## 6. Input/Output Contract

### Full Request Schema

```json
{
  "sequences": [
    {
      "type": "protein" | "dna" | "rna" | "ligand",
      "id": "<chain_id>",
      "sequence": "<amino_acid_or_nucleotide_string>",  // protein/dna/rna
      "smiles": "<SMILES_string>",    // ligand (alternative to ccd)
      "ccd": "<CCD_code>",            // ligand (alternative to smiles)
      "msa": "<path_or_null>"         // protein; null = use MSA server
    }
  ],
  "recycling_steps": 3,
  "sampling_steps": 200,
  "diffusion_samples": 1,
  "output_format": "mmcif",      // "mmcif" | "pdb"
  "size_dp": 1,                  // data-parallel size (distributed only)
  "size_cp": 4,                  // context-parallel size (must be perfect square)
  "triattn_backend": "cueq",     // "cueq" | "trifast" | "reference"
  "precision": "BF16_MIXED",     // "BF16" | "BF16_MIXED" | "TF32" | "FP32"
  "seed": null                   // int | null
}
```

### Full Response Schema (terminal states)

```json
{
  "job_id": "<uuid>",
  "status": "queued" | "running" | "done" | "error",
  "structures": [
    {
      "model_index": 0,
      "format": "mmcif",
      "content": "<base64-encoded structure file>"
    }
  ],
  "confidence": {
    "plddt": 0.87,
    "ptm": 0.72,
    "iptm": 0.65,
    "complex_plddt": 0.84,
    "chains_ptm": [0.78],
    "pair_chains_iptm": [[1.0]]
  },
  "error": null | "<error message>"
}
```

Note: `confidence` is populated only by the serial boltz path. If `size_cp > 1`, the
distributed path is used and `confidence` will be `null` (limitation of boltz-cp
distributed mode as of current implementation).

______________________________________________________________________

## 7. Deployment

### Step 1: Start BoltzServer

```bash
pip install fastapi uvicorn

# Activate environment with boltz-cp installed
pip install -e /mnt/nvme1n1/ml_research/majinming/src/boltz-cp

# Start BoltzServer
uvicorn boltz_server.server:app --host 0.0.0.0 --port 8200 --workers 1
```

Use `--workers 1` because boltz-cp itself uses all GPUs via `torchrun`; running multiple
worker processes would cause GPU contention.

### Step 2: Register with NanoCtrl

```bash
curl -X POST http://localhost:3000/register \
  -H "Content-Type: application/json" \
  -d '{
    "service_type": "boltz",
    "endpoint": "http://127.0.0.1:8200",
    "scope": "default"
  }'
```

### Step 3: Configure NanoRoute

Add to your NanoRoute TOML config:

```toml
[boltz]
server_url = "http://127.0.0.1:8200"
timeout_s = 600
```

Or rely on NanoCtrl auto-discovery if `nanoctrl_address` is set in `[engine]`.

### Step 4: Verify

```bash
# Via NanoRoute
curl -X POST http://localhost:8080/v1/structure/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sequences": [{"type": "protein", "id": "A", "sequence": "MKTAYIAKQRQISFVK"}],
    "recycling_steps": 1,
    "sampling_steps": 10,
    "diffusion_samples": 1,
    "size_dp": 1,
    "size_cp": 1
  }'

# Poll result
JOB_ID=<from above>
curl http://localhost:8080/v1/structure/jobs/${JOB_ID}
```

______________________________________________________________________

## 8. Non-Goals

The following are explicitly out of scope for this integration:

- **Token streaming / chunked generation** — not applicable; structure prediction is not
  autoregressive token generation.
- **KV cache** — boltz-cp uses a diffusion model; there is no KV cache concept.
- **Cross-sequence batching** — boltz-cp processes one structure at a time per torchrun job;
  batching multiple unrelated sequences in one job is not supported.
- **FlatBuffers protocol** — not needed; see §5 for rationale.
- **NanoDeploy changes** — BoltzServer is standalone; no changes to the NanoDeploy C++
  engine or scheduler are required.
- **Affinity prediction** — `boltz2_aff.ckpt` works in serial mode only; the distributed
  path does not support affinity. This can be added as a separate `POST /v1/affinity/predict`
  endpoint backed by the serial boltz path.
