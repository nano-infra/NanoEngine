# Protenix Context Parallelism — Design Document

## 1. Motivation

Protenix (368 M params) uses the same O(N²) pair-representation stack as AlphaFold3 and
Boltz-2. At inference on a single A100-80 G:

| N_token | Peak VRAM | Status |
| ------- | --------- | ------ |
| 1 024   | ~38 GB    | ok     |
| 2 048   | ~62 GB    | ok     |
| 3 072   | ~78 GB    | tight  |
| 4 096   | >80 GB    | OOM    |

The bottleneck is the Pairformer (`PairformerBlock` × 48 + MSA stack):
triangle attention stores `[B, N, N, d_pair]` activations; triangle multiplicative update
allocates intermediate `[B, N, N, d_pair]` tensors; all layers are O(N²) in memory.

**Goal**: shard the N×N pair representation across *P* GPUs using 2-D context parallelism
(CP), enabling arbitrarily large structures at the cost of P-fold communication overhead.
Mirror the design of `boltz-cp` (`NVIDIA/Fold-CP`), adapting each component to Protenix's
module hierarchy and kernel choices.

______________________________________________________________________

## 2. Background: boltz-cp 2-D CP Design

`boltz-cp` uses PyTorch DTensor with a 2-D `DeviceMesh`:

```
DeviceMesh("cuda", [[0, 1], [2, 3]])   # size_cp = 4 (= 2 × 2)
axis 0: "row"   (row shard of N)
axis 1: "col"   (col shard of N)
```

Each GPU owns a tile `[B, N/√P, N/√P, d_pair]`.
Three communication patterns are used:

| Operation                      | Pattern                                                    | File                                               |
| ------------------------------ | ---------------------------------------------------------- | -------------------------------------------------- |
| Triangle attention             | Ring2DCommTriAttn (2-stage bias scatter + diagonal rotate) | `distributed/model/layers/triangular_attention.py` |
| Triangle multiplicative update | \_distributed_bmm ring + transposed ring                   | `distributed/model/layers/triangular_mult.py`      |
| Outer product mean             | Ring-rotate a (by row) + b (by col)                        | `distributed/model/layers/outer_product_mean.py`   |
| Linear, LayerNorm              | DTensor Replicated params, Shard(1)/Shard(2) activations   | wrapper shims                                      |

______________________________________________________________________

## 3. Protenix CP Architecture

### 3a. Terminology

```
size_cp   — total CP degree (must be a perfect square: 1, 4, 9, 16, …)
cp_size_r — √size_cp  — row-dim CP size
cp_size_c — √size_cp  — col-dim CP size
rank_r    — row rank  (0 .. cp_size_r-1)
rank_c    — col rank  (0 .. cp_size_c-1)
```

### 3b. DeviceMesh

```python
from torch.distributed.device_mesh import init_device_mesh

# Given world_size = size_dp * size_cp, with size_cp a perfect square:
cp_mesh = init_device_mesh(
    "cuda",
    mesh_shape=(size_dp, cp_size_r, cp_size_c),
    mesh_dim_names=("dp", "cp_row", "cp_col"),
)
cp_submesh = cp_mesh["cp_row", "cp_col"]   # shape (cp_size_r, cp_size_c)
```

### 3c. Pair Representation Layout

For a full `[B, N, N, d]` pair tensor, each rank stores:

```
pair[B, rank_r*N_r : (rank_r+1)*N_r, rank_c*N_c : (rank_c+1)*N_c, d]
```

where `N_r = N_c = N / √size_cp`.

In DTensor notation:

```python
from torch.distributed._tensor import Shard, Replicate, DTensor

pair_dtensor = DTensor.from_local(
    local_pair,                     # [B, N_r, N_c, d]
    device_mesh=cp_submesh,
    placements=[Shard(1), Shard(2)],  # shard dim 1 on row-mesh, dim 2 on col-mesh
)
```

MSA tensor `[B, S, N, d_msa]` is only sharded on N (col-mesh axis):

```python
placements=[Replicate(), Shard(2)]   # S replicated; N sharded on col
```

Single-chain tensors (`[B, N, d]`) are sharded on N (row-mesh axis, by convention):

```python
placements=[Shard(1), Replicate()]
```

______________________________________________________________________

## 4. Module-by-Module Design

### 4a. Triangle Attention — Ring Replacement for TriAttentionFunction

**Problem**: Protenix's current `TriangleAttention` calls `_tri_attention()` which invokes
the custom `TriAttentionFunction` Triton kernel (`protenix/model/tri_attention/op.py`).
This kernel fuses all heads and computes the full N×N attention matrix in a single pass —
it cannot be split across ranks or interleaved with P2P communication.

**Solution**: Replace with a PyTorch-native ring attention path when CP is active, keeping
the Triton path as a fallback for CP=1.

#### 4a-i. Ring Triangle Attention Algorithm

The algorithm mirrors `_RingMultiHeadTriangleAttentionImpl` from boltz-cp exactly:

```
For triangle attention "starting" (attn over rows, bias from cols):
  Local pair tile: [B, N_r, N_c, d]
  Q, K, V from local rows; bias accumulated across CP col-ring.

Phase 1 — bias redistribution:
  Each rank prepares local_bias = pair_tile @ W_bias  → [B, N_r, N_c, heads]
  Ring-reduce bias along col-mesh (P2P isend/irecv, cp_size_c steps)
  After ring: each rank has full row-bias for its local rows.

Phase 2 — diagonal rotate for KV:
  All ranks hold local K, V tiles [B, N_r, N_c, d_k]
  Step s (0 .. cp_size_c-1):
    Receive KV from prev col-rank (P2P isend/irecv)
    Compute partial attention scores: Q · K^T for current KV shard
    Accumulate into running softmax (log-sum-exp trick)
    Send KV to next col-rank

Output: each rank holds its local output [B, N_r, N_c, d_v]
```

Triangle attention "ending" (attn over cols) transposes roles of row/col mesh.

#### 4a-ii. Implementation Sketch

```python
# protenix/model/distributed/triangular_attention.py

class RingTriangleAttention(nn.Module):
    """CP-aware triangle attention replacing TriAttentionFunction."""

    def __init__(self, c_pair: int, num_heads: int, orientation: str,
                 inf: float = 1e9, cp_mesh=None):
        super().__init__()
        self.orientation = orientation  # "per_row" | "per_column"
        self.cp_mesh = cp_mesh
        # Same projections as Protenix TriangleAttention
        self.linear_q = OpenfoldLinear(c_pair, num_heads * c_pair // num_heads, bias=False)
        self.linear_k = OpenfoldLinear(c_pair, num_heads * c_pair // num_heads, bias=False)
        self.linear_v = OpenfoldLinear(c_pair, num_heads * c_pair // num_heads, bias=False)
        self.linear_b = OpenfoldLinear(c_pair, num_heads, bias=False)
        self.linear_g = OpenfoldLinear(c_pair, num_heads * c_pair // num_heads)
        self.linear_o = OpenfoldLinear(num_heads * c_pair // num_heads, c_pair)
        self.layer_norm = OpenFoldLayerNorm(c_pair)

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.cp_mesh is None or self.cp_mesh.size() == 1:
            # Fallback: original Protenix TriAttentionFunction path
            return self._serial_forward(z, mask)
        return self._ring_forward(z, mask)

    def _ring_forward(self, z, mask):
        # Implementation follows _RingMultiHeadTriangleAttentionImpl
        # Phase 1: bias ring-reduce along col mesh
        # Phase 2: KV diagonal rotation across col mesh
        ...
```

#### 4a-iii. Triton Kernel Disposition

The existing `TriAttentionFunction` is **retained** for CP=1 (single GPU) because it is
significantly faster than pure PyTorch. When `size_cp > 1`, the Triton kernel is bypassed.
This is a deliberate trade-off: CP introduces P2P latency that already dominates over
kernel speed at large N.

**Long-term**: a custom Triton kernel that operates on local tiles and issues NCCL
collectives (similar to FlashAttention-3's ring variant) could recover some performance.
This is out of scope for the initial CP implementation.

______________________________________________________________________

### 4b. Triangle Multiplicative Update — Distributed BMM

`TriangleMultiplicativeUpdate` computes:

```
z_out = LayerNorm(a ⊗ b)    where  a = σ(g_a) ⊙ (z W_a),  b = σ(g_b) ⊙ (z W_b)
```

The outer product `a_left[i, k] · b_right[j, k] → z_out[i, j]` is an N×N BMM.

With CP sharding each rank holds tiles `a: [B, N_r, N_c, d]` and `b: [B, N_r, N_c, d]`.
To compute the output tile at `(i, j)` we need the full `k` dimension of both `a` and `b`.

#### Algorithm (mirrors `_distributed_bmm`)

**"Outgoing" mode** (z_out\[i, j\] = Σ_k a\[i,k\] · b\[j,k\]):

```
Ring-reduce a along col-mesh (cp_size_c steps, each rank gets full-k slice for its rows):
  → a_local: [B, N_r, N, d_r]

Ring-reduce b along row-mesh (cp_size_r steps, full-k slice for local cols):
  → b_local: [B, N_c, N, d_r]  (transposed view)

Compute: z_out_tile[B, N_r, N_c] = einsum("bik,bjk->bij", a_local, b_local)
```

**"Incoming" mode** (z_out\[i, j\] = Σ_k a\[k,i\] · b\[k,j\]) requires a transposed ring.

```python
# protenix/model/distributed/triangular_mult.py

def distributed_bmm(a_tile, b_tile, mode: str, cp_mesh) -> torch.Tensor:
    """Ring-reduce a and b, then compute local BMM tile."""
    cp_size_r = cp_mesh.size(0)  # "cp_row"
    cp_size_c = cp_mesh.size(1)  # "cp_col"
    ...
```

______________________________________________________________________

### 4c. Outer Product Mean — Ring Outer Product

`OuterProductMean` computes:

```
z += mean_s( (m W_a)[b, s, i, :] ⊗ (m W_b)[b, s, j, :] )
```

where `m` is the MSA tensor `[B, S, N, d_msa]`.

With CP: `m` is sharded on N (col-mesh); output `z[B, N_r, N_c, d]` is sharded `[Shard(1), Shard(2)]`.

#### Algorithm (mirrors `_OuterProductMeanImpl`)

```
Each rank holds m_tile: [B, S, N_c, d_msa]  (col-shard of N)
Compute: a_tile = m_tile @ W_a  →  [B, S, N_c, d_r]
         b_tile = m_tile @ W_b  →  [B, S, N_c, d_r]

Ring-reduce a along col-mesh (so each rank has full N for its row shard):
  → a_full: [B, S, N, d_r]  (sliced to local N_r rows after rotation)

Outer product:
  z_tile[b, i_r, j_c, :] = mean_s( a_full[b, s, i_r, :] ⊗ b_tile[b, s, j_c, :] )
```

The 2-D grid rotation pattern:

- `a` is rotated along the row-mesh (provides full N across the row shard)
- `b` stays local (already at column shard)

```python
# protenix/model/distributed/outer_product_mean.py

class RingOuterProductMean(nn.Module):
    def __init__(self, c_msa: int, c_pair: int, c_hidden: int = 32, cp_mesh=None):
        ...

    def forward(self, m: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.cp_mesh is None or self.cp_mesh.size() == 1:
            return self._serial_forward(m, mask)
        return self._ring_forward(m, mask)
```

______________________________________________________________________

### 4d. LayerNorm and Linear — DTensor Wrappers

`OpenfoldLinear` and `OpenFoldLayerNorm` are already standard `nn.Module` instances with
no custom distributed logic. They work correctly with DTensor when:

1. Parameters are **Replicated** across the CP mesh (standard DDP behavior):

   ```python
   from torch.distributed._tensor import DTensor, Replicate
   for param in module.parameters():
       param.data = DTensor.from_local(param.data, cp_mesh,
                                       [Replicate(), Replicate()])
   ```

2. Inputs are DTensors with the appropriate sharding (Shard(1) or Shard(2)); the linear
   `F.linear` and `F.layer_norm` ops propagate sharding automatically via DTensor's
   op dispatch table.

3. LayerNorm over the last dim (`d`) works per-tile (no communication needed because `d`
   is Replicated). LayerNorm over N (unusual, but used in some outer product paths) requires
   an all-reduce for the mean/variance — handled by DTensor automatically.

______________________________________________________________________

### 4e. Attention in MSAModule

`MSAStack` (`protenix/model/modules/pairformer.py`) runs row/col gated self-attention over
the MSA tensor `[B, S, N, d_msa]`. With CP this is sharded on N.

Row attention is already independent per row — no change needed.
Column attention is independent per column — no change needed.

The only communication point is the `OuterProductMean` (MSA → pair), handled in §4c.

______________________________________________________________________

### 4f. Template Embedder

`TemplateEmbedder` uses `PairformerBlock` internally, so the same ring triangle attention
applies once the blocks are replaced. Template MSA attention follows §4e.

______________________________________________________________________

## 5. Communication Primitives

Copy the `One2OneComm`, `Ring2DComm`, and `Ring2DCommTriAttn` classes from boltz-cp
verbatim (Apache 2.0 compatible) into:

```
protenix/model/distributed/comm.py
```

The only adaptation is replacing `boltz.distributed.comm` import paths and ensuring the
`cp_mesh` sub-mesh indexing matches Protenix's `("cp_row", "cp_col")` dim names.

### Key primitives

```python
class One2OneComm:
    """P2P send/recv wrapper with isend/irecv for pipeline overlap."""
    def send_recv(self, send_tensor, recv_tensor, send_rank, recv_rank, group): ...

class Ring2DComm:
    """Ring communication along one CP mesh axis."""
    def ring_send_recv(self, tensor, axis: str) -> Iterator[torch.Tensor]: ...

class Ring2DCommTriAttn:
    """Two-phase ring for triangle attention bias + KV rotation."""
    def bias_ring(self, bias, axis: str) -> torch.Tensor: ...
    def kv_rotate(self, k, v, axis: str) -> Iterator[Tuple[Tensor, Tensor]]: ...
```

______________________________________________________________________

## 6. ProtenixDistributed Wrapper

Following the `Boltz2Distributed` pattern from boltz-cp, create a wrapper class that:

1. Accepts a serial `Protenix` model instance
2. Replaces each `PairformerBlock`, `MSAStack`, and `TemplateEmbedder` submodule with its
   CP-aware variant
3. Wraps all parameters with DTensor Replicated placement
4. Provides the same `forward()` signature as the serial model

```python
# protenix/model/distributed/protenix_distributed.py

from protenix.model.protenix import Protenix
from protenix.model.distributed.pairformer import (
    DistributedPairformerBlock,
    DistributedMSAStack,
    DistributedTemplateEmbedder,
)

class ProtenixDistributed(nn.Module):
    """
    Wraps a serial Protenix model for 2-D context parallelism.

    Usage:
        model = Protenix(configs)
        cp_mesh = init_device_mesh("cuda", (cp_size_r, cp_size_c),
                                   mesh_dim_names=("cp_row", "cp_col"))
        dist_model = ProtenixDistributed(model, cp_mesh)
    """

    def __init__(self, model: Protenix, cp_mesh):
        super().__init__()
        self.cp_mesh = cp_mesh
        self._replace_submodules(model)
        self.model = model
        self._wrap_params_with_dtensor()

    def _replace_submodules(self, model: Protenix):
        """Replace serial blocks with CP-aware variants."""
        # PairformerStack
        stack = model.trunk.pairformer_stack
        for i, block in enumerate(stack.blocks):
            stack.blocks[i] = DistributedPairformerBlock.from_serial(block, self.cp_mesh)

        # MSAModule
        msa_module = model.trunk.msa_stack
        msa_module.outer_product_mean = RingOuterProductMean.from_serial(
            msa_module.outer_product_mean, self.cp_mesh)
        for i, block in enumerate(msa_module.msa_stack.blocks):
            msa_module.msa_stack.blocks[i] = DistributedMSABlock.from_serial(
                block, self.cp_mesh)

        # TemplateEmbedder (if templates are used)
        if hasattr(model.trunk, "template_embedder"):
            model.trunk.template_embedder = DistributedTemplateEmbedder.from_serial(
                model.trunk.template_embedder, self.cp_mesh)

    def _wrap_params_with_dtensor(self):
        for param in self.parameters():
            if not isinstance(param.data, DTensor):
                param.data = DTensor.from_local(
                    param.data, self.cp_mesh,
                    placements=[Replicate(), Replicate()])

    def forward(self, batch):
        # Shard input pair/MSA tensors before trunk
        batch = _shard_inputs(batch, self.cp_mesh)
        out = self.model(batch)
        # Gather output pair tensor for structure module
        out = _gather_outputs(out, self.cp_mesh)
        return out
```

### Input/Output Sharding Helpers

```python
def _shard_inputs(batch: dict, cp_mesh) -> dict:
    """Shard pair tensor [B,N,N,d] and MSA [B,S,N,d] to local tiles."""
    if "pair" in batch:
        batch["pair"] = DTensor.from_local(
            _local_slice(batch["pair"], cp_mesh, dims=(1, 2)),
            cp_mesh, placements=[Shard(1), Shard(2)])
    if "msa" in batch:
        batch["msa"] = DTensor.from_local(
            _local_slice(batch["msa"], cp_mesh, dims=(2,)),
            cp_mesh, placements=[Replicate(), Shard(2)])
    return batch

def _gather_outputs(out: dict, cp_mesh) -> dict:
    """All-gather the pair tensor before passing to structure module."""
    if isinstance(out.get("pair"), DTensor):
        out["pair"] = out["pair"].full_tensor()  # all-gather across CP mesh
    return out
```

______________________________________________________________________

## 7. Structure Module (Diffusion)

The diffusion module (`DiffusionModule`) in Protenix operates on per-atom coordinates
(`[B, N_atom, 3]`) which are O(N), not O(N²). At inference, structure module memory is
negligible compared to the Pairformer trunk.

**Recommendation**: do not shard the structure module in the initial implementation.
All-gather the pair tensor after the Pairformer trunk (see `_gather_outputs` above) and
run the structure module on rank 0 only (or on a replicated sub-mesh).

If structure module VRAM becomes a bottleneck at N > 8192, a future pass can add 1-D
sequence-parallel sharding of the atom coordinates.

______________________________________________________________________

## 8. Data Pipeline

Protenix's input is a JSON list of `{name, sequences, covalent_bonds}` dicts, processed
into an `AtomData` / `TokenData` batch by `protenix/data/`.

With CP, all preprocessing is performed on a single process (rank 0) and the resulting
batch tensors are broadcast to all CP ranks before `_shard_inputs` slices them:

```python
# protenix_server/distributed_runner.py
if cp_rank == 0:
    batch = preprocess(input_json)
    batch_tensors = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
else:
    batch_tensors = {}

# Broadcast all tensors
for key, val in batch_tensors.items():
    dist.broadcast(val, src=0, group=cp_process_group)
    batch_tensors[key] = val

batch.update(batch_tensors)
batch = _shard_inputs(batch, cp_mesh)
```

This is identical to boltz-cp's single-node distributed inference approach.

______________________________________________________________________

## 9. CLI and ProtenixServer Integration

### Serial path (CP=1)

No change. `protenix pred` continues to use the existing path.

### Distributed path (CP > 1)

```bash
torchrun \
  --nproc_per_node=${size_cp} \
  --nnodes=${nnodes} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  -m protenix.distributed.predict \
  --input input.json \
  --out_dir ./output \
  --model_name protenix_base_default_v1.0.0 \
  --size_cp ${size_cp} \
  --seeds 101 \
  --dtype bf16
```

Constraints: `size_cp` must be a perfect square (1, 4, 9, 16). `world_size == size_cp`
(no DP in inference; DP only in training).

### ProtenixServer Multi-GPU Mode

For very large structures, ProtenixServer can launch a `torchrun` subprocess:

```python
# protenix_server/server.py (extended)

async def _run_job_cp(job_id: str, req: PredictRequest, gpus: list[int]):
    """Distributed inference via torchrun for size_cp > 1."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    cmd = [
        "torchrun",
        f"--nproc_per_node={len(gpus)}",
        "-m", "protenix.distributed.predict",
        "--input", str(input_path),
        "--out_dir", str(out_dir),
        "--model_name", req.model_name,
        "--size_cp", str(len(gpus)),
        "--seeds", ",".join(str(s) for s in req.seeds),
        "--dtype", req.dtype,
    ]
    proc = await asyncio.create_subprocess_exec(*cmd, stderr=asyncio.subprocess.PIPE,
                                                 env=env)
    _, stderr = await proc.communicate()
    ...
```

______________________________________________________________________

## 10. Implementation Roadmap

### Phase 1 — Communication Primitives (Week 1-2)

- [ ] Copy `comm.py` from boltz-cp → `protenix/model/distributed/comm.py`
- [ ] Add DeviceMesh setup utility (`protenix/model/distributed/mesh.py`)
- [ ] Unit test: P2P ring on 4 GPUs, validate data integrity

### Phase 2 — Distributed Layers (Week 3-5)

- [ ] `RingOuterProductMean` (simplest: no bias redistribution phase)
- [ ] `DistributedTriangleMult` (\_distributed_bmm port)
- [ ] `RingTriangleAttention` (most complex: 2-phase bias + KV rotation)
- [ ] Per-layer unit tests: compare distributed vs serial output on small N (tolerance ≤ 1e-3)

### Phase 3 — ProtenixDistributed Wrapper (Week 6)

- [ ] `ProtenixDistributed` class with `_replace_submodules`
- [ ] `_shard_inputs` / `_gather_outputs` helpers
- [ ] End-to-end test: N=512 protein, 4-GPU CP, verify structure RMSD to serial run

### Phase 4 — CLI + Server Integration (Week 7-8)

- [ ] `protenix/distributed/predict.py` entry point
- [ ] `ProtenixServer` `_run_job_cp` path
- [ ] NanoRoute `ProtenixConfig` extended with optional `size_cp` per-request hint

### Phase 5 — Optimization (Week 9-10)

- [ ] Overlap communication with compute via `torch.cuda.Stream`
- [ ] Profile ring step latency vs compute at N=1024, 2048, 4096
- [ ] Optionally: NCCL comm group for each CP ring axis (avoid default process group)

______________________________________________________________________

## 11. Testing Strategy

### Correctness

```python
# tests/distributed/test_ring_triangle_attn.py

@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires 4 GPUs")
def test_ring_vs_serial_triangle_attention():
    N = 128; d = 64; num_heads = 4; B = 1
    z = torch.randn(B, N, N, d, device="cuda")
    mask = torch.ones(B, N, N, device="cuda")

    # Serial
    serial_layer = TriangleAttention(c_pair=d, num_heads=num_heads, orientation="per_row")
    out_serial = serial_layer(z, mask)

    # Distributed
    cp_mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("cp_row", "cp_col"))
    ring_layer = RingTriangleAttention(c_pair=d, num_heads=num_heads,
                                       orientation="per_row", cp_mesh=cp_mesh)
    ring_layer.load_state_dict(serial_layer.state_dict())
    z_shard = _shard_pair(z, cp_mesh)
    out_shard = ring_layer(z_shard, mask)
    out_gathered = out_shard.full_tensor()

    torch.testing.assert_close(out_serial, out_gathered, atol=1e-3, rtol=1e-3)
```

Same pattern for `DistributedTriangleMult` and `RingOuterProductMean`.

### Scale Test

```python
# Run with torchrun --nproc_per_node=4
python -m protenix.distributed.predict \
  --input tests/fixtures/long_chain_N4096.json \
  --size_cp 4 --out_dir /tmp/cp_test
```

Validate: (a) no OOM, (b) output CIF RMSD ≤ 0.5 Å vs 1-GPU run at smaller N.

______________________________________________________________________

## 12. Non-Goals

- **Training with CP** — training requires gradient through ring ops; out of scope for v1
- **CP + DDP combined at inference** — wasteful; all GPUs needed for one structure
- **Triton ring kernel** — replacing `TriAttentionFunction` with a ring-aware Triton kernel
  is a performance optimization, not a correctness concern; deferred
- **Structure module sharding** — the diffusion module is O(N) and not the bottleneck
- **Multi-node inference for normal structures** — single-node 4-GPU (size_cp=4) covers
  N up to ~16 000 on H100-80G; multi-node CP adds NCCL network latency
- **MSA preprocessing** — preprocessing (`protenix prep`) remains serial; CP only covers
  the neural-network forward pass
- **Affinity prediction** — not currently in Protenix v1; would follow same CP pattern
  if added
