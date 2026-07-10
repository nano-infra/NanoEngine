# LoongServe-style SP<=8 Pure Decode Profile Execution Plan

## Goal

We need enough profile data to decide how the pure-decode scheduler chooses:

```text
d_attn in {1, 2, 4, 8}
```

The target policy is LoongServe-style:

```text
offline profile -> build SIB / threshold table -> runtime table lookup
```

The target runtime is pure decode only:

```text
dummy_prefill=True
mode=decode
loop_count=1
cuda_graph_mode=full
DLSlime RPC enabled
```

Prefill is not measured. Expert routing skew is not measured because the target
setting uses forced uniform EP.

All GPU/Ray/CUDA execution commands must be run with elevated permission.

## Current Status

Existing profile files:

```text
docs-dev/profile-results/loongserve_16g_pilot_20260709.csv
docs-dev/profile-results/loongserve_16g_longctx_640k_20260709.csv
docs-dev/profile-results/loongserve_16g_longctx_1m_20260709.csv
```

Current 16-GPU data can support only this interim rule:

```text
if d=1 fits:
    use d=1
elif d=8 fits:
    use d=8
elif d=4 fits:
    use d=4
elif d=2 fits:
    use d=2
else:
    no normal intra-node SP<=8 plan
```

It cannot yet determine production performance thresholds because it only covers
`B=16`, mostly uniform lengths, and has not proven that `d<=8` communication is
strictly node-local.

## Profile Questions

The profile must answer these questions:

1. While `d=1` fits, is there any `(B, W_attn, L distribution)` where `d=2/4/8`
   is at least 5% faster at p90?
2. When `d=1` does not fit, which viable `d in {2,4,8}` is fastest?
3. Are `d=2` and `d=4` ever useful, or are they only fallback states when `d=8`
   is unavailable?
4. Does `d=8` actually avoid cross-node communication, or does the backend still
   pay hidden full-world SP16 overhead?
5. Does the profile-table scheduler beat fixed baselines on DPSK issue1/issue5
   pure-decode replay?

If the answer to question 1 is "no", then the final scheduler has no
performance-only scale-up threshold for the measured domain; it is a memory-first
scheduler.

## Required Runner Work

The existing runner:

```text
docs-dev/loongserve_decode_profile_runner.py
```

currently supports uniform synthetic decode points:

```text
B, L, d_attn
```

It needs two additions before final profile:

```text
--dataset-path PATH
--dataset-mode {short_only,natural,one_long,long_heavy,tail_stress}
--decode-step-offsets 0,128,512,1024
--sample-seed INT
--sample-repeat INT
```

Dataset mode should construct pure-decode batches without running prefill:

```text
L_i = prompt_len_i + decode_step_offset
active only if decode_step_offset < output_len_i
```

The runner must write these extra fields:

```text
profile_kind              # uniform or dataset_replay
dataset_name
dataset_mode
sample_seed
decode_step_offset
B
L_avg
L_p50
L_p90
L_max
W_attn
d_attn
viable
skip_reason
node_local
```

Keep existing fields:

```text
step_mean_ms
step_p50_ms
step_p90_ms
model_mean_ms
model_p50_ms
model_p90_ms
scheduler_mean_ms
postprocess_mean_ms
master_counts
occupied
append
draining
sp_size_hist
sp_send_counts
sp_recv_counts
```

## Phase 0: Backend Locality Validation

Purpose: make sure `d<=8` profile data represents node-local SP, not hidden SP16
communication.

Run minimal points:

```text
B in {16}
W_attn in {640K, 1M}
d_attn in {1,2,4,8}
warmup=3
steps=10
```

Expected checks:

```text
occupied ranks for d=8 are within one 8-rank local group
append ranks for d=8 are within one 8-rank local group
sp_send_counts outside occupied group are zero
sp_recv_counts outside occupied group are zero
worker logs / NCCL traces show no cross-node ranks for d<=8
```

Decision:

```text
if locality fails:
    label all existing d<=8 data as "full-world backend profile"
    do not use it as final intra-node threshold
    fix active subgroup / DLSlime occupied-rank communication first
else:
    continue to threshold profile
```

Output:

```text
docs-dev/profile-results/loongserve_sp8_phase0_locality_YYYYMMDD.{jsonl,csv}
```

## Phase 1: Controlled Uniform Pilot

Purpose: quickly find whether any performance-only scale-up exists before
spending time on the full matrix.

Matrix:

```text
B in {16,32,64,128}
W_attn in {256K,512K,640K,768K,850K,1M,2M}
d_attn in {1,2,4,8}
```

For each point:

```text
L = ceil(W_attn / B)
skip if L < d_attn
skip d if memory fit is impossible
warmup=5
steps=20
```

Viability by observed 16-GPU raw capacity:

```text
raw_tokens_per_rank = 913472
initial safety_margin = 0.90
usable_tokens_per_rank = floor(raw_tokens_per_rank * safety_margin)
```

The runner should still use actual allocator success/failure as final truth.

Example commands, to be run with GPU escalation:

```bash
python docs-dev/loongserve_decode_profile_runner.py \
  --batches 16,32,64,128 \
  --lengths 16384,32768,40960,49152,53248,65536,131072 \
  --dops 1,2,4,8 \
  --warmup 5 \
  --steps 20 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 262144 \
  --out docs-dev/profile-results/loongserve_sp8_phase1_uniform_pilot_YYYYMMDD.jsonl
```

Note: the current runner takes lengths, not W buckets. For clean W-bucket output,
prefer adding `--w-attn-buckets` before running the final version.

Decision after Phase 1:

```text
if d=1 remains fastest for all memory-fit comparable points:
    mark performance scale-up as unproven
    focus Phase 2 on memory-forced and high-B edge points
else:
    densify around the first W/B buckets where d>1 wins
```

Output:

```text
docs-dev/profile-results/loongserve_sp8_phase1_uniform_pilot_YYYYMMDD.{jsonl,csv}
```

## Phase 2: Controlled Uniform Full Profile

Purpose: build the controlled part of the SIB.

Matrix:

```text
B in {1,2,4,8,16,32,64,128}
W_attn in {64K,128K,256K,512K,640K,768K,850K,1M,1536K,2M,3M,4M,6M}
d_attn in {1,2,4,8}
warmup=20
steps=100
repeat=2
```

For each point:

```text
L = ceil(W_attn / B)
skip if L < d_attn
skip d if allocator cannot fit
```

Required shape variants:

```text
uniform:
    all L_i = L

one_long:
    one request carries most W_attn, remaining B-1 are short

two_long:
    two long requests, remaining B-2 are short
```

The shape variants matter because `W_attn` alone does not capture all metadata,
remote receive count, or master load imbalance.

Output:

```text
docs-dev/profile-results/loongserve_sp8_phase2_uniform_full_YYYYMMDD.{jsonl,csv}
```

## Phase 3: DPSK Dataset Replay Validation

Purpose: validate the SIB-derived threshold on workload-realistic decode shapes.

This phase is not the source of the low-level threshold. The paper-aligned
threshold source is the controlled profile in Phase 1/2. DPSK replay answers a
different question: after the threshold is derived from controlled `(B, W_attn,
L distribution, d_attn)` measurements, does it make the right decisions on the
actual long/short mixture we care about?

Datasets:

```text
DPSK-issue1:
/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.01_n60000_60k.csv

DPSK-issue5:
/mnt/nvme1n1/ml_research/linbinbin1/paper-nanolmdeploy/dataset/sharegpt-4o-mixlong-0326/sharegpt4o-random_geminiissue_r0.05_n60000_60k.csv
```

Batch families:

```text
short_only:
    sample total_len < 8K

natural:
    sample from full CSV distribution

one_long:
    force exactly one total_len >= 64K request plus B-1 short requests

long_heavy:
    sample only total_len >= 64K

tail_stress:
    sample only total_len >= 512K
```

Matrix:

```text
dataset in {DPSK-issue1,DPSK-issue5}
dataset_mode in {short_only,natural,one_long,long_heavy,tail_stress}
B in {16,32,64,128}
decode_step_offset in {0,128,512,1024}
d_attn in {1,2,4,8}
warmup=20
steps=100
sample_repeat=3
```

Skip rules:

```text
skip sampled request if decode_step_offset >= output_len
skip d if memory fit is impossible
skip d if it violates node-local SP<=8 policy
```

Output:

```text
docs-dev/profile-results/loongserve_sp8_phase3_dpsk_replay_YYYYMMDD.{jsonl,csv}
```

## Phase 4: SIB and Threshold Extraction

Input:

```text
phase1/phase2 jsonl files
phase3 jsonl files are validation/calibration inputs, not the primary threshold source
```

Generate from Phase 1/2:

```text
docs-dev/profile-results/loongserve_sp8_decode_sib_YYYYMMDD.json
docs-dev/profile-results/loongserve_sp8_decode_thresholds_YYYYMMDD.md
```

For every controlled runtime bucket:

```text
key = (
    profile_kind,
    dataset_mode,
    B_bucket,
    W_attn_bucket,
    L_p90_bucket,
    L_max_bucket
)
```

Candidate set:

```text
D_viable = {d in {1,2,4,8} where d fits memory and is node-local}
```

Best and near-optimal:

```text
T(d) = step_p90_ms, or model_p90_ms if scheduler/postprocess noise dominates
best = min_d T(d)
d_near = smallest d where T(d) <= 1.05 * best
```

Performance scale-up threshold exists only if:

```text
T(current_d) - T(candidate_d) >= max(5% of T(current_d), 0.1 ms/layer)
and the result is stable across repeats
and node-locality validation passed
```

Otherwise mark:

```text
performance_threshold = none
reason = "d=1 remains near-optimal" or "data not stable" or "backend not node-local"
```

Memory threshold:

```text
d_mem = smallest d where placement fits usable KV capacity
```

Final table entry:

```text
d_target = max_by_constraint(d_mem, d_near)
```

But if `d=1` is near-optimal and memory fits, store `d_target=1`.

Use Phase 3 only to check:

```text
1. whether runtime buckets cover the DPSK length distribution;
2. whether same-W_attn mixed-length batches need a separate L_p90/L_max bucket;
3. whether the selected d is stable under realistic batching;
4. whether profile_scheduler beats fixed_d1/fixed_d8 on pure-decode replay.
```

If Phase 3 disagrees with Phase 1/2, do not directly overwrite thresholds with
dataset-specific constants. First add the missing controlled bucket or shape
variant, rerun it, then regenerate the SIB.

## Phase 5: Scheduler Validation

After implementing the profile-table scheduler, run end-to-end pure decode:

```text
fixed_d1
fixed_d8
profile_scheduler
```

Datasets:

```text
DPSK-issue1
DPSK-issue5
```

Metrics:

```text
decode step p50/p90/p99
model p50/p90/p99
request output latency p50/p90/p99
tokens/s
time spent at d=1/2/4/8
scale-up count
scale-down count
no-fit d<=8 count
node-locality violation count
```

Pass conditions:

```text
1. d_target is always in {1,2,4,8}.
2. Normal mode never crosses node.
3. Short-heavy DPSK traffic is not worse than fixed_d1 by more than 3%.
4. Long memory-forced traffic succeeds where fixed_d1 cannot.
5. If d>1 is selected for performance, profile data shows >=5% stable p90 gain.
```

## Minimum Data Needed To Decide Policy

Do not claim a production threshold until all are true:

```text
Phase 0 node-locality validation passed
Phase 2 includes B>=64 and W_attn>=640K comparable points
each threshold bucket has at least 2 stable repeats
raw JSONL/CSV and derived SIB are saved under docs-dev/profile-results/
```

Do not claim the scheduler is validated for DPSK traffic until:

```text
Phase 3 includes DPSK-issue1 and DPSK-issue5 replay
profile_scheduler has been compared with fixed_d1 and fixed_d8
```

If Phase 2/3 still show no stable performance win for `d>1` while `d=1` fits,
then the final policy should explicitly be:

```text
memory-first scheduler:
    d=1 while it fits;
    when memory forces scale-up, choose fastest viable d<=8 from SIB;
    otherwise keep d=1.
```

That is still a valid LoongServe-style result because the profile/SIB proves the
absence of a performance scale-up threshold in the measured domain.
