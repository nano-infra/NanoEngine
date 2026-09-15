# Performance Profiling

Use timeline evidence and wall-clock measurements together. A profiler explains where time goes, but its instrumentation overhead means profiled wall time should not be compared directly with an unprofiled result.

## Reproducible comparison

Keep these inputs identical between baseline and optimized runs:

- commit and dependency build;
- model and numerical format;
- GPU type, count, topology, and parallel configuration;
- prompt token IDs and output-token limit;
- sampling parameters, including temperature and top-p;
- cache state, prefix-cache policy, and warmup count;
- profiler options.

Report cold startup separately from warmed serving. Run multiple unprofiled repetitions for latency and throughput, then capture a smaller number of representative profiled requests.

## Capture a bounded timeline

Configure a persistent output root with `--profiler_dir`. Start and stop profiling through each engine's direct management endpoint:

```bash
curl -sS -X POST "http://${ENGINE_IP}:${ENGINE_PORT}/start_profiler" \
  -H "content-type: application/json" \
  -d '{"trace_name":"glm-mtp-baseline"}'

# Send the inference request or fixed request batch.

curl -sS -X POST "http://${ENGINE_IP}:${ENGINE_PORT}/stop_profiler"
```

Each worker writes below `<profiler_dir>/<trace_name>/`. Use a durable mounted directory rather than container-local temporary storage. Keep the request payload, server command, commit, environment summary, and log beside the trace.

## Interpret serving metrics

The completion log reports several different quantities:

| Metric           | Definition                                                               | MTP interpretation                                                                                                                                                                  |
| ---------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Output Length`  | Number of committed generated tokens                                     | Includes every accepted draft and recovery or bonus token.                                                                                                                          |
| `Output Steps`   | Number of scheduler output steps                                         | One MTP step may commit multiple tokens.                                                                                                                                            |
| `Tokens/Step`    | `Output Length / Output Steps`                                           | Direct aggregate signal for speculative acceptance and bonus-token yield.                                                                                                           |
| `ITL Wo Queue`   | Mean wall-clock gap between recorded output tokens after the first token | Bundle-internal tokens have near-zero gaps, so MTP amortizes one model step across them. Initial request queueing and TTFT are excluded; active scheduling stalls can still appear. |
| `ITL With Queue` | `E2E / Output Length`                                                    | Includes prompt processing and initial queueing when those timestamps are available.                                                                                                |
| `TTFT`           | Arrival to first emitted token                                           | For PD, includes routing, prefill, migration, and first decode readiness.                                                                                                           |

Do not diagnose MTP from ITL alone. Report wall time, E2E, model-step latency, Tokens/Step, and output throughput together. A low acceptance rate raises the number of target steps even when each target verify is fast.

## MTP timeline checklist

For recurrent `N=5` GLM MTP, separate:

1. target verification GPU span and kernel count;
2. the draft-extend predictor call;
3. the remaining four recurrent predictor calls;
4. rejection sampling and postprocessing;
5. CPU scheduling gaps between verify steps;
6. expert dispatch/combine communication and wait time;
7. CUDA synchronization and graph replay boundaries.

Compare active and padded attention ranks. Collective participants may execute padding work to preserve ordering even when only one attention rank owns a request.

## PP prefill checklist

For each pipeline stage, report:

- useful GPU kernel time and kernel count;
- stage input receive and output send duration;
- pipeline bubbles;
- CPU gaps between microbatches;
- peak allocated and reserved device memory;
- prompt tokens per stage forward;
- total scheduler admission window.

`max_num_batched_tokens` bounds one stage forward. `pp_prefill_scheduler_depth` controls how many microbatches one scheduler step may admit; zero selects the automatic window. Do not infer the scheduler window by multiplying the microbatch size by PP degree.

## PD migration checklist

Measure these phases independently:

1. peer discovery and first connection;
2. assignment and descriptor construction;
3. batch submission;
4. data-transfer completion;
5. device visibility synchronization;
6. destination cache registration or restoration;
7. first target verify.

A batch read submits many scatter/gather descriptors together. It does not automatically transform a fragmented source and destination KV layout into contiguous regions. Report descriptor count before and after coalescing, bytes transferred, submission time, completion time, and effective bandwidth.

Submit independent peer batches before draining their completions so transfers can overlap. Synchronize device visibility once after all submitted reads complete unless the backend provides a stronger ordering guarantee.

If raw transfer is fast but migration plus first verify is slow, inspect Python descriptor materialization, layout fragmentation, first-connection setup, distributed scheduling, cache restoration, and first-use kernel setup before attributing the entire interval to the network.

## CUDA stream discipline

Sequential model layers should reuse process-local streams by device, role, and priority. Allocate separate roles only for operations that must overlap within a layer, and join those side streams before the next layer consumes their outputs.

Creating one persistent stream per layer does not add concurrency to a serial decoder stack. It increases retained runtime state and makes multi-rank profiler timelines unnecessarily wide.

## Artifact checklist

Every performance result should include:

- commit and dirty-tree state;
- complete launch commands with secrets and machine-specific paths redacted;
- prompt token count and a reproducible payload description;
- sampling parameters;
- cold or warmed status;
- unprofiled wall-clock repetitions;
- profiler trace and server logs;
- metric summary and the exact aggregation method;
- known limitations and pending validation.

Keep incomplete long-context measurements in the associated performance issue. Promote them to stable documentation only after the full comparison matrix is reproduced.
