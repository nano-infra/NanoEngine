# Routing Metadata Fusion P/D ShareGPT Validation

Date: 2026-08-23 UTC

## Result

The corrected two-node, real-weight ShareGPT P/D run passed transport and
execution validation: 101/101 requests completed, every request produced a
nonempty decoded string, the process exited with code 0, and no Python, CUDA,
NCCL, or Ray runtime error appeared.

The generated text is not acceptable as a complete text-quality result. Only
13/101 outputs naturally emitted the DeepSeek end-of-sentence token. The other
88 outputs reached the benchmark's per-request reference-length limit, and 84
of those ended without terminal punctuation. Many are visibly cut in the
middle of a sentence, list item, code block, or word.

## Configuration

- Ray GCS: `10.102.243.60:6380`
- Prefill: `10.102.98.154:6006`, 8 H200 GPUs
- Decode: `10.102.243.60:6006`, 8 H200 GPUs, `bucket-sp8`
- Parallel builder:
  `examples/pd_disagg_deepseek_v3_parallel.py::build_engines_parallel`
- Dataset: `/mnt/nvme1n1/ml_research/linbinbin1/sharegpt.json`
- Valid human/assistant pairs scanned: 92,704
- Reservoir sample: 101 pairs, seed 0
- Prompt tokens P50/P90/P99/max: 28/291/2,082/2,303
- Arrival window: 120 seconds
- Target rate: 1 request/second, Poisson
- Sampled rate: 101 requests, 0.842 requests/second
- Real DeepSeek-V3 weights
- Warmup requests: 8
- Maximum model length: 1,000,000
- Maximum sequences: 8
- Decode policy:
  `1:1024-63488;5:63489-210944;6:210945-399360;7:399361-428032;8:428033-1048576`

Both engines loaded concurrently. Prefill initialized in 178.92 seconds,
decode in 289.27 seconds, and parallel wall time was 289.27 seconds. Each
decode worker captured 4 local and 8 SP CUDA graphs. P/D warmup took 6.51
seconds.

## Serving metrics

| Metric | Result |
| --- | ---: |
| Requests sent/completed | 101 / 101 |
| Input tokens | 14,264 |
| Output tokens | 27,702 |
| Injection window | 120.00 s |
| Drain time | 60.96 s |
| Total measured time | 180.96 s |
| Throughput | 153.08 tokens/s |
| Average TTFT | 1,010.91 ms |
| Average E2E latency | 28.27 s |
| TPOT average | 97.53 ms/token |
| TPOT P50 / P90 / P95 / P99 | 98.73 / 99.20 / 99.31 / 100.13 ms/token |
| Queued ITL average | 97.64 ms/token |
| Queued ITL P50 / P90 / P95 / P99 | 98.57 / 98.71 / 99.00 / 99.48 ms/token |

## Text review

Automated integrity checks over all 101 JSONL records found:

- 101 nonempty decoded outputs
- no Unicode replacement characters
- no NUL or unexpected ASCII control characters
- one duplicate output, explained by the distinct prompts `hi` and `hello`
  both naturally producing the same greeting
- no evidence of outputs being assigned to a different request

The output-limit diagnosis is exact:

- 13 outputs ended with tokenizer EOS
- 88 outputs had no EOS and their generated-token count exactly equaled the
  tokenized reference-response length used as `max_tokens`
- 84 of those 88 capped strings ended without terminal punctuation

The behavior comes from `examples/bench_pd_serving.py`: it initially sets the
output limit to the reference response's token count and then applies
`--sharegpt-max-output-tokens` as an additional upper bound. A sampled model
response need not have the same length as the reference, so this is suitable
for controlling benchmark workload length but not for checking whether a
free-form response finishes correctly.

Manual review of all prompt/output starts and endings found that most generated
text begins on topic and remains fluent until the hard cutoff. There was no
systematic gibberish, token corruption, or cross-request mixing. Some sampled
prompts are context-dependent fragments such as `continue`, `Please go on`, or
references to a prior file structure or chapter; without the missing earlier
turns, their semantic correctness cannot be evaluated and the model often
responds generically.

One sampled source prompt contains real-looking names and email addresses. The
generated JSONL stores complete prompts and references, so it should be treated
as potentially sensitive and not published without sanitization.

## Artifacts

- Full log:
  `bench_logs/pd_sharegpt_rate1_2min_parallel_20260823_135644/run.log`
- Generated prompt/reference/output records:
  `bench_logs/pd_sharegpt_rate1_2min_parallel_20260823_135644/generated_text.jsonl`
- Per-request ITL records:
  `bench_logs/pd_sharegpt_rate1_2min_parallel_20260823_135644/itl_samples.jsonl`
- Process exit code: 0
- Post-run Ray resources: `0.0/16.0 GPU` in use, no pending nodes or failures
