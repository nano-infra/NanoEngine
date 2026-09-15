# GLM Recurrent MTP

GLM checkpoints expose one physical multi-token predictor layer. With `num_speculative_tokens=N`, DLEngine recurrently executes that same layer `N` times, retains `N` draft tokens, and derives a target verification width of `K=N+1`. The production GLM configuration uses `N=5` and `K=6`; it does not load five or six predictor layers.

## Linear draft and verify

DLEngine implements a line rather than a speculation tree:

```text
target input:  [base, draft_1, draft_2, draft_3, draft_4, draft_5]
target output: [verify_1, verify_2, verify_3, verify_4, verify_5, bonus]
```

One decode step has three phases:

1. The target model verifies the base token and five saved drafts in one fixed-width forward.
2. Sampling commits the accepted draft prefix plus one recovery or bonus token.
3. The predictor replays from the committed target hidden state and recurrently generates the next five-token draft line.

This fixed sequence-major layout is CUDA Graph friendly and avoids tree-shaped cache and recurrent-state ownership.

## Sampling semantics

The predictor uses greedy draft tokens. This does not force target sampling to be greedy:

- With `temperature=0`, verification compares the target argmax with each draft.
- With `temperature>0`, each greedy draft is a one-hot proposal. DLEngine applies rejection sampling against the original target distribution and samples the residual distribution after a rejection.
- Completion log probabilities come from the original target distribution.

The accepted length may range from zero to five. When all five drafts are accepted, the sixth target row supplies the bonus token. These rules preserve the target distribution; acceptance rate affects performance, not sampling correctness.

## Predictor cache and Indexer invariants

Because the checkpoint contains one physical predictor layer, predictor MLA KV occupies one cache layer regardless of recurrent depth.

The first predictor call computes the DSA Indexer TopK state. Remaining recurrent calls reuse that state when the model enables shared Indexer iteration; they do not rerun the Indexer five times. The reused state must be staged into stable hot storage before an earlier temporary output page can be recycled.

Target verification still owns six distinct output KV slots. A HiSparse decode engine may merge the six TopK rows into one request-wide hot union, but it must not alias the six output locations.

## Cache reservation

For `N` speculative tokens:

```text
prefill reservation = prompt tokens + N
decode reservation  = visible tokens + 2N
```

Prefill reserves the first draft line. Decode reserves the current `N+1` verification span and capacity for the next recurrent draft line. The scheduler truncates a committed bundle at EOS, `max_tokens`, or `max_model_len`.

For GLM HiSparse with `N=5` and `index_topk=2048`, the request hot-buffer lower bound is:

```text
(N + 1) × index_topk = 6 × 2048 = 12288
```

## Supported topologies

| Topology                             | Status            | Notes                                                                                               |
| ------------------------------------ | ----------------- | --------------------------------------------------------------------------------------------------- |
| Hybrid, `pp=1`                       | Supported         | Target, predictor, and cache remain colocated.                                                      |
| PD with PP prefill and `pp=1` decode | Supported         | Predictor runs on the final prefill stage; predictor KV and draft handoff migrate with the request. |
| HiSparse decode plus recurrent MTP   | Supported for GLM | Six TopK rows share a hot union and retain distinct output slots.                                   |
| PP decode                            | Unsupported       | Decode workers must use `pp=1`.                                                                     |
| Tree speculation                     | Unsupported       | Only the linear draft line is implemented.                                                          |
| Non-GLM HiSparse plus multi-step MTP | Unsupported       | Runtime configuration rejects unsupported combinations.                                             |

Multi-step GLM MTP currently targets Hopper with `attention_tp=1` and `attention_sp=1`.

## PD handoff

Only the final prefill PP stage executes the predictor. The handoff contains:

- target MLA/DSA cache state;
- the single predictor-layer KV state;
- the five-token draft line and its row ownership metadata.

The decode engine restores those components before the first target verify. A missing, stale, or mismatched handoff row must fall back safely to a normal target decode instead of consuming invalid drafts.

## Configuration

Use the same speculative depth on both PD roles:

```bash
# Prefill role
--mode prefill
--pp 8
--max_num_batched_tokens 8192
--pp_prefill_scheduler_depth 0
--num_speculative_tokens 5

# Decode role
--mode decode
--attention_dp 8
--ffn_ep 8
--num_speculative_tokens 5
--enable_hisparse true
--hisparse_device_buffer_size 12288
```

`max_num_batched_tokens` bounds one stage forward. It is not the total scheduler admission window. With `pp_prefill_scheduler_depth=0`, DLEngine automatically admits enough microbatches to keep the pipeline full and expands to at most 64 consecutive microbatches for long prompts.

See [Production Serving](./online-serving.md) for the complete service topology.

## Validation checklist

- Exercise both `temperature=0` and `temperature>0`.
- Compare output distribution semantics, not byte-identical text across different parallel layouts.
- Record output wall time, output tokens, output steps, and Tokens/Step together.
- Test batch reorder, shrink, prefix-cache hits, and a prompt that crosses cache-page boundaries.
- For PD, confirm predictor KV and draft handoff restoration before the first verify.
- For HiSparse, verify six distinct output slots and sufficient hot-buffer capacity.
- Treat compressed synthetic long prompts as capacity tests, not acceptance-rate benchmarks.
