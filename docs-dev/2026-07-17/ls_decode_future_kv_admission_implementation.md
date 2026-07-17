# LS Decode future-KV admission implementation

Date: 2026-07-17

## Outcome

NanoDeploy LS Decode admission now applies the same future-KV high-water-mark
formula used by LoongServe's `ReqQueue._can_add_new_req()` before committing a
logical batch. The guard is enabled by default through
`ls_decode_enable_future_kv_admission=True`.

The target benchmark sets `ignore_eos=True` and supplies the exact output length
as `max_tokens`. Therefore, the implementation follows LoongServe's
`ignore_eos`/busy branch and uses the full user-provided maximum output length.
It intentionally does not copy LoongServe's optional light-load 1024-token/1.1x
soft horizon.

## Peak formula

For each live or candidate request, define:

- `held_tokens`: KV held now or immediately after admission;
- `remaining_iterations`: remaining Decode iterations derived from
  `max_tokens`.

NanoDeploy uses the same frontier convention as LoongServe:

- running: `(prompt + generated, max_tokens - generated - 1)`;
- waiting: `(prompt + 1, max_tokens - 2)`.

After sorting requests by descending `remaining_iterations`, the projected peak
is:

```text
peak = max_k(prefix_sum(held_tokens, k) + k * remaining_iterations[k])
```

This evaluates the KV high-water mark immediately before each request-completion
boundary. Requests that finish earlier release their KV before longer requests
reach their terminal lengths, so heterogeneous output lengths do not get
naively summed as if all terminal KV were simultaneously resident.

## Integration points

The guard runs in both admission feasibility layers:

1. `_ls_batch_fits_empty_system()` checks future feasibility before a logical
   batch is sealed. It may shrink only the still-unsealed candidate and prevents
   a permanently impossible sealed batch.
2. `_plan_ls_initial_placement()` checks the candidate batch together with the
   existing requests in a prospective merge target. A future-infeasible batch
   stays queued without allocating or publishing any partial state.

Capacity is scoped to the prospective group. It includes:

- physical KV blocks already owned by that group's existing requests;
- currently free blocks on ranks that the group may use now or through memory
  scale-up.

It excludes capacity held by other groups. Block capacity is converted to token
slots using the configured KV block size. Existing per-rank prompt placement,
receiver metadata, pending-append, and reservation-headroom checks remain in
force after this aggregate filter.

When memory scale-up is disabled, the estimate only includes the base allocation
and the ranks selected by the candidate initial placement. When it is enabled,
all currently available scale-up ranks are included.

## Behavioral change

A request whose future maximum cannot fit even in an otherwise empty DP now
fails during pre-seal feasibility instead of being admitted and inevitably
preempted later. A batch whose prompts fit now but whose projected future peak
does not fit stays pending until capacity is released.

The default-on flag can be disabled for controlled A/B comparison or rollback:

```python
ls_decode_enable_future_kv_admission = False
```

## CPU validation

- Future-infeasible merge remains pending without changing physical block
  counts.
- After the occupying request releases capacity, the same pending batch admits
  atomically.
- Heterogeneous requests are admitted using completion overlap when the
  LoongServe peak fits, even if summing all terminal lengths would not fit.
- LS scheduler, planner, metadata, and config tests pass without GPU execution.

## Remaining limitation

This is LoongServe's aggregate token-envelope admission policy adapted to
NanoDeploy's group-owned block pool. It does not reserve a time-indexed per-rank
future placement. Physical 64-token block fragmentation and owner/receiver
constraints continue to be enforced by the existing current-placement and
per-iteration planners. If aggregate admission still proves optimistic in a
long run, the next step is a per-rank shadow growth-owner plan rather than
weakening the aggregate high-water guard.
