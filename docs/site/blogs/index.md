# DLEngine Blog

These notes are grouped by the question they answer.

## K3 model and performance

- [Kimi K3 Overview](dlengine-kda-overview.md) — model structure, capacity, common layer analysis, and intra-layer parallelism.
- [Kimi K3 Prefill](dlengine-kda-prefill.md) — Prefill capacity, chunk scaling, sharding, and TTFT.
- [Kimi K3 Decode](dlengine-kda-decode.md) — Decode cache capacity, batch/context scaling, and serving selection.

## Engine and serving architecture

- [DLEngine Engine](dlengine-engine.md) — runtime architecture and distributed execution.
- [MTP speculative decoding](dlengine-mtp.md) — recurrent MTP execution and verification.

## Sparse attention and capacity

- [HiSparse Capacity](dlengine-hisparse-capacity.md)
- [NSA](dlengine-nsa.md)

The K3 evaluation is the main end-to-end study. The other notes document
specific runtime mechanisms and capacity models; their measurements should not
be mixed with the K3 GB200 results unless the hardware and workload are stated.
