# DLEngine Blog

These notes are grouped by the question they answer.

## K3 model and performance

- [Kimi K3 Structure and Capacity](dlengine-kda-evaluation.md) — model structure, capacity, intra-layer parallelism, Prefill, Decode, and 1M-context serving.

## Engine and serving architecture

- [DLEngine Engine](dlengine-engine.md) — runtime architecture and distributed execution.
- [MTP speculative decoding](dlengine-mtp.md) — recurrent MTP execution and verification.

## Sparse attention and capacity

- [HiSparse Capacity](dlengine-hisparse-capacity.md)
- [HiSparse Capacity（中文）](dlengine-hisparse-capacity.zh.md)
- [NSA](dlengine-nsa.md)

The K3 evaluation is the main end-to-end study. The other notes document
specific runtime mechanisms and capacity models; their measurements should not
be mixed with the K3 GB200 results unless the hardware and workload are stated.
