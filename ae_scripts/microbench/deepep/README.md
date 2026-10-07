# Shared DeepEP low-latency measurement engine

The launcher runs the DeepEP `tests/test_low_latency.py` implementation vendored
with this artifact for an explicit token list. It measures low-latency dispatch,
combine, and their combined latency on all distributed ranks. It does not
decide which token points a figure needs.

Files:

- `run_one_low_latency.py`: imports the vendored `test_main` and supplies one
  configurable token count.
- `run_low_latency_sweep.sh`: runs the requested token list on every node and
  retains one raw log per node/token.
- `parse_low_latency_logs.py`: creates the representative-rank summary CSV
  consumed by Fig. 3. Raw logs retain all rank lines needed by Fig. 5's
  figure-specific aggregation.
- `third_party/deepep/`: pinned upstream benchmark sources, license, and
  provenance. The compiled `deep_ep` package is still installed separately.

Run all commands from the `ae_scripts` directory. Apply the following
setup in the same environment on every participating node:

```bash
export MASTER_ADDR=<node-0-ip>
export MASTER_PORT=<free-base-port>
export WORLD_SIZE=<number-of-nodes> # node count, not GPU rank count
export EP_TEST_NUM_PROCESSES=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export EP_TEST_NUM_EXPERTS=256      # DeepSeek-V3 routed experts
export EP_TEST_HIDDEN=7168
export EP_TEST_NUM_TOPK=8
export EP_TEST_SEED=1
export TOKENS="<space-separated-points-from-the-figure>"
export OUTPUT_DIR=<shared-or-node-local-output-directory>
```

Launch concurrently on every node:

```bash
RANK=<node-rank> bash microbench/deepep/run_low_latency_sweep.sh
```

`MASTER_PORT + token` is used for each rendezvous, so the base port and token
order must match on all nodes. Keep the allocation's validated NCCL, NVSHMEM,
RDMA, HCA/GID, traffic-class, socket-interface, and library-path settings.

Fig. 3 defines `TOKENS` as `1..8` followed by `16..256` in steps of 8. Fig. 5
reads `deepep.microbenchmark_batch_cases` from its extracted snapshot JSON and
uses those values instead. See the figure READMEs for the exact presets and
postprocessing semantics.
