# Basic Test

This test exercises both distributed serving implementations without using a
figure-specific launcher. It directly invokes the shared NanoDeploy and vLLM
E2E entry points under `start-e2e/` and runs the following cases sequentially
on two nodes with eight H200 GPUs per node:

| System | Topology | Workload | Request rate | Send duration |
|---|---|---|---:|---:|
| NanoDeploy | DP2-CP8-DCP | Issue 1% | 5 requests/s | 60 seconds |
| vLLM | DP2-TP8-DCP8 | Issue 1% | 5 requests/s | 60 seconds |

The input used by both systems is a run-local copy of the Issue 1% workload
containing requests with at most 750K total prompt and output tokens. This
keeps the smoke test within the validated two-node KV-cache capacity.

This is a functional kick-the-tires test. Its TPOT values are recorded for
diagnosis but are not compared with the four-node paper results.

## Requirements

Complete the shared setup in the root `README.md`. In particular:

- Connect exactly two eight-GPU nodes to the NanoDeploy Ray cluster.
- Reserve all 16 GPUs exclusively for the test.
- Configure the worker SSH endpoint used by the vLLM container.
- Make the model, dataset, system checkouts, and repository visible at the same
  paths on both nodes.

## Run

Run from the `ae_scripts` directory:

```bash
python3 basic-test/run_basic_test.py
```

The two 60-second measurements run sequentially. Including service startup,
request draining, validation, and cleanup, the command typically finishes
within 10--20 minutes.

To run only one implementation:

```bash
python3 basic-test/run_basic_test.py --system nano
python3 basic-test/run_basic_test.py --system vllm
```

If the current allocation differs from the preconfigured one, supply its
addresses explicitly:

```bash
python3 basic-test/run_basic_test.py \
  --ray-address <ray-head-ip>:6380 \
  --nano-master-address <ray-head-ip>:29500 \
  --vllm-master-address <head-ip> \
  --vllm-worker-host <worker-ssh-host>
```

Unless `VLLM_2NODE_H200_MASTER_ADDR` is set, the vLLM master address defaults
to the host portion of `--ray-address`. The script verifies that this address
can be bound on the local frontend node before launching either vLLM rank.

Use `--dry-run` to generate and validate both launch plans without starting GPU
processes:

```bash
python3 basic-test/run_basic_test.py --dry-run
```

## Expected result

A successful real run ends with:

```text
NanoDeploy DP2-CP8-DCP: PASS
vLLM DP2-TP8-DCP8: PASS
Basic test: PASS
```

The command returns a nonzero exit status if either system fails, does not
complete every request, or does not produce TPOT data. Results are retained in:

```text
basic-test/results/<run-id>/
├── inputs/
├── nanodeploy/
├── vllm/
└── summary.json
```

`summary.json` records the resolved configuration, request counts, Mean/P99
TPOT, and the relative paths of the underlying NanoDeploy and vLLM artifacts.
Neither this wrapper nor its dry run modifies files under `start-e2e/`.
