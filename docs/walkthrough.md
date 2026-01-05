# Debugging Distributed AllReduce in NanoDeploy

## Goal

Achieve a stable distributed execution of `DummyRunner` across 8 instances, verifying correct `c10d` (NCCL) initialization and accurate AllReduce summation (Rank 0-7 sum to 28).

## Critical Issue: `malloc(): invalid size` Panic

During the initial multi-process test spawn, the Spoke Daemon crashed with persistent heap corruption errors.

### Root Cause Analysis

The Spoke Daemon is multi-threaded (network IO, client handling). Standard `fork()` in a multi-threaded program only replicates the calling thread. If another thread holds a `malloc` lock during `fork()`, the child process inherits a locked mutex with no owner, leading to deadlocks or state corruption when it attempts to allocate memory.

## Solution Journey

### Final Solution: Fork-Exec Architecture (Robust)

We migrated Spoke's process management to a true **Fork-Exec** model, similar to industrial-grade servers (Nginx, Redis).

1. **Daemon CLI Upgrade**: Added `--worker <type> <id> ...` mode to `nanodeploy_agent`.
2. **Agent Logic**: Replaced `fork()` with `fork() + execv("/proc/self/exe", args)`.

**Why this works**: `execv` completely replaces the process image. The child process starts with a fresh heap, fresh stack, and NO inherited lock states. It is 100% thread-safe by definition.

## Verification

We ran the `test_dummy_runner` which spawns 8 actors and performs an AllReduce.

### Results

1. **Stability**: Zero crashes during rapid spawning.
2. **Initialization**: 8/8 Ranks connected to TCPStore and initialized ProcessGroupNCCL via concurrent Client Pool.
3. **Correctness**: Total sum = 28.0 (Rank 0-7: 0+1+2+3+4+5+6+7). **PASSED**.

## Conclusion

The Distributed AllReduce primitive is now fully functional and robust. The new `Fork-Exec` architecture provides a solid foundation for scaling NanoDeploy to complex distributed inference workloads.
