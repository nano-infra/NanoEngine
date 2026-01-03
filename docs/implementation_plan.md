# Distributed Process Group Plan

## Goal Description

Enable multi-GPU/distributed execution by launching multiple `DummyRunner` instances (world_size=8) and establishing a communication group (ProcessGroup).

- **Spoke**: Spawn 8 `DummyRunner` Actors.
- **Initialization**: Implement `init_process_group` logic in `DummyRunner`.
- **Communication**: Use `c10d` (Gloo/NCCL) or `DLSlime` to sync tensors.

## Architecture Design

### Components

- **Manager Actor (Executor)**:
  - Spawns 8 `DummyRunner` instances.
  - Generates `rank` and `world_size` config for each.
  - Distributes the `Master Address/Port` or `Store` info for C10D rendezvous.
- **DummyRunner (Worker)**:
  - Accepts `rank`, `world_size`, and `rendezvous_info` in `init()`.
  - Calls `c10d::ProcessGroup::create(...)` or `torch::distributed::init_process_group`.
  - Performs on-device (Mock) collective ops (e.g., `all_reduce`).

## Proposed Changes

### \[MODIFY\] `nanodeploy/worker/dummy_runner.h`

- Add `init_distributed(int rank, int world_size, std::string master_addr, int master_port)` method.
- Store `c10d::ProcessGroup` shared pointer.

### \[NEW\] `tests/test_distributed.cpp`

- Spawns 8 actors.
- Triggers initialization on all.
- Triggers a collective op verify synchronization.
