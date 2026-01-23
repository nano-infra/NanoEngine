# Task: Refactor Spoke into an SDK

- [x] Explore new project structure
- [x] Define CMake installation logic
  - [x] Create/Update `CMakeLists.txt` for core library
  - [x] Create/Update `CMakeLists.txt` for agent library
  - [x] Create/Update `CMakeLists.txt` for hub tool
- [x] Implement CMake Export support
  - [x] Create `SpokeConfig.cmake.in`
  - [x] Configure `install(EXPORT ...)`
- [x] Configure build output layout (bin, tools, lib)
- [x] Refactor tests to examples and add installation
- [x] Refine CMake export path (move to share/cmake)
- [x] Fix shared library runtime loading (RPATH)
- [x] Rename binary directory back to bin
- [x] Refine Spoke elegant exports (Spoke::daemon, Spoke::core)
- [x] Refine DLSlime CMake export path (move to share/cmake)
- [x] Debug NanoDeploy Spoke integration (fix find_package order)
- [x] Integrate DummyRunner as Spoke Actor
  - [x] Create nanodeploy_agent executable
  - [x] Configure installation for python bin
  - [x] Refactor DummyRunner (Pure C++)
  - [x] Implement DummyRunnerActor (In Executor)
- [x] Verify with C++ Test
  - [x] Create test_dummy_runner.cpp
  - [x] Create tests/CMakeLists.txt
- [x] Verify installation layout

# Phase 2: Distributed Process Group (World Size 8)

- [x] Design Distributed Architecture
  - [x] Update Implementation Plan
- [x] Implement Distributed Logic
  - [x] Update DummyRunner with c10d initialization
  - [x] Create Distributed Manager/Test Client
- [x] Verify Distributed Execution
  - [x] Create tests/test_distributed.cpp (Spawn 8 actors)
- [x] Protocol Upgrade: Synchronous Spawn (Ack-based flow)
- [x] Architecture Upgrade: Fork-Exec Model (Robust Multi-Threading Support)
  - [x] Implement `--worker` in `daemon_main.cpp`
  - [x] Implement `execv` in `Agent::spawnActor`
- [x] Verify Distributed Execution
  - [x] Start 8 `DummyRunner` instances
  - [x] Verify c10d Initialization (Success)
  - [x] Verify AllReduce (Sum=28)
- [x] Reliability: Automatic Process Cleanup
  - [x] Implement `prctl(PR_SET_PDEATHSIG, SIGTERM)` in `Agent::spawnActor`
  - [x] Handle `SIGINT` in `run_daemon`
- [x] Usability: Client-side Actor Shutdown
  - [x] Implement `stopRemote` in `Client`
  - [x] Call `stopRemote` in `test_dummy_runner.cpp`

# Phase 3: Rust Server Enhancements

- [x] Implement Dynamic Configuration
  - [x] Create `server/src/config.rs`
  - [x] Add CLI argument parsing for model and parallelization params
  - [x] Integrate with `EngineInitReq` in `main.rs`
