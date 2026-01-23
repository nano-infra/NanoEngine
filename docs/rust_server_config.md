# Implement Rust Server Configuration

## Objective

Enable dynamic configuration of the Engine initialization parameters via command-line arguments in the Rust server, replacing hardcoded values.

## Changes

1. **Configuration Module (`server/src/config.rs`)**:

   - Created a `Config` struct to hold model and runtime parameters.
   - Implemented a dependency-free custom argument parser `Config::parse()` to handle command-line flags.

2. **Main Integration (`server/src/main.rs`)**:

   - Integrated `Config` into the main application flow.
   - The `EngineInitReq` payload is now populated with values from the command line.

## Usage

The server now accepts the following arguments:

- `--model-config-path <PATH>` (Required): Path to the model configuration JSON file.
- `--ffn-ep <INT>` (Default: 1): Feed-Forward Network Expert Parallelism.
- `--ffn-tp <INT>` (Default: 1): Feed-Forward Network Tensor Parallelism.
- `--ffn-dp <INT>` (Default: 1): Feed-Forward Network Data Parallelism.
- `--pp <INT>` (Default: 1): Pipeline Parallelism.
- `--attention-dp <INT>` (Default: 1): Attention Data Parallelism.
- `--attention-tp <INT>` (Default: 1): Attention Tensor Parallelism.
- `--attention-sp <INT>` (Default: 1): Attention Sequence Parallelism.
- `--enable-cuda-graph`: Enable CUDA Graph support (flag).

### Example

```bash
./target/debug/nanodeploy-server \
  --model-config-path /models/qwen3-235B-Instruct-2507-FP8/config.json \
  --pp 2 \
  --attention-tp 4 \
  --enable-cuda-graph
```

## Implementation Details

The argument parser manually iterates over `std::env::args()` to avoid introducing external dependencies like `clap` which were facing registry resolution issues in the environment.
