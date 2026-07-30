# Repository Guidelines

## Project Structure & Module Organization

`nanodeploy/` contains the Python package. Runtime orchestration lives in `engine/`, distributed worker code in `worker/`, model and kernel implementations in `models/`, `layers/`, and `kernels/`, and request routing in `router/`. Native C++20 code and pybind11 bindings are under `csrc/nanodeploy/` and `csrc/python/`. Put regression tests in `tests/` using `test_*.py`; keep runnable demonstrations in `examples/`. Operational and analysis tools belong in `scripts/` and `utils_analysis/`. Design notes live in `docs/`; dated investigation notes may go in `docs-dev/`. Treat `build/`, profiler output, and `bench_logs/` as generated artifacts.

## Build, Test, and Development Commands

- `python -m pip install -v -e .` builds the CMake/Ninja extension and installs NanoDeploy in editable mode. Re-run it after changing C++ or bindings.
- `python -m pytest tests/test_hierarchical_control_plane.py` runs a focused CPU-friendly test module.
- `python -m pytest tests` runs test discovery; expect some suites to require CUDA, Ray, compiled extensions, or project-specific dependencies.
- `python tests/test_sequence_proxy.py` runs the standalone sequence proxy checks without pytest.
- `torchrun --nproc_per_node=4 tests/test_mla_sp_backend_correctness.py --mode eager` exercises the distributed SP backend on four visible GPUs. Adjust process count to the intended SP size.

## Coding Style & Naming Conventions

Use four-space indentation in Python, `snake_case` for functions and modules, `PascalCase` for classes, and uppercase constants. Prefer type hints for public interfaces and keep orchestration separate from model/kernel mechanics. C++ targets C++20; follow nearby formatting, use descriptive `snake_case` functions, and keep Python exposure in the binding files. No repository-wide formatter is configured, so avoid unrelated formatting churn.

## Testing Guidelines

Add a focused regression test for each behavior change. Name tests `test_<behavior>` and use pytest parametrization when validating policy or topology variants. Run the smallest relevant CPU test first, then the CUDA/Ray or multi-rank suite matching the affected path. There is no configured numeric coverage gate; meaningful edge-case and failure-path assertions are expected.

## Agent-Specific Repository Instructions

- Before internet access or any external HTTP(S) request, configure the repository proxy:

  ```bash
  export http_proxy=http://127.0.0.1:15409 https_proxy=http://127.0.0.1:15409 HTTP_PROXY=http://127.0.0.1:15409 HTTPS_PROXY=http://127.0.0.1:15409
  ```

- Ray operations are the exception. Before `ray start/status/stop`, Ray Client or GCS communication, or connecting NanoDeploy to a Ray cluster, remove all HTTP proxy variables:

  ```bash
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  ```

- Before launching any experiment or benchmark, set `SLIME_QP_NUM=4` in the
  driver environment so the value is propagated to newly created Ray actors:

  ```bash
  export SLIME_QP_NUM=4
  ```

- After changing C++ sources, reinstall with `pip install -v -e .` before running the full project.
- Modify NanoDeploy code only; do not patch external dependency libraries.
- Save useful interim reasoning or research notes under `docs-dev/` for later reference.
- Subagents may be used for suitable independent work.
- Commit changes promptly with clear, accurate messages so work remains traceable.
- Request elevated permissions before every GPU operation.
- When context exceeds 220,000 characters, and before context compaction, record the current time, objective, and completed work in `docs-dev/<date>/Progress.md`. Read that file before resuming after compaction to avoid duplicate work.

## Commit & Pull Request Guidelines

Recent history favors short imperative subjects, sometimes with `feat:`, `fix:`, or `test:` prefixes. Keep commits narrowly scoped (for example, `fix: stabilize fixed SP graph capture`). Pull requests should explain the problem and approach, list exact validation commands, link related issues or design notes, and state the GPU count/topology for distributed results. Include benchmark summaries for performance changes, but do not add bulky raw logs unless they are intentional review artifacts.
