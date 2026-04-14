#!/usr/bin/env sh
# 清理所有编译产物和缓存（适用于 POSIX shell，例如 Git Bash / WSL）
set -eu

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ] || [ "${1:-}" = "-n" ]; then
  DRY_RUN=1
fi

run() {
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY-RUN] $*"
  else
    echo "[RUN] $*"
    eval "$*"
  fi
}

echo "Cleaning build artifacts and caches (dry-run=$DRY_RUN)"

# Python artifacts
run "rm -rf build dist *.egg-info nanodeploy.egg-info"
run "rm -rf .pytest_cache .mypy_cache venv .venv"
run "find . -type d -name '__pycache__' -exec rm -rf {} +"
run "find . -type f -name '*.pyc' -delete"
run "find . -type f -name '*.pyo' -delete"
run "find . -type d -name '*.egg-info' -exec rm -rf {} +"

# C/C++/CMake artifacts
run "rm -rf csrc/build"
run "find . -type d -name 'CMakeFiles' -exec rm -rf {} +"
run "rm -f CMakeCache.txt Makefile cmake_install.cmake"
run "find . -path './csrc/*' -type f \( -name '*.o' -o -name '*.obj' -o -name '*.so' -o -name '*.pyd' -o -name '*.dll' -o -name '*.exe' -o -name '*.lib' \) -delete"

# Remove compiled extensions inside package (Unix/Windows)
run "find nanodeploy -type f \( -name '_nanodeploy_cpp.*' -o -name '*.so' -o -name '*.pyd' \) -delete || true"

echo "Cleaning complete."
