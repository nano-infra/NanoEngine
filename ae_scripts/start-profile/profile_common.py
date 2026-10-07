#!/usr/bin/env python3
"""Small helpers shared by the profiler launchers."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
from typing import Iterable


def parse_named_inputs(values: Iterable[str]) -> list[tuple[str, Path]]:
    """Parse repeated NAME=PATH arguments while preserving their order."""
    inputs: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(
                f"input must use NAME=PATH syntax, got: {value!r}"
            )
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name or "/" in name:
            raise argparse.ArgumentTypeError(
                f"input name must be a non-empty path component, got: {name!r}"
            )
        if name in seen:
            raise argparse.ArgumentTypeError(f"duplicate input name: {name}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise argparse.ArgumentTypeError(f"input JSON does not exist: {path}")
        seen.add(name)
        inputs.append((name, path))
    return inputs


def shell_command(command: list[str], env_overrides: dict[str, str] | None = None) -> str:
    words = []
    if env_overrides:
        words.extend(f"{key}={shlex.quote(value)}" for key, value in env_overrides.items())
    words.extend(shlex.quote(value) for value in command)
    return " ".join(words)


def tee_process(
    command: list[str],
    log_path: Path,
    *,
    cwd: Path,
    env: dict[str, str],
) -> int:
    """Run a command and mirror combined output to the console and a log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
        return process.wait()
