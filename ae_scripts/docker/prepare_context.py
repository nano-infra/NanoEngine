#!/usr/bin/env python3
"Prepare an AE Docker build context without an existing container."

import argparse
from pathlib import Path
import re
import shutil
import subprocess


RECIPE_DIR = Path(__file__).resolve().parent
BUILD_FILES = (
    "Dockerfile", ".dockerignore", "constraints-base.txt", "requirements-extra.txt",
    "export_vllm_precompiled.py", "prepare_flashmla.py", "verify_environment.py",
)


def git(source, *args):
    return subprocess.check_output([
        "git", "--no-optional-locks", "-C", str(source), *args,
    ])


def revision(name):
    text = (RECIPE_DIR / "Dockerfile").read_text()
    match = re.search(rf"^ARG {name}_COMMIT=([0-9a-f]{{40}})$", text, re.MULTILINE)
    if not match:
        raise SystemExit(f"Missing {name}_COMMIT in Dockerfile")
    return match.group(1)


def export_commit(source, destination, expected):
    destination.mkdir(parents=True)
    with subprocess.Popen(
        ["git", "-C", str(source), "archive", expected], stdout=subprocess.PIPE,
    ) as archive:
        result = subprocess.run(
            ["tar", "-xf", "-", "-C", str(destination)], stdin=archive.stdout,
        )
        archive.stdout.close()
        archive_status = archive.wait()
    if result.returncode or archive_status:
        raise SystemExit(f"Failed to export {source} at {expected}")
    (destination / "SOURCE_COMMIT").write_text(expected + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nanodeploy", type=Path, default=RECIPE_DIR.parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"Build context already exists; choose a fresh directory: {output}")
    nanodeploy = args.nanodeploy.expanduser().resolve()
    nano_revision = revision("NANODEPLOY")
    git(nanodeploy, "cat-file", "-e", nano_revision + "^{commit}")
    output.mkdir(parents=True)
    for name in BUILD_FILES:
        shutil.copy2(RECIPE_DIR / name, output / name)
    export_commit(nanodeploy, output / "sources/NanoDeploy", nano_revision)
    print(f"Build context ready: {output}")


if __name__ == "__main__":
    main()
