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


def export_worktree(source, destination, expected):
    destination.mkdir(parents=True)
    for raw in git(source, "ls-files", "-z").split(b"\0"):
        if not raw:
            continue
        relative = Path(raw.decode())
        original, target = source / relative, destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if original.is_symlink():
            target.symlink_to(original.readlink())
        else:
            shutil.copy2(original, target)
    (destination / "SOURCE_COMMIT").write_text(expected + "\n")
    (destination / "SOURCE_WORKTREE.patch").write_bytes(
        git(source, "diff", "--no-ext-diff", "HEAD")
    )


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
    parser.add_argument("--dlslime", type=Path, required=True)
    parser.add_argument("--nano-intra-alltoall", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"Build context already exists; choose a fresh directory: {output}")
    sources = (
        (args.dlslime.expanduser().resolve(), "DLSlime", revision("DLSLIME")),
        (args.nano_intra_alltoall.expanduser().resolve(), "nano_intra_alltoall",
         revision("NANO_INTRA")),
    )
    for source, name, expected in sources:
        actual = git(source, "rev-parse", "HEAD").decode().strip()
        if actual != expected:
            raise SystemExit(f"{name}: expected {expected}, found {actual}")
    nanodeploy = args.nanodeploy.expanduser().resolve()
    nano_revision = revision("NANODEPLOY")
    git(nanodeploy, "cat-file", "-e", nano_revision + "^{commit}")
    output.mkdir(parents=True)
    for name in BUILD_FILES:
        shutil.copy2(RECIPE_DIR / name, output / name)
    for source, name, expected in sources:
        export_worktree(source, output / "sources" / name, expected)
    export_commit(nanodeploy, output / "sources/NanoDeploy", nano_revision)
    print(f"Build context ready: {output}")


if __name__ == "__main__":
    main()
