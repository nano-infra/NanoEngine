# Releasing DLEngine

DLEngine uses one repository-wide version and one annotated tag named `v<version>`. The authoritative Python version is in `pyproject.toml`; the release script aligns the Rust package, lockfile, and `dlengine.vl` version to it.

## Release the current version

From a clean, up-to-date `Pure_dp` checkout:

```bash
./scripts/release.sh
```

For non-interactive confirmation:

```bash
./scripts/release.sh --yes
```

The first release therefore tags the current `0.2.0` version as `v0.2.0`.

## Bump and release

Pass the next version when a bump is required:

```bash
./scripts/release.sh 0.2.1
```

The script synchronizes:

- `pyproject.toml`
- `Cargo.toml`
- the `dlengine-rust` entry in `Cargo.lock`
- `dlengine/vl/__init__.py`

It then commits the version change as `release: v0.2.1`, creates an annotated tag, and atomically pushes the `Pure_dp` branch and tag.

## Pre-commit behavior

The script runs pre-commit against the four release-managed version files before changing versions. If a hook changes a managed file or a check fails, the release stops before creating a commit or tag. Review and commit the hook changes, then rerun the command.

After a version bump, release-managed files are checked again. The normal `git commit` hooks still run; if they reject or modify the commit, the script stops before tagging.

## Safety options

```bash
./scripts/release.sh --dry-run
./scripts/release.sh 0.2.1 --no-push
```

`--dry-run` performs clean-tree, branch, remote-sync, tag-collision, and pre-commit validation without modifying git state. `--no-push` creates the release commit and tag locally for inspection.

Tags matching `v*` also trigger the versioned documentation workflow. Release only from `Pure_dp`; the script rejects feature branches and a local branch that is not exactly synchronized with `origin/Pure_dp`.
