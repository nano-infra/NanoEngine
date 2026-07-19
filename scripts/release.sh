#!/usr/bin/env bash
#
# One-command DLEngine release: validate, align versions, run pre-commit,
# create an annotated v<version> tag, and atomically push branch + tag.
#
# Usage:
#   scripts/release.sh                  # release current pyproject version
#   scripts/release.sh 0.2.1            # bump all DLEngine versions and release
#   scripts/release.sh 0.2.1 --yes      # non-interactive confirmation
#   scripts/release.sh --dry-run        # validate without changing git state
#   scripts/release.sh --no-push        # create commit/tag locally only

set -Eeuo pipefail

DRY_RUN=0
PUSH=1
ASSUME_YES=0
REQUESTED_VERSION=""
REMOTE="${RELEASE_REMOTE:-origin}"
RELEASE_BRANCH="${RELEASE_BRANCH:-Pure_dp}"

usage() {
  sed -n '2,12p' "$0"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --no-push) PUSH=0 ;;
    --yes|-y) ASSUME_YES=1 ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -n "$REQUESTED_VERSION" ]]; then
        die "unexpected extra argument: $arg"
      fi
      REQUESTED_VERSION="$arg"
      ;;
  esac
done

for command in git python3 pre-commit; do
  command -v "$command" >/dev/null 2>&1 || die "required command not found: $command"
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

CURRENT_VERSION="$(
  python3 - <<'PY'
import pathlib
import re

text = pathlib.Path("pyproject.toml").read_text()
match = re.search(r'(?m)^version = "([^"]+)"$', text)
if not match:
    raise SystemExit("could not read [project] version from pyproject.toml")
print(match.group(1))
PY
)"

NEW_VERSION="${REQUESTED_VERSION:-$CURRENT_VERSION}"
if ! [[ "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-](rc|post)[0-9]+)?$ ]]; then
  die "'$NEW_VERSION' is not a supported version; expected MAJOR.MINOR.PATCH"
fi

TAG="v$NEW_VERSION"
VERSION_FILES=(
  pyproject.toml
  Cargo.toml
  Cargo.lock
  dlengine/vl/__init__.py
)

[[ -z "$(git status --porcelain=v1)" ]] ||
  die "working tree is not clean; commit or stash tracked and untracked changes first"

CURRENT_BRANCH="$(git branch --show-current)"
[[ "$CURRENT_BRANCH" == "$RELEASE_BRANCH" ]] ||
  die "release must run on '$RELEASE_BRANCH' (current: '$CURRENT_BRANCH')"

echo "Fetching $REMOTE/$RELEASE_BRANCH and tags..."
git fetch "$REMOTE" "$RELEASE_BRANCH" --tags

LOCAL_HEAD="$(git rev-parse HEAD)"
REMOTE_HEAD="$(git rev-parse "$REMOTE/$RELEASE_BRANCH")"
[[ "$LOCAL_HEAD" == "$REMOTE_HEAD" ]] ||
  die "local $RELEASE_BRANCH is not exactly $REMOTE/$RELEASE_BRANCH; update it first"

if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
  die "local tag $TAG already exists"
fi
if git ls-remote --exit-code --tags "$REMOTE" "refs/tags/$TAG" >/dev/null 2>&1; then
  die "remote tag $TAG already exists"
fi

echo "Running pre-commit on release-managed files before release..."
set +e
pre-commit run --files "${VERSION_FILES[@]}"
PRECOMMIT_RC=$?
set -e
if [[ "$PRECOMMIT_RC" -ne 0 ]]; then
  if [[ -n "$(git status --porcelain=v1)" ]]; then
    echo "ERROR: pre-commit modified files. No release commit or tag was created." >&2
    echo "Review and commit the hook fixes, then rerun this command." >&2
  else
    echo "ERROR: pre-commit checks failed. No release commit or tag was created." >&2
  fi
  exit "$PRECOMMIT_RC"
fi
[[ -z "$(git status --porcelain=v1)" ]] ||
  die "pre-commit left changes despite returning success; review them before release"

echo "DLEngine release: $CURRENT_VERSION -> $NEW_VERSION ($TAG)"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] Validation passed; no version files, commits, tags, or remotes changed."
  exit 0
fi

python3 - "$NEW_VERSION" <<'PY'
from pathlib import Path
import re
import sys

version = sys.argv[1]

def replace_once(path: str, pattern: str, replacement: str) -> None:
    file = Path(path)
    text = file.read_text()
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"expected exactly one version field in {path}, found {count}")
    file.write_text(updated)

replace_once("pyproject.toml", r'^version = "[^"]+"$', f'version = "{version}"')
replace_once("Cargo.toml", r'^version = "[^"]+"$', f'version = "{version}"')
replace_once(
    "Cargo.lock",
    r'(\[\[package\]\]\nname = "dlengine-rust"\nversion = ")[^"]+(")',
    rf'\g<1>{version}\2',
)
replace_once(
    "dlengine/vl/__init__.py",
    r'^__version__ = "[^"]+"$',
    f'__version__ = "{version}"',
)
PY

mapfile -t CHANGED_FILES < <(git diff --name-only)
for changed in "${CHANGED_FILES[@]}"; do
  allowed=0
  for managed in "${VERSION_FILES[@]}"; do
    [[ "$changed" == "$managed" ]] && allowed=1
  done
  [[ "$allowed" -eq 1 ]] || die "unexpected release change: $changed"
done

if [[ "${#CHANGED_FILES[@]}" -gt 0 ]]; then
  echo "Running pre-commit on release-managed version files..."
  set +e
  pre-commit run --files "${VERSION_FILES[@]}"
  VERSION_HOOK_RC=$?
  set -e

  if [[ "$VERSION_HOOK_RC" -ne 0 ]]; then
    echo "Hooks updated or rejected release-managed files; rerunning once..."
    pre-commit run --files "${VERSION_FILES[@]}" ||
      die "pre-commit still fails on release-managed files"
  fi
fi

git diff --check

echo
echo "Release commit:"
git --no-pager diff --stat
git --no-pager diff
echo "Tag: $TAG"
echo "Target: $REMOTE/$RELEASE_BRANCH"

if [[ "$ASSUME_YES" -ne 1 ]]; then
  read -r -p "Create and publish $TAG? [y/N] " answer
  case "$answer" in
    y|Y|yes|YES) ;;
    *) echo "Aborted; version changes remain in the working tree."; exit 1 ;;
  esac
fi

if [[ -n "$(git diff --name-only)" ]]; then
  git add "${VERSION_FILES[@]}"
  if ! git commit -m "release: $TAG"; then
    echo "ERROR: release commit failed, possibly because a commit hook changed files." >&2
    echo "No tag was created. Review the working tree and rerun the release." >&2
    exit 1
  fi
fi

[[ -z "$(git status --porcelain=v1)" ]] ||
  die "working tree is not clean after the release commit; refusing to tag"

git tag -a "$TAG" -m "DLEngine $TAG"

if [[ "$PUSH" -eq 1 ]]; then
  if ! git push --atomic "$REMOTE" \
    "HEAD:refs/heads/$RELEASE_BRANCH" \
    "refs/tags/$TAG"; then
    echo "ERROR: atomic push failed. Nothing was partially published remotely." >&2
    echo "The local tag remains; fix the cause and rerun the push or remove it manually." >&2
    exit 1
  fi
  echo "Released $TAG from $REMOTE/$RELEASE_BRANCH."
else
  echo "Created local $TAG. Publish it atomically with:"
  echo "  git push --atomic $REMOTE HEAD:refs/heads/$RELEASE_BRANCH refs/tags/$TAG"
fi
