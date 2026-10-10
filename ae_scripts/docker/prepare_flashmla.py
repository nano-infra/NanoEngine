"""Prepare official FlashMLA archives for a build without Git metadata.

Only adjust setup metadata: the already-extracted CUTLASS replaces the
submodule update, and the supplied source revision replaces `git rev-parse`.
The kernel code and compiler options are unchanged.
"""

import pathlib
import re
import sys


def main():
    root = pathlib.Path(sys.argv[1])
    revision = sys.argv[2]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SystemExit("Expected a full FlashMLA commit ID")
    if not (root / "csrc/cutlass/include/cutlass/cutlass.h").is_file():
        raise SystemExit("The pinned CUTLASS source archive has not been extracted")

    setup = root / "setup.py"
    text = setup.read_text()
    replacements = {
        'subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"])':
            "# CUTLASS was extracted from its pinned official source archive.",
        "rev = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()":
            f'rev = "+{revision[:7]}"',
    }
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise SystemExit(f"Unexpected FlashMLA setup.py: {old}")
        text = text.replace(old, new, 1)
    setup.write_text(text)
    (root / "SOURCE_COMMIT").write_text(revision + "\n")


if __name__ == "__main__":
    main()
