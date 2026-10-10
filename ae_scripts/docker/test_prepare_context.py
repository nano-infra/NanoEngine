"CPU regression checks for preparing the published Docker build context."

from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import prepare_context


class PrepareContextTest(unittest.TestCase):
    def make_repo(self, path):
        path.mkdir()
        subprocess.run(["git", "init", "-q", str(path)], check=True)
        (path / "tracked.txt").write_text("committed source\n")
        subprocess.run(["git", "-C", str(path), "add", "tracked.txt"], check=True)
        subprocess.run([
            "git", "-C", str(path), "-c", "user.name=AE Test",
            "-c", "user.email=ae-test@example.invalid", "-c", "core.hooksPath=/dev/null",
            "commit", "-q", "-m", "fixture",
        ], check=True)
        return prepare_context.git(path, "rev-parse", "HEAD").decode().strip()

    def test_context_without_local_dlslime(self):
        with tempfile.TemporaryDirectory(prefix="ae-context-test-") as temp:
            root = Path(temp)
            nano, intra = root / "NanoDeploy", root / "nano_intra_alltoall"
            nano_commit = self.make_repo(nano)
            intra_commit = self.make_repo(intra)
            (intra / "tracked.txt").write_text("local build fix\n")
            (intra / "untracked.txt").write_text("exclude this\n")
            recipe = root / "recipe"
            recipe.mkdir()
            for name in prepare_context.BUILD_FILES:
                (recipe / name).write_text("")
            (recipe / "Dockerfile").write_text(
                f"ARG NANODEPLOY_COMMIT={nano_commit}\n"
                f"ARG NANO_INTRA_COMMIT={intra_commit}\n"
            )
            output = root / "context"
            argv = ["prepare_context.py", "--nanodeploy", str(nano),
                    "--nano-intra-alltoall", str(intra), "--output", str(output)]
            with patch.object(prepare_context, "RECIPE_DIR", recipe), \
                    patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                prepare_context.main()
            self.assertEqual(
                (output / "sources/NanoDeploy/SOURCE_COMMIT").read_text().strip(),
                nano_commit,
            )
            exported_intra = output / "sources/nano_intra_alltoall"
            self.assertEqual((exported_intra / "tracked.txt").read_text(), "local build fix\n")
            self.assertIn("+local build fix", (exported_intra / "SOURCE_WORKTREE.patch").read_text())
            self.assertFalse((exported_intra / "untracked.txt").exists())
            self.assertFalse((output / "sources/DLSlime").exists())
            with patch.object(prepare_context, "RECIPE_DIR", recipe), \
                    patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(SystemExit, "already exists"):
                    prepare_context.main()


if __name__ == "__main__":
    unittest.main()
