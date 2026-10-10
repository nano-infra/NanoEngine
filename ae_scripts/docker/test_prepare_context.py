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

    def test_context_with_only_nanodeploy_checkout(self):
        with tempfile.TemporaryDirectory(prefix="ae-context-test-") as temp:
            root = Path(temp)
            nano = root / "NanoDeploy"
            nano_commit = self.make_repo(nano)
            (nano / "tracked.txt").write_text("uncommitted change\n")
            (nano / "untracked.txt").write_text("exclude this\n")
            recipe = root / "recipe"
            recipe.mkdir()
            for name in prepare_context.BUILD_FILES:
                (recipe / name).write_text("")
            (recipe / "Dockerfile").write_text(f"ARG NANODEPLOY_COMMIT={nano_commit}\n")
            output = root / "context"
            argv = ["prepare_context.py", "--nanodeploy", str(nano),
                    "--output", str(output)]
            with patch.object(prepare_context, "RECIPE_DIR", recipe), \
                    patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                prepare_context.main()
            exported = output / "sources/NanoDeploy"
            self.assertEqual((exported / "SOURCE_COMMIT").read_text().strip(), nano_commit)
            self.assertEqual((exported / "tracked.txt").read_text(), "committed source\n")
            self.assertFalse((exported / "untracked.txt").exists())
            self.assertEqual([p.name for p in (output / "sources").iterdir()], ["NanoDeploy"])
            with patch.object(prepare_context, "RECIPE_DIR", recipe), \
                    patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(SystemExit, "already exists"):
                    prepare_context.main()


if __name__ == "__main__":
    unittest.main()
