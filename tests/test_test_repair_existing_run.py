import tempfile
import unittest
import os
from pathlib import Path

from scripts.test_repair_existing_run import copy_project_output


class TestRepairExistingRunTest(unittest.TestCase):
    def test_copy_project_output_skips_unreadable_runtime_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            (source / "src").mkdir()
            (source / "src" / "lib.rs").write_text("pub fn ok() {}\n", encoding="utf-8")
            runtime_file = source / "testfile.mvp"
            runtime_file.write_text("runtime\n", encoding="utf-8")
            runtime_file.chmod(0o300)

            try:
                copy_project_output(source, destination)
            finally:
                runtime_file.chmod(0o600)

            self.assertTrue((destination / "src" / "lib.rs").exists())
            self.assertFalse((destination / "testfile.mvp").exists())

    @unittest.skipUnless(hasattr(os, "mkfifo"), "mkfifo is unavailable on this platform")
    def test_copy_project_output_skips_runtime_pipes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            (source / "src").mkdir()
            (source / "src" / "lib.rs").write_text("pub fn ok() {}\n", encoding="utf-8")
            os.mkfifo(source / ".feedback.1.pipe")

            copy_project_output(source, destination)

            self.assertTrue((destination / "src" / "lib.rs").exists())
            self.assertFalse((destination / ".feedback.1.pipe").exists())


if __name__ == "__main__":
    unittest.main()
