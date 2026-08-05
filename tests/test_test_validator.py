import os
import tempfile
import time
import unittest
from pathlib import Path

from c2rust_translator.compilation_validator import CompilationValidator, DEFAULT_COMPILE_TIMEOUT_SECONDS
from c2rust_translator.test_validator import MAX_TEST_OUTPUT_STREAM_BYTES, TestValidator


class TestValidatorRuntimeArtifactTest(unittest.TestCase):
    def test_default_compile_timeout_covers_cold_dependency_builds(self):
        self.assertEqual(DEFAULT_COMPILE_TIMEOUT_SECONDS, 180)

    def test_restore_runtime_artifacts_removes_test_side_effects_without_touching_rust_tests(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            bin_dir = project / "src" / "bin"
            bin_dir.mkdir(parents=True)
            (bin_dir / "test.rs").write_text("fn main() {}\n", encoding="utf-8")
            (bin_dir / "fixture").write_text("expected\n", encoding="utf-8")

            validator = TestValidator(str(project))
            snapshot = validator.snapshot_runtime_artifacts()

            (project / "tmp").mkdir()
            (project / "tmp" / "symlink").symlink_to("./tmp/file")
            (project / ".vsync").mkdir()
            (project / ".vsync" / "index").write_text("generated\n", encoding="utf-8")
            (bin_dir / "result.txt").write_text("generated\n", encoding="utf-8")
            (bin_dir / "test.rs").write_text("fn main() { println!(\"patched\"); }\n", encoding="utf-8")

            validator.restore_runtime_artifacts(snapshot)

            self.assertFalse((project / "tmp").exists())
            self.assertFalse((project / ".vsync").exists())
            self.assertFalse((bin_dir / "result.txt").exists())
            self.assertEqual((bin_dir / "fixture").read_text(encoding="utf-8"), "expected\n")
            self.assertIn("patched", (bin_dir / "test.rs").read_text(encoding="utf-8"))

    def test_restore_runtime_artifacts_restores_original_symlink_fixture(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            bin_dir = project / "src" / "bin"
            bin_dir.mkdir(parents=True)
            (bin_dir / "target").write_text("fixture\n", encoding="utf-8")
            os.symlink("target", bin_dir / "expected.txt")

            validator = TestValidator(str(project))
            snapshot = validator.snapshot_runtime_artifacts()

            (bin_dir / "expected.txt").unlink()
            (bin_dir / "expected.txt").write_text("mutated\n", encoding="utf-8")

            validator.restore_runtime_artifacts(snapshot)

            self.assertTrue((bin_dir / "expected.txt").is_symlink())
            self.assertEqual(os.readlink(bin_dir / "expected.txt"), "target")

    def test_test_timeout_kills_spawned_cargo_children(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir) / "project"
            project.mkdir()
            fake_bin = Path(tmpdir) / "bin"
            fake_bin.mkdir()
            child_pid_file = Path(tmpdir) / "child.pid"
            cargo = fake_bin / "cargo"
            cargo.write_text(
                "#!/bin/sh\n"
                "(sleep 30) &\n"
                "echo $! > \"$CHILD_PID_FILE\"\n"
                "wait\n",
                encoding="utf-8",
            )
            cargo.chmod(0o755)

            old_path = os.environ.get("PATH", "")
            old_child_pid_file = os.environ.get("CHILD_PID_FILE")
            os.environ["PATH"] = f"{fake_bin}{os.pathsep}{old_path}"
            os.environ["CHILD_PID_FILE"] = str(child_pid_file)
            try:
                ok, output, _stats = TestValidator(str(project)).run_all_tests(timeout=1)
            finally:
                os.environ["PATH"] = old_path
                if old_child_pid_file is None:
                    os.environ.pop("CHILD_PID_FILE", None)
                else:
                    os.environ["CHILD_PID_FILE"] = old_child_pid_file

            self.assertFalse(ok)
            self.assertIn("Tests timeout after 1 seconds", output)
            self._assert_process_exits(int(child_pid_file.read_text(encoding="utf-8")))

    def test_test_output_limit_kills_noisy_cargo_process(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir) / "project"
            project.mkdir()
            fake_bin = Path(tmpdir) / "bin"
            fake_bin.mkdir()
            cargo = fake_bin / "cargo"
            cargo.write_text(
                "#!/bin/sh\n"
                "python - <<'PY'\n"
                "import sys\n"
                "chunk = 'x' * 1000000\n"
                "for _ in range(25):\n"
                "    print(chunk)\n"
                "    sys.stdout.flush()\n"
                "PY\n",
                encoding="utf-8",
            )
            cargo.chmod(0o755)

            old_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{fake_bin}{os.pathsep}{old_path}"
            try:
                ok, output, _stats = TestValidator(str(project)).run_all_tests(timeout=30)
            finally:
                os.environ["PATH"] = old_path

            self.assertFalse(ok)
            self.assertIn(f"Tests output exceeded {MAX_TEST_OUTPUT_STREAM_BYTES} bytes", output)
            self.assertIn("truncated", output)
            self.assertLess(len(output), 3_000_000)

    def test_compile_timeout_kills_spawned_cargo_children(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir) / "project"
            project.mkdir()
            fake_bin = Path(tmpdir) / "bin"
            fake_bin.mkdir()
            child_pid_file = Path(tmpdir) / "child.pid"
            cargo = fake_bin / "cargo"
            cargo.write_text(
                "#!/bin/sh\n"
                "(sleep 30) &\n"
                "echo $! > \"$CHILD_PID_FILE\"\n"
                "wait\n",
                encoding="utf-8",
            )
            cargo.chmod(0o755)

            old_path = os.environ.get("PATH", "")
            old_child_pid_file = os.environ.get("CHILD_PID_FILE")
            os.environ["PATH"] = f"{fake_bin}{os.pathsep}{old_path}"
            os.environ["CHILD_PID_FILE"] = str(child_pid_file)
            try:
                ok, output = CompilationValidator(str(project)).compile_project(timeout=1)
            finally:
                os.environ["PATH"] = old_path
                if old_child_pid_file is None:
                    os.environ.pop("CHILD_PID_FILE", None)
                else:
                    os.environ["CHILD_PID_FILE"] = old_child_pid_file

            self.assertFalse(ok)
            self.assertIn("Compilation timeout after 1 seconds", output)
            self._assert_process_exits(int(child_pid_file.read_text(encoding="utf-8")))

    def _assert_process_exits(self, pid: int) -> None:
        for _ in range(50):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.05)
        self.fail(f"process {pid} was still alive after timeout cleanup")


if __name__ == "__main__":
    unittest.main()
