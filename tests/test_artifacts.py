import tempfile
import unittest
from pathlib import Path

from c2rust_translator.artifacts import (
    default_full_dataset_log_path,
    default_project_root,
    default_run_root,
    legacy_project_root,
    legacy_run_root,
    project_root,
    resolve_full_dataset_log_path,
    resolve_project_root,
    resolve_run_root,
    run_logs_root,
    runs_root,
)


class ArtifactPathTest(unittest.TestCase):
    def test_default_run_root_uses_structured_runs_directory(self):
        root = Path("/repo")

        self.assertEqual(default_run_root(root, "run1"), Path("/repo/output/runs/run1"))

    def test_default_project_root_uses_structured_runs_directory(self):
        root = Path("/repo")

        self.assertEqual(
            default_project_root(root, "run1", "leftpad"),
            Path("/repo/output/runs/run1/leftpad"),
        )

    def test_project_root_appends_project_to_existing_run_root(self):
        run_root = Path("/repo/output/runs/run1")

        self.assertEqual(project_root(run_root, "leftpad"), Path("/repo/output/runs/run1/leftpad"))

    def test_runs_root_names_structured_runs_directory(self):
        root = Path("/repo")

        self.assertEqual(runs_root(root), Path("/repo/output/runs"))

    def test_resolve_run_root_prefers_new_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            default_run_root(root, "run1").mkdir(parents=True)
            legacy_run_root(root, "run1").mkdir(parents=True)

            self.assertEqual(resolve_run_root(root, "run1"), default_run_root(root, "run1"))

    def test_resolve_run_root_falls_back_to_legacy_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            legacy_run_root(root, "old-run").mkdir(parents=True)

            self.assertEqual(resolve_run_root(root, "old-run"), legacy_run_root(root, "old-run"))

    def test_resolve_project_root_falls_back_to_legacy_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            legacy_project_root(root, "old-run", "leftpad").mkdir(parents=True)

            self.assertEqual(
                resolve_project_root(root, "old-run", "leftpad"),
                legacy_project_root(root, "old-run", "leftpad"),
            )

    def test_default_full_dataset_log_path_uses_logs_runs(self):
        root = Path("/repo")

        self.assertEqual(default_full_dataset_log_path(root, "run1"), Path("/repo/logs/runs/run1.log"))

    def test_run_logs_root_names_structured_run_log_directory(self):
        root = Path("/repo")

        self.assertEqual(run_logs_root(root), Path("/repo/logs/runs"))

    def test_resolve_full_dataset_log_path_places_bare_names_under_logs_runs(self):
        root = Path("/repo")

        self.assertEqual(
            resolve_full_dataset_log_path(root, "run1", "custom.log"),
            Path("/repo/logs/runs/custom.log"),
        )

    def test_resolve_full_dataset_log_path_places_old_logs_root_names_under_logs_runs(self):
        root = Path("/repo")

        self.assertEqual(
            resolve_full_dataset_log_path(root, "run1", "logs/custom.log"),
            Path("/repo/logs/runs/custom.log"),
        )

    def test_resolve_full_dataset_log_path_preserves_other_explicit_relative_parent(self):
        root = Path("/repo")

        self.assertEqual(
            resolve_full_dataset_log_path(root, "run1", "logs/archive/custom.log"),
            Path("/repo/logs/archive/custom.log"),
        )
        self.assertEqual(
            resolve_full_dataset_log_path(root, "run1", "tmp/custom.log"),
            Path("/repo/tmp/custom.log"),
        )

    def test_resolve_full_dataset_log_path_preserves_absolute_path(self):
        root = Path("/repo")

        self.assertEqual(
            resolve_full_dataset_log_path(root, "run1", "/tmp/custom.log"),
            Path("/tmp/custom.log"),
        )

    def test_resolve_full_dataset_log_path_keeps_structured_logs_runs_path(self):
        root = Path("/repo")

        self.assertEqual(
            resolve_full_dataset_log_path(root, "run1", "logs/runs/custom.log"),
            Path("/repo/logs/runs/custom.log"),
        )


if __name__ == "__main__":
    unittest.main()
