import tempfile
import unittest
from pathlib import Path

from scripts import experiment_metrics
from scripts.experiment_metrics import metadata_projects, reported_projects


class ExperimentMetricsTest(unittest.TestCase):
    def test_metadata_projects_ignores_dep_graph_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metadata_dir = root / "data" / "c_metadata"
            metadata_dir.mkdir(parents=True)
            (metadata_dir / "Megalania.json").write_text("{}", encoding="utf-8")
            (metadata_dir / "Megalania_dep_graph.json").write_text("{}", encoding="utf-8")
            (metadata_dir / "libfor.json").write_text("{}", encoding="utf-8")

            projects = metadata_projects(root, include_excluded=False)

        self.assertEqual(projects, {"Megalania"})

    def test_reported_projects_reads_new_output_runs_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            project_dir = root / "output" / "runs" / "run1" / "leftpad"
            project_dir.mkdir(parents=True)
            (project_dir / "translation_report.json").write_text("{}", encoding="utf-8")

            projects = reported_projects(root, "run1", include_excluded=False)

        self.assertEqual(projects, {"leftpad"})

    def test_reported_projects_falls_back_to_legacy_output_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            project_dir = root / "output" / "legacy-run" / "leftpad"
            project_dir.mkdir(parents=True)
            (project_dir / "translation_report.json").write_text("{}", encoding="utf-8")

            projects = reported_projects(root, "legacy-run", include_excluded=False)

        self.assertEqual(projects, {"leftpad"})

    def test_parser_accepts_quality_table_mode(self):
        args = experiment_metrics.build_parser().parse_args(["run1", "--quality-table"])

        self.assertTrue(args.quality_table)

    def test_callable_safety_metrics_reads_translation_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            (project_root / "translation_history.json").write_text(
                """
{
  "src::function::safe": {
    "final_code": "pub fn safe() -> i32 { 1 }"
  },
  "src::function::unsafe_block": {
    "final_code": "pub fn unsafe_block() { unsafe { dangerous(); } }"
  },
  "src::function::unsafe_extern_only": {
    "final_code": "unsafe extern \\"C\\" { fn c_func(); }\\npub fn wrapper() {}"
  }
}
""",
                encoding="utf-8",
            )

            safe, total = experiment_metrics.callable_safety_metrics(project_root)

        self.assertEqual((safe, total), (2, 3))

    def test_project_quality_metrics_reads_report_history_and_quality_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            project_root = root / "output" / "runs" / "run1" / "leftpad"
            project_root.mkdir(parents=True)
            (project_root / "translation_report.json").write_text(
                """
{
  "success": true,
  "config": {
    "experiment_profile": "full",
    "experiment_ablation_label": "TRAIL",
    "model_display_name": "DeepSeek-V4-Flash"
  },
  "statistics": {
    "total_modules": 2,
    "translated_modules": 2,
    "failed_modules": 0
  }
}
""",
                encoding="utf-8",
            )
            (project_root / "translation_history.json").write_text(
                """
{
  "a": {"final_code": "pub fn a() {}"},
  "b": {"final_code": "pub fn b() { unsafe { b(); } }"}
}
""",
                encoding="utf-8",
            )
            (project_root / "quality_metrics.json").write_text(
                '{"lint_pass": true, "idiom_penalty": 4.25}',
                encoding="utf-8",
            )

            metrics = experiment_metrics.project_quality_metrics(root, "run1", "leftpad")

        self.assertTrue(metrics["compile_pass"])
        self.assertTrue(metrics["test_pass"])
        self.assertEqual(metrics["profile"], "TRAIL")
        self.assertEqual(metrics["model"], "DeepSeek-V4-Flash")
        self.assertEqual(metrics["safe_callables"], 1)
        self.assertEqual(metrics["total_callables"], 2)
        self.assertTrue(metrics["lint_pass"])
        self.assertEqual(metrics["idiom_penalty"], 4.25)


if __name__ == "__main__":
    unittest.main()
