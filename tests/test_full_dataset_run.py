import argparse
import tempfile
import unittest
from pathlib import Path

from c2rust_translator import translation_runner
from scripts import full_dataset_run


class FullDatasetProjectOrderingTest(unittest.TestCase):
    def test_parser_accepts_profile_experiment_profile_and_rounds(self):
        args = full_dataset_run.build_parser().parse_args(
            ["--experiment-profile", "no-compositional", "--adversarial-rounds", "3"]
        )

        self.assertEqual(args.experiment_profile, "no-compositional")
        self.assertEqual(args.adversarial_rounds, 3)

    def test_full_profile_parser_namespace_enables_live_adversarial_defaults_for_sequential_runs(self):
        args = full_dataset_run.build_parser().parse_args(["--experiment-profile", "full", "--jobs", "1"])

        options = translation_runner.adversarial_learning_option_metadata(args)

        self.assertTrue(options["admit_trace_candidates"])
        self.assertTrue(options["run_synthetic_challenges"])
        self.assertTrue(options["run_compositional_challenges"])
        self.assertTrue(options["admit_composition_failure_candidates"])
        self.assertTrue(options["pre_translation_challenges"])
        self.assertTrue(options["pre_translation_refinement"])
        self.assertEqual(options["pre_translation_refinement_rounds"], 5)
        self.assertTrue(options["pre_translation_refinement_writeback"])
        self.assertTrue(options["live_structured_agents"])
        self.assertTrue(options["live_semantic_oracle_validation"])
        self.assertTrue(options["live_activation_validation"])
        self.assertTrue(options["live_translator_replay"])

    def test_full_profile_parser_namespace_enables_writeback_for_parallel_project_local_bank(self):
        args = full_dataset_run.build_parser().parse_args(["--experiment-profile", "full", "--jobs", "15"])

        options = translation_runner.adversarial_learning_option_metadata(args)

        self.assertTrue(options["pre_translation_challenges"])
        self.assertTrue(options["pre_translation_refinement"])
        self.assertEqual(options["pre_translation_refinement_rounds"], 5)
        self.assertTrue(options["pre_translation_refinement_writeback"])

    def test_full_profile_parser_namespace_enables_writeback_for_parallel_shared_bank(self):
        args = full_dataset_run.build_parser().parse_args(
            [
                "--experiment-profile",
                "full",
                "--jobs",
                "15",
                "--rules-json-bank",
                "banks/shared.json",
            ]
        )

        options = translation_runner.adversarial_learning_option_metadata(args)

        self.assertTrue(options["pre_translation_challenges"])
        self.assertTrue(options["pre_translation_refinement"])
        self.assertEqual(options["pre_translation_refinement_rounds"], 5)
        self.assertTrue(options["pre_translation_refinement_writeback"])

    def test_parser_namespace_allows_pre_translation_refinement_round_override(self):
        args = full_dataset_run.build_parser().parse_args(
            [
                "--experiment-profile",
                "full",
                "--adversarial-pre-translation-refinement-rounds",
                "4",
            ]
        )

        options = translation_runner.adversarial_learning_option_metadata(args)

        self.assertEqual(options["pre_translation_refinement_rounds"], 4)

    def test_full_profile_parser_namespace_enables_online_updates_for_sequential_runs(self):
        args = full_dataset_run.build_parser().parse_args(["--experiment-profile", "full", "--jobs", "1"])

        self.assertTrue(translation_runner.online_adversarial_rules_enabled(args))

    def test_full_profile_parser_namespace_enables_online_updates_for_parallel_project_local_bank(self):
        args = full_dataset_run.build_parser().parse_args(["--experiment-profile", "full", "--jobs", "15"])

        self.assertTrue(translation_runner.online_adversarial_rules_enabled(args))

    def test_full_profile_parser_namespace_enables_online_updates_for_parallel_shared_bank(self):
        args = full_dataset_run.build_parser().parse_args(
            [
                "--experiment-profile",
                "full",
                "--jobs",
                "15",
                "--rules-json-bank",
                "banks/shared.json",
            ]
        )

        self.assertTrue(translation_runner.online_adversarial_rules_enabled(args))
        self.assertTrue(translation_runner.parallel_online_adversarial_updates_supported(args))

    def test_custom_parser_namespace_defaults_online_updates_for_parallel_shared_bank(self):
        args = full_dataset_run.build_parser().parse_args(
            [
                "--jobs",
                "15",
                "--rules-json-bank",
                "banks/shared.json",
            ]
        )

        self.assertTrue(translation_runner.online_adversarial_rules_enabled(args))
        self.assertTrue(translation_runner.parallel_online_adversarial_updates_supported(args))

    def test_parser_accepts_adversarial_rule_update_options(self):
        args = full_dataset_run.build_parser().parse_args(
            [
                "--adversarial-rule-bank",
                "banks/input.json",
                "--adversarial-rule-output-bank",
                "banks/output.json",
                "--adversarial-rule-report",
                "reports/learning.json",
                "--online-adversarial-rules",
                "--adversarial-admit-trace-candidates",
                "--adversarial-run-synthetic-challenges",
                "--adversarial-run-compositional-challenges",
                "--adversarial-admit-composition-failure-candidates",
                "--adversarial-live-structured-agents",
                "--adversarial-live-trace-synthesis",
                "--adversarial-live-trace-contradiction-review",
                "--adversarial-live-composition-relation-review",
                "--adversarial-live-semantic-oracle-validation",
                "--adversarial-live-activation-validation",
                "--adversarial-live-translator-replay",
                "--adversarial-pre-translation-challenges",
                "--adversarial-pre-translation-refinement",
                "--adversarial-pre-translation-refinement-writeback",
            ]
        )

        self.assertEqual(args.adversarial_rule_bank, "banks/input.json")
        self.assertEqual(args.adversarial_rule_output_bank, "banks/output.json")
        self.assertEqual(args.adversarial_rule_report, "reports/learning.json")
        self.assertTrue(args.online_adversarial_rules)
        self.assertTrue(args.adversarial_admit_trace_candidates)
        self.assertTrue(args.adversarial_run_synthetic_challenges)
        self.assertTrue(args.adversarial_run_compositional_challenges)
        self.assertTrue(args.adversarial_admit_composition_failure_candidates)
        self.assertTrue(args.adversarial_live_structured_agents)
        self.assertTrue(args.adversarial_live_trace_synthesis)
        self.assertTrue(args.adversarial_live_trace_contradiction_review)
        self.assertTrue(args.adversarial_live_composition_relation_review)
        self.assertTrue(args.adversarial_live_semantic_oracle_validation)
        self.assertTrue(args.adversarial_live_activation_validation)
        self.assertTrue(args.adversarial_live_translator_replay)
        self.assertTrue(args.adversarial_pre_translation_challenges)
        self.assertTrue(args.adversarial_pre_translation_refinement)
        self.assertTrue(args.adversarial_pre_translation_refinement_writeback)

    def test_parser_accepts_json_rule_bank_for_prompt_retrieval(self):
        args = full_dataset_run.build_parser().parse_args(
            [
                "--rules-json-bank",
                "banks/prompt_bank.json",
                "--rule-retrieval-backend",
                "auto",
                "--rule-retrieval-index-cache",
                "cache/prompt_bank.faiss",
                "--rule-interaction-scorer",
                "live",
            ]
        )

        self.assertEqual(args.rules_json_bank, "banks/prompt_bank.json")
        self.assertEqual(args.rule_retrieval_backend, "auto")
        self.assertEqual(args.rule_retrieval_index_cache, "cache/prompt_bank.faiss")
        self.assertEqual(args.rule_interaction_scorer, "live")

    def test_clear_runs_dir_preserves_legacy_output_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            legacy_project = root / "output" / "legacy-run" / "leftpad"
            current_project = root / "output" / "runs" / "new-run" / "leftpad"
            legacy_project.mkdir(parents=True)
            current_project.mkdir(parents=True)
            (legacy_project / "translation_report.json").write_text("{}", encoding="utf-8")
            (current_project / "translation_report.json").write_text("{}", encoding="utf-8")

            full_dataset_run.clear_runs_dir(root / "output" / "runs")

            self.assertTrue(legacy_project.exists())
            self.assertFalse(current_project.exists())

    def test_selected_projects_orders_non_reruns_by_descending_c_code_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metadata_dir = root / "data" / "c_metadata"
            metadata_dir.mkdir(parents=True)
            cbench_dir = root / "data" / "project" / "CBench"

            for project in ["m_small", "z_large", "a_medium", "b_same_size"]:
                (metadata_dir / f"{project}.json").write_text("{}", encoding="utf-8")
                (cbench_dir / project / "src").mkdir(parents=True)

            (cbench_dir / "z_large" / "src" / "lib.c").write_text("1\n2\n3\n", encoding="utf-8")
            (cbench_dir / "a_medium" / "src" / "lib.c").write_text("1\n2\n", encoding="utf-8")
            (cbench_dir / "m_small" / "src" / "lib.c").write_text("1\n", encoding="utf-8")
            (cbench_dir / "b_same_size" / "src" / "lib.h").write_text("1\n2\n", encoding="utf-8")
            (cbench_dir / "m_small" / "src" / "generated.i").write_text("ignored\n" * 100, encoding="utf-8")

            args = argparse.Namespace(resume_from_log=False, rerun_project=[])

            self.assertEqual(
                full_dataset_run.selected_projects(root, args, root / "run.log"),
                ["z_large", "a_medium", "b_same_size", "m_small"],
            )

    def test_selected_projects_keeps_reruns_first_then_orders_remaining_by_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metadata_dir = root / "data" / "c_metadata"
            metadata_dir.mkdir(parents=True)
            cbench_dir = root / "data" / "project" / "CBench"

            for project, lines in [("m_small", 1), ("z_large", 3), ("a_medium", 2)]:
                (metadata_dir / f"{project}.json").write_text("{}", encoding="utf-8")
                project_dir = cbench_dir / project / "src"
                project_dir.mkdir(parents=True)
                (project_dir / "lib.c").write_text("x\n" * lines, encoding="utf-8")

            args = argparse.Namespace(resume_from_log=False, rerun_project=["m_small"])

            self.assertEqual(
                full_dataset_run.selected_projects(root, args, root / "run.log"),
                ["m_small", "z_large", "a_medium"],
            )

    def test_selected_projects_excludes_difficult_projects_even_when_requested_for_rerun(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metadata_dir = root / "data" / "c_metadata"
            metadata_dir.mkdir(parents=True)
            cbench_dir = root / "data" / "project" / "CBench"

            for project, lines in [
                ("keep", 1),
                ("fslib", 100),
                ("libfor", 100),
                ("tisp", 100),
            ]:
                (metadata_dir / f"{project}.json").write_text("{}", encoding="utf-8")
                project_dir = cbench_dir / project / "src"
                project_dir.mkdir(parents=True)
                (project_dir / "lib.c").write_text("x\n" * lines, encoding="utf-8")

            args = argparse.Namespace(
                resume_from_log=False,
                rerun_project=["libfor", "keep", "tisp", "fslib"],
            )

            self.assertEqual(
                full_dataset_run.selected_projects(root, args, root / "run.log"),
                ["keep"],
            )

    def test_selected_projects_can_include_difficult_projects_when_explicitly_requested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metadata_dir = root / "data" / "c_metadata"
            metadata_dir.mkdir(parents=True)
            cbench_dir = root / "data" / "project" / "CBench"

            for project, lines in [
                ("keep", 1),
                ("fslib", 4),
                ("libfor", 3),
                ("tisp", 2),
            ]:
                (metadata_dir / f"{project}.json").write_text("{}", encoding="utf-8")
                project_dir = cbench_dir / project / "src"
                project_dir.mkdir(parents=True)
                (project_dir / "lib.c").write_text("x\n" * lines, encoding="utf-8")

            args = argparse.Namespace(
                resume_from_log=False,
                rerun_project=["libfor"],
                include_excluded=True,
            )

            self.assertEqual(
                full_dataset_run.selected_projects(root, args, root / "run.log"),
                ["libfor", "fslib", "tisp", "keep"],
            )


if __name__ == "__main__":
    unittest.main()
