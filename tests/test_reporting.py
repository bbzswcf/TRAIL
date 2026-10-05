import json
import tempfile
import unittest
from pathlib import Path

from c2rust_translator.orchestrator_reporting import generate_translation_report


class ReportingTest(unittest.TestCase):
    def test_translation_report_includes_alignment_statistics_and_tokens(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            report_path = generate_translation_report(
                project_name="amp",
                metadata_path="data/c_metadata/amp.json",
                output_dir=output_dir,
                max_fix_attempts=5,
                dep_analysis={"total_modules": 1},
                translated_modules=set(),
                translation_log=[],
                translation_history={},
                alignment_stats={
                    "alignment_llm_calls": 1,
                    "llm_bound_callables": 1,
                    "rule_fallback_callables": 0,
                    "empty_fallback_callables": 0,
                    "prompt_tokens": 3,
                    "completion_tokens": 4,
                    "total_tokens": 7,
                },
                success=True,
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(report["alignment_statistics"]["alignment_llm_calls"], 1)
        self.assertEqual(report["alignment_statistics"]["llm_bound_callables"], 1)
        self.assertEqual(report["llm_statistics"]["llm_calls"], 1)
        self.assertEqual(report["llm_statistics"]["prompt_tokens"], 3)
        self.assertEqual(report["llm_statistics"]["completion_tokens"], 4)
        self.assertEqual(report["llm_statistics"]["total_tokens"], 7)

    def test_translation_report_includes_profile_experiment_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            report_path = generate_translation_report(
                project_name="amp",
                metadata_path="data/c_metadata/amp.json",
                output_dir=output_dir,
                max_fix_attempts=5,
                dep_analysis={"total_modules": 1},
                translated_modules=set(),
                translation_log=[],
                translation_history={},
                success=True,
                experiment_profile="full",
                experiment_ablation_label="TRAIL",
                rule_stage="multi",
                rule_mining=True,
                single_rule_adversarial_learning=True,
                compositional_rule_adversarial_learning=True,
                adversarial_rounds=5,
                model_display_name="DeepSeek-V4-Flash",
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))

        config = report["config"]
        self.assertEqual(config["experiment_profile"], "full")
        self.assertEqual(config["experiment_ablation_label"], "TRAIL")
        self.assertEqual(config["rule_stage"], "multi")
        self.assertTrue(config["rule_mining"])
        self.assertTrue(config["single_rule_adversarial_learning"])
        self.assertTrue(config["compositional_rule_adversarial_learning"])
        self.assertEqual(config["adversarial_rounds"], 5)
        self.assertEqual(config["model_display_name"], "DeepSeek-V4-Flash")

    def test_translation_report_includes_adversarial_rule_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            report_path = generate_translation_report(
                project_name="amp",
                metadata_path="data/c_metadata/amp.json",
                output_dir=output_dir,
                max_fix_attempts=5,
                dep_analysis={"total_modules": 1},
                translated_modules=set(),
                translation_log=[],
                translation_history={},
                success=True,
                online_adversarial_rules=True,
                adversarial_rule_bank_path="/repo/banks/input.json",
                adversarial_rule_output_bank_path="/repo/output/runs/run/adversarial_rules.updated.json",
                adversarial_rule_report_path="/repo/output/runs/run/adversarial_rule_report.json",
                adversarial_learning_options={
                    "admit_trace_candidates": True,
                    "run_synthetic_challenges": True,
                    "run_compositional_challenges": True,
                    "admit_composition_failure_candidates": True,
                    "live_structured_agents": True,
                    "live_trace_synthesis": True,
                    "live_trace_contradiction_review": True,
                    "live_composition_relation_review": True,
                    "live_semantic_oracle_validation": True,
                    "live_activation_validation": True,
                    "live_translator_replay": True,
                    "pre_translation_challenges": True,
                    "pre_translation_refinement": True,
                    "pre_translation_refinement_rounds": 3,
                    "pre_translation_refinement_writeback": True,
                },
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))

        config = report["config"]
        self.assertTrue(config["online_adversarial_rules"])
        self.assertEqual(config["adversarial_rule_bank_path"], "/repo/banks/input.json")
        self.assertEqual(
            config["adversarial_rule_output_bank_path"],
            "/repo/output/runs/run/adversarial_rules.updated.json",
        )
        self.assertEqual(
            config["adversarial_rule_report_path"],
            "/repo/output/runs/run/adversarial_rule_report.json",
        )
        self.assertEqual(
            config["adversarial_learning_options"],
            {
                "admit_trace_candidates": True,
                "run_synthetic_challenges": True,
                "run_compositional_challenges": True,
                "admit_composition_failure_candidates": True,
                "live_structured_agents": True,
                "live_trace_synthesis": True,
                "live_trace_contradiction_review": True,
                "live_composition_relation_review": True,
                "live_semantic_oracle_validation": True,
                "live_activation_validation": True,
                "live_translator_replay": True,
                "pre_translation_challenges": True,
                "pre_translation_refinement": True,
                "pre_translation_refinement_rounds": 3,
                "pre_translation_refinement_writeback": True,
            },
        )

    def test_translation_report_summarizes_pre_translation_refinement_writebacks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            report_path = generate_translation_report(
                project_name="amp",
                metadata_path="data/c_metadata/amp.json",
                output_dir=output_dir,
                max_fix_attempts=5,
                dep_analysis={"total_modules": 1},
                translated_modules=set(),
                translation_log=[],
                translation_history={
                    ("mod", "fn", "parse"): {
                        "selected_rules": [
                            {
                                "id": "capacity-contract",
                                "pre_translation_refinement": {
                                    "accepted": True,
                                    "field": "risk",
                                    "writeback": "updated",
                                    "writeback_path": "/repo/output/runs/run/rules.active.json",
                                },
                            },
                            {
                                "id": "nul-contract",
                                "pre_translation_refinement": {
                                    "accepted": False,
                                    "field": "constraint",
                                    "reason": "no improvement",
                                },
                            },
                        ],
                        "compilation_attempts": [
                            {
                                "selected_rules": [
                                    {
                                        "id": "parser-state",
                                        "pre_translation_refinement": {
                                            "accepted": True,
                                            "field": "trigger",
                                            "writeback": "error",
                                            "writeback_error": "permission denied",
                                        },
                                    }
                                ]
                            }
                        ],
                    }
                },
                success=True,
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(
            report["adversarial_refinement_statistics"],
            {
                "pre_translation_refinements": 3,
                "accepted_pre_translation_refinements": 2,
                "pre_translation_refinement_writebacks": 1,
                "pre_translation_refinement_writeback_errors": 1,
                "pre_translation_refinement_writeback_paths": [
                    "/repo/output/runs/run/rules.active.json"
                ],
            },
        )

    def test_translation_report_includes_json_rule_bank_retrieval_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            report_path = generate_translation_report(
                project_name="amp",
                metadata_path="data/c_metadata/amp.json",
                output_dir=output_dir,
                max_fix_attempts=5,
                dep_analysis={"total_modules": 1},
                translated_modules=set(),
                translation_log=[],
                translation_history={},
                success=True,
                rules_json_bank_path="/repo/banks/prompt_bank.json",
                rule_retrieval_backend="keyword",
                rule_retrieval_status="keyword backend requested",
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))

        config = report["config"]
        self.assertEqual(config["rules_json_bank_path"], "/repo/banks/prompt_bank.json")
        self.assertEqual(config["rule_retrieval_backend"], "keyword")
        self.assertEqual(config["rule_retrieval_status"], "keyword backend requested")


if __name__ == "__main__":
    unittest.main()
