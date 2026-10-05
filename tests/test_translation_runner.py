import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from c2rust_translator import translation_runner


class EnvFileTest(unittest.TestCase):
    def test_load_env_file_sets_missing_values_and_preserves_existing_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENAI_API_KEY=from-file",
                        'LLM_SESSION_ID="session from file"',
                        "LLM_LOGID=log-from-file",
                    ]
                ),
                encoding="utf-8",
            )

            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "from-env"}, clear=False):
                translation_runner.load_env_file(env_path)

                self.assertEqual(os.environ["OPENAI_API_KEY"], "from-env")
                self.assertEqual(os.environ["LLM_SESSION_ID"], "session from file")
                self.assertEqual(os.environ["LLM_LOGID"], "log-from-file")


class ProjectParallelismTest(unittest.TestCase):
    def test_run_project_batch_runs_multiple_projects_concurrently_when_jobs_is_greater_than_one(self):
        active = 0
        max_active = 0
        lock = threading.Lock()

        class FakeTranslator:
            def __init__(self, config):
                self.config = config

            def run(self):
                nonlocal active, max_active
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                time.sleep(0.05)
                with lock:
                    active -= 1

        args = argparse.Namespace(
            model="gpt-5.4-mini-2026-03-17",
            api_key="secret",
            api_url="https://example.test/responses",
            base_url="https://unused.test",
            llm_protocol="openai-responses",
            session_id="session-123",
            logid="log-456",
            max_fix_attempts=5,
            max_test_fix_attempts=3,
            continue_on_failure=False,
            jobs=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_root = root / "output" / "runs" / "run"
            translation_runner.run_project_batch(
                ["leftpad", "morton"],
                root_dir=root,
                run_root=run_root,
                args=args,
                translator_cls=FakeTranslator,
            )

        self.assertGreaterEqual(max_active, 2)

    def test_parallel_full_profile_uses_project_local_online_banks_without_rejecting_batch(self):
        class FakeTranslator:
            configs = []

            def __init__(self, config):
                self.config = config
                self.configs.append(config)

            def run(self):
                Path(self.config["output_dir"]).mkdir(parents=True, exist_ok=True)

        args = argparse.Namespace(
            model="gpt-5.4-mini-2026-03-17",
            api_key="secret",
            api_url="https://example.test/responses",
            base_url="https://unused.test",
            llm_protocol="openai-responses",
            session_id="session-123",
            logid="log-456",
            max_fix_attempts=5,
            max_test_fix_attempts=3,
            continue_on_failure=False,
            jobs=2,
            alignment_mode="llm",
            alignment_confidence_threshold=0.65,
            rules_mode="default",
            rule_limit=5,
            rules_path="",
            rules_json_bank="",
            rule_retrieval_backend="keyword",
            rule_retrieval_index_cache="",
            rule_interaction_scorer="heuristic",
            experiment_profile="full",
            adversarial_rounds=None,
            adversarial_rule_bank="",
            adversarial_rule_output_bank="",
            adversarial_rule_report="",
            online_adversarial_rules=False,
            adversarial_admit_trace_candidates=False,
            adversarial_run_synthetic_challenges=False,
            adversarial_run_compositional_challenges=False,
            adversarial_admit_composition_failure_candidates=False,
            adversarial_live_structured_agents=False,
            adversarial_live_trace_synthesis=False,
            adversarial_live_trace_contradiction_review=False,
            adversarial_live_composition_relation_review=False,
            adversarial_live_semantic_oracle_validation=False,
            adversarial_live_activation_validation=False,
            adversarial_live_translator_replay=False,
            adversarial_pre_translation_challenges=False,
            adversarial_pre_translation_refinement=False,
            adversarial_pre_translation_refinement_rounds=None,
            adversarial_pre_translation_refinement_writeback=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path.cwd()
            run_root = Path(tmpdir) / "output" / "runs" / "run"
            translation_runner.run_project_batch(
                ["leftpad", "morton"],
                root_dir=root,
                run_root=run_root,
                args=args,
                translator_cls=FakeTranslator,
            )

        self.assertEqual(len(FakeTranslator.configs), 2)
        bank_paths = {config["rules_json_bank_path"] for config in FakeTranslator.configs}
        self.assertEqual(
            bank_paths,
            {
                str(run_root / "leftpad" / "rules.active.json"),
                str(run_root / "morton" / "rules.active.json"),
            },
        )
        self.assertTrue(all(config["online_adversarial_rules"] for config in FakeTranslator.configs))


class AdversarialRuleRunnerIntegrationTest(unittest.TestCase):
    def _all_adversarial_options_args(self, *, experiment_profile=""):
        return argparse.Namespace(
            experiment_profile=experiment_profile,
            adversarial_admit_trace_candidates=True,
            adversarial_run_synthetic_challenges=True,
            adversarial_run_compositional_challenges=True,
            adversarial_admit_composition_failure_candidates=True,
            adversarial_live_structured_agents=True,
            adversarial_live_trace_synthesis=True,
            adversarial_live_trace_contradiction_review=True,
            adversarial_live_composition_relation_review=True,
            adversarial_live_semantic_oracle_validation=True,
            adversarial_live_activation_validation=True,
            adversarial_live_translator_replay=True,
            adversarial_pre_translation_challenges=True,
            adversarial_pre_translation_refinement=True,
            adversarial_pre_translation_refinement_rounds=None,
            adversarial_pre_translation_refinement_writeback=True,
            api_key="secret",
            api_url="https://example.test/responses",
            base_url="https://unused.test",
            model="model-x",
            llm_protocol="openai-responses",
            min_request_interval=0.0,
            rate_limit_retry_delay=60.0,
            llm_timeout=300,
            session_id="session-123",
            logid="log-456",
        )

    def test_parser_accepts_json_rule_bank_for_prompt_retrieval(self):
        parser = translation_runner.build_arg_parser()

        args = parser.parse_args(
            [
                "leftpad",
                "--rules-json-bank",
                "banks/prompt_bank.json",
                "--rule-retrieval-backend",
                "embedding",
                "--rule-retrieval-index-cache",
                "cache/prompt_bank.faiss",
                "--rule-interaction-scorer",
                "live",
            ]
        )

        self.assertEqual(args.rules_json_bank, "banks/prompt_bank.json")
        self.assertEqual(args.rule_retrieval_backend, "embedding")
        self.assertEqual(args.rule_retrieval_index_cache, "cache/prompt_bank.faiss")
        self.assertEqual(args.rule_interaction_scorer, "live")

    def test_parser_accepts_adversarial_rule_update_options(self):
        parser = translation_runner.build_arg_parser()

        args = parser.parse_args(
            [
                "leftpad",
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

    def test_online_adversarial_update_defaults_to_active_json_bank(self):
        args = argparse.Namespace(
            rules_json_bank="banks/prompt_bank.json",
            adversarial_rule_bank="",
            adversarial_rule_output_bank="",
            adversarial_rule_report="",
            online_adversarial_rules=True,
        )

        bank_path, output_bank_path, report_path = translation_runner.adversarial_rule_update_paths(
            Path("/repo"),
            Path("/repo/output/runs/run1"),
            args,
        )

        self.assertEqual(bank_path, Path("/repo/banks/prompt_bank.json"))
        self.assertEqual(output_bank_path, Path("/repo/banks/prompt_bank.json"))
        self.assertEqual(report_path, Path("/repo/output/runs/run1/adversarial_rule_report.json"))

    def test_online_adversarial_project_config_bootstraps_active_json_bank(self):
        args = argparse.Namespace(
            model="gpt-5.4-mini-2026-03-17",
            api_key="secret",
            api_url="https://example.test/responses",
            base_url="https://unused.test",
            llm_protocol="openai-responses",
            session_id="session-123",
            logid="log-456",
            max_fix_attempts=5,
            max_test_fix_attempts=3,
            continue_on_failure=False,
            jobs=1,
            alignment_mode="llm",
            alignment_confidence_threshold=0.65,
            rules_mode="default",
            rule_limit=5,
            rules_path="",
            rules_json_bank="",
            rule_retrieval_backend="keyword",
            rule_retrieval_index_cache="",
            rule_interaction_scorer="heuristic",
            experiment_profile="",
            adversarial_rounds=None,
            adversarial_rule_bank="",
            adversarial_rule_output_bank="",
            adversarial_rule_report="",
            online_adversarial_rules=True,
            adversarial_admit_trace_candidates=False,
            adversarial_run_synthetic_challenges=False,
            adversarial_run_compositional_challenges=False,
            adversarial_admit_composition_failure_candidates=False,
            adversarial_live_structured_agents=False,
            adversarial_live_trace_synthesis=False,
            adversarial_live_trace_contradiction_review=False,
            adversarial_live_composition_relation_review=False,
            adversarial_live_semantic_oracle_validation=False,
            adversarial_live_activation_validation=False,
            adversarial_live_translator_replay=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_root = root / "output" / "runs" / "run1"
            config = translation_runner.project_config(root, run_root, "amp", args)

            bank_path = run_root / "rules.active.json"
            self.assertTrue(bank_path.exists())
            raw_bank = bank_path.read_text(encoding="utf-8")

        self.assertEqual(config["rules_json_bank_path"], str(bank_path))
        self.assertEqual(config["adversarial_rule_bank_path"], str(bank_path))
        self.assertEqual(config["adversarial_rule_output_bank_path"], str(bank_path))
        self.assertIn("interface-contract-first", raw_bank)

    def test_project_config_bootstraps_active_json_bank_for_default_prompt_retrieval(self):
        args = argparse.Namespace(
            model="gpt-5.4-mini-2026-03-17",
            api_key="secret",
            api_url="https://example.test/responses",
            base_url="https://unused.test",
            llm_protocol="openai-responses",
            session_id="session-123",
            logid="log-456",
            max_fix_attempts=5,
            max_test_fix_attempts=3,
            continue_on_failure=False,
            jobs=1,
            alignment_mode="llm",
            alignment_confidence_threshold=0.65,
            rules_mode="default",
            rule_limit=5,
            rules_path="",
            rules_json_bank="",
            rule_retrieval_backend="keyword",
            rule_retrieval_index_cache="",
            rule_interaction_scorer="heuristic",
            experiment_profile="",
            adversarial_rounds=None,
            adversarial_rule_bank="",
            adversarial_rule_output_bank="",
            adversarial_rule_report="",
            online_adversarial_rules=False,
            adversarial_admit_trace_candidates=False,
            adversarial_run_synthetic_challenges=False,
            adversarial_run_compositional_challenges=False,
            adversarial_admit_composition_failure_candidates=False,
            adversarial_live_structured_agents=False,
            adversarial_live_trace_synthesis=False,
            adversarial_live_trace_contradiction_review=False,
            adversarial_live_composition_relation_review=False,
            adversarial_live_semantic_oracle_validation=False,
            adversarial_live_activation_validation=False,
            adversarial_live_translator_replay=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_root = root / "output" / "runs" / "run1"
            config = translation_runner.project_config(root, run_root, "amp", args)

            bank_path = run_root / "rules.active.json"
            self.assertTrue(bank_path.exists())
            raw_bank = bank_path.read_text(encoding="utf-8")

        self.assertEqual(config["rules_json_bank_path"], str(bank_path))
        self.assertTrue(config["online_adversarial_rules"])
        self.assertEqual(config["adversarial_rule_bank_path"], str(bank_path))
        self.assertEqual(config["adversarial_rule_output_bank_path"], str(bank_path))
        self.assertEqual(
            config["adversarial_rule_report_path"],
            str(run_root / "adversarial_rule_report.json"),
        )
        self.assertIn("interface-contract-first", raw_bank)

    def test_project_config_uses_project_local_active_bank_for_custom_parallel_online_updates(self):
        parser = translation_runner.build_arg_parser()
        args = parser.parse_args(["leftpad", "--jobs", "2"])

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_root = root / "output" / "runs" / "run1"
            config = translation_runner.project_config(root, run_root, "amp", args)

            bank_path = run_root / "amp" / "rules.active.json"

        self.assertEqual(config["rules_json_bank_path"], str(bank_path))
        self.assertTrue(config["online_adversarial_rules"])
        self.assertEqual(config["adversarial_rule_bank_path"], str(bank_path))
        self.assertEqual(config["adversarial_rule_output_bank_path"], str(bank_path))
        self.assertEqual(
            config["adversarial_rule_report_path"],
            str(run_root / "amp" / "adversarial_rule_report.json"),
        )

    def test_active_json_bank_bootstrap_is_shared_across_parallel_project_configs(self):
        class FakeBank:
            def save(self, path):
                time.sleep(0.05)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{"rules": []}', encoding="utf-8")

        from_legacy_calls = 0

        def fake_from_legacy(_rules):
            nonlocal from_legacy_calls
            from_legacy_calls += 1
            return FakeBank()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_root = root / "output" / "runs" / "run1"
            with mock.patch.object(
                translation_runner.RuleBank,
                "from_legacy",
                side_effect=fake_from_legacy,
            ):
                with ThreadPoolExecutor(max_workers=5) as executor:
                    results = list(
                        executor.map(
                            lambda _: translation_runner.ensure_active_json_rule_bank(
                                root,
                                run_root,
                                rules_path="",
                            ),
                            range(5),
                        )
                    )

        self.assertEqual(results, [str(run_root / "rules.active.json")] * 5)
        self.assertEqual(from_legacy_calls, 1)

    def test_project_config_records_adversarial_rule_paths(self):
        args = argparse.Namespace(
            model="gpt-5.4-mini-2026-03-17",
            api_key="secret",
            api_url="https://example.test/responses",
            base_url="https://unused.test",
            llm_protocol="openai-responses",
            session_id="session-123",
            logid="log-456",
            max_fix_attempts=5,
            max_test_fix_attempts=3,
            continue_on_failure=False,
            jobs=1,
            alignment_mode="llm",
            alignment_confidence_threshold=0.65,
            rules_mode="default",
            rule_limit=5,
            rules_path="",
            rules_json_bank="banks/prompt_bank.json",
            rule_retrieval_backend="auto",
            rule_retrieval_index_cache="",
            rule_interaction_scorer="live",
            experiment_profile="",
            adversarial_rounds=None,
            adversarial_rule_bank="banks/input.json",
            adversarial_rule_output_bank="",
            adversarial_rule_report="reports/learning.json",
            online_adversarial_rules=True,
            adversarial_admit_trace_candidates=True,
            adversarial_run_synthetic_challenges=True,
            adversarial_run_compositional_challenges=True,
            adversarial_admit_composition_failure_candidates=True,
            adversarial_live_structured_agents=True,
            adversarial_live_trace_synthesis=True,
            adversarial_live_trace_contradiction_review=True,
            adversarial_live_composition_relation_review=True,
            adversarial_live_semantic_oracle_validation=True,
            adversarial_live_activation_validation=True,
            adversarial_live_translator_replay=True,
            adversarial_pre_translation_challenges=True,
            adversarial_pre_translation_refinement=True,
            adversarial_pre_translation_refinement_writeback=True,
        )

        root = Path("/repo")
        run_root = root / "output" / "runs" / "run"
        config = translation_runner.project_config(root, run_root, "amp", args)

        self.assertEqual(config["rules_json_bank_path"], "/repo/banks/prompt_bank.json")
        self.assertEqual(config["rule_retrieval_backend_preference"], "auto")
        self.assertEqual(
            config["rule_retrieval_index_cache_path"],
            "/repo/output/runs/run/rule_retrieval.faiss",
        )
        self.assertEqual(config["rule_interaction_scorer"], "live")
        self.assertTrue(config["online_adversarial_rules"])
        self.assertEqual(config["adversarial_rule_bank_path"], "/repo/banks/input.json")
        self.assertEqual(
            config["adversarial_rule_output_bank_path"],
            "/repo/output/runs/run/adversarial_rules.updated.json",
        )
        self.assertEqual(config["adversarial_rule_report_path"], "/repo/reports/learning.json")
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
                "pre_translation_refinement_rounds": 1,
                "pre_translation_refinement_writeback": True,
            },
        )

    def test_no_compositional_profile_gates_compositional_adversarial_metadata(self):
        args = self._all_adversarial_options_args(experiment_profile="no-compositional")

        options = translation_runner.adversarial_learning_option_metadata(args)

        self.assertTrue(options["admit_trace_candidates"])
        self.assertTrue(options["run_synthetic_challenges"])
        self.assertTrue(options["live_structured_agents"])
        self.assertTrue(options["live_trace_synthesis"])
        self.assertTrue(options["live_trace_contradiction_review"])
        self.assertTrue(options["live_semantic_oracle_validation"])
        self.assertTrue(options["live_activation_validation"])
        self.assertTrue(options["live_translator_replay"])
        self.assertTrue(options["pre_translation_challenges"])
        self.assertTrue(options["pre_translation_refinement"])
        self.assertTrue(options["pre_translation_refinement_writeback"])
        self.assertFalse(options["run_compositional_challenges"])
        self.assertFalse(options["admit_composition_failure_candidates"])
        self.assertFalse(options["live_composition_relation_review"])

    def test_full_profile_enables_live_adversarial_runtime_defaults_for_sequential_runs(self):
        parser = translation_runner.build_arg_parser()
        args = parser.parse_args(["leftpad", "--experiment-profile", "full", "--jobs", "1"])

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
        self.assertTrue(options["live_trace_synthesis"])
        self.assertTrue(options["live_trace_contradiction_review"])
        self.assertTrue(options["live_composition_relation_review"])
        self.assertTrue(options["live_semantic_oracle_validation"])
        self.assertTrue(options["live_activation_validation"])
        self.assertTrue(options["live_translator_replay"])

    def test_full_profile_enables_writeback_for_parallel_project_local_bank(self):
        parser = translation_runner.build_arg_parser()
        args = parser.parse_args(["leftpad", "--experiment-profile", "full", "--jobs", "2"])

        options = translation_runner.adversarial_learning_option_metadata(args)

        self.assertTrue(options["pre_translation_challenges"])
        self.assertTrue(options["pre_translation_refinement"])
        self.assertEqual(options["pre_translation_refinement_rounds"], 5)
        self.assertTrue(options["pre_translation_refinement_writeback"])

    def test_full_profile_enables_writeback_for_parallel_shared_bank(self):
        parser = translation_runner.build_arg_parser()
        args = parser.parse_args(
            [
                "leftpad",
                "--experiment-profile",
                "full",
                "--jobs",
                "2",
                "--rules-json-bank",
                "banks/shared.json",
            ]
        )

        options = translation_runner.adversarial_learning_option_metadata(args)

        self.assertTrue(options["pre_translation_challenges"])
        self.assertTrue(options["pre_translation_refinement"])
        self.assertEqual(options["pre_translation_refinement_rounds"], 5)
        self.assertTrue(options["pre_translation_refinement_writeback"])

    def test_custom_pre_translation_refinement_defaults_to_single_round(self):
        parser = translation_runner.build_arg_parser()
        args = parser.parse_args(["leftpad", "--adversarial-pre-translation-refinement"])

        options = translation_runner.adversarial_learning_option_metadata(args)

        self.assertTrue(options["pre_translation_refinement"])
        self.assertEqual(options["pre_translation_refinement_rounds"], 1)

    def test_pre_translation_refinement_rounds_can_be_overridden(self):
        parser = translation_runner.build_arg_parser()
        args = parser.parse_args(
            [
                "leftpad",
                "--experiment-profile",
                "full",
                "--adversarial-pre-translation-refinement-rounds",
                "4",
            ]
        )

        options = translation_runner.adversarial_learning_option_metadata(args)

        self.assertEqual(options["pre_translation_refinement_rounds"], 4)

    def test_full_profile_defaults_online_updates_for_sequential_runs(self):
        parser = translation_runner.build_arg_parser()
        args = parser.parse_args(["leftpad", "--experiment-profile", "full", "--jobs", "1"])

        self.assertTrue(translation_runner.online_adversarial_rules_enabled(args))

    def test_custom_profile_defaults_online_updates_for_sequential_active_bank(self):
        parser = translation_runner.build_arg_parser()
        args = parser.parse_args(["leftpad", "--jobs", "1"])

        self.assertTrue(translation_runner.online_adversarial_rules_enabled(args))

    def test_custom_profile_defaults_online_updates_for_parallel_project_local_bank(self):
        parser = translation_runner.build_arg_parser()
        args = parser.parse_args(["leftpad", "--jobs", "2"])

        self.assertTrue(translation_runner.online_adversarial_rules_enabled(args))

    def test_custom_profile_defaults_online_updates_for_parallel_shared_bank(self):
        parser = translation_runner.build_arg_parser()
        args = parser.parse_args(
            [
                "leftpad",
                "--jobs",
                "2",
                "--rules-json-bank",
                "banks/shared.json",
            ]
        )

        self.assertTrue(translation_runner.online_adversarial_rules_enabled(args))
        self.assertTrue(translation_runner.parallel_online_adversarial_updates_supported(args))

    def test_custom_parallel_shared_prompt_bank_online_update_uses_project_report_path(self):
        parser = translation_runner.build_arg_parser()
        args = parser.parse_args(
            [
                "leftpad",
                "--jobs",
                "2",
                "--rules-json-bank",
                "banks/shared.json",
            ]
        )

        root = Path("/repo")
        run_root = root / "output" / "runs" / "run1"
        config = translation_runner.project_config(root, run_root, "amp", args)

        self.assertEqual(config["adversarial_rule_bank_path"], "/repo/banks/shared.json")
        self.assertEqual(config["adversarial_rule_output_bank_path"], "/repo/banks/shared.json")
        self.assertEqual(
            config["adversarial_rule_report_path"],
            "/repo/output/runs/run1/amp/adversarial_rule_report.json",
        )

    def test_explicit_online_updates_are_supported_for_parallel_shared_prompt_bank(self):
        parser = translation_runner.build_arg_parser()
        args = parser.parse_args(
            [
                "leftpad",
                "--jobs",
                "2",
                "--rules-json-bank",
                "banks/shared.json",
                "--online-adversarial-rules",
            ]
        )

        self.assertTrue(translation_runner.online_adversarial_rules_enabled(args))
        self.assertTrue(translation_runner.parallel_online_adversarial_updates_supported(args))

    def test_parallel_shared_prompt_bank_online_update_uses_project_report_path(self):
        parser = translation_runner.build_arg_parser()
        args = parser.parse_args(
            [
                "leftpad",
                "--jobs",
                "2",
                "--rules-json-bank",
                "banks/shared.json",
                "--online-adversarial-rules",
            ]
        )

        root = Path("/repo")
        run_root = root / "output" / "runs" / "run1"
        config = translation_runner.project_config(root, run_root, "amp", args)

        self.assertEqual(config["adversarial_rule_bank_path"], "/repo/banks/shared.json")
        self.assertEqual(config["adversarial_rule_output_bank_path"], "/repo/banks/shared.json")
        self.assertEqual(
            config["adversarial_rule_report_path"],
            "/repo/output/runs/run1/amp/adversarial_rule_report.json",
        )

    def test_explicit_online_updates_are_supported_for_parallel_adversarial_bank(self):
        parser = translation_runner.build_arg_parser()
        args = parser.parse_args(
            [
                "leftpad",
                "--jobs",
                "2",
                "--adversarial-rule-bank",
                "banks/input.json",
                "--online-adversarial-rules",
            ]
        )

        self.assertTrue(translation_runner.online_adversarial_rules_enabled(args))
        self.assertTrue(translation_runner.parallel_online_adversarial_updates_supported(args))

    def test_parallel_adversarial_bank_online_update_uses_same_bank_with_project_report(self):
        parser = translation_runner.build_arg_parser()
        args = parser.parse_args(
            [
                "leftpad",
                "--jobs",
                "2",
                "--adversarial-rule-bank",
                "banks/input.json",
                "--online-adversarial-rules",
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_root = root / "output" / "runs" / "run1"
            config = translation_runner.project_config(root, run_root, "amp", args)

            self.assertEqual(
                config["adversarial_rule_bank_path"],
                str(root / "banks" / "input.json"),
            )
            self.assertEqual(
                config["adversarial_rule_output_bank_path"],
                str(root / "banks" / "input.json"),
            )
            self.assertEqual(
                config["adversarial_rule_report_path"],
                str(run_root / "amp" / "adversarial_rule_report.json"),
            )

    def test_explicit_adversarial_bank_defaults_to_primary_online_prompt_bank(self):
        parser = translation_runner.build_arg_parser()
        args = parser.parse_args(
            [
                "leftpad",
                "--jobs",
                "1",
                "--adversarial-rule-bank",
                "banks/input.json",
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_root = root / "output" / "runs" / "run1"
            config = translation_runner.project_config(root, run_root, "amp", args)

        expected_bank = str(root / "banks" / "input.json")
        self.assertTrue(translation_runner.online_adversarial_rules_enabled(args))
        self.assertEqual(config["rules_json_bank_path"], expected_bank)
        self.assertEqual(config["adversarial_rule_bank_path"], expected_bank)
        self.assertEqual(config["adversarial_rule_output_bank_path"], expected_bank)

    def test_parallel_explicit_adversarial_bank_defaults_to_locked_shared_online_bank(self):
        parser = translation_runner.build_arg_parser()
        args = parser.parse_args(
            [
                "leftpad",
                "--jobs",
                "2",
                "--adversarial-rule-bank",
                "banks/shared.json",
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_root = root / "output" / "runs" / "run1"
            config = translation_runner.project_config(root, run_root, "amp", args)

        expected_bank = str(root / "banks" / "shared.json")
        self.assertTrue(translation_runner.online_adversarial_rules_enabled(args))
        self.assertTrue(translation_runner.parallel_online_adversarial_updates_supported(args))
        self.assertEqual(config["rules_json_bank_path"], expected_bank)
        self.assertEqual(config["adversarial_rule_bank_path"], expected_bank)
        self.assertEqual(config["adversarial_rule_output_bank_path"], expected_bank)
        self.assertEqual(
            config["adversarial_rule_report_path"],
            str(root / "output" / "runs" / "run1" / "amp" / "adversarial_rule_report.json"),
        )

    def test_full_profile_defaults_online_updates_for_parallel_runs_with_project_local_bank(self):
        parser = translation_runner.build_arg_parser()
        args = parser.parse_args(["leftpad", "--experiment-profile", "full", "--jobs", "2"])

        self.assertTrue(translation_runner.online_adversarial_rules_enabled(args))

    def test_full_profile_defaults_online_updates_for_parallel_shared_prompt_bank(self):
        parser = translation_runner.build_arg_parser()
        args = parser.parse_args(
            [
                "leftpad",
                "--experiment-profile",
                "full",
                "--jobs",
                "2",
                "--rules-json-bank",
                "banks/shared.json",
            ]
        )

        self.assertTrue(translation_runner.online_adversarial_rules_enabled(args))
        self.assertTrue(translation_runner.parallel_online_adversarial_updates_supported(args))

    def test_full_profile_parallel_shared_prompt_bank_online_update_uses_project_report_path(self):
        parser = translation_runner.build_arg_parser()
        args = parser.parse_args(
            [
                "leftpad",
                "--experiment-profile",
                "full",
                "--jobs",
                "2",
                "--rules-json-bank",
                "banks/shared.json",
            ]
        )

        root = Path("/repo")
        run_root = root / "output" / "runs" / "run1"
        config = translation_runner.project_config(root, run_root, "amp", args)

        self.assertEqual(config["adversarial_rule_bank_path"], "/repo/banks/shared.json")
        self.assertEqual(config["adversarial_rule_output_bank_path"], "/repo/banks/shared.json")
        self.assertEqual(
            config["adversarial_rule_report_path"],
            "/repo/output/runs/run1/amp/adversarial_rule_report.json",
        )

    def test_custom_pre_translation_refinement_defaults_writeback_to_mutable_bank(self):
        parser = translation_runner.build_arg_parser()
        args = parser.parse_args(
            [
                "leftpad",
                "--adversarial-pre-translation-refinement",
            ]
        )

        options = translation_runner.adversarial_learning_option_metadata(args)

        self.assertTrue(options["pre_translation_refinement"])
        self.assertTrue(options["pre_translation_refinement_writeback"])

    def test_no_compositional_profile_defaults_keep_single_adversarial_paths(self):
        parser = translation_runner.build_arg_parser()
        args = parser.parse_args(["leftpad", "--experiment-profile", "no-compositional", "--jobs", "1"])

        options = translation_runner.adversarial_learning_option_metadata(args)

        self.assertTrue(options["admit_trace_candidates"])
        self.assertTrue(options["run_synthetic_challenges"])
        self.assertTrue(options["pre_translation_challenges"])
        self.assertTrue(options["pre_translation_refinement"])
        self.assertTrue(options["pre_translation_refinement_writeback"])
        self.assertFalse(options["run_compositional_challenges"])
        self.assertFalse(options["admit_composition_failure_candidates"])

    def test_no_single_profile_gates_single_and_compositional_adversarial_metadata(self):
        args = self._all_adversarial_options_args(experiment_profile="no-single")

        options = translation_runner.adversarial_learning_option_metadata(args)

        self.assertEqual(
            options,
            {
                "admit_trace_candidates": False,
                "run_synthetic_challenges": False,
                "run_compositional_challenges": False,
                "admit_composition_failure_candidates": False,
                "live_structured_agents": False,
                "live_trace_synthesis": False,
                "live_trace_contradiction_review": False,
                "live_composition_relation_review": False,
                "live_semantic_oracle_validation": False,
                "live_activation_validation": False,
                "live_translator_replay": False,
                "pre_translation_challenges": False,
                "pre_translation_refinement": False,
                "pre_translation_refinement_rounds": 0,
                "pre_translation_refinement_writeback": False,
            },
        )

    def test_baseline_profile_gates_all_adversarial_learning_runtime_options(self):
        args = self._all_adversarial_options_args(experiment_profile="baseline")
        args.adversarial_rule_bank = "banks/input.json"
        args.adversarial_rule_output_bank = ""
        args.adversarial_rule_report = ""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_root = root / "output" / "runs" / "run1"
            with mock.patch.object(
                translation_runner,
                "run_offline_adversarial_learning",
                return_value={"challenges": 0, "accepted_updates": 0},
            ) as offline, mock.patch.object(
                translation_runner,
                "create_agent",
                return_value=object(),
                create=True,
            ) as create_agent:
                translation_runner.run_adversarial_rule_update(root, run_root, args)

        kwargs = offline.call_args.kwargs
        self.assertFalse(kwargs["admit_trace_candidates"])
        self.assertFalse(kwargs["run_synthetic_challenges"])
        self.assertFalse(kwargs["run_compositional_challenges"])
        self.assertFalse(kwargs["admit_composition_failure_candidates"])
        self.assertIsNone(kwargs["challenger_agent"])
        self.assertIsNone(kwargs["refiner_agent"])
        self.assertIsNone(kwargs["trace_synthesizer"])
        self.assertIsNone(kwargs["trace_contradiction_reviewer"])
        self.assertIsNone(kwargs["composition_relation_reviewer"])
        self.assertIsNone(kwargs["oracle_validator"])
        self.assertIsNone(kwargs["activation_validator"])
        self.assertIsNone(kwargs["replay_translate_fn"])
        create_agent.assert_not_called()

    def test_post_batch_adversarial_update_runs_only_when_bank_is_configured(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_root = root / "output" / "runs" / "run1"
            args = argparse.Namespace(
                adversarial_rule_bank="banks/input.json",
                adversarial_rule_output_bank="",
                adversarial_rule_report="",
                adversarial_admit_trace_candidates=True,
                adversarial_run_synthetic_challenges=True,
                adversarial_run_compositional_challenges=True,
                adversarial_admit_composition_failure_candidates=True,
                adversarial_live_structured_agents=True,
                adversarial_live_trace_synthesis=True,
                adversarial_live_trace_contradiction_review=True,
                adversarial_live_composition_relation_review=True,
                adversarial_live_semantic_oracle_validation=True,
                adversarial_live_activation_validation=True,
                adversarial_live_translator_replay=True,
                api_key="secret",
                api_url="https://example.test/responses",
                base_url="https://unused.test",
                model="model-x",
                llm_protocol="openai-responses",
                min_request_interval=0.0,
                rate_limit_retry_delay=60.0,
                llm_timeout=300,
                session_id="session-123",
                logid="log-456",
            )

            with mock.patch.object(
                translation_runner,
                "run_offline_adversarial_learning",
                return_value={"challenges": 2, "accepted_updates": 1},
            ) as offline, mock.patch.object(translation_runner, "create_agent", return_value=object(), create=True):
                result = translation_runner.run_adversarial_rule_update(root, run_root, args)

        self.assertEqual(result, {"challenges": 2, "accepted_updates": 1})
        offline.assert_called_once()
        self.assertEqual(
            offline.call_args.args,
            (
                run_root,
                root / "banks" / "input.json",
                run_root / "adversarial_rules.updated.json",
                run_root / "adversarial_rule_report.json",
            ),
        )
        kwargs = offline.call_args.kwargs
        self.assertTrue(kwargs["admit_trace_candidates"])
        self.assertTrue(kwargs["run_synthetic_challenges"])
        self.assertTrue(kwargs["run_compositional_challenges"])
        self.assertTrue(kwargs["admit_composition_failure_candidates"])
        self.assertIsNotNone(kwargs["challenger_agent"])
        self.assertIsNotNone(kwargs["refiner_agent"])
        self.assertIsNotNone(kwargs["trace_synthesizer"])
        self.assertIsNotNone(kwargs["trace_contradiction_reviewer"])
        self.assertIsNotNone(kwargs["composition_relation_reviewer"])
        self.assertIsNotNone(kwargs["oracle_validator"])
        self.assertIsNotNone(kwargs["activation_validator"])
        self.assertIsNotNone(kwargs["replay_translate_fn"])

    def test_post_batch_adversarial_update_skips_when_bank_is_absent(self):
        args = argparse.Namespace(
            rules_json_bank="",
            adversarial_rule_bank="",
            adversarial_rule_output_bank="",
            adversarial_rule_report="",
            online_adversarial_rules=False,
        )

        with mock.patch.object(translation_runner, "run_offline_adversarial_learning") as offline:
            result = translation_runner.run_adversarial_rule_update(Path("/repo"), Path("/repo/output/runs/run1"), args)

        self.assertIsNone(result)
        offline.assert_not_called()

    def test_online_adversarial_update_uses_bootstrapped_active_bank_without_explicit_bank(self):
        args = argparse.Namespace(
            rules_mode="default",
            rules_path="",
            rules_json_bank="",
            experiment_profile="",
            adversarial_rule_bank="",
            adversarial_rule_output_bank="",
            adversarial_rule_report="",
            online_adversarial_rules=True,
            adversarial_admit_trace_candidates=False,
            adversarial_run_synthetic_challenges=False,
            adversarial_run_compositional_challenges=False,
            adversarial_admit_composition_failure_candidates=False,
            adversarial_live_structured_agents=False,
            adversarial_live_trace_synthesis=False,
            adversarial_live_trace_contradiction_review=False,
            adversarial_live_composition_relation_review=False,
            adversarial_live_semantic_oracle_validation=False,
            adversarial_live_activation_validation=False,
            adversarial_live_translator_replay=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_root = root / "output" / "runs" / "run1"
            with mock.patch.object(
                translation_runner,
                "run_offline_adversarial_learning",
                return_value={"challenges": 0, "accepted_updates": 0},
            ) as offline:
                result = translation_runner.run_adversarial_rule_update(root, run_root, args)

            bank_path = run_root / "rules.active.json"
            bank_exists = bank_path.exists()

        self.assertEqual(result, {"challenges": 0, "accepted_updates": 0})
        self.assertTrue(bank_exists)
        self.assertEqual(
            offline.call_args.args,
            (
                run_root,
                bank_path,
                bank_path,
                run_root / "adversarial_rule_report.json",
            ),
        )

    def test_online_adversarial_update_runs_after_each_sequential_project(self):
        class FakeTranslator:
            configs = []

            def __init__(self, config):
                self.config = config
                self.configs.append(config)

            def run(self):
                Path(self.config["output_dir"]).mkdir(parents=True, exist_ok=True)
                (Path(self.config["output_dir"]) / "translation_report.json").write_text(
                    "{}",
                    encoding="utf-8",
                )

        args = argparse.Namespace(
            model="gpt-5.4-mini-2026-03-17",
            api_key="secret",
            api_url="https://example.test/responses",
            base_url="https://unused.test",
            llm_protocol="openai-responses",
            session_id="session-123",
            logid="log-456",
            max_fix_attempts=5,
            max_test_fix_attempts=3,
            continue_on_failure=False,
            jobs=1,
            alignment_mode="llm",
            alignment_confidence_threshold=0.65,
            rules_mode="default",
            rule_limit=5,
            rules_path="",
            rules_json_bank="banks/prompt_bank.json",
            rule_retrieval_backend="keyword",
            rule_retrieval_index_cache="",
            rule_interaction_scorer="heuristic",
            experiment_profile="",
            adversarial_rounds=None,
            adversarial_rule_bank="",
            adversarial_rule_output_bank="",
            adversarial_rule_report="",
            online_adversarial_rules=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "banks").mkdir()
            (root / "banks" / "prompt_bank.json").write_text('{"rules": []}', encoding="utf-8")
            run_root = root / "output" / "runs" / "run1"
            with mock.patch.object(
                translation_runner,
                "run_adversarial_rule_update",
                return_value={"accepted_updates": 0},
            ) as update:
                translation_runner.run_project_batch(
                    ["leftpad", "morton"],
                    root_dir=root,
                    run_root=run_root,
                    args=args,
                    translator_cls=FakeTranslator,
                )

        self.assertEqual(update.call_count, 2)
        self.assertEqual([call.args[1] for call in update.call_args_list], [run_root, run_root])


class AlignmentConfigTest(unittest.TestCase):
    def test_project_config_passes_alignment_options(self):
        args = argparse.Namespace(
            model="gpt-5.4-mini-2026-03-17",
            api_key="secret",
            api_url="https://example.test/responses",
            base_url="https://unused.test",
            llm_protocol="openai-responses",
            session_id="session-123",
            logid="log-456",
            max_fix_attempts=5,
            max_test_fix_attempts=3,
            continue_on_failure=False,
            jobs=1,
            alignment_mode="llm",
            alignment_confidence_threshold=0.72,
            rules_mode="default",
            rule_limit=5,
            rules_json_bank="banks/prompt_bank.json",
        )

        config = translation_runner.project_config(Path("/repo"), Path("/repo/output/runs/run"), "amp", args)

        self.assertEqual(config["alignment_mode"], "llm")
        self.assertEqual(config["alignment_confidence_threshold"], 0.72)

    def test_project_config_passes_rule_options(self):
        args = argparse.Namespace(
            model="gpt-5.4-mini-2026-03-17",
            api_key="secret",
            api_url="https://example.test/responses",
            base_url="https://unused.test",
            llm_protocol="openai-responses",
            session_id="session-123",
            logid="log-456",
            max_fix_attempts=5,
            max_test_fix_attempts=3,
            continue_on_failure=False,
            jobs=1,
            alignment_mode="llm",
            alignment_confidence_threshold=0.65,
            rules_mode="off",
            rule_limit=0,
            rules_path="snapshots/rules/model_variant.py",
        )

        config = translation_runner.project_config(Path("/repo"), Path("/repo/output/runs/run"), "amp", args)

        self.assertFalse(config["rules_enabled"])
        self.assertEqual(config["rule_limit"], 0)
        self.assertEqual(config["rules_path"], "/repo/snapshots/rules/model_variant.py")

    def test_project_config_passes_json_rule_bank_path(self):
        args = argparse.Namespace(
            model="gpt-5.4-mini-2026-03-17",
            api_key="secret",
            api_url="https://example.test/responses",
            base_url="https://unused.test",
            llm_protocol="openai-responses",
            session_id="session-123",
            logid="log-456",
            max_fix_attempts=5,
            max_test_fix_attempts=3,
            continue_on_failure=False,
            jobs=1,
            alignment_mode="llm",
            alignment_confidence_threshold=0.65,
            rules_mode="default",
            rule_limit=5,
            rules_path="",
            rules_json_bank="banks/prompt_bank.json",
        )

        config = translation_runner.project_config(Path("/repo"), Path("/repo/output/runs/run"), "amp", args)

        self.assertEqual(config["rules_json_bank_path"], "/repo/banks/prompt_bank.json")

    def test_project_config_applies_full_experiment_profile_defaults(self):
        args = argparse.Namespace(
            model="ali-deepseek-v4-flash",
            api_key="secret",
            api_url="https://example.test/responses",
            base_url="https://unused.test",
            llm_protocol="openai-responses",
            session_id="session-123",
            logid="log-456",
            max_fix_attempts=5,
            max_test_fix_attempts=3,
            continue_on_failure=False,
            jobs=1,
            alignment_mode="llm",
            alignment_confidence_threshold=0.65,
            rules_mode="default",
            rule_limit=5,
            rules_path="",
            rules_json_bank="",
            rule_retrieval_backend="keyword",
            rule_retrieval_index_cache="",
            experiment_profile="full",
            adversarial_rounds=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path.cwd()
            run_root = Path(tmpdir) / "run"
            config = translation_runner.project_config(root, run_root, "amp", args)
            bank_path = run_root / "rules.active.json"
            self.assertTrue(bank_path.exists())
            bank_record_count = len(translation_runner.RuleBank.load(bank_path).records)

        self.assertTrue(config["rules_enabled"])
        self.assertEqual(config["rules_path"], "")
        self.assertEqual(config["rules_json_bank_path"], str(bank_path))
        self.assertEqual(bank_record_count, 49)
        self.assertEqual(config["experiment_profile"], "full")
        self.assertEqual(config["experiment_ablation_label"], "TRAIL")
        self.assertEqual(config["rule_stage"], "multi")
        self.assertTrue(config["rule_mining"])
        self.assertTrue(config["single_rule_adversarial_learning"])
        self.assertTrue(config["compositional_rule_adversarial_learning"])
        self.assertEqual(config["rule_interaction_scorer"], "live")
        self.assertEqual(config["adversarial_rounds"], 5)
        self.assertTrue(config["online_adversarial_rules"])
        self.assertTrue(config["adversarial_learning_options"]["run_synthetic_challenges"])
        self.assertTrue(config["adversarial_learning_options"]["run_compositional_challenges"])
        self.assertTrue(config["adversarial_learning_options"]["pre_translation_challenges"])
        self.assertTrue(config["adversarial_learning_options"]["pre_translation_refinement"])
        self.assertEqual(config["adversarial_learning_options"]["pre_translation_refinement_rounds"], 5)
        self.assertTrue(config["adversarial_learning_options"]["pre_translation_refinement_writeback"])
        self.assertEqual(config["model_display_name"], "DeepSeek-V4-Flash")

    def test_project_config_keeps_no_compositional_interaction_scorer_heuristic(self):
        parser = translation_runner.build_arg_parser()
        args = parser.parse_args(["amp", "--experiment-profile", "no-compositional"])

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path.cwd()
            run_root = Path(tmpdir) / "run"
            config = translation_runner.project_config(root, run_root, "amp", args)

        self.assertEqual(config["experiment_profile"], "without_compositional")
        self.assertTrue(config["single_rule_adversarial_learning"])
        self.assertFalse(config["compositional_rule_adversarial_learning"])
        self.assertEqual(config["rule_interaction_scorer"], "heuristic")

    def test_parallel_full_profile_uses_project_local_active_bank_for_online_updates(self):
        args = argparse.Namespace(
            model="ali-deepseek-v4-flash",
            api_key="secret",
            api_url="https://example.test/responses",
            base_url="https://unused.test",
            llm_protocol="openai-responses",
            session_id="session-123",
            logid="log-456",
            max_fix_attempts=5,
            max_test_fix_attempts=3,
            continue_on_failure=False,
            jobs=2,
            alignment_mode="llm",
            alignment_confidence_threshold=0.65,
            rules_mode="default",
            rule_limit=5,
            rules_path="",
            rules_json_bank="",
            rule_retrieval_backend="keyword",
            rule_retrieval_index_cache="",
            experiment_profile="full",
            adversarial_rounds=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path.cwd()
            run_root = Path(tmpdir) / "run"
            config = translation_runner.project_config(root, run_root, "amp", args)
            bank_path = run_root / "amp" / "rules.active.json"
            self.assertTrue(bank_path.exists())

        self.assertTrue(config["online_adversarial_rules"])
        self.assertEqual(config["rules_json_bank_path"], str(bank_path))
        self.assertEqual(config["adversarial_rule_bank_path"], str(bank_path))
        self.assertEqual(config["adversarial_rule_output_bank_path"], str(bank_path))
        self.assertEqual(
            config["adversarial_rule_report_path"],
            str(run_root / "amp" / "adversarial_rule_report.json"),
        )

    def test_project_config_keeps_explicit_rule_path_over_profile_default(self):
        args = argparse.Namespace(
            model="gpt-5.4-mini-2026-03-17",
            api_key="secret",
            api_url="https://example.test/responses",
            base_url="https://unused.test",
            llm_protocol="openai-responses",
            session_id="session-123",
            logid="log-456",
            max_fix_attempts=5,
            max_test_fix_attempts=3,
            continue_on_failure=False,
            jobs=1,
            alignment_mode="llm",
            alignment_confidence_threshold=0.65,
            rules_mode="default",
            rule_limit=5,
            rules_path="snapshots/rules/custom.py",
            rules_json_bank="banks/prompt_bank.json",
            experiment_profile="full",
            adversarial_rounds=3,
        )

        config = translation_runner.project_config(Path("/repo"), Path("/repo/output/runs/run"), "amp", args)

        self.assertEqual(config["rules_path"], "/repo/snapshots/rules/custom.py")
        self.assertEqual(config["experiment_profile"], "full")
        self.assertEqual(config["adversarial_rounds"], 3)
        self.assertEqual(config["model_display_name"], "GPT-5.4-mini")

    def test_project_config_baseline_profile_disables_rules(self):
        args = argparse.Namespace(
            model="kimi-k2.5",
            api_key="secret",
            api_url="https://example.test/responses",
            base_url="https://unused.test",
            llm_protocol="openai-responses",
            session_id="session-123",
            logid="log-456",
            max_fix_attempts=5,
            max_test_fix_attempts=3,
            continue_on_failure=False,
            jobs=1,
            alignment_mode="llm",
            alignment_confidence_threshold=0.65,
            rules_mode="default",
            rule_limit=5,
            rules_path="",
            experiment_profile="baseline",
            adversarial_rounds=None,
        )

        config = translation_runner.project_config(Path("/repo"), Path("/repo/output/runs/run"), "amp", args)

        self.assertFalse(config["rules_enabled"])
        self.assertEqual(config["rules_path"], "")
        self.assertEqual(config["experiment_profile"], "without_rule_mining")
        self.assertEqual(config["experiment_ablation_label"], "w/o Rule Mining")
        self.assertEqual(config["rule_stage"], "off")
        self.assertFalse(config["rule_mining"])
        self.assertEqual(config["model_display_name"], "Kimi-K2.5")

    def test_llm_config_defaults_to_twenty_retries_with_bounded_backoff(self):
        args = argparse.Namespace(
            model="gpt-5.4-mini-2026-03-17",
            api_key="secret",
            api_url="https://example.test/responses",
            base_url="https://unused.test",
            llm_protocol="openai-responses",
            session_id="session-123",
            logid="log-456",
            min_request_interval=7.5,
            rate_limit_retry_delay=120,
        )

        config = translation_runner.build_llm_config(args)

        self.assertEqual(config["max_retries"], 20)
        self.assertEqual(config["retry_initial_delay"], 4)
        self.assertEqual(config["retry_max_delay"], 60)
        self.assertEqual(config["max_tokens"], 128000)
        self.assertEqual(config["temperature"], 1.0)
        self.assertEqual(config["reasoning_effort"], "medium")
        self.assertEqual(config["include"], ["reasoning.encrypted_content"])
        self.assertEqual(config["min_request_interval"], 7.5)
        self.assertEqual(config["rate_limit_retry_delay"], 120)
        self.assertEqual(config["timeout"], 300)

    def test_llm_config_reads_request_timeout(self):
        args = argparse.Namespace(
            model="gpt-5.4-mini-2026-03-17",
            api_key="secret",
            api_url="https://example.test/responses",
            base_url="https://unused.test",
            llm_protocol="openai-responses",
            session_id="session-123",
            logid="log-456",
            min_request_interval=0,
            rate_limit_retry_delay=60,
            llm_timeout=900,
        )

        config = translation_runner.build_llm_config(args)

        self.assertEqual(config["timeout"], 900)

    def test_llm_config_forces_ali_deepseek_flash_to_chat_with_thinking(self):
        args = argparse.Namespace(
            model="ali-deepseek-v4-flash",
            api_key="secret",
            api_url="https://example.test/responses",
            base_url="https://example.test/chat",
            llm_protocol="openai-responses",
            session_id="session-123",
            logid="log-456",
            min_request_interval=0,
            rate_limit_retry_delay=60,
        )

        config = translation_runner.build_llm_config(args)

        self.assertEqual(config["protocol"], "openai-chat")
        self.assertEqual(config["max_retries"], 20)
        self.assertEqual(config["max_tokens"], 128000)
        self.assertEqual(config["chat_token_parameter"], "max_tokens")
        self.assertEqual(config["chat_extra_body"], {"enable_thinking": True})

    def test_llm_config_forces_kimi_to_chat_with_max_completion_tokens(self):
        args = argparse.Namespace(
            model="kimi-k2.5",
            api_key="secret",
            api_url="https://example.test/responses",
            base_url="https://example.test/chat",
            llm_protocol="openai-responses",
            session_id="session-123",
            logid="log-456",
            min_request_interval=0,
            rate_limit_retry_delay=60,
        )

        config = translation_runner.build_llm_config(args)

        self.assertEqual(config["protocol"], "openai-chat")
        self.assertEqual(config["max_retries"], 20)
        self.assertEqual(config["max_tokens"], 128000)
        self.assertEqual(config["chat_token_parameter"], "max_completion_tokens")
        self.assertIsNone(config["chat_extra_body"])

    def test_llm_config_forces_gemini_priority_to_google_genai(self):
        args = argparse.Namespace(
            model="gemini-3.1-p-priority",
            api_key="secret",
            api_url="https://example.test/responses",
            base_url="https://example.test/chat",
            llm_protocol="openai-responses",
            session_id="session-123",
            logid="log-456",
            min_request_interval=0,
            rate_limit_retry_delay=60,
        )

        config = translation_runner.build_llm_config(args)

        self.assertEqual(config["protocol"], "google-genai")
        self.assertEqual(
            config["api_url"],
            "https://generativelanguage.googleapis.com",
        )
        self.assertEqual(config["max_retries"], 20)
        self.assertEqual(config["max_tokens"], 65536)
        self.assertEqual(config["google_api_version"], "v1")

    def test_llm_config_forces_gemini_standard_to_google_genai(self):
        args = argparse.Namespace(
            model="gemini-3.1-p",
            api_key="secret",
            api_url="https://example.test/responses",
            base_url="https://example.test/chat",
            llm_protocol="openai-responses",
            session_id="session-123",
            logid="log-456",
            min_request_interval=0,
            rate_limit_retry_delay=60,
        )

        config = translation_runner.build_llm_config(args)

        self.assertEqual(config["protocol"], "google-genai")
        self.assertEqual(
            config["api_url"],
            "https://generativelanguage.googleapis.com",
        )
        self.assertEqual(config["max_tokens"], 65536)
        self.assertEqual(config["google_api_version"], "v1")


if __name__ == "__main__":
    unittest.main()
