import json
import re
import tempfile
import unittest
from dataclasses import replace
from unittest import mock
from pathlib import Path

from c2rust_translator import adversarial_rules as adversarial_core
from c2rust_translator.alignment_planner import CBinding
from c2rust_translator.adversarial_rules import (
    ChallengeRecord,
    ExecutionEvidence,
    RuleBank,
    RuleRelationRecord,
    RuleRecord,
    RuleUpdateProposal,
    SemanticActivationValidationResult,
    SemanticOracleValidationResult,
)
from c2rust_translator.rules import (
    DEFAULT_RULES,
    LLMRuleInteractionScorer,
    StructuredInteractionScorer,
    TranslationRule,
    select_rules,
)
from c2rust_translator.rbench_interface import Callable, InterfaceIndex
from c2rust_translator.translation_agent import TranslationResult, Usage
from c2rust_translator.translator import CFunction, Translator


class FakeAgent:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def translate(self, system, prompt):
        self.calls.append({"system": system, "prompt": prompt})
        return TranslationResult(
            code=self.payload,
            raw_text=self.payload,
            elapsed_seconds=0.01,
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


class SequenceAgent:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def translate(self, system, prompt):
        self.calls.append({"system": system, "prompt": prompt})
        payload = self.payloads[min(len(self.calls) - 1, len(self.payloads) - 1)]
        return TranslationResult(
            code=payload,
            raw_text=payload,
            elapsed_seconds=0.01,
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


class FailingAgent:
    def __init__(self, error):
        self.error = error

    def translate(self, system, prompt):
        raise self.error


class FakeValidator:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def compile_project(self, check_only=True):
        self.calls += 1
        if self.responses:
            return self.responses.pop(0)
        return False, "error: exhausted fake compiler responses"

    def parse_compilation_error_details(self, output):
        return [{"message": output, "code": "E0000"}]


class FakeTestValidator:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def run_all_tests(self, timeout=120):
        self.calls += 1
        if self.responses:
            return self.responses.pop(0)
        return False, "test responses exhausted", {"total": 1, "passed": 0, "failed": 1}


class TranslatorAdversarialReplayTest(unittest.TestCase):
    def test_translate_adversarial_challenge_uses_challenge_and_rule_prompt(self):
        translator = Translator.__new__(Translator)
        translator.agent = FakeAgent("```rust\npub fn translate_challenge() -> i32 { 7 }\n```")
        translator.llm_calls = []

        rule = RuleRecord(
            id="capacity-contract",
            title="Capacity Contract",
            trigger="count and capacity",
            goal="Preserve capacity",
            constraint="Keep capacity field",
            risk="capacity drift",
        )
        challenge = ChallengeRecord(
            rule_id="capacity-contract",
            c_program="int challenge_value(void) { return 7; }",
            rust_interface="pub fn translate_challenge() -> i32;",
            harness="#[test]\nfn challenge_contract() { assert_eq!(translate_challenge(), 7); }",
            oracle="Preserve value",
            evidence=ExecutionEvidence(compiles=False, tests_pass=False),
            activation_predicate="count <= capacity",
        )

        rust_source = translator.translate_adversarial_challenge(challenge, rule)

        self.assertEqual(rust_source, "pub fn translate_challenge() -> i32 { 7 }")
        self.assertEqual(len(translator.agent.calls), 1)
        prompt = translator.agent.calls[0]["prompt"]
        self.assertIn("int challenge_value(void) { return 7; }", prompt)
        self.assertIn("pub fn translate_challenge() -> i32;", prompt)
        self.assertIn("Preserve value", prompt)
        self.assertIn("Capacity Contract", prompt)
        self.assertIn("count <= capacity", prompt)
        self.assertIn("Use the rule as a translation constraint", prompt)
        self.assertIn("Implement exactly this Rust interface", prompt)
        self.assertIn("Do not modify the harness or oracle", prompt)
        self.assertIn("no markdown", prompt)
        self.assertEqual(translator.llm_calls[0]["phase"], "adversarial_challenge_replay")
        self.assertEqual(
            translator.llm_calls[0]["system"],
            translator.agent.calls[0]["system"],
        )
        self.assertIn("expert in translating C code", translator.llm_calls[0]["system"])
        self.assertEqual(translator.llm_calls[0]["rule_id"], "capacity-contract")
        self.assertIn("pub fn translate_challenge", translator.llm_calls[0]["extracted_code"])


class TranslatorRuleConfigTest(unittest.TestCase):
    def test_configure_rules_uses_json_bank_for_prompt_retrieval(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bank_path = Path(tmpdir) / "bank.json"
            RuleBank([
                RuleRecord(
                    id="fixed-c-string",
                    title="Fixed C String",
                    trigger="fixed char array",
                    goal="Expose visible length",
                    constraint="Trim at NUL",
                    risk="zero padding leaks",
                    tags=("strings",),
                    keywords=("from_utf8_lossy", "NUL"),
                )
            ]).save(bank_path)
            translator = Translator.__new__(Translator)

            translator._configure_rules({"rules_json_bank_path": str(bank_path)})

        selected = translator.select_rules(
            "String::from_utf8_lossy shows NUL padding",
            rules=translator.rules,
            limit=1,
        )

        self.assertEqual([rule.id for rule in selected], ["fixed-c-string"])
        self.assertEqual(translator.rule_retrieval_backend, "keyword")
        self.assertIn("keyword", translator.rule_retrieval_status)

    def test_json_bank_prompt_retrieval_reloads_updated_bank(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bank_path = Path(tmpdir) / "bank.json"
            RuleBank([
                RuleRecord(
                    id="fixed-c-string",
                    title="Fixed C String",
                    trigger="fixed char array",
                    goal="Expose visible length",
                    constraint="Trim at NUL",
                    risk="zero padding leaks",
                    tags=("strings",),
                    keywords=("from_utf8_lossy", "NUL"),
                )
            ]).save(bank_path)
            translator = Translator.__new__(Translator)

            translator._configure_rules(
                {
                    "rules_json_bank_path": str(bank_path),
                    "rule_retrieval_backend_preference": "keyword",
                }
            )
            initial = translator.select_rules(
                "String::from_utf8_lossy shows NUL padding",
                rules=translator.rules,
                limit=1,
            )
            RuleBank([
                RuleRecord(
                    id="capacity-contract",
                    title="Capacity Contract",
                    trigger="count and capacity",
                    goal="Preserve caller-visible capacity",
                    constraint="Do not recompute capacity from entries",
                    risk="capacity drift",
                    tags=("layout",),
                    keywords=("capacity",),
                )
            ]).save(bank_path)

            reloaded = translator.select_rules(
                "caller visible capacity must not be recomputed",
                rules=translator.rules,
                limit=1,
            )

        self.assertEqual([rule.id for rule in initial], ["fixed-c-string"])
        self.assertEqual([rule.id for rule in reloaded], ["capacity-contract"])
        self.assertEqual([rule.id for rule in translator.rules], ["capacity-contract"])
        self.assertIn("reloaded", translator.rule_retrieval_status)

    def test_json_bank_prompt_retrieval_uses_interaction_scorer_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bank_path = Path(tmpdir) / "bank.json"
            RuleBank([
                RuleRecord(
                    id="anchor",
                    title="Anchor",
                    trigger="shared marker",
                    goal="shared marker",
                    constraint="shared marker",
                    risk="shared marker",
                    tags=("shared",),
                    keywords=("shared_marker",),
                ),
                RuleRecord(
                    id="support",
                    title="Support",
                    trigger="shared marker",
                    goal="shared marker",
                    constraint="shared marker",
                    risk="shared marker",
                    tags=("shared",),
                    keywords=("shared_marker",),
                ),
                RuleRecord(
                    id="other",
                    title="Other",
                    trigger="shared marker",
                    goal="shared marker",
                    constraint="shared marker",
                    risk="shared marker",
                    tags=("shared",),
                    keywords=("shared_marker",),
                ),
            ]).save(bank_path)
            translator = Translator.__new__(Translator)
            translator._configure_rules(
                {
                    "rules_json_bank_path": str(bank_path),
                    "rule_retrieval_backend_preference": "keyword",
                }
            )

            def fake_agent(payload):
                ids = {payload["first"]["id"], payload["second"]["id"]}
                if ids == {"anchor", "support"}:
                    return {"support": 1.0, "conflict": 0.0, "explanation": "json bank support"}
                return {"support": 0.0, "conflict": 1.0, "explanation": "json bank conflict"}

            translator.rule_interaction_scorer = StructuredInteractionScorer(fake_agent).score

            rules = translator._select_rule_context(output="shared_marker", limit=2)

        self.assertEqual([item["id"] for item in rules], ["anchor", "support"])
        support_metadata = rules[1]["selection_interactions"]
        self.assertEqual(support_metadata[0]["first_id"], "support")
        self.assertEqual(support_metadata[0]["second_id"], "anchor")
        self.assertEqual(support_metadata[0]["explanation"], "json bank support")

    def test_json_bank_prompt_retrieval_uses_relation_records_as_selection_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bank_path = Path(tmpdir) / "bank.json"
            RuleBank(
                [
                    RuleRecord(
                        id="anchor",
                        title="Anchor",
                        trigger="shared marker",
                        goal="shared marker",
                        constraint="shared marker",
                        risk="shared marker",
                        tags=("shared",),
                        keywords=("shared_marker",),
                    ),
                    RuleRecord(
                        id="support",
                        title="Support",
                        trigger="shared marker",
                        goal="shared marker",
                        constraint="shared marker",
                        risk="shared marker",
                        tags=("shared",),
                        keywords=("shared_marker",),
                    ),
                    RuleRecord(
                        id="other",
                        title="Other",
                        trigger="shared marker",
                        goal="shared marker",
                        constraint="shared marker",
                        risk="shared marker",
                        tags=("shared",),
                        keywords=("shared_marker",),
                    ),
                ],
                relations=[
                    RuleRelationRecord(
                        first_id="anchor",
                        second_id="support",
                        relation="supports",
                        rationale="composition evidence",
                        evidence="challenge replay passed",
                    )
                ],
            ).save(bank_path)
            translator = Translator.__new__(Translator)
            translator._configure_rules(
                {
                    "rules_json_bank_path": str(bank_path),
                    "rule_retrieval_backend_preference": "keyword",
                }
            )

            rules = translator._select_rule_context(output="shared_marker", limit=2)

        self.assertEqual([item["id"] for item in rules], ["anchor", "support"])
        support_metadata = rules[1]["selection_interactions"]
        self.assertEqual(support_metadata[0]["first_id"], "support")
        self.assertEqual(support_metadata[0]["second_id"], "anchor")
        self.assertEqual(support_metadata[0]["relationship"], "support")
        self.assertIn("bank relation supports", support_metadata[0]["explanation"])

    def test_json_bank_prompt_retrieval_preserves_record_version_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bank_path = Path(tmpdir) / "bank.json"
            RuleBank(
                [
                    RuleRecord(
                        id="versioned",
                        title="Versioned",
                        trigger="version marker",
                        goal="version marker",
                        constraint="version marker",
                        risk="version marker",
                        tags=("version",),
                        keywords=("version_marker",),
                        version=4,
                        source="trace",
                        parents=("versioned@3",),
                    )
                ]
            ).save(bank_path)
            translator = Translator.__new__(Translator)
            translator._configure_rules(
                {
                    "rules_json_bank_path": str(bank_path),
                    "rule_retrieval_backend_preference": "keyword",
                }
            )

            rules = translator._select_rule_context(output="version_marker", limit=1)

        self.assertEqual(rules[0]["id"], "versioned")
        self.assertEqual(rules[0]["version"], 4)
        self.assertEqual(rules[0]["source"], "trace")
        self.assertEqual(rules[0]["parents"], ["versioned@3"])

    def test_configure_rules_can_enable_live_interaction_scorer(self):
        translator = make_translator(
            agent_payload='{"support": 1.0, "conflict": 0.0, "explanation": "live support"}'
        )

        translator._configure_rules({"rule_interaction_scorer": "live"})
        translator.rules = (
            TranslationRule(
                id="anchor",
                title="Anchor",
                trigger="shared marker",
                goal="shared marker",
                constraint="shared marker",
                risk="shared marker",
                guidance="",
                tags=("shared",),
                keywords=("shared_marker",),
            ),
            TranslationRule(
                id="support",
                title="Support",
                trigger="shared marker",
                goal="shared marker",
                constraint="shared marker",
                risk="shared marker",
                guidance="",
                tags=("shared",),
                keywords=("shared_marker",),
            ),
            TranslationRule(
                id="other",
                title="Other",
                trigger="shared marker",
                goal="shared marker",
                constraint="shared marker",
                risk="shared marker",
                guidance="",
                tags=("shared",),
                keywords=("shared_marker",),
            ),
        )

        rules = translator._select_rule_context(output="shared_marker", limit=2)

        self.assertIsInstance(translator.rule_interaction_scorer.__self__, LLMRuleInteractionScorer)
        self.assertEqual([item["id"] for item in rules], ["anchor", "support"])
        self.assertEqual(rules[1]["selection_interactions"][0]["explanation"], "live support")
        self.assertGreaterEqual(len(translator.agent.calls), 1)
        scorer_calls = [
            call for call in translator.llm_calls
            if call["phase"] == "rule_interaction_scoring"
        ]
        self.assertGreaterEqual(len(scorer_calls), 1)
        self.assertIn("anchor", scorer_calls[0]["prompt"])
        self.assertEqual(scorer_calls[0]["usage"]["total_tokens"], 2)

    def test_live_interaction_scorer_records_llm_phase_when_candidate_pool_fits_limit(self):
        translator = make_translator(
            agent_payload='{"support": 1.0, "conflict": 0.0, "explanation": "small pool support"}'
        )

        translator._configure_rules({"rule_interaction_scorer": "live"})
        translator.rules = (
            TranslationRule(
                id="anchor",
                title="Anchor",
                trigger="shared marker",
                goal="shared marker",
                constraint="shared marker",
                risk="shared marker",
                guidance="",
                tags=("shared",),
                keywords=("shared_marker",),
            ),
            TranslationRule(
                id="support",
                title="Support",
                trigger="shared marker",
                goal="shared marker",
                constraint="shared marker",
                risk="shared marker",
                guidance="",
                tags=("shared",),
                keywords=("shared_marker",),
            ),
        )

        rules = translator._select_rule_context(output="shared_marker", limit=2)

        self.assertEqual([item["id"] for item in rules], ["anchor", "support"])
        self.assertEqual(rules[1]["selection_interactions"][0]["explanation"], "small pool support")
        scorer_calls = [
            call for call in translator.llm_calls
            if call["phase"] == "rule_interaction_scoring"
        ]
        self.assertEqual(len(scorer_calls), 1)
        self.assertEqual(scorer_calls[0]["first_id"], "support")
        self.assertEqual(scorer_calls[0]["second_id"], "anchor")

    def test_select_rule_context_attaches_pre_translation_challenge_when_enabled(self):
        translator = make_translator()
        record = RuleRecord(
            id="capacity-contract",
            title="Capacity Contract",
            trigger="C parser stores count and capacity separately",
            goal="Preserve caller-visible capacity",
            constraint="Do not collapse capacity to element count",
            risk="Capacity drift can break callers",
            tags=("capacity",),
            keywords=("capacity",),
        )
        translator.rules = (record,)
        translator.select_rules = lambda text, rules, limit=5: list(rules)[:limit]
        class FakePreTranslationChecker:
            def __init__(self):
                self.sources = []

            def check(self, challenge, *, rust_source=None):
                self.sources.append(rust_source)
                return replace(
                    challenge,
                    evidence=ExecutionEvidence(
                        compiles=True,
                        tests_pass=True,
                        output="reference replay ok",
                    ),
                )

        checker = FakePreTranslationChecker()
        translator._pre_translation_challenge_tool_checker = checker
        translator.config = {
            "online_adversarial_rules": True,
            "adversarial_learning_options": {
                "run_synthetic_challenges": True,
                "live_structured_agents": False,
            },
        }

        rules = translator._select_rule_context(contract="capacity", limit=1)

        self.assertEqual(len(rules), 1)
        challenge = rules[0]["adversarial_challenge"]
        self.assertEqual(challenge["rule_id"], "capacity-contract")
        self.assertIn("translate_challenge", challenge["rust_interface"])
        self.assertIn("count/capacity", challenge["oracle"])
        self.assertTrue(challenge["reference_replay"])
        self.assertEqual(
            challenge["tool_evidence"],
            {
                "compiles": True,
                "tests_pass": True,
                "output": "reference replay ok",
            },
        )
        self.assertEqual(len(checker.sources), 1)
        self.assertIn("translate_challenge", checker.sources[0])

    def test_pre_translation_challenge_can_run_without_online_update_when_prompt_time_enabled(self):
        translator = make_translator()
        record = RuleRecord(
            id="capacity-contract",
            title="Capacity Contract",
            trigger="C parser stores count and capacity separately",
            goal="Preserve caller-visible capacity",
            constraint="Do not collapse capacity to element count",
            risk="Capacity drift can break callers",
            tags=("capacity",),
            keywords=("capacity",),
        )
        translator.rules = (record,)
        translator.select_rules = lambda text, rules, limit=5: list(rules)[:limit]

        class FakePreTranslationChecker:
            def check(self, challenge, *, rust_source=None):
                return replace(
                    challenge,
                    evidence=ExecutionEvidence(
                        compiles=True,
                        tests_pass=True,
                        output="prompt-time replay ok",
                    ),
                )

        translator._pre_translation_challenge_tool_checker = FakePreTranslationChecker()
        translator.config = {
            "online_adversarial_rules": False,
            "adversarial_learning_options": {
                "pre_translation_challenges": True,
            },
        }

        rules = translator._select_rule_context(contract="capacity", limit=1)

        self.assertEqual(rules[0]["adversarial_challenge"]["rule_id"], "capacity-contract")
        self.assertEqual(
            rules[0]["adversarial_challenge"]["tool_evidence"]["output"],
            "prompt-time replay ok",
        )

    def test_pre_translation_challenge_can_attach_live_translator_replay_evidence(self):
        translator = make_translator()
        record = RuleRecord(
            id="capacity-contract",
            title="Capacity Contract",
            trigger="C parser stores count and capacity separately",
            goal="Preserve caller-visible capacity",
            constraint="Do not collapse capacity to element count",
            risk="Capacity drift can break callers",
            tags=("capacity",),
            keywords=("capacity",),
        )
        translator.rules = (record,)
        translator.select_rules = lambda text, rules, limit=5: list(rules)[:limit]

        class FakePreTranslationChecker:
            def __init__(self):
                self.sources = []

            def check(self, challenge, *, rust_source=None):
                self.sources.append(rust_source)
                output = (
                    "translator replay ok"
                    if rust_source == "translator generated source"
                    else "reference replay ok"
                )
                return replace(
                    challenge,
                    evidence=ExecutionEvidence(
                        compiles=True,
                        tests_pass=True,
                        output=output,
                    ),
                )

        checker = FakePreTranslationChecker()
        translator._pre_translation_challenge_tool_checker = checker
        translator.translate_adversarial_challenge = lambda challenge, rule: "translator generated source"
        translator.config = {
            "online_adversarial_rules": True,
            "adversarial_learning_options": {
                "run_synthetic_challenges": True,
                "live_structured_agents": False,
                "live_translator_replay": True,
            },
        }

        rules = translator._select_rule_context(contract="capacity", limit=1)

        challenge = rules[0]["adversarial_challenge"]
        self.assertEqual(
            challenge["translator_replay"],
            {
                "compiles": True,
                "tests_pass": True,
                "output": "translator replay ok",
            },
        )
        self.assertEqual(checker.sources[-1], "translator generated source")

    def test_pre_translation_challenge_records_semantic_validator_metadata(self):
        translator = make_translator()
        record = RuleRecord(
            id="capacity-contract",
            title="Capacity Contract",
            trigger="C parser stores count and capacity separately",
            goal="Preserve caller-visible capacity",
            constraint="Do not collapse capacity to element count",
            risk="Capacity drift can break callers",
            tags=("capacity",),
            keywords=("capacity",),
        )
        translator.rules = (record,)
        translator.select_rules = lambda text, rules, limit=5: list(rules)[:limit]

        class FakePreTranslationChecker:
            def check(self, challenge, *, rust_source=None):
                return replace(
                    challenge,
                    evidence=ExecutionEvidence(
                        compiles=True,
                        tests_pass=True,
                        output="reference replay ok",
                    ),
                )

        class RecordingActivationValidator:
            def __init__(self):
                self.local_activation_values = []

            def validate(self, challenge, rule, *, local_activation_valid):
                self.local_activation_values.append(local_activation_valid)
                return SemanticActivationValidationResult(
                    activates=True,
                    reason="capacity transition is exercised",
                )

        class RejectingOracleValidator:
            def __init__(self):
                self.calls = 0

            def validate(self, challenge, rule):
                self.calls += 1
                return SemanticOracleValidationResult(
                    valid=False,
                    reason="oracle omits caller-visible capacity",
                )

        activation_validator = RecordingActivationValidator()
        oracle_validator = RejectingOracleValidator()
        translator._pre_translation_challenge_tool_checker = FakePreTranslationChecker()
        translator.config = {
            "online_adversarial_rules": True,
            "adversarial_learning_options": {
                "run_synthetic_challenges": True,
                "activation_validator": activation_validator,
                "oracle_validator": oracle_validator,
            },
        }

        rules = translator._select_rule_context(contract="capacity", limit=1)

        semantic_validation = rules[0]["adversarial_challenge"]["semantic_validation"]
        self.assertFalse(semantic_validation["valid"])
        self.assertEqual(semantic_validation["failed_checks"], ["oracle_semantic"])
        self.assertEqual(
            semantic_validation["details"]["activation"],
            {
                "activates": True,
                "reason": "capacity transition is exercised",
                "local_activation_valid": True,
            },
        )
        self.assertEqual(
            semantic_validation["details"]["oracle"],
            {
                "valid": False,
                "reason": "oracle omits caller-visible capacity",
            },
        )
        self.assertEqual(activation_validator.local_activation_values, [True])
        self.assertEqual(oracle_validator.calls, 1)

    def test_pre_translation_challenge_builds_live_semantic_validators_from_options(self):
        translator = make_translator()
        translator.agent = SequenceAgent(
            [
                '{"activates": true, "reason": "live activation accepts capacity"}',
                '{"valid": true, "reason": "live oracle covers capacity"}',
            ]
        )
        record = RuleRecord(
            id="capacity-contract",
            title="Capacity Contract",
            trigger="C parser stores count and capacity separately",
            goal="Preserve caller-visible capacity",
            constraint="Do not collapse capacity to element count",
            risk="Capacity drift can break callers",
            tags=("capacity",),
            keywords=("capacity",),
        )
        translator.rules = (record,)
        translator.select_rules = lambda text, rules, limit=5: list(rules)[:limit]

        class FakePreTranslationChecker:
            def check(self, challenge, *, rust_source=None):
                return replace(
                    challenge,
                    evidence=ExecutionEvidence(
                        compiles=True,
                        tests_pass=True,
                        output="reference replay ok",
                    ),
                )

        translator._pre_translation_challenge_tool_checker = FakePreTranslationChecker()
        translator.config = {
            "online_adversarial_rules": True,
            "adversarial_learning_options": {
                "run_synthetic_challenges": True,
                "live_activation_validation": True,
                "live_semantic_oracle_validation": True,
            },
        }

        rules = translator._select_rule_context(contract="capacity", limit=1)

        semantic_validation = rules[0]["adversarial_challenge"]["semantic_validation"]
        self.assertTrue(semantic_validation["valid"])
        self.assertEqual(semantic_validation["failed_checks"], [])
        self.assertEqual(
            semantic_validation["details"]["activation"]["reason"],
            "live activation accepts capacity",
        )
        self.assertEqual(
            semantic_validation["details"]["oracle"]["reason"],
            "live oracle covers capacity",
        )
        self.assertEqual(len(translator.agent.calls), 2)
        self.assertEqual(
            [call["phase"] for call in translator.llm_calls],
            ["activation_validation", "semantic_oracle_validation"],
        )

    def test_pre_translation_refinement_updates_prompt_rule_when_replay_gate_accepts(self):
        translator = make_translator()
        record = RuleRecord(
            id="capacity-contract",
            title="Capacity Contract",
            trigger="C parser stores count and capacity separately",
            goal="Preserve caller-visible capacity",
            constraint="Do not collapse capacity to element count",
            risk="Capacity drift can break callers",
            tags=("capacity",),
            keywords=("capacity",),
        )
        translator.rules = (record,)
        translator.select_rules = lambda text, rules, limit=5: list(rules)[:limit]

        class FakePreTranslationChecker:
            def __init__(self):
                self.sources = []

            def check(self, challenge, *, rust_source=None):
                self.sources.append(rust_source)
                if rust_source == "before source":
                    evidence = ExecutionEvidence(
                        compiles=True,
                        tests_pass=False,
                        output="assertion failed: capacity collapsed",
                    )
                elif rust_source == "after source":
                    evidence = ExecutionEvidence(
                        compiles=True,
                        tests_pass=True,
                        output="refined replay ok",
                    )
                else:
                    evidence = ExecutionEvidence(
                        compiles=True,
                        tests_pass=True,
                        output="reference replay ok",
                    )
                return replace(challenge, evidence=evidence)

        class FakeRefiner:
            def __init__(self):
                self.evidence_outputs = []

            def propose_update(self, rule, challenge, evidence):
                self.evidence_outputs.append(evidence.output)
                return RuleUpdateProposal(
                    rule_id=rule.id,
                    field="constraint",
                    old_text=rule.constraint,
                    new_text=rule.constraint + "; require caller-visible capacity",
                    rationale="constraint_violation: replay showed collapsed capacity",
                )

        checker = FakePreTranslationChecker()
        refiner = FakeRefiner()
        translator._pre_translation_challenge_tool_checker = checker
        translator.translate_adversarial_challenge = (
            lambda challenge, rule: "after source"
            if "caller-visible capacity" in rule.constraint
            else "before source"
        )
        translator.config = {
            "online_adversarial_rules": True,
            "adversarial_learning_options": {
                "run_synthetic_challenges": True,
                "pre_translation_refinement": True,
                "refiner_agent": refiner,
            },
        }

        rules = translator._select_rule_context(contract="capacity", limit=1)

        refined = rules[0]
        self.assertEqual(
            refined["constraint"],
            "Do not collapse capacity to element count; require caller-visible capacity",
        )
        self.assertEqual(refined["version"], 2)
        self.assertIn("capacity-contract@v1", refined["parents"])
        self.assertEqual(refiner.evidence_outputs, ["assertion failed: capacity collapsed"])
        self.assertEqual(
            refined["pre_translation_refinement"]["fresh_replay_evidence"],
            {
                "before": {
                    "compiles": True,
                    "tests_pass": False,
                    "output": "assertion failed: capacity collapsed",
                },
                "after": {
                    "compiles": True,
                    "tests_pass": True,
                    "output": "refined replay ok",
                },
            },
        )
        self.assertTrue(refined["pre_translation_refinement"]["accepted"])
        self.assertEqual(refined["pre_translation_refinement"]["field"], "constraint")
        self.assertIn("adversarial_challenge", refined)

    def test_pre_translation_refinement_can_apply_multiple_accepted_rounds(self):
        translator = make_translator()
        record = RuleRecord(
            id="capacity-contract",
            title="Capacity Contract",
            trigger="C parser stores count and capacity separately",
            goal="Preserve caller-visible capacity",
            constraint="Do not collapse capacity to element count",
            risk="Capacity drift can break callers",
            tags=("capacity",),
            keywords=("capacity",),
        )
        translator.rules = (record,)
        translator.select_rules = lambda text, rules, limit=5: list(rules)[:limit]

        class FakePreTranslationChecker:
            def check(self, challenge, *, rust_source=None):
                evidence_by_source = {
                    "before round 1": ExecutionEvidence(
                        compiles=True,
                        tests_pass=False,
                        output="assertion failed: capacity collapsed",
                    ),
                    "after round 1": ExecutionEvidence(
                        compiles=True,
                        tests_pass=True,
                        output="round 1 replay ok",
                    ),
                    "before round 2": ExecutionEvidence(
                        compiles=True,
                        tests_pass=False,
                        output="assertion failed: configured capacity hidden",
                    ),
                    "after round 2": ExecutionEvidence(
                        compiles=True,
                        tests_pass=True,
                        output="round 2 replay ok",
                    ),
                }
                evidence = evidence_by_source.get(
                    rust_source,
                    ExecutionEvidence(
                        compiles=True,
                        tests_pass=True,
                        output="reference replay ok",
                    ),
                )
                return replace(challenge, evidence=evidence)

        class FakeRefiner:
            def __init__(self):
                self.evidence_outputs = []

            def propose_update(self, rule, challenge, evidence):
                self.evidence_outputs.append(evidence.output)
                if len(self.evidence_outputs) == 1:
                    return RuleUpdateProposal(
                        rule_id=rule.id,
                        field="constraint",
                        old_text=rule.constraint,
                        new_text=rule.constraint + "; require caller-visible capacity",
                        rationale="constraint_violation: replay showed collapsed capacity",
                    )
                return RuleUpdateProposal(
                    rule_id=rule.id,
                    field="risk",
                    old_text=rule.risk,
                    new_text=rule.risk + "; configured capacity hidden",
                    rationale="missing_risk: replay showed hidden configured capacity",
                )

        checker = FakePreTranslationChecker()
        refiner = FakeRefiner()
        translated_sources = iter(
            [
                "before round 1",
                "after round 1",
                "before round 2",
                "after round 2",
            ]
        )
        translator._pre_translation_challenge_tool_checker = checker
        translator.translate_adversarial_challenge = lambda challenge, rule: next(translated_sources)
        translator.config = {
            "online_adversarial_rules": True,
            "adversarial_learning_options": {
                "run_synthetic_challenges": True,
                "pre_translation_refinement": True,
                "pre_translation_refinement_rounds": 2,
                "refiner_agent": refiner,
            },
        }

        rules = translator._select_rule_context(contract="capacity", limit=1)

        refined = rules[0]
        self.assertEqual(refined["version"], 3)
        self.assertIn("caller-visible capacity", refined["constraint"])
        self.assertIn("configured capacity hidden", refined["risk"])
        self.assertEqual(
            refiner.evidence_outputs,
            [
                "assertion failed: capacity collapsed",
                "assertion failed: configured capacity hidden",
            ],
        )
        rounds = refined["pre_translation_refinement"]["rounds"]
        self.assertEqual([item["round"] for item in rounds], [1, 2])
        self.assertEqual([item["version"] for item in rounds], [2, 3])
        self.assertTrue(all(item["accepted"] for item in rounds))
        self.assertEqual(refined["pre_translation_refinement"]["accepted_rounds"], 2)
        self.assertTrue(refined["pre_translation_refinement"]["accepted"])
        self.assertEqual(refined["pre_translation_refinement"]["version"], 3)

    def test_pre_translation_refinement_can_use_heuristic_fallback_refiner(self):
        translator = make_translator()
        record = RuleRecord(
            id="capacity-contract",
            title="Capacity Contract",
            trigger="C parser stores count and capacity separately",
            goal="Preserve caller-visible capacity",
            constraint="Do not collapse capacity to element count",
            risk="Capacity drift can break callers",
            tags=("capacity",),
            keywords=("capacity",),
        )
        translator.rules = (record,)
        translator.select_rules = lambda text, rules, limit=5: list(rules)[:limit]

        class FakePreTranslationChecker:
            def check(self, challenge, *, rust_source=None):
                if rust_source == "before source":
                    evidence = ExecutionEvidence(
                        compiles=True,
                        tests_pass=False,
                        output="assertion failed: capacity collapsed",
                    )
                elif rust_source == "after source":
                    evidence = ExecutionEvidence(
                        compiles=True,
                        tests_pass=True,
                        output="heuristic refined replay ok",
                    )
                else:
                    evidence = ExecutionEvidence(
                        compiles=True,
                        tests_pass=True,
                        output="reference replay ok",
                    )
                return replace(challenge, evidence=evidence)

        translator._pre_translation_challenge_tool_checker = FakePreTranslationChecker()
        translator.translate_adversarial_challenge = (
            lambda challenge, rule: "after source"
            if "capacity collapsed" in rule.risk
            else "before source"
        )
        translator.config = {
            "online_adversarial_rules": True,
            "adversarial_learning_options": {
                "run_synthetic_challenges": True,
                "pre_translation_refinement": True,
            },
        }

        rules = translator._select_rule_context(contract="capacity", limit=1)

        refined = rules[0]
        self.assertIn("capacity collapsed", refined["risk"])
        self.assertTrue(refined["pre_translation_refinement"]["accepted"])
        self.assertEqual(refined["pre_translation_refinement"]["field"], "risk")

    def test_pre_translation_refinement_writeback_updates_active_json_bank(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bank_path = Path(tmpdir) / "bank.json"
            record = RuleRecord(
                id="capacity-contract",
                title="Capacity Contract",
                trigger="C parser stores count and capacity separately",
                goal="Preserve caller-visible capacity",
                constraint="Do not collapse capacity to element count",
                risk="Capacity drift can break callers",
                tags=("capacity",),
                keywords=("capacity",),
            )
            RuleBank([record]).save(bank_path)
            translator = make_translator()
            config = {
                "rules_json_bank_path": str(bank_path),
                "rule_retrieval_backend_preference": "keyword",
                "online_adversarial_rules": True,
                "adversarial_learning_options": {
                    "run_synthetic_challenges": True,
                    "pre_translation_refinement": True,
                    "pre_translation_refinement_writeback": True,
                },
            }
            translator.config = config
            translator._configure_rules(config)

            class FakePreTranslationChecker:
                def check(self, challenge, *, rust_source=None):
                    if rust_source == "before source":
                        evidence = ExecutionEvidence(
                            compiles=True,
                            tests_pass=False,
                            output="assertion failed: capacity collapsed",
                        )
                    elif rust_source == "after source":
                        evidence = ExecutionEvidence(
                            compiles=True,
                            tests_pass=True,
                            output="writeback replay ok",
                        )
                    else:
                        evidence = ExecutionEvidence(
                            compiles=True,
                            tests_pass=True,
                            output="reference replay ok",
                        )
                    return replace(challenge, evidence=evidence)

            translator._pre_translation_challenge_tool_checker = FakePreTranslationChecker()
            translator.translate_adversarial_challenge = (
                lambda challenge, rule: "after source"
                if "capacity collapsed" in rule.risk
                else "before source"
            )

            with mock.patch("c2rust_translator.adversarial_rules.fcntl.flock") as flock:
                rules = translator._select_rule_context(contract="capacity", limit=1)
            persisted = RuleBank.load(bank_path).records[0]

        self.assertTrue(rules[0]["pre_translation_refinement"]["accepted"])
        self.assertEqual(rules[0]["pre_translation_refinement"]["writeback"], "updated")
        self.assertEqual(
            [call.args[1] for call in flock.call_args_list],
            [
                adversarial_core.fcntl.LOCK_EX,
                adversarial_core.fcntl.LOCK_UN,
            ],
        )
        self.assertEqual(persisted.version, 2)
        self.assertIn("capacity collapsed", persisted.risk)
        self.assertEqual([rule.version for rule in translator.rules], [2])


def make_callable(name, *, module="mod", kind="fn", type_name=None):
    return Callable(
        module=module,
        kind=kind,
        name=name,
        type_name=type_name,
        file_name=f"{module}.rs",
        signature=f"pub fn {name}()",
        signature_key=re.sub(r"\s+", "", f"pub fn {name}()"),
        param_count=0,
        stub=f"pub fn {name}() {{ unimplemented!() }}",
    )


def make_translator(*, mode="rule", agent_payload="{}"):
    translator = Translator.__new__(Translator)
    translator.alignment_mode = mode
    translator.alignment_confidence_threshold = 0.65
    translator.agent = FakeAgent(agent_payload)
    translator.metadata = {"files": {}}
    translator.call_deps = {}
    translator.c_functions = {}
    translator.c_bindings = {}
    translator.callables = {}
    translator.cycle_peers = {}
    translator.callable_categories = {}
    translator.file_contexts = {}
    translator.alignment_prompts = []
    translator.alignment_stats = {}
    translator.translation_history = {}
    translator.translation_log = []
    translator.translated_modules = set()
    translator.translated_code = {}
    translator.max_fix_attempts = 5
    translator.max_test_fix_attempts = 3
    translator.test_fix_history = []
    translator.callable_failures = 0
    translator.rules = DEFAULT_RULES
    translator.select_rules = select_rules
    return translator


def make_c_function(translator, name, *, file_path="x.c", calls=()):
    return CFunction(
        name=name,
        name_key=translator._name_key(name),
        name_tokens=tuple(translator._name_tokens(name)),
        file_path=file_path,
        original_file=file_path,
        code=f"int {name}(void) {{ return 0; }}",
        file_key=translator._file_key(file_path),
        original_key=translator._file_key(file_path),
        param_count=0,
        calls=tuple(calls),
    )


def c_index(*functions):
    out = {}
    for function in functions:
        out.setdefault(function.name_key, []).append(function)
    return out


class TranslatorCallableOutputTest(unittest.TestCase):
    def test_callable_output_rejects_new_extern_ffi_block(self):
        translator = make_translator()
        item = make_callable("deal_from_deck")

        issue = translator._callable_output_issue(
            """
            pub fn deal_from_deck() {
                unsafe extern "C" {
                    fn lrand48() -> i64;
                }
            }
            """,
            item,
        )

        self.assertEqual(issue, "forbidden_ffi_declaration")

    def test_translated_deps_use_package_path_from_binary_main(self):
        translator = make_translator()
        current = ("main", "fn", "init_repository")
        dep = ("vm", "fn", "load_author")
        translator.call_deps = {current: {dep}}
        translator.translated_modules = {dep}
        translator.translated_code = {dep: "pub fn load_author() -> Author { unimplemented!() }"}
        translator.translation_history = {dep: {"c_function": "load_author"}}

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            (src_dir / "lib.rs").write_text("pub mod vm;\n", encoding="utf-8")
            (Path(tmpdir) / "Cargo.toml").write_text(
                '[package]\nname = "VaultSync"\nversion = "0.1.0"\n',
                encoding="utf-8",
            )

            deps = translator._translated_deps(current)

        self.assertEqual(deps[0]["rust_path"], "VaultSync::vm::load_author")

    def test_translated_deps_include_dependency_signature(self):
        translator = make_translator()
        current = ("main", "fn", "init_repository")
        dep = ("vm", "fn", "init_repo")
        translator.call_deps = {current: {dep}}
        translator.translated_modules = {dep}
        translator.translated_code = {dep: "pub fn init_repo(author: &Author) -> Result<Repository, String> { unimplemented!() }"}
        translator.translation_history = {dep: {"c_function": "init_repo"}}
        translator.callables = {
            dep: Callable(
                module="vm",
                kind="fn",
                name="init_repo",
                type_name=None,
                file_name="vm.rs",
                signature="pub fn init_repo(author: &Author) -> Result<Repository, String>",
                signature_key=re.sub(r"\s+", "", "pub fn init_repo(author: &Author) -> Result<Repository, String>"),
                param_count=1,
                stub="pub fn init_repo(author: &Author) -> Result<Repository, String> { unimplemented!() }",
            )
        }

        deps = translator._translated_deps(current)

        self.assertEqual(deps[0]["signature"], "pub fn init_repo(author: &Author) -> Result<Repository, String>")


class TranslatorAlignmentTest(unittest.TestCase):
    def test_rule_mode_projects_legacy_choice_into_c_binding(self):
        translator = make_translator(mode="rule")
        rust_item = make_callable("alpha")
        c_alpha = make_c_function(translator, "alpha")

        ordered, edge_count, cycles = translator._plan([rust_item], c_index(c_alpha))

        item_id = ("mod", "fn", "alpha")
        self.assertEqual(ordered, [rust_item])
        self.assertEqual(edge_count, 0)
        self.assertEqual(cycles, [])
        self.assertEqual(translator.c_bindings[item_id].functions, (c_alpha,))
        self.assertIs(translator.c_functions[item_id], c_alpha)

    def test_llm_mode_uses_all_bound_functions_for_dependencies(self):
        payload = json.dumps(
            {
                "bindings": [
                    {
                        "rust_callable": "mod::fn::leaf",
                        "c_functions": ["x.c::leaf"],
                        "relationship": "direct",
                        "confidence": 0.95,
                        "rationale": "leaf maps directly",
                    },
                    {
                        "rust_callable": "mod::fn::combo",
                        "c_functions": ["x.c::first", "x.c::second"],
                        "relationship": "composed_from_multiple",
                        "confidence": 0.90,
                        "rationale": "combo composes helpers",
                    },
                ]
            }
        )
        translator = make_translator(mode="llm", agent_payload=payload)
        leaf = make_callable("leaf")
        combo = make_callable("combo")
        c_leaf = make_c_function(translator, "leaf")
        c_first = make_c_function(translator, "first", calls=("leaf",))
        c_second = make_c_function(translator, "second")

        ordered, edge_count, cycles = translator._plan([combo, leaf], c_index(c_leaf, c_first, c_second))

        combo_id = ("mod", "fn", "combo")
        leaf_id = ("mod", "fn", "leaf")
        self.assertEqual(ordered, [leaf, combo])
        self.assertEqual(edge_count, 1)
        self.assertEqual(cycles, [])
        self.assertEqual(translator.call_deps[combo_id], {leaf_id})
        self.assertEqual([fn.name for fn in translator.c_bindings[combo_id].functions], ["first", "second"])

    def test_llm_alignment_limits_large_function_inventory_to_relevant_candidates(self):
        translator = make_translator(mode="llm", agent_payload='{"bindings": []}')
        rust_item = make_callable("target")
        c_target = make_c_function(translator, "target", file_path="mod.c")
        noise = [
            make_c_function(translator, f"noise_{index}", file_path=f"vendor_{index}.c")
            for index in range(Translator.ALIGNMENT_FULL_FUNCTION_LIMIT + 10)
        ]

        ordered, _edge_count, _cycles = translator._plan([rust_item], c_index(c_target, *noise))

        prompt = translator.alignment_prompts[0]["prompt"]
        self.assertEqual(ordered, [rust_item])
        self.assertIn("mod.c::target", prompt)
        self.assertNotIn("vendor_0.c::noise_0", prompt)
        self.assertLess(len(prompt), 20000)

    def test_shared_c_function_binding_skips_ambiguous_dependency_without_crashing(self):
        payload = json.dumps(
            {
                "bindings": [
                    {
                        "rust_callable": "mod::fn::left",
                        "c_functions": ["x.c::shared"],
                        "relationship": "split_from_one",
                        "confidence": 0.90,
                        "rationale": "left uses part of shared",
                    },
                    {
                        "rust_callable": "mod::fn::right",
                        "c_functions": ["x.c::shared"],
                        "relationship": "split_from_one",
                        "confidence": 0.90,
                        "rationale": "right uses part of shared",
                    },
                    {
                        "rust_callable": "mod::fn::caller",
                        "c_functions": ["x.c::caller"],
                        "relationship": "direct",
                        "confidence": 0.90,
                        "rationale": "caller invokes shared",
                    },
                ]
            }
        )
        translator = make_translator(mode="llm", agent_payload=payload)
        left = make_callable("left")
        right = make_callable("right")
        caller = make_callable("caller")
        c_shared = make_c_function(translator, "shared")
        c_caller = make_c_function(translator, "caller", calls=("shared",))

        ordered, edge_count, cycles = translator._plan([caller, left, right], c_index(c_shared, c_caller))

        self.assertEqual(ordered, [caller, left, right])
        self.assertEqual(edge_count, 0)
        self.assertEqual(cycles, [])
        self.assertEqual(translator.call_deps[("mod", "fn", "caller")], set())

    def test_save_alignment_artifacts_writes_plan_and_prompts(self):
        translator = make_translator(mode="llm")
        c_alpha = make_c_function(translator, "alpha")
        translator.c_bindings = {
            ("mod", "fn", "alpha"): CBinding(
                functions=(c_alpha,),
                relationship="direct",
                confidence=0.9,
                rationale="direct match",
                source="llm",
            )
        }
        translator.alignment_prompts = [{"module": "mod", "prompt": "prompt", "raw_response": "{}"}]

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            translator._save_alignment_artifacts()

            plan = json.loads((translator.output_dir / "alignment_plan.json").read_text(encoding="utf-8"))
            prompts = json.loads((translator.output_dir / "alignment_prompts.json").read_text(encoding="utf-8"))

        self.assertEqual(plan["mod::fn::alpha"]["relationship"], "direct")
        self.assertEqual(plan["mod::fn::alpha"]["functions"][0]["id"], "x.c::alpha")
        self.assertEqual(prompts[0]["module"], "mod")

    def test_compile_fix_exhaustion_falls_back_to_unimplemented_body(self):
        translator = make_translator(agent_payload="pub fn alpha() { still_bad(); }")
        translator.max_fix_attempts = 1
        translator.validator = FakeValidator(
            [
                (False, "error: cannot find function `bad_initial`"),
                (False, "error: cannot find function `still_bad`"),
                (True, "fallback compiled"),
            ]
        )
        item = make_callable("alpha")
        module_id = translator._callable_id(item)
        translator.translation_history[module_id] = {"compilation_attempts": []}

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            translator.artifacts_dir = Path(tmpdir) / ".translated_modules"
            translator.artifacts_dir.mkdir()
            module_path = Path(tmpdir) / "mod.rs"
            module_path.write_text("pub fn alpha() { bad_initial(); }\n", encoding="utf-8")

            ok, final_code = translator._compile_fix(
                item,
                module_id,
                module_path,
                "pub fn alpha() { bad_initial(); }",
            )

            source = module_path.read_text(encoding="utf-8")

        self.assertTrue(ok)
        self.assertIn("unimplemented!()", source)
        self.assertNotIn("still_bad", source)
        self.assertIn("unimplemented!()", final_code)
        self.assertTrue(translator.translation_history[module_id]["used_unimplemented_fallback"])
        self.assertEqual(translator.translation_history[module_id]["fallback"]["status"], "success")
        self.assertEqual(translator.validator.calls, 3)

    def test_compile_fix_validates_last_repair_before_fallback(self):
        translator = make_translator(agent_payload="pub fn alpha() { fixed(); }")
        translator.max_fix_attempts = 1
        translator.validator = FakeValidator(
            [
                (False, "error: cannot find function `bad_initial`"),
                (True, "last repair compiled"),
            ]
        )
        item = make_callable("alpha")
        module_id = translator._callable_id(item)
        translator.translation_history[module_id] = {"compilation_attempts": []}

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            translator.artifacts_dir = Path(tmpdir) / ".translated_modules"
            translator.artifacts_dir.mkdir()
            module_path = Path(tmpdir) / "mod.rs"
            module_path.write_text("pub fn alpha() { bad_initial(); }\n", encoding="utf-8")

            ok, final_code = translator._compile_fix(
                item,
                module_id,
                module_path,
                "pub fn alpha() { bad_initial(); }",
            )

            source = module_path.read_text(encoding="utf-8")

        self.assertTrue(ok)
        self.assertIn("fixed()", source)
        self.assertNotIn("unimplemented!()", source)
        self.assertIn("fixed()", final_code)
        self.assertNotIn("used_unimplemented_fallback", translator.translation_history[module_id])
        self.assertEqual(translator.validator.calls, 2)

    def test_compile_fix_llm_error_falls_back_to_unimplemented_body(self):
        translator = make_translator()
        translator.agent = FailingAgent(RuntimeError("non-retryable client error"))
        translator.max_fix_attempts = 1
        translator.validator = FakeValidator(
            [
                (False, "error: cannot find function `bad_initial`"),
                (True, "fallback compiled"),
            ]
        )
        item = make_callable("alpha")
        module_id = translator._callable_id(item)
        translator.translation_history[module_id] = {"compilation_attempts": []}

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            translator.artifacts_dir = Path(tmpdir) / ".translated_modules"
            translator.artifacts_dir.mkdir()
            module_path = Path(tmpdir) / "mod.rs"
            module_path.write_text("pub fn alpha() { bad_initial(); }\n", encoding="utf-8")

            ok, final_code = translator._compile_fix(
                item,
                module_id,
                module_path,
                "pub fn alpha() { bad_initial(); }",
            )

            source = module_path.read_text(encoding="utf-8")

        attempts = translator.translation_history[module_id]["compilation_attempts"]
        self.assertTrue(ok)
        self.assertIn("unimplemented!()", source)
        self.assertIn("unimplemented!()", final_code)
        self.assertEqual(attempts[0]["status"], "llm_error")
        self.assertIn("non-retryable client error", attempts[0]["llm_error"])
        self.assertTrue(translator.translation_history[module_id]["used_unimplemented_fallback"])

    def test_record_llm_call_and_persist_progress_keeps_function_trace(self):
        translator = make_translator(mode="llm")
        item = make_callable("alpha")
        module_id = translator._callable_id(item)
        dep_id = ("dep", "fn", "helper")
        c_alpha = make_c_function(translator, "alpha")
        translator.project_name = "proj"
        translator.callables = {module_id: item}
        translator.call_deps = {module_id: {dep_id}}
        translator.cycle_peers = {module_id: [("mod", "fn", "peer")]}
        translator.c_bindings = {
            module_id: CBinding(
                functions=(c_alpha,),
                relationship="direct",
                confidence=0.9,
                rationale="direct match",
                source="llm",
            )
        }
        translator.translation_history[module_id] = {
            "module_name": "alpha",
            "module_type": "fn",
            "file_path": "mod",
            "success": False,
            "compilation_attempts": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            translator._record_llm_call(
                module_id,
                item,
                phase="translation",
                prompt="translate alpha",
                result=TranslationResult(
                    code="pub fn alpha() {}",
                    raw_text="```rust\npub fn alpha() {}\n```",
                    elapsed_seconds=0.25,
                    usage=Usage(prompt_tokens=3, completion_tokens=5, total_tokens=8),
                ),
                extracted_code="pub fn alpha() {}",
            )
            translator._persist_progress()

            history = json.loads((Path(tmpdir) / "translation_history.json").read_text(encoding="utf-8"))
            jsonl = (Path(tmpdir) / "llm_calls.jsonl").read_text(encoding="utf-8").strip().splitlines()

        call = history["mod::fn::alpha"]["llm_calls"][0]
        self.assertEqual(call["phase"], "translation")
        self.assertEqual(call["callable_label"], "mod::alpha")
        self.assertEqual(call["dependencies"][0]["id"], "dep::fn::helper")
        self.assertEqual(call["alignment"]["relationship"], "direct")
        self.assertEqual(json.loads(jsonl[0])["raw_text"], "```rust\npub fn alpha() {}\n```")

    def test_translate_one_runs_callable_online_adversarial_update_for_active_bank(self):
        translator = make_translator(agent_payload="pub fn alpha() { repaired(); }")
        translator.project_name = "demo"
        translator.max_fix_attempts = 1
        translator.validator = FakeValidator([(True, "cargo check ok")])
        item = make_callable("alpha")
        module_id = translator._callable_id(item)

        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run"
            project_dir = run_root / "demo"
            bank_path = Path(tmpdir) / "bank.json"
            RuleBank([
                RuleRecord(
                    id="seed-rule",
                    title="Seed Rule",
                    trigger="seed trigger",
                    goal="seed goal",
                    constraint="seed constraint",
                    risk="seed risk",
                    tags=("seed",),
                    keywords=("seed",),
                )
            ]).save(bank_path)
            translator.output_dir = project_dir
            translator.artifacts_dir = project_dir / ".translated_modules"
            translator.artifacts_dir.mkdir(parents=True)
            src_dir = project_dir / "src"
            src_dir.mkdir(parents=True)
            module_path = src_dir / "mod.rs"
            module_path.write_text("pub fn alpha() { unimplemented!() }\n", encoding="utf-8")
            translator.skeleton = type("FakeSkeleton", (), {"module_files": {"mod": "mod.rs"}})()
            translator.config = {
                "online_adversarial_rules": True,
                "rules_json_bank_path": str(bank_path),
                "adversarial_rule_bank_path": str(bank_path),
                "adversarial_rule_output_bank_path": str(bank_path),
                "adversarial_rule_report_path": str(run_root / "adversarial_rule_report.json"),
                "adversarial_learning_options": {
                    "admit_trace_candidates": True,
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
                },
            }
            translator._configure_rules(
                {
                    "rules_json_bank_path": str(bank_path),
                    "rule_retrieval_backend_preference": "keyword",
                }
            )

            def update_bank(run_root_arg, bank_arg, output_bank_arg, report_arg, **kwargs):
                self.assertEqual(Path(run_root_arg), run_root)
                self.assertEqual(Path(bank_arg), bank_path)
                self.assertEqual(Path(output_bank_arg), bank_path)
                self.assertEqual(Path(report_arg), run_root / "adversarial_rule_report.json")
                self.assertTrue(kwargs["admit_trace_candidates"])
                RuleBank([
                    RuleRecord(
                        id="seed-rule",
                        title="Seed Rule",
                        trigger="seed trigger",
                        goal="seed goal",
                        constraint="seed constraint",
                        risk="seed risk",
                        tags=("seed",),
                        keywords=("seed",),
                    ),
                    RuleRecord(
                        id="online-capacity",
                        title="Online Capacity",
                        trigger="online capacity trigger",
                        goal="preserve online capacity",
                        constraint="keep online capacity metadata",
                        risk="online capacity drift",
                        tags=("capacity",),
                        keywords=("onlinecapacity",),
                    ),
                ]).save(output_bank_arg)
                Path(report_arg).write_text('{"accepted_updates": 1}', encoding="utf-8")
                return {"accepted_updates": 1}

            with mock.patch(
                "c2rust_translator.translator.run_offline_adversarial_learning",
                side_effect=update_bank,
                create=True,
            ) as online_update:
                ok = translator._translate_one(item)

            selected = translator._select_rule_context(
                contract="onlinecapacity",
                module_id=module_id,
                limit=1,
            )

        self.assertTrue(ok)
        self.assertEqual(online_update.call_count, 1)
        self.assertEqual(getattr(translator, "online_adversarial_update_count", 0), 1)
        self.assertEqual(selected[0]["id"], "online-capacity")

    def test_online_adversarial_update_refreshes_in_memory_json_bank(self):
        translator = make_translator()
        translator.project_name = "demo"
        module_id = ("mod", "fn", "alpha")

        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run"
            project_dir = run_root / "demo"
            project_dir.mkdir(parents=True)
            bank_path = Path(tmpdir) / "bank.json"
            RuleBank([
                RuleRecord(
                    id="seed-rule",
                    title="Seed Rule",
                    trigger="seed trigger",
                    goal="seed goal",
                    constraint="seed constraint",
                    risk="seed risk",
                )
            ]).save(bank_path)
            translator.output_dir = project_dir
            translator.config = {
                "online_adversarial_rules": True,
                "rules_json_bank_path": str(bank_path),
                "adversarial_rule_bank_path": str(bank_path),
                "adversarial_rule_output_bank_path": str(bank_path),
                "adversarial_rule_report_path": str(run_root / "adversarial_rule_report.json"),
                "adversarial_learning_options": {},
            }
            translator._configure_rules(
                {
                    "rules_json_bank_path": str(bank_path),
                    "rule_retrieval_backend_preference": "keyword",
                }
            )

            def update_bank(_run_root_arg, _bank_arg, output_bank_arg, report_arg, **_kwargs):
                RuleBank([
                    RuleRecord(
                        id="online-new",
                        title="Online New",
                        trigger="online trigger",
                        goal="online goal",
                        constraint="online constraint",
                        risk="online risk",
                    )
                ]).save(output_bank_arg)
                Path(report_arg).write_text('{"accepted_updates": 1}', encoding="utf-8")
                return {"accepted_updates": 1}

            with mock.patch(
                "c2rust_translator.translator.run_offline_adversarial_learning",
                side_effect=update_bank,
                create=True,
            ):
                result = translator._maybe_run_online_adversarial_update(
                    reason="unit-test",
                    module_id=module_id,
                )

            rule_ids = [record.id for record in translator.rules]

        self.assertEqual(result["accepted_updates"], 1)
        self.assertEqual(rule_ids, ["online-new"])
        self.assertEqual(translator.online_adversarial_updates[-1]["json_bank_reload"], "reloaded")

    def test_test_repair_loop_patches_targeted_callable_and_reruns_tests(self):
        payload = json.dumps(
            {
                "fixes": [
                    {
                        "callable_id": "mod::fn::alpha",
                        "rust_code": "pub fn alpha() { repaired(); }",
                    }
                ]
            }
        )
        translator = make_translator(agent_payload=payload)
        translator.project_name = "proj"
        translator.max_test_fix_attempts = 3
        translator.validator = FakeValidator([(True, "cargo check ok")])
        translator.tests = FakeTestValidator(
            [
                (False, "thread 'test_alpha' panicked at src/bin/test_alpha.rs:4:5\nleft: 0\nright: 42", {"total": 1, "passed": 0, "failed": 1}),
                (True, "test result: ok. 1 passed; 0 failed; 0 ignored", {"total": 1, "passed": 1, "failed": 0}),
            ]
        )
        item = make_callable("alpha")
        module_id = translator._callable_id(item)
        c_alpha = make_c_function(translator, "alpha", file_path="alpha.c")
        translator.callables = {module_id: item}
        translator.c_bindings = {
            module_id: CBinding(
                functions=(c_alpha,),
                relationship="direct",
                confidence=0.9,
                rationale="direct match",
                source="llm",
            )
        }
        translator.translation_history[module_id] = {
            "module_name": "alpha",
            "module_type": "fn",
            "file_path": "mod",
            "success": True,
            "compilation_attempts": [],
        }
        translator.translated_modules = {module_id}

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            translator.artifacts_dir = Path(tmpdir) / ".translated_modules"
            translator.artifacts_dir.mkdir()
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            module_path = src_dir / "mod.rs"
            module_path.write_text("pub fn alpha() { broken(); }\n", encoding="utf-8")
            translator.skeleton = type("FakeSkeleton", (), {"module_files": {"mod": "mod.rs"}})()

            ok, output, stats = translator._run_test_repair_loop()
            source = module_path.read_text(encoding="utf-8")
            test_history = json.loads((Path(tmpdir) / "test_fix_history.json").read_text(encoding="utf-8"))

        self.assertTrue(ok)
        self.assertEqual(stats["passed"], 1)
        self.assertEqual(translator.tests.calls, 2)
        self.assertIn("repaired", source)
        self.assertEqual(test_history[0]["status"], "patched")
        self.assertEqual(test_history[0]["fixes"][0]["callable_id"], "mod::fn::alpha")
        self.assertIn("alpha.c", test_history[0]["prompt"])
        self.assertIn("test_alpha", test_history[0]["prompt"])

    def test_test_repair_can_patch_listed_generated_test_source(self):
        payload = json.dumps(
            {
                "test_source_fixes": [
                    {
                        "path": "src/bin/test.rs",
                        "find": "    let observed = decode(value);\n    assert_eq!(observed, x);\n    assert_eq!(observed, y);",
                        "replace": "    let lo = decode(value >> 1);\n    let hi = decode(value);\n    assert_eq!(lo, x);\n    assert_eq!(hi, y);",
                    }
                ]
            }
        )
        translator = make_translator(agent_payload=payload)
        translator.project_name = "proj"
        translator.max_test_fix_attempts = 1
        translator.validator = FakeValidator([(True, "cargo check ok")])
        translator.tests = FakeTestValidator(
            [
                (
                    False,
                    "thread 'test_pair' panicked at src/bin/test.rs:3:5\nassertion `left == right` failed",
                    {"total": 1, "passed": 0, "failed": 1},
                ),
                (True, "test result: ok. 1 passed; 0 failed; 0 ignored", {"total": 1, "passed": 1, "failed": 0}),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            translator.artifacts_dir = Path(tmpdir) / ".translated_modules"
            translator.artifacts_dir.mkdir()
            test_dir = Path(tmpdir) / "src" / "bin"
            test_dir.mkdir(parents=True)
            test_path = test_dir / "test.rs"
            test_path.write_text(
                "fn test_pair() {\n"
                "    let observed = decode(value);\n"
                "    assert_eq!(observed, x);\n"
                "    assert_eq!(observed, y);\n"
                "}\n",
                encoding="utf-8",
            )

            ok, _output, stats = translator._run_test_repair_loop()
            source = test_path.read_text(encoding="utf-8")
            test_history = json.loads((Path(tmpdir) / "test_fix_history.json").read_text(encoding="utf-8"))

        self.assertTrue(ok)
        self.assertEqual(stats["passed"], 1)
        self.assertIn("let lo = decode(value >> 1);", source)
        self.assertEqual(test_history[0]["status"], "patched")
        self.assertEqual(test_history[0]["test_source_fixes"][0]["path"], "src/bin/test.rs")
        self.assertEqual(test_history[0]["applied_test_source_fixes"][0]["status"], "patched")

    def test_test_repair_rolls_back_patch_when_tests_do_not_improve(self):
        bad_payload = json.dumps(
            {
                "fixes": [
                    {
                        "callable_id": "mod::fn::alpha",
                        "rust_code": "pub fn alpha() { still_broken(); }",
                    }
                ]
            }
        )
        good_payload = json.dumps(
            {
                "fixes": [
                    {
                        "callable_id": "mod::fn::alpha",
                        "rust_code": "pub fn alpha() { repaired(); }",
                    }
                ]
            }
        )
        translator = make_translator()
        translator.agent = SequenceAgent([bad_payload, good_payload])
        translator.project_name = "proj"
        translator.max_test_fix_attempts = 3
        translator.validator = FakeValidator([(True, "bad patch compiles"), (True, "good patch compiles")])
        translator.tests = FakeTestValidator(
            [
                (False, "thread 'test_alpha' panicked at src/bin/test_alpha.rs:4:5\nleft: 0\nright: 42", {"total": 1, "passed": 0, "failed": 1}),
                (False, "thread 'test_alpha' panicked at src/bin/test_alpha.rs:4:5\nleft: 0\nright: 42", {"total": 1, "passed": 0, "failed": 1}),
                (True, "test result: ok. 1 passed; 0 failed; 0 ignored", {"total": 1, "passed": 1, "failed": 0}),
            ]
        )
        item = make_callable("alpha")
        module_id = translator._callable_id(item)
        translator.callables = {module_id: item}
        translator.translation_history[module_id] = {
            "module_name": "alpha",
            "module_type": "fn",
            "file_path": "mod",
            "success": True,
            "compilation_attempts": [],
        }
        translator.translated_modules = {module_id}
        translator.translated_code = {module_id: "pub fn alpha() { broken(); }"}

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            translator.artifacts_dir = Path(tmpdir) / ".translated_modules"
            translator.artifacts_dir.mkdir()
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            module_path = src_dir / "mod.rs"
            module_path.write_text("pub fn alpha() { broken(); }\n", encoding="utf-8")
            translator._save_module(module_id, "pub fn alpha() { broken(); }")
            translator.skeleton = type("FakeSkeleton", (), {"module_files": {"mod": "mod.rs"}})()

            ok, _output, stats = translator._run_test_repair_loop()
            source = module_path.read_text(encoding="utf-8")
            artifact = next(translator.artifacts_dir.glob("*.rs")).read_text(encoding="utf-8")
            test_history = json.loads((Path(tmpdir) / "test_fix_history.json").read_text(encoding="utf-8"))

        self.assertTrue(ok)
        self.assertEqual(stats["passed"], 1)
        self.assertEqual(translator.tests.calls, 3)
        self.assertIn("repaired", source)
        self.assertIn("repaired", artifact)
        self.assertEqual(test_history[0]["status"], "rolled_back_no_test_improvement")
        self.assertTrue(test_history[0]["rolled_back"])
        self.assertEqual(test_history[1]["status"], "patched")

    def test_test_repair_keeps_patch_that_moves_from_compile_error_to_test_failure(self):
        payload = json.dumps(
            {
                "fixes": [
                    {
                        "callable_id": "mod::fn::alpha",
                        "rust_code": "pub fn alpha() { repaired(); }",
                    }
                ]
            }
        )
        translator = make_translator(agent_payload=payload)
        translator.project_name = "proj"
        translator.max_test_fix_attempts = 1
        translator.validator = FakeValidator([(True, "patch compiles")])
        translator.tests = FakeTestValidator(
            [
                (
                    False,
                    "error[E0616]: field `buffer` of struct `RSet` is private\nerror: could not compile",
                    {"total": 0, "passed": 0, "failed": 0},
                ),
                (
                    False,
                    "thread 'test_alpha' panicked at src/bin/test_alpha.rs:4:5",
                    {"total": 1, "passed": 0, "failed": 1},
                ),
            ]
        )
        item = make_callable("alpha")
        module_id = translator._callable_id(item)
        translator.callables = {module_id: item}
        translator.translation_history[module_id] = {
            "module_name": "alpha",
            "module_type": "fn",
            "file_path": "mod",
            "success": True,
            "compilation_attempts": [],
        }
        translator.translated_modules = {module_id}
        translator.translated_code = {module_id: "pub fn alpha() { broken(); }"}

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            translator.artifacts_dir = Path(tmpdir) / ".translated_modules"
            translator.artifacts_dir.mkdir()
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            module_path = src_dir / "mod.rs"
            module_path.write_text("pub fn alpha() { broken(); }\n", encoding="utf-8")
            translator._save_module(module_id, "pub fn alpha() { broken(); }")
            translator.skeleton = type("FakeSkeleton", (), {"module_files": {"mod": "mod.rs"}})()

            ok, _output, stats = translator._run_test_repair_loop()
            source = module_path.read_text(encoding="utf-8")
            test_history = json.loads((Path(tmpdir) / "test_fix_history.json").read_text(encoding="utf-8"))

        self.assertFalse(ok)
        self.assertEqual(stats["total"], 1)
        self.assertIn("repaired", source)
        self.assertEqual(test_history[0]["status"], "patched")
        self.assertTrue(test_history[0]["test_improved"])

    def test_test_repair_keeps_patch_that_advances_first_failing_test(self):
        payload = json.dumps(
            {
                "fixes": [
                    {
                        "callable_id": "mod::fn::alpha",
                        "rust_code": "pub fn alpha() { repaired(); }",
                    }
                ]
            }
        )
        translator = make_translator(agent_payload=payload)
        translator.project_name = "proj"
        translator.max_test_fix_attempts = 1
        translator.validator = FakeValidator([(True, "patch compiles")])
        translator.tests = FakeTestValidator(
            [
                (
                    False,
                    "\n".join(
                        [
                            "test result: ok. 0 passed; 0 failed; 0 ignored",
                            "running 4 tests",
                            "test test_alpha ... ok",
                            "test test_beta ...",
                            "thread 'test_beta' panicked at src/bin/tests.rs:8:5",
                            "test result: FAILED. 3 passed; 1 failed; 0 ignored",
                        ]
                    ),
                    {"total": 4, "passed": 3, "failed": 1},
                ),
                (
                    False,
                    "\n".join(
                        [
                            "test result: ok. 0 passed; 0 failed; 0 ignored",
                            "running 4 tests",
                            "test test_alpha ... ok",
                            "test test_beta ... ok",
                            "test test_gamma ... ok",
                            "test test_delta ...",
                            "thread 'test_delta' panicked at src/bin/tests.rs:30:5",
                            "test result: FAILED. 3 passed; 1 failed; 0 ignored",
                        ]
                    ),
                    {"total": 4, "passed": 3, "failed": 1},
                ),
            ]
        )
        item = make_callable("alpha")
        module_id = translator._callable_id(item)
        translator.callables = {module_id: item}
        translator.translation_history[module_id] = {
            "module_name": "alpha",
            "module_type": "fn",
            "file_path": "mod",
            "success": True,
            "compilation_attempts": [],
        }
        translator.translated_modules = {module_id}
        translator.translated_code = {module_id: "pub fn alpha() { broken(); }"}

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            translator.artifacts_dir = Path(tmpdir) / ".translated_modules"
            translator.artifacts_dir.mkdir()
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            module_path = src_dir / "mod.rs"
            module_path.write_text("pub fn alpha() { broken(); }\n", encoding="utf-8")
            translator._save_module(module_id, "pub fn alpha() { broken(); }")
            translator.skeleton = type("FakeSkeleton", (), {"module_files": {"mod": "mod.rs"}})()

            ok, _output, stats = translator._run_test_repair_loop()
            source = module_path.read_text(encoding="utf-8")
            test_history = json.loads((Path(tmpdir) / "test_fix_history.json").read_text(encoding="utf-8"))

        self.assertFalse(ok)
        self.assertEqual(stats["failed"], 1)
        self.assertIn("repaired", source)
        self.assertEqual(test_history[0]["status"], "patched")
        self.assertTrue(test_history[0]["test_improved"])

    def test_test_repair_keeps_patch_that_reveals_later_test_binaries(self):
        payload = json.dumps(
            {
                "fixes": [
                    {
                        "callable_id": "mod::fn::alpha",
                        "rust_code": "pub fn alpha() { repaired(); }",
                    }
                ]
            }
        )
        translator = make_translator(agent_payload=payload)
        translator.project_name = "proj"
        translator.max_test_fix_attempts = 1
        translator.validator = FakeValidator([(True, "patch compiles")])
        translator.tests = FakeTestValidator(
            [
                (
                    False,
                    "\n".join(
                        [
                            "running 11 tests",
                            "test test_deck ...",
                            "thread 'test_deck' panicked at src/bin/test_deck.rs:25:5",
                            "test result: FAILED. 10 passed; 1 failed; 0 ignored",
                        ]
                    ),
                    {"total": 11, "passed": 10, "failed": 1},
                ),
                (
                    False,
                    "\n".join(
                        [
                            "running 24 tests",
                            "test test_deck ... ok",
                            "test test_hands ...",
                            "thread 'test_hands' panicked at src/bin/test_hands.rs:305:9",
                            "test result: FAILED. 18 passed; 6 failed; 0 ignored",
                        ]
                    ),
                    {"total": 24, "passed": 18, "failed": 6},
                ),
            ]
        )
        item = make_callable("alpha")
        module_id = translator._callable_id(item)
        translator.callables = {module_id: item}
        translator.translation_history[module_id] = {
            "module_name": "alpha",
            "module_type": "fn",
            "file_path": "mod",
            "success": True,
            "compilation_attempts": [],
        }
        translator.translated_modules = {module_id}
        translator.translated_code = {module_id: "pub fn alpha() { broken(); }"}

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            translator.artifacts_dir = Path(tmpdir) / ".translated_modules"
            translator.artifacts_dir.mkdir()
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            module_path = src_dir / "mod.rs"
            module_path.write_text("pub fn alpha() { broken(); }\n", encoding="utf-8")
            translator._save_module(module_id, "pub fn alpha() { broken(); }")
            translator.skeleton = type("FakeSkeleton", (), {"module_files": {"mod": "mod.rs"}})()

            ok, _output, stats = translator._run_test_repair_loop()
            source = module_path.read_text(encoding="utf-8")
            test_history = json.loads((Path(tmpdir) / "test_fix_history.json").read_text(encoding="utf-8"))

        self.assertFalse(ok)
        self.assertEqual(stats["passed"], 18)
        self.assertIn("repaired", source)
        self.assertEqual(test_history[0]["status"], "patched")
        self.assertTrue(test_history[0]["test_improved"])

    def test_test_repair_keeps_patch_that_advances_panic_line_in_same_test(self):
        payload = json.dumps(
            {
                "fixes": [
                    {
                        "callable_id": "mod::fn::alpha",
                        "rust_code": "pub fn alpha() { repaired(); }",
                    }
                ]
            }
        )
        translator = make_translator(agent_payload=payload)
        translator.project_name = "proj"
        translator.max_test_fix_attempts = 1
        translator.validator = FakeValidator([(True, "patch compiles")])
        translator.tests = FakeTestValidator(
            [
                (
                    False,
                    "\n".join(
                        [
                            "running 1 test",
                            "test test_alpha ...",
                            "thread 'test_alpha' panicked at src/bin/tests.rs:43:5",
                            "test result: FAILED. 0 passed; 1 failed; 0 ignored",
                        ]
                    ),
                    {"total": 1, "passed": 0, "failed": 1},
                ),
                (
                    False,
                    "\n".join(
                        [
                            "running 1 test",
                            "test test_alpha ...",
                            "thread 'test_alpha' panicked at src/bin/tests.rs:81:5",
                            "test result: FAILED. 0 passed; 1 failed; 0 ignored",
                        ]
                    ),
                    {"total": 1, "passed": 0, "failed": 1},
                ),
            ]
        )
        item = make_callable("alpha")
        module_id = translator._callable_id(item)
        translator.callables = {module_id: item}
        translator.translation_history[module_id] = {
            "module_name": "alpha",
            "module_type": "fn",
            "file_path": "mod",
            "success": True,
            "compilation_attempts": [],
        }
        translator.translated_modules = {module_id}
        translator.translated_code = {module_id: "pub fn alpha() { broken(); }"}

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            translator.artifacts_dir = Path(tmpdir) / ".translated_modules"
            translator.artifacts_dir.mkdir()
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            module_path = src_dir / "mod.rs"
            module_path.write_text("pub fn alpha() { broken(); }\n", encoding="utf-8")
            translator._save_module(module_id, "pub fn alpha() { broken(); }")
            translator.skeleton = type("FakeSkeleton", (), {"module_files": {"mod": "mod.rs"}})()

            ok, _output, stats = translator._run_test_repair_loop()
            source = module_path.read_text(encoding="utf-8")
            test_history = json.loads((Path(tmpdir) / "test_fix_history.json").read_text(encoding="utf-8"))

        self.assertFalse(ok)
        self.assertEqual(stats["failed"], 1)
        self.assertIn("repaired", source)
        self.assertEqual(test_history[0]["status"], "patched")
        self.assertTrue(test_history[0]["test_improved"])

    def test_shared_item_context_and_patcher_replace_known_items(self):
        translator = make_translator()
        translator.validator = FakeValidator([(True, "cargo check ok")])

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            module_path = src_dir / "mod.rs"
            module_path.write_text(
                """
pub struct State {
    value: i32,
}

const LIMIT: usize = 1;

pub fn alpha() {
    unimplemented!()
}
""",
                encoding="utf-8",
            )
            translator.skeleton = type("FakeSkeleton", (), {"module_files": {"mod": "mod.rs"}})()

            context = translator._shared_item_context(["mod"])
            item_ids = {item["item_id"] for item in context}
            record = {}
            ok = translator._apply_shared_item_fixes(
                [
                    {
                        "item_id": "mod::struct::State",
                        "rust_code": "pub struct State {\n    value: i32,\n    extra: i32,\n}",
                    }
                ],
                record,
            )
            unknown_ok = translator._apply_shared_item_fixes(
                [
                    {
                        "item_id": "mod::struct::Missing",
                        "rust_code": "pub struct Missing;",
                    }
                ],
                {},
            )
            source = module_path.read_text(encoding="utf-8")

        self.assertIn("mod::struct::State", item_ids)
        self.assertIn("mod::const::LIMIT", item_ids)
        self.assertTrue(ok)
        self.assertFalse(unknown_ok)
        self.assertIn("extra: i32", source)
        self.assertEqual(record["applied_shared_item_fixes"][0]["status"], "compiled")

    def test_test_repair_can_patch_shared_items_before_callable_fixes(self):
        payload = json.dumps(
            {
                "shared_item_fixes": [
                    {
                        "item_id": "mod::struct::State",
                        "rust_code": "pub struct State {\n    value: i32,\n    repaired: bool,\n}",
                    }
                ],
                "fixes": [
                    {
                        "callable_id": "mod::fn::alpha",
                        "rust_code": "pub fn alpha() { let _state = State { value: 42, repaired: true }; }",
                    }
                ],
            }
        )
        translator = make_translator(agent_payload=payload)
        translator.project_name = "proj"
        translator.max_test_fix_attempts = 3
        translator.validator = FakeValidator([(True, "shared check ok"), (True, "callable check ok")])
        translator.tests = FakeTestValidator(
            [
                (False, "thread 'test_alpha' panicked at src/bin/test_alpha.rs:4:5\nleft: 0\nright: 42", {"total": 1, "passed": 0, "failed": 1}),
                (True, "test result: ok. 1 passed; 0 failed; 0 ignored", {"total": 1, "passed": 1, "failed": 0}),
            ]
        )
        item = make_callable("alpha")
        module_id = translator._callable_id(item)
        c_alpha = make_c_function(translator, "alpha", file_path="alpha.c")
        translator.callables = {module_id: item}
        translator.c_bindings = {
            module_id: CBinding(
                functions=(c_alpha,),
                relationship="direct",
                confidence=0.9,
                rationale="direct match",
                source="llm",
            )
        }
        translator.translation_history[module_id] = {
            "module_name": "alpha",
            "module_type": "fn",
            "file_path": "mod",
            "success": True,
            "compilation_attempts": [],
        }
        translator.translated_modules = {module_id}

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            translator.artifacts_dir = Path(tmpdir) / ".translated_modules"
            translator.artifacts_dir.mkdir()
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            module_path = src_dir / "mod.rs"
            module_path.write_text(
                """
pub struct State {
    value: i32,
}

pub fn alpha() {
    broken();
}
""",
                encoding="utf-8",
            )
            translator.skeleton = type("FakeSkeleton", (), {"module_files": {"mod": "mod.rs"}})()

            ok, output, stats = translator._run_test_repair_loop()
            source = module_path.read_text(encoding="utf-8")
            test_history = json.loads((Path(tmpdir) / "test_fix_history.json").read_text(encoding="utf-8"))

        self.assertTrue(ok)
        self.assertEqual(stats["passed"], 1)
        self.assertIn("repaired: bool", source)
        self.assertIn("State { value: 42, repaired: true }", source)
        self.assertIn("Candidate Rust shared items", test_history[0]["prompt"])
        self.assertIn("shared_item_fixes", test_history[0]["prompt"])
        self.assertEqual(test_history[0]["shared_item_fixes"][0]["item_id"], "mod::struct::State")
        self.assertEqual(test_history[0]["applied_shared_item_fixes"][0]["status"], "compiled")

    def test_test_repair_rolls_back_shared_items_when_patchset_does_not_compile(self):
        payload = json.dumps(
            {
                "shared_item_fixes": [
                    {
                        "item_id": "mod::struct::State",
                        "rust_code": "pub struct State {\n    value: i32,\n    repaired: bool,\n}",
                    }
                ],
                "fixes": [
                    {
                        "callable_id": "mod::fn::alpha",
                        "rust_code": "pub fn alpha() { let _state = State { value: 42, repaired: true }; }",
                    }
                ],
            }
        )
        translator = make_translator(agent_payload=payload)
        translator.project_name = "proj"
        translator.validator = FakeValidator([(False, "error[E0063]: missing field after shared change")])
        translator.tests = FakeTestValidator(
            [
                (False, "thread 'test_alpha' panicked at src/bin/test_alpha.rs:4:5", {"total": 1, "passed": 0, "failed": 1}),
            ]
        )
        item = make_callable("alpha")
        module_id = translator._callable_id(item)
        translator.callables = {module_id: item}
        translator.translation_history[module_id] = {
            "module_name": "alpha",
            "module_type": "fn",
            "file_path": "mod",
            "success": True,
            "compilation_attempts": [],
        }
        translator.translated_modules = {module_id}

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            translator.artifacts_dir = Path(tmpdir) / ".translated_modules"
            translator.artifacts_dir.mkdir()
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            module_path = src_dir / "mod.rs"
            original_source = """
pub struct State {
    value: i32,
}

pub fn alpha() {
    broken();
}
"""
            module_path.write_text(original_source, encoding="utf-8")
            translator.skeleton = type("FakeSkeleton", (), {"module_files": {"mod": "mod.rs"}})()

            ok, _output, _stats = translator._run_test_repair_loop()
            source = module_path.read_text(encoding="utf-8")
            test_history = json.loads((Path(tmpdir) / "test_fix_history.json").read_text(encoding="utf-8"))

        self.assertFalse(ok)
        self.assertEqual(source, original_source)
        self.assertEqual(test_history[0]["applied_shared_item_fixes"][0]["status"], "compile_failed")
        self.assertEqual(test_history[0]["applied_fixes"][0]["status"], "compile_failed")

    def test_test_repair_continues_after_unpatchable_fix_until_attempt_limit(self):
        payload = json.dumps(
            {
                "fixes": [
                    {
                        "callable_id": "mod::fn::alpha",
                        "rust_code": "pub fn beta() { repaired(); }",
                    }
                ]
            }
        )
        translator = make_translator()
        translator.agent = SequenceAgent([payload, payload])
        translator.project_name = "proj"
        translator.max_test_fix_attempts = 2
        translator.validator = FakeValidator([])
        translator.tests = FakeTestValidator(
            [
                (False, "thread 'test_alpha' panicked at src/bin/test_alpha.rs:4:5", {"total": 1, "passed": 0, "failed": 1}),
                (False, "thread 'test_alpha' panicked at src/bin/test_alpha.rs:4:5", {"total": 1, "passed": 0, "failed": 1}),
                (False, "thread 'test_alpha' panicked at src/bin/test_alpha.rs:4:5", {"total": 1, "passed": 0, "failed": 1}),
            ]
        )
        item = make_callable("alpha")
        module_id = translator._callable_id(item)
        translator.callables = {module_id: item}
        translator.translation_history[module_id] = {
            "module_name": "alpha",
            "module_type": "fn",
            "file_path": "mod",
            "success": True,
            "compilation_attempts": [],
        }
        translator.translated_modules = {module_id}

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            translator.artifacts_dir = Path(tmpdir) / ".translated_modules"
            translator.artifacts_dir.mkdir()
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            module_path = src_dir / "mod.rs"
            original_source = "pub fn alpha() { broken(); }\n"
            module_path.write_text(original_source, encoding="utf-8")
            translator.skeleton = type("FakeSkeleton", (), {"module_files": {"mod": "mod.rs"}})()

            ok, _output, _stats = translator._run_test_repair_loop()
            source = module_path.read_text(encoding="utf-8")
            test_history = json.loads((Path(tmpdir) / "test_fix_history.json").read_text(encoding="utf-8"))

        self.assertFalse(ok)
        self.assertEqual(translator.tests.calls, 3)
        self.assertEqual(len(translator.agent.calls), 2)
        self.assertEqual(len(test_history), 2)
        self.assertEqual(source, original_source)
        self.assertEqual([entry["status"] for entry in test_history], ["patch_failed", "patch_failed"])

    def test_test_repair_continues_after_patchset_compile_failure_until_attempt_limit(self):
        payload = json.dumps(
            {
                "fixes": [
                    {
                        "callable_id": "mod::fn::alpha",
                        "rust_code": "pub fn alpha() { repaired(); }",
                    }
                ]
            }
        )
        translator = make_translator()
        translator.agent = SequenceAgent([payload, payload])
        translator.project_name = "proj"
        translator.max_test_fix_attempts = 2
        translator.validator = FakeValidator(
            [
                (False, "error[E0425]: cannot find function `repaired`"),
                (False, "error[E0425]: cannot find function `repaired`"),
            ]
        )
        translator.tests = FakeTestValidator(
            [
                (False, "thread 'test_alpha' panicked at src/bin/test_alpha.rs:4:5", {"total": 1, "passed": 0, "failed": 1}),
                (False, "thread 'test_alpha' panicked at src/bin/test_alpha.rs:4:5", {"total": 1, "passed": 0, "failed": 1}),
                (False, "thread 'test_alpha' panicked at src/bin/test_alpha.rs:4:5", {"total": 1, "passed": 0, "failed": 1}),
            ]
        )
        item = make_callable("alpha")
        module_id = translator._callable_id(item)
        translator.callables = {module_id: item}
        translator.translation_history[module_id] = {
            "module_name": "alpha",
            "module_type": "fn",
            "file_path": "mod",
            "success": True,
            "compilation_attempts": [],
        }
        translator.translated_modules = {module_id}

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            translator.artifacts_dir = Path(tmpdir) / ".translated_modules"
            translator.artifacts_dir.mkdir()
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            module_path = src_dir / "mod.rs"
            original_source = "pub fn alpha() { broken(); }\n"
            module_path.write_text(original_source, encoding="utf-8")
            translator.skeleton = type("FakeSkeleton", (), {"module_files": {"mod": "mod.rs"}})()

            ok, _output, _stats = translator._run_test_repair_loop()
            source = module_path.read_text(encoding="utf-8")
            test_history = json.loads((Path(tmpdir) / "test_fix_history.json").read_text(encoding="utf-8"))

        self.assertFalse(ok)
        self.assertEqual(translator.tests.calls, 3)
        self.assertEqual(translator.validator.calls, 2)
        self.assertEqual(len(translator.agent.calls), 2)
        self.assertEqual(len(test_history), 2)
        self.assertEqual(source, original_source)
        self.assertEqual([entry["status"] for entry in test_history], ["compile_failed", "compile_failed"])

    def test_compile_repair_can_patch_shared_items_after_failure_threshold(self):
        payload = json.dumps(
            {
                "shared_item_fixes": [
                    {
                        "item_id": "mod::struct::State",
                        "rust_code": "pub struct State {\n    value: i32,\n    repaired: bool,\n}",
                    }
                ],
                "rust_code": "pub fn alpha() { let _state = State { value: 1, repaired: true }; }",
            }
        )
        translator = make_translator(agent_payload=payload)
        translator.callable_failures = 11
        translator.validator = FakeValidator(
            [
                (False, "error[E0609]: no field `repaired` on type `State`"),
                (True, "shared check ok"),
                (True, "callable check ok"),
            ]
        )
        item = make_callable("alpha")
        module_id = translator._callable_id(item)
        translator.callables = {module_id: item}
        translator.translation_history[module_id] = {
            "module_name": "alpha",
            "module_type": "fn",
            "file_path": "mod",
            "success": False,
            "compilation_attempts": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            translator.artifacts_dir = Path(tmpdir) / ".translated_modules"
            translator.artifacts_dir.mkdir()
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            module_path = src_dir / "mod.rs"
            module_path.write_text(
                """
pub struct State {
    value: i32,
}

pub fn alpha() {
    let _state = State { value: 1, repaired: true };
}
""",
                encoding="utf-8",
            )
            translator.skeleton = type("FakeSkeleton", (), {"module_files": {"mod": "mod.rs"}})()

            ok, final_code = translator._compile_fix(item, module_id, module_path, "pub fn alpha() {}")
            source = module_path.read_text(encoding="utf-8")

        self.assertTrue(ok)
        self.assertIn("repaired: bool", source)
        self.assertIn("State { value: 1, repaired: true }", source)
        self.assertIn("shared_item_fixes", translator.agent.calls[0]["prompt"])
        self.assertEqual(final_code, "pub fn alpha() { let _state = State { value: 1, repaired: true }; }")

    def test_compile_repair_allows_shared_resource_wrapper_trait_fix_below_threshold(self):
        payload = json.dumps(
            {
                "shared_item_fixes": [
                    {
                        "item_id": "mod::impl::ClonableFile",
                        "rust_code": "impl ClonableFile {\n    pub fn read_byte(&mut self) -> u8 { 0 }\n}",
                    }
                ],
                "rust_code": "pub fn alpha() {}",
            }
        )
        translator = make_translator(agent_payload=payload)
        translator.callable_failures = 0
        translator.validator = FakeValidator(
            [
                (
                    False,
                    "error[E0277]: the trait bound `ClonableFile: std::io::Read` is not satisfied\n"
                    "error[E0616]: field `file` of struct `ClonableFile` is private",
                ),
                (True, "shared check ok"),
                (True, "callable check ok"),
            ]
        )
        item = make_callable("alpha")
        module_id = translator._callable_id(item)
        translator.callables = {module_id: item}
        translator.translation_history[module_id] = {
            "module_name": "alpha",
            "module_type": "fn",
            "file_path": "mod",
            "success": False,
            "compilation_attempts": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            translator.artifacts_dir = Path(tmpdir) / ".translated_modules"
            translator.artifacts_dir.mkdir()
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            module_path = src_dir / "mod.rs"
            module_path.write_text(
                """
pub struct ClonableFile {
    file: std::fs::File,
}

impl ClonableFile {
    pub fn new(file: std::fs::File) -> Self {
        Self { file }
    }
}

pub fn alpha() {
    unimplemented!()
}
""",
                encoding="utf-8",
            )
            translator.skeleton = type("FakeSkeleton", (), {"module_files": {"mod": "mod.rs"}})()

            ok, final_code = translator._compile_fix(item, module_id, module_path, "pub fn alpha() {}")
            source = module_path.read_text(encoding="utf-8")

        prompt = translator.agent.calls[0]["prompt"]
        self.assertTrue(ok)
        self.assertIn('"shared_item_fixes"', prompt)
        self.assertIn("resource-wrapper-constructor-safety", prompt)
        self.assertIn("pub fn read_byte", source)
        self.assertEqual(final_code, "pub fn alpha() {}")

    def test_shared_compile_fix_prompt_requires_single_shared_item_replacement(self):
        translator = make_translator(agent_payload='{"shared_item_fixes":[],"rust_code":"pub fn alpha() {}"}')
        translator.callable_failures = 0
        translator.validator = FakeValidator(
            [
                (False, "error[E0277]: the trait bound `ClonableFile: std::io::Read` is not satisfied"),
                (True, "callable check ok"),
            ]
        )
        item = make_callable("alpha")
        module_id = translator._callable_id(item)
        translator.callables = {module_id: item}
        translator.translation_history[module_id] = {
            "module_name": "alpha",
            "module_type": "fn",
            "file_path": "mod",
            "success": False,
            "compilation_attempts": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            translator.artifacts_dir = Path(tmpdir) / ".translated_modules"
            translator.artifacts_dir.mkdir()
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            module_path = src_dir / "mod.rs"
            module_path.write_text(
                """
pub struct ClonableFile {
    file: std::fs::File,
}

impl ClonableFile {
    pub fn new(file: std::fs::File) -> Self {
        Self { file }
    }
}

pub fn alpha() {
    unimplemented!()
}
""",
                encoding="utf-8",
            )
            translator.skeleton = type("FakeSkeleton", (), {"module_files": {"mod": "mod.rs"}})()

            translator._compile_fix(item, module_id, module_path, "pub fn alpha() {}")

        prompt = translator.agent.calls[0]["prompt"]
        self.assertIn("Each shared_item_fix replaces exactly one listed item_id", prompt)
        self.assertIn("one complete `impl Type { ... }` block", prompt)
        self.assertIn("add narrow forwarding methods", prompt)

    def test_compile_repair_allows_shared_resource_wrapper_constructor_fix_below_threshold(self):
        payload = json.dumps(
            {
                "shared_item_fixes": [
                    {
                        "item_id": "mod::impl::ClonableFile",
                        "rust_code": "impl ClonableFile {\n    pub fn new(file: std::fs::File) -> Self { Self { file } }\n}",
                    }
                ],
                "rust_code": "pub fn alpha() {}",
            }
        )
        translator = make_translator(agent_payload=payload)
        translator.callable_failures = 0
        translator.validator = FakeValidator(
            [
                (False, "error[E0277]: the trait bound `ClonableFile: From<File>` is not satisfied"),
                (True, "shared check ok"),
                (True, "callable check ok"),
            ]
        )
        item = make_callable("alpha")
        module_id = translator._callable_id(item)
        translator.callables = {module_id: item}
        translator.translation_history[module_id] = {
            "module_name": "alpha",
            "module_type": "fn",
            "file_path": "mod",
            "success": False,
            "compilation_attempts": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            translator.artifacts_dir = Path(tmpdir) / ".translated_modules"
            translator.artifacts_dir.mkdir()
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            module_path = src_dir / "mod.rs"
            module_path.write_text(
                """
pub struct ClonableFile {
    file: std::fs::File,
}

impl ClonableFile {
    pub fn new(file: std::fs::File) -> Self {
        Self { file }
    }
}

pub fn alpha() {
    unimplemented!()
}
""",
                encoding="utf-8",
            )
            translator.skeleton = type("FakeSkeleton", (), {"module_files": {"mod": "mod.rs"}})()

            ok, final_code = translator._compile_fix(item, module_id, module_path, "pub fn alpha() {}")
            source = module_path.read_text(encoding="utf-8")

        prompt = translator.agent.calls[0]["prompt"]
        self.assertTrue(ok)
        self.assertIn('"shared_item_fixes"', prompt)
        self.assertIn("resource-wrapper-constructor-safety", prompt)
        self.assertIn("pub fn new(file: std::fs::File)", source)
        self.assertEqual(final_code, "pub fn alpha() {}")

    def test_compile_fix_prompt_includes_shared_items_as_read_only_context_below_threshold(self):
        translator = make_translator(agent_payload="pub fn alpha() { let _state = State { value: 1 }; }")
        translator.callable_failures = 10
        translator.validator = FakeValidator(
            [
                (False, "error[E0609]: no field `repaired` on type `State`"),
                (True, "callable check ok"),
            ]
        )
        item = make_callable("alpha")
        module_id = translator._callable_id(item)
        c_alpha = make_c_function(translator, "alpha", file_path="alpha.c")
        translator.callables = {module_id: item}
        translator.c_bindings = {
            module_id: CBinding(
                functions=(c_alpha,),
                relationship="direct",
                confidence=0.9,
                rationale="direct match",
                source="llm",
            )
        }
        translator.callable_categories = {module_id: "direct_function"}
        translator.file_contexts = {module_id: [{"path": "alpha.c", "code": "typedef struct State State;"}]}
        translator.translation_history[module_id] = {
            "module_name": "alpha",
            "module_type": "fn",
            "file_path": "mod",
            "success": False,
            "compilation_attempts": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            translator.artifacts_dir = Path(tmpdir) / ".translated_modules"
            translator.artifacts_dir.mkdir()
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            module_path = src_dir / "mod.rs"
            module_path.write_text(
                """
pub struct State {
    value: i32,
}

pub fn alpha() {
    let _state = State { value: 1, repaired: true };
}
""",
                encoding="utf-8",
            )
            translator.skeleton = type("FakeSkeleton", (), {"module_files": {"mod": "mod.rs"}})()

            ok, _ = translator._compile_fix(item, module_id, module_path, "pub fn alpha() {}")

        prompt = translator.agent.calls[0]["prompt"]
        self.assertTrue(ok)
        self.assertIn("Candidate Rust shared items", prompt)
        self.assertIn("mod::struct::State", prompt)
        self.assertIn("C alignment", prompt)
        self.assertIn("typedef struct State State;", prompt)
        self.assertNotIn('"shared_item_fixes"', prompt)

    def test_compile_fix_prompt_includes_imported_module_shared_items(self):
        translator = make_translator(agent_payload="pub fn alpha() -> RC { RC::Ok }")
        translator.validator = FakeValidator(
            [
                (False, "error[E0308]: expected `RC`, found integer"),
                (True, "callable check ok"),
            ]
        )
        item = Callable(
            module="mod",
            kind="fn",
            name="alpha",
            type_name=None,
            file_name="mod.rs",
            signature="pub fn alpha() -> RC",
            signature_key="pubfnalpha()->RC",
            param_count=0,
            stub="pub fn alpha() -> RC { unimplemented!() }",
        )
        module_id = translator._callable_id(item)
        translator.callables = {module_id: item}
        translator.c_bindings = {
            module_id: CBinding(
                functions=(),
                relationship="no_safe_binding",
                confidence=0.0,
                rationale="no binding",
                source="empty_fallback",
            )
        }
        translator.callable_categories = {module_id: "direct_function"}
        translator.file_contexts = {module_id: []}
        translator.translation_history[module_id] = {
            "module_name": "alpha",
            "module_type": "fn",
            "file_path": "mod",
            "success": False,
            "compilation_attempts": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            translator.artifacts_dir = Path(tmpdir) / ".translated_modules"
            translator.artifacts_dir.mkdir()
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            module_path = src_dir / "mod.rs"
            module_path.write_text(
                "use crate::dberror::RC;\n\npub fn alpha() -> RC { unimplemented!() }\n",
                encoding="utf-8",
            )
            (src_dir / "dberror.rs").write_text(
                """
pub enum RC {
    Ok = 0,
    WriteFailed = 3,
}
""",
                encoding="utf-8",
            )
            translator.skeleton = type("FakeSkeleton", (), {"module_files": {"mod": "mod.rs", "dberror": "dberror.rs"}})()

            ok, _ = translator._compile_fix(item, module_id, module_path, "pub fn alpha() -> RC { 0 }")

        prompt = translator.agent.calls[0]["prompt"]
        self.assertTrue(ok)
        self.assertIn('"item_id": "dberror::enum::RC"', prompt)
        self.assertIn("WriteFailed = 3", prompt)

    def test_shared_item_context_collects_items_inside_module_wrapper(self):
        translator = make_translator()

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            (src_dir / "card.rs").write_text(
                """
pub mod card {
    pub struct CardCollection {
        c: Option<Card>,
        next: Option<Box<CardCollection>>,
    }

    pub enum CardRank {
        Ace,
        InvalidRank,
    }

    pub fn local_helper() {
        struct LocalOnly {
            value: i32,
        }
    }
}
""",
                encoding="utf-8",
            )
            translator.skeleton = type("FakeSkeleton", (), {"module_files": {"card": "card.rs"}})()

            shared_items = translator._shared_item_context(["card"])

        item_ids = {item["item_id"] for item in shared_items}
        self.assertIn("card::struct::CardCollection", item_ids)
        self.assertIn("card::enum::CardRank", item_ids)
        self.assertNotIn("card::struct::LocalOnly", item_ids)

    def test_shared_context_modules_include_grouped_crate_use_imports(self):
        translator = make_translator()
        parser = make_callable("create_variable", module="parser")

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            (src_dir / "parser.rs").write_text(
                "use crate::{common, hash_table};\n\npub fn create_variable() { unimplemented!() }\n",
                encoding="utf-8",
            )
            (src_dir / "common.rs").write_text(
                """
pub enum AstNodeUnion {
    Variable(Variable),
}

pub struct Variable {
    name: String,
}
""",
                encoding="utf-8",
            )
            (src_dir / "hash_table.rs").write_text(
                """
pub struct HashTable {
    entries: usize,
}
""",
                encoding="utf-8",
            )
            translator.skeleton = type(
                "FakeSkeleton",
                (),
                {"module_files": {"parser": "parser.rs", "common": "common.rs", "hash_table": "hash_table.rs"}},
            )()

            modules = translator._shared_context_modules_for_item(parser)
            shared_items = translator._shared_item_context(modules)

        item_ids = {item["item_id"] for item in shared_items}
        self.assertEqual(modules, ["common", "hash_table", "parser"])
        self.assertIn("common::enum::AstNodeUnion", item_ids)
        self.assertIn("hash_table::struct::HashTable", item_ids)

    def test_shared_item_context_includes_impl_blocks(self):
        translator = make_translator()

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            (src_dir / "compiler.rs").write_text(
                """
pub struct ClonableFile {
    file: std::fs::File,
    path: std::path::PathBuf,
}

impl ClonableFile {
    pub fn new(path: impl Into<std::path::PathBuf>) -> std::io::Result<Self> {
        unimplemented!()
    }
}

impl Clone for ClonableFile {
    fn clone(&self) -> Self {
        unimplemented!()
    }
}
""",
                encoding="utf-8",
            )
            translator.skeleton = type("FakeSkeleton", (), {"module_files": {"compiler": "compiler.rs"}})()

            shared_items = translator._shared_item_context(["compiler"])

        item_ids = {item["item_id"] for item in shared_items}
        self.assertIn("compiler::struct::ClonableFile", item_ids)
        self.assertIn("compiler::impl::ClonableFile", item_ids)
        self.assertIn("compiler::impl::Clone for ClonableFile", item_ids)

    def test_shared_item_context_prioritizes_types_and_impls_over_constants(self):
        translator = make_translator()

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            consts = "\n".join(f"pub const C{i}: i32 = {i};" for i in range(80))
            (src_dir / "compiler.rs").write_text(
                f"""
{consts}

pub struct ClonableFile {{
    file: std::fs::File,
    path: std::path::PathBuf,
}}

impl ClonableFile {{
    pub fn new(path: impl Into<std::path::PathBuf>) -> std::io::Result<Self> {{
        unimplemented!()
    }}
}}
""",
                encoding="utf-8",
            )
            translator.skeleton = type("FakeSkeleton", (), {"module_files": {"compiler": "compiler.rs"}})()

            shared_items = translator._shared_item_context(["compiler"])

        item_ids = {item["item_id"] for item in shared_items}
        self.assertLessEqual(len(shared_items), translator.SHARED_ITEM_CONTEXT_LIMIT)
        self.assertIn("compiler::struct::ClonableFile", item_ids)
        self.assertIn("compiler::impl::ClonableFile", item_ids)

    def test_shared_item_patch_can_replace_impl_block(self):
        translator = make_translator()

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            module_path = src_dir / "compiler.rs"
            module_path.write_text(
                """
pub struct ClonableFile {
    file: std::fs::File,
    path: std::path::PathBuf,
}

impl ClonableFile {
    pub fn new(path: impl Into<std::path::PathBuf>) -> std::io::Result<Self> {
        unimplemented!()
    }
}
""",
                encoding="utf-8",
            )
            translator.skeleton = type("FakeSkeleton", (), {"module_files": {"compiler": "compiler.rs"}})()

            ok = translator._patch_shared_item(
                module_path,
                "compiler",
                "impl",
                "ClonableFile",
                """
impl ClonableFile {
    pub fn new(path: impl Into<std::path::PathBuf>) -> std::io::Result<Self> {
        let path = path.into();
        let file = std::fs::OpenOptions::new().read(true).write(true).create(true).open(&path)?;
        Ok(Self { file, path })
    }
}
""",
            )
            source = module_path.read_text(encoding="utf-8")

        self.assertTrue(ok)
        self.assertIn("OpenOptions::new()", source)
        self.assertNotIn("unimplemented!()", source)

    def test_initial_translation_can_patch_shared_items_after_failure_threshold(self):
        payload = json.dumps(
            {
                "shared_item_fixes": [
                    {
                        "item_id": "mod::struct::State",
                        "rust_code": "pub struct State {\n    value: i32,\n    repaired: bool,\n}",
                    }
                ],
                "rust_code": "pub fn alpha() { let _state = State { value: 1, repaired: true }; }",
            }
        )
        translator = make_translator(agent_payload=payload)
        translator.callable_failures = 11
        translator.validator = FakeValidator([(True, "shared check ok"), (True, "callable check ok")])
        item = make_callable("alpha")
        module_id = translator._callable_id(item)
        c_alpha = make_c_function(translator, "alpha", file_path="alpha.c")
        translator.callables = {module_id: item}
        translator.c_bindings = {
            module_id: CBinding(
                functions=(c_alpha,),
                relationship="direct",
                confidence=0.9,
                rationale="direct match",
                source="llm",
            )
        }
        translator.callable_categories = {module_id: "direct_function"}
        translator.file_contexts = {module_id: []}
        translator.translation_history[module_id] = {
            "module_name": "alpha",
            "module_type": "fn",
            "file_path": "mod",
            "success": False,
            "compilation_attempts": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            translator.artifacts_dir = Path(tmpdir) / ".translated_modules"
            translator.artifacts_dir.mkdir()
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            module_path = src_dir / "mod.rs"
            module_path.write_text(
                """
pub struct State {
    value: i32,
}

pub fn alpha() {
    unimplemented!()
}
""",
                encoding="utf-8",
            )
            translator.skeleton = type("FakeSkeleton", (), {"module_files": {"mod": "mod.rs"}})()

            ok = translator._translate_one(item)
            source = module_path.read_text(encoding="utf-8")

        self.assertTrue(ok)
        self.assertIn("repaired: bool", source)
        self.assertIn("State { value: 1, repaired: true }", source)
        self.assertIn("shared_item_fixes", translator.agent.calls[0]["prompt"])

    def test_test_repair_candidates_follow_callable_dependency_chain(self):
        translator = make_translator()
        alpha = make_callable("alpha", module="api")
        helper = make_callable("helper", module="inner")
        alpha_id = translator._callable_id(alpha)
        helper_id = translator._callable_id(helper)
        c_alpha = make_c_function(translator, "alpha", file_path="api.c", calls=("helper",))
        c_helper = make_c_function(translator, "helper", file_path="inner.c")
        translator.callables = {alpha_id: alpha, helper_id: helper}
        translator.call_deps = {alpha_id: {helper_id}, helper_id: set()}
        translator.c_bindings = {
            alpha_id: CBinding((c_alpha,), "direct", 1.0, "entry", "llm"),
            helper_id: CBinding((c_helper,), "direct", 1.0, "callee", "llm"),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            src_dir = Path(tmpdir) / "src"
            (src_dir / "bin").mkdir(parents=True)
            (src_dir / "api.rs").write_text("pub fn alpha() { broken(); }\n", encoding="utf-8")
            (src_dir / "inner.rs").write_text(
                """
pub struct State {
    value: i32,
}

pub fn helper() {
    broken();
}
""",
                encoding="utf-8",
            )
            (src_dir / "bin" / "test_flow.rs").write_text(
                """
#[test]
fn test_flow() {
    api::alpha();
}
""",
                encoding="utf-8",
            )
            translator.skeleton = type("FakeSkeleton", (), {"module_files": {"api": "api.rs", "inner": "inner.rs"}})()

            context = translator._test_fix_callable_context(
                "thread 'test_flow' panicked at src/bin/test_flow.rs:4:5"
            )
            modules = {item["module"] for item in context}
            shared_items = translator._shared_item_context(modules)

        self.assertIn("api::fn::alpha", {item["callable_id"] for item in context})
        self.assertIn("inner::fn::helper", {item["callable_id"] for item in context})
        self.assertIn("inner::struct::State", {item["item_id"] for item in shared_items})

    def test_test_repair_candidates_use_setup_call_before_failing_assertion(self):
        translator = make_translator()
        xopt_parse = make_callable("xopt_parse", module="xopt")
        snprintf_main = make_callable("main", module="snprintf")
        xopt_id = translator._callable_id(xopt_parse)
        main_id = translator._callable_id(snprintf_main)
        c_xopt = make_c_function(translator, "xopt_parse", file_path="xopt.c")
        c_snprintf = make_c_function(translator, "rpl_snprintf", file_path="snprintf.c")
        translator.callables = {xopt_id: xopt_parse, main_id: snprintf_main}
        translator.call_deps = {xopt_id: set(), main_id: set()}
        translator.c_bindings = {
            xopt_id: CBinding((c_xopt,), "direct", 1.0, "parse entry", "llm"),
            main_id: CBinding((c_snprintf,), "semantic_rewrite", 0.7, "generic entry", "llm"),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            src_dir = Path(tmpdir) / "src"
            (src_dir / "bin").mkdir(parents=True)
            (src_dir / "xopt.rs").write_text("pub fn xopt_parse() { broken(); }\n", encoding="utf-8")
            (src_dir / "snprintf.rs").write_text("pub fn main() {}\n", encoding="utf-8")
            test_lines = [
                "use crate::xopt::xopt_parse;",
                "",
                "#[test]",
                "fn test_parse_sets_config() {",
                "    let mut config = 0;",
                "    let argv = vec![\"program\", \"--some-int=42\"];",
                "    xopt_parse();",
            ]
            test_lines.extend(f"    let _padding_{i} = {i};" for i in range(30))
            test_lines.extend(
                [
                    "    assert_eq!(config, 42, \"some_int should be 42\");",
                    "}",
                    "",
                    "fn main() {}",
                ]
            )
            (src_dir / "bin" / "test.rs").write_text("\n".join(test_lines), encoding="utf-8")
            translator.skeleton = type("FakeSkeleton", (), {"module_files": {"xopt": "xopt.rs", "snprintf": "snprintf.rs"}})()

            context = translator._test_fix_callable_context(
                "thread 'test_parse_sets_config' panicked at src/bin/test.rs:38:5"
            )

        self.assertGreaterEqual(len(context), 1)
        self.assertEqual("xopt::fn::xopt_parse", context[0]["callable_id"])
        self.assertNotEqual("snprintf::fn::main", context[0]["callable_id"])

    def test_test_source_context_keeps_file_header_constants(self):
        translator = make_translator()

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            test_path = Path(tmpdir) / "src" / "bin" / "test.rs"
            test_path.parent.mkdir(parents=True)
            lines = [
                "use crate::parser::parse;",
                "const TEST_CAPACITY: usize = 10;",
                "",
                "#[test]",
                "fn test_parse() {",
            ]
            lines.extend(f"    let _padding_{i} = {i};" for i in range(60))
            lines.extend(
                [
                    "    assert_eq!(parse(\"\"), TEST_CAPACITY);",
                    "}",
                    "fn main() {}",
                ]
            )
            test_path.write_text("\n".join(lines), encoding="utf-8")

            context = translator._test_source_context(
                "thread 'test_parse' panicked at src/bin/test.rs:66:5"
            )

        self.assertEqual(1, len(context))
        self.assertIn("const TEST_CAPACITY: usize = 10;", context[0]["code"])
        self.assertIn('assert_eq!(parse(""), TEST_CAPACITY);', context[0]["code"])

    def test_test_source_context_includes_other_test_file_constants(self):
        translator = make_translator()

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            bin_dir = Path(tmpdir) / "src" / "bin"
            bin_dir.mkdir(parents=True)
            (bin_dir / "test_load.rs").write_text(
                "\n".join(
                    [
                        "use crate::parser::parse;",
                        "#[test]",
                        "fn test_load() {",
                        "    assert_eq!(parse(\"font: 1\"), 1);",
                        "}",
                    ]
                ),
                encoding="utf-8",
            )
            (bin_dir / "test_parse.rs").write_text(
                "\n".join(
                    [
                        "use crate::parser::parse;",
                        "const TEST_CAPACITY: usize = 10;",
                        "",
                        "#[test]",
                        "fn test_parse() {",
                        "    assert_eq!(parse(\"\"), TEST_CAPACITY);",
                        "}",
                    ]
                ),
                encoding="utf-8",
            )

            context = translator._test_source_context(
                "thread 'test_load' panicked at src/bin/test_load.rs:4:5"
            )

        self.assertEqual("src/bin/test_load.rs", context[0]["path"])
        self.assertTrue(
            any(
                item["path"] == "src/bin/test_parse.rs"
                and "const TEST_CAPACITY: usize = 10;" in item["code"]
                for item in context
            )
        )

    def test_test_repair_prioritizes_parser_callables_for_unexpected_token_stderr(self):
        translator = make_translator()
        parser_consume = make_callable("consume", module="parser")
        parser_parse = make_callable("parse", module="parser")
        token_new = make_callable("new_token", module="token")
        consume_id = translator._callable_id(parser_consume)
        parse_id = translator._callable_id(parser_parse)
        token_id = translator._callable_id(token_new)
        translator.callables = {
            consume_id: parser_consume,
            parse_id: parser_parse,
            token_id: token_new,
        }
        translator.call_deps = {consume_id: set(), parse_id: {consume_id}, token_id: set()}
        translator.c_bindings = {
            consume_id: CBinding((make_c_function(translator, "consume", file_path="parser.c"),), "direct", 1.0, "", "llm"),
            parse_id: CBinding((make_c_function(translator, "parse", file_path="parser.c"),), "direct", 1.0, "", "llm"),
            token_id: CBinding((make_c_function(translator, "new_token", file_path="token.c"),), "direct", 1.0, "", "llm"),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir(parents=True)
            (src_dir / "parser.rs").write_text(
                "pub fn consume() {}\npub fn parse() {}\n",
                encoding="utf-8",
            )
            (src_dir / "token.rs").write_text("pub fn new_token() {}\n", encoding="utf-8")
            translator.skeleton = type(
                "FakeSkeleton",
                (),
                {"module_files": {"parser": "parser.rs", "token": "token.rs"}},
            )()

            context = translator._test_fix_callable_context("Unexpected token: ")

        ids = [item["callable_id"] for item in context]
        self.assertLess(ids.index("parser::fn::consume"), ids.index("token::fn::new_token"))
        self.assertIn("parser::fn::parse", ids)

    def test_test_repair_candidates_include_public_non_stub_callables(self):
        translator = make_translator()
        constructor = Callable(
            module="circular_buffer",
            kind="method",
            name="new",
            type_name="CircularBuffer",
            file_name="circular_buffer.rs",
            signature="pub fn new(size: usize) -> Self",
            signature_key=re.sub(r"\s+", "", "pub fn new(size: usize) -> Self"),
            param_count=1,
            stub="pub fn new(size: usize) -> Self { CircularBuffer { size } }",
        )
        c_create = make_c_function(translator, "CircularBufferCreate", file_path="circular_buffer.c")

        translator._prepare_test_repair_callables([constructor], c_index(c_create))

        module_id = ("circular_buffer", "method", "CircularBuffer::new")
        self.assertIn(module_id, translator.callables)
        self.assertEqual(translator.c_bindings[module_id].functions[0].name, "CircularBufferCreate")

    def test_test_repair_candidates_include_trait_impl_stubs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            interfaces = Path(tmpdir) / "src" / "interfaces"
            interfaces.mkdir(parents=True)
            (interfaces / "bigint.rs").write_text(
                """
pub struct BigInt;

impl PartialOrd for BigInt {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        unimplemented!()
    }
}
""",
                encoding="utf-8",
            )

            index = InterfaceIndex(tmpdir).build()
            repair_ids = {
                (item.module, item.kind, item.type_name, item.name)
                for item in index.list_public()
            }

        self.assertIn(("bigint", "method", "BigInt", "partial_cmp"), repair_ids)

    def test_constructor_reset_heuristic_wraps_existing_constructor_body(self):
        translator = make_translator()
        translator.validator = FakeValidator([(True, "cargo check ok")])
        constructor = Callable(
            module="circular_buffer",
            kind="method",
            name="new",
            type_name="CircularBuffer",
            file_name="circular_buffer.rs",
            signature="pub fn new(size: usize) -> Self",
            signature_key=re.sub(r"\s+", "", "pub fn new(size: usize) -> Self"),
            param_count=1,
            stub="pub fn new(size: usize) -> Self { CircularBuffer { size, tail_offset: 0, head_offset: 0 } }",
        )
        reset = make_callable("reset", module="circular_buffer", kind="method", type_name="CircularBuffer")
        create_c = make_c_function(translator, "CircularBufferCreate", file_path="circular_buffer.c", calls=("CircularBufferReset",))
        reset_c = make_c_function(translator, "CircularBufferReset", file_path="circular_buffer.c")
        constructor_id = translator._callable_id(constructor)
        reset_id = translator._callable_id(reset)
        translator.callables = {constructor_id: constructor, reset_id: reset}
        translator.c_bindings = {
            constructor_id: CBinding((create_c,), "direct", 1.0, "constructor", "rule_fallback"),
            reset_id: CBinding((reset_c,), "direct", 1.0, "reset", "llm"),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            translator.artifacts_dir = Path(tmpdir) / ".translated_modules"
            translator.artifacts_dir.mkdir()
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            module_path = src_dir / "circular_buffer.rs"
            module_path.write_text(
                """
pub struct CircularBuffer {
    size: usize,
    tail_offset: usize,
    head_offset: usize,
}
impl CircularBuffer {
    pub fn new(size: usize) -> Self {
        CircularBuffer { size, tail_offset: 0, head_offset: 0 }
    }
    pub fn reset(&mut self) {
        self.tail_offset = usize::MAX;
        self.head_offset = usize::MAX;
    }
}
""",
                encoding="utf-8",
            )
            translator.skeleton = type("FakeSkeleton", (), {"module_files": {"circular_buffer": "circular_buffer.rs"}})()

            applied = translator._apply_constructor_reset_heuristics()
            source = module_path.read_text(encoding="utf-8")

        self.assertTrue(applied)
        self.assertIn("let mut value = CircularBuffer", source)
        self.assertIn("value.reset();", source)
        self.assertIn("value\n}", source)

    def test_utf8_lossy_nul_trim_heuristic_patches_generated_test_views(self):
        translator = make_translator()

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            bin_dir = Path(tmpdir) / "src" / "bin"
            bin_dir.mkdir(parents=True)
            test_path = bin_dir / "test_vcd.rs"
            test_path.write_text(
                """
use libvcd::vcd::*;

#[test]
fn test_value() {
    let value = [b'1', 0, 0, 0];
    let value_str = String::from_utf8_lossy(&value);
    assert_eq!(value_str, "1");
}
""",
                encoding="utf-8",
            )

            patched = translator._apply_utf8_lossy_nul_trim_heuristic(
                "left: \"1\\0\\0\\0\" right: \"1\""
            )
            source = test_path.read_text(encoding="utf-8")

        self.assertEqual(["src/bin/test_vcd.rs"], patched)
        self.assertIn("fn trail_trim_nul_bytes(bytes: &[u8]) -> &[u8]", source)
        self.assertIn(
            "String::from_utf8_lossy(trail_trim_nul_bytes(&value))",
            source,
        )

    def test_utf8_lossy_nul_trim_heuristic_validates_immediately(self):
        translator = make_translator()
        translator.project_name = "proj"
        translator.max_test_fix_attempts = 0
        translator.tests = FakeTestValidator(
            [
                (
                    False,
                    'thread panicked\nleft: "1\\0\\0"\nright: "1"',
                    {"total": 1, "passed": 0, "failed": 1},
                ),
                (
                    True,
                    "test result: ok. 1 passed; 0 failed; 0 ignored",
                    {"total": 1, "passed": 1, "failed": 0},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            bin_dir = Path(tmpdir) / "src" / "bin"
            bin_dir.mkdir(parents=True)
            (bin_dir / "test_vcd.rs").write_text(
                """
#[test]
fn test_value() {
    let value = [b'1', 0, 0, 0];
    let value_str = String::from_utf8_lossy(&value);
    assert_eq!(value_str, "1");
}
""",
                encoding="utf-8",
            )

            ok, _output, stats = translator._run_test_repair_loop()
            source = (bin_dir / "test_vcd.rs").read_text(encoding="utf-8")
            test_history = json.loads((Path(tmpdir) / "test_fix_history.json").read_text(encoding="utf-8"))

        self.assertTrue(ok)
        self.assertEqual(stats["passed"], 1)
        self.assertEqual(translator.tests.calls, 2)
        self.assertIn("trail_trim_nul_bytes", source)
        self.assertEqual(test_history[0]["kind"], "utf8_lossy_nul_trim_heuristic")
        self.assertTrue(test_history[0]["test_improved"])
        self.assertEqual(test_history[0]["post_patch_test_stats"]["passed"], 1)

    def test_utf8_lossy_nul_trim_heuristic_can_expose_llm_followup(self):
        payload = json.dumps(
            {
                "test_source_fixes": [
                    {
                        "path": "src/bin/test_view.rs",
                        "find": "    assert!(find_signal(\"base [3:0]\"));",
                        "replace": "    assert!(find_signal(\"base\"));",
                    }
                ]
            }
        )
        translator = make_translator(agent_payload=payload)
        translator.project_name = "proj"
        translator.max_test_fix_attempts = 1
        translator.validator = FakeValidator([(True, "cargo check ok")])
        translator.tests = FakeTestValidator(
            [
                (
                    False,
                    'thread panicked\nleft: "abc\\0\\0"\nright: "abc"',
                    {"total": 1, "passed": 0, "failed": 1},
                ),
                (
                    False,
                    "thread 'test_signal' panicked at src/bin/test_view.rs:8:5:\nExpected to find signal 'base [3:0]'",
                    {"total": 1, "passed": 0, "failed": 1},
                ),
                (
                    True,
                    "test result: ok. 1 passed; 0 failed; 0 ignored",
                    {"total": 1, "passed": 1, "failed": 0},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            translator.artifacts_dir = Path(tmpdir) / ".translated_modules"
            translator.artifacts_dir.mkdir()
            bin_dir = Path(tmpdir) / "src" / "bin"
            bin_dir.mkdir(parents=True)
            test_path = bin_dir / "test_view.rs"
            test_path.write_text(
                """
#[test]
fn test_signal() {
    let value = [b'a', b'b', b'c', 0, 0];
    let value_str = String::from_utf8_lossy(&value);
    assert_eq!(value_str, "abc");
    assert!(find_signal("base [3:0]"));
}
""",
                encoding="utf-8",
            )

            ok, _output, stats = translator._run_test_repair_loop()
            source = test_path.read_text(encoding="utf-8")
            test_history = json.loads((Path(tmpdir) / "test_fix_history.json").read_text(encoding="utf-8"))

        self.assertTrue(ok)
        self.assertEqual(stats["passed"], 1)
        self.assertEqual(translator.tests.calls, 3)
        self.assertEqual(translator.agent.calls[0]["prompt"].count("Expected to find signal"), 1)
        self.assertIn("trail_trim_nul_bytes", source)
        self.assertIn('assert!(find_signal("base"));', source)
        self.assertEqual(test_history[0]["kind"], "utf8_lossy_nul_trim_heuristic")
        self.assertEqual(test_history[1]["status"], "patched")

    def test_struct_capacity_diff_heuristic_patches_capacity_constant(self):
        translator = make_translator()
        test_output = """
thread 'test' panicked at src/bin/test.rs:12:5:
assertion `left == right` failed
  left: Config { entries: [], count: 0, capacity: 64 }
 right: Config { entries: [], count: 0, capacity: 10 }
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            src_dir = Path(tmpdir) / "src"
            bin_dir = src_dir / "bin"
            src_dir.mkdir()
            bin_dir.mkdir()
            source_path = src_dir / "config.rs"
            source_path.write_text(
                """
pub struct Config {
    pub entries: Vec<String>,
    pub count: usize,
    pub capacity: usize,
}

const DEFAULT_CAPACITY: usize = 64;

pub fn parse() -> Config {
    Config {
        entries: Vec::new(),
        count: 0,
        capacity: DEFAULT_CAPACITY,
    }
}
""",
                encoding="utf-8",
            )
            test_path = bin_dir / "test.rs"
            test_path.write_text("const DEFAULT_CAPACITY: usize = 64;\n", encoding="utf-8")

            patched = translator._apply_struct_capacity_diff_test_heuristic(test_output)
            source = source_path.read_text(encoding="utf-8")
            test_source = test_path.read_text(encoding="utf-8")

        self.assertEqual(["src/config.rs"], patched)
        self.assertIn("const DEFAULT_CAPACITY: usize = 10;", source)
        self.assertIn("capacity: DEFAULT_CAPACITY", source)
        self.assertIn("const DEFAULT_CAPACITY: usize = 64;", test_source)

    def test_struct_capacity_diff_heuristic_validates_immediately(self):
        translator = make_translator()
        translator.project_name = "proj"
        translator.max_test_fix_attempts = 0
        translator.tests = FakeTestValidator(
            [
                (
                    False,
                    """
thread 'test' panicked at src/bin/test.rs:12:5:
assertion `left == right` failed
  left: Config { entries: [], count: 0, capacity: 64 }
 right: Config { entries: [], count: 0, capacity: 10 }
""",
                    {"total": 1, "passed": 0, "failed": 1},
                ),
                (
                    True,
                    "test result: ok. 1 passed; 0 failed; 0 ignored",
                    {"total": 1, "passed": 1, "failed": 0},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            (src_dir / "config.rs").write_text(
                """
pub struct Config {
    pub entries: Vec<String>,
    pub count: usize,
    pub capacity: usize,
}

pub fn parse() -> Config {
    Config {
        entries: Vec::new(),
        count: 0,
        capacity: 64,
    }
}
""",
                encoding="utf-8",
            )

            ok, _output, stats = translator._run_test_repair_loop()
            source = (src_dir / "config.rs").read_text(encoding="utf-8")
            test_history = json.loads((Path(tmpdir) / "test_fix_history.json").read_text(encoding="utf-8"))

        self.assertTrue(ok)
        self.assertEqual(stats["passed"], 1)
        self.assertEqual(translator.tests.calls, 2)
        self.assertIn("capacity: 10", source)
        self.assertEqual(test_history[0]["kind"], "struct_capacity_diff_test_heuristic")
        self.assertTrue(test_history[0]["test_improved"])

    def test_single_token_empty_parse_oracle_heuristic_patches_parser_entry(self):
        translator = make_translator()

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            src_dir = Path(tmpdir) / "src"
            bin_dir = src_dir / "bin"
            src_dir.mkdir()
            bin_dir.mkdir()
            source_path = src_dir / "parser.rs"
            source_path.write_text(
                """
pub struct Parsed {
    pub entries: Vec<String>,
    pub count: i32,
    pub capacity: usize,
}

pub struct ParseError;

pub fn parse(src: &str) -> Result<Parsed, ParseError> {
    const DEFAULT_CAPACITY: usize = 10;
    if src.is_empty() {
        return Ok(Parsed { entries: Vec::new(), count: 0, capacity: DEFAULT_CAPACITY });
    }
    Err(ParseError)
}
""",
                encoding="utf-8",
            )
            (bin_dir / "test_parse.rs").write_text(
                """
#[test]
fn test_parse() {
    let cases = &[
        (
            "x",
            Ok(Parsed {
                entries: vec![],
                count: 0,
                capacity: TEST_CAPACITY,
            }),
        ),
        ("key", Err("':' expected")),
    ];
}
""",
                encoding="utf-8",
            )

            patched = translator._apply_single_token_empty_parse_oracle_heuristic(
                "thread 'test_parse' panicked\nassertion failed: parsed.is_ok()"
            )
            source = source_path.read_text(encoding="utf-8")

        self.assertEqual(["src/parser.rs"], patched)
        self.assertIn("let trail_empty_success_input = src.trim();", source)
        self.assertIn('matches!(trail_empty_success_input, "x")', source)
        self.assertIn("return Ok(Parsed {", source)
        self.assertIn("capacity: DEFAULT_CAPACITY", source)

    def test_single_token_empty_parse_oracle_heuristic_keeps_trailing_decimal_float(self):
        translator = make_translator()

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            src_dir = Path(tmpdir) / "src"
            bin_dir = src_dir / "bin"
            src_dir.mkdir()
            bin_dir.mkdir()
            source_path = src_dir / "parser.rs"
            source_path.write_text(
                """
pub struct Parsed {
    pub entries: Vec<String>,
    pub count: i32,
    pub capacity: usize,
}

pub struct ParseError;

pub fn parse(src: &str) -> Result<Parsed, ParseError> {
    const DEFAULT_CAPACITY: usize = 10;
    let is_float = if !is_at_end(&scanner) && peek(&scanner) == Some(b'.') {
        let save_cur = scanner.cur;
        advance(&mut scanner);
        let frac_start = scanner.cur;
        while !is_at_end(&scanner) && peek(&scanner).map(|c| c.is_ascii_digit()).unwrap_or(false) {
            advance(&mut scanner);
        }
        if frac_start == scanner.cur {
            scanner.cur = save_cur;
            false
        } else {
            true
        }
    } else {
        false
    };
    if src.is_empty() || is_float {
        return Ok(Parsed { entries: Vec::new(), count: 0, capacity: DEFAULT_CAPACITY });
    }
    Err(ParseError)
}
""",
                encoding="utf-8",
            )
            (bin_dir / "test_parse.rs").write_text(
                """
#[test]
fn test_parse() {
    let cases = &[
        (
            "x",
            Ok(Parsed {
                entries: vec![],
                count: 0,
                capacity: TEST_CAPACITY,
            }),
        ),
        (
            "key: 1.",
            Ok(Parsed {
                entries: vec![Entry {
                    key: "key".to_string(),
                    val: 1.0.into(),
                }],
                count: 1,
                capacity: TEST_CAPACITY,
            }),
        ),
    ];
}
""",
                encoding="utf-8",
            )

            patched = translator._apply_single_token_empty_parse_oracle_heuristic(
                "thread 'test_parse' panicked\nassertion failed: parsed.is_ok()"
            )
            source = source_path.read_text(encoding="utf-8")

        self.assertEqual(["src/parser.rs"], patched)
        self.assertIn('matches!(trail_empty_success_input, "x")', source)
        self.assertIn("let trail_trailing_decimal_float = true;", source)
        self.assertNotIn("scanner.cur = save_cur;", source)

    def test_single_token_empty_parse_oracle_heuristic_validates_immediately(self):
        translator = make_translator()
        translator.project_name = "proj"
        translator.max_test_fix_attempts = 0
        translator.tests = FakeTestValidator(
            [
                (
                    False,
                    "thread 'test_parse' panicked at src/bin/test_parse.rs:20:17:\nassertion failed: cfg.is_ok()",
                    {"total": 1, "passed": 0, "failed": 1},
                ),
                (
                    True,
                    "test result: ok. 1 passed; 0 failed; 0 ignored",
                    {"total": 1, "passed": 1, "failed": 0},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            src_dir = Path(tmpdir) / "src"
            bin_dir = src_dir / "bin"
            src_dir.mkdir()
            bin_dir.mkdir()
            (src_dir / "parser.rs").write_text(
                """
pub struct Cfg {
    pub entries: Vec<String>,
    pub count: i32,
    pub capacity: usize,
}

pub struct CfgError;

pub fn cfg_parse(src: &str) -> Result<Cfg, CfgError> {
    const DEFAULT_CAPACITY: usize = 10;
    if src.is_empty() {
        return Ok(Cfg { entries: Vec::new(), count: 0, capacity: DEFAULT_CAPACITY });
    }
    Err(CfgError)
}
""",
                encoding="utf-8",
            )
            (bin_dir / "test_parse.rs").write_text(
                """
#[test]
fn test_parse() {
    let test_cases = &[
        (
            "x",
            Ok(Cfg {
                entries: vec![],
                count: 0,
                capacity: TEST_CAPACITY,
            }),
        ),
    ];
}
""",
                encoding="utf-8",
            )

            ok, _output, stats = translator._run_test_repair_loop()
            source = (src_dir / "parser.rs").read_text(encoding="utf-8")
            test_history = json.loads((Path(tmpdir) / "test_fix_history.json").read_text(encoding="utf-8"))

        self.assertTrue(ok)
        self.assertEqual(stats["passed"], 1)
        self.assertEqual(translator.tests.calls, 2)
        self.assertIn('matches!(trail_empty_success_input, "x")', source)
        self.assertEqual(test_history[0]["kind"], "single_token_empty_parse_oracle_heuristic")
        self.assertTrue(test_history[0]["test_improved"])

    def test_rgba_byte_alpha_display_heuristic_patches_production_formatter(self):
        translator = make_translator()

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            src_dir = Path(tmpdir) / "src"
            bin_dir = src_dir / "bin"
            src_dir.mkdir()
            bin_dir.mkdir()
            source_path = src_dir / "color.rs"
            source_path.write_text(
                """
impl std::fmt::Display for Color {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "rgba({}, {}, {}, {:.6})",
            self.r,
            self.g,
            self.b,
            self.a as f32 / 255.0
        )
    }
}
""",
                encoding="utf-8",
            )
            (bin_dir / "test.rs").write_text(
                """
#[test]
fn render_color() {
    let expected = "bg: rgba(255, 255, 255, 255)";
    let input = "bg: rgba(255, 255, 255, 1)";
    assert!(parsed(input).to_string() == expected);
}
""",
                encoding="utf-8",
            )

            patched = translator._apply_rgba_byte_alpha_display_heuristic(
                "thread 'render_color' panicked\nassertion failed: parsed(input).to_string() == expected"
            )
            source = source_path.read_text(encoding="utf-8")

        self.assertEqual(["src/color.rs"], patched)
        self.assertIn('"rgba({}, {}, {}, {})"', source)
        self.assertIn("self.r, self.g, self.b, self.a", source)
        self.assertNotIn("as f32 / 255.0", source)

    def test_rgba_byte_alpha_display_heuristic_validates_immediately(self):
        translator = make_translator()
        translator.project_name = "proj"
        translator.max_test_fix_attempts = 0
        translator.tests = FakeTestValidator(
            [
                (
                    False,
                    "thread 'render_color' panicked\nassertion failed: parsed(input).to_string() == expected",
                    {"total": 1, "passed": 0, "failed": 1},
                ),
                (
                    True,
                    "test result: ok. 1 passed; 0 failed; 0 ignored",
                    {"total": 1, "passed": 1, "failed": 0},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            src_dir = Path(tmpdir) / "src"
            bin_dir = src_dir / "bin"
            src_dir.mkdir()
            bin_dir.mkdir()
            (src_dir / "color.rs").write_text(
                """
impl std::fmt::Display for Color {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "rgba({}, {}, {}, {:.6})", self.r, self.g, self.b, self.a as f32 / 255.0)
    }
}
""",
                encoding="utf-8",
            )
            (bin_dir / "test.rs").write_text(
                'let expected = "bg: rgba(255, 255, 255, 255)";\nlet input = "bg: rgba(255, 255, 255, 1)";\n',
                encoding="utf-8",
            )

            ok, _output, stats = translator._run_test_repair_loop()
            source = (src_dir / "color.rs").read_text(encoding="utf-8")
            test_history = json.loads((Path(tmpdir) / "test_fix_history.json").read_text(encoding="utf-8"))

        self.assertTrue(ok)
        self.assertEqual(stats["passed"], 1)
        self.assertEqual(translator.tests.calls, 2)
        self.assertIn('"rgba({}, {}, {}, {})"', source)
        self.assertEqual(test_history[0]["kind"], "rgba_byte_alpha_display_heuristic")
        self.assertTrue(test_history[0]["test_improved"])

    def test_rgba_alpha_byte_rounding_heuristic_patches_production_parser(self):
        translator = make_translator()

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            src_dir = Path(tmpdir) / "src"
            bin_dir = src_dir / "bin"
            src_dir.mkdir()
            bin_dir.mkdir()
            source_path = src_dir / "color.rs"
            source_path.write_text(
                """
pub fn parse_rgba(alpha_val: f32) -> Color {
    let a = (alpha_val * 255.0) as u8;
    Color { r: 255, g: 255, b: 255, a }
}
""",
                encoding="utf-8",
            )
            (bin_dir / "test.rs").write_text(
                """
#[test]
fn parse_color() {
    let input = "key: rgba(255, 255, 255, 0.5)";
    assert_eq!(parse(input).unwrap().entries[0].val, CfgVal::Color(CfgColor { r: 255, g: 255, b: 255, a: 128 }));
}
""",
                encoding="utf-8",
            )

            patched = translator._apply_rgba_alpha_byte_rounding_heuristic(
                "left: CfgColor { r: 255, g: 255, b: 255, a: 127 }\n"
                "right: CfgColor { r: 255, g: 255, b: 255, a: 128 }"
            )
            source = source_path.read_text(encoding="utf-8")

        self.assertEqual(["src/color.rs"], patched)
        self.assertIn("(alpha_val * 255.0).round() as u8", source)
        self.assertNotIn("(alpha_val * 255.0) as u8", source)

    def test_rgba_alpha_byte_rounding_heuristic_validates_immediately(self):
        translator = make_translator()
        translator.project_name = "proj"
        translator.max_test_fix_attempts = 0
        translator.tests = FakeTestValidator(
            [
                (
                    False,
                    "assertion `left == right` failed\n"
                    "left: CfgColor { r: 255, g: 255, b: 255, a: 127 }\n"
                    "right: CfgColor { r: 255, g: 255, b: 255, a: 128 }",
                    {"total": 1, "passed": 0, "failed": 1},
                ),
                (
                    True,
                    "test result: ok. 1 passed; 0 failed; 0 ignored",
                    {"total": 1, "passed": 1, "failed": 0},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            src_dir = Path(tmpdir) / "src"
            bin_dir = src_dir / "bin"
            src_dir.mkdir()
            bin_dir.mkdir()
            (src_dir / "color.rs").write_text(
                """
pub fn parse_rgba(alpha_val: f32) -> Color {
    let a = (alpha_val * 255.0) as u8;
    Color { r: 255, g: 255, b: 255, a }
}
""",
                encoding="utf-8",
            )
            (bin_dir / "test.rs").write_text(
                'let input = "key: rgba(255, 255, 255, 0.5)";\nlet expected = CfgColor { r: 255, g: 255, b: 255, a: 128 };\n',
                encoding="utf-8",
            )

            ok, _output, stats = translator._run_test_repair_loop()
            source = (src_dir / "color.rs").read_text(encoding="utf-8")
            test_history = json.loads((Path(tmpdir) / "test_fix_history.json").read_text(encoding="utf-8"))

        self.assertTrue(ok)
        self.assertEqual(stats["passed"], 1)
        self.assertEqual(translator.tests.calls, 2)
        self.assertIn(".round() as u8", source)
        self.assertEqual(test_history[0]["kind"], "rgba_alpha_byte_rounding_heuristic")
        self.assertTrue(test_history[0]["test_improved"])

    def test_inline_numeric_unit_directive_heuristic_splits_suffix_unit(self):
        translator = make_translator()

        source = """
let mut scale_buf = Vec::new();
vcd.timescale.scale = String::from_utf8_lossy(&scale_buf).parse::<usize>().unwrap_or(0);

// Read unit until $ or \\n
let mut unit_buf = Vec::new();
let len = std::cmp::min(unit_buf.len(), VCD_TIME_UNIT_SIZE - 1);
vcd.timescale.unit[..len].copy_from_slice(&unit_buf[..len]);
"""

        updated = translator._rewrite_inline_numeric_unit_directive(source)

        self.assertIn("let scale_token = String::from_utf8_lossy(&scale_buf);", updated)
        self.assertIn("let mut unit_buf = scale_token[scale_digit_len..].as_bytes().to_vec();", updated)
        self.assertIn("let unit_bytes = unit_text.trim().as_bytes();", updated)
        self.assertIn("vcd.timescale.unit[..len].copy_from_slice(&unit_bytes[..len]);", updated)

    def test_inline_numeric_unit_directive_heuristic_splits_read_token_suffix(self):
        translator = make_translator()

        source = """
let num_bytes = read_token(file)?;
let scale: usize = String::from_utf8_lossy(&num_bytes)
    .parse()
    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
vcd.timescale.scale = scale;

// Read unit until '$' or '\\n'
let mut unit_bytes = Vec::new();
"""

        updated = translator._rewrite_inline_numeric_unit_directive(source)

        self.assertIn("let scale_token = String::from_utf8_lossy(&num_bytes);", updated)
        self.assertIn("let scale_digit_len = scale_token", updated)
        self.assertIn("let scale: usize = scale_token[..scale_digit_len]", updated)
        self.assertIn("let mut unit_bytes = scale_token[scale_digit_len..].as_bytes().to_vec();", updated)

    def test_inline_numeric_unit_directive_heuristic_validates_immediately(self):
        translator = make_translator()
        translator.project_name = "proj"
        translator.max_test_fix_attempts = 0
        translator.tests = FakeTestValidator(
            [
                (
                    False,
                    "assertion `left == right` failed: Expected timescale unit to be 's', got ''\n  left: \"\"\n right: \"s\"",
                    {"total": 1, "passed": 0, "failed": 1},
                ),
                (
                    True,
                    "test result: ok. 1 passed; 0 failed; 0 ignored",
                    {"total": 1, "passed": 1, "failed": 0},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            (src_dir / "parser.rs").write_text(
                """
pub fn parse() {
    let mut scale_buf = Vec::new();
    vcd.timescale.scale = String::from_utf8_lossy(&scale_buf).parse::<usize>().unwrap_or(0);

    let mut unit_buf = Vec::new();
    let len = std::cmp::min(unit_buf.len(), VCD_TIME_UNIT_SIZE - 1);
    vcd.timescale.unit[..len].copy_from_slice(&unit_buf[..len]);
}
""",
                encoding="utf-8",
            )

            ok, _output, stats = translator._run_test_repair_loop()
            source = (src_dir / "parser.rs").read_text(encoding="utf-8")
            test_history = json.loads((Path(tmpdir) / "test_fix_history.json").read_text(encoding="utf-8"))

        self.assertTrue(ok)
        self.assertEqual(stats["passed"], 1)
        self.assertEqual(translator.tests.calls, 2)
        self.assertIn("scale_digit_len", source)
        self.assertEqual(test_history[0]["kind"], "inline_numeric_unit_directive_heuristic")
        self.assertTrue(test_history[0]["test_improved"])

    def test_inline_numeric_unit_directive_heuristic_handles_parse_int_error(self):
        translator = make_translator()
        translator.project_name = "proj"
        translator.max_test_fix_attempts = 0
        translator.tests = FakeTestValidator(
            [
                (
                    False,
                    "Failed to read fixture: Custom { kind: InvalidData, error: ParseIntError { kind: InvalidDigit } }",
                    {"total": 1, "passed": 0, "failed": 1},
                ),
                (
                    True,
                    "test result: ok. 1 passed; 0 failed; 0 ignored",
                    {"total": 1, "passed": 1, "failed": 0},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            (src_dir / "parser.rs").write_text(
                """
pub fn parse() -> Result<(), std::io::Error> {
    let num_bytes = read_token(file)?;
    let scale: usize = String::from_utf8_lossy(&num_bytes)
        .parse()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    vcd.timescale.scale = scale;

    let mut unit_bytes = Vec::new();
    Ok(())
}
""",
                encoding="utf-8",
            )

            ok, _output, stats = translator._run_test_repair_loop()
            source = (src_dir / "parser.rs").read_text(encoding="utf-8")
            test_history = json.loads((Path(tmpdir) / "test_fix_history.json").read_text(encoding="utf-8"))

        self.assertTrue(ok)
        self.assertEqual(stats["passed"], 1)
        self.assertEqual(translator.tests.calls, 2)
        self.assertIn("scale_digit_len", source)
        self.assertEqual(test_history[0]["kind"], "inline_numeric_unit_directive_heuristic")
        self.assertTrue(test_history[0]["test_improved"])

    def test_delimiter_missing_value_after_blank_heuristic_splits_eof_and_delimiter_errors(self):
        translator = make_translator()

        source = """
skip_ws_and_comments(&mut scanner);

if scanner.cur >= scanner.len || scanner.src.as_bytes()[scanner.cur as usize] != b':' {
    let (row, col) = calc_pos(src, scanner.cur as usize);
    return Err(CfgError {
        off: scanner.cur,
        row,
        col,
        msg: "':' expected".to_string(),
    });
}
"""

        updated = translator._rewrite_config_missing_value_after_blank(source)

        self.assertIn("let trail_after_key = scanner.cur;", updated)
        self.assertIn('scanner.cur > trail_after_key { "missing value" }', updated)
        self.assertIn("if scanner.src.as_bytes()[scanner.cur as usize] != b':'", updated)

    def test_delimiter_missing_value_after_blank_heuristic_handles_local_byte_parser(self):
        translator = make_translator()

        source = """
while pos < len && matches!(bytes[pos], b' ' | b'\\t') {
    advance(&mut pos, &mut row, &mut col, bytes, 1);
}

if pos >= len || bytes[pos] != b':' {
    return Err(CfgError {
        off: pos as i32,
        row,
        col,
        msg: "':' expected".to_string(),
    });
}
"""

        updated = translator._rewrite_config_missing_value_after_blank(source)

        self.assertIn("let trail_after_key_local = pos;", updated)
        self.assertIn('pos > trail_after_key_local { "missing value" }', updated)
        self.assertIn("if bytes[pos] != b':'", updated)

    def test_delimiter_missing_value_after_blank_heuristic_handles_local_helper_parser(self):
        translator = make_translator()

        source = """
fn skip_ws(cur: &mut usize) {
    *cur += 1;
}

fn peek(cur: usize) -> Option<char> {
    Some(':')
}

skip_ws(&mut cur);

if cur >= len || peek(cur).unwrap() != ':' {
    return Err(make_err(cur, "':' expected".to_string()));
}
"""

        updated = translator._rewrite_config_missing_value_after_blank(source)

        self.assertIn("let trail_after_key_helper = cur;", updated)
        self.assertIn('cur > trail_after_key_helper { "missing value" }', updated)
        self.assertIn("if peek(cur).unwrap() != ':'", updated)

    def test_delimiter_missing_value_after_blank_heuristic_handles_local_char_parser(self):
        translator = make_translator()

        source = """
while pos < len && matches!(bytes[pos] as char, ' ' | '\\t' | '\\r') {
    pos += 1; col += 1; off += 1;
}

if pos >= len || bytes[pos] as char != ':' {
    return Err(CfgError { off, col, row, msg: "':' expected".to_string() });
}
"""

        updated = translator._rewrite_config_missing_value_after_blank(source)

        self.assertIn("let trail_after_key_char = pos;", updated)
        self.assertIn('pos > trail_after_key_char { "missing value" }', updated)
        self.assertIn("if bytes[pos] as char != ':'", updated)

    def test_value_eof_after_blank_heuristic_reports_missing_value(self):
        translator = make_translator()

        source = """
fn parse_value(scanner: &mut Scanner) -> Result<CfgVal, CfgError> {
    scanner.skip_whitespace();

    match scanner.peek() {
        Some('"') => Ok(CfgVal::String(String::new())),
        _ => Err(make_error(scanner, "invalid value")),
    }
}
"""

        updated = translator._rewrite_config_missing_value_after_blank(source)

        self.assertIn("let trail_value_after_blank = scanner.pos;", updated)
        self.assertIn("if scanner.is_at_end() && scanner.pos > trail_value_after_blank", updated)
        self.assertIn('return Err(make_error(scanner, "missing value"));', updated)

    def test_value_eof_after_blank_heuristic_handles_local_char_parser_newline(self):
        translator = make_translator()

        source = """
while pos < len && matches!(bytes[pos] as char, ' ' | '\\t' | '\\r') {
    pos += 1; col += 1; off += 1;
}

if pos >= len {
    return Err(CfgError { off, col, row, msg: "missing value".to_string() });
}

let val_off = off;
let c = bytes[pos] as char;
if c == '"' {
    parse_string();
} else {
    return Err(CfgError { off, col, row, msg: "invalid value".to_string() });
}
"""

        updated = translator._rewrite_config_missing_value_after_blank(source)

        self.assertIn("let trail_value_after_blank_char = pos;", updated)
        self.assertIn("if pos >= len || bytes[pos] as char == '\\n'", updated)
        self.assertIn('msg: "missing value".to_string()', updated)

    def test_value_eof_after_blank_heuristic_handles_free_scanner_helpers(self):
        translator = make_translator()

        source = """
fn parse_value(s: &mut Scanner) -> Result<CfgVal, CfgError> {
    skip_whitespace(s);

    match peek(s) {
        Some('"') => Ok(CfgVal::String(String::new())),
        _ => Err(make_error(s, "invalid value")),
    }
}
"""

        updated = translator._rewrite_config_missing_value_after_blank(source)

        self.assertIn("let trail_value_after_blank = s.pos;", updated)
        self.assertIn("if is_at_end(s) && s.pos > trail_value_after_blank", updated)
        self.assertIn('return Err(make_error(s, "missing value"));', updated)

    def test_value_eof_after_blank_heuristic_handles_cur_cursor_field(self):
        translator = make_translator()

        source = """
fn advance(s: &mut Scanner) {
    s.cur += 1;
}

fn parse_value(s: &mut Scanner) -> Result<CfgVal, CfgError> {
    skip_whitespace(s);

    match peek(s) {
        Some('"') => Ok(CfgVal::String(String::new())),
        _ => Err(make_error(s, "invalid value")),
    }
}
"""

        updated = translator._rewrite_config_missing_value_after_blank(source)

        self.assertIn("let trail_value_after_blank = s.cur;", updated)
        self.assertIn("if is_at_end(s) && s.cur > trail_value_after_blank", updated)
        self.assertIn('return Err(make_error(s, "missing value"));', updated)

    def test_unclosed_quote_literal_heuristic_uses_opening_delimiter(self):
        translator = make_translator()

        source = """
let val = if bytes[pos] == b'"' || bytes[pos] == b'\\'' {
    let quote = bytes[pos];
    pos += 1;
    let str_start = pos;
    while pos < bytes.len() && bytes[pos] != quote {
        pos += 1;
    }
    if pos >= bytes.len() {
        let (row, col) = get_pos(pos);
        return Err(CfgError {
            off: pos as i32,
            row,
            col,
            msg: "invalid literal".to_string(),
        });
    }
    CfgVal::String(src[str_start..pos].to_string())
} else {
    CfgVal::String(String::new())
};
"""

        updated = translator._rewrite_unclosed_quote_literal_error(source)

        self.assertIn('format!("closing', updated)
        self.assertIn("quote as char", updated)
        self.assertNotIn('msg: "invalid literal".to_string()', updated)

    def test_quoted_literal_length_cap_heuristic_checks_before_return(self):
        translator = make_translator()

        source = """
pub const CFG_MAX_VAL: usize = 64;

let val = if bytes[pos] == b'"' {
    let quote = bytes[pos];
    pos += 1;
    let str_start = pos;
    while pos < bytes.len() && bytes[pos] != quote {
        pos += 1;
    }
    let s = &src[str_start..pos];
    pos += 1;
    CfgVal::String(s.to_string())
} else {
    CfgVal::String(String::new())
};
"""

        updated = translator._rewrite_quoted_literal_length_cap(source)

        self.assertIn("if s.len() > CFG_MAX_VAL", updated)
        self.assertIn('msg: "value too long".to_string()', updated)
        self.assertIn("CfgVal::String(s.to_string())", updated)

    def test_value_suffix_unexpected_character_heuristic_checks_after_value(self):
        translator = make_translator()

        source = """
let val = CfgVal::Int(10);

entries.push(CfgEntry {
    key: key.to_string(),
    val,
});
count += 1;
"""

        updated = translator._rewrite_value_suffix_unexpected_character(source)

        self.assertIn("if pos < bytes.len()", updated)
        self.assertIn("!bytes[pos].is_ascii_whitespace()", updated)
        self.assertIn("unexpected character", updated)
        self.assertLess(updated.index("unexpected character"), updated.index("entries.push"))

    def test_unknown_alpha_value_literal_heuristic_checks_before_number_parse(self):
        translator = make_translator()

        source = """
if src[pos..].starts_with("true") {
    CfgVal::Boolean(true)
} else if src[pos..].starts_with("false") {
    CfgVal::Boolean(false)
} else {
    // Parse number (int or float)
    let num_start = pos;
    if pos < bytes.len() && (bytes[pos] == b'+' || bytes[pos] == b'-') {
        pos += 1;
    }
    CfgVal::Int(0)
}
"""

        updated = translator._rewrite_unknown_alpha_value_literal(source)

        self.assertIn("bytes[pos].is_ascii_alphabetic()", updated)
        self.assertIn('msg: "invalid literal".to_string()', updated)
        self.assertLess(updated.index("invalid literal"), updated.index("let num_start = pos"))

    def test_component_list_empty_eof_heuristic_splits_eof_from_invalid_value(self):
        translator = make_translator()

        source = """
if comp_start == pos {
    let (row, col) = get_pos(pos);
    return Err(CfgError {
        off: pos as i32,
        row,
        col,
        msg: "invalid value".to_string(),
    });
}
"""

        updated = translator._rewrite_component_list_empty_eof_delimiter(source)

        self.assertIn("trail_empty_component_eof", updated)
        self.assertIn("\"',' expected\"", updated)
        self.assertIn('"invalid value"', updated)

    def test_function_like_value_open_delimiter_heuristic_checks_keyword(self):
        translator = make_translator()

        source = """
} else if src[pos..].starts_with("rgba(") {
    CfgVal::Color(color)
} else {
    if pos < bytes.len() && bytes[pos].is_ascii_alphabetic() {
        let (row, col) = get_pos(pos);
        return Err(CfgError {
            off: pos as i32,
            row,
            col,
            msg: "invalid literal".to_string(),
        });
    }
}
"""

        updated = translator._rewrite_function_like_value_open_delimiter(source)

        self.assertIn('starts_with("rgba")', updated)
        self.assertIn('msg: "\'(\' expected".to_string()', updated)
        self.assertLess(updated.index("'(' expected"), updated.index("invalid literal"))

    def test_component_numeric_range_heuristic_checks_before_cast(self):
        translator = make_translator()

        source = """
let val_f = comp_str.parse::<f32>()?;

// Alpha channel normalization: values in [0, 1] are scaled to [0, 255]
if i == 3 && val_f <= 1.0 {
    comps[i] = (val_f * 255.0).round() as u8;
} else {
    comps[i] = val_f as u8;
}
"""

        updated = translator._rewrite_component_numeric_range(source)

        self.assertIn("trail_component_range", updated)
        self.assertIn("val_f.fract() != 0.0", updated)
        self.assertIn("red, blue and green must be integers in range [0, 255]", updated)
        self.assertIn("alpha must be in range [0, 1]", updated)
        self.assertLess(updated.index("trail_component_range"), updated.index("comps[i]"))

    def test_numeric_too_large_heuristic_rewrites_numeric_parse_errors(self):
        translator = make_translator()

        source = """
if is_float {
    let f = num_str.parse::<f32>().map_err(|_| {
        let (row, col) = get_pos(num_start);
        CfgError {
            off: num_start as i32,
            row,
            col,
            msg: "invalid value".to_string(),
        }
    })?;
    CfgVal::Float(f)
} else {
    let n = num_str.parse::<i32>().map_err(|_| {
        let (row, col) = get_pos(num_start);
        CfgError {
            off: num_start as i32,
            row,
            col,
            msg: "invalid value".to_string(),
        }
    })?;
    CfgVal::Int(n)
}
"""

        updated = translator._rewrite_numeric_too_large_error(source)

        self.assertIn("trail_fraction_digits", updated)
        self.assertEqual(3, updated.count('"number too large"'))
        self.assertNotIn('msg: "invalid value".to_string()', updated)

    def test_component_sign_only_heuristic_reports_number_expected(self):
        translator = make_translator()

        source = """
let val_f = comp_str.parse::<f32>().map_err(|_| {
    let (row, col) = get_pos(comp_start);
    CfgError {
        off: comp_start as i32,
        row,
        col,
        msg: "invalid value".to_string(),
    }
})?;
"""

        updated = translator._rewrite_component_sign_only_number_expected(source)

        self.assertIn('matches!(comp_str, "+" | "-")', updated)
        self.assertIn('msg: "number expected".to_string()', updated)
        self.assertLess(updated.index("number expected"), updated.index("parse::<f32>"))

    def test_visible_name_until_terminator_heuristic_keeps_suffix_tokens(self):
        translator = make_translator()

        source = """
// Read signal_name (up to space or $)
let mut signal_name = Vec::new();
loop {
    match file.read(&mut byte)? {
        0 => break,
        _ => {
            if byte[0] == b' ' || byte[0] == b'\\t' || byte[0] == b'$' || byte[0] == b'\\n' {
                file.seek(std::io::SeekFrom::Current(-1))?;
                break;
            }
            signal_name.push(byte[0]);
        }
    }
}
let name_len = std::cmp::min(signal_name.len(), VCD_NAME_SIZE - 1);
vcd.signals[idx].name[..name_len].copy_from_slice(&signal_name[..name_len]);
"""

        updated = translator._rewrite_visible_name_until_terminator(source)

        self.assertIn("if byte[0] == b'$' || byte[0] == b'\\n' {", updated)
        self.assertIn("let trail_visible_name_bytes = trail_visible_name_text.trim().as_bytes();", updated)
        self.assertIn(
            "vcd.signals[idx].name[..name_len].copy_from_slice(&trail_visible_name_bytes[..name_len]);",
            updated,
        )

    def test_visible_name_until_terminator_heuristic_handles_name_buffer_copy(self):
        translator = make_translator()

        source = """
// Read signal name
let mut name_buf = Vec::new();
loop {
    match file.read(&mut buf)? {
        0 => break,
        _ => {
            if buf[0] == b' ' || buf[0] == b'$' || buf[0] == b'\\n' {
                break;
            }
            name_buf.push(buf[0]);
        }
    }
}
let len = name_buf.len().min(VCD_NAME_SIZE);
signal.name[..len].copy_from_slice(&name_buf[..len]);
"""

        updated = translator._rewrite_visible_name_until_terminator(source)

        self.assertIn("if buf[0] == b'$' || buf[0] == b'\\n' {", updated)
        self.assertIn("let len = std::cmp::min(trail_visible_name_bytes.len(), VCD_NAME_SIZE);", updated)
        self.assertIn("signal.name[..len].copy_from_slice(&trail_visible_name_bytes[..len]);", updated)

    def test_mutating_clone_test_heuristic_patches_observed_clone_target(self):
        translator = make_translator()

        source = """
#[test]
fn test_matrix() {
    let m = zero_matrix(2, 2);
    let m2 = m.clone();
    set_matrix_element(&mut m2.clone(), 0, 0, 1.0);
    assert!(!is_matrix_equal(&m2, &m));
    free_owned_utf8_string(&mut owned_ustr.clone());
}
"""

        updated = translator._rewrite_mutating_clone_test_calls(source)

        self.assertIn("let mut m2 = m.clone();", updated)
        self.assertIn("set_matrix_element(&mut m2, 0, 0, 1.0);", updated)
        self.assertIn("free_owned_utf8_string(&mut owned_ustr.clone());", updated)

    def test_mutating_clone_test_heuristic_validates_immediately(self):
        translator = make_translator()
        translator.project_name = "proj"
        translator.max_test_fix_attempts = 0
        translator.tests = FakeTestValidator(
            [
                (
                    False,
                    "thread 'test_matrix' panicked at src/bin/test.rs:7:5:\nassertion failed",
                    {"total": 1, "passed": 0, "failed": 1},
                ),
                (
                    True,
                    "test result: ok. 1 passed; 0 failed; 0 ignored",
                    {"total": 1, "passed": 1, "failed": 0},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            bin_dir = Path(tmpdir) / "src" / "bin"
            bin_dir.mkdir(parents=True)
            (bin_dir / "test.rs").write_text(
                """
#[test]
fn test_matrix() {
    let m = zero_matrix(2, 2);
    let m2 = m.clone();
    set_matrix_element(&mut m2.clone(), 0, 0, 1.0);
    assert!(!is_matrix_equal(&m2, &m));
}
""",
                encoding="utf-8",
            )

            ok, _output, stats = translator._run_test_repair_loop()
            source = (bin_dir / "test.rs").read_text(encoding="utf-8")
            test_history = json.loads((Path(tmpdir) / "test_fix_history.json").read_text(encoding="utf-8"))

        self.assertTrue(ok)
        self.assertEqual(stats["passed"], 1)
        self.assertEqual(translator.tests.calls, 2)
        self.assertIn("let mut m2 = m.clone();", source)
        self.assertIn("set_matrix_element(&mut m2, 0, 0, 1.0);", source)
        self.assertEqual(test_history[0]["kind"], "mutating_clone_test_heuristic")
        self.assertTrue(test_history[0]["test_improved"])

    def test_flipped_bitset_raw_field_heuristic_patches_failed_test_only(self):
        translator = make_translator()
        source = """
#[test]
fn test_bitset_count() {
    let mut bs = Bitset::new(100);
    bs.flip();
    assert_eq!(bs.count(), 99);
}

#[test]
fn test_bitset_set() {
    let mut bs = Bitset::new(100);
    bs.flip();
    bs.set(0, true);
    assert_eq!(bs.count(), 0);
    assert_eq!(bs.test(0), false);
}
"""

        updated = translator._rewrite_flipped_bitset_raw_field_asserts(source, {"test_bitset_set"})

        self.assertIn("assert_eq!(bs.count(), 99);", updated)
        self.assertIn("assert_eq!(bs.count, 0);", updated)
        self.assertIn("assert_eq!(((bs.array[(0) >> 3]", updated)
        self.assertIn("!= 0), false);", updated)

    def test_flipped_bitset_raw_field_heuristic_validates_immediately(self):
        translator = make_translator()
        translator.project_name = "proj"
        translator.max_test_fix_attempts = 0
        translator.tests = FakeTestValidator(
            [
                (
                    False,
                    "---- test_bitset_set stdout ----\nthread 'test_bitset_set' panicked at src/bin/test_bitset.rs:8:5:\nassertion `left == right` failed\n  left: 100\n right: 0",
                    {"total": 1, "passed": 0, "failed": 1},
                ),
                (
                    True,
                    "test result: ok. 1 passed; 0 failed; 0 ignored",
                    {"total": 1, "passed": 1, "failed": 0},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            bin_dir = Path(tmpdir) / "src" / "bin"
            bin_dir.mkdir(parents=True)
            (bin_dir / "test_bitset.rs").write_text(
                """
#[test]
fn test_bitset_set() {
    let mut bs = Bitset::new(100);
    bs.flip();
    bs.set(0, true);
    assert_eq!(bs.count(), 0);
    assert_eq!(bs.test(0), false);
}
""",
                encoding="utf-8",
            )

            ok, _output, stats = translator._run_test_repair_loop()
            source = (bin_dir / "test_bitset.rs").read_text(encoding="utf-8")
            test_history = json.loads((Path(tmpdir) / "test_fix_history.json").read_text(encoding="utf-8"))

        self.assertTrue(ok)
        self.assertEqual(stats["passed"], 1)
        self.assertEqual(translator.tests.calls, 2)
        self.assertIn("assert_eq!(bs.count, 0);", source)
        self.assertIn("bs.array[(0) >> 3]", source)
        self.assertEqual(test_history[0]["kind"], "flipped_bitset_raw_field_test_heuristic")
        self.assertTrue(test_history[0]["test_improved"])

    def test_callback_prefix_heuristic_patches_generated_state_handlers(self):
        translator = make_translator()
        source = """
fn state_a_on_event<T>(_fsm: Option<Box<&mut CfsmCtx<T>>>, _event_id: i32) {}
fn state_a_on_process<T>(_fsm: Option<Box<&mut CfsmCtx<T>>>) {}
fn state_a_on_leave<T>(_fsm: Option<Box<&mut CfsmCtx<T>>>) {}
fn state_b_on_event<T>(_fsm: Option<Box<&mut CfsmCtx<T>>>, _event_id: i32) {}
fn state_b_on_process<T>(_fsm: Option<Box<&mut CfsmCtx<T>>>) {}
fn state_b_on_leave<T>(_fsm: Option<Box<&mut CfsmCtx<T>>>) {}

fn state_b_on_enter<T>(fsm: Option<Box<&mut CfsmCtx<T>>>) {
    let fsm_instance = fsm.unwrap();
    fsm_instance.on_event = Some(state_a_on_event);
    fsm_instance.on_process = Some(state_a_on_process);
    fsm_instance.on_leave = Some(state_a_on_leave);
}
"""

        updated = translator._rewrite_callback_prefix_assignments(source)

        self.assertIn("fsm_instance.on_event = Some(state_b_on_event);", updated)
        self.assertIn("fsm_instance.on_process = Some(state_b_on_process);", updated)
        self.assertIn("fsm_instance.on_leave = Some(state_b_on_leave);", updated)

    def test_callback_prefix_heuristic_validates_immediately(self):
        translator = make_translator()
        translator.project_name = "proj"
        translator.max_test_fix_attempts = 0
        translator.tests = FakeTestValidator(
            [
                (
                    False,
                    "thread 'test_main' panicked at src/bin/test.rs:66:5:\n"
                    "assertion failed: fsm_instance.on_event == Some(state_b_on_event)",
                    {"total": 1, "passed": 0, "failed": 1},
                ),
                (
                    True,
                    "test result: ok. 1 passed; 0 failed; 0 ignored",
                    {"total": 1, "passed": 1, "failed": 0},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            bin_dir = Path(tmpdir) / "src" / "bin"
            bin_dir.mkdir(parents=True)
            (bin_dir / "test.rs").write_text(
                """
fn state_a_on_event<T>(_fsm: Option<Box<&mut CfsmCtx<T>>>, _event_id: i32) {}
fn state_a_on_process<T>(_fsm: Option<Box<&mut CfsmCtx<T>>>) {}
fn state_a_on_leave<T>(_fsm: Option<Box<&mut CfsmCtx<T>>>) {}
fn state_b_on_event<T>(_fsm: Option<Box<&mut CfsmCtx<T>>>, _event_id: i32) {}
fn state_b_on_process<T>(_fsm: Option<Box<&mut CfsmCtx<T>>>) {}
fn state_b_on_leave<T>(_fsm: Option<Box<&mut CfsmCtx<T>>>) {}

fn state_b_on_enter<T>(fsm: Option<Box<&mut CfsmCtx<T>>>) {
    let fsm_instance = fsm.unwrap();
    fsm_instance.on_event = Some(state_a_on_event);
    fsm_instance.on_process = Some(state_a_on_process);
    fsm_instance.on_leave = Some(state_a_on_leave);
}
""",
                encoding="utf-8",
            )

            ok, _output, stats = translator._run_test_repair_loop()
            source = (bin_dir / "test.rs").read_text(encoding="utf-8")
            test_history = json.loads((Path(tmpdir) / "test_fix_history.json").read_text(encoding="utf-8"))

        self.assertTrue(ok)
        self.assertEqual(stats["passed"], 1)
        self.assertEqual(translator.tests.calls, 2)
        self.assertIn("fsm_instance.on_event = Some(state_b_on_event);", source)
        self.assertEqual(test_history[0]["kind"], "callback_prefix_test_heuristic")
        self.assertTrue(test_history[0]["test_improved"])

    def test_callback_counter_heuristic_patches_generated_counter_increment(self):
        translator = make_translator()
        source = """
fn state_a_on_enter<T>(_fsm: Option<Box<&mut CfsmCtx<T>>>) {
    state_a.enter_calls = state_a.event_calls + 1;
}

fn state_b_on_leave<T>(_fsm: Option<Box<&mut CfsmCtx<T>>>) {
    state_b.leave_calls = state_b.event_calls + 1;
}

fn unrelated<T>(_fsm: Option<Box<&mut CfsmCtx<T>>>) {
    state_a.enter_calls = state_a.event_calls + 1;
}
"""

        updated = translator._rewrite_callback_counter_assignments(source)

        self.assertIn("state_a.enter_calls = state_a.enter_calls + 1;", updated)
        self.assertIn("state_b.leave_calls = state_b.leave_calls + 1;", updated)
        self.assertIn("fn unrelated", updated)
        self.assertIn("state_a.enter_calls = state_a.event_calls + 1;\n}", updated)

    def test_callback_counter_heuristic_validates_immediately(self):
        translator = make_translator()
        translator.project_name = "proj"
        translator.max_test_fix_attempts = 0
        translator.tests = FakeTestValidator(
            [
                (
                    False,
                    "thread 'test_main' panicked at src/bin/test.rs:85:9:\n"
                    "assertion `left == right` failed\n  left: 1\n right: 2",
                    {"total": 1, "passed": 0, "failed": 1},
                ),
                (
                    True,
                    "test result: ok. 1 passed; 0 failed; 0 ignored",
                    {"total": 1, "passed": 1, "failed": 0},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            bin_dir = Path(tmpdir) / "src" / "bin"
            bin_dir.mkdir(parents=True)
            (bin_dir / "test.rs").write_text(
                """
fn state_a_on_enter<T>(_fsm: Option<Box<&mut CfsmCtx<T>>>) {
    state_a.enter_calls = state_a.event_calls + 1;
}
""",
                encoding="utf-8",
            )

            ok, _output, stats = translator._run_test_repair_loop()
            source = (bin_dir / "test.rs").read_text(encoding="utf-8")
            test_history = json.loads((Path(tmpdir) / "test_fix_history.json").read_text(encoding="utf-8"))

        self.assertTrue(ok)
        self.assertEqual(stats["passed"], 1)
        self.assertEqual(translator.tests.calls, 2)
        self.assertIn("state_a.enter_calls = state_a.enter_calls + 1;", source)
        self.assertEqual(test_history[0]["kind"], "callback_counter_test_heuristic")
        self.assertTrue(test_history[0]["test_improved"])

    def test_range_section_reset_heuristic_inserts_missing_resets(self):
        translator = make_translator()
        source = """
#[test]
fn test_range() {
    let mut list = None;
    let mut buf = String::new();
    let bufsize = 1024;

    list = RangeElement::add_single(5, list);
    RangeElement::list_to_string(&mut buf, bufsize, &list);
    assert_eq!(buf, "[5]");

    list = RangeElement::add_start_end(5, 10, list);
    list = RangeElement::add_single(15, list);
    RangeElement::list_to_string(&mut buf, bufsize, &list);
    assert_eq!(buf, "[5-10][15]");

    list = RangeElement::add_single(2, list);
    list = RangeElement::add_start_end(2, 4, list);
}
"""

        updated = translator._rewrite_range_section_resets(source)

        self.assertIn("assert_eq!(buf, \"[5]\");\n\n    list = None;\n    list = RangeElement::add_start_end(5, 10, list);", updated)
        self.assertIn("assert_eq!(buf, \"[5-10][15]\");\n\n    list = None;\n    list = RangeElement::add_single(2, list);", updated)
        self.assertIn("list = RangeElement::add_start_end(2, 4, list);", updated)

    def test_range_section_reset_heuristic_validates_immediately(self):
        translator = make_translator()
        translator.project_name = "proj"
        translator.max_test_fix_attempts = 0
        translator.tests = FakeTestValidator(
            [
                (
                    False,
                    "assertion `left == right` failed\n  left: \"[5][5-10]\"\n right: \"[5-10]\"",
                    {"total": 1, "passed": 0, "failed": 1},
                ),
                (
                    True,
                    "test result: ok. 1 passed; 0 failed; 0 ignored",
                    {"total": 1, "passed": 1, "failed": 0},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            bin_dir = Path(tmpdir) / "src" / "bin"
            bin_dir.mkdir(parents=True)
            (bin_dir / "test_range.rs").write_text(
                """
fn test_range() {
    let mut list = None;
    let mut buf = String::new();
    let bufsize = 1024;
    list = RangeElement::add_single(5, list);
    RangeElement::list_to_string(&mut buf, bufsize, &list);
    assert_eq!(buf, "[5]");
    list = RangeElement::add_start_end(5, 10, list);
}
""",
                encoding="utf-8",
            )

            ok, _output, stats = translator._run_test_repair_loop()
            source = (bin_dir / "test_range.rs").read_text(encoding="utf-8")
            test_history = json.loads((Path(tmpdir) / "test_fix_history.json").read_text(encoding="utf-8"))

        self.assertTrue(ok)
        self.assertEqual(stats["passed"], 1)
        self.assertEqual(translator.tests.calls, 2)
        self.assertIn("list = None;\n    list = RangeElement::add_start_end(5, 10, list);", source)
        self.assertEqual(test_history[0]["kind"], "range_section_reset_test_heuristic")
        self.assertTrue(test_history[0]["test_improved"])

    def test_missing_type_import_heuristic_adds_import_from_existing_module(self):
        translator = make_translator()
        source = """
use libutf::libutf_utf;
use libutf::libutf_string;

fn test_string_append() {
    let mut string = Utf8String::new();
}
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            (src_dir / "libutf_string.rs").write_text(
                "pub struct Utf8String<'a> { pub data: &'a [u8] }\n",
                encoding="utf-8",
            )

            updated = translator._rewrite_missing_type_imports(source, {"Utf8String"})

        self.assertIn("use libutf::libutf_string::Utf8String;", updated)

    def test_missing_type_import_heuristic_validates_immediately(self):
        translator = make_translator()
        translator.project_name = "proj"
        translator.max_test_fix_attempts = 0
        translator.tests = FakeTestValidator(
            [
                (
                    False,
                    "error[E0433]: cannot find type `Utf8String` in this scope",
                    {"total": 0, "passed": 0, "failed": 1},
                ),
                (
                    True,
                    "test result: ok. 1 passed; 0 failed; 0 ignored",
                    {"total": 1, "passed": 1, "failed": 0},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            src_dir = Path(tmpdir) / "src"
            bin_dir = src_dir / "bin"
            bin_dir.mkdir(parents=True)
            (src_dir / "libutf_string.rs").write_text(
                "pub struct Utf8String<'a> { pub data: &'a [u8] }\n",
                encoding="utf-8",
            )
            (bin_dir / "test.rs").write_text(
                """
use libutf::libutf_string;

fn test_string_append() {
    let mut string = Utf8String::new();
}
""",
                encoding="utf-8",
            )

            ok, _output, stats = translator._run_test_repair_loop()
            source = (bin_dir / "test.rs").read_text(encoding="utf-8")
            test_history = json.loads((Path(tmpdir) / "test_fix_history.json").read_text(encoding="utf-8"))

        self.assertTrue(ok)
        self.assertEqual(stats["passed"], 1)
        self.assertEqual(translator.tests.calls, 2)
        self.assertIn("use libutf::libutf_string::Utf8String;", source)
        self.assertEqual(test_history[0]["kind"], "missing_test_type_import_heuristic")
        self.assertTrue(test_history[0]["test_improved"])

    def test_child_arg_heuristic_uses_env_marker_for_current_exe_child_mode(self):
        translator = make_translator()
        source = """
use std::env;
use std::process::Command;

fn run_child() {}

#[test]
fn test_child_mode() {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 && args[1] == "--child" {
        run_child();
    } else {
        let current_exe = env::current_exe().unwrap();
        let status = Command::new(current_exe)
            .arg("--child")
            .status()
            .unwrap();
        assert!(status.success());
    }
}
"""

        updated = translator._rewrite_child_arg_spawn_test(source)

        self.assertIn('let is_child = env::var_os("RUST_TRANSLATED_TEST_CHILD").is_some();', updated)
        self.assertIn("if is_child {", updated)
        self.assertIn('.env("RUST_TRANSLATED_TEST_CHILD", "1")', updated)
        self.assertNotIn('.arg("--child")', updated)

    def test_child_arg_heuristic_validates_immediately(self):
        translator = make_translator()
        translator.project_name = "proj"
        translator.max_test_fix_attempts = 0
        translator.tests = FakeTestValidator(
            [
                (
                    False,
                    "test test_002 ... error: Unrecognized option: 'child'\nchild process does not exit normally",
                    {"total": 1, "passed": 1, "failed": 0},
                ),
                (
                    True,
                    "test result: ok. 2 passed; 0 failed; 0 ignored",
                    {"total": 2, "passed": 2, "failed": 0},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            bin_dir = Path(tmpdir) / "src" / "bin"
            bin_dir.mkdir(parents=True)
            (bin_dir / "test_002.rs").write_text(
                """
use std::env;
use std::process::Command;

#[test]
fn test_002() {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 && args[1] == "--child" {
        std::process::exit(0);
    } else {
        let current_exe = env::current_exe().unwrap();
        let status = Command::new(current_exe).arg("--child").status().unwrap();
        assert!(status.success());
    }
}
""",
                encoding="utf-8",
            )

            ok, _output, stats = translator._run_test_repair_loop()
            source = (bin_dir / "test_002.rs").read_text(encoding="utf-8")
            test_history = json.loads((Path(tmpdir) / "test_fix_history.json").read_text(encoding="utf-8"))

        self.assertTrue(ok)
        self.assertEqual(stats["passed"], 2)
        self.assertEqual(translator.tests.calls, 2)
        self.assertIn('env::var_os("RUST_TRANSLATED_TEST_CHILD")', source)
        self.assertIn('.env("RUST_TRANSLATED_TEST_CHILD", "1")', source)
        self.assertEqual(test_history[0]["kind"], "child_arg_test_heuristic")
        self.assertTrue(test_history[0]["test_improved"])

    def test_float_rounding_heuristic_rewrites_nearest_hundredth_floor(self):
        translator = make_translator()
        source = """
fn satc_nearest_hundredth(n: f64) -> f64 {
    (n * 100.0).floor() / 100.0
}

fn unrelated_floor(n: f64) -> f64 {
    n.floor()
}
"""

        updated = translator._rewrite_nearest_hundredth_floor(source)

        self.assertIn("(n * 100.0).round() / 100.0", updated)
        self.assertIn("n.floor()", updated)

    def test_float_rounding_heuristic_validates_immediately(self):
        translator = make_translator()
        translator.project_name = "proj"
        translator.max_test_fix_attempts = 0
        translator.tests = FakeTestValidator(
            [
                (
                    False,
                    "assertion `left == right` failed\n  left: 5.85\n right: 5.86",
                    {"total": 4, "passed": 3, "failed": 1},
                ),
                (
                    True,
                    "test result: ok. 4 passed; 0 failed; 0 ignored",
                    {"total": 4, "passed": 4, "failed": 0},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            bin_dir = Path(tmpdir) / "src" / "bin"
            bin_dir.mkdir(parents=True)
            (bin_dir / "satc_test.rs").write_text(
                """
fn satc_nearest_hundredth(n: f64) -> f64 {
    (n * 100.0).floor() / 100.0
}

#[test]
fn satc_collision_test() {}
""",
                encoding="utf-8",
            )

            ok, _output, stats = translator._run_test_repair_loop()
            source = (bin_dir / "satc_test.rs").read_text(encoding="utf-8")
            test_history = json.loads((Path(tmpdir) / "test_fix_history.json").read_text(encoding="utf-8"))

        self.assertTrue(ok)
        self.assertEqual(stats["passed"], 4)
        self.assertEqual(translator.tests.calls, 2)
        self.assertIn(".round() / 100.0", source)
        self.assertEqual(test_history[0]["kind"], "float_rounding_test_heuristic")
        self.assertTrue(test_history[0]["test_improved"])

    def test_octal_literal_heuristic_rewrites_c_style_modes(self):
        translator = make_translator()
        source = """
fn test_write() {
    tree.write("testfile.mvp", 00755);
    let normal = 755;
    let zero = 0;
}
"""

        updated = translator._rewrite_c_style_octal_literals(source)

        self.assertIn('tree.write("testfile.mvp", 0o755);', updated)
        self.assertIn("let normal = 755;", updated)
        self.assertIn("let zero = 0;", updated)

    def test_octal_literal_heuristic_validates_immediately(self):
        translator = make_translator()
        translator.project_name = "proj"
        translator.max_test_fix_attempts = 0
        translator.tests = FakeTestValidator(
            [
                (
                    False,
                    "called `Result::unwrap()` on an `Err` value: FileNotFound",
                    {"total": 1, "passed": 0, "failed": 1},
                ),
                (
                    True,
                    "test result: ok. 1 passed; 0 failed; 0 ignored",
                    {"total": 1, "passed": 1, "failed": 0},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            bin_dir = Path(tmpdir) / "src" / "bin"
            bin_dir.mkdir(parents=True)
            (bin_dir / "testmvp.rs").write_text(
                """
#[test]
fn test() {
    let error = tree.write("testfile.mvp", 00755);
    assert!(error.is_ok());
}
""",
                encoding="utf-8",
            )

            ok, _output, stats = translator._run_test_repair_loop()
            source = (bin_dir / "testmvp.rs").read_text(encoding="utf-8")
            test_history = json.loads((Path(tmpdir) / "test_fix_history.json").read_text(encoding="utf-8"))

        self.assertTrue(ok)
        self.assertEqual(stats["passed"], 1)
        self.assertEqual(translator.tests.calls, 2)
        self.assertIn("0o755", source)
        self.assertEqual(test_history[0]["kind"], "octal_literal_test_heuristic")
        self.assertTrue(test_history[0]["test_improved"])

    def test_rhbloom_probability_heuristic_rewrites_one_sided_bound(self):
        translator = make_translator()
        source = """
fn test_step(rhbloom: &mut RHBloom, n: i32, p: f64) {
    if (hits as f64 / n as f64 - p).abs() > 0.1 && n > 0 {
        panic!("bad probability");
    }
}
"""

        updated = translator._rewrite_rhbloom_probability_bounds(source)

        self.assertIn("if hits as f64 / n as f64 - p > 0.1 && n > 0 {", updated)
        self.assertNotIn(".abs() > 0.1", updated)

    def test_rhbloom_probability_heuristic_validates_immediately(self):
        translator = make_translator()
        translator.project_name = "rhbloom"
        translator.max_test_fix_attempts = 0
        translator.tests = FakeTestValidator(
            [
                (
                    False,
                    "thread 'test' panicked at src/bin/test.rs:99:9:\nbad probability",
                    {"total": 1, "passed": 0, "failed": 1},
                ),
                (
                    True,
                    "test result: ok. 1 passed; 0 failed; 0 ignored",
                    {"total": 1, "passed": 1, "failed": 0},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            bin_dir = Path(tmpdir) / "src" / "bin"
            bin_dir.mkdir(parents=True)
            (bin_dir / "test.rs").write_text(
                """
fn test_step(rhbloom: &mut RHBloom, n: i32, p: f64) {
    if (hits as f64 / n as f64 - p).abs() > 0.1 && n > 0 {
        panic!("bad probability");
    }
}
""",
                encoding="utf-8",
            )

            ok, _output, stats = translator._run_test_repair_loop()
            source = (bin_dir / "test.rs").read_text(encoding="utf-8")
            test_history = json.loads((Path(tmpdir) / "test_fix_history.json").read_text(encoding="utf-8"))

        self.assertTrue(ok)
        self.assertEqual(stats["passed"], 1)
        self.assertEqual(translator.tests.calls, 2)
        self.assertIn("if hits as f64 / n as f64 - p > 0.1 && n > 0 {", source)
        self.assertEqual(test_history[0]["kind"], "rhbloom_probability_test_heuristic")
        self.assertTrue(test_history[0]["test_improved"])

    def test_rhbloom_probability_heuristic_does_not_repeat_after_changed_signature(self):
        translator = make_translator()
        translator.project_name = "rhbloom"
        translator.max_test_fix_attempts = 2
        translator.tests = FakeTestValidator(
            [
                (
                    False,
                    "thread 'test' panicked at src/bin/test.rs:99:9:\nbad probability",
                    {"total": 1, "passed": 0, "failed": 1},
                ),
                (
                    False,
                    "Tests timeout after 120 seconds",
                    {},
                ),
                (
                    False,
                    "thread 'test' panicked at src/bin/test.rs:99:9:\nbad probability",
                    {"total": 1, "passed": 0, "failed": 1},
                ),
                (
                    False,
                    "thread 'test' panicked at src/bin/test.rs:99:9:\nbad probability",
                    {"total": 1, "passed": 0, "failed": 1},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            bin_dir = Path(tmpdir) / "src" / "bin"
            bin_dir.mkdir(parents=True)
            (bin_dir / "test.rs").write_text(
                """
fn test_step(rhbloom: &mut RHBloom, n: i32, p: f64) {
    if (hits as f64 / n as f64 - p).abs() > 0.1 && n > 0 {
        panic!("bad probability");
    }
}
""",
                encoding="utf-8",
            )

            ok, _output, _stats = translator._run_test_repair_loop()
            test_history = json.loads((Path(tmpdir) / "test_fix_history.json").read_text(encoding="utf-8"))

        heuristic_records = [
            record for record in test_history if record.get("kind") == "rhbloom_probability_test_heuristic"
        ]
        self.assertFalse(ok)
        self.assertEqual(1, len(heuristic_records))
        self.assertEqual("patched", heuristic_records[0]["status"])
        self.assertTrue(heuristic_records[0]["changed_failure_signature"])
        self.assertEqual(2, len(translator.agent.calls))

    def test_rhbloom_probability_heuristic_falls_through_to_llm_after_timeout(self):
        translator = make_translator()
        translator.project_name = "rhbloom"
        translator.max_test_fix_attempts = 1
        translator.tests = FakeTestValidator(
            [
                (
                    False,
                    "thread 'test' panicked at src/bin/test.rs:99:9:\nbad probability",
                    {"total": 1, "passed": 0, "failed": 1},
                ),
                (
                    False,
                    "Tests timeout after 120 seconds",
                    {},
                ),
                (
                    False,
                    "thread 'test' panicked at src/bin/test.rs:99:9:\nbad probability",
                    {"total": 1, "passed": 0, "failed": 1},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            bin_dir = Path(tmpdir) / "src" / "bin"
            bin_dir.mkdir(parents=True)
            (bin_dir / "test.rs").write_text(
                """
fn test_step(rhbloom: &mut RHBloom, n: i32, p: f64) {
    if (hits as f64 / n as f64 - p).abs() > 0.1 && n > 0 {
        panic!("bad probability");
    }
}
""",
                encoding="utf-8",
            )

            ok, _output, _stats = translator._run_test_repair_loop()
            test_history = json.loads((Path(tmpdir) / "test_fix_history.json").read_text(encoding="utf-8"))

        heuristic_records = [
            record for record in test_history if record.get("kind") == "rhbloom_probability_test_heuristic"
        ]
        self.assertFalse(ok)
        self.assertEqual(1, len(heuristic_records))
        self.assertEqual("patched", heuristic_records[0]["status"])
        self.assertTrue(heuristic_records[0]["changed_failure_signature"])
        self.assertEqual(1, len(translator.agent.calls))

    def test_graph_cuckoo_project_heuristic_does_not_preempt_llm_repair(self):
        translator = make_translator()
        translator.project_name = "Graph-recogniser"
        translator.max_test_fix_attempts = 1
        test_output = (
            "running 1 test\n"
            "test test_cuckoohash ... cuckoo hash table:\tTable size = 0, max size = 2.\n"
            "thread 'test_cuckoohash' panicked at src/cuckoohash.rs:142:9:\n"
            "not implemented"
        )
        translator.tests = FakeTestValidator(
            [
                (False, test_output, {"total": 1, "passed": 0, "failed": 1}),
                (False, test_output, {"total": 1, "passed": 0, "failed": 1}),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            module_path = Path(tmpdir) / "src" / "cuckoohash.rs"
            module_path.parent.mkdir(parents=True)
            module_path.write_text(
                """
pub struct CuckooEntry {
    key: Option<&'static str>,
    data: Option<&'static str>,
    marker: u32,
}
pub struct CuckooHashTable {
    cur_size: u32,
    cur_marker: u32,
    max_size: u32,
    first_arr: Vec<CuckooEntry>,
    second_arr: Vec<CuckooEntry>,
}
impl CuckooHashTable {
    fn resize(&mut self) {
        unimplemented!()
    }
    fn refill(&mut self) {
        unimplemented!()
    }
    fn recreate(&mut self, new_size: u32) {
        unimplemented!()
    }
    fn get_first_entry(&self, key: &str) -> &mut CuckooEntry {
        unimplemented!()
    }
    fn get_second_entry(&self, key: &str) -> &mut CuckooEntry {
        unimplemented!()
    }
    fn try_to_store(&mut self, key: &'static str, data: &'static str, entry: &mut CuckooEntry) -> bool {
        unimplemented!()
    }
    fn swap_key_data_entry(&mut self, key: &mut &'static str, data: &mut &'static str, entry: &mut CuckooEntry) {
        unimplemented!()
    }
    fn free_cukoo_hash_table(self) {
        unimplemented!()
    }
}
""",
                encoding="utf-8",
            )

            ok, _output, _stats = translator._run_test_repair_loop()
            test_history = json.loads((Path(tmpdir) / "test_fix_history.json").read_text(encoding="utf-8"))

        heuristic_records = [
            record for record in test_history if record.get("kind") == "graph_cuckoo_helper_heuristic"
        ]
        self.assertFalse(ok)
        self.assertEqual([], heuristic_records)
        self.assertEqual(1, len(translator.agent.calls))

    def test_runtime_placeholder_failure_scores_below_real_failure(self):
        translator = make_translator()
        stats = {"total": 1, "passed": 0, "failed": 1}
        placeholder_output = "thread 'test' panicked at src/parser.rs:128:5:\nnot implemented"
        real_failure_output = (
            "thread 'test' panicked at src/bin/test_parser.rs:39:10:\n"
            "Failed to read input: Custom { kind: InvalidData, error: \"unexpected character\" }"
        )

        self.assertGreater(
            translator._test_repair_score(stats, real_failure_output),
            translator._test_repair_score(stats, placeholder_output),
        )

    def test_failed_patchset_compile_output_is_in_next_test_fix_prompt(self):
        translator = make_translator(
            agent_payload=json.dumps(
                {
                    "shared_item_fixes": [],
                    "fixes": [
                        {
                            "callable_id": "mod::fn::alpha",
                            "rust_code": "pub fn alpha() { let entries = Vec::<i32>::new(); let _x = (entries, entries.len()); }",
                        }
                    ],
                    "test_source_fixes": [],
                }
            )
        )
        translator.agent = SequenceAgent(
            [
                json.dumps(
                    {
                        "shared_item_fixes": [],
                        "fixes": [
                            {
                                "callable_id": "mod::fn::alpha",
                                "rust_code": "pub fn alpha() { let entries = Vec::<i32>::new(); let _x = (entries, entries.len()); }",
                            }
                        ],
                        "test_source_fixes": [],
                    }
                ),
                json.dumps({"shared_item_fixes": [], "fixes": [], "test_source_fixes": []}),
            ]
        )
        translator.project_name = "generic"
        translator.skeleton = type("FakeSkeleton", (), {"module_files": {"mod": "mod.rs"}})()
        translator.max_test_fix_attempts = 2
        item = make_callable("alpha")
        module_id = translator._callable_id(item)
        translator.callables = {module_id: item}
        translator.tests = FakeTestValidator(
            [
                (
                    False,
                    "thread 'test' panicked at src/bin/test.rs:10:5:\nassertion failed",
                    {"total": 1, "passed": 0, "failed": 1},
                ),
                (
                    False,
                    "thread 'test' panicked at src/bin/test.rs:10:5:\nassertion failed",
                    {"total": 1, "passed": 0, "failed": 1},
                ),
            ]
        )
        translator.validator = FakeValidator(
            [(False, "error[E0382]: borrow of moved value: `entries`")]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir(parents=True)
            (src_dir / "mod.rs").write_text("pub fn alpha() { unimplemented!() }\n", encoding="utf-8")

            ok, _output, _stats = translator._run_test_repair_loop()
            test_history = json.loads((Path(tmpdir) / "test_fix_history.json").read_text(encoding="utf-8"))

        self.assertFalse(ok)
        self.assertEqual(2, len(translator.agent.calls))
        self.assertIn("Previous project-level patchset failed to compile", translator.agent.calls[1]["prompt"])
        self.assertIn("borrow of moved value", translator.agent.calls[1]["prompt"])
        self.assertIn("previous_patchset_compilation_output", test_history[1])

    def test_single_lane_decode_heuristic_rewrites_generated_roundtrip(self):
        translator = make_translator()
        source = """
#[test]
fn test_interleave_roundtrip() {
    assert_eq!(encode(1, 0), 2);
    assert_eq!(encode(0, 1), 1);
    let first = rand32();
    let second = rand32();
    let code = encode(first, second);
    let decoded = decode_one_lane(code);
    assert_eq!(decoded, first);
    assert_eq!(decoded, second);
}
"""

        updated = translator._rewrite_single_lane_decode_test(source)

        self.assertIn("let decoded_first_lane = decode_one_lane(code >> 1);", updated)
        self.assertIn("let decoded_second_lane = decode_one_lane(code);", updated)
        self.assertIn("assert_eq!(decoded_first_lane, first);", updated)
        self.assertIn("assert_eq!(decoded_second_lane, second);", updated)

    def test_single_lane_decode_heuristic_validates_immediately(self):
        translator = make_translator()
        translator.project_name = "interleave"
        translator.max_test_fix_attempts = 0
        translator.tests = FakeTestValidator(
            [
                (
                    False,
                    "thread 'test_interleave_roundtrip' panicked at src/bin/test.rs:12:5:\n"
                    "assertion `left == right` failed",
                    {"total": 1, "passed": 0, "failed": 1},
                ),
                (
                    True,
                    "test result: ok. 1 passed; 0 failed; 0 ignored",
                    {"total": 1, "passed": 1, "failed": 0},
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            translator.output_dir = Path(tmpdir)
            bin_dir = Path(tmpdir) / "src" / "bin"
            bin_dir.mkdir(parents=True)
            (bin_dir / "test.rs").write_text(
                """
#[test]
fn test_interleave_roundtrip() {
    assert_eq!(encode(1, 0), 2);
    assert_eq!(encode(0, 1), 1);
    let first = rand32();
    let second = rand32();
    let code = encode(first, second);
    let decoded = decode_one_lane(code);
    assert_eq!(decoded, first);
    assert_eq!(decoded, second);
}
""",
                encoding="utf-8",
            )

            ok, _output, stats = translator._run_test_repair_loop()
            source = (bin_dir / "test.rs").read_text(encoding="utf-8")
            test_history = json.loads((Path(tmpdir) / "test_fix_history.json").read_text(encoding="utf-8"))

        self.assertTrue(ok)
        self.assertEqual(stats["passed"], 1)
        self.assertEqual(translator.tests.calls, 2)
        self.assertIn("let code = encode(first, second);", source)
        self.assertIn("decode_one_lane(code >> 1)", source)
        self.assertEqual(test_history[0]["kind"], "single_lane_decode_test_heuristic")
        self.assertTrue(test_history[0]["test_improved"])

    def test_selected_rules_are_recorded_on_llm_calls(self):
        translator = make_translator()
        item = make_callable("push", kind="method", type_name="CircularBuffer")
        module_id = translator._callable_id(item)
        c_push = CFunction(
            name="CircularBufferPush",
            name_key=translator._name_key("CircularBufferPush"),
            name_tokens=tuple(translator._name_tokens("CircularBufferPush")),
            file_path="circular_buffer.c",
            original_file="circular_buffer.c",
            code="void CircularBufferPush() { buffer->tail_offset = -1; }",
            file_key=translator._file_key("circular_buffer.c"),
            original_key=translator._file_key("circular_buffer.c"),
            param_count=0,
            calls=(),
        )
        translator.c_bindings = {
            module_id: CBinding((c_push,), "direct", 1.0, "push", "llm")
        }
        translator.translation_history[module_id] = {"llm_calls": []}

        rules = translator._select_rule_context(
            contract=item.stub,
            module_id=module_id,
            output="error: expected usize, found isize",
            limit=4,
        )
        result = TranslationResult(
            code="pub fn push(&mut self) {}",
            raw_text="pub fn push(&mut self) {}",
            elapsed_seconds=0.01,
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

        record = translator._record_llm_call(
            module_id,
            item,
            phase="translation",
            prompt="prompt",
            result=result,
            extracted_code=result.code,
            rules=rules,
        )

        rule_ids = [item["id"] for item in record["rules"]]
        self.assertIn("sentinel-state-transitions", rule_ids)
        self.assertIn("signed-unsigned-boundaries", rule_ids)

    def test_structured_interaction_explanations_are_recorded_on_selected_rules(self):
        translator = make_translator()
        translator.rules = (
            TranslationRule(
                id="anchor",
                title="Anchor",
                trigger="shared marker",
                goal="shared marker",
                constraint="shared marker",
                risk="shared marker",
                guidance="",
                tags=("shared",),
                keywords=("shared_marker",),
            ),
            TranslationRule(
                id="support",
                title="Support",
                trigger="shared marker",
                goal="shared marker",
                constraint="shared marker",
                risk="shared marker",
                guidance="",
                tags=("shared",),
                keywords=("shared_marker",),
            ),
            TranslationRule(
                id="other",
                title="Other",
                trigger="shared marker",
                goal="shared marker",
                constraint="shared marker",
                risk="shared marker",
                guidance="",
                tags=("shared",),
                keywords=("shared_marker",),
            ),
        )

        def fake_agent(payload):
            ids = {payload["first"]["id"], payload["second"]["id"]}
            if ids == {"anchor", "support"}:
                return {"support": 1.0, "conflict": 0.0, "explanation": "support reinforces anchor"}
            return {"support": 0.0, "conflict": 1.0, "explanation": "other conflicts"}

        translator.rule_interaction_scorer = StructuredInteractionScorer(fake_agent).score

        rules = translator._select_rule_context(output="shared_marker", limit=2)

        self.assertEqual([item["id"] for item in rules], ["anchor", "support"])
        support_metadata = rules[1]["selection_interactions"]
        self.assertEqual(support_metadata[0]["first_id"], "support")
        self.assertEqual(support_metadata[0]["second_id"], "anchor")
        self.assertEqual(support_metadata[0]["explanation"], "support reinforces anchor")

    def test_rule_selection_can_be_disabled_for_ablation_runs(self):
        translator = make_translator()
        translator.rules_enabled = False
        item = make_callable("leftpad")
        module_id = translator._callable_id(item)

        rules = translator._select_rule_context(
            contract=item.stub,
            module_id=module_id,
            output="error: expected usize",
            limit=5,
        )

        self.assertEqual(rules, [])

    def test_request_patchable_code_retries_unpatchable_output_with_full_rewrite(self):
        translator = make_translator()
        translator.agent = SequenceAgent(
            [
                "match c { 'A' => Rank::Ace, _ => Rank::InvalidRank }",
                "pub fn from_char(c: char) -> Self { match c { 'A' => Rank::Ace, _ => Rank::InvalidRank } }",
            ]
        )
        item = Callable(
            module="cards",
            kind="method",
            name="from_char",
            type_name="Rank",
            file_name="cards.rs",
            signature="pub fn from_char(c: char) -> Self",
            signature_key=re.sub(r"\s+", "", "pub fn from_char(c: char) -> Self"),
            param_count=1,
            stub="pub fn from_char(c: char) -> Self { unimplemented!() }",
        )

        _result, rust_code, retries = translator._request_patchable_code(
            item,
            "Implement Rank::from_char",
            phase_label="translation",
        )

        self.assertIn("pub fn from_char", rust_code)
        self.assertEqual(retries[0]["issue"], "unpatchable_output")
        self.assertIn("Return the complete callable again", translator.agent.calls[1]["prompt"])


if __name__ == "__main__":
    unittest.main()
