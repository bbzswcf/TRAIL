import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from unittest import mock
from types import SimpleNamespace
from pathlib import Path

from c2rust_translator.rules import DEFAULT_RULES
from c2rust_translator import adversarial_rules as adversarial_core
from c2rust_translator.adversarial_rules import (
    ChallengeRecord,
    ChallengeSanityChecker,
    ChallengeToolChecker,
    CompositionalChallengeGenerator,
    EvidenceGate,
    ExecutionEvidence,
    FailureAttributor,
    RuleBank,
    RuleConstrainedReplayRunner,
    RuleRecord,
    RuleRelationRecord,
    RuleUpdateProposal,
    LLMActivationValidator,
    LLMCompositionRelationReviewer,
    LLMSemanticOracleValidator,
    LLMStructuredChallengerAgent,
    LLMStructuredRefinerAgent,
    LLMTraceContradictionReviewer,
    LLMTraceRuleSynthesizer,
    RetrievalEngine,
    SemanticActivationValidationResult,
    SemanticOracleValidationResult,
    StructuredActivationValidator,
    StructuredChallengerAgent,
    StructuredCompositionRelationReviewer,
    StructuredOracleValidator,
    StructuredRefinerAgent,
    StructuredTraceContradictionReviewer,
    StructuredTraceSynthesizer,
    SyntheticChallengeGenerator,
    TraceReflector,
    TraceRuleMiner,
    TranslatorChallengeReplayAdapter,
    run_offline_adversarial_learning,
)


class AdversarialRuleBankTest(unittest.TestCase):
    def test_json_bank_round_trips_records_and_versions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bank.json"
            bank = RuleBank([
                RuleRecord(
                    id="pointer-buffer",
                    title="Pointer Buffer",
                    trigger="C pointer and length",
                    goal="Preserve slice length",
                    constraint="Keep terminator semantics",
                    risk="NUL padding can leak",
                    guidance="Trim public views",
                    tags=("pointers", "strings"),
                    keywords=("pointer", "length"),
                    version=2,
                    source="test",
                )
            ])

            bank.save(path)
            loaded = RuleBank.load(path)

            self.assertEqual(len(loaded.records), 1)
            self.assertEqual(loaded.records[0].id, "pointer-buffer")
            self.assertEqual(loaded.records[0].version, 2)
            self.assertEqual(loaded.records[0].tags, ("pointers", "strings"))
            self.assertEqual(loaded.records[0].to_prompt_dict()["trigger"], "C pointer and length")

    def test_json_bank_round_trips_relation_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bank.json"
            bank = RuleBank(
                [
                    RuleRecord(
                        id="fixed-c-string",
                        title="Fixed C String",
                        trigger="fixed char array",
                        goal="Expose visible length",
                        constraint="Trim at NUL",
                        risk="padding may leak",
                    ),
                    RuleRecord(
                        id="capacity-contract",
                        title="Capacity Contract",
                        trigger="count and capacity",
                        goal="Preserve capacity",
                        constraint="Keep capacity field",
                        risk="capacity drift",
                    ),
                ],
                relations=[
                    RuleRelationRecord(
                        first_id="fixed-c-string",
                        second_id="capacity-contract",
                        relation="supports",
                        rationale="composition challenge passed",
                        evidence="fingerprint:abc",
                    )
                ],
            )

            bank.save(path)
            loaded = RuleBank.load(path)

            self.assertEqual(len(loaded.relations), 1)
            self.assertEqual(loaded.relations[0].first_id, "fixed-c-string")
            self.assertEqual(loaded.relations[0].second_id, "capacity-contract")
            self.assertEqual(loaded.relations[0].relation, "supports")

    def test_same_path_offline_update_uses_exclusive_bank_lock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_root = root / "run"
            run_root.mkdir()
            bank_path = root / "bank.json"
            report_path = root / "report.json"
            RuleBank([]).save(bank_path)

            with mock.patch("c2rust_translator.adversarial_rules.fcntl.flock") as flock:
                run_offline_adversarial_learning(run_root, bank_path, bank_path, report_path)

            lock_ops = [call.args[1] for call in flock.call_args_list]
            self.assertEqual(
                lock_ops,
                [
                    adversarial_core.fcntl.LOCK_EX,
                    adversarial_core.fcntl.LOCK_UN,
                ],
            )


class AdversarialRuleRetrievalTest(unittest.TestCase):
    def test_imports_legacy_rules_into_json_bank(self):
        bank = RuleBank.from_legacy(DEFAULT_RULES[:2])

        self.assertEqual(len(bank.records), 2)
        self.assertEqual(bank.records[0].source, "legacy")
        self.assertTrue(bank.records[0].keywords)

    def test_retrieval_reports_keyword_fallback_when_embedding_deps_missing(self):
        bank = RuleBank([
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
        ])

        engine = RetrievalEngine(bank, prefer_embeddings=True)
        selected = engine.select("String::from_utf8_lossy shows NUL padding", limit=1)

        self.assertEqual(selected[0].id, "fixed-c-string")
        self.assertIn(engine.status.backend, {"keyword", "faiss-bge"})
        if engine.status.backend == "keyword":
            self.assertIn("unavailable", engine.status.reason)

    def test_retrieval_uses_injected_embedding_index_when_available(self):
        class FakeEmbeddingModel:
            def encode(self, texts, normalize_embeddings=True):
                vectors = []
                for text in texts:
                    lowered = text.lower()
                    vectors.append([
                        1.0 if "pointer" in lowered else 0.0,
                        1.0 if "capacity" in lowered else 0.0,
                    ])
                return vectors

        class FakeIndex:
            def __init__(self, _dimension):
                self.vectors = []

            def add(self, vectors):
                self.vectors = list(vectors)

            def search(self, queries, limit):
                scores = []
                indexes = []
                for query in queries:
                    ranked = []
                    for index, vector in enumerate(self.vectors):
                        score = sum(left * right for left, right in zip(query, vector))
                        ranked.append((score, index))
                    ranked.sort(reverse=True)
                    scores.append([score for score, _index in ranked[:limit]])
                    indexes.append([index for _score, index in ranked[:limit]])
                return scores, indexes

        class FakeFaiss:
            @staticmethod
            def IndexFlatIP(dimension):
                return FakeIndex(dimension)

        bank = RuleBank([
            RuleRecord(
                id="pointer-buffer",
                title="Pointer Buffer",
                trigger="pointer and length",
                goal="Preserve pointed buffer",
                constraint="Keep slice length",
                risk="pointer aliasing",
            ),
            RuleRecord(
                id="capacity-contract",
                title="Capacity Contract",
                trigger="count and capacity",
                goal="Preserve capacity",
                constraint="Do not recompute capacity",
                risk="capacity drift",
            ),
        ])

        engine = RetrievalEngine(
            bank,
            prefer_embeddings=True,
            embedding_model=FakeEmbeddingModel(),
            faiss_module=FakeFaiss(),
        )
        selected = engine.select("caller-visible capacity must be preserved", limit=1)

        self.assertEqual(engine.status.backend, "faiss-bge")
        self.assertEqual([record.id for record in selected], ["capacity-contract"])

    def test_embedding_retrieval_reuses_cached_faiss_index(self):
        class RecordingEmbeddingModel:
            def __init__(self):
                self.encoded_batches = []

            def encode(self, texts, normalize_embeddings=True):
                self.encoded_batches.append(list(texts))
                vectors = []
                for text in texts:
                    lowered = text.lower()
                    vectors.append([
                        1.0 if "pointer" in lowered else 0.0,
                        1.0 if "capacity" in lowered else 0.0,
                    ])
                return vectors

        class FakeIndex:
            def __init__(self, _dimension):
                self.vectors = []

            def add(self, vectors):
                self.vectors = [
                    list(vector.tolist() if hasattr(vector, "tolist") else vector)
                    for vector in vectors
                ]

            def search(self, queries, limit):
                scores = []
                indexes = []
                for query in queries:
                    ranked = []
                    for index, vector in enumerate(self.vectors):
                        score = sum(left * right for left, right in zip(query, vector))
                        ranked.append((score, index))
                    ranked.sort(reverse=True)
                    scores.append([score for score, _index in ranked[:limit]])
                    indexes.append([index for _score, index in ranked[:limit]])
                return scores, indexes

        class FakeFaiss:
            @staticmethod
            def IndexFlatIP(dimension):
                return FakeIndex(dimension)

            @staticmethod
            def write_index(index, path):
                Path(path).write_text(json.dumps(index.vectors), encoding="utf-8")

            @staticmethod
            def read_index(path):
                vectors = json.loads(Path(path).read_text(encoding="utf-8"))
                index = FakeIndex(len(vectors[0]))
                index.vectors = vectors
                return index

        bank = RuleBank([
            RuleRecord(
                id="pointer-buffer",
                title="Pointer Buffer",
                trigger="pointer and length",
                goal="Preserve pointed buffer",
                constraint="Keep slice length",
                risk="pointer aliasing",
            ),
            RuleRecord(
                id="capacity-contract",
                title="Capacity Contract",
                trigger="count and capacity",
                goal="Preserve capacity",
                constraint="Do not recompute capacity",
                risk="capacity drift",
            ),
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "retrieval.faiss"
            first_model = RecordingEmbeddingModel()
            first = RetrievalEngine(
                bank,
                prefer_embeddings=True,
                embedding_model=first_model,
                faiss_module=FakeFaiss(),
                index_cache_path=cache_path,
            )

            self.assertEqual(first.cache_status, "written")
            self.assertTrue(cache_path.exists())
            self.assertTrue(cache_path.with_suffix(".faiss.meta.json").exists())
            self.assertEqual(len(first_model.encoded_batches[0]), 2)

            second_model = RecordingEmbeddingModel()
            second = RetrievalEngine(
                bank,
                prefer_embeddings=True,
                embedding_model=second_model,
                faiss_module=FakeFaiss(),
                index_cache_path=cache_path,
            )
            selected = second.select("capacity must remain visible", limit=1)

            self.assertEqual(second.cache_status, "loaded")
            self.assertEqual([record.id for record in selected], ["capacity-contract"])
            self.assertEqual(second_model.encoded_batches, [["capacity must remain visible"]])

    def test_build_rule_retrieval_index_writes_manifest(self):
        from scripts.build_rule_retrieval_index import build_index_manifest

        class RecordingEmbeddingModel:
            def encode(self, texts, normalize_embeddings=True):
                return [
                    [1.0 if "pointer" in text.lower() else 0.0, 0.0]
                    for text in texts
                ]

        class FakeIndex:
            def __init__(self, _dimension):
                self.vectors = []

            def add(self, vectors):
                self.vectors = [
                    list(vector.tolist() if hasattr(vector, "tolist") else vector)
                    for vector in vectors
                ]

        class FakeFaiss:
            @staticmethod
            def IndexFlatIP(dimension):
                return FakeIndex(dimension)

            @staticmethod
            def write_index(index, path):
                Path(path).write_text(json.dumps(index.vectors), encoding="utf-8")

        bank = RuleBank([
            RuleRecord(
                id="pointer-buffer",
                title="Pointer Buffer",
                trigger="pointer and length",
                goal="Preserve pointed buffer",
                constraint="Keep slice length",
                risk="pointer aliasing",
            )
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "rules.faiss"
            manifest_path = Path(tmpdir) / "manifest.json"

            manifest = build_index_manifest(
                bank,
                index_path,
                manifest_path=manifest_path,
                embedding_model=RecordingEmbeddingModel(),
                faiss_module=FakeFaiss(),
            )

            self.assertEqual(manifest["backend"], "faiss-bge")
            self.assertEqual(manifest["cache_status"], "written")
            self.assertIn("embedding_dependencies", manifest)
            self.assertEqual(manifest["record_ids"], ["pointer-buffer"])
            self.assertTrue(index_path.exists())
            self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8")), manifest)

    def test_build_rule_retrieval_index_manifest_reports_missing_dependencies(self):
        from scripts.build_rule_retrieval_index import build_index_manifest

        bank = RuleBank([
            RuleRecord(
                id="pointer-buffer",
                title="Pointer Buffer",
                trigger="pointer and length",
                goal="Preserve pointed buffer",
                constraint="Keep slice length",
                risk="pointer aliasing",
            )
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "rules.faiss"
            manifest = build_index_manifest(
                bank,
                index_path,
                find_spec=lambda _name: None,
            )

            dependencies = manifest["embedding_dependencies"]

            self.assertFalse(dependencies["available"])
            self.assertEqual(dependencies["modules"]["faiss"], "missing")
            self.assertEqual(dependencies["modules"]["sentence_transformers"], "missing")
            self.assertIn("uv", dependencies["install_hint"])

    def test_build_rule_retrieval_index_loads_python_rules_path(self):
        from scripts.build_rule_retrieval_index import load_bank

        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = Path(tmpdir) / "snapshot.py"
            snapshot_path.write_text(
                "\n".join(
                    [
                        "from c2rust_translator.rules import TranslationRule",
                        "DEFAULT_RULES = (",
                        "    TranslationRule(",
                        "        id='snapshot-rule',",
                        "        title='Snapshot Rule',",
                        "        trigger='python snapshot trigger',",
                        "        goal='load the Python snapshot',",
                        "        constraint='preserve snapshot fields',",
                        "        risk='using the wrong retrieval bank',",
                        "        guidance='Index this exact profile snapshot.',",
                        "        tags=('snapshot',),",
                        "        keywords=('profile',),",
                        "    ),",
                        ")",
                    ]
                ),
                encoding="utf-8",
            )

            bank = load_bank(None, rules_path=snapshot_path)

        self.assertEqual([record.id for record in bank.records], ["snapshot-rule"])
        self.assertEqual(bank.records[0].trigger, "python snapshot trigger")


class AdversarialChallengeTest(unittest.TestCase):
    def test_compositional_generator_builds_joint_challenge_for_rule_set(self):
        string_rule = RuleRecord(
            id="fixed-c-string",
            title="Fixed C String",
            trigger="fixed char array",
            goal="Expose visible string length",
            constraint="Stop at first NUL",
            risk="zero padding leaks",
            tags=("strings",),
            keywords=("NUL",),
        )
        capacity_rule = RuleRecord(
            id="capacity-contract",
            title="Capacity Contract",
            trigger="output struct has count and capacity",
            goal="Preserve caller-visible capacity",
            constraint="Do not recompute capacity from entries length",
            risk="capacity drift",
            tags=("capacity",),
            keywords=("capacity",),
        )

        challenge = CompositionalChallengeGenerator().generate([string_rule, capacity_rule])

        self.assertEqual(challenge.rule_id, "composition:fixed-c-string+capacity-contract")
        self.assertIn("fixed-c-string", challenge.c_program)
        self.assertIn("capacity-contract", challenge.c_program)
        self.assertIn("challenge_composition", challenge.rust_interface)
        self.assertIn("assert_eq!", challenge.harness)
        self.assertIn("jointly satisfy", challenge.oracle)
        self.assertFalse(challenge.evidence.tool_checked)

    def test_compositional_generator_includes_counterfactual_dependency_probes(self):
        string_rule = RuleRecord(
            id="fixed-c-string",
            title="Fixed C String",
            trigger="fixed char array",
            goal="Expose visible string length",
            constraint="Stop at first NUL",
            risk="zero padding leaks",
            tags=("strings",),
            keywords=("NUL",),
        )
        capacity_rule = RuleRecord(
            id="capacity-contract",
            title="Capacity Contract",
            trigger="output struct has count and capacity",
            goal="Preserve caller-visible capacity",
            constraint="Do not recompute capacity from entries length",
            risk="capacity drift",
            tags=("capacity",),
            keywords=("capacity",),
        )

        challenge = CompositionalChallengeGenerator().generate([string_rule, capacity_rule])
        probes = challenge.trace_bundle["composition_dependency_probes"]

        self.assertIn("reference_source", probes)
        self.assertIn("challenge_composition", probes["reference_source"])
        self.assertEqual(
            [mutant["rule_id"] for mutant in probes["mutants"]],
            ["fixed-c-string", "capacity-contract"],
        )
        self.assertTrue(all("composition_mutant" in mutant["rust_source"] for mutant in probes["mutants"]))

    def test_composition_candidate_sets_use_formulaic_greedy_selection_with_bounded_size(self):
        records = [
            RuleRecord(
                id="string-boundary",
                title="String Boundary",
                trigger="fixed C buffer exposed through Rust string APIs",
                goal="Expose visible string length",
                constraint="Stop at the first NUL",
                risk="zero padding leaks",
                tags=("string",),
                keywords=("buffer", "visible"),
            ),
            RuleRecord(
                id="capacity-contract",
                title="Capacity Contract",
                trigger="C struct carries count and capacity",
                goal="Preserve capacity",
                constraint="Keep capacity independent from count",
                risk="capacity drift",
                tags=("capacity",),
                keywords=("buffer", "capacity"),
            ),
            RuleRecord(
                id="ownership-transfer",
                title="Ownership Transfer",
                trigger="C caller transfers allocation ownership",
                goal="Represent ownership transfer",
                constraint="Do not borrow after free",
                risk="aliasing after ownership transfer",
                tags=("ownership",),
                keywords=("buffer", "ownership"),
            ),
            RuleRecord(
                id="late-extra-contract",
                title="Late Extra Contract",
                trigger="C code exposes a late extra observable field",
                goal="Preserve late extra metadata",
                constraint="Keep the extra field stable",
                risk="extra metadata drift",
                tags=("extra",),
                keywords=("buffer", "extra"),
            ),
        ]

        selected = adversarial_core._composition_candidate_sets(records, limit=1)

        self.assertEqual(
            [[rule.id for rule in rule_set] for rule_set in selected],
            [["string-boundary", "capacity-contract", "late-extra-contract"]],
        )

    def test_composition_candidate_sets_skip_unresolved_conflicts_above_threshold(self):
        string_rule = RuleRecord(
            id="string-boundary",
            title="String Boundary",
            trigger="fixed C buffer exposed through Rust string APIs",
            goal="Expose visible string length",
            constraint="Stop at the first NUL",
            risk="zero padding leaks",
            tags=("string",),
            keywords=("buffer",),
        )
        capacity_rule = RuleRecord(
            id="capacity-contract",
            title="Capacity Contract",
            trigger="C struct carries count and capacity",
            goal="Preserve capacity",
            constraint="Keep capacity independent from count",
            risk="capacity drift",
            tags=("capacity",),
            keywords=("buffer",),
        )
        conflicting_rule = RuleRecord(
            id="ownership-transfer",
            title="Ownership Transfer",
            trigger="C caller transfers allocation ownership",
            goal="Represent ownership transfer",
            constraint="Take ownership of the buffer",
            risk="conflicts with borrowed string views",
            tags=("ownership",),
            keywords=("buffer",),
        )

        selected = adversarial_core._composition_candidate_sets(
            [string_rule, capacity_rule, conflicting_rule],
            limit=1,
            relations=[
                RuleRelationRecord(
                    first_id="string-boundary",
                    second_id="ownership-transfer",
                    relation="conflicts",
                    rationale="Borrowed string view conflicts with ownership transfer.",
                    evidence="review",
                )
            ],
        )

        self.assertEqual(
            [[rule.id for rule in rule_set] for rule_set in selected],
            [["string-boundary", "capacity-contract"]],
        )

    def test_structured_composition_relation_reviewer_parses_precondition_relation(self):
        string_rule = RuleRecord(
            id="fixed-c-string",
            title="Fixed C String",
            trigger="fixed char array",
            goal="Expose visible length",
            constraint="Trim at NUL",
            risk="zero padding leaks",
        )
        capacity_rule = RuleRecord(
            id="capacity-contract",
            title="Capacity Contract",
            trigger="count and capacity",
            goal="Preserve capacity",
            constraint="Do not recompute capacity",
            risk="capacity drift",
        )
        payloads = []

        def fake_reviewer(payload):
            payloads.append(payload)
            self.assertEqual(payload["rules"][0]["id"], "fixed-c-string")
            self.assertEqual(payload["rules"][1]["id"], "capacity-contract")
            return {
                "relation": "requires",
                "first_id": "fixed-c-string",
                "second_id": "capacity-contract",
                "rationale": "String trimming requires preserving the paired capacity field.",
            }

        relation = StructuredCompositionRelationReviewer(fake_reviewer).review(
            (string_rule, capacity_rule),
            {
                "fingerprint": "abc123",
                "tool_evidence": {"compiles": True, "tests_pass": True},
                "sanity_valid": True,
            },
        )

        self.assertIsNotNone(relation)
        self.assertEqual(relation.relation, "requires")
        self.assertEqual(relation.first_id, "fixed-c-string")
        self.assertEqual(relation.second_id, "capacity-contract")
        self.assertIn("requires preserving", relation.rationale)
        self.assertIn("challenge_fingerprint:abc123", relation.evidence)
        self.assertEqual(len(payloads), 1)

    def test_llm_composition_relation_reviewer_records_pair_ids(self):
        string_rule = RuleRecord(
            id="fixed-c-string",
            title="Fixed C String",
            trigger="fixed char array",
            goal="Expose visible length",
            constraint="Trim at NUL",
            risk="zero padding leaks",
        )
        capacity_rule = RuleRecord(
            id="capacity-contract",
            title="Capacity Contract",
            trigger="count and capacity",
            goal="Preserve capacity",
            constraint="Do not recompute capacity",
            risk="capacity drift",
        )
        calls = []

        class FakeAgent:
            def translate(self, _system, _prompt):
                return SimpleNamespace(
                    raw_text=json.dumps(
                        {
                            "relation": "requires",
                            "first_id": "fixed-c-string",
                            "second_id": "capacity-contract",
                            "rationale": "String handling requires capacity preservation.",
                        }
                    ),
                    elapsed_seconds=0.25,
                    usage=SimpleNamespace(
                        prompt_tokens=3,
                        completion_tokens=4,
                        total_tokens=7,
                    ),
                )

        relation = LLMCompositionRelationReviewer(FakeAgent(), call_recorder=calls.append).review(
            (string_rule, capacity_rule),
            {
                "fingerprint": "abc123",
                "tool_evidence": {"compiles": True, "tests_pass": True},
                "sanity_valid": True,
            },
        )

        self.assertIsNotNone(relation)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["phase"], "composition_relation_review")
        self.assertEqual(calls[0]["first_id"], "fixed-c-string")
        self.assertEqual(calls[0]["second_id"], "capacity-contract")
        self.assertEqual(calls[0]["usage"]["total_tokens"], 7)

    def test_synthetic_generator_builds_string_boundary_challenge_without_toolcheck(self):
        rule = RuleRecord(
            id="fixed-c-string",
            title="Fixed C String",
            trigger="fixed char array exposed through String::from_utf8_lossy",
            goal="Expose only the Rust-visible string length",
            constraint="Stop public views at the first NUL byte",
            risk="zero padding after NUL can leak into equality checks",
            tags=("strings", "nul-terminated-string"),
            keywords=("NUL", "from_utf8_lossy", "fixed char array"),
        )

        challenge = SyntheticChallengeGenerator().generate(rule)

        self.assertEqual(challenge.rule_id, "fixed-c-string")
        self.assertIn("challenge_buffer", challenge.c_program)
        self.assertIn("'\\0'", challenge.c_program)
        self.assertIn("pub fn translate_challenge", challenge.rust_interface)
        self.assertIn("#[test]", challenge.harness)
        self.assertIn("first NUL", challenge.oracle)
        self.assertFalse(challenge.evidence.tool_checked)

        sanity = ChallengeSanityChecker().check(challenge, rule)

        self.assertIn("toolcheck", sanity.failed_checks)
        self.assertNotIn("activates", sanity.failed_checks)

    def test_generated_synthetic_challenge_replays_reference_source_when_available(self):
        rule = RuleRecord(
            id="fixed-c-string",
            title="Fixed C String",
            trigger="fixed char array exposed through String::from_utf8_lossy",
            goal="Expose only the Rust-visible string length",
            constraint="Stop public views at the first NUL byte",
            risk="zero padding after NUL can leak into equality checks",
            tags=("strings", "nul-terminated-string"),
            keywords=("NUL", "from_utf8_lossy", "fixed char array"),
        )
        challenge = SyntheticChallengeGenerator().generate(rule)
        reference_source = challenge.trace_bundle["reference_source"]

        class FakeChecker:
            def __init__(self):
                self.rust_source = None

            def check(self, challenge, *, work_dir=None, rust_source=None):
                self.rust_source = rust_source
                return replace(
                    challenge,
                    evidence=ExecutionEvidence(
                        compiles=True,
                        tests_pass=rust_source == reference_source,
                        output="toolcheck ok NUL",
                    ),
                )

        checker = FakeChecker()

        event = adversarial_core._execute_generated_challenge(
            challenge,
            rule,
            tool_checker=checker,
            sanity_checker=ChallengeSanityChecker(),
        )

        self.assertIn("pub fn translate_challenge(input: &[u8]) -> String", reference_source)
        self.assertEqual(checker.rust_source, reference_source)
        self.assertTrue(event["reference_replay"])
        self.assertTrue(event["tool_evidence"]["tests_pass"])

    def test_synthetic_generator_builds_default_challenge_for_generic_rule(self):
        rule = RuleRecord(
            id="capacity-contract",
            title="Capacity Contract",
            trigger="C output struct has count and capacity fields",
            goal="Preserve caller-visible capacity",
            constraint="Do not recompute capacity from entries length",
            risk="tests can observe stale or changed capacity",
            tags=("struct", "capacity"),
            keywords=("capacity", "count"),
        )

        challenge = SyntheticChallengeGenerator().generate(rule)

        self.assertEqual(challenge.rule_id, "capacity-contract")
        self.assertIn("capacity", challenge.c_program.lower())
        self.assertIn("translate_challenge", challenge.rust_interface)
        self.assertIn("assert_eq!", challenge.harness)
        self.assertIn("Preserve caller-visible capacity", challenge.oracle)
        self.assertFalse(challenge.evidence.tool_checked)

    def test_synthetic_generator_builds_pointer_length_challenge(self):
        rule = RuleRecord(
            id="pointer-buffer-length",
            title="Pointer Buffer Length",
            trigger="C returns a pointer paired with an explicit buffer length",
            goal="Preserve the caller-visible byte length",
            constraint="Do not stop at interior zeros or infer length from sentinel bytes",
            risk="Pointer buffers with explicit length can lose trailing bytes",
            tags=("pointers", "buffers"),
            keywords=("pointer", "buffer", "length"),
        )

        challenge = SyntheticChallengeGenerator().generate(rule)

        self.assertEqual(challenge.rule_id, "pointer-buffer-length")
        self.assertIn("challenge_buffer", challenge.c_program)
        self.assertIn("translate_challenge(input: &[u8], len: usize)", challenge.rust_interface)
        self.assertIn("vec![1, 2, 0, 3, 4]", challenge.harness)
        self.assertIn("explicit buffer length", challenge.oracle)

    def test_synthetic_generator_builds_parser_cursor_state_challenge(self):
        rule = RuleRecord(
            id="parser-cursor-state",
            title="Parser Cursor State",
            trigger="C parser helpers share a cursor and consumed byte state",
            goal="Preserve parser cursor progress across helper calls",
            constraint="Do not reset cursor state between parse and consume helpers",
            risk="Parser translation can revisit bytes or report the wrong consumed offset",
            tags=("parser", "cursor", "state"),
            keywords=("parse", "consume", "cursor", "consumed"),
        )

        challenge = SyntheticChallengeGenerator().generate(rule)

        self.assertEqual(challenge.rule_id, "parser-cursor-state")
        self.assertIn("ParserState", challenge.c_program)
        self.assertIn("cursor", challenge.c_program)
        self.assertIn("translate_challenge(input: &[u8]) -> (usize, usize)", challenge.rust_interface)
        self.assertIn("assert_eq!(translate_challenge(input), (3, 3))", challenge.harness)
        self.assertIn("shared parser cursor", challenge.oracle)

    def test_synthetic_generator_builds_prng_sequence_challenge(self):
        rule = RuleRecord(
            id="prng-sequence-state",
            title="PRNG Sequence State",
            trigger="C code uses rand, srand, drand48, lrand48, or random calls",
            goal="Keep the same deterministic seeded pseudo-random sequence",
            constraint="Do not reseed or use wall-clock randomness for each call",
            risk="Random state changes tested observable order",
            tags=("random", "state", "sequence"),
            keywords=("lrand48", "srand48", "seed", "PRNG"),
        )

        challenge = SyntheticChallengeGenerator().generate(rule)

        self.assertEqual(challenge.rule_id, "prng-sequence-state")
        self.assertIn("challenge_prng_sequence", challenge.c_program)
        self.assertIn("1103515245u", challenge.c_program)
        self.assertIn("translate_challenge(seed: u32) -> (u32, u32)", challenge.rust_interface)
        self.assertIn("assert_eq!(translate_challenge(1), (1103527590, 377401575))", challenge.harness)
        self.assertIn("deterministic PRNG state", challenge.oracle)

    def test_synthetic_generator_builds_callback_user_data_state_challenge(self):
        rule = RuleRecord(
            id="callback-user-data-state",
            title="Preserve callback user_data state across invocations",
            trigger="C callbacks receive void *user_data state across multiple visitor calls",
            goal="Keep callback-carried state persistent across the traversal",
            constraint="Do not translate user_data into a fresh local callback variable",
            risk="The callback step counter can reset on every callback invocation",
            tags=("callbacks", "state", "user-data"),
            keywords=("user_data", "callback", "handler"),
        )

        challenge = SyntheticChallengeGenerator().generate(rule)

        self.assertEqual(challenge.rule_id, "callback-user-data-state")
        self.assertIn("challenge_visit_with_callback", challenge.c_program)
        self.assertIn("CallbackState", challenge.c_program)
        self.assertIn("user_data", challenge.c_program)
        self.assertIn("translate_challenge(values: &[i32]) -> i32", challenge.rust_interface)
        self.assertIn("assert_eq!(translate_challenge(&[2, 3, 5]), 10)", challenge.harness)
        self.assertIn("callback user_data state", challenge.oracle)

    def test_tool_checker_writes_c_and_rust_stub_and_marks_compiling_challenge(self):
        rule = RuleRecord(
            id="fixed-c-string",
            title="Fixed C String",
            trigger="fixed char array exposed through String::from_utf8_lossy",
            goal="Expose only the Rust-visible string length",
            constraint="Stop public views at the first NUL byte",
            risk="zero padding after NUL can leak into equality checks",
            tags=("strings",),
            keywords=("NUL", "fixed char array"),
        )
        challenge = SyntheticChallengeGenerator().generate(rule)
        calls = []

        def fake_runner(command, *, cwd, text, stdout, stderr, timeout):
            calls.append((command, Path(cwd)))
            if command[0] == "cc":
                self.assertTrue((Path(cwd) / "challenge.c").exists())
            if command[0] == "cargo":
                lib_rs = Path(cwd) / "src" / "lib.rs"
                self.assertIn("pub fn translate_challenge(input: &[u8]) -> String", lib_rs.read_text())
                self.assertIn("String::new()", lib_rs.read_text())
                self.assertIn("challenge_stops_at_first_nul", lib_rs.read_text())
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        with tempfile.TemporaryDirectory() as tmpdir:
            checked = ChallengeToolChecker(
                command_runner=fake_runner,
                c_compiler="cc",
                cargo="cargo",
            ).check(challenge, work_dir=Path(tmpdir))

        self.assertTrue(checked.evidence.tool_checked)
        self.assertTrue(checked.evidence.compiles)
        self.assertTrue(checked.evidence.tests_pass)
        self.assertEqual([call[0] for call in calls], [
            ["cc", "-std=c11", "-Wall", "-Wextra", "-c", "challenge.c", "-o", "challenge.o"],
            ["cc", "-std=c11", "-Wall", "-Wextra", "challenge_main.c", "-o", "challenge_c_check"],
            ["./challenge_c_check"],
            ["cargo", "test", "--no-run", "--quiet"],
            ["cargo", "test", "--quiet"],
        ])

    def test_tool_checker_runs_rust_harness_after_typecheck(self):
        challenge = ChallengeRecord(
            rule_id="capacity-contract",
            c_program="int challenge_value(void) { return 7; }",
            rust_interface="pub fn translate_challenge() -> i32;",
            harness="#[test]\nfn challenge_contract() { assert_eq!(translate_challenge(), 0); }",
            oracle="Preserve value",
            evidence=ExecutionEvidence(compiles=False, tests_pass=False),
        )
        calls = []

        def fake_runner(command, *, cwd, text, stdout, stderr, timeout):
            calls.append(command)
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        with tempfile.TemporaryDirectory() as tmpdir:
            checked = ChallengeToolChecker(
                command_runner=fake_runner,
                c_compiler="cc",
                cargo="cargo",
            ).check(challenge, work_dir=Path(tmpdir))

        self.assertTrue(checked.evidence.compiles)
        self.assertTrue(checked.evidence.tests_pass)
        self.assertEqual(
            calls,
            [
                ["cc", "-std=c11", "-Wall", "-Wextra", "-c", "challenge.c", "-o", "challenge.o"],
                ["cc", "-std=c11", "-Wall", "-Wextra", "challenge_main.c", "-o", "challenge_c_check"],
                ["./challenge_c_check"],
                ["cargo", "test", "--no-run", "--quiet"],
                ["cargo", "test", "--quiet"],
            ],
        )

    def test_tool_checker_can_replay_harness_against_supplied_rust_source(self):
        challenge = ChallengeRecord(
            rule_id="capacity-contract",
            c_program="int challenge_value(void) { return 7; }",
            rust_interface="pub fn translate_challenge() -> i32;",
            harness="#[test]\nfn challenge_contract() { assert_eq!(translate_challenge(), 7); }",
            oracle="Preserve value",
            evidence=ExecutionEvidence(compiles=False, tests_pass=False),
        )
        calls = []

        def fake_runner(command, *, cwd, text, stdout, stderr, timeout):
            calls.append(command)
            if command[0] == "cargo":
                lib_rs = Path(cwd) / "src" / "lib.rs"
                source = lib_rs.read_text()
                self.assertIn("pub fn translate_challenge() -> i32 { 7 }", source)
                self.assertNotIn("pub fn translate_challenge() -> i32 {\n    0\n}", source)
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        with tempfile.TemporaryDirectory() as tmpdir:
            checked = ChallengeToolChecker(
                command_runner=fake_runner,
                c_compiler="cc",
                cargo="cargo",
            ).check(
                challenge,
                work_dir=Path(tmpdir),
                rust_source="pub fn translate_challenge() -> i32 { 7 }",
            )

        self.assertTrue(checked.evidence.compiles)
        self.assertTrue(checked.evidence.tests_pass)
        self.assertIn(["cargo", "test", "--quiet"], calls)

    def test_tool_checker_reports_rust_harness_replay_failure(self):
        challenge = ChallengeRecord(
            rule_id="capacity-contract",
            c_program="int challenge_value(void) { return 7; }",
            rust_interface="pub fn translate_challenge() -> i32;",
            harness="#[test]\nfn challenge_contract() { assert_eq!(translate_challenge(), 7); }",
            oracle="Preserve value",
            evidence=ExecutionEvidence(compiles=False, tests_pass=False),
        )

        def fake_runner(command, *, cwd, text, stdout, stderr, timeout):
            if command == ["cargo", "test", "--quiet"]:
                return SimpleNamespace(returncode=1, stdout="", stderr="assertion failed")
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        with tempfile.TemporaryDirectory() as tmpdir:
            checked = ChallengeToolChecker(
                command_runner=fake_runner,
                c_compiler="cc",
                cargo="cargo",
            ).check(challenge, work_dir=Path(tmpdir))

        self.assertTrue(checked.evidence.compiles)
        self.assertFalse(checked.evidence.tests_pass)
        self.assertIn("cargo test failed", checked.evidence.output)
        self.assertIn("assertion failed", checked.evidence.output)

    def test_tool_checker_runs_executable_c_snippet_when_main_exists(self):
        challenge = ChallengeRecord(
            rule_id="capacity-contract",
            c_program="int main(void) { return 0; }",
            rust_interface="pub fn translate_challenge() -> i32;",
            harness="#[test]\nfn challenge_contract() { assert_eq!(translate_challenge(), 0); }",
            oracle="C snippet should be executable",
            evidence=ExecutionEvidence(compiles=False, tests_pass=False),
        )
        calls = []

        def fake_runner(command, *, cwd, text, stdout, stderr, timeout):
            calls.append(command)
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        with tempfile.TemporaryDirectory() as tmpdir:
            checked = ChallengeToolChecker(
                command_runner=fake_runner,
                c_compiler="cc",
                cargo="cargo",
            ).check(challenge, work_dir=Path(tmpdir))

        self.assertTrue(checked.evidence.compiles)
        self.assertTrue(checked.evidence.tests_pass)
        self.assertEqual(
            calls[:3],
            [
                ["cc", "-std=c11", "-Wall", "-Wextra", "-c", "challenge.c", "-o", "challenge.o"],
                ["cc", "-std=c11", "-Wall", "-Wextra", "challenge.c", "-o", "challenge_c_check"],
                ["./challenge_c_check"],
            ],
        )

    def test_tool_checker_generates_executable_wrapper_for_c_snippet_without_main(self):
        challenge = ChallengeRecord(
            rule_id="capacity-contract",
            c_program="int challenge_value(void) { return 7; }",
            rust_interface="pub fn translate_challenge() -> i32;",
            harness="#[test]\nfn challenge_contract() { assert_eq!(translate_challenge(), 0); }",
            oracle="C snippet should be linkable through a generated wrapper",
            evidence=ExecutionEvidence(compiles=False, tests_pass=False),
        )
        calls = []

        def fake_runner(command, *, cwd, text, stdout, stderr, timeout):
            calls.append(command)
            if command[0] == "cc" and "challenge_main.c" in command:
                wrapper = Path(cwd) / "challenge_main.c"
                self.assertTrue(wrapper.exists())
                self.assertIn('#include "challenge.c"', wrapper.read_text(encoding="utf-8"))
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        with tempfile.TemporaryDirectory() as tmpdir:
            checked = ChallengeToolChecker(
                command_runner=fake_runner,
                c_compiler="cc",
                cargo="cargo",
            ).check(challenge, work_dir=Path(tmpdir))

        self.assertTrue(checked.evidence.compiles)
        self.assertEqual(
            calls[:3],
            [
                ["cc", "-std=c11", "-Wall", "-Wextra", "-c", "challenge.c", "-o", "challenge.o"],
                ["cc", "-std=c11", "-Wall", "-Wextra", "challenge_main.c", "-o", "challenge_c_check"],
                ["./challenge_c_check"],
            ],
        )

    def test_tool_checker_reports_executable_c_snippet_failure(self):
        challenge = ChallengeRecord(
            rule_id="capacity-contract",
            c_program="int main(void) { return 1; }",
            rust_interface="pub fn translate_challenge() -> i32;",
            harness="#[test]\nfn challenge_contract() { assert_eq!(translate_challenge(), 0); }",
            oracle="C snippet should be executable",
            evidence=ExecutionEvidence(compiles=False, tests_pass=False),
        )
        calls = []

        def fake_runner(command, *, cwd, text, stdout, stderr, timeout):
            calls.append(command)
            if command == ["./challenge_c_check"]:
                return SimpleNamespace(returncode=1, stdout="", stderr="returned nonzero")
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        with tempfile.TemporaryDirectory() as tmpdir:
            checked = ChallengeToolChecker(
                command_runner=fake_runner,
                c_compiler="cc",
                cargo="cargo",
            ).check(challenge, work_dir=Path(tmpdir))

        self.assertFalse(checked.evidence.compiles)
        self.assertFalse(checked.evidence.tests_pass)
        self.assertIn("C snippet execution failed", checked.evidence.output)
        self.assertEqual([command[0] for command in calls], ["cc", "cc", "./challenge_c_check"])

    def test_tool_checker_reports_rust_typecheck_failure(self):
        challenge = ChallengeRecord(
            rule_id="capacity-contract",
            c_program="int challenge_value(void) { return 7; }",
            rust_interface="pub fn translate_challenge() -> (i32, i32);",
            harness="#[test]\nfn challenge_contract() { assert_eq!(translate_challenge(), (2, 8)); }",
            oracle="Preserve pair",
            evidence=ExecutionEvidence(compiles=False, tests_pass=False),
        )

        def fake_runner(command, *, cwd, text, stdout, stderr, timeout):
            if command[0] == "cargo":
                return SimpleNamespace(returncode=1, stdout="", stderr="type mismatch")
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        with tempfile.TemporaryDirectory() as tmpdir:
            checked = ChallengeToolChecker(
                command_runner=fake_runner,
                c_compiler="cc",
                cargo="cargo",
            ).check(challenge, work_dir=Path(tmpdir))

        self.assertFalse(checked.evidence.compiles)
        self.assertIn("cargo test --no-run failed", checked.evidence.output)

    def test_sanity_checker_rejects_duplicates_and_inactive_challenges(self):
        rule = RuleRecord(
            id="fixed-c-string",
            title="Fixed C String",
            trigger="fixed char array",
            goal="Expose visible length",
            constraint="Trim at NUL",
            risk="zero padding leaks",
            tags=("strings",),
            keywords=("from_utf8_lossy", "NUL"),
        )
        checker = ChallengeSanityChecker()
        first = ChallengeRecord(
            rule_id="fixed-c-string",
            c_program="char name[8];",
            rust_interface="pub fn name() -> String",
            harness="assert_eq!(name(), \"abc\")",
            oracle="abc",
            evidence=ExecutionEvidence(compiles=True, tests_pass=False, output="NUL padding"),
        )
        duplicate = ChallengeRecord(
            rule_id="fixed-c-string",
            c_program="char name[8];",
            rust_interface="pub fn name() -> String",
            harness="assert_eq!(name(), \"abc\")",
            oracle="abc",
            evidence=ExecutionEvidence(compiles=True, tests_pass=False, output="NUL padding"),
        )
        inactive = ChallengeRecord(
            rule_id="fixed-c-string",
            c_program="int add(int a, int b) { return a + b; }",
            rust_interface="pub fn add(a: i32, b: i32) -> i32",
            harness="assert_eq!(add(1, 2), 3)",
            oracle="3",
            evidence=ExecutionEvidence(compiles=True, tests_pass=True, output="ok"),
        )

        first_result = checker.check(first, rule)
        duplicate_result = checker.check(duplicate, rule)
        inactive_result = checker.check(inactive, rule)

        self.assertTrue(first_result.valid)
        self.assertIn("leak", duplicate_result.failed_checks)
        self.assertIn("activates", inactive_result.failed_checks)

    def test_sanity_checker_rejects_near_duplicate_challenges(self):
        rule = RuleRecord(
            id="fixed-c-string",
            title="Fixed C String",
            trigger="fixed char array",
            goal="Expose visible length",
            constraint="Trim at NUL",
            risk="zero padding leaks",
            tags=("strings",),
            keywords=("NUL",),
        )
        first = ChallengeRecord(
            rule_id="fixed-c-string",
            c_program="unsigned char name[8] = {'a','b','c','\\0','x','y','z','\\0'};",
            rust_interface="pub fn name() -> String;",
            harness="assert_eq!(name(), \"abc\");",
            oracle="Stop at first NUL",
            evidence=ExecutionEvidence(compiles=True, tests_pass=True, output="cargo test ok"),
        )
        same_boundary_with_noise = ChallengeRecord(
            rule_id="fixed-c-string",
            c_program="\n/* generated again */\nunsigned char name [ 8 ] = { 'a', 'b', 'c', '\\0', 'x', 'y', 'z', '\\0' };\n",
            rust_interface="pub fn name ( ) -> String;",
            harness="\nassert_eq!( name(), \"abc\" );\n",
            oracle="Stop at the first NUL byte",
            evidence=ExecutionEvidence(compiles=True, tests_pass=True, output="cargo test ok"),
        )
        checker = ChallengeSanityChecker()

        first_result = checker.check(first, rule)
        near_duplicate_result = checker.check(same_boundary_with_noise, rule)

        self.assertTrue(first_result.valid)
        self.assertIn("leak", near_duplicate_result.failed_checks)

    def test_sanity_checker_rejects_boundary_rule_without_boundary_payload(self):
        rule = RuleRecord(
            id="fixed-c-string",
            title="Fixed C String",
            trigger="fixed char array exposed through String::from_utf8_lossy",
            goal="Expose visible length",
            constraint="Trim at first NUL",
            risk="zero padding after NUL can leak",
            tags=("strings",),
            keywords=("NUL", "fixed char array"),
        )
        challenge = ChallengeRecord(
            rule_id="fixed-c-string",
            c_program="unsigned char name[8] = {'a','b','c','x','y','z','q','r'};",
            rust_interface="pub fn name() -> String;",
            harness="assert_eq!(name(), \"abc\");",
            oracle="The result should stop at the first NUL byte.",
            evidence=ExecutionEvidence(compiles=True, tests_pass=True, output="cargo test ok"),
        )

        result = ChallengeSanityChecker().check(challenge, rule)

        self.assertIn("activates", result.failed_checks)

    def test_sanity_checker_rejects_challenge_without_self_validating_oracle(self):
        rule = RuleRecord(
            id="capacity-contract",
            title="Capacity Contract",
            trigger="count and capacity",
            goal="Preserve capacity",
            constraint="Keep capacity field",
            risk="capacity drift",
            tags=("capacity",),
            keywords=("capacity",),
        )
        no_assertion = ChallengeRecord(
            rule_id="capacity-contract",
            c_program="typedef struct S { int count; int capacity; } S;",
            rust_interface="pub fn translate_challenge() -> (i32, i32);",
            harness="let _observed = translate_challenge();",
            oracle="",
            evidence=ExecutionEvidence(compiles=True, tests_pass=True, output="cargo test ok"),
        )

        result = ChallengeSanityChecker().check(no_assertion, rule)

        self.assertIn("oracle", result.failed_checks)

    def test_sanity_checker_rejects_harness_that_does_not_call_declared_interface(self):
        rule = RuleRecord(
            id="capacity-contract",
            title="Capacity Contract",
            trigger="count and capacity",
            goal="Preserve capacity",
            constraint="Keep capacity field",
            risk="capacity drift",
            tags=("capacity",),
            keywords=("capacity",),
        )
        wrong_call = ChallengeRecord(
            rule_id="capacity-contract",
            c_program="typedef struct S { int count; int capacity; } S;",
            rust_interface="pub fn translate_challenge() -> (i32, i32);",
            harness="#[test]\nfn challenge_contract() { assert_eq!(other_function(), (2, 8)); }",
            oracle="Preserve count and capacity",
            evidence=ExecutionEvidence(compiles=True, tests_pass=True, output="cargo test ok"),
        )

        result = ChallengeSanityChecker().check(wrong_call, rule)

        self.assertIn("oracle", result.failed_checks)

    def test_sanity_checker_rejects_mismatched_activation_predicate(self):
        rule = RuleRecord(
            id="fixed-c-string",
            title="Fixed C String",
            trigger="fixed char array exposed through String::from_utf8_lossy",
            goal="Expose only the Rust-visible string length",
            constraint="Stop public views at the first NUL byte",
            risk="zero padding after NUL can leak into equality checks",
            tags=("strings",),
            keywords=("NUL", "fixed char array"),
        )
        challenge = ChallengeRecord(
            rule_id="fixed-c-string",
            c_program="const unsigned char *challenge(void) { return (const unsigned char *)\"abc\\0pad\"; }",
            rust_interface="pub fn translate_challenge(input: &[u8]) -> String;",
            harness="#[test]\nfn generated() { assert_eq!(translate_challenge(b\"abc\\0pad\"), \"abc\"); }",
            oracle="The result should stop at the first NUL byte.",
            activation_predicate="binary heap sorting preserves floating point order",
            evidence=ExecutionEvidence(compiles=True, tests_pass=True, output="toolcheck ok"),
        )

        result = ChallengeSanityChecker().check(challenge, rule)

        self.assertIn("activates", result.failed_checks)

    def test_sanity_checker_can_use_activation_validator_for_generic_activation(self):
        rule = RuleRecord(
            id="callback-state",
            title="Callback State",
            trigger="C library invokes a cleanup callback after ownership transfer",
            goal="Preserve callback-visible state transitions",
            constraint="Do not drop the state before the cleanup callback observes it",
            risk="Callback state can be lost before cleanup",
            tags=("callback",),
            keywords=("ownership",),
        )
        challenge = ChallengeRecord(
            rule_id="callback-state",
            c_program="int challenge_value(int x) { return x + 1; }",
            rust_interface="pub fn translate_challenge(x: i32) -> i32;",
            harness="#[test]\nfn generated() { assert_eq!(translate_challenge(1), 2); }",
            oracle="The generated wrapper must preserve the observed transition result.",
            evidence=ExecutionEvidence(compiles=True, tests_pass=True, output="toolcheck ok"),
        )
        calls = []

        def fake_validator(payload):
            calls.append(payload)
            self.assertFalse(payload["local_activation_valid"])
            self.assertEqual(payload["rule"]["id"], "callback-state")
            return {"activates": True, "reason": "The transition result exercises the callback-state rule."}

        result = ChallengeSanityChecker(
            activation_validator=StructuredActivationValidator(fake_validator)
        ).check(challenge, rule)

        self.assertTrue(result.valid)
        self.assertEqual(calls[0]["challenge"]["rule_id"], "callback-state")

    def test_sanity_checker_rejects_activation_validator_negative_result(self):
        rule = RuleRecord(
            id="callback-state",
            title="Callback State",
            trigger="C library invokes a cleanup callback after ownership transfer",
            goal="Preserve callback-visible state transitions",
            constraint="Do not drop the state before the cleanup callback observes it",
            risk="Callback state can be lost before cleanup",
            tags=("callback",),
            keywords=("ownership",),
        )
        challenge = ChallengeRecord(
            rule_id="callback-state",
            c_program="int unrelated(int x) { return x + 1; }",
            rust_interface="pub fn translate_challenge(x: i32) -> i32;",
            harness="#[test]\nfn generated() { assert_eq!(translate_challenge(1), 2); }",
            oracle="The wrapper preserves an arithmetic result.",
            evidence=ExecutionEvidence(compiles=True, tests_pass=True, output="toolcheck ok"),
        )

        result = ChallengeSanityChecker(
            activation_validator=StructuredActivationValidator(
                lambda _payload: {"activates": False, "reason": "No callback behavior is exercised."}
            )
        ).check(challenge, rule)

        self.assertIn("activates_semantic", result.failed_checks)

    def test_sanity_checker_can_use_semantic_oracle_validator(self):
        rule = RuleRecord(
            id="fixed-c-string",
            title="Fixed C String",
            trigger="fixed char array exposed through String::from_utf8_lossy",
            goal="Expose only the Rust-visible string length",
            constraint="Stop public views at the first NUL byte",
            risk="zero padding after NUL can leak into equality checks",
            tags=("strings",),
            keywords=("NUL", "fixed char array"),
        )
        challenge = ChallengeRecord(
            rule_id="fixed-c-string",
            c_program="const unsigned char *challenge(void) { return (const unsigned char *)\"abc\\0pad\"; }",
            rust_interface="pub fn translate_challenge(input: &[u8]) -> String;",
            harness="#[test]\nfn generated() { assert_eq!(translate_challenge(b\"abc\\0pad\"), \"abc\"); }",
            oracle="The result should stop at the first NUL byte.",
            activation_predicate="fixed char array includes NUL padding",
            evidence=ExecutionEvidence(compiles=True, tests_pass=True, output="toolcheck ok"),
        )

        class RejectingOracleValidator:
            def validate(self, input_challenge, input_rule):
                self.challenge = input_challenge
                self.rule = input_rule
                return SemanticOracleValidationResult(False, "oracle does not cover visible length")

        validator = RejectingOracleValidator()
        result = ChallengeSanityChecker(oracle_validator=validator).check(challenge, rule)

        self.assertIn("oracle_semantic", result.failed_checks)
        self.assertIs(validator.challenge, challenge)
        self.assertIs(validator.rule, rule)


class AdversarialRefinementTest(unittest.TestCase):
    def test_structured_challenger_agent_parses_challenge_schema(self):
        rule = RuleRecord(
            id="fixed-c-string",
            title="Fixed C String",
            trigger="fixed char array",
            goal="Expose visible length",
            constraint="Trim at NUL",
            risk="padding may leak",
            tags=("strings",),
            keywords=("NUL",),
        )
        payloads = []

        def fake_agent(payload):
            payloads.append(payload)
            return json.dumps(
                {
                    "rule_id": "fixed-c-string",
                    "c_program": "const char *challenge(void) { return \"abc\"; }",
                    "rust_interface": "pub fn translate_challenge() -> String;",
                    "harness": "#[test]\nfn generated() { assert_eq!(translate_challenge(), \"abc\"); }",
                    "oracle": "The translated challenge must expose abc.",
                    "activation_predicate": "fixed char array is present",
                    "rationale": "string boundary check",
                }
            )

        challenge = StructuredChallengerAgent(fake_agent).generate(rule)

        self.assertEqual(payloads[0]["rule"]["id"], "fixed-c-string")
        self.assertIn("schema", payloads[0])
        self.assertEqual(challenge.rule_id, "fixed-c-string")
        self.assertIn("challenge", challenge.c_program)
        self.assertIn("fixed char array", challenge.activation_predicate)
        self.assertFalse(challenge.evidence.compiles)
        self.assertFalse(challenge.evidence.tests_pass)

    def test_structured_challenger_agent_rejects_invalid_schema(self):
        rule = RuleRecord(
            id="fixed-c-string",
            title="Fixed C String",
            trigger="fixed char array",
            goal="Expose visible length",
            constraint="Trim at NUL",
            risk="padding may leak",
        )

        with self.assertRaises(ValueError):
            StructuredChallengerAgent(lambda _payload: {"rule_id": "other"}).generate(rule)

    def test_llm_structured_challenger_agent_uses_translation_agent(self):
        rule = RuleRecord(
            id="fixed-c-string",
            title="Fixed C String",
            trigger="fixed char array",
            goal="Expose visible length",
            constraint="Trim at NUL",
            risk="padding may leak",
        )

        class FakeLiveAgent:
            def __init__(self):
                self.calls = []

            def translate(self, system, prompt):
                self.calls.append({"system": system, "prompt": prompt})
                return SimpleNamespace(
                    raw_text=(
                        "```json\n"
                        "{"
                        "\"rule_id\":\"fixed-c-string\","
                        "\"c_program\":\"const char *challenge(void) { return \\\"abc\\\"; }\","
                        "\"rust_interface\":\"pub fn translate_challenge() -> String;\","
                        "\"harness\":\"#[test]\\nfn generated() { assert_eq!(translate_challenge(), \\\"abc\\\"); }\","
                        "\"oracle\":\"The translated challenge must expose abc.\","
                        "\"activation_predicate\":\"fixed char array is present\""
                        "}\n"
                        "```"
                    ),
                    code="",
                )

        agent = FakeLiveAgent()
        challenge = LLMStructuredChallengerAgent(agent).generate(rule)

        self.assertEqual(challenge.rule_id, "fixed-c-string")
        self.assertIn("fixed char array", challenge.activation_predicate)
        self.assertEqual(len(agent.calls), 1)
        self.assertIn("Challenger", agent.calls[0]["system"])
        self.assertIn('"schema"', agent.calls[0]["prompt"])

    def test_llm_structured_challenger_agent_records_live_call_metadata(self):
        rule = RuleRecord(
            id="fixed-c-string",
            title="Fixed C String",
            trigger="fixed char array",
            goal="Expose visible length",
            constraint="Trim at NUL",
            risk="padding may leak",
        )

        class FakeLiveAgent:
            def translate(self, system, prompt):
                return SimpleNamespace(
                    raw_text=json.dumps(
                        {
                            "rule_id": "fixed-c-string",
                            "c_program": "const char *challenge(void) { return \"abc\"; }",
                            "rust_interface": "pub fn translate_challenge() -> String;",
                            "harness": "#[test]\nfn generated() { assert_eq!(translate_challenge(), \"abc\"); }",
                            "oracle": "The translated challenge must expose abc.",
                        }
                    ),
                    code="",
                    elapsed_seconds=0.25,
                    usage=SimpleNamespace(
                        prompt_tokens=11,
                        completion_tokens=13,
                        total_tokens=24,
                    ),
                )

        recorded_calls = []
        LLMStructuredChallengerAgent(FakeLiveAgent(), call_recorder=recorded_calls.append).generate(rule)

        self.assertEqual(len(recorded_calls), 1)
        self.assertEqual(recorded_calls[0]["phase"], "adversarial_structured_challenger")
        self.assertIn("Challenger", recorded_calls[0]["system"])
        self.assertIn('"schema"', recorded_calls[0]["prompt"])
        self.assertEqual(recorded_calls[0]["usage"]["total_tokens"], 24)

    def test_live_adversarial_system_prompts_are_detailed_json_contracts(self):
        prompts = [
            adversarial_core.LLM_STRUCTURED_CHALLENGER_SYSTEM,
            adversarial_core.LLM_STRUCTURED_REFINER_SYSTEM,
            adversarial_core.LLM_SEMANTIC_ORACLE_SYSTEM,
            adversarial_core.LLM_ACTIVATION_VALIDATOR_SYSTEM,
            adversarial_core.LLM_TRACE_SYNTHESIS_SYSTEM,
            adversarial_core.LLM_TRACE_CONTRADICTION_SYSTEM,
            adversarial_core.LLM_COMPOSITION_RELATION_SYSTEM,
        ]
        for prompt in prompts:
            self.assertIn("Return exactly one JSON object", prompt)
            self.assertIn("no markdown", prompt)
            self.assertIn("provided schema", prompt)
            self.assertIn("Do not add fields outside the schema", prompt)

        self.assertIn("self-contained C", adversarial_core.LLM_STRUCTURED_CHALLENGER_SYSTEM)
        self.assertIn("minimal counterexample", adversarial_core.LLM_STRUCTURED_CHALLENGER_SYSTEM)
        self.assertIn("harness must call the declared Rust interface", adversarial_core.LLM_STRUCTURED_CHALLENGER_SYSTEM)
        self.assertIn("EvidenceGate", adversarial_core.LLM_STRUCTURED_REFINER_SYSTEM)
        self.assertIn("Preserve the rule id", adversarial_core.LLM_STRUCTURED_REFINER_SYSTEM)
        self.assertIn("cite concrete challenge or trace evidence", adversarial_core.LLM_STRUCTURED_REFINER_SYSTEM)
        self.assertIn("tautological", adversarial_core.LLM_SEMANTIC_ORACLE_SYSTEM)
        self.assertIn("observable behavior", adversarial_core.LLM_SEMANTIC_ORACLE_SYSTEM)
        self.assertIn("similar vocabulary", adversarial_core.LLM_ACTIVATION_VALIDATOR_SYSTEM)
        self.assertIn("concrete input or state condition", adversarial_core.LLM_ACTIVATION_VALIDATOR_SYSTEM)
        self.assertIn("not a project-specific patch", adversarial_core.LLM_TRACE_SYNTHESIS_SYSTEM)
        self.assertIn("observable C/Rust semantic mismatch", adversarial_core.LLM_TRACE_SYNTHESIS_SYSTEM)
        self.assertIn("semantic conflicts", adversarial_core.LLM_TRACE_CONTRADICTION_SYSTEM)
        self.assertIn("mutually incompatible constraints", adversarial_core.LLM_TRACE_CONTRADICTION_SYSTEM)
        self.assertIn("passed sanity, compile, and test replay gates", adversarial_core.LLM_COMPOSITION_RELATION_SYSTEM)
        self.assertIn("Do not infer relations from id, title, or wording overlap", adversarial_core.LLM_COMPOSITION_RELATION_SYSTEM)

    def test_structured_refiner_agent_parses_update_schema(self):
        rule = RuleRecord(
            id="fixed-c-string",
            title="Fixed C String",
            trigger="fixed char array",
            goal="Expose visible length",
            constraint="Trim at NUL",
            risk="padding may leak",
        )
        challenge = SyntheticChallengeGenerator().generate(rule)

        def fake_agent(payload):
            self.assertEqual(payload["rule"]["id"], "fixed-c-string")
            self.assertIn("challenge", payload)
            self.assertIn("schema", payload)
            return {
                "rule_id": "fixed-c-string",
                "field": "constraint",
                "new_text": "Trim at NUL; preserve only bytes before the first NUL in public views",
                "rationale": "failure shows public NUL padding leakage",
                "failure_kind": "constraint_violation",
                "actionable": True,
            }

        proposal = StructuredRefinerAgent(fake_agent).propose_update(
            rule,
            challenge,
            ExecutionEvidence(compiles=True, tests_pass=False, output="assertion failed: NUL padding"),
        )

        self.assertEqual(proposal.rule_id, "fixed-c-string")
        self.assertEqual(proposal.field, "constraint")
        self.assertIn("first NUL", proposal.new_text)
        self.assertIn("constraint_violation", proposal.rationale)

    def test_structured_refiner_payload_includes_trace_bundle(self):
        rule = RuleRecord(
            id="fixed-c-string",
            title="Fixed C String",
            trigger="fixed char array",
            goal="Expose visible length",
            constraint="Trim at NUL",
            risk="padding may leak",
        )
        challenge = ChallengeRecord(
            rule_id="fixed-c-string",
            c_program="const char *challenge(void) { return \"abc\"; }",
            rust_interface="pub fn translate_challenge() -> String;",
            harness="#[test]\nfn generated() { assert_eq!(translate_challenge(), \"abc\"); }",
            oracle="The translated challenge must expose abc.",
            evidence=ExecutionEvidence(compiles=True, tests_pass=False, output="assertion failed"),
            trace_bundle={
                "generated_rust": "pub fn translate_challenge() -> String { String::new() }",
                "compiler_errors": ["error[E0308]"],
                "failing_assertions": ["assert_eq failed"],
            },
        )

        def fake_agent(payload):
            self.assertEqual(payload["trace_bundle"]["compiler_errors"], ["error[E0308]"])
            self.assertIn("generated_rust", payload["trace_bundle"])
            return {
                "rule_id": "fixed-c-string",
                "field": "risk",
                "new_text": "padding may leak; assertion failed in generated harness",
                "rationale": "trace bundle points to a missing risk",
                "actionable": True,
            }

        proposal = StructuredRefinerAgent(fake_agent).propose_update(
            rule,
            challenge,
            challenge.evidence,
        )

        self.assertEqual(proposal.field, "risk")

    def test_llm_structured_refiner_agent_uses_translation_agent(self):
        rule = RuleRecord(
            id="fixed-c-string",
            title="Fixed C String",
            trigger="fixed char array",
            goal="Expose visible length",
            constraint="Trim at NUL",
            risk="padding may leak",
        )
        challenge = SyntheticChallengeGenerator().generate(rule)

        class FakeLiveAgent:
            def __init__(self):
                self.calls = []

            def translate(self, system, prompt):
                self.calls.append({"system": system, "prompt": prompt})
                return SimpleNamespace(
                    raw_text=json.dumps(
                        {
                            "rule_id": "fixed-c-string",
                            "field": "constraint",
                            "new_text": "Trim at NUL and keep only visible bytes before it",
                            "rationale": "generated Rust leaked padded bytes",
                            "failure_kind": "constraint_violation",
                            "actionable": True,
                        }
                    ),
                    code="",
                )

        agent = FakeLiveAgent()
        proposal = LLMStructuredRefinerAgent(agent).propose_update(
            rule,
            challenge,
            ExecutionEvidence(compiles=True, tests_pass=False, output="assertion failed"),
        )

        self.assertEqual(proposal.field, "constraint")
        self.assertIn("visible bytes", proposal.new_text)
        self.assertEqual(len(agent.calls), 1)
        self.assertIn("Refiner", agent.calls[0]["system"])
        self.assertIn('"trace_bundle"', agent.calls[0]["prompt"])

    def test_structured_oracle_validator_parses_schema(self):
        rule = RuleRecord(
            id="fixed-c-string",
            title="Fixed C String",
            trigger="fixed char array",
            goal="Expose visible length",
            constraint="Trim at NUL",
            risk="padding may leak",
        )
        challenge = SyntheticChallengeGenerator().generate(rule)
        payloads = []

        def fake_validator(payload):
            payloads.append(payload)
            return {"valid": True, "reason": "oracle covers the challenge assertion"}

        result = StructuredOracleValidator(fake_validator).validate(challenge, rule)

        self.assertTrue(result.valid)
        self.assertEqual(result.reason, "oracle covers the challenge assertion")
        self.assertEqual(payloads[0]["rule"]["id"], "fixed-c-string")
        self.assertIn("schema", payloads[0])

    def test_structured_refiner_agent_rejects_non_actionable_updates(self):
        rule = RuleRecord(
            id="fixed-c-string",
            title="Fixed C String",
            trigger="fixed char array",
            goal="Expose visible length",
            constraint="Trim at NUL",
            risk="padding may leak",
        )
        challenge = SyntheticChallengeGenerator().generate(rule)

        with self.assertRaises(ValueError):
            StructuredRefinerAgent(lambda _payload: {"actionable": False}).propose_update(
                rule,
                challenge,
                ExecutionEvidence(compiles=True, tests_pass=False, output="invalid challenge"),
            )

    def test_failure_attributor_maps_nul_failure_to_risk_update(self):
        rule = RuleRecord(
            id="fixed-c-string",
            title="Fixed C String",
            trigger="fixed char array",
            goal="Expose visible length",
            constraint="Trim at NUL",
            risk="padding may leak",
            tags=("strings",),
            keywords=("NUL",),
        )

        proposal = FailureAttributor().propose_update(
            rule,
            ExecutionEvidence(compiles=True, tests_pass=False, output="assertion failed: zero padding after NUL"),
        )

        self.assertEqual(proposal.field, "risk")
        self.assertIn("zero padding after NUL", proposal.new_text)
        self.assertIn("missing_risk", proposal.rationale)

    def test_failure_attributor_classifies_common_failure_kinds(self):
        rule = RuleRecord(
            id="capacity-contract",
            title="Capacity Contract",
            trigger="count and capacity",
            goal="Preserve public API behavior",
            constraint="Keep capacity field",
            risk="capacity drift",
        )

        constraint = FailureAttributor().propose_update(
            rule,
            ExecutionEvidence(compiles=True, tests_pass=False, output="bounds check changed capacity"),
        )
        goal = FailureAttributor().propose_update(
            rule,
            ExecutionEvidence(compiles=True, tests_pass=False, output="public API signature mismatch"),
        )
        trigger = FailureAttributor().propose_update(
            rule,
            ExecutionEvidence(compiles=True, tests_pass=False, output="unexpected branch for empty input"),
        )

        self.assertEqual(constraint.field, "constraint")
        self.assertIn("constraint_violation", constraint.rationale)
        self.assertEqual(goal.field, "goal")
        self.assertIn("goal_mismatch", goal.rationale)
        self.assertEqual(trigger.field, "trigger")
        self.assertIn("trigger_gap", trigger.rationale)

    def test_evidence_gate_accepts_only_improving_replay(self):
        rule = RuleRecord(
            id="fixed-c-string",
            title="Fixed C String",
            trigger="fixed char array",
            goal="Expose visible length",
            constraint="Trim at NUL",
            risk="padding may leak",
        )
        proposal = RuleUpdateProposal(
            rule_id="fixed-c-string",
            field="risk",
            old_text="padding may leak",
            new_text="padding may leak; zero padding after NUL",
            rationale="test failure",
        )
        gate = EvidenceGate()

        accepted = gate.evaluate(
            rule,
            proposal,
            before=ExecutionEvidence(compiles=True, tests_pass=False, output="failed"),
            after=ExecutionEvidence(compiles=True, tests_pass=True, output="ok"),
        )
        rejected = gate.evaluate(
            rule,
            proposal,
            before=ExecutionEvidence(compiles=True, tests_pass=False, output="failed"),
            after=ExecutionEvidence(compiles=True, tests_pass=False, output="failed"),
        )

        self.assertTrue(accepted.accepted)
        self.assertEqual(accepted.updated_rule.version, 2)
        self.assertFalse(rejected.accepted)

    def test_rule_constrained_replay_runner_compares_before_and_after_updates(self):
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
        )
        proposal = RuleUpdateProposal(
            rule_id="capacity-contract",
            field="risk",
            old_text="capacity drift",
            new_text="capacity drift; replay fixed",
            rationale="after update uses passing translation",
        )
        translated_risks = []

        def fake_translate(input_challenge, input_rule):
            self.assertEqual(input_challenge.rule_id, "capacity-contract")
            translated_risks.append(input_rule.risk)
            if "replay fixed" in input_rule.risk:
                return "pub fn translate_challenge() -> i32 { 7 }"
            return "pub fn translate_challenge() -> i32 { 0 }"

        class FakeChecker:
            def check(self, input_challenge, *, rust_source=None, work_dir=None):
                return replace(
                    input_challenge,
                    evidence=ExecutionEvidence(
                        compiles=True,
                        tests_pass=" { 7 }" in str(rust_source),
                        output=str(rust_source),
                    ),
                )

        result = RuleConstrainedReplayRunner(
            fake_translate,
            tool_checker=FakeChecker(),
        ).evaluate_update(rule, challenge, proposal)

        self.assertTrue(result.accepted)
        self.assertEqual(translated_risks, ["capacity drift", "capacity drift; replay fixed"])
        self.assertEqual(result.updated_rule.version, 2)

    def test_translator_challenge_replay_adapter_uses_injected_translator_factory(self):
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
        created_configs = []

        class FakeTranslator:
            def __init__(self, config):
                self.config = config

            def translate_adversarial_challenge(self, input_challenge, input_rule):
                self.config["called"] = True
                return f"pub fn translate_challenge() -> i32 {{ {len(input_rule.risk)} }}"

        def factory(config):
            created_configs.append(config)
            return FakeTranslator(config)

        adapter = TranslatorChallengeReplayAdapter(
            factory,
            base_config={"model_display_name": "unit-model"},
        )
        rust_source = adapter(challenge, rule)

        self.assertIn("pub fn translate_challenge", rust_source)
        self.assertEqual(len(created_configs), 1)
        config = created_configs[0]
        self.assertEqual(config["model_display_name"], "unit-model")
        self.assertTrue(config["called"])
        self.assertEqual(config["adversarial_replay_challenge"]["rule_id"], "capacity-contract")
        self.assertEqual(
            config["adversarial_replay_challenge"]["activation_predicate"],
            "count <= capacity",
        )
        self.assertEqual(config["adversarial_replay_rule"]["id"], "capacity-contract")
        self.assertEqual(config["adversarial_replay_rule"]["risk"], "capacity drift")


class AdversarialTraceReflectionTest(unittest.TestCase):
    def test_trace_miner_extracts_candidate_rule_from_failed_trace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir) / "run" / "demo"
            project.mkdir(parents=True)
            (project / "translation_report.json").write_text(json.dumps({"success": False}), encoding="utf-8")
            (project / "test_fix_history.json").write_text(json.dumps([
                {
                    "callable_id": "parse_config",
                    "c_context": "C parser keeps caller-visible capacity 8 while count is 2",
                    "test_output": "assertion failed: capacity changed from 8 to 2",
                    "test_stats": {"failed": 1, "passed": 0},
                }
            ]), encoding="utf-8")

            candidates = TraceRuleMiner(RuleBank()).mine(Path(tmpdir) / "run")

            self.assertEqual(len(candidates), 1)
            self.assertTrue(candidates[0].id.startswith("trace-demo-parse-config-"))
            self.assertEqual(candidates[0].source, "trace")
            self.assertIn("caller-visible capacity 8", candidates[0].trigger)
            self.assertIn("capacity changed from 8 to 2", candidates[0].constraint)
            self.assertIn("trace-mined", candidates[0].tags)
            self.assertIn("capacity", candidates[0].keywords)

    def test_trace_miner_extracts_candidate_from_failed_compile_attempt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir) / "run" / "demo"
            project.mkdir(parents=True)
            (project / "translation_report.json").write_text(json.dumps({"success": False}), encoding="utf-8")
            (project / "translation_history.json").write_text(json.dumps({
                "src::fn::parse_config": {
                    "callable_id": "src::fn::parse_config",
                    "callable_label": "parse_config",
                    "c_binding": {
                        "functions": [
                            {
                                "name": "parse_config",
                                "code": (
                                    "int parse_config(char *buf, int capacity) "
                                    "{ return capacity; }"
                                ),
                            }
                        ]
                    },
                    "compilation_attempts": [
                        {
                            "attempt": 1,
                            "code": "pub fn parse_config() -> usize { entries.len() }",
                            "compilation_success": False,
                            "compilation_output": (
                                "error[E0308]: mismatched types while preserving "
                                "caller visible capacity metadata"
                            ),
                            "errors": [
                                {
                                    "message": "mismatched types while preserving capacity",
                                    "excerpt": "expected usize, found i32",
                                }
                            ],
                        }
                    ],
                    "success": False,
                }
            }), encoding="utf-8")

            candidates = TraceRuleMiner(RuleBank()).mine(Path(tmpdir) / "run")

            self.assertEqual(len(candidates), 1)
            candidate = candidates[0]
            self.assertTrue(candidate.id.startswith("trace-demo-src-fn-parse-config-"))
            self.assertEqual(candidate.source, "trace")
            self.assertIn("capacity", candidate.trigger)
            self.assertIn("mismatched types", candidate.constraint)
            self.assertIn("trace-mined", candidate.tags)
            self.assertIn("capacity", candidate.keywords)

    def test_trace_miner_can_use_structured_trace_synthesizer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir) / "run" / "demo"
            project.mkdir(parents=True)
            (project / "translation_report.json").write_text(json.dumps({"success": False}), encoding="utf-8")
            (project / "translation_history.json").write_text(json.dumps({
                "src::fn::parse_config": {
                    "final_code": "pub fn parse_config() -> (i32, i32) { (2, 2) }",
                    "compilation_attempts": [{"compilation_output": "ok"}],
                }
            }), encoding="utf-8")
            (project / "test_fix_history.json").write_text(json.dumps([
                {
                    "callable_id": "parse_config",
                    "c_context": "C parser stores count and capacity separately",
                    "test_output": "assertion failed: capacity changed from 8 to 2",
                    "test_stats": {"failed": 1, "passed": 0},
                }
            ]), encoding="utf-8")
            payloads = []

            def fake_synthesizer(payload):
                payloads.append(payload)
                self.assertIn("trace_bundle", payload)
                self.assertIn("generated_rust", payload["trace_bundle"])
                return {
                    "trigger": "C parser stores count and capacity separately",
                    "goal": "Preserve caller-visible allocation capacity",
                    "constraint": "Do not collapse capacity to the visible count",
                    "risk": "Capacity drift breaks caller allocation contracts",
                    "guidance": "Keep count and capacity as independent observable fields.",
                    "tags": ["trace-synthesized", "capacity"],
                    "keywords": ["capacity", "count"],
                }

            candidates = TraceRuleMiner(
                RuleBank(),
                synthesizer=StructuredTraceSynthesizer(fake_synthesizer),
            ).mine(Path(tmpdir) / "run")

            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0].source, "trace-llm")
            self.assertIn("allocation capacity", candidates[0].goal)
            self.assertEqual(payloads[0]["project"], "demo")
            self.assertEqual(payloads[0]["callable_id"], "parse_config")

    def test_trace_miner_can_use_structured_contradiction_reviewer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir) / "run" / "demo"
            project.mkdir(parents=True)
            (project / "translation_report.json").write_text(json.dumps({"success": False}), encoding="utf-8")
            (project / "test_fix_history.json").write_text(json.dumps([
                {
                    "callable_id": "parse_config",
                    "c_context": "C parser stores reservation and count as separate observable metadata",
                    "test_output": "assertion failed: reservation changed from 8 to 2 after translation",
                    "test_stats": {"failed": 1, "passed": 0},
                }
            ]), encoding="utf-8")
            bank = RuleBank([
                RuleRecord(
                    id="capacity-contract",
                    title="Capacity Contract",
                    trigger="C parser exposes allocation reservation separately from count",
                    goal="Preserve caller-visible reservation metadata",
                    constraint="Keep reservation independent from count.",
                    risk="Collapsing reservation to count loses allocation contract.",
                    tags=("capacity",),
                    keywords=("reservation", "count"),
                )
            ])
            payloads = []

            def fake_reviewer(payload):
                payloads.append(payload)
                self.assertEqual(payload["candidate"]["source"], "trace")
                self.assertEqual(payload["existing_rules"][0]["id"], "capacity-contract")
                return {
                    "contradicts": True,
                    "reason": "The candidate would reverse the existing reservation contract.",
                    "conflicting_rule_id": "capacity-contract",
                }

            candidates = TraceRuleMiner(
                bank,
                contradiction_reviewer=StructuredTraceContradictionReviewer(fake_reviewer),
            ).mine(Path(tmpdir) / "run")

            self.assertEqual(candidates, [])
            self.assertEqual(len(payloads), 1)

    def test_trace_miner_keeps_candidate_when_structured_contradiction_reviewer_allows_it(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir) / "run" / "demo"
            project.mkdir(parents=True)
            (project / "translation_report.json").write_text(json.dumps({"success": False}), encoding="utf-8")
            (project / "test_fix_history.json").write_text(json.dumps([
                {
                    "callable_id": "parse_config",
                    "c_context": "C parser stores reservation and count as separate observable metadata",
                    "test_output": "assertion failed: reservation changed from 8 to 2 after translation",
                    "test_stats": {"failed": 1, "passed": 0},
                }
            ]), encoding="utf-8")
            bank = RuleBank([
                RuleRecord(
                    id="unrelated",
                    title="Unrelated String Contract",
                    trigger="C fixed char array contains a NUL terminator",
                    goal="Expose only bytes before the terminator",
                    constraint="Trim public string views at the first NUL.",
                    risk="Padding bytes may leak into Rust strings.",
                    tags=("strings",),
                    keywords=("nul", "string"),
                )
            ])

            candidates = TraceRuleMiner(
                bank,
                contradiction_reviewer=StructuredTraceContradictionReviewer(
                    lambda _payload: {"contradicts": False, "reason": "No semantic conflict."}
                ),
            ).mine(Path(tmpdir) / "run")

            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0].source, "trace")

    def test_trace_miner_screens_duplicate_and_low_signal_candidates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir) / "run" / "demo"
            project.mkdir(parents=True)
            (project / "translation_report.json").write_text(json.dumps({"success": False}), encoding="utf-8")
            (project / "test_fix_history.json").write_text(json.dumps([
                {
                    "callable_id": "parse_config",
                    "c_context": "C parser keeps caller-visible capacity 8 while count is 2",
                    "test_output": "assertion failed: capacity changed from 8 to 2",
                    "test_stats": {"failed": 1, "passed": 0},
                },
                {
                    "callable_id": "other",
                    "test_output": "failed",
                    "test_stats": {"failed": 1, "passed": 0},
                },
            ]), encoding="utf-8")
            bank = RuleBank([
                RuleRecord(
                    id="capacity-existing",
                    title="Capacity Existing",
                    trigger="C parser keeps caller-visible capacity",
                    goal="Preserve capacity",
                    constraint="capacity changed from 8 to 2",
                    risk="capacity changed from 8 to 2",
                    tags=("capacity",),
                    keywords=("capacity",),
                )
            ])

            candidates = TraceRuleMiner(bank).mine(Path(tmpdir) / "run")

            self.assertEqual(candidates, [])

    def test_trace_miner_screens_project_specific_path_candidates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir) / "run" / "demo"
            project.mkdir(parents=True)
            leaked_path = f"{tmpdir}/output/runs/run/demo/generated/expected.txt"
            (project / "translation_report.json").write_text(json.dumps({"success": False}), encoding="utf-8")
            (project / "test_fix_history.json").write_text(json.dumps([
                {
                    "callable_id": "read_fixture",
                    "c_context": f"Generated test hard-codes project-local path {leaked_path}",
                    "test_output": f"assertion failed: fixture path {leaked_path} was not found",
                    "test_stats": {"failed": 1, "passed": 0},
                }
            ]), encoding="utf-8")

            candidates = TraceRuleMiner(RuleBank()).mine(Path(tmpdir) / "run")

            self.assertEqual(candidates, [])

    def test_trace_miner_screens_candidates_that_contradict_existing_rules(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir) / "run" / "demo"
            project.mkdir(parents=True)
            (project / "translation_report.json").write_text(json.dumps({"success": False}), encoding="utf-8")
            (project / "test_fix_history.json").write_text(json.dumps([
                {
                    "callable_id": "parse_config",
                    "c_context": (
                        "C parser count/capacity failure suggests recompute capacity from the "
                        "visible element count"
                    ),
                    "test_output": (
                        "assertion failed: recompute capacity from visible element count when "
                        "capacity changed"
                    ),
                    "test_stats": {"failed": 1, "passed": 0},
                }
            ]), encoding="utf-8")
            bank = RuleBank([
                RuleRecord(
                    id="capacity-contract",
                    title="Capacity Contract",
                    trigger="C parser stores count and capacity separately",
                    goal="Preserve caller-visible capacity",
                    constraint="Do not recompute capacity from count.",
                    risk="Recomputing capacity from count loses allocation contract.",
                    tags=("capacity",),
                    keywords=("capacity", "count"),
                )
            ])

            candidates = TraceRuleMiner(bank).mine(Path(tmpdir) / "run")

            self.assertEqual(candidates, [])

    def test_reflector_builds_challenge_from_failed_trace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir) / "run" / "demo"
            project.mkdir(parents=True)
            (project / "translation_report.json").write_text(json.dumps({"success": False}), encoding="utf-8")
            (project / "translation_history.json").write_text(json.dumps({
                "src::fn::name": {
                    "selected_rules": [{"id": "fixed-c-string"}],
                    "compilation_attempts": [
                        {"compilation_success": True, "compilation_output": "ok"}
                    ],
                    "final_code": "pub fn name() -> String { String::new() }",
                    "success": True,
                }
            }), encoding="utf-8")
            (project / "test_fix_history.json").write_text(json.dumps([
                {
                    "selected_rules": [{"id": "fixed-c-string"}],
                    "test_output": "assertion failed: zero padding after NUL",
                    "test_stats": {"failed": 1, "passed": 0},
                }
            ]), encoding="utf-8")

            bank = RuleBank([
                RuleRecord(
                    id="fixed-c-string",
                    title="Fixed C String",
                    trigger="fixed char array",
                    goal="Expose visible length",
                    constraint="Trim at NUL",
                    risk="padding may leak",
                    tags=("strings",),
                    keywords=("NUL",),
                )
            ])

            challenges = TraceReflector(bank).reflect(Path(tmpdir) / "run")

            self.assertEqual(len(challenges), 1)
            self.assertEqual(challenges[0].rule_id, "fixed-c-string")
            self.assertIn("zero padding", challenges[0].evidence.output)
            self.assertEqual(challenges[0].trace_bundle["test_stats"], {"failed": 1, "passed": 0})
            self.assertIn("zero padding", challenges[0].trace_bundle["test_output"])

    def test_reflector_builds_challenge_from_failed_compile_attempt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir) / "run" / "demo"
            project.mkdir(parents=True)
            (project / "translation_report.json").write_text(json.dumps({"success": False}), encoding="utf-8")
            (project / "translation_history.json").write_text(json.dumps({
                "src::fn::parse_config": {
                    "callable_id": "src::fn::parse_config",
                    "callable_label": "parse_config",
                    "c_binding": {
                        "functions": [
                            {
                                "name": "parse_config",
                                "code": "int parse_config(char *buf, int capacity) { return capacity; }",
                            }
                        ]
                    },
                    "selected_rules": [{"id": "capacity-contract"}],
                    "compilation_attempts": [
                        {
                            "attempt": 1,
                            "code": "pub fn parse_config() -> usize { entries.len() }",
                            "compilation_success": False,
                            "compilation_output": "error[E0308]: mismatched types for capacity",
                            "errors": [
                                {
                                    "message": "mismatched types for capacity",
                                    "excerpt": "expected usize, found i32",
                                }
                            ],
                            "selected_rules": [{"id": "capacity-contract"}],
                        }
                    ],
                    "final_code": "pub fn parse_config() -> usize { entries.len() }",
                    "success": False,
                }
            }), encoding="utf-8")

            bank = RuleBank([
                RuleRecord(
                    id="capacity-contract",
                    title="Capacity Contract",
                    trigger="C parser stores capacity separately from count",
                    goal="Preserve caller-visible capacity",
                    constraint="Do not collapse capacity to element count",
                    risk="Capacity drift can compile-fail or break callers",
                    tags=("capacity",),
                    keywords=("capacity",),
                )
            ])

            challenges = TraceReflector(bank).reflect(Path(tmpdir) / "run")

            self.assertEqual(len(challenges), 1)
            challenge = challenges[0]
            self.assertEqual(challenge.rule_id, "capacity-contract")
            self.assertFalse(challenge.evidence.compiles)
            self.assertFalse(challenge.evidence.tests_pass)
            self.assertIn("mismatched types", challenge.evidence.output)
            self.assertIn("parse_config", challenge.rust_interface)
            self.assertIn("parse_config", challenge.c_program)
            self.assertEqual(challenge.trace_bundle["source"], "translation_history")
            self.assertEqual(challenge.trace_bundle["callable_id"], "src::fn::parse_config")
            self.assertIn("compiler_errors", challenge.trace_bundle)
            self.assertIn("mismatched types", challenge.trace_bundle["compiler_errors"][0])
            self.assertIn("generated_rust", challenge.trace_bundle)

    def test_reflector_reads_in_progress_project_without_final_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir) / "run" / "demo"
            project.mkdir(parents=True)
            (project / "translation_history.json").write_text(json.dumps({
                "src::fn::parse_config": {
                    "callable_id": "src::fn::parse_config",
                    "callable_label": "parse_config",
                    "c_binding": {
                        "functions": [
                            {
                                "name": "parse_config",
                                "code": "int parse_config(char *buf, int capacity) { return capacity; }",
                            }
                        ]
                    },
                    "selected_rules": [{"id": "capacity-contract"}],
                    "compilation_attempts": [
                        {
                            "attempt": 1,
                            "code": "pub fn parse_config() -> usize { entries.len() }",
                            "compilation_success": False,
                            "compilation_output": "error[E0308]: mismatched capacity types",
                            "selected_rules": [{"id": "capacity-contract"}],
                        }
                    ],
                    "success": False,
                }
            }), encoding="utf-8")
            bank = RuleBank([
                RuleRecord(
                    id="capacity-contract",
                    title="Capacity Contract",
                    trigger="C parser stores capacity separately from count",
                    goal="Preserve caller-visible capacity",
                    constraint="Do not collapse capacity to element count",
                    risk="Capacity drift can compile-fail or break callers",
                    tags=("capacity",),
                    keywords=("capacity",),
                )
            ])

            challenges = TraceReflector(bank).reflect(Path(tmpdir) / "run")

            self.assertEqual(len(challenges), 1)
            self.assertEqual(challenges[0].rule_id, "capacity-contract")
            self.assertEqual(challenges[0].trace_bundle["project"], "demo")

    def test_reflector_attaches_later_successful_compile_replay_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir) / "run" / "demo"
            project.mkdir(parents=True)
            (project / "translation_report.json").write_text(json.dumps({"success": False}), encoding="utf-8")
            (project / "translation_history.json").write_text(json.dumps({
                "src::fn::parse_config": {
                    "callable_id": "src::fn::parse_config",
                    "callable_label": "parse_config",
                    "c_binding": {
                        "functions": [
                            {
                                "name": "parse_config",
                                "code": "int parse_config(char *buf, int capacity) { return capacity; }",
                            }
                        ]
                    },
                    "selected_rules": [{"id": "capacity-contract"}],
                    "compilation_attempts": [
                        {
                            "attempt": 1,
                            "code": "pub fn parse_config() -> usize { entries.len() }",
                            "compilation_success": False,
                            "compilation_output": "error[E0308]: mismatched capacity types",
                            "selected_rules": [{"id": "capacity-contract"}],
                        },
                        {
                            "attempt": 2,
                            "code": "pub fn parse_config() -> usize { capacity }",
                            "compilation_success": True,
                            "compilation_output": "Finished dev profile",
                        },
                    ],
                    "success": True,
                }
            }), encoding="utf-8")
            bank = RuleBank([
                RuleRecord(
                    id="capacity-contract",
                    title="Capacity Contract",
                    trigger="C parser stores capacity separately from count",
                    goal="Preserve caller-visible capacity",
                    constraint="Do not collapse capacity to element count",
                    risk="Capacity drift can compile-fail or break callers",
                    tags=("capacity",),
                    keywords=("capacity",),
                )
            ])

            challenges = TraceReflector(bank).reflect(Path(tmpdir) / "run")

            self.assertEqual(len(challenges), 1)
            replay = challenges[0].replay_evidence
            self.assertIsNotNone(replay)
            self.assertTrue(replay.compiles)
            self.assertFalse(replay.tests_pass)
            self.assertIn("Finished dev profile", replay.output)
            self.assertIn("replay_compilation_attempt", challenges[0].trace_bundle)

    def test_reflector_builds_challenge_from_llm_output_issue(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir) / "run" / "demo"
            project.mkdir(parents=True)
            (project / "translation_report.json").write_text(json.dumps({"success": False}), encoding="utf-8")
            llm_call = {
                "callable_id": "src::fn::parse_config",
                "callable_label": "parse_config",
                "phase": "translation",
                "output_issue": "unpatchable_output",
                "prompt": "Translate parse_config while preserving capacity metadata",
                "raw_text": "fn parse_config( {",
                "extracted_code": "",
                "rules": [{"id": "capacity-contract"}],
            }
            (project / "llm_calls.jsonl").write_text(
                json.dumps(llm_call, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            bank = RuleBank([
                RuleRecord(
                    id="capacity-contract",
                    title="Capacity Contract",
                    trigger="C parser stores capacity separately from count",
                    goal="Preserve caller-visible capacity",
                    constraint="Do not collapse capacity to element count",
                    risk="Capacity drift can compile-fail or break callers",
                    tags=("capacity",),
                    keywords=("capacity",),
                )
            ])

            challenges = TraceReflector(bank).reflect(Path(tmpdir) / "run")

            self.assertEqual(len(challenges), 1)
            challenge = challenges[0]
            self.assertEqual(challenge.rule_id, "capacity-contract")
            self.assertFalse(challenge.evidence.compiles)
            self.assertFalse(challenge.evidence.tests_pass)
            self.assertIsNone(challenge.replay_evidence)
            self.assertIn("unpatchable_output", challenge.evidence.output)
            self.assertIn("parse_config", challenge.rust_interface)
            self.assertEqual(challenge.trace_bundle["source"], "llm_calls")
            self.assertEqual(challenge.trace_bundle["output_issue"], "unpatchable_output")
            self.assertEqual(challenge.trace_bundle["phase"], "translation")

    def test_offline_runner_writes_updated_bank_and_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run"
            project = run_root / "demo"
            project.mkdir(parents=True)
            (project / "translation_report.json").write_text(json.dumps({"success": False}), encoding="utf-8")
            (project / "test_fix_history.json").write_text(json.dumps([
                {
                    "selected_rules": [{"id": "fixed-c-string"}],
                    "test_output": "assertion failed: zero padding after NUL",
                    "post_patch_test_output": "ok",
                    "post_patch_test_stats": {"failed": 0, "passed": 1},
                    "test_stats": {"failed": 1, "passed": 0},
                    "test_improved": True,
                }
            ]), encoding="utf-8")
            bank_path = Path(tmpdir) / "bank.json"
            out_bank = Path(tmpdir) / "updated.json"
            report_path = Path(tmpdir) / "report.json"
            RuleBank([
                RuleRecord(
                    id="fixed-c-string",
                    title="Fixed C String",
                    trigger="fixed char array",
                    goal="Expose visible length",
                    constraint="Trim at NUL",
                    risk="padding may leak",
                    tags=("strings",),
                    keywords=("NUL",),
                )
            ]).save(bank_path)

            result = run_offline_adversarial_learning(run_root, bank_path, out_bank, report_path)

            self.assertEqual(result["accepted_updates"], 1)
            self.assertTrue(out_bank.exists())
            self.assertTrue(report_path.exists())
            updated = RuleBank.load(out_bank)
            self.assertEqual(updated.records[0].version, 2)

    def test_offline_runner_can_use_fresh_replay_translation_for_trace_updates(self):
        class FakeToolChecker:
            def check(self, challenge, *, rust_source=None, work_dir=None):
                return replace(
                    challenge,
                    evidence=ExecutionEvidence(
                        compiles=True,
                        tests_pass="PASS" in str(rust_source),
                        output=str(rust_source),
                    ),
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run"
            project = run_root / "demo"
            project.mkdir(parents=True)
            (project / "translation_report.json").write_text(json.dumps({"success": False}), encoding="utf-8")
            (project / "test_fix_history.json").write_text(json.dumps([
                {
                    "selected_rules": [{"id": "fixed-c-string"}],
                    "test_output": "assertion failed: zero padding after NUL",
                    "test_stats": {"failed": 1, "passed": 0},
                }
            ]), encoding="utf-8")
            bank_path = Path(tmpdir) / "bank.json"
            out_bank = Path(tmpdir) / "updated.json"
            report_path = Path(tmpdir) / "report.json"
            RuleBank([
                RuleRecord(
                    id="fixed-c-string",
                    title="Fixed C String",
                    trigger="fixed char array",
                    goal="Expose visible length",
                    constraint="Trim at NUL",
                    risk="padding may leak",
                    tags=("strings",),
                    keywords=("NUL",),
                )
            ]).save(bank_path)
            translated_risks = []

            def fake_translate(input_challenge, input_rule):
                self.assertEqual(input_challenge.rule_id, "fixed-c-string")
                translated_risks.append(input_rule.risk)
                if "observed:" in input_rule.risk:
                    return "PASS"
                return "FAIL"

            result = run_offline_adversarial_learning(
                run_root,
                bank_path,
                out_bank,
                report_path,
                challenge_tool_checker=FakeToolChecker(),
                replay_translate_fn=fake_translate,
            )

            self.assertEqual(result["accepted_updates"], 1)
            self.assertEqual(len(translated_risks), 2)
            self.assertEqual(translated_risks[0], "padding may leak")
            self.assertIn("observed:", translated_risks[1])
            updated = RuleBank.load(out_bank)
            self.assertEqual(updated.records[0].version, 2)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(report["events"][0]["fresh_replay"])
            self.assertTrue(report["events"][0]["fresh_replay_evidence"]["after"]["tests_pass"])

    def test_offline_runner_can_use_structured_refiner_agent_for_trace_updates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run"
            project = run_root / "demo"
            project.mkdir(parents=True)
            (project / "translation_report.json").write_text(json.dumps({"success": False}), encoding="utf-8")
            (project / "test_fix_history.json").write_text(json.dumps([
                {
                    "selected_rules": [{"id": "fixed-c-string"}],
                    "test_output": "assertion failed: zero padding after NUL",
                    "post_patch_test_output": "ok",
                    "post_patch_test_stats": {"failed": 0, "passed": 1},
                    "test_stats": {"failed": 1, "passed": 0},
                    "test_improved": True,
                }
            ]), encoding="utf-8")
            bank_path = Path(tmpdir) / "bank.json"
            out_bank = Path(tmpdir) / "updated.json"
            report_path = Path(tmpdir) / "report.json"
            RuleBank([
                RuleRecord(
                    id="fixed-c-string",
                    title="Fixed C String",
                    trigger="fixed char array",
                    goal="Expose visible length",
                    constraint="Trim at NUL",
                    risk="padding may leak",
                    tags=("strings",),
                    keywords=("NUL",),
                )
            ]).save(bank_path)
            payloads = []

            def fake_refiner(payload):
                payloads.append(payload)
                return {
                    "rule_id": "fixed-c-string",
                    "field": "risk",
                    "new_text": "padding may leak; zero padding after NUL must be handled",
                    "rationale": "trace replay improved after the risk update",
                    "failure_kind": "missing_risk",
                    "actionable": True,
                }

            result = run_offline_adversarial_learning(
                run_root,
                bank_path,
                out_bank,
                report_path,
                refiner_agent=StructuredRefinerAgent(fake_refiner),
            )

            self.assertEqual(result["accepted_updates"], 1)
            self.assertEqual(payloads[0]["rule"]["id"], "fixed-c-string")
            self.assertIn("schema", payloads[0])
            updated = RuleBank.load(out_bank)
            self.assertIn("must be handled", updated.records[0].risk)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIn("missing_risk", report["events"][0]["proposal"]["rationale"])

    def test_offline_runner_can_admit_validated_trace_candidates(self):
        class PassingToolChecker:
            def check(self, challenge, *, work_dir=None):
                return ChallengeRecord(
                    rule_id=challenge.rule_id,
                    c_program=challenge.c_program,
                    rust_interface=challenge.rust_interface,
                    harness=challenge.harness,
                    oracle=challenge.oracle,
                    evidence=ExecutionEvidence(
                        compiles=True,
                        tests_pass=True,
                        output="cargo test ok count capacity",
                    ),
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run"
            project = run_root / "demo"
            project.mkdir(parents=True)
            (project / "translation_report.json").write_text(json.dumps({"success": False}), encoding="utf-8")
            (project / "test_fix_history.json").write_text(json.dumps([
                {
                    "callable_id": "parse_config",
                    "c_context": "C parser keeps caller-visible capacity 8 while count is 2",
                    "test_output": "assertion failed: capacity changed from 8 to 2",
                    "test_stats": {"failed": 1, "passed": 0},
                }
            ]), encoding="utf-8")
            bank_path = Path(tmpdir) / "bank.json"
            out_bank = Path(tmpdir) / "updated.json"
            report_path = Path(tmpdir) / "report.json"
            RuleBank([]).save(bank_path)

            result = run_offline_adversarial_learning(
                run_root,
                bank_path,
                out_bank,
                report_path,
                admit_trace_candidates=True,
                challenge_tool_checker=PassingToolChecker(),
            )

            self.assertEqual(result["admitted_candidates"], 1)
            updated = RuleBank.load(out_bank)
            self.assertEqual(len(updated.records), 1)
            self.assertEqual(updated.records[0].source, "trace")
            self.assertTrue(updated.records[0].id.startswith("trace-demo-parse-config-"))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["candidate_events"][0]["status"], "admitted")

    def test_offline_runner_can_admit_trace_synthesizer_candidates(self):
        class PassingToolChecker:
            def check(self, challenge, *, work_dir=None):
                return replace(
                    challenge,
                    evidence=ExecutionEvidence(
                        compiles=True,
                        tests_pass=True,
                        output="cargo test ok capacity count",
                    ),
                )

        def fake_synthesizer(_payload):
            return {
                "trigger": "C parser stores count and capacity separately",
                "goal": "Preserve caller-visible allocation capacity",
                "constraint": "Do not collapse capacity to the visible count",
                "risk": "Capacity drift breaks caller allocation contracts",
                "tags": ["trace-synthesized", "capacity"],
                "keywords": ["capacity", "count"],
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run"
            project = run_root / "demo"
            project.mkdir(parents=True)
            (project / "translation_report.json").write_text(json.dumps({"success": False}), encoding="utf-8")
            (project / "test_fix_history.json").write_text(json.dumps([
                {
                    "callable_id": "parse_config",
                    "c_context": "C parser stores count and capacity separately",
                    "test_output": "assertion failed: capacity changed from 8 to 2",
                    "test_stats": {"failed": 1, "passed": 0},
                }
            ]), encoding="utf-8")
            bank_path = Path(tmpdir) / "bank.json"
            out_bank = Path(tmpdir) / "updated.json"
            report_path = Path(tmpdir) / "report.json"
            RuleBank([]).save(bank_path)

            result = run_offline_adversarial_learning(
                run_root,
                bank_path,
                out_bank,
                report_path,
                admit_trace_candidates=True,
                challenge_tool_checker=PassingToolChecker(),
                trace_synthesizer=StructuredTraceSynthesizer(fake_synthesizer),
            )

            self.assertEqual(result["admitted_candidates"], 1)
            updated = RuleBank.load(out_bank)
            self.assertEqual(updated.records[0].source, "trace-llm")
            self.assertIn("allocation capacity", updated.records[0].goal)

    def test_offline_runner_can_reject_trace_candidates_with_contradiction_reviewer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run"
            project = run_root / "demo"
            project.mkdir(parents=True)
            (project / "translation_report.json").write_text(json.dumps({"success": False}), encoding="utf-8")
            (project / "test_fix_history.json").write_text(json.dumps([
                {
                    "callable_id": "parse_config",
                    "c_context": "C parser stores reservation and count as separate observable metadata",
                    "test_output": "assertion failed: reservation changed from 8 to 2 after translation",
                    "test_stats": {"failed": 1, "passed": 0},
                }
            ]), encoding="utf-8")
            bank_path = Path(tmpdir) / "bank.json"
            out_bank = Path(tmpdir) / "updated.json"
            report_path = Path(tmpdir) / "report.json"
            RuleBank([
                RuleRecord(
                    id="capacity-contract",
                    title="Capacity Contract",
                    trigger="C parser exposes allocation reservation separately from count",
                    goal="Preserve caller-visible reservation metadata",
                    constraint="Keep reservation independent from count.",
                    risk="Collapsing reservation to count loses allocation contract.",
                    tags=("capacity",),
                    keywords=("reservation", "count"),
                )
            ]).save(bank_path)

            result = run_offline_adversarial_learning(
                run_root,
                bank_path,
                out_bank,
                report_path,
                admit_trace_candidates=True,
                trace_contradiction_reviewer=StructuredTraceContradictionReviewer(
                    lambda _payload: {"contradicts": True, "reason": "Conflicts with capacity contract."}
                ),
            )

            self.assertEqual(result["admitted_candidates"], 0)
            updated = RuleBank.load(out_bank)
            self.assertEqual([record.id for record in updated.records], ["capacity-contract"])

    def test_cli_parser_accepts_trace_candidate_admission_flag(self):
        from scripts.adversarial_rule_run import build_parser

        args = build_parser().parse_args([
            "output/runs/demo",
            "--bank",
            "bank.json",
            "--output-bank",
            "updated.json",
            "--report",
            "report.json",
            "--admit-trace-candidates",
        ])

        self.assertTrue(args.admit_trace_candidates)

    def test_offline_runner_can_execute_synthetic_and_compositional_challenges(self):
        class FakeChecker:
            def __init__(self):
                self.checked_ids = []

            def check(self, challenge, *, work_dir=None, rust_source=None):
                self.checked_ids.append(challenge.rule_id)
                if rust_source is not None:
                    tests_pass = "composition_mutant" not in rust_source
                    return replace(
                        challenge,
                        evidence=ExecutionEvidence(compiles=True, tests_pass=tests_pass, output="probe"),
                    )
                return replace(
                    challenge,
                    evidence=ExecutionEvidence(compiles=True, tests_pass=True, output="toolcheck ok"),
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_root = root / "run"
            run_root.mkdir()
            bank_path = root / "bank.json"
            out_bank = root / "updated.json"
            report_path = root / "report.json"
            RuleBank([
                RuleRecord(
                    id="fixed-c-string",
                    title="Fixed C String",
                    trigger="fixed char array",
                    goal="Expose visible length",
                    constraint="Trim at NUL",
                    risk="zero padding leaks",
                    tags=("strings",),
                    keywords=("NUL",),
                ),
                RuleRecord(
                    id="capacity-contract",
                    title="Capacity Contract",
                    trigger="count and capacity",
                    goal="Preserve capacity",
                    constraint="Do not recompute capacity",
                    risk="capacity drift",
                    tags=("layout",),
                    keywords=("capacity",),
                ),
            ]).save(bank_path)
            checker = FakeChecker()

            result = run_offline_adversarial_learning(
                run_root,
                bank_path,
                out_bank,
                report_path,
                run_synthetic_challenges=True,
                run_compositional_challenges=True,
                challenge_tool_checker=checker,
            )
            relations = RuleBank.load(out_bank).relations

        self.assertEqual(result["synthetic_challenges"], 2)
        self.assertEqual(result["compositional_challenges"], 1)
        self.assertEqual(len(result["synthetic_events"]), 2)
        self.assertEqual(len(result["composition_events"]), 1)
        self.assertIn("composition:fixed-c-string+capacity-contract", checker.checked_ids)
        self.assertTrue(all(event["sanity_valid"] for event in result["synthetic_events"]))
        self.assertTrue(all(event["sanity_valid"] for event in result["composition_events"]))
        self.assertTrue(result["composition_events"][0]["interaction_dependency_valid"])
        self.assertEqual(result["relation_updates"], 1)
        self.assertEqual(result["composition_events"][0]["relation_update"]["relation"], "supports")
        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0].first_id, "fixed-c-string")
        self.assertEqual(relations[0].second_id, "capacity-contract")

    def test_offline_runner_replays_composition_with_translator_when_available(self):
        class FakeChecker:
            def check(self, challenge, *, work_dir=None, rust_source=None):
                if rust_source is not None:
                    tests_pass = (
                        "translator-combined" in rust_source
                        or ("composition_mutant" not in rust_source and "translator-single" not in rust_source)
                    )
                    return replace(
                        challenge,
                        evidence=ExecutionEvidence(compiles=True, tests_pass=tests_pass, output="probe"),
                    )
                return replace(
                    challenge,
                    evidence=ExecutionEvidence(compiles=True, tests_pass=True, output="toolcheck ok"),
                )

        translated_pairs = []

        def fake_translate(challenge, rule):
            translated_pairs.append((challenge.rule_id, rule.id))
            if rule.id.startswith("composition:"):
                return "translator-combined"
            return f"translator-single {rule.id}"

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_root = root / "run"
            run_root.mkdir()
            bank_path = root / "bank.json"
            out_bank = root / "updated.json"
            report_path = root / "report.json"
            RuleBank([
                RuleRecord(
                    id="fixed-c-string",
                    title="Fixed C String",
                    trigger="fixed char array",
                    goal="Expose visible length",
                    constraint="Trim at NUL",
                    risk="zero padding leaks",
                    tags=("strings",),
                    keywords=("NUL",),
                ),
                RuleRecord(
                    id="capacity-contract",
                    title="Capacity Contract",
                    trigger="count and capacity",
                    goal="Preserve capacity",
                    constraint="Do not recompute capacity",
                    risk="capacity drift",
                    tags=("layout",),
                    keywords=("capacity",),
                ),
            ]).save(bank_path)

            result = run_offline_adversarial_learning(
                run_root,
                bank_path,
                out_bank,
                report_path,
                run_compositional_challenges=True,
                challenge_tool_checker=FakeChecker(),
                replay_translate_fn=fake_translate,
            )

        event = result["composition_events"][0]
        self.assertTrue(event["translator_replay_dependency_valid"])
        self.assertEqual(len(event["translator_replay_dependency_probes"]), 3)
        self.assertIn(
            (
                "composition:fixed-c-string+capacity-contract",
                "composition:fixed-c-string+capacity-contract",
            ),
            translated_pairs,
        )
        self.assertIn(
            ("composition:fixed-c-string+capacity-contract", "fixed-c-string"),
            translated_pairs,
        )
        self.assertEqual(result["relation_updates"], 1)

    def test_composition_relation_requires_interaction_dependency_evidence(self):
        rules = (
            RuleRecord(
                id="fixed-c-string",
                title="Fixed C String",
                trigger="fixed char array",
                goal="Expose visible length",
                constraint="Trim at NUL",
                risk="zero padding leaks",
            ),
            RuleRecord(
                id="capacity-contract",
                title="Capacity Contract",
                trigger="count and capacity",
                goal="Preserve capacity",
                constraint="Do not recompute capacity",
                risk="capacity drift",
            ),
        )
        event = {
            "sanity_valid": True,
            "fingerprint": "abc",
            "interaction_dependency_valid": False,
            "tool_evidence": {"compiles": True, "tests_pass": True, "output": "ok"},
        }

        relation = adversarial_core._composition_relation_record(rules, event)

        self.assertIsNone(relation)

    def test_composition_relation_requires_translator_replay_dependency_when_present(self):
        rules = (
            RuleRecord(
                id="fixed-c-string",
                title="Fixed C String",
                trigger="fixed char array",
                goal="Expose visible length",
                constraint="Trim at NUL",
                risk="zero padding leaks",
            ),
            RuleRecord(
                id="capacity-contract",
                title="Capacity Contract",
                trigger="count and capacity",
                goal="Preserve capacity",
                constraint="Do not recompute capacity",
                risk="capacity drift",
            ),
        )
        event = {
            "sanity_valid": True,
            "fingerprint": "abc",
            "interaction_dependency_valid": True,
            "translator_replay_dependency_valid": False,
            "tool_evidence": {"compiles": True, "tests_pass": True, "output": "ok"},
        }

        relation = adversarial_core._composition_relation_record(rules, event)

        self.assertIsNone(relation)

    def test_composition_relation_infers_requires_from_local_relation_hint(self):
        rules = (
            RuleRecord(
                id="string-boundary",
                title="String Boundary",
                trigger="fixed char array",
                goal="Expose visible string bytes",
                constraint="String boundary handling requires capacity preservation",
                risk="trimming can hide capacity drift",
                tags=("strings",),
                keywords=("string",),
            ),
            RuleRecord(
                id="capacity-contract",
                title="Capacity Contract",
                trigger="count and capacity",
                goal="Preserve capacity",
                constraint="Do not recompute capacity",
                risk="capacity drift",
                tags=("layout",),
                keywords=("capacity",),
            ),
        )
        event = {
            "sanity_valid": True,
            "fingerprint": "abc",
            "interaction_dependency_valid": True,
            "tool_evidence": {"compiles": True, "tests_pass": True, "output": "ok"},
        }

        relation = adversarial_core._composition_relation_record(rules, event)

        self.assertIsNotNone(relation)
        self.assertEqual(relation.relation, "requires")
        self.assertEqual(relation.first_id, "string-boundary")
        self.assertEqual(relation.second_id, "capacity-contract")
        self.assertIn("requires", relation.rationale)

    def test_composition_relation_infers_precedes_from_local_relation_hint(self):
        rules = (
            RuleRecord(
                id="cursor-normalization",
                title="Cursor Normalization",
                trigger="scanner byte cursor",
                goal="Normalize cursor before parser state is consumed",
                constraint="Apply cursor normalization before parser state sharing",
                risk="late normalization revisits bytes",
                tags=("scanner",),
                keywords=("cursor",),
            ),
            RuleRecord(
                id="parser-state",
                title="Parser State",
                trigger="parser state sharing",
                goal="Preserve parser state",
                constraint="Share parser cursor state across calls",
                risk="state drift",
                tags=("parser",),
                keywords=("parser", "state"),
            ),
        )
        event = {
            "sanity_valid": True,
            "fingerprint": "abc",
            "interaction_dependency_valid": True,
            "tool_evidence": {"compiles": True, "tests_pass": True, "output": "ok"},
        }

        relation = adversarial_core._composition_relation_record(rules, event)

        self.assertIsNotNone(relation)
        self.assertEqual(relation.relation, "precedes")
        self.assertEqual(relation.first_id, "cursor-normalization")
        self.assertEqual(relation.second_id, "parser-state")
        self.assertIn("before", relation.rationale)

    def test_composition_dependency_ignores_bare_rule_id_mentions(self):
        rules = (
            RuleRecord(
                id="alpha-contract",
                title="Alpha Contract",
                trigger="pointer length buffer",
                goal="Preserve pointed buffer length",
                constraint="Keep slice length exact",
                risk="buffer aliasing",
                tags=("pointer",),
                keywords=("slice", "length"),
            ),
            RuleRecord(
                id="beta-contract",
                title="Beta Contract",
                trigger="capacity count layout",
                goal="Preserve caller-visible capacity",
                constraint="Do not recompute capacity",
                risk="capacity drift",
                tags=("capacity",),
                keywords=("count", "capacity"),
            ),
        )
        challenge = ChallengeRecord(
            rule_id="composition:alpha-contract+beta-contract",
            c_program="",
            rust_interface="",
            harness="",
            oracle="The selected rules alpha-contract and beta-contract must both hold.",
            evidence=ExecutionEvidence(compiles=True, tests_pass=True),
        )

        dependency = adversarial_core._composition_interaction_dependency(rules, challenge)

        self.assertFalse(dependency["valid"])
        self.assertEqual(dependency["matches"], {"alpha-contract": [], "beta-contract": []})

    def test_composition_dependency_replays_counterfactual_probes_when_available(self):
        rules = (
            RuleRecord(
                id="fixed-c-string",
                title="Fixed C String",
                trigger="fixed char array",
                goal="Expose visible length",
                constraint="Trim at NUL",
                risk="zero padding leaks",
                tags=("strings",),
                keywords=("NUL",),
            ),
            RuleRecord(
                id="capacity-contract",
                title="Capacity Contract",
                trigger="count and capacity",
                goal="Preserve capacity",
                constraint="Do not recompute capacity",
                risk="capacity drift",
                tags=("layout",),
                keywords=("capacity",),
            ),
        )
        challenge = CompositionalChallengeGenerator().generate(rules)

        class FakeChecker:
            def __init__(self):
                self.sources = []

            def check(self, challenge, *, work_dir=None, rust_source=None):
                self.sources.append(rust_source)
                tests_pass = "composition_mutant" not in (rust_source or "")
                return replace(
                    challenge,
                    evidence=ExecutionEvidence(compiles=True, tests_pass=tests_pass, output="probe"),
                )

        checker = FakeChecker()

        dependency = adversarial_core._composition_interaction_dependency(
            rules,
            challenge,
            tool_checker=checker,
        )

        self.assertTrue(dependency["valid"])
        self.assertTrue(dependency["executable_dependency_valid"])
        self.assertEqual(len(dependency["executable_dependency_probes"]), 3)
        self.assertEqual(len(checker.sources), 3)
        self.assertTrue(any(source and "reference composition implementation" in source for source in checker.sources))

    def test_offline_runner_can_write_reviewed_composition_precondition_relation(self):
        class FakeChecker:
            def check(self, challenge, *, work_dir=None, rust_source=None):
                if rust_source is not None:
                    tests_pass = "composition_mutant" not in rust_source
                    return replace(
                        challenge,
                        evidence=ExecutionEvidence(compiles=True, tests_pass=tests_pass, output="probe"),
                    )
                return replace(
                    challenge,
                    evidence=ExecutionEvidence(compiles=True, tests_pass=True, output="toolcheck ok"),
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_root = root / "run"
            run_root.mkdir()
            bank_path = root / "bank.json"
            out_bank = root / "updated.json"
            report_path = root / "report.json"
            RuleBank([
                RuleRecord(
                    id="fixed-c-string",
                    title="Fixed C String",
                    trigger="fixed char array",
                    goal="Expose visible length",
                    constraint="Trim at NUL",
                    risk="zero padding leaks",
                    tags=("strings",),
                    keywords=("NUL",),
                ),
                RuleRecord(
                    id="capacity-contract",
                    title="Capacity Contract",
                    trigger="count and capacity",
                    goal="Preserve capacity",
                    constraint="Do not recompute capacity",
                    risk="capacity drift",
                    tags=("layout",),
                    keywords=("capacity",),
                ),
            ]).save(bank_path)

            result = run_offline_adversarial_learning(
                run_root,
                bank_path,
                out_bank,
                report_path,
                run_compositional_challenges=True,
                challenge_tool_checker=FakeChecker(),
                composition_relation_reviewer=StructuredCompositionRelationReviewer(
                    lambda _payload: {
                        "relation": "requires",
                        "first_id": "fixed-c-string",
                        "second_id": "capacity-contract",
                        "rationale": "String boundary handling requires capacity preservation.",
                    }
                ),
            )
            relations = RuleBank.load(out_bank).relations

        self.assertEqual(result["relation_updates"], 1)
        self.assertEqual(result["composition_events"][0]["relation_update"]["relation"], "requires")
        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0].relation, "requires")
        self.assertIn("capacity preservation", relations[0].rationale)

    def test_offline_runner_can_apply_semantic_oracle_validator_to_synthetic_challenges(self):
        class FakeChecker:
            def check(self, challenge, *, work_dir=None):
                return replace(
                    challenge,
                    evidence=ExecutionEvidence(compiles=True, tests_pass=True, output="toolcheck ok NUL"),
                )

        class RejectingOracleValidator:
            def validate(self, challenge, rule):
                return SemanticOracleValidationResult(False, "semantic oracle rejected")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_root = root / "run"
            run_root.mkdir()
            bank_path = root / "bank.json"
            out_bank = root / "updated.json"
            report_path = root / "report.json"
            RuleBank([
                RuleRecord(
                    id="fixed-c-string",
                    title="Fixed C String",
                    trigger="fixed char array",
                    goal="Expose visible length",
                    constraint="Trim at NUL",
                    risk="zero padding leaks",
                    tags=("strings",),
                    keywords=("NUL",),
                )
            ]).save(bank_path)

            result = run_offline_adversarial_learning(
                run_root,
                bank_path,
                out_bank,
                report_path,
                run_synthetic_challenges=True,
                challenge_tool_checker=FakeChecker(),
                oracle_validator=RejectingOracleValidator(),
            )

        self.assertIn("oracle_semantic", result["synthetic_events"][0]["failed_checks"])

    def test_offline_runner_can_admit_composition_failure_candidate(self):
        class FakeChecker:
            def check(self, challenge, *, work_dir=None, rust_source=None):
                if rust_source is not None:
                    tests_pass = "composition_mutant" not in rust_source
                    return replace(
                        challenge,
                        evidence=ExecutionEvidence(compiles=True, tests_pass=tests_pass, output="probe"),
                    )
                if challenge.rule_id.startswith("composition:"):
                    return replace(
                        challenge,
                        evidence=ExecutionEvidence(
                            compiles=True,
                            tests_pass=False,
                            output="assertion failed for NUL count capacity interaction",
                        ),
                    )
                return replace(
                    challenge,
                    evidence=ExecutionEvidence(
                        compiles=True,
                        tests_pass=True,
                        output="candidate toolcheck ok NUL count capacity",
                    ),
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_root = root / "run"
            run_root.mkdir()
            bank_path = root / "bank.json"
            out_bank = root / "updated.json"
            report_path = root / "report.json"
            RuleBank([
                RuleRecord(
                    id="fixed-c-string",
                    title="Fixed C String",
                    trigger="fixed char array",
                    goal="Expose visible length",
                    constraint="Trim at NUL",
                    risk="zero padding leaks",
                    tags=("strings",),
                    keywords=("NUL",),
                ),
                RuleRecord(
                    id="capacity-contract",
                    title="Capacity Contract",
                    trigger="count and capacity",
                    goal="Preserve capacity",
                    constraint="Do not recompute capacity",
                    risk="capacity drift",
                    tags=("layout",),
                    keywords=("capacity",),
                ),
            ]).save(bank_path)

            result = run_offline_adversarial_learning(
                run_root,
                bank_path,
                out_bank,
                report_path,
                run_compositional_challenges=True,
                challenge_tool_checker=FakeChecker(),
                admit_composition_failure_candidates=True,
            )
            updated = RuleBank.load(out_bank)

        self.assertEqual(result["admitted_candidates"], 1)
        self.assertEqual(result["candidate_events"][0]["source"], "composition")
        self.assertEqual(result["candidate_events"][0]["status"], "admitted")
        self.assertTrue(any(record.source == "composition" for record in updated.records))

    def test_offline_runner_can_use_structured_challenger_agent_for_synthetic_challenges(self):
        class FakeChecker:
            def check(self, challenge, *, work_dir=None):
                return replace(
                    challenge,
                    evidence=ExecutionEvidence(compiles=True, tests_pass=True, output="toolcheck ok NUL"),
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_root = root / "run"
            run_root.mkdir()
            bank_path = root / "bank.json"
            out_bank = root / "updated.json"
            report_path = root / "report.json"
            RuleBank([
                RuleRecord(
                    id="fixed-c-string",
                    title="Fixed C String",
                    trigger="fixed char array",
                    goal="Expose visible length",
                    constraint="Trim at NUL",
                    risk="zero padding leaks",
                    tags=("strings",),
                    keywords=("NUL",),
                )
            ]).save(bank_path)
            payloads = []

            def fake_challenger(payload):
                payloads.append(payload)
                return {
                    "rule_id": "fixed-c-string",
                    "c_program": "const char *challenge(void) { return \"abc\\0pad\"; }",
                    "rust_interface": "pub fn translate_challenge() -> String;",
                    "harness": (
                        "#[test]\n"
                        "fn generated() { assert_eq!(translate_challenge(), \"abc\"); }"
                    ),
                    "oracle": "The translated challenge trims public output at the first NUL.",
                    "activation_predicate": "fixed char array includes a NUL boundary",
                }

            result = run_offline_adversarial_learning(
                run_root,
                bank_path,
                out_bank,
                report_path,
                run_synthetic_challenges=True,
                challenge_tool_checker=FakeChecker(),
                challenger_agent=StructuredChallengerAgent(fake_challenger),
            )

        self.assertEqual(result["synthetic_challenges"], 1)
        self.assertEqual(payloads[0]["rule"]["id"], "fixed-c-string")
        self.assertIn("schema", payloads[0])
        self.assertTrue(result["synthetic_events"][0]["sanity_valid"])

    def test_cli_parser_accepts_synthetic_and_compositional_execution_flags(self):
        from scripts.adversarial_rule_run import build_parser

        args = build_parser().parse_args([
            "output/runs/demo",
            "--bank",
            "bank.json",
            "--output-bank",
            "updated.json",
            "--report",
            "report.json",
            "--run-synthetic-challenges",
            "--run-compositional-challenges",
        ])

        self.assertTrue(args.run_synthetic_challenges)
        self.assertTrue(args.run_compositional_challenges)

    def test_cli_full_profile_defaults_live_adversarial_flags(self):
        from scripts import adversarial_rule_run as cli

        args = cli.build_parser().parse_args([
            "output/runs/demo",
            "--bank",
            "bank.json",
            "--output-bank",
            "updated.json",
            "--report",
            "report.json",
            "--experiment-profile",
            "full",
        ])

        flags = cli.adversarial_learning_flags(args)

        self.assertTrue(flags["admit_trace_candidates"])
        self.assertTrue(flags["run_synthetic_challenges"])
        self.assertTrue(flags["run_compositional_challenges"])
        self.assertTrue(flags["admit_composition_failure_candidates"])
        self.assertTrue(flags["live_structured_agents"])
        self.assertTrue(flags["live_trace_synthesis"])
        self.assertTrue(flags["live_trace_contradiction_review"])
        self.assertTrue(flags["live_composition_relation_review"])
        self.assertTrue(flags["live_semantic_oracle_validation"])
        self.assertTrue(flags["live_activation_validation"])
        self.assertTrue(flags["live_translator_replay"])

    def test_cli_no_compositional_profile_gates_compositional_flags(self):
        from scripts import adversarial_rule_run as cli

        args = cli.build_parser().parse_args([
            "output/runs/demo",
            "--bank",
            "bank.json",
            "--output-bank",
            "updated.json",
            "--report",
            "report.json",
            "--experiment-profile",
            "no-compositional",
            "--run-compositional-challenges",
            "--live-composition-relation-review",
        ])

        flags = cli.adversarial_learning_flags(args)

        self.assertTrue(flags["admit_trace_candidates"])
        self.assertTrue(flags["run_synthetic_challenges"])
        self.assertTrue(flags["live_structured_agents"])
        self.assertTrue(flags["live_trace_synthesis"])
        self.assertTrue(flags["live_trace_contradiction_review"])
        self.assertFalse(flags["run_compositional_challenges"])
        self.assertFalse(flags["admit_composition_failure_candidates"])
        self.assertFalse(flags["live_composition_relation_review"])

    def test_cli_no_single_profile_gates_explicit_adversarial_flags(self):
        from scripts import adversarial_rule_run as cli

        args = cli.build_parser().parse_args([
            "output/runs/demo",
            "--bank",
            "bank.json",
            "--output-bank",
            "updated.json",
            "--report",
            "report.json",
            "--experiment-profile",
            "no-single",
            "--admit-trace-candidates",
            "--run-synthetic-challenges",
            "--live-structured-agents",
        ])

        flags = cli.adversarial_learning_flags(args)

        self.assertFalse(any(flags.values()))

    def test_cli_parser_accepts_composition_failure_candidate_flag(self):
        from scripts.adversarial_rule_run import build_parser

        args = build_parser().parse_args([
            "output/runs/demo",
            "--bank",
            "bank.json",
            "--output-bank",
            "updated.json",
            "--report",
            "report.json",
            "--run-compositional-challenges",
            "--admit-composition-failure-candidates",
        ])

        self.assertTrue(args.admit_composition_failure_candidates)

    def test_cli_parser_accepts_live_composition_relation_review(self):
        from scripts.adversarial_rule_run import build_parser

        args = build_parser().parse_args([
            "output/runs/demo",
            "--bank",
            "bank.json",
            "--output-bank",
            "updated.json",
            "--report",
            "report.json",
            "--run-compositional-challenges",
            "--live-composition-relation-review",
            "--api-key",
            "secret",
            "--model",
            "model-x",
        ])

        self.assertTrue(args.live_composition_relation_review)

    def test_cli_parser_accepts_structured_agent_response_files(self):
        from scripts.adversarial_rule_run import build_parser

        args = build_parser().parse_args([
            "output/runs/demo",
            "--bank",
            "bank.json",
            "--output-bank",
            "updated.json",
            "--report",
            "report.json",
            "--structured-challenger-responses",
            "challenger.json",
            "--structured-refiner-responses",
            "refiner.json",
        ])

        self.assertEqual(args.structured_challenger_responses, "challenger.json")
        self.assertEqual(args.structured_refiner_responses, "refiner.json")

    def test_cli_parser_accepts_live_structured_agents(self):
        from scripts.adversarial_rule_run import build_parser

        args = build_parser().parse_args([
            "output/runs/demo",
            "--bank",
            "bank.json",
            "--output-bank",
            "updated.json",
            "--report",
            "report.json",
            "--live-structured-agents",
            "--api-key",
            "secret",
            "--model",
            "model-x",
            "--llm-protocol",
            "openai-responses",
        ])

        self.assertTrue(args.live_structured_agents)
        self.assertEqual(args.api_key, "secret")
        self.assertEqual(args.model, "model-x")

    def test_cli_builds_live_structured_agents_from_configured_llm(self):
        from scripts import adversarial_rule_run as cli

        args = cli.build_parser().parse_args([
            "output/runs/demo",
            "--bank",
            "bank.json",
            "--output-bank",
            "updated.json",
            "--report",
            "report.json",
            "--live-structured-agents",
            "--api-key",
            "secret",
            "--model",
            "model-x",
            "--llm-protocol",
            "openai-responses",
        ])
        fake_agent = object()

        with mock.patch.object(cli, "create_agent", return_value=fake_agent) as create:
            challenger, refiner = cli._structured_agents_from_args(args)

        self.assertIsInstance(challenger, LLMStructuredChallengerAgent)
        self.assertIsInstance(refiner, LLMStructuredRefinerAgent)
        self.assertIs(challenger.agent, fake_agent)
        self.assertIs(refiner.agent, fake_agent)
        create.assert_called_once()
        config = create.call_args.args[0]
        self.assertEqual(config["api_key"], "secret")
        self.assertEqual(config["model_name"], "model-x")
        self.assertEqual(config["protocol"], "openai-responses")

    def test_cli_builds_live_composition_relation_reviewer_from_configured_llm(self):
        from scripts import adversarial_rule_run as cli

        args = cli.build_parser().parse_args([
            "output/runs/demo",
            "--bank",
            "bank.json",
            "--output-bank",
            "updated.json",
            "--report",
            "report.json",
            "--live-composition-relation-review",
            "--api-key",
            "secret",
            "--model",
            "model-x",
        ])
        fake_agent = object()

        with mock.patch.object(cli, "create_agent", return_value=fake_agent) as create:
            reviewer = cli._composition_relation_reviewer_from_args(args)

        self.assertIsInstance(reviewer, LLMCompositionRelationReviewer)
        self.assertIs(reviewer.agent, fake_agent)
        create.assert_called_once()
        self.assertEqual(create.call_args.args[0]["model_name"], "model-x")

    def test_cli_parser_accepts_live_semantic_oracle_validation(self):
        from scripts.adversarial_rule_run import build_parser

        args = build_parser().parse_args([
            "output/runs/demo",
            "--bank",
            "bank.json",
            "--output-bank",
            "updated.json",
            "--report",
            "report.json",
            "--live-semantic-oracle-validation",
            "--api-key",
            "secret",
            "--model",
            "model-x",
        ])

        self.assertTrue(args.live_semantic_oracle_validation)

    def test_cli_builds_live_semantic_oracle_validator_from_configured_llm(self):
        from scripts import adversarial_rule_run as cli

        args = cli.build_parser().parse_args([
            "output/runs/demo",
            "--bank",
            "bank.json",
            "--output-bank",
            "updated.json",
            "--report",
            "report.json",
            "--live-semantic-oracle-validation",
            "--api-key",
            "secret",
            "--model",
            "model-x",
        ])
        fake_agent = object()

        with mock.patch.object(cli, "create_agent", return_value=fake_agent) as create:
            validator = cli._oracle_validator_from_args(args)

        self.assertIsInstance(validator, LLMSemanticOracleValidator)
        self.assertIs(validator.agent, fake_agent)
        create.assert_called_once()
        self.assertEqual(create.call_args.args[0]["model_name"], "model-x")

    def test_cli_parser_accepts_live_activation_validation(self):
        from scripts.adversarial_rule_run import build_parser

        args = build_parser().parse_args([
            "output/runs/demo",
            "--bank",
            "bank.json",
            "--output-bank",
            "updated.json",
            "--report",
            "report.json",
            "--live-activation-validation",
            "--api-key",
            "secret",
            "--model",
            "model-x",
        ])

        self.assertTrue(args.live_activation_validation)

    def test_cli_parser_accepts_live_translator_replay(self):
        from scripts.adversarial_rule_run import build_parser

        args = build_parser().parse_args([
            "output/runs/demo",
            "--bank",
            "bank.json",
            "--output-bank",
            "updated.json",
            "--report",
            "report.json",
            "--live-translator-replay",
            "--api-key",
            "secret",
            "--model",
            "model-x",
        ])

        self.assertTrue(args.live_translator_replay)

    def test_cli_builds_live_translator_replay_from_configured_llm(self):
        from scripts import adversarial_rule_run as cli

        args = cli.build_parser().parse_args([
            "output/runs/demo",
            "--bank",
            "bank.json",
            "--output-bank",
            "updated.json",
            "--report",
            "report.json",
            "--live-translator-replay",
            "--api-key",
            "secret",
            "--model",
            "model-x",
        ])

        class FakeAgent:
            def __init__(self):
                self.calls = []

            def translate(self, system, prompt):
                self.calls.append({"system": system, "prompt": prompt})
                return SimpleNamespace(
                    code="```rust\npub fn translate_challenge() -> i32 { 7 }\n```",
                    raw_text="```rust\npub fn translate_challenge() -> i32 { 7 }\n```",
                    elapsed_seconds=0.01,
                    usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                )

        fake_agent = FakeAgent()
        with mock.patch.object(cli, "create_agent", return_value=fake_agent) as create:
            replay_translate = cli._replay_translate_fn_from_args(args)

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
        )

        self.assertIsNotNone(replay_translate)
        self.assertEqual(
            replay_translate(challenge, rule),
            "pub fn translate_challenge() -> i32 { 7 }",
        )
        create.assert_called_once()
        self.assertEqual(create.call_args.args[0]["model_name"], "model-x")
        self.assertIn("Preserve value", fake_agent.calls[0]["prompt"])

    def test_cli_builds_live_activation_validator_from_configured_llm(self):
        from scripts import adversarial_rule_run as cli

        args = cli.build_parser().parse_args([
            "output/runs/demo",
            "--bank",
            "bank.json",
            "--output-bank",
            "updated.json",
            "--report",
            "report.json",
            "--live-activation-validation",
            "--api-key",
            "secret",
            "--model",
            "model-x",
        ])
        fake_agent = object()

        with mock.patch.object(cli, "create_agent", return_value=fake_agent) as create:
            validator = cli._activation_validator_from_args(args)

        self.assertIsInstance(validator, LLMActivationValidator)
        self.assertIs(validator.agent, fake_agent)
        create.assert_called_once()
        self.assertEqual(create.call_args.args[0]["model_name"], "model-x")

    def test_cli_parser_accepts_live_trace_synthesis(self):
        from scripts.adversarial_rule_run import build_parser

        args = build_parser().parse_args([
            "output/runs/demo",
            "--bank",
            "bank.json",
            "--output-bank",
            "updated.json",
            "--report",
            "report.json",
            "--live-trace-synthesis",
            "--api-key",
            "secret",
            "--model",
            "model-x",
        ])

        self.assertTrue(args.live_trace_synthesis)

    def test_cli_parser_accepts_live_trace_contradiction_review(self):
        from scripts.adversarial_rule_run import build_parser

        args = build_parser().parse_args([
            "output/runs/demo",
            "--bank",
            "bank.json",
            "--output-bank",
            "updated.json",
            "--report",
            "report.json",
            "--live-trace-contradiction-review",
            "--api-key",
            "secret",
            "--model",
            "model-x",
        ])

        self.assertTrue(args.live_trace_contradiction_review)

    def test_cli_builds_live_trace_synthesizer_from_configured_llm(self):
        from scripts import adversarial_rule_run as cli

        args = cli.build_parser().parse_args([
            "output/runs/demo",
            "--bank",
            "bank.json",
            "--output-bank",
            "updated.json",
            "--report",
            "report.json",
            "--live-trace-synthesis",
            "--api-key",
            "secret",
            "--model",
            "model-x",
        ])
        fake_agent = object()

        with mock.patch.object(cli, "create_agent", return_value=fake_agent) as create:
            synthesizer = cli._trace_synthesizer_from_args(args)

        self.assertIsInstance(synthesizer, LLMTraceRuleSynthesizer)
        self.assertIs(synthesizer.agent, fake_agent)
        create.assert_called_once()
        self.assertEqual(create.call_args.args[0]["model_name"], "model-x")

    def test_cli_builds_live_trace_contradiction_reviewer_from_configured_llm(self):
        from scripts import adversarial_rule_run as cli

        args = cli.build_parser().parse_args([
            "output/runs/demo",
            "--bank",
            "bank.json",
            "--output-bank",
            "updated.json",
            "--report",
            "report.json",
            "--live-trace-contradiction-review",
            "--api-key",
            "secret",
            "--model",
            "model-x",
        ])
        fake_agent = object()

        with mock.patch.object(cli, "create_agent", return_value=fake_agent) as create:
            reviewer = cli._trace_contradiction_reviewer_from_args(args)

        self.assertIsInstance(reviewer, LLMTraceContradictionReviewer)
        self.assertIs(reviewer.agent, fake_agent)
        create.assert_called_once()
        self.assertEqual(create.call_args.args[0]["model_name"], "model-x")

    def test_cli_uses_structured_refiner_response_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run"
            project = run_root / "demo"
            project.mkdir(parents=True)
            (project / "translation_report.json").write_text(json.dumps({"success": False}), encoding="utf-8")
            (project / "test_fix_history.json").write_text(json.dumps([
                {
                    "selected_rules": [{"id": "fixed-c-string"}],
                    "test_output": "assertion failed: zero padding after NUL",
                    "post_patch_test_output": "ok",
                    "post_patch_test_stats": {"failed": 0, "passed": 1},
                    "test_stats": {"failed": 1, "passed": 0},
                    "test_improved": True,
                }
            ]), encoding="utf-8")
            bank_path = Path(tmpdir) / "bank.json"
            out_bank = Path(tmpdir) / "updated.json"
            report_path = Path(tmpdir) / "report.json"
            refiner_responses = Path(tmpdir) / "refiner.json"
            RuleBank([
                RuleRecord(
                    id="fixed-c-string",
                    title="Fixed C String",
                    trigger="fixed char array",
                    goal="Expose visible length",
                    constraint="Trim at NUL",
                    risk="padding may leak",
                    tags=("strings",),
                    keywords=("NUL",),
                )
            ]).save(bank_path)
            refiner_responses.write_text(json.dumps([
                {
                    "rule_id": "fixed-c-string",
                    "field": "risk",
                    "new_text": "padding may leak; zero padding after NUL must be handled",
                    "rationale": "trace replay improved after the risk update",
                    "failure_kind": "missing_risk",
                    "actionable": True,
                }
            ]), encoding="utf-8")

            from scripts.adversarial_rule_run import main

            with mock.patch.object(sys, "argv", [
                "adversarial_rule_run.py",
                str(run_root),
                "--bank",
                str(bank_path),
                "--output-bank",
                str(out_bank),
                "--report",
                str(report_path),
                "--structured-refiner-responses",
                str(refiner_responses),
            ]):
                main()

            updated = RuleBank.load(out_bank)
            self.assertIn("must be handled", updated.records[0].risk)

    def test_cli_writes_updated_bank_and_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run"
            project = run_root / "demo"
            project.mkdir(parents=True)
            (project / "translation_report.json").write_text(json.dumps({"success": False}), encoding="utf-8")
            (project / "test_fix_history.json").write_text(json.dumps([
                {
                    "selected_rules": [{"id": "fixed-c-string"}],
                    "test_output": "assertion failed: zero padding after NUL",
                    "post_patch_test_output": "ok",
                    "post_patch_test_stats": {"failed": 0, "passed": 1},
                    "test_stats": {"failed": 1, "passed": 0},
                    "test_improved": True,
                }
            ]), encoding="utf-8")
            bank_path = Path(tmpdir) / "bank.json"
            out_bank = Path(tmpdir) / "updated.json"
            report_path = Path(tmpdir) / "report.json"
            RuleBank([
                RuleRecord(
                    id="fixed-c-string",
                    title="Fixed C String",
                    trigger="fixed char array",
                    goal="Expose visible length",
                    constraint="Trim at NUL",
                    risk="padding may leak",
                    tags=("strings",),
                    keywords=("NUL",),
                )
            ]).save(bank_path)

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/adversarial_rule_run.py",
                    str(run_root),
                    "--bank",
                    str(bank_path),
                    "--output-bank",
                    str(out_bank),
                    "--report",
                    str(report_path),
                ],
                cwd=Path(__file__).resolve().parent.parent,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("accepted_updates=1", result.stdout)
            self.assertIn("relation_updates=0", result.stdout)
            self.assertTrue(out_bank.exists())
            self.assertTrue(report_path.exists())


if __name__ == "__main__":
    unittest.main()
