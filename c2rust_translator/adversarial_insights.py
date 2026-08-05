from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import importlib.util
import re
import subprocess
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Sequence

COMPOSITION_MAX_INSIGHTS = 3
COMPOSITION_SUPPORT_WEIGHT = 1.0
COMPOSITION_CONFLICT_WEIGHT = 1.0
COMPOSITION_CONFLICT_THRESHOLD = 0.7


@dataclass(frozen=True)
class InsightRecord:
    id: str
    title: str
    trigger: str
    goal: str
    constraint: str
    risk: str
    guidance: str = ""
    tags: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    version: int = 1
    source: str = "json"
    parents: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "trigger": self.trigger,
            "goal": self.goal,
            "constraint": self.constraint,
            "risk": self.risk,
            "guidance": self.guidance,
            "tags": list(self.tags),
            "keywords": list(self.keywords),
            "version": self.version,
            "source": self.source,
            "parents": list(self.parents),
        }

    def to_prompt_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "trigger": self.trigger,
            "goal": self.goal,
            "constraint": self.constraint,
            "risk": self.risk,
            "guidance": self.guidance,
            "tags": list(self.tags),
            "version": self.version,
            "source": self.source,
            "parents": list(self.parents),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "InsightRecord":
        return cls(
            id=str(raw.get("id", "")),
            title=str(raw.get("title", "")),
            trigger=str(raw.get("trigger", "")),
            goal=str(raw.get("goal", "")),
            constraint=str(raw.get("constraint", "")),
            risk=str(raw.get("risk", "")),
            guidance=str(raw.get("guidance", "")),
            tags=tuple(str(item) for item in raw.get("tags", ())),
            keywords=tuple(str(item) for item in raw.get("keywords", ())),
            version=int(raw.get("version", 1) or 1),
            source=str(raw.get("source", "json")),
            parents=tuple(str(item) for item in raw.get("parents", ())),
        )


@dataclass(frozen=True)
class InsightRelationRecord:
    first_id: str
    second_id: str
    relation: str
    rationale: str
    evidence: str
    version: int = 1

    def to_dict(self) -> dict:
        return {
            "first_id": self.first_id,
            "second_id": self.second_id,
            "relation": self.relation,
            "rationale": self.rationale,
            "evidence": self.evidence,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "InsightRelationRecord":
        return cls(
            first_id=str(raw.get("first_id", "")),
            second_id=str(raw.get("second_id", "")),
            relation=str(raw.get("relation", "")),
            rationale=str(raw.get("rationale", "")),
            evidence=str(raw.get("evidence", "")),
            version=int(raw.get("version", 1) or 1),
        )


@dataclass
class InsightBank:
    records: list[InsightRecord] = field(default_factory=list)
    relations: list[InsightRelationRecord] = field(default_factory=list)

    def save(self, path: str | Path) -> Path:
        bank_path = Path(path)
        bank_path.parent.mkdir(parents=True, exist_ok=True)
        bank_path.write_text(
            json.dumps(
                {
                    "insights": [record.to_dict() for record in self.records],
                    "relations": [relation.to_dict() for relation in self.relations],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return bank_path

    @classmethod
    def load(cls, path: str | Path) -> "InsightBank":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            [InsightRecord.from_dict(item) for item in raw.get("insights", [])],
            [InsightRelationRecord.from_dict(item) for item in raw.get("relations", [])],
        )

    @classmethod
    def from_legacy(cls, insights: Iterable[object]) -> "InsightBank":
        records: list[InsightRecord] = []
        for insight in insights:
            item = insight.to_prompt_dict() if hasattr(insight, "to_prompt_dict") else dict(insight)
            records.append(
                InsightRecord(
                    id=str(item.get("id", "")),
                    title=str(item.get("title", "")),
                    trigger=str(item.get("trigger", "")),
                    goal=str(item.get("goal", "")),
                    constraint=str(item.get("constraint", "")),
                    risk=str(item.get("risk", "")),
                    guidance=str(item.get("guidance", "")),
                    tags=tuple(str(tag) for tag in item.get("tags", ())),
                    keywords=tuple(str(keyword) for keyword in getattr(insight, "keywords", ())),
                    source="legacy",
                )
            )
        return cls(records)


@dataclass(frozen=True)
class RetrievalStatus:
    backend: str
    reason: str


class RetrievalEngine:
    def __init__(
        self,
        bank: InsightBank,
        *,
        prefer_embeddings: bool = False,
        index_cache_path: str | Path | None = None,
        embedding_model=None,
        faiss_module=None,
        find_spec: Callable[[str], object | None] | None = None,
    ):
        self.bank = bank
        self.index_cache_path = Path(index_cache_path) if index_cache_path else None
        self.cache_status = "disabled"
        self.embedding_model = embedding_model
        self.faiss_module = faiss_module
        self.find_spec = find_spec or importlib.util.find_spec
        self.embedding_index = None
        self.embedding_records: list[InsightRecord] = []
        self.status = self._configure_backend(prefer_embeddings)
        if self.status.backend == "faiss-bge":
            self._build_embedding_index()

    def _configure_backend(self, prefer_embeddings: bool) -> RetrievalStatus:
        if not prefer_embeddings:
            return RetrievalStatus("keyword", "keyword backend requested")
        if self.embedding_model is not None and self.faiss_module is not None:
            return RetrievalStatus("faiss-bge", "injected FAISS/BGE-compatible backend")
        if not self.find_spec("faiss") or not self.find_spec("sentence_transformers"):
            return RetrievalStatus("keyword", "embedding backend unavailable; using keyword fallback")
        try:
            import faiss  # type: ignore
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional local packages
            return RetrievalStatus("keyword", f"embedding backend import failed: {exc}")
        self.faiss_module = faiss
        self.embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        return RetrievalStatus("faiss-bge", "FAISS and sentence-transformers available")

    def _build_embedding_index(self) -> None:
        if not self.bank.records:
            return
        if self._load_embedding_index():
            return
        try:
            texts = [_record_retrieval_text(record) for record in self.bank.records]
            vectors = self._encode(texts)
            if not vectors:
                self.status = RetrievalStatus("keyword", "embedding backend produced no vectors")
                return
            dimension = len(vectors[0])
            index = self.faiss_module.IndexFlatIP(dimension)
            index.add(_float_matrix(vectors))
        except Exception as exc:
            self.embedding_index = None
            self.embedding_records = []
            self.status = RetrievalStatus("keyword", f"embedding index build failed: {exc}")
            return
        self.embedding_index = index
        self.embedding_records = list(self.bank.records)
        self._write_embedding_index()

    def _load_embedding_index(self) -> bool:
        if self.index_cache_path is None:
            self.cache_status = "disabled"
            return False
        metadata_path = _embedding_index_metadata_path(self.index_cache_path)
        if not self.index_cache_path.exists() or not metadata_path.exists():
            self.cache_status = "miss"
            return False
        if not hasattr(self.faiss_module, "read_index"):
            self.cache_status = "unsupported"
            return False
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("bank_fingerprint") != _bank_fingerprint(self.bank):
                self.cache_status = "stale"
                return False
            self.embedding_index = self.faiss_module.read_index(str(self.index_cache_path))
        except Exception as exc:
            self.cache_status = f"load failed: {exc}"
            return False
        self.embedding_records = list(self.bank.records)
        self.cache_status = "loaded"
        return True

    def _write_embedding_index(self) -> None:
        if self.index_cache_path is None:
            self.cache_status = "disabled"
            return
        if not hasattr(self.faiss_module, "write_index"):
            self.cache_status = "unsupported"
            return
        try:
            self.index_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.faiss_module.write_index(self.embedding_index, str(self.index_cache_path))
            _embedding_index_metadata_path(self.index_cache_path).write_text(
                json.dumps(
                    {
                        "bank_fingerprint": _bank_fingerprint(self.bank),
                        "record_ids": [record.id for record in self.bank.records],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            self.cache_status = f"write failed: {exc}"
            return
        self.cache_status = "written"

    def select(self, query: str, *, limit: int = 5) -> list[InsightRecord]:
        if limit <= 0:
            return []
        if self.status.backend == "faiss-bge" and self.embedding_index is not None:
            try:
                return self._select_embedding(query, limit=limit)
            except Exception:
                return self._select_keyword(query, limit=limit)
        return self._select_keyword(query, limit=limit)

    def _select_embedding(self, query: str, *, limit: int) -> list[InsightRecord]:
        query_vectors = self._encode([query])
        if not query_vectors:
            return []
        scores, indexes = self.embedding_index.search(_float_matrix(query_vectors), limit)
        selected: list[InsightRecord] = []
        for score, index in zip(list(scores[0]), list(indexes[0])):
            if index < 0 or index >= len(self.embedding_records) or float(score) <= 0.0:
                continue
            selected.append(self.embedding_records[int(index)])
        return selected

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.embedding_model.encode(texts, normalize_embeddings=True)
        if hasattr(vectors, "tolist"):
            vectors = vectors.tolist()
        return [list(vector) for vector in vectors]

    def _select_keyword(self, query: str, *, limit: int = 5) -> list[InsightRecord]:
        scored: list[tuple[int, int, InsightRecord]] = []
        normalized = _normalize(query)
        query_tokens = set(_tokens(normalized))
        for order, record in enumerate(self.bank.records):
            haystack = _normalize(_record_retrieval_text(record))
            score = len(query_tokens & set(_tokens(haystack)))
            for keyword in record.keywords:
                normalized_keyword = _normalize(keyword)
                if normalized_keyword and normalized_keyword in normalized:
                    score += 5
            for tag in record.tags:
                normalized_tag = _normalize(tag)
                if normalized_tag and normalized_tag in normalized:
                    score += 3
            if score > 0:
                scored.append((score, -order, record))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [record for _score, _order, record in scored[: max(limit, 0)]]


def _record_retrieval_text(record: InsightRecord) -> str:
    return " ".join(
        [
            record.id,
            record.title,
            record.trigger,
            record.goal,
            record.constraint,
            record.risk,
            record.guidance,
            " ".join(record.tags),
            " ".join(record.keywords),
        ]
    )


def _bank_fingerprint(bank: InsightBank) -> str:
    payload = json.dumps([record.to_dict() for record in bank.records], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _embedding_index_metadata_path(index_path: Path) -> Path:
    if index_path.suffix:
        return index_path.with_suffix(index_path.suffix + ".meta.json")
    return Path(str(index_path) + ".meta.json")


def _float_matrix(vectors):
    try:
        import numpy as np  # type: ignore

        return np.asarray(vectors, dtype="float32")
    except Exception:  # pragma: no cover - only used when numpy is absent
        return vectors


def _normalize(text: str) -> str:
    return text.lower().replace("_", " ").replace("-", " ")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass(frozen=True)
class ExecutionEvidence:
    compiles: bool
    tests_pass: bool
    output: str = ""

    @property
    def tool_checked(self) -> bool:
        return self.compiles or bool(self.output)

    @property
    def score(self) -> int:
        return int(self.tool_checked) + int(self.compiles) + int(self.tests_pass)


@dataclass(frozen=True)
class ChallengeRecord:
    insight_id: str
    c_program: str
    rust_interface: str
    harness: str
    oracle: str
    evidence: ExecutionEvidence
    replay_evidence: ExecutionEvidence | None = None
    activation_predicate: str = ""
    trace_bundle: dict[str, object] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        payload = "\n".join(
            [
                self.insight_id,
                self.c_program,
                self.rust_interface,
                self.harness,
                self.oracle,
                self.activation_predicate,
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def activation_text(self) -> str:
        return "\n".join(
            [
                self.c_program,
                self.rust_interface,
                self.harness,
                self.oracle,
                self.activation_predicate,
                self.evidence.output,
            ]
        )


class SyntheticChallengeGenerator:
    def generate(self, insight: InsightRecord) -> ChallengeRecord:
        if _is_string_boundary_insight(insight):
            return self._string_boundary_challenge(insight)
        if _is_pointer_length_insight(insight):
            return self._pointer_length_challenge(insight)
        if _is_callback_user_data_state_insight(insight):
            return self._callback_user_data_state_challenge(insight)
        if _is_parser_cursor_state_insight(insight):
            return self._parser_cursor_state_challenge(insight)
        if _is_prng_state_insight(insight):
            return self._prng_sequence_challenge(insight)
        return self._default_challenge(insight)

    def _string_boundary_challenge(self, insight: InsightRecord) -> ChallengeRecord:
        c_program = "\n".join(
            [
                "/*",
                f"Insight: {insight.id}",
                f"Trigger: {insight.trigger}",
                f"Constraint: {insight.constraint}",
                f"Risk: {insight.risk}",
                "*/",
                "#include <stddef.h>",
                "",
                "const unsigned char *challenge_bytes(size_t *len) {",
                "    static const unsigned char challenge_buffer[8] = {'a', 'b', 'c', '\\0', 'x', 'y', 'z', '\\0'};",
                "    *len = sizeof(challenge_buffer);",
                "    return challenge_buffer;",
                "}",
            ]
        )
        rust_interface = "pub fn translate_challenge(input: &[u8]) -> String;"
        harness = "\n".join(
            [
                "#[test]",
                "fn challenge_stops_at_first_nul() {",
                "    let input = b\"abc\\0xyz\\0\";",
                "    assert_eq!(translate_challenge(input), \"abc\");",
                "}",
            ]
        )
        oracle = (
            "Rust-visible string output must stop at the first NUL byte; "
            "bytes after the first NUL are storage padding and must not affect equality."
        )
        return ChallengeRecord(
            insight_id=insight.id,
            c_program=c_program,
            rust_interface=rust_interface,
            harness=harness,
            oracle=oracle,
            evidence=ExecutionEvidence(compiles=False, tests_pass=False),
            trace_bundle={"reference_source": _synthetic_reference_source("string_boundary")},
        )

    def _pointer_length_challenge(self, insight: InsightRecord) -> ChallengeRecord:
        c_program = "\n".join(
            [
                "/*",
                f"Insight: {insight.id}",
                f"Trigger: {insight.trigger}",
                f"Constraint: {insight.constraint}",
                f"Risk: {insight.risk}",
                "*/",
                "#include <stddef.h>",
                "",
                "const unsigned char *challenge_buffer(size_t *len) {",
                "    static const unsigned char challenge_bytes[5] = {1, 2, 0, 3, 4};",
                "    *len = sizeof(challenge_bytes);",
                "    return challenge_bytes;",
                "}",
            ]
        )
        rust_interface = "pub fn translate_challenge(input: &[u8], len: usize) -> Vec<u8>;"
        harness = "\n".join(
            [
                "#[test]",
                "fn challenge_preserves_explicit_length() {",
                "    let input = &[1, 2, 0, 3, 4];",
                "    assert_eq!(translate_challenge(input, 5), vec![1, 2, 0, 3, 4]);",
                "}",
            ]
        )
        oracle = (
            f"{insight.goal} The translated result must preserve the explicit buffer length, "
            "including interior zero bytes and trailing bytes."
        )
        return ChallengeRecord(
            insight_id=insight.id,
            c_program=c_program,
            rust_interface=rust_interface,
            harness=harness,
            oracle=oracle,
            evidence=ExecutionEvidence(compiles=False, tests_pass=False),
            trace_bundle={"reference_source": _synthetic_reference_source("pointer_length")},
        )

    def _parser_cursor_state_challenge(self, insight: InsightRecord) -> ChallengeRecord:
        c_program = "\n".join(
            [
                "/*",
                f"Insight: {insight.id}",
                f"Trigger: {insight.trigger}",
                f"Constraint: {insight.constraint}",
                f"Risk: {insight.risk}",
                "*/",
                "#include <stddef.h>",
                "",
                "typedef struct ParserState {",
                "    const unsigned char *input;",
                "    size_t cursor;",
                "    size_t consumed;",
                "} ParserState;",
                "",
                "static void parse_key(ParserState *state) {",
                "    while (state->input[state->cursor] && state->input[state->cursor] != ':') {",
                "        state->cursor++;",
                "    }",
                "    state->consumed = state->cursor;",
                "}",
                "",
                "static void consume_colon(ParserState *state) {",
                "    if (state->input[state->cursor] == ':') {",
                "        state->cursor++;",
                "        state->consumed = state->cursor;",
                "    }",
                "}",
                "",
                "ParserState challenge_parse(const unsigned char *input) {",
                "    ParserState state = {input, 0, 0};",
                "    parse_key(&state);",
                "    consume_colon(&state);",
                "    return state;",
                "}",
            ]
        )
        rust_interface = "pub fn translate_challenge(input: &[u8]) -> (usize, usize);"
        harness = "\n".join(
            [
                "#[test]",
                "fn challenge_preserves_shared_parser_cursor() {",
                "    let input = b\"ab:cd\\0\";",
                "    assert_eq!(translate_challenge(input), (3, 3));",
                "}",
            ]
        )
        oracle = (
            f"{insight.goal} The translated parser must keep one shared parser cursor "
            "and consumed byte offset across parse and consume helper steps."
        )
        return ChallengeRecord(
            insight_id=insight.id,
            c_program=c_program,
            rust_interface=rust_interface,
            harness=harness,
            oracle=oracle,
            evidence=ExecutionEvidence(compiles=False, tests_pass=False),
            trace_bundle={"reference_source": _synthetic_reference_source("parser_cursor_state")},
        )

    def _prng_sequence_challenge(self, insight: InsightRecord) -> ChallengeRecord:
        c_program = "\n".join(
            [
                "/*",
                f"Insight: {insight.id}",
                f"Trigger: {insight.trigger}",
                f"Constraint: {insight.constraint}",
                f"Risk: {insight.risk}",
                "*/",
                "",
                "typedef struct PrngPair {",
                "    unsigned int first;",
                "    unsigned int second;",
                "} PrngPair;",
                "",
                "static unsigned int challenge_next(unsigned int *state) {",
                "    *state = ((*state * 1103515245u) + 12345u) & 0x7fffffffu;",
                "    return *state;",
                "}",
                "",
                "PrngPair challenge_prng_sequence(unsigned int seed) {",
                "    unsigned int state = seed;",
                "    PrngPair pair = {challenge_next(&state), challenge_next(&state)};",
                "    return pair;",
                "}",
            ]
        )
        rust_interface = "pub fn translate_challenge(seed: u32) -> (u32, u32);"
        harness = "\n".join(
            [
                "#[test]",
                "fn challenge_preserves_prng_sequence_state() {",
                "    assert_eq!(translate_challenge(1), (1103527590, 377401575));",
                "}",
            ]
        )
        oracle = (
            f"{insight.goal} The translated code must preserve deterministic PRNG state "
            "advancement across consecutive calls from the same seed."
        )
        return ChallengeRecord(
            insight_id=insight.id,
            c_program=c_program,
            rust_interface=rust_interface,
            harness=harness,
            oracle=oracle,
            evidence=ExecutionEvidence(compiles=False, tests_pass=False),
            trace_bundle={"reference_source": _synthetic_reference_source("prng_sequence")},
        )

    def _callback_user_data_state_challenge(self, insight: InsightRecord) -> ChallengeRecord:
        c_program = "\n".join(
            [
                "/*",
                f"Insight: {insight.id}",
                f"Trigger: {insight.trigger}",
                f"Constraint: {insight.constraint}",
                f"Risk: {insight.risk}",
                "*/",
                "#include <stddef.h>",
                "",
                "typedef struct CallbackState {",
                "    int sum;",
                "    int calls;",
                "} CallbackState;",
                "",
                "typedef void (*ChallengeCallback)(int value, void *user_data);",
                "",
                "static void challenge_accumulate(int value, void *user_data) {",
                "    CallbackState *state = (CallbackState *)user_data;",
                "    state->sum += value;",
                "    state->calls++;",
                "}",
                "",
                "CallbackState challenge_visit_with_callback(",
                "    const int *values,",
                "    size_t len,",
                "    ChallengeCallback callback,",
                "    void *user_data",
                ") {",
                "    for (size_t i = 0; i < len; i++) {",
                "        callback(values[i], user_data);",
                "    }",
                "    return *(CallbackState *)user_data;",
                "}",
                "",
                "CallbackState challenge_run_callback(void) {",
                "    const int values[3] = {2, 3, 5};",
                "    CallbackState state = {0, 0};",
                "    return challenge_visit_with_callback(values, 3, challenge_accumulate, &state);",
                "}",
            ]
        )
        rust_interface = "pub fn translate_challenge(values: &[i32]) -> i32;"
        harness = "\n".join(
            [
                "#[test]",
                "fn challenge_preserves_callback_user_data_state() {",
                "    assert_eq!(translate_challenge(&[2, 3, 5]), 10);",
                "}",
            ]
        )
        oracle = (
            f"{insight.goal} The translated traversal must preserve callback user_data state "
            "across each invocation instead of reinitializing counters or accumulators inside "
            "the callback body."
        )
        return ChallengeRecord(
            insight_id=insight.id,
            c_program=c_program,
            rust_interface=rust_interface,
            harness=harness,
            oracle=oracle,
            evidence=ExecutionEvidence(compiles=False, tests_pass=False),
            trace_bundle={"reference_source": _synthetic_reference_source("callback_user_data_state")},
        )

    def _default_challenge(self, insight: InsightRecord) -> ChallengeRecord:
        c_program = "\n".join(
            [
                "/*",
                f"Insight: {insight.id}",
                f"Trigger: {insight.trigger}",
                f"Goal: {insight.goal}",
                f"Constraint: {insight.constraint}",
                f"Risk: {insight.risk}",
                "*/",
                "typedef struct ChallengeState {",
                "    int count;",
                "    int capacity;",
                "} ChallengeState;",
                "",
                "ChallengeState challenge_state(void) {",
                "    ChallengeState state = {2, 8};",
                "    return state;",
                "}",
            ]
        )
        rust_interface = "pub fn translate_challenge() -> (i32, i32);"
        harness = "\n".join(
            [
                "#[test]",
                "fn challenge_preserves_observable_contract() {",
                "    assert_eq!(translate_challenge(), (2, 8));",
                "}",
            ]
        )
        oracle = (
            f"{insight.goal} The translated result must preserve the observable "
            "count/capacity pair from the C challenge state."
        )
        return ChallengeRecord(
            insight_id=insight.id,
            c_program=c_program,
            rust_interface=rust_interface,
            harness=harness,
            oracle=oracle,
            evidence=ExecutionEvidence(compiles=False, tests_pass=False),
            trace_bundle={"reference_source": _synthetic_reference_source("count_capacity")},
        )


def _synthetic_reference_source(kind: str) -> str:
    sources = {
        "string_boundary": "\n".join(
            [
                "pub fn translate_challenge(input: &[u8]) -> String {",
                "    let end = input.iter().position(|&byte| byte == 0).unwrap_or(input.len());",
                "    String::from_utf8_lossy(&input[..end]).to_string()",
                "}",
            ]
        ),
        "pointer_length": "\n".join(
            [
                "pub fn translate_challenge(input: &[u8], len: usize) -> Vec<u8> {",
                "    input[..len.min(input.len())].to_vec()",
                "}",
            ]
        ),
        "parser_cursor_state": "\n".join(
            [
                "pub fn translate_challenge(input: &[u8]) -> (usize, usize) {",
                "    let limit = input.iter().position(|&byte| byte == 0).unwrap_or(input.len());",
                "    let cursor = input[..limit]",
                "        .iter()",
                "        .position(|&byte| byte == b':')",
                "        .map(|index| index + 1)",
                "        .unwrap_or(limit);",
                "    (cursor, cursor)",
                "}",
            ]
        ),
        "prng_sequence": "\n".join(
            [
                "pub fn translate_challenge(seed: u32) -> (u32, u32) {",
                "    fn next(state: &mut u32) -> u32 {",
                "        *state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;",
                "        *state",
                "    }",
                "    let mut state = seed;",
                "    (next(&mut state), next(&mut state))",
                "}",
            ]
        ),
        "callback_user_data_state": "\n".join(
            [
                "pub fn translate_challenge(values: &[i32]) -> i32 {",
                "    values.iter().copied().sum()",
                "}",
            ]
        ),
        "count_capacity": "\n".join(
            [
                "pub fn translate_challenge() -> (i32, i32) {",
                "    (2, 8)",
                "}",
            ]
        ),
    }
    return sources[kind]


def _is_string_boundary_insight(insight: InsightRecord) -> bool:
    text = _normalize(
        " ".join(
            [
                insight.trigger,
                insight.goal,
                insight.constraint,
                insight.risk,
                insight.guidance,
                " ".join(insight.tags),
                " ".join(insight.keywords),
            ]
        )
    )
    return any(marker in text for marker in ("nul", "string", "char array", "utf8 lossy"))


def _is_pointer_length_insight(insight: InsightRecord) -> bool:
    text = _normalize_record_text(insight)
    return any(marker in text for marker in ("pointer", "buffer", "slice")) and "length" in text


def _is_callback_user_data_state_insight(insight: InsightRecord) -> bool:
    text = _normalize_record_text(insight)
    callback_terms = ("callback", "handler", "visitor")
    user_data_terms = ("user data", "user_data", "baton", "context pointer")
    return any(term in text for term in callback_terms) and any(term in text for term in user_data_terms)


def _is_parser_cursor_state_insight(insight: InsightRecord) -> bool:
    text = _normalize_record_text(insight)
    parser_terms = ("parser", "parse", "scanner", "token")
    state_terms = ("cursor", "consume", "consumed", "lookahead", "state")
    return any(term in text for term in parser_terms) and any(term in text for term in state_terms)


def _is_prng_state_insight(insight: InsightRecord) -> bool:
    text = _normalize_record_text(insight)
    prng_terms = ("prng", "rand", "random", "lrand48", "drand48", "srand", "seed")
    state_terms = ("state", "sequence", "seed", "deterministic")
    return any(term in text for term in prng_terms) and any(term in text for term in state_terms)


class CompositionalChallengeGenerator:
    def generate(self, insights: Sequence[InsightRecord]) -> ChallengeRecord:
        if not insights:
            raise ValueError("At least one insight is required for a compositional challenge")
        insight_ids = [insight.id for insight in insights]
        c_program = "\n".join(
            [
                "/*",
                "Compositional insight challenge",
                *[
                    f"- {insight.id}: trigger={insight.trigger}; constraint={insight.constraint}; risk={insight.risk}"
                    for insight in insights
                ],
                "*/",
                "#include <stddef.h>",
                "",
                "typedef struct ChallengeComposition {",
                "    unsigned char name[8];",
                "    int count;",
                "    int capacity;",
                "} ChallengeComposition;",
                "",
                "ChallengeComposition challenge_composition(void) {",
                "    ChallengeComposition state = {{'a', 'b', 'c', '\\0', 'x', 'y', 'z', '\\0'}, 2, 8};",
                "    return state;",
                "}",
            ]
        )
        rust_interface = (
            "pub fn challenge_composition(input: &[u8], count: i32, capacity: i32) "
            "-> (String, i32, i32);"
        )
        harness = "\n".join(
            [
                "#[test]",
                "fn challenge_joint_insight_contract() {",
                "    let input = b\"abc\\0xyz\\0\";",
                "    assert_eq!(challenge_composition(input, 2, 8), (\"abc\".to_string(), 2, 8));",
                "}",
            ]
        )
        oracle = (
            "The translation must jointly satisfy the selected insights: "
            + ", ".join(insight_ids)
            + ". String-visible output stops at the first NUL while count and capacity remain observable."
        )
        dependency_probes = _composition_dependency_probes(insights)
        return ChallengeRecord(
            insight_id="composition:" + "+".join(insight_ids),
            c_program=c_program,
            rust_interface=rust_interface,
            harness=harness,
            oracle=oracle,
            evidence=ExecutionEvidence(compiles=False, tests_pass=False),
            trace_bundle={"composition_dependency_probes": dependency_probes},
        )


def _composition_dependency_probes(insights: Sequence[InsightRecord]) -> dict:
    return {
        "reference_source": _composition_reference_source(),
        "mutants": [
            {
                "insight_id": insight.id,
                "kind": _composition_mutant_kind(insight),
                "rust_source": _composition_mutant_source(_composition_mutant_kind(insight)),
            }
            for insight in insights
        ],
    }


def _composition_reference_source() -> str:
    return "\n".join(
        [
            "// reference composition implementation",
            "pub fn challenge_composition(input: &[u8], count: i32, capacity: i32) -> (String, i32, i32) {",
            "    let visible: String = input",
            "        .iter()",
            "        .take_while(|&&byte| byte != 0)",
            "        .map(|&byte| byte as char)",
            "        .collect();",
            "    (visible, count, capacity)",
            "}",
        ]
    )


def _composition_mutant_kind(insight: InsightRecord) -> str:
    text = _normalize_record_text(insight)
    if any(term in text for term in ("nul", "string", "char array", "utf8 lossy")):
        return "string_boundary"
    if any(term in text for term in ("capacity", "count", "layout")):
        return "capacity_contract"
    return "generic_contract"


def _composition_mutant_source(kind: str) -> str:
    if kind == "string_boundary":
        body = [
            "    let visible: String = input.iter().map(|&byte| byte as char).collect();",
            "    (visible, count, capacity)",
        ]
    elif kind == "capacity_contract":
        body = [
            "    let visible: String = input",
            "        .iter()",
            "        .take_while(|&&byte| byte != 0)",
            "        .map(|&byte| byte as char)",
            "        .collect();",
            "    (visible, count, count)",
        ]
    else:
        body = [
            "    let _ = input;",
            "    (String::new(), count, capacity)",
        ]
    return "\n".join(
        [
            f"// composition_mutant:{kind}",
            "pub fn challenge_composition(input: &[u8], count: i32, capacity: i32) -> (String, i32, i32) {",
            *body,
            "}",
        ]
    )


CHALLENGER_OUTPUT_SCHEMA = {
    "insight_id": "string; must match the input insight id",
    "c_program": "string; self-contained C snippet that activates the insight",
    "rust_interface": "string; Rust signature ending with semicolon",
    "harness": "string; Rust test harness with at least one assertion",
    "oracle": "string; expected semantic property checked by the harness",
    "activation_predicate": "optional string; why the challenge activates the insight",
    "rationale": "optional string; short generation rationale",
}

REFINER_OUTPUT_SCHEMA = {
    "insight_id": "string; must match the input insight id",
    "field": "one of trigger, goal, constraint, risk",
    "new_text": "string; replacement text for the selected insight field",
    "rationale": "string; why the update addresses the failure",
    "failure_kind": "optional string; trigger_gap, goal_mismatch, constraint_violation, missing_risk, translator_noise, invalid_challenge, or non_actionable_success",
    "actionable": "optional boolean; false rejects the proposal",
}

ORACLE_VALIDATOR_OUTPUT_SCHEMA = {
    "valid": "boolean; true only when the oracle and harness check the claimed semantic property",
    "reason": "string; short explanation of the validation decision",
}

ACTIVATION_VALIDATOR_OUTPUT_SCHEMA = {
    "activates": "boolean; true only when the challenge exercises the input insight",
    "reason": "string; short explanation of the validation decision",
}

TRACE_SYNTHESIZER_OUTPUT_SCHEMA = {
    "trigger": "string; generalized condition under which the trace failure applies",
    "goal": "string; behavior the translation should preserve",
    "constraint": "string; actionable translation constraint inferred from the trace",
    "risk": "string; failure mode if the constraint is ignored",
    "guidance": "optional string; prompt guidance for future translations",
    "tags": "optional array of short strings",
    "keywords": "optional array of short strings",
    "actionable": "optional boolean; false rejects the candidate",
}

TRACE_CONTRADICTION_REVIEW_OUTPUT_SCHEMA = {
    "contradicts": "boolean; true only when the candidate semantically conflicts with existing insights",
    "reason": "string; short explanation for the decision",
    "conflicting_insight_id": "optional string; existing insight id most directly involved in the conflict",
}

COMPOSITION_RELATION_REVIEW_OUTPUT_SCHEMA = {
    "relation": "one of: supports, requires, precedes, none",
    "first_id": "optional insight id for the directed relation source; defaults to the first insight",
    "second_id": "optional insight id for the directed relation target; defaults to the second insight",
    "rationale": "string; why the gated composition evidence implies this relation",
}

LLM_JSON_ONLY_CONTRACT = (
    "Return exactly one JSON object matching the provided schema; no markdown, "
    "no code fences, no comments, and no explanatory prose. Do not add fields "
    "outside the schema."
)

LLM_STRUCTURED_CHALLENGER_SYSTEM = (
    "You are the Challenger in an adversarial Rust translation insight loop. "
    "Generate one executable challenge that activates the input insight: a "
    "self-contained C snippet, a Rust interface, an assertion harness, and an "
    "oracle that states the semantic property being checked. Preserve the exact "
    "input insight_id and do not invent project-local paths or fixtures. Make "
    "the challenge a minimal counterexample, not a broad benchmark; the harness "
    "must call the declared Rust interface and assert observable behavior from "
    "the oracle. Do not rely on external files, randomness, wall clock time, "
    "network access, or undefined behavior. "
    + LLM_JSON_ONLY_CONTRACT
)

LLM_STRUCTURED_REFINER_SYSTEM = (
    "You are the Refiner in an adversarial Rust translation insight loop. "
    "Propose the smallest evidence-backed update to trigger, goal, constraint, "
    "or risk; EvidenceGate replay must still be able to approve it before any "
    "bank write. Mark actionable=false for invalid challenges, translator noise, "
    "or non-actionable successes, and avoid project-specific patches. Preserve "
    "the insight id, title, source, tags, and parent provenance unless the schema "
    "explicitly asks for an updated field; cite concrete challenge or trace "
    "evidence in the rationale, and do not rewrite the whole insight when one "
    "field is enough. "
    + LLM_JSON_ONLY_CONTRACT
)

LLM_SEMANTIC_ORACLE_SYSTEM = (
    "You validate adversarial translation challenge oracles. Accept only when "
    "the oracle is a semantic property implied by the insight and the harness "
    "assertion checks that property through the declared Rust interface; reject "
    "tautological, unrelated, or under-specified assertions. The decision must "
    "be about observable behavior, not wording similarity, preferred style, or "
    "private implementation details that the harness cannot observe. "
    + LLM_JSON_ONLY_CONTRACT
)

LLM_ACTIVATION_VALIDATOR_SYSTEM = (
    "You validate whether a generated adversarial challenge actually activates "
    "a Rust translation insight. Decide from the insight trigger, constraint, "
    "risk, C snippet, interface, oracle, and harness; reject challenges that "
    "merely mention similar vocabulary without testing the insight behavior. "
    "Require at least one concrete input or state condition that makes the "
    "insight relevant and is exercised by the harness. "
    + LLM_JSON_ONLY_CONTRACT
)

LLM_TRACE_SYNTHESIS_SYSTEM = (
    "You synthesize reusable Rust translation insights from failed trace "
    "evidence. Infer a general translation rule, not a project-specific patch; "
    "avoid filesystem paths, fixture names, and one-off test details. Mark "
    "actionable=false when the trace is too noisy or not generalizable. Anchor "
    "each proposed rule in an observable C/Rust semantic mismatch, compiler "
    "diagnostic, failing assertion, or replay result from the trace bundle. "
    + LLM_JSON_ONLY_CONTRACT
)

LLM_TRACE_CONTRADICTION_SYSTEM = (
    "You review candidate Rust translation insights for semantic contradictions "
    "against an existing insight bank. Flag true semantic conflicts, not ordinary "
    "overlap, reinforcement, or different levels of specificity; include the "
    "main conflicting insight id when applicable. A contradiction requires "
    "mutually incompatible constraints, goals, or risk mitigations for the same "
    "translation situation. "
    + LLM_JSON_ONLY_CONTRACT
)

LLM_COMPOSITION_RELATION_SYSTEM = (
    "You infer evidence-gated semantic relations between two Rust translation "
    "insights from a compositional challenge event. Infer supports, requires, "
    "or precedes only after the event has passed sanity, compile, and test "
    "replay gates; return none when the evidence does not justify a relation. "
    "Do not infer relations from id, title, or wording overlap; use dependency "
    "evidence, replay probes, and oracle-facing behavior only. "
    + LLM_JSON_ONLY_CONTRACT
)


class StructuredChallengerAgent:
    def __init__(self, challenge_fn: Callable[[dict[str, object]], dict[str, object] | str]):
        self.challenge_fn = challenge_fn

    def generate(self, insight: InsightRecord) -> ChallengeRecord:
        raw = self.challenge_fn(_challenger_payload(insight))
        data = _parse_schema_object(raw, "challenger")
        insight_id = _required_text_field(data, "insight_id")
        if insight_id != insight.id:
            raise ValueError("Challenger output insight_id must match the input insight")
        c_program = _required_text_field(data, "c_program")
        rust_interface = _required_text_field(data, "rust_interface")
        harness = _required_text_field(data, "harness")
        oracle = _required_text_field(data, "oracle")
        if "assert" not in _normalize(harness):
            raise ValueError("Challenger harness must contain an assertion")
        return ChallengeRecord(
            insight_id=insight_id,
            c_program=c_program,
            rust_interface=rust_interface,
            harness=harness,
            oracle=oracle,
            evidence=ExecutionEvidence(compiles=False, tests_pass=False),
            activation_predicate=str(data.get("activation_predicate") or "").strip(),
        )


def _usage_to_dict(usage) -> dict[str, int]:
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _record_live_agent_call(
    call_recorder: Callable[[dict[str, object]], None] | None,
    *,
    phase: str,
    system: str,
    prompt: str,
    result,
    extra_fields: dict[str, object] | None = None,
) -> None:
    if call_recorder is None:
        return
    record = {
        "phase": phase,
        "system": system,
        "prompt": prompt,
        "raw_text": str(getattr(result, "raw_text", "") or ""),
        "elapsed_seconds": float(getattr(result, "elapsed_seconds", 0.0) or 0.0),
        "usage": _usage_to_dict(getattr(result, "usage", None)),
    }
    if extra_fields:
        record.update(extra_fields)
    call_recorder(record)


class LLMStructuredChallengerAgent(StructuredChallengerAgent):
    def __init__(self, agent, call_recorder: Callable[[dict[str, object]], None] | None = None):
        self.agent = agent
        self.call_recorder = call_recorder
        super().__init__(self._generate_with_agent)

    def _generate_with_agent(self, payload: dict[str, object]) -> str:
        prompt = json.dumps(payload, ensure_ascii=False, indent=2)
        result = self.agent.translate(LLM_STRUCTURED_CHALLENGER_SYSTEM, prompt)
        _record_live_agent_call(
            self.call_recorder,
            phase="adversarial_structured_challenger",
            system=LLM_STRUCTURED_CHALLENGER_SYSTEM,
            prompt=prompt,
            result=result,
        )
        return _agent_result_text(result)


class StructuredRefinerAgent:
    def __init__(self, refine_fn: Callable[[dict[str, object]], dict[str, object] | str]):
        self.refine_fn = refine_fn

    def propose_update(
        self,
        insight: InsightRecord,
        challenge: ChallengeRecord,
        evidence: ExecutionEvidence,
    ) -> InsightUpdateProposal:
        raw = self.refine_fn(_refiner_payload(insight, challenge, evidence))
        data = _parse_schema_object(raw, "refiner")
        if data.get("actionable") is False:
            raise ValueError("Refiner output marked the failure as non-actionable")
        insight_id = _required_text_field(data, "insight_id")
        if insight_id != insight.id:
            raise ValueError("Refiner output insight_id must match the input insight")
        field = _required_text_field(data, "field")
        if field not in {"trigger", "goal", "constraint", "risk"}:
            raise ValueError(f"Unsupported refiner update field: {field}")
        new_text = _required_text_field(data, "new_text")
        rationale = _required_text_field(data, "rationale")
        failure_kind = str(data.get("failure_kind") or "").strip()
        if failure_kind:
            rationale = f"{failure_kind}: {rationale}"
        return InsightUpdateProposal(
            insight_id=insight_id,
            field=field,
            old_text=getattr(insight, field),
            new_text=new_text,
            rationale=rationale,
        )


class LLMStructuredRefinerAgent(StructuredRefinerAgent):
    def __init__(self, agent, call_recorder: Callable[[dict[str, object]], None] | None = None):
        self.agent = agent
        self.call_recorder = call_recorder
        super().__init__(self._refine_with_agent)

    def _refine_with_agent(self, payload: dict[str, object]) -> str:
        prompt = json.dumps(payload, ensure_ascii=False, indent=2)
        result = self.agent.translate(LLM_STRUCTURED_REFINER_SYSTEM, prompt)
        _record_live_agent_call(
            self.call_recorder,
            phase="adversarial_structured_refiner",
            system=LLM_STRUCTURED_REFINER_SYSTEM,
            prompt=prompt,
            result=result,
        )
        return _agent_result_text(result)


@dataclass(frozen=True)
class SemanticOracleValidationResult:
    valid: bool
    reason: str = ""


class StructuredOracleValidator:
    def __init__(self, validate_fn: Callable[[dict[str, object]], dict[str, object] | str]):
        self.validate_fn = validate_fn

    def validate(self, challenge: ChallengeRecord, insight: InsightRecord) -> SemanticOracleValidationResult:
        raw = self.validate_fn(_oracle_validation_payload(challenge, insight))
        data = _parse_schema_object(raw, "oracle validator")
        valid = data.get("valid")
        if not isinstance(valid, bool):
            raise ValueError("Oracle validator output must include boolean field: valid")
        reason = str(data.get("reason") or "").strip()
        return SemanticOracleValidationResult(valid=valid, reason=reason)


class LLMSemanticOracleValidator(StructuredOracleValidator):
    def __init__(self, agent, call_recorder: Callable[[dict[str, object]], None] | None = None):
        self.agent = agent
        self.call_recorder = call_recorder
        super().__init__(self._validate_with_agent)

    def _validate_with_agent(self, payload: dict[str, object]) -> str:
        prompt = json.dumps(payload, ensure_ascii=False, indent=2)
        result = self.agent.translate(LLM_SEMANTIC_ORACLE_SYSTEM, prompt)
        _record_live_agent_call(
            self.call_recorder,
            phase="semantic_oracle_validation",
            system=LLM_SEMANTIC_ORACLE_SYSTEM,
            prompt=prompt,
            result=result,
        )
        return _agent_result_text(result)


@dataclass(frozen=True)
class SemanticActivationValidationResult:
    activates: bool
    reason: str = ""


class StructuredActivationValidator:
    def __init__(self, validate_fn: Callable[[dict[str, object]], dict[str, object] | str]):
        self.validate_fn = validate_fn

    def validate(
        self,
        challenge: ChallengeRecord,
        insight: InsightRecord,
        *,
        local_activation_valid: bool,
    ) -> SemanticActivationValidationResult:
        raw = self.validate_fn(_activation_validation_payload(challenge, insight, local_activation_valid))
        data = _parse_schema_object(raw, "activation validator")
        activates = data.get("activates")
        if not isinstance(activates, bool):
            raise ValueError("Activation validator output must include boolean field: activates")
        return SemanticActivationValidationResult(
            activates=activates,
            reason=str(data.get("reason") or "").strip(),
        )


class LLMActivationValidator(StructuredActivationValidator):
    def __init__(self, agent, call_recorder: Callable[[dict[str, object]], None] | None = None):
        self.agent = agent
        self.call_recorder = call_recorder
        super().__init__(self._validate_with_agent)

    def _validate_with_agent(self, payload: dict[str, object]) -> str:
        prompt = json.dumps(payload, ensure_ascii=False, indent=2)
        result = self.agent.translate(LLM_ACTIVATION_VALIDATOR_SYSTEM, prompt)
        _record_live_agent_call(
            self.call_recorder,
            phase="activation_validation",
            system=LLM_ACTIVATION_VALIDATOR_SYSTEM,
            prompt=prompt,
            result=result,
        )
        return _agent_result_text(result)


class StructuredTraceSynthesizer:
    def __init__(self, synthesize_fn: Callable[[dict[str, object]], dict[str, object] | str]):
        self.synthesize_fn = synthesize_fn

    def synthesize(
        self,
        project_dir: Path,
        record: dict,
        trace_bundle: dict[str, object],
    ) -> InsightRecord | None:
        payload = _trace_synthesis_payload(project_dir, record, trace_bundle)
        raw = self.synthesize_fn(payload)
        data = _parse_schema_object(raw, "trace synthesizer")
        if data.get("actionable") is False:
            return None
        trigger = _required_text_field(data, "trigger")
        goal = _required_text_field(data, "goal")
        constraint = _required_text_field(data, "constraint")
        risk = _required_text_field(data, "risk")
        callable_id = str(record.get("callable_id") or record.get("project") or project_dir.name)
        digest = hashlib.sha1(
            json.dumps(
                {
                    "project": project_dir.name,
                    "callable_id": callable_id,
                    "trigger": trigger,
                    "constraint": constraint,
                    "risk": risk,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:10]
        candidate_id = "trace-llm-" + "-".join([_slug(project_dir.name), _slug(callable_id), digest])
        tags = tuple(_string_list_field(data, "tags")) or ("trace-synthesized", _slug(project_dir.name))
        keywords = tuple(_string_list_field(data, "keywords")) or tuple(
            _keyword_candidates(" ".join([trigger, goal, constraint, risk]))[:8]
        )
        return InsightRecord(
            id=candidate_id,
            title=f"Trace LLM candidate: {project_dir.name}/{callable_id}",
            trigger=trigger,
            goal=goal,
            constraint=constraint,
            risk=risk,
            guidance=str(data.get("guidance") or "").strip(),
            tags=tags,
            keywords=keywords,
            source="trace-llm",
        )


class LLMTraceInsightSynthesizer(StructuredTraceSynthesizer):
    def __init__(self, agent, call_recorder: Callable[[dict[str, object]], None] | None = None):
        self.agent = agent
        self.call_recorder = call_recorder
        super().__init__(self._synthesize_with_agent)

    def _synthesize_with_agent(self, payload: dict[str, object]) -> str:
        prompt = json.dumps(payload, ensure_ascii=False, indent=2)
        result = self.agent.translate(LLM_TRACE_SYNTHESIS_SYSTEM, prompt)
        _record_live_agent_call(
            self.call_recorder,
            phase="trace_insight_synthesis",
            system=LLM_TRACE_SYNTHESIS_SYSTEM,
            prompt=prompt,
            result=result,
        )
        return _agent_result_text(result)


@dataclass(frozen=True)
class TraceContradictionReviewResult:
    contradicts: bool
    reason: str = ""
    conflicting_insight_id: str = ""


class StructuredTraceContradictionReviewer:
    def __init__(self, review_fn: Callable[[dict[str, object]], dict[str, object] | str]):
        self.review_fn = review_fn

    def review(
        self,
        candidate: InsightRecord,
        existing_records: Sequence[InsightRecord],
    ) -> TraceContradictionReviewResult:
        raw = self.review_fn(_trace_contradiction_payload(candidate, existing_records))
        data = _parse_schema_object(raw, "trace contradiction reviewer")
        contradicts = data.get("contradicts")
        if not isinstance(contradicts, bool):
            raise ValueError("Trace contradiction reviewer output must include boolean field: contradicts")
        return TraceContradictionReviewResult(
            contradicts=contradicts,
            reason=str(data.get("reason") or "").strip(),
            conflicting_insight_id=str(data.get("conflicting_insight_id") or "").strip(),
        )


class LLMTraceContradictionReviewer(StructuredTraceContradictionReviewer):
    def __init__(self, agent, call_recorder: Callable[[dict[str, object]], None] | None = None):
        self.agent = agent
        self.call_recorder = call_recorder
        super().__init__(self._review_with_agent)

    def _review_with_agent(self, payload: dict[str, object]) -> str:
        prompt = json.dumps(payload, ensure_ascii=False, indent=2)
        result = self.agent.translate(LLM_TRACE_CONTRADICTION_SYSTEM, prompt)
        _record_live_agent_call(
            self.call_recorder,
            phase="trace_contradiction_review",
            system=LLM_TRACE_CONTRADICTION_SYSTEM,
            prompt=prompt,
            result=result,
        )
        return _agent_result_text(result)


class StructuredCompositionRelationReviewer:
    def __init__(self, review_fn: Callable[[dict[str, object]], dict[str, object] | str]):
        self.review_fn = review_fn

    def review(
        self,
        insights: Sequence[InsightRecord],
        event: dict,
    ) -> InsightRelationRecord | None:
        if len(insights) != 2:
            return None
        raw = self.review_fn(_composition_relation_payload(insights, event))
        data = _parse_schema_object(raw, "composition relation reviewer")
        relation = _required_text_field(data, "relation").lower()
        if relation == "none":
            return None
        if relation not in {"supports", "requires", "precedes"}:
            raise ValueError(f"Unsupported composition relation: {relation}")
        insight_ids = {insight.id for insight in insights}
        first_id = str(data.get("first_id") or insights[0].id).strip()
        second_id = str(data.get("second_id") or insights[1].id).strip()
        if first_id not in insight_ids or second_id not in insight_ids:
            raise ValueError("Composition relation ids must refer to the reviewed insights")
        rationale = _required_text_field(data, "rationale")
        fingerprint = str(event.get("fingerprint", ""))
        return InsightRelationRecord(
            first_id=first_id,
            second_id=second_id,
            relation=relation,
            rationale=rationale,
            evidence=f"challenge_fingerprint:{fingerprint}",
        )


class LLMCompositionRelationReviewer(StructuredCompositionRelationReviewer):
    def __init__(self, agent, call_recorder: Callable[[dict[str, object]], None] | None = None):
        self.agent = agent
        self.call_recorder = call_recorder
        super().__init__(self._review_with_agent)

    def _review_with_agent(self, payload: dict[str, object]) -> str:
        prompt = json.dumps(payload, ensure_ascii=False, indent=2)
        result = self.agent.translate(LLM_COMPOSITION_RELATION_SYSTEM, prompt)
        call_metadata: dict[str, object] = {}
        insights = payload.get("insights")
        if isinstance(insights, Sequence) and len(insights) >= 2:
            first = insights[0]
            second = insights[1]
            if isinstance(first, dict) and isinstance(second, dict):
                first_id = str(first.get("id") or "").strip()
                second_id = str(second.get("id") or "").strip()
                if first_id and second_id:
                    call_metadata["first_id"] = first_id
                    call_metadata["second_id"] = second_id
        _record_live_agent_call(
            self.call_recorder,
            phase="composition_relation_review",
            system=LLM_COMPOSITION_RELATION_SYSTEM,
            prompt=prompt,
            result=result,
            extra_fields=call_metadata,
        )
        return _agent_result_text(result)


def _challenger_payload(insight: InsightRecord) -> dict[str, object]:
    return {
        "agent": "Challenger",
        "insight": insight.to_dict(),
        "schema": CHALLENGER_OUTPUT_SCHEMA,
        "acceptance_criteria": [
            "insight_id must match the input insight",
            "c_program, rust_interface, harness, and oracle must be non-empty strings",
            "harness must contain an assertion that calls the declared Rust interface",
            "generated evidence starts unchecked and must pass ChallengeToolChecker before admission",
        ],
    }


def _refiner_payload(
    insight: InsightRecord,
    challenge: ChallengeRecord,
    evidence: ExecutionEvidence,
) -> dict[str, object]:
    return {
        "agent": "Refiner",
        "insight": insight.to_dict(),
        "challenge": _challenge_to_dict(challenge),
        "trace_bundle": dict(challenge.trace_bundle),
        "evidence": {
            "compiles": evidence.compiles,
            "tests_pass": evidence.tests_pass,
            "output": evidence.output,
        },
        "schema": REFINER_OUTPUT_SCHEMA,
        "acceptance_criteria": [
            "actionable=false rejects the proposal",
            "insight_id must match the input insight",
            "field must be trigger, goal, constraint, or risk",
            "new_text and rationale must be non-empty",
            "EvidenceGate must still approve the update before persistence",
        ],
    }


def _oracle_validation_payload(challenge: ChallengeRecord, insight: InsightRecord) -> dict[str, object]:
    return {
        "agent": "SemanticOracleValidator",
        "insight": insight.to_dict(),
        "challenge": _challenge_to_dict(challenge),
        "schema": ORACLE_VALIDATOR_OUTPUT_SCHEMA,
        "acceptance_criteria": [
            "valid=true only when the oracle states a semantic property implied by the insight",
            "valid=true only when the harness assertion checks that property through the declared Rust interface",
            "valid=false for tautological, unrelated, or under-specified assertions",
        ],
    }


def _activation_validation_payload(
    challenge: ChallengeRecord,
    insight: InsightRecord,
    local_activation_valid: bool,
) -> dict[str, object]:
    return {
        "agent": "ActivationValidator",
        "insight": insight.to_dict(),
        "challenge": _challenge_to_dict(challenge),
        "local_activation_valid": local_activation_valid,
        "schema": ACTIVATION_VALIDATOR_OUTPUT_SCHEMA,
        "acceptance_criteria": [
            "activates=true only when the challenge exercises the trigger/constraint/risk of the insight",
            "activates=true may override conservative local token checks for non-template insight types",
            "activates=false when the challenge merely mentions similar vocabulary without testing the insight",
            "reason must explain the activation decision in terms of the insight and challenge behavior",
        ],
    }


def _trace_synthesis_payload(
    project_dir: Path,
    record: dict,
    trace_bundle: dict[str, object],
) -> dict[str, object]:
    callable_id = str(record.get("callable_id") or record.get("project") or project_dir.name)
    return {
        "agent": "TraceInsightSynthesizer",
        "project": project_dir.name,
        "callable_id": callable_id,
        "record": dict(record),
        "trace_bundle": dict(trace_bundle),
        "schema": TRACE_SYNTHESIZER_OUTPUT_SCHEMA,
        "acceptance_criteria": [
            "infer a reusable translation insight, not a project-specific patch",
            "do not include local filesystem paths, generated fixture names, or one-off test details",
            "set actionable=false when the failure is too noisy or not generalizable",
            "trigger, goal, constraint, and risk must be concrete enough for future prompts",
        ],
    }


def _trace_contradiction_payload(
    candidate: InsightRecord,
    existing_records: Sequence[InsightRecord],
) -> dict[str, object]:
    return {
        "agent": "TraceContradictionReviewer",
        "candidate": candidate.to_dict(),
        "existing_insights": [record.to_dict() for record in existing_records],
        "schema": TRACE_CONTRADICTION_REVIEW_OUTPUT_SCHEMA,
        "acceptance_criteria": [
            "contradicts=true only for semantic conflicts, not ordinary overlap or reinforcement",
            "ignore project-specific names unless they change the reusable insight meaning",
            "prefer contradicts=false when the candidate can coexist with existing insight constraints",
            "include conflicting_insight_id when one existing insight is the main conflict source",
        ],
    }


def _composition_relation_payload(
    insights: Sequence[InsightRecord],
    event: dict,
) -> dict[str, object]:
    return {
        "agent": "CompositionRelationReviewer",
        "insights": [insight.to_dict() for insight in insights],
        "composition_event": dict(event),
        "schema": COMPOSITION_RELATION_REVIEW_OUTPUT_SCHEMA,
        "acceptance_criteria": [
            "Only infer a relation after the local runner has passed sanity, compile, and test replay gates",
            "supports means the pair should be retrieved together but neither insight depends on the other",
            "requires means first_id has a precondition on second_id",
            "precedes means first_id should be applied or considered before second_id",
            "return relation=none when the evidence does not justify a semantic relation",
        ],
    }


def _challenge_to_dict(challenge: ChallengeRecord) -> dict[str, object]:
    return {
        "insight_id": challenge.insight_id,
        "c_program": challenge.c_program,
        "rust_interface": challenge.rust_interface,
        "harness": challenge.harness,
        "oracle": challenge.oracle,
        "activation_predicate": challenge.activation_predicate,
        "trace_bundle": dict(challenge.trace_bundle),
        "evidence": {
            "compiles": challenge.evidence.compiles,
            "tests_pass": challenge.evidence.tests_pass,
            "output": challenge.evidence.output,
        },
    }


def _parse_schema_object(raw: dict[str, object] | str, label: str) -> dict[str, object]:
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            parsed = _parse_json_object_from_text(raw, exc, label)
    else:
        parsed = raw
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} agent must return a JSON object")
    return parsed


def _parse_json_object_from_text(
    text: str,
    original_error: json.JSONDecodeError,
    label: str,
) -> dict[str, object]:
    candidates: list[str] = []
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        candidates.append(text[first : last + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"Invalid {label} JSON: {original_error}") from original_error


def _agent_result_text(result) -> str:
    return str(getattr(result, "raw_text", "") or getattr(result, "code", ""))


def _required_text_field(data: dict[str, object], field_name: str) -> str:
    value = data.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing required string field: {field_name}")
    return value.strip()


def _string_list_field(data: dict[str, object], field_name: str) -> list[str]:
    value = data.get(field_name)
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of strings")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field_name} must be a list of strings")
        text = item.strip()
        if text:
            items.append(text)
    return items


class ChallengeToolChecker:
    def __init__(
        self,
        *,
        command_runner=None,
        c_compiler: str = "cc",
        cargo: str = "cargo",
        timeout: int = 30,
    ):
        self.command_runner = command_runner or subprocess.run
        self.c_compiler = c_compiler
        self.cargo = cargo
        self.timeout = timeout

    def check(
        self,
        challenge: ChallengeRecord,
        *,
        work_dir: str | Path | None = None,
        rust_source: str | None = None,
    ) -> ChallengeRecord:
        temp_dir = None
        if work_dir is None:
            temp_dir = tempfile.TemporaryDirectory()
            root = Path(temp_dir.name)
        else:
            root = Path(work_dir)
        try:
            root.mkdir(parents=True, exist_ok=True)
            self._write_project(root, challenge, rust_source=rust_source)
            c_ok, c_output = self._run_c_check(root)
            c_runtime_ok = True
            c_runtime_output = ""
            if c_ok:
                c_runtime_ok, c_runtime_output = self._run_c_executable_check(
                    root,
                    has_main=_c_program_has_main(challenge.c_program),
                )
            rust_ok = False
            rust_output = ""
            if c_ok and c_runtime_ok:
                rust_ok, rust_output = self._run_rust_typecheck(root)
            tests_ok = False
            test_output = ""
            if c_ok and c_runtime_ok and rust_ok:
                tests_ok, test_output = self._run_rust_tests(root)
            output = "\n".join(
                part for part in (c_output, c_runtime_output, rust_output, test_output) if part
            )
            return replace(
                challenge,
                evidence=ExecutionEvidence(
                    compiles=c_ok and c_runtime_ok and rust_ok,
                    tests_pass=tests_ok,
                    output=output,
                ),
            )
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()

    def _write_project(
        self,
        root: Path,
        challenge: ChallengeRecord,
        *,
        rust_source: str | None = None,
    ) -> None:
        (root / "challenge.c").write_text(challenge.c_program, encoding="utf-8")
        (root / "Cargo.toml").write_text(
            "\n".join(
                [
                    "[package]",
                    'name = "adversarial_challenge_toolcheck"',
                    'version = "0.1.0"',
                    'edition = "2021"',
                    "",
                    "[lib]",
                    'path = "src/lib.rs"',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        src_dir = root / "src"
        src_dir.mkdir(exist_ok=True)
        (src_dir / "lib.rs").write_text(
            "\n\n".join(
                [
                    rust_source or _rust_stub_from_interface(challenge.rust_interface),
                    "#[cfg(test)]",
                    "mod generated_challenge {",
                    "    use super::*;",
                    _indent(challenge.harness, "    "),
                    "}",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def _run_c_check(self, root: Path) -> tuple[bool, str]:
        command = [
            self.c_compiler,
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-c",
            "challenge.c",
            "-o",
            "challenge.o",
        ]
        return self._run_command(command, root, failure_label="C snippet compile failed")

    def _run_c_executable_check(self, root: Path, *, has_main: bool) -> tuple[bool, str]:
        source = "challenge.c" if has_main else "challenge_main.c"
        if not has_main:
            (root / source).write_text(
                "\n".join(
                    [
                        '#include "challenge.c"',
                        "",
                        "int main(void) {",
                        "    return 0;",
                        "}",
                    ]
                ),
                encoding="utf-8",
            )
        build_command = [
            self.c_compiler,
            "-std=c11",
            "-Wall",
            "-Wextra",
            source,
            "-o",
            "challenge_c_check",
        ]
        build_ok, build_output = self._run_command(
            build_command,
            root,
            failure_label="C snippet executable build failed",
        )
        if not build_ok:
            return False, build_output
        run_ok, run_output = self._run_command(
            ["./challenge_c_check"],
            root,
            failure_label="C snippet execution failed",
        )
        return run_ok, "\n".join(part for part in (build_output, run_output) if part)

    def _run_rust_typecheck(self, root: Path) -> tuple[bool, str]:
        command = [self.cargo, "test", "--no-run", "--quiet"]
        return self._run_command(command, root, failure_label="cargo test --no-run failed")

    def _run_rust_tests(self, root: Path) -> tuple[bool, str]:
        command = [self.cargo, "test", "--quiet"]
        return self._run_command(command, root, failure_label="cargo test failed")

    def _run_command(self, command: list[str], root: Path, *, failure_label: str) -> tuple[bool, str]:
        try:
            result = self.command_runner(
                command,
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout,
            )
        except FileNotFoundError as exc:
            return False, f"{failure_label}: {exc}"
        except subprocess.TimeoutExpired as exc:
            return False, f"{failure_label}: timed out after {exc.timeout} seconds"
        output = "\n".join(
            part.strip()
            for part in (getattr(result, "stdout", ""), getattr(result, "stderr", ""))
            if part and part.strip()
        )
        if getattr(result, "returncode", 1) == 0:
            return True, output
        return False, f"{failure_label}: {output}".rstrip()


def _rust_stub_from_interface(interface: str) -> str:
    match = re.search(
        r"pub\s+fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*(?:->\s*([^;{]+))?\s*;",
        interface.strip(),
        re.DOTALL,
    )
    if not match:
        return interface
    name = match.group(1)
    args = re.sub(r"\s+", " ", match.group(2).strip())
    return_type = re.sub(r"\s+", " ", (match.group(3) or "").strip())
    if not return_type:
        return f"pub fn {name}({args}) {{}}"
    return f"pub fn {name}({args}) -> {return_type} {{\n    {_stub_value_for_return(return_type)}\n}}"


def _c_program_has_main(c_program: str) -> bool:
    return bool(re.search(r"\bmain\s*\(", _strip_c_comments(c_program)))


def _strip_c_comments(c_program: str) -> str:
    without_block = re.sub(r"/\*.*?\*/", " ", c_program, flags=re.DOTALL)
    return re.sub(r"//.*", " ", without_block)


def _stub_value_for_return(return_type: str) -> str:
    compact = return_type.replace(" ", "")
    if compact == "String":
        return "String::new()"
    if compact.startswith("Vec<"):
        return "Vec::new()"
    if compact == "bool":
        return "false"
    if compact in {"i8", "i16", "i32", "i64", "i128", "isize", "u8", "u16", "u32", "u64", "u128", "usize"}:
        return "0"
    if compact in {"f32", "f64"}:
        return "0.0"
    if compact.startswith("Option<"):
        return "None"
    if compact in {"&str", "&'staticstr"}:
        return '""'
    if compact.startswith("(") and compact.endswith(")"):
        inner = compact[1:-1]
        parts = [part for part in inner.split(",") if part]
        if not parts:
            return "()"
        values = [_stub_value_for_return(part) for part in parts]
        if len(values) == 1:
            return f"({values[0]},)"
        return f"({', '.join(values)})"
    return "unimplemented!()"


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line if line else prefix.rstrip() for line in text.splitlines())


@dataclass(frozen=True)
class SanityCheckResult:
    valid: bool
    failed_checks: tuple[str, ...]
    details: dict[str, object] = field(default_factory=dict)


class ChallengeSanityChecker:
    def __init__(
        self,
        oracle_validator: StructuredOracleValidator | None = None,
        activation_validator: StructuredActivationValidator | None = None,
    ):
        self._seen: set[str] = set()
        self._seen_signatures: list[tuple[str, set[str]]] = []
        self.oracle_validator = oracle_validator
        self.activation_validator = activation_validator

    def check(self, challenge: ChallengeRecord, insight: InsightRecord) -> SanityCheckResult:
        failed = []
        details: dict[str, object] = {}
        signature = _challenge_signature(challenge)
        if challenge.fingerprint in self._seen or self._is_near_duplicate(challenge.insight_id, signature):
            failed.append("leak")
        else:
            self._seen.add(challenge.fingerprint)
            self._seen_signatures.append((challenge.insight_id, signature))
        local_activation_valid = (
            _activates(challenge.activation_text(), insight)
            and _boundary_activates(challenge, insight)
            and _activation_predicate_matches(challenge, insight)
        )
        if self.activation_validator is not None:
            activation_result = self.activation_validator.validate(
                challenge,
                insight,
                local_activation_valid=local_activation_valid,
            )
            details["activation"] = {
                "activates": bool(activation_result.activates),
                "reason": str(activation_result.reason),
                "local_activation_valid": bool(local_activation_valid),
            }
            if not activation_result.activates:
                failed.append("activates_semantic")
        elif not local_activation_valid:
            failed.append("activates")
        if not _oracle_self_validates(challenge):
            failed.append("oracle")
        elif self.oracle_validator is not None:
            oracle_result = self.oracle_validator.validate(challenge, insight)
            details["oracle"] = {
                "valid": bool(oracle_result.valid),
                "reason": str(oracle_result.reason),
            }
            if not oracle_result.valid:
                failed.append("oracle_semantic")
        if not challenge.evidence.tool_checked:
            failed.append("toolcheck")
        return SanityCheckResult(valid=not failed, failed_checks=tuple(failed), details=details)

    def _is_near_duplicate(self, insight_id: str, signature: set[str]) -> bool:
        return any(
            seen_insight_id == insight_id and _signatures_are_near_duplicates(signature, seen_signature)
            for seen_insight_id, seen_signature in self._seen_signatures
        )


def _challenge_signature(challenge: ChallengeRecord) -> set[str]:
    text = "\n".join([challenge.c_program, challenge.rust_interface, challenge.harness, challenge.oracle])
    return {token for token in _tokens(_normalize(text)) if len(token) >= 2}


def _signatures_are_near_duplicates(left: set[str], right: set[str]) -> bool:
    if not left or not right:
        return False
    overlap = len(left & right)
    containment = overlap / min(len(left), len(right))
    jaccard = overlap / len(left | right)
    return containment >= 0.85 and jaccard >= 0.65


def _activates(text: str, insight: InsightRecord) -> bool:
    normalized = _normalize(text)
    terms = [*insight.tags, *insight.keywords, insight.trigger]
    for term in terms:
        tokens = [token for token in _tokens(_normalize(term)) if len(token) >= 3]
        if tokens and any(token in normalized for token in tokens):
            return True
    return False


def _boundary_activates(challenge: ChallengeRecord, insight: InsightRecord) -> bool:
    insight_text = _normalize_record_text(insight)
    payload = "\n".join([challenge.c_program, challenge.rust_interface, challenge.harness])
    normalized_payload = _normalize(payload)
    normalized_evidence = _normalize(challenge.evidence.output)
    if "nul" in insight_text or "char array" in insight_text:
        return any(marker in payload for marker in ("\\0", "\\x00", "\x00")) or "nul" in normalized_evidence
    if "count" in insight_text and "capacity" in insight_text:
        combined = " ".join([normalized_payload, normalized_evidence])
        return "count" in combined and "capacity" in combined
    return True


def _activation_predicate_matches(challenge: ChallengeRecord, insight: InsightRecord) -> bool:
    predicate = challenge.activation_predicate.strip()
    if not predicate:
        return True
    predicate_tokens = _meaningful_tokens(predicate)
    if not predicate_tokens:
        return False
    insight_tokens = _meaningful_tokens(_normalize_record_text(insight))
    return bool(predicate_tokens & insight_tokens)


def _oracle_self_validates(challenge: ChallengeRecord) -> bool:
    if not challenge.oracle.strip():
        return False
    harness = challenge.harness
    normalized_harness = _normalize(harness)
    if "assert" not in normalized_harness and "assertion" not in normalized_harness:
        return False
    function_name = _rust_interface_function_name(challenge.rust_interface)
    if function_name and not re.search(rf"\b{re.escape(function_name)}\s*\(", harness):
        return False
    return True


def _rust_interface_function_name(interface: str) -> str:
    match = re.search(r"pub\s+fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", interface)
    return match.group(1) if match else ""


def _meaningful_tokens(text: str) -> set[str]:
    stopwords = {
        "and",
        "are",
        "for",
        "from",
        "has",
        "into",
        "must",
        "not",
        "only",
        "present",
        "the",
        "this",
        "with",
    }
    return {token for token in _tokens(_normalize(text)) if len(token) >= 3 and token not in stopwords}


@dataclass(frozen=True)
class InsightUpdateProposal:
    insight_id: str
    field: str
    old_text: str
    new_text: str
    rationale: str


@dataclass(frozen=True)
class EvidenceGateResult:
    accepted: bool
    reason: str
    updated_insight: InsightRecord


class FailureAttributor:
    def propose_update(self, insight: InsightRecord, evidence: ExecutionEvidence) -> InsightUpdateProposal:
        output = evidence.output.strip()
        lowered = output.lower()
        if any(marker in lowered for marker in ("invalid challenge", "invalid oracle", "bad harness")):
            field = "risk"
            failure_kind = "invalid_challenge"
        elif any(marker in lowered for marker in ("non-actionable", "non actionable", "already passes")):
            field = "risk"
            failure_kind = "non_actionable_success"
        elif any(marker in lowered for marker in ("unpatchable", "model output", "llm", "malformed rust")):
            field = "risk"
            failure_kind = "translator_noise"
        elif any(marker in lowered for marker in ("nul", "padding", "panic", "assertion")):
            field = "risk"
            failure_kind = "missing_risk"
        elif any(marker in lowered for marker in ("signature", "interface", "api")):
            field = "goal"
            failure_kind = "goal_mismatch"
        elif any(marker in lowered for marker in ("borrow", "alias", "capacity", "bounds")):
            field = "constraint"
            failure_kind = "constraint_violation"
        else:
            field = "trigger"
            failure_kind = "trigger_gap"
        old_text = getattr(insight, field)
        evidence_text = output[:500] if output else "tool-observed failure"
        new_text = old_text if evidence_text in old_text else f"{old_text}; observed: {evidence_text}"
        return InsightUpdateProposal(
            insight_id=insight.id,
            field=field,
            old_text=old_text,
            new_text=new_text,
            rationale=f"{failure_kind}: failure attribution from execution evidence",
        )


class EvidenceGate:
    def evaluate(
        self,
        insight: InsightRecord,
        proposal: InsightUpdateProposal,
        *,
        before: ExecutionEvidence,
        after: ExecutionEvidence,
    ) -> EvidenceGateResult:
        if proposal.insight_id != insight.id:
            return EvidenceGateResult(False, "proposal targets a different insight", insight)
        if after.score <= before.score or not after.tests_pass:
            return EvidenceGateResult(False, "replay evidence did not improve", insight)
        updated = _replace_insight_field(insight, proposal.field, proposal.new_text)
        return EvidenceGateResult(True, "replay evidence improved", updated)


class InsightConstrainedReplayRunner:
    """Replay one challenge before and after an insight update via injected translation."""

    def __init__(
        self,
        translate_fn: Callable[[ChallengeRecord, InsightRecord], str],
        *,
        tool_checker: ChallengeToolChecker | None = None,
        gate: EvidenceGate | None = None,
    ):
        self.translate_fn = translate_fn
        self.tool_checker = tool_checker or ChallengeToolChecker()
        self.gate = gate or EvidenceGate()

    def evaluate_update(
        self,
        insight: InsightRecord,
        challenge: ChallengeRecord,
        proposal: InsightUpdateProposal,
    ) -> EvidenceGateResult:
        before_source = self.translate_fn(challenge, insight)
        before = self.tool_checker.check(challenge, rust_source=before_source)
        updated_candidate = _replace_insight_field(insight, proposal.field, proposal.new_text)
        after_source = self.translate_fn(challenge, updated_candidate)
        after = self.tool_checker.check(challenge, rust_source=after_source)
        return self.gate.evaluate(
            insight,
            proposal,
            before=before.evidence,
            after=after.evidence,
        )


class TranslatorChallengeReplayAdapter:
    """Callable adapter that lets replay use a caller-supplied Translator factory."""

    def __init__(
        self,
        translator_factory: Callable[[dict], object],
        *,
        base_config: dict | None = None,
    ):
        self.translator_factory = translator_factory
        self.base_config = dict(base_config or {})

    def __call__(self, challenge: ChallengeRecord, insight: InsightRecord) -> str:
        config = dict(self.base_config)
        config["adversarial_replay_challenge"] = self._challenge_payload(challenge)
        config["adversarial_replay_insight"] = insight.to_dict()
        translator = self.translator_factory(config)
        translate = getattr(translator, "translate_adversarial_challenge", None)
        if not callable(translate):
            raise TypeError("translator must provide translate_adversarial_challenge(challenge, insight)")
        rust_source = translate(challenge, insight)
        if not isinstance(rust_source, str):
            raise TypeError("translate_adversarial_challenge() must return Rust source text")
        return rust_source

    @staticmethod
    def _evidence_payload(evidence: ExecutionEvidence | None) -> dict | None:
        if evidence is None:
            return None
        return {
            "compiles": evidence.compiles,
            "tests_pass": evidence.tests_pass,
            "output": evidence.output,
        }

    @classmethod
    def _challenge_payload(cls, challenge: ChallengeRecord) -> dict:
        return {
            "insight_id": challenge.insight_id,
            "c_program": challenge.c_program,
            "rust_interface": challenge.rust_interface,
            "harness": challenge.harness,
            "oracle": challenge.oracle,
            "activation_predicate": challenge.activation_predicate,
            "fingerprint": challenge.fingerprint,
            "evidence": cls._evidence_payload(challenge.evidence),
            "replay_evidence": cls._evidence_payload(challenge.replay_evidence),
            "trace_bundle": dict(challenge.trace_bundle),
        }


def _replace_insight_field(insight: InsightRecord, field: str, value: str) -> InsightRecord:
    raw = insight.to_dict()
    if field not in {"trigger", "goal", "constraint", "risk"}:
        raise ValueError(f"Unsupported insight update field: {field}")
    raw[field] = value
    raw["version"] = insight.version + 1
    raw["parents"] = [*insight.parents, f"{insight.id}@v{insight.version}"]
    return InsightRecord.from_dict(raw)


class TraceReflector:
    def __init__(self, bank: InsightBank):
        self.bank = bank
        self._known_ids = {record.id for record in bank.records}

    def reflect(self, run_root: str | Path) -> list[ChallengeRecord]:
        challenges: list[ChallengeRecord] = []
        for project_dir in _iter_project_artifacts(Path(run_root)):
            translation_history = _load_json(project_dir / "translation_history.json", {})
            for record in _load_json(project_dir / "test_fix_history.json", []):
                if not isinstance(record, dict):
                    continue
                evidence = _evidence_from_test_record(record)
                if evidence.tests_pass:
                    continue
                for insight_id in _insight_ids_from_record(record):
                    if insight_id not in self._known_ids:
                        continue
                    challenges.append(
                        ChallengeRecord(
                            insight_id=insight_id,
                            c_program=str(record.get("c_context") or record.get("test_output") or ""),
                            rust_interface=str(record.get("callable_id") or record.get("project") or project_dir.name),
                            harness=str(record.get("test_output", ""))[:2000],
                            oracle=str(record.get("post_patch_test_output") or record.get("test_output") or "")[:2000],
                            evidence=evidence,
                            replay_evidence=_replay_evidence_from_test_record(record),
                            trace_bundle=_trace_bundle_from_test_record(
                                project_dir,
                                record,
                                translation_history,
                            ),
                        )
                    )
            challenges.extend(
                _compile_challenges_from_translation_history(
                    project_dir,
                    translation_history,
                    self._known_ids,
                )
            )
            challenges.extend(_llm_call_challenges_from_project(project_dir, self._known_ids))
        return challenges


class TraceInsightMiner:
    def __init__(
        self,
        bank: InsightBank,
        *,
        min_signal_tokens: int = 4,
        synthesizer: StructuredTraceSynthesizer | None = None,
        contradiction_reviewer: StructuredTraceContradictionReviewer | None = None,
    ):
        self.bank = bank
        self.min_signal_tokens = min_signal_tokens
        self.synthesizer = synthesizer
        self.contradiction_reviewer = contradiction_reviewer
        self._known_text = [_normalize_record_text(record) for record in bank.records]
        self._known_contradiction_rules = [
            _contradiction_rules_from_record(record) for record in bank.records
        ]

    def mine(self, run_root: str | Path) -> list[InsightRecord]:
        candidates: list[InsightRecord] = []
        seen: set[str] = set()
        for project_dir in _iter_project_artifacts(Path(run_root)):
            translation_history = _load_json(project_dir / "translation_history.json", {})
            for record in _load_json(project_dir / "test_fix_history.json", []):
                if not isinstance(record, dict):
                    continue
                candidate = self._candidate_from_record(project_dir, record, translation_history)
                if candidate is None:
                    continue
                fingerprint = _candidate_fingerprint(candidate)
                if fingerprint in seen:
                    continue
                if self._duplicates_existing(candidate):
                    continue
                if self._contradicts_existing(candidate):
                    continue
                if self._semantically_contradicts_existing(candidate):
                    continue
                seen.add(fingerprint)
                candidates.append(candidate)
            for record in _compile_trace_records_from_translation_history(project_dir, translation_history):
                candidate = self._candidate_from_record(project_dir, record, translation_history)
                if candidate is None:
                    continue
                fingerprint = _candidate_fingerprint(candidate)
                if fingerprint in seen:
                    continue
                if self._duplicates_existing(candidate):
                    continue
                if self._contradicts_existing(candidate):
                    continue
                if self._semantically_contradicts_existing(candidate):
                    continue
                seen.add(fingerprint)
                candidates.append(candidate)
        return candidates

    def _candidate_from_record(
        self,
        project_dir: Path,
        record: dict,
        translation_history: dict | None = None,
    ) -> InsightRecord | None:
        evidence = _evidence_from_test_record(record)
        if evidence.tests_pass:
            return None
        output = str(record.get("test_output", "")).strip()
        if len(_keyword_candidates(output)) < self.min_signal_tokens:
            return None
        callable_id = str(record.get("callable_id") or record.get("project") or project_dir.name)
        context = str(
            record.get("c_context")
            or record.get("c_code")
            or record.get("rust_context")
            or record.get("callable_context")
            or ""
        ).strip()
        if _looks_project_specific_trace(" ".join([context, output])):
            return None
        if self.synthesizer is not None:
            return self.synthesizer.synthesize(
                project_dir,
                record,
                _trace_bundle_from_test_record(project_dir, record, translation_history or {}),
            )
        trigger = context or f"Failure observed while translating {callable_id}"
        if len(_tokens(trigger)) < 3:
            return None
        evidence_text = _truncate(output, 300)
        candidate_id = "trace-" + "-".join(
            [
                _slug(project_dir.name),
                _slug(callable_id),
                hashlib.sha1((trigger + "\n" + output).encode("utf-8")).hexdigest()[:10],
            ]
        )
        return InsightRecord(
            id=candidate_id,
            title=f"Trace candidate: {project_dir.name}/{callable_id}",
            trigger=_truncate(trigger, 300),
            goal=f"Preserve behavior observed in failing trace for {callable_id}.",
            constraint=f"Account for failing evidence: {evidence_text}",
            risk=f"Without this insight, translations may reproduce: {evidence_text}",
            guidance="Review this trace-mined candidate before admitting it to the insight bank.",
            tags=("trace-mined", _slug(project_dir.name)),
            keywords=tuple(_keyword_candidates(" ".join([trigger, output]))[:8]),
            source="trace",
        )

    def _duplicates_existing(self, candidate: InsightRecord) -> bool:
        evidence = _failure_signature(candidate.constraint)
        if not evidence:
            return False
        return any(evidence in known_text for known_text in self._known_text)

    def _contradicts_existing(self, candidate: InsightRecord) -> bool:
        candidate_rules = _contradiction_rules_from_record(candidate)
        candidate_text = _normalize_record_text(candidate)
        for known_text, known_rules in zip(self._known_text, self._known_contradiction_rules):
            if _rules_contradict_text(known_rules["forbidden"], candidate_text):
                return True
            if _rules_contradict_text(candidate_rules["forbidden"], known_text):
                return True
        return False

    def _semantically_contradicts_existing(self, candidate: InsightRecord) -> bool:
        if self.contradiction_reviewer is None or not self.bank.records:
            return False
        review = self.contradiction_reviewer.review(candidate, self.bank.records)
        return review.contradicts


def _looks_project_specific_trace(text: str) -> bool:
    normalized = text.replace("\\", "/")
    path_markers = (
        "/tmp/",
        "/var/tmp/",
        "/data00/",
        "/home/",
        "output/runs/",
        "target/debug/",
        "generated/",
    )
    if not any(marker in normalized for marker in path_markers):
        return False
    lowered = _normalize(normalized)
    local_markers = (
        "hard codes",
        "hard coded",
        "project local",
        "fixture path",
        "generated test",
        "not found",
        "no such file",
    )
    return any(marker in lowered for marker in local_markers)


@contextlib.contextmanager
def exclusive_insight_bank_lock(bank_path: str | Path):
    lock_target = Path(bank_path).resolve()
    lock_path = lock_target.with_name(lock_target.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def _exclusive_bank_update_lock(bank_path: str | Path, output_bank_path: str | Path):
    input_path = Path(bank_path).resolve()
    output_path = Path(output_bank_path).resolve()
    if input_path != output_path:
        yield
        return

    with exclusive_insight_bank_lock(output_path):
        yield


def run_offline_adversarial_learning(
    run_root: str | Path,
    bank_path: str | Path,
    output_bank_path: str | Path,
    report_path: str | Path,
    *,
    admit_trace_candidates: bool = False,
    admit_composition_failure_candidates: bool = False,
    run_synthetic_challenges: bool = False,
    run_compositional_challenges: bool = False,
    compositional_limit: int = 5,
    challenge_tool_checker: ChallengeToolChecker | None = None,
    challenger_agent: StructuredChallengerAgent | None = None,
    refiner_agent: StructuredRefinerAgent | None = None,
    oracle_validator: StructuredOracleValidator | None = None,
    activation_validator: StructuredActivationValidator | None = None,
    trace_synthesizer: StructuredTraceSynthesizer | None = None,
    trace_contradiction_reviewer: StructuredTraceContradictionReviewer | None = None,
    composition_relation_reviewer: StructuredCompositionRelationReviewer | None = None,
    replay_translate_fn: Callable[[ChallengeRecord, InsightRecord], str] | None = None,
) -> dict:
    with _exclusive_bank_update_lock(bank_path, output_bank_path):
        return _run_offline_adversarial_learning_unlocked(
            run_root,
            bank_path,
            output_bank_path,
            report_path,
            admit_trace_candidates=admit_trace_candidates,
            admit_composition_failure_candidates=admit_composition_failure_candidates,
            run_synthetic_challenges=run_synthetic_challenges,
            run_compositional_challenges=run_compositional_challenges,
            compositional_limit=compositional_limit,
            challenge_tool_checker=challenge_tool_checker,
            challenger_agent=challenger_agent,
            refiner_agent=refiner_agent,
            oracle_validator=oracle_validator,
            activation_validator=activation_validator,
            trace_synthesizer=trace_synthesizer,
            trace_contradiction_reviewer=trace_contradiction_reviewer,
            composition_relation_reviewer=composition_relation_reviewer,
            replay_translate_fn=replay_translate_fn,
        )


def _run_offline_adversarial_learning_unlocked(
    run_root: str | Path,
    bank_path: str | Path,
    output_bank_path: str | Path,
    report_path: str | Path,
    *,
    admit_trace_candidates: bool = False,
    admit_composition_failure_candidates: bool = False,
    run_synthetic_challenges: bool = False,
    run_compositional_challenges: bool = False,
    compositional_limit: int = 5,
    challenge_tool_checker: ChallengeToolChecker | None = None,
    challenger_agent: StructuredChallengerAgent | None = None,
    refiner_agent: StructuredRefinerAgent | None = None,
    oracle_validator: StructuredOracleValidator | None = None,
    activation_validator: StructuredActivationValidator | None = None,
    trace_synthesizer: StructuredTraceSynthesizer | None = None,
    trace_contradiction_reviewer: StructuredTraceContradictionReviewer | None = None,
    composition_relation_reviewer: StructuredCompositionRelationReviewer | None = None,
    replay_translate_fn: Callable[[ChallengeRecord, InsightRecord], str] | None = None,
) -> dict:
    bank = InsightBank.load(bank_path)
    records_by_id = {record.id: record for record in bank.records}
    checker = ChallengeSanityChecker(
        oracle_validator=oracle_validator,
        activation_validator=activation_validator,
    )
    attributor = FailureAttributor()
    gate = EvidenceGate()
    report_events: list[dict] = []
    candidate_events: list[dict] = []
    synthetic_events: list[dict] = []
    composition_events: list[dict] = []
    accepted_updates = 0
    admitted_candidates = 0
    relation_updates = 0
    relation_records = list(bank.relations)

    for challenge in TraceReflector(bank).reflect(run_root):
        insight = records_by_id.get(challenge.insight_id)
        if insight is None:
            continue
        sanity = checker.check(challenge, insight)
        event = {
            "insight_id": challenge.insight_id,
            "fingerprint": challenge.fingerprint,
            "sanity_valid": sanity.valid,
            "failed_checks": list(sanity.failed_checks),
            "accepted": False,
        }
        if not sanity.valid:
            report_events.append(event)
            continue
        proposal = _propose_insight_update(
            insight,
            challenge,
            refiner_agent=refiner_agent,
            fallback_attributor=attributor,
        )
        if replay_translate_fn is not None:
            replay_checker = challenge_tool_checker or ChallengeToolChecker()
            updated_candidate = _replace_insight_field(insight, proposal.field, proposal.new_text)
            before_source = replay_translate_fn(challenge, insight)
            before_checked = replay_checker.check(challenge, rust_source=before_source)
            after_source = replay_translate_fn(challenge, updated_candidate)
            after_checked = replay_checker.check(challenge, rust_source=after_source)
            result = gate.evaluate(
                insight,
                proposal,
                before=before_checked.evidence,
                after=after_checked.evidence,
            )
            event["fresh_replay"] = True
            event["fresh_replay_evidence"] = {
                "before": _evidence_to_dict(before_checked.evidence),
                "after": _evidence_to_dict(after_checked.evidence),
            }
        else:
            replay = challenge.replay_evidence or challenge.evidence
            result = gate.evaluate(insight, proposal, before=challenge.evidence, after=replay)
            event["fresh_replay"] = False
        event.update(
            {
                "proposal": {
                    "field": proposal.field,
                    "old_text": proposal.old_text,
                    "new_text": proposal.new_text,
                    "rationale": proposal.rationale,
                },
                "accepted": result.accepted,
                "reason": result.reason,
            }
        )
        if result.accepted:
            records_by_id[challenge.insight_id] = result.updated_insight
            accepted_updates += 1
        report_events.append(event)

    if run_synthetic_challenges:
        synthetic_tool_checker = challenge_tool_checker or ChallengeToolChecker()
        synthetic_sanity_checker = ChallengeSanityChecker(
            oracle_validator=oracle_validator,
            activation_validator=activation_validator,
        )
        generator = SyntheticChallengeGenerator()
        for insight in list(records_by_id.values()):
            challenge = _generate_single_insight_challenge(
                insight,
                generator=generator,
                challenger_agent=challenger_agent,
            )
            synthetic_events.append(
                _execute_generated_challenge(
                    challenge,
                    insight,
                    tool_checker=synthetic_tool_checker,
                    sanity_checker=synthetic_sanity_checker,
                )
            )

    if run_compositional_challenges:
        composition_tool_checker = challenge_tool_checker or ChallengeToolChecker()
        composition_sanity_checker = ChallengeSanityChecker(
            oracle_validator=oracle_validator,
            activation_validator=activation_validator,
        )
        generator = CompositionalChallengeGenerator()
        for insight_set in _composition_candidate_sets(
            list(records_by_id.values()),
            limit=max(compositional_limit, 0),
            relations=relation_records,
        ):
            challenge = generator.generate(insight_set)
            event = _execute_generated_challenge(
                challenge,
                _combined_insight_for_challenge(challenge, insight_set),
                tool_checker=composition_tool_checker,
                sanity_checker=composition_sanity_checker,
            )
            dependency = _composition_interaction_dependency(
                insight_set,
                challenge,
                tool_checker=composition_tool_checker,
            )
            event["interaction_dependency_valid"] = dependency["valid"]
            event["interaction_dependency_matches"] = dependency["matches"]
            if "executable_dependency_valid" in dependency:
                event["executable_dependency_valid"] = dependency["executable_dependency_valid"]
                event["executable_dependency_probes"] = dependency["executable_dependency_probes"]
            translator_dependency = _composition_translator_replay_dependency(
                insight_set,
                challenge,
                tool_checker=composition_tool_checker,
                replay_translate_fn=replay_translate_fn,
            )
            if translator_dependency is not None:
                event["translator_replay_dependency_valid"] = translator_dependency["valid"]
                event["translator_replay_dependency_probes"] = translator_dependency["probes"]
            relation = _composition_relation_record(
                insight_set,
                event,
                reviewer=composition_relation_reviewer,
            )
            if relation is not None and not _has_relation(relation_records, relation):
                relation_records.append(relation)
                relation_updates += 1
                event["relation_update"] = relation.to_dict()
            if admit_composition_failure_candidates:
                candidate = _composition_failure_candidate(insight_set, event)
                if candidate is not None:
                    candidate_event, admitted = _validate_candidate_insight(
                        candidate,
                        generator=SyntheticChallengeGenerator(),
                        challenger_agent=challenger_agent,
                        tool_checker=composition_tool_checker,
                        sanity_checker=ChallengeSanityChecker(
                            oracle_validator=oracle_validator,
                            activation_validator=activation_validator,
                        ),
                    )
                    if admitted:
                        records_by_id[candidate.id] = candidate
                        admitted_candidates += 1
                    candidate_events.append(candidate_event)
            composition_events.append(event)

    if admit_trace_candidates:
        candidate_bank = InsightBank(records_by_id.values())
        candidate_tool_checker = challenge_tool_checker or ChallengeToolChecker()
        candidate_sanity_checker = ChallengeSanityChecker(
            oracle_validator=oracle_validator,
            activation_validator=activation_validator,
        )
        generator = SyntheticChallengeGenerator()
        for candidate in TraceInsightMiner(
            candidate_bank,
            synthesizer=trace_synthesizer,
            contradiction_reviewer=trace_contradiction_reviewer,
        ).mine(run_root):
            event, admitted = _validate_candidate_insight(
                candidate,
                generator=generator,
                challenger_agent=challenger_agent,
                tool_checker=candidate_tool_checker,
                sanity_checker=candidate_sanity_checker,
            )
            if admitted:
                records_by_id[candidate.id] = candidate
                admitted_candidates += 1
            candidate_events.append(event)

    updated_bank = InsightBank([records_by_id[record.id] for record in bank.records], relation_records)
    if admitted_candidates:
        existing_ids = {record.id for record in updated_bank.records}
        updated_bank.records.extend(
            record for record_id, record in records_by_id.items() if record_id not in existing_ids
        )
    updated_bank.save(output_bank_path)
    summary = {
        "run_root": str(run_root),
        "input_bank": str(bank_path),
        "output_bank": str(output_bank_path),
        "challenges": len(report_events),
        "accepted_updates": accepted_updates,
        "admitted_candidates": admitted_candidates,
        "relation_updates": relation_updates,
        "synthetic_challenges": len(synthetic_events),
        "compositional_challenges": len(composition_events),
        "events": report_events,
        "candidate_events": candidate_events,
        "synthetic_events": synthetic_events,
        "composition_events": composition_events,
    }
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _evidence_to_dict(evidence: ExecutionEvidence) -> dict:
    return {
        "compiles": evidence.compiles,
        "tests_pass": evidence.tests_pass,
        "output": evidence.output,
    }


def _execute_generated_challenge(
    challenge: ChallengeRecord,
    insight: InsightRecord,
    *,
    tool_checker: ChallengeToolChecker,
    sanity_checker: ChallengeSanityChecker,
) -> dict:
    reference_source = challenge.trace_bundle.get("reference_source")
    reference_replay = isinstance(reference_source, str) and bool(reference_source.strip())
    if reference_replay:
        try:
            checked = tool_checker.check(challenge, rust_source=reference_source)
        except TypeError as exc:
            if "rust_source" not in str(exc):
                raise
            checked = tool_checker.check(challenge)
            reference_replay = False
    else:
        checked = tool_checker.check(challenge)
    sanity = sanity_checker.check(checked, insight)
    return {
        "insight_id": checked.insight_id,
        "fingerprint": checked.fingerprint,
        "sanity_valid": sanity.valid,
        "failed_checks": list(sanity.failed_checks),
        "reference_replay": reference_replay,
        "tool_evidence": {
            "compiles": checked.evidence.compiles,
            "tests_pass": checked.evidence.tests_pass,
            "output": checked.evidence.output,
        },
    }


def _validate_candidate_insight(
    candidate: InsightRecord,
    *,
    generator: SyntheticChallengeGenerator,
    challenger_agent: StructuredChallengerAgent | None,
    tool_checker: ChallengeToolChecker,
    sanity_checker: ChallengeSanityChecker,
) -> tuple[dict, bool]:
    event = {
        "candidate_id": candidate.id,
        "source": candidate.source,
        "status": "rejected",
        "failed_checks": [],
    }
    challenge = _generate_single_insight_challenge(
        candidate,
        generator=generator,
        challenger_agent=challenger_agent,
    )
    checked_challenge = tool_checker.check(challenge)
    sanity = sanity_checker.check(checked_challenge, candidate)
    event.update(
        {
            "challenge_fingerprint": checked_challenge.fingerprint,
            "sanity_valid": sanity.valid,
            "failed_checks": list(sanity.failed_checks),
            "tool_evidence": {
                "compiles": checked_challenge.evidence.compiles,
                "tests_pass": checked_challenge.evidence.tests_pass,
                "output": checked_challenge.evidence.output,
            },
        }
    )
    admitted = sanity.valid and checked_challenge.evidence.compiles
    if admitted:
        event["status"] = "admitted"
    return event, admitted


def _composition_failure_candidate(
    insights: Sequence[InsightRecord],
    event: dict,
) -> InsightRecord | None:
    evidence = event.get("tool_evidence")
    if not isinstance(evidence, dict):
        return None
    if event.get("interaction_dependency_valid") is False:
        return None
    if not event.get("sanity_valid") or not evidence.get("compiles") or evidence.get("tests_pass"):
        return None
    failure_text = str(evidence.get("output") or "").strip()
    insight_ids = [insight.id for insight in insights]
    digest = hashlib.sha1(("+".join(insight_ids) + "\n" + failure_text).encode("utf-8")).hexdigest()[:10]
    candidate_id = "composition-candidate-" + "-".join(_slug(insight_id) for insight_id in insight_ids) + f"-{digest}"
    interaction = " + ".join(insight_ids)
    evidence_text = _truncate(failure_text or "composition challenge failed", 300)
    combined_text = " ".join(
        [
            interaction,
            evidence_text,
            " ".join(insight.trigger for insight in insights),
            " ".join(insight.constraint for insight in insights),
            " ".join(insight.risk for insight in insights),
        ]
    )
    return InsightRecord(
        id=candidate_id,
        title=f"Composition candidate: {interaction}",
        trigger=f"Joint activation of {interaction} produced failing compositional evidence.",
        goal="Preserve behavior when the contributing insights are active together.",
        constraint=f"Account for the observed interaction failure: {evidence_text}",
        risk=f"Applying these insights independently may still fail when combined: {evidence_text}",
        guidance="Validate this composition-mined candidate before using it as a prompt insight.",
        tags=("composition-mined", *tuple(_slug(insight_id) for insight_id in insight_ids)),
        keywords=tuple(_keyword_candidates(combined_text)[:8]),
        source="composition",
    )


def _propose_insight_update(
    insight: InsightRecord,
    challenge: ChallengeRecord,
    *,
    refiner_agent: StructuredRefinerAgent | None,
    fallback_attributor: FailureAttributor,
) -> InsightUpdateProposal:
    if refiner_agent is not None:
        return refiner_agent.propose_update(insight, challenge, challenge.evidence)
    return fallback_attributor.propose_update(insight, challenge.evidence)


def _generate_single_insight_challenge(
    insight: InsightRecord,
    *,
    generator: SyntheticChallengeGenerator,
    challenger_agent: StructuredChallengerAgent | None,
) -> ChallengeRecord:
    if challenger_agent is not None:
        return challenger_agent.generate(insight)
    return generator.generate(insight)


@dataclass(frozen=True)
class _CompositionPairScore:
    support: float
    conflict: float


def _composition_candidate_sets(
    records: list[InsightRecord],
    *,
    limit: int,
    relations: Sequence[InsightRelationRecord] = (),
) -> list[tuple[InsightRecord, ...]]:
    if limit <= 0:
        return []
    unique_records = _dedupe_insight_records(records)
    if len(unique_records) < 2:
        return []

    relation_scores = _composition_relation_scores(relations)
    tag_vocabulary = {
        tag
        for record in unique_records
        for tag in record.tags
        if str(tag).strip()
    }
    selected_sets: list[tuple[InsightRecord, ...]] = []
    seen: set[tuple[str, ...]] = set()

    for seed in unique_records:
        selected = [seed]
        while len(selected) < COMPOSITION_MAX_INSIGHTS:
            selected_ids = {record.id for record in selected}
            current_score = _composition_set_score(
                selected,
                tag_vocabulary=tag_vocabulary,
                relation_scores=relation_scores,
            )
            best_candidate: InsightRecord | None = None
            best_gain = 0.0
            best_order = len(unique_records)
            for order, candidate in enumerate(unique_records):
                if candidate.id in selected_ids:
                    continue
                candidate_set = [*selected, candidate]
                if _composition_has_unresolved_conflict(
                    candidate_set,
                    relation_scores=relation_scores,
                ):
                    continue
                candidate_score = _composition_set_score(
                    candidate_set,
                    tag_vocabulary=tag_vocabulary,
                    relation_scores=relation_scores,
                )
                gain = candidate_score - current_score
                if gain > best_gain or (
                    gain == best_gain and best_candidate is not None and order < best_order
                ):
                    best_candidate = candidate
                    best_gain = gain
                    best_order = order
            if best_candidate is None or best_gain <= 0.0:
                break
            selected.append(best_candidate)

        if len(selected) < 2:
            continue
        key = tuple(sorted(record.id for record in selected))
        if key in seen:
            continue
        seen.add(key)
        selected_sets.append(tuple(selected))
        if len(selected_sets) >= limit:
            break

    return selected_sets


def _dedupe_insight_records(records: Sequence[InsightRecord]) -> list[InsightRecord]:
    seen: set[str] = set()
    unique: list[InsightRecord] = []
    for record in records:
        if record.id in seen:
            continue
        seen.add(record.id)
        unique.append(record)
    return unique


def _composition_relation_scores(
    relations: Sequence[InsightRelationRecord],
) -> dict[frozenset[str], _CompositionPairScore]:
    scores: dict[frozenset[str], _CompositionPairScore] = {}
    for relation in relations:
        key = frozenset((relation.first_id, relation.second_id))
        if len(key) != 2:
            continue
        label = _normalize(relation.relation)
        if label in {"supports", "support", "requires", "require", "precedes", "precede"}:
            scores[key] = _CompositionPairScore(support=1.0, conflict=0.0)
        elif label in {"conflicts", "conflict", "incompatible", "contradicts", "contradict"}:
            scores[key] = _CompositionPairScore(support=0.0, conflict=1.0)
    return scores


def _composition_set_score(
    records: Sequence[InsightRecord],
    *,
    tag_vocabulary: set[str],
    relation_scores: dict[frozenset[str], _CompositionPairScore],
) -> float:
    return (
        _composition_coverage(records, tag_vocabulary)
        + COMPOSITION_SUPPORT_WEIGHT * _composition_support(records, relation_scores)
        - COMPOSITION_CONFLICT_WEIGHT * _composition_conflict(records, relation_scores)
    )


def _composition_coverage(records: Sequence[InsightRecord], tag_vocabulary: set[str]) -> float:
    if not records:
        return 0.0
    selected_tags = {
        tag
        for record in records
        for tag in record.tags
        if str(tag).strip()
    }
    if selected_tags and tag_vocabulary:
        return len(selected_tags) / min(COMPOSITION_MAX_INSIGHTS, len(tag_vocabulary))
    return min(len(records), COMPOSITION_MAX_INSIGHTS) / COMPOSITION_MAX_INSIGHTS


def _composition_support(
    records: Sequence[InsightRecord],
    relation_scores: dict[frozenset[str], _CompositionPairScore],
) -> float:
    pair_scores = _composition_pair_scores(records, relation_scores)
    if not pair_scores:
        return 0.0
    return sum(score.support for score in pair_scores) / len(pair_scores)


def _composition_conflict(
    records: Sequence[InsightRecord],
    relation_scores: dict[frozenset[str], _CompositionPairScore],
) -> float:
    pair_scores = _composition_pair_scores(records, relation_scores)
    if not pair_scores:
        return 0.0
    return sum(score.conflict for score in pair_scores) / len(pair_scores)


def _composition_pair_scores(
    records: Sequence[InsightRecord],
    relation_scores: dict[frozenset[str], _CompositionPairScore],
) -> list[_CompositionPairScore]:
    scores: list[_CompositionPairScore] = []
    for first_index, first in enumerate(records):
        for second in records[first_index + 1 :]:
            scores.append(_composition_pair_score(first, second, relation_scores))
    return scores


def _composition_pair_score(
    first: InsightRecord,
    second: InsightRecord,
    relation_scores: dict[frozenset[str], _CompositionPairScore],
) -> _CompositionPairScore:
    relation_score = relation_scores.get(frozenset((first.id, second.id)))
    if relation_score is not None:
        return relation_score

    first_tags = set(first.tags)
    second_tags = set(second.tags)
    tag_support = _jaccard(first_tags, second_tags)

    first_keywords = {_normalize(keyword) for keyword in first.keywords if _normalize(keyword)}
    second_keywords = {_normalize(keyword) for keyword in second.keywords if _normalize(keyword)}
    keyword_support = _jaccard(first_keywords, second_keywords)

    first_tokens = set(_composition_score_tokens(first))
    second_tokens = set(_composition_score_tokens(second))
    text_support = _jaccard(first_tokens, second_tokens)

    support = min(1.0, 0.5 * tag_support + 0.3 * keyword_support + 0.2 * text_support)
    conflict = _heuristic_composition_conflict(first, second)
    return _CompositionPairScore(support=support, conflict=conflict)


def _composition_score_tokens(record: InsightRecord) -> list[str]:
    return [
        token
        for token in re.findall(
            r"[a-z0-9_]+",
            _normalize(
                " ".join([record.trigger, record.goal, record.constraint, record.risk, record.guidance])
            ),
        )
        if len(token) >= 5
    ]


def _heuristic_composition_conflict(first: InsightRecord, second: InsightRecord) -> float:
    joined = " ".join(
        [
            _normalize(first.constraint),
            _normalize(first.risk),
            _normalize(first.guidance),
            _normalize(second.constraint),
            _normalize(second.risk),
            _normalize(second.guidance),
        ]
    )
    if "conflict" in joined or "contradict" in joined or "incompatible" in joined:
        return COMPOSITION_CONFLICT_THRESHOLD
    return 0.0


def _composition_has_unresolved_conflict(
    records: Sequence[InsightRecord],
    *,
    relation_scores: dict[frozenset[str], _CompositionPairScore],
) -> bool:
    return any(
        score.conflict >= COMPOSITION_CONFLICT_THRESHOLD
        for score in _composition_pair_scores(records, relation_scores)
    )


def _jaccard(first: set[str], second: set[str]) -> float:
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def _combined_insight_for_challenge(
    challenge: ChallengeRecord,
    insights: Sequence[InsightRecord],
) -> InsightRecord:
    return InsightRecord(
        id=challenge.insight_id,
        title="Compositional challenge insight",
        trigger=" ".join(insight.trigger for insight in insights),
        goal=" ".join(insight.goal for insight in insights),
        constraint=" ".join(insight.constraint for insight in insights),
        risk=" ".join(insight.risk for insight in insights),
        guidance=" ".join(insight.guidance for insight in insights),
        tags=tuple(tag for insight in insights for tag in insight.tags),
        keywords=tuple(keyword for insight in insights for keyword in insight.keywords),
        source="composition",
    )


def _composition_relation_record(
    insights: Sequence[InsightRecord],
    event: dict,
    *,
    reviewer: StructuredCompositionRelationReviewer | None = None,
) -> InsightRelationRecord | None:
    if len(insights) != 2:
        return None
    if event.get("interaction_dependency_valid") is False:
        return None
    if event.get("translator_replay_dependency_valid") is False:
        return None
    tool_evidence = event.get("tool_evidence") if isinstance(event.get("tool_evidence"), dict) else {}
    if not event.get("sanity_valid") or not tool_evidence.get("compiles") or not tool_evidence.get("tests_pass"):
        return None
    if reviewer is not None:
        return reviewer.review(insights, event)
    hinted = _heuristic_composition_relation_record(insights, event)
    if hinted is not None:
        return hinted
    first, second = insights
    fingerprint = str(event.get("fingerprint", ""))
    return InsightRelationRecord(
        first_id=first.id,
        second_id=second.id,
        relation="supports",
        rationale=(
            "Pairwise composition challenge passed sanity, compile, and test replay gates."
            + (
                " Translator replay dependency also passed."
                if event.get("translator_replay_dependency_valid") is True
                else ""
            )
        ),
        evidence=f"challenge_fingerprint:{fingerprint}",
    )


def _heuristic_composition_relation_record(
    insights: Sequence[InsightRecord],
    event: dict,
) -> InsightRelationRecord | None:
    if len(insights) != 2:
        return None
    for source, target in ((insights[0], insights[1]), (insights[1], insights[0])):
        source_text = _normalize(_insight_relation_text(source))
        target_matches = sorted(
            term for term in _composition_dependency_terms(target) if _whole_term_in_text(term, source_text)
        )[:8]
        if not target_matches:
            continue
        relation = ""
        marker = ""
        if _relation_marker_present(source_text, ("requires", "require", "depends on", "precondition", "needs")):
            relation = "requires"
            marker = "requires"
        elif _relation_marker_present(source_text, ("before", "precedes", "precede", "prior to")):
            relation = "precedes"
            marker = "before"
        if not relation:
            continue
        return InsightRelationRecord(
            first_id=source.id,
            second_id=target.id,
            relation=relation,
            rationale=(
                f"Local composition relation hint '{marker}' in {source.id} matched "
                f"{target.id} terms: {', '.join(target_matches)}."
            ),
            evidence=f"challenge_fingerprint:{str(event.get('fingerprint', ''))}",
        )
    return None


def _insight_relation_text(insight: InsightRecord) -> str:
    return " ".join(
        [
            insight.title,
            insight.trigger,
            insight.goal,
            insight.constraint,
            insight.risk,
            insight.guidance,
        ]
    )


def _relation_marker_present(text: str, markers: Sequence[str]) -> bool:
    return any(_whole_term_in_text(marker, text) for marker in markers)


def _whole_term_in_text(term: str, text: str) -> bool:
    normalized = _normalize(term)
    return re.search(rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])", text) is not None


def _composition_interaction_dependency(
    insights: Sequence[InsightRecord],
    challenge: ChallengeRecord,
    *,
    tool_checker: ChallengeToolChecker | None = None,
) -> dict:
    text = _normalize(
        " ".join(
            [
                challenge.oracle,
                challenge.harness,
                challenge.rust_interface,
                challenge.activation_predicate,
            ]
        )
    )
    matches = {
        insight.id: sorted(term for term in _composition_dependency_terms(insight) if term in text)[:8]
        for insight in insights
    }
    static_valid = len(insights) >= 2 and all(matches.get(insight.id) for insight in insights)
    dependency = {
        "valid": static_valid,
        "matches": matches,
    }
    executable = _composition_executable_dependency(challenge, tool_checker) if static_valid else None
    if executable is not None:
        dependency["executable_dependency_valid"] = executable["valid"]
        dependency["executable_dependency_probes"] = executable["probes"]
        dependency["valid"] = static_valid and executable["valid"]
    return dependency


def _composition_executable_dependency(
    challenge: ChallengeRecord,
    tool_checker: ChallengeToolChecker | None,
) -> dict | None:
    if tool_checker is None:
        return None
    probes = challenge.trace_bundle.get("composition_dependency_probes")
    if not isinstance(probes, dict):
        return None
    reference_source = probes.get("reference_source")
    mutants = probes.get("mutants")
    if not isinstance(reference_source, str) or not isinstance(mutants, list):
        return None

    probe_results = []
    reference_checked = tool_checker.check(challenge, rust_source=reference_source)
    reference_passed = reference_checked.evidence.compiles and reference_checked.evidence.tests_pass
    probe_results.append(
        {
            "kind": "reference",
            "insight_id": None,
            "compiles": reference_checked.evidence.compiles,
            "tests_pass": reference_checked.evidence.tests_pass,
        }
    )

    mutant_failures = []
    for mutant in mutants:
        if not isinstance(mutant, dict):
            continue
        rust_source = mutant.get("rust_source")
        insight_id = mutant.get("insight_id")
        if not isinstance(rust_source, str):
            continue
        checked = tool_checker.check(challenge, rust_source=rust_source)
        failed_as_expected = checked.evidence.compiles and not checked.evidence.tests_pass
        mutant_failures.append(failed_as_expected)
        probe_results.append(
            {
                "kind": str(mutant.get("kind", "mutant")),
                "insight_id": str(insight_id) if insight_id is not None else None,
                "compiles": checked.evidence.compiles,
                "tests_pass": checked.evidence.tests_pass,
                "failed_as_expected": failed_as_expected,
            }
        )

    return {
        "valid": reference_passed and bool(mutant_failures) and all(mutant_failures),
        "probes": probe_results,
    }


def _composition_translator_replay_dependency(
    insights: Sequence[InsightRecord],
    challenge: ChallengeRecord,
    *,
    tool_checker: ChallengeToolChecker | None,
    replay_translate_fn: Callable[[ChallengeRecord, InsightRecord], str] | None,
) -> dict | None:
    if tool_checker is None or replay_translate_fn is None or len(insights) < 2:
        return None

    probe_results: list[dict] = []
    combined_insight = _combined_insight_for_challenge(challenge, insights)

    try:
        combined_source = replay_translate_fn(challenge, combined_insight)
        combined_checked = tool_checker.check(challenge, rust_source=combined_source)
        combined_passed = combined_checked.evidence.compiles and combined_checked.evidence.tests_pass
        probe_results.append(
            {
                "kind": "translator_combined",
                "insight_id": combined_insight.id,
                "compiles": combined_checked.evidence.compiles,
                "tests_pass": combined_checked.evidence.tests_pass,
            }
        )
    except Exception as exc:
        probe_results.append(
            {
                "kind": "translator_combined",
                "insight_id": combined_insight.id,
                "error": str(exc),
            }
        )
        return {"valid": False, "probes": probe_results}

    single_failures = []
    for insight in insights:
        try:
            single_source = replay_translate_fn(challenge, insight)
            checked = tool_checker.check(challenge, rust_source=single_source)
            failed_as_expected = checked.evidence.compiles and not checked.evidence.tests_pass
            single_failures.append(failed_as_expected)
            probe_results.append(
                {
                    "kind": "translator_single",
                    "insight_id": insight.id,
                    "compiles": checked.evidence.compiles,
                    "tests_pass": checked.evidence.tests_pass,
                    "failed_as_expected": failed_as_expected,
                }
            )
        except Exception as exc:
            single_failures.append(False)
            probe_results.append(
                {
                    "kind": "translator_single",
                    "insight_id": insight.id,
                    "error": str(exc),
                    "failed_as_expected": False,
                }
            )

    return {
        "valid": combined_passed and len(single_failures) == len(insights) and all(single_failures),
        "probes": probe_results,
    }


def _composition_dependency_terms(insight: InsightRecord) -> set[str]:
    stopwords = {
        "and",
        "the",
        "must",
        "with",
        "from",
        "this",
        "that",
        "when",
        "keep",
        "preserve",
        "translation",
        "insight",
        "risk",
        "goal",
        "trigger",
        "constraint",
    }
    raw_terms = [
        *insight.tags,
        *insight.keywords,
        *_keyword_candidates(
            " ".join([insight.trigger, insight.goal, insight.constraint, insight.risk])
        ),
    ]
    terms: set[str] = set()
    for raw in raw_terms:
        for token in _tokens(str(raw)):
            if len(token) > 2 and token not in stopwords:
                terms.add(token)
    return terms


def _has_relation(records: Sequence[InsightRelationRecord], candidate: InsightRelationRecord) -> bool:
    return any(
        record.first_id == candidate.first_id
        and record.second_id == candidate.second_id
        and record.relation == candidate.relation
        for record in records
    )


def _iter_project_artifacts(run_root: Path):
    if not run_root.exists():
        return
    if _has_project_artifacts(run_root):
        yield run_root
    for project_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        if _has_project_artifacts(project_dir):
            yield project_dir


def _has_project_artifacts(path: Path) -> bool:
    return any(
        (path / name).exists()
        for name in (
            "translation_report.json",
            "translation_history.json",
            "test_fix_history.json",
            "llm_calls.jsonl",
        )
    )


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return default


def _load_jsonl(path: Path) -> list[object]:
    if not path.exists():
        return []
    records: list[object] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError:
            continue
    return records


def _insight_ids_from_record(record: dict) -> list[str]:
    raw_items = record.get("selected_insights") or record.get("insights") or []
    ids: list[str] = []
    seen = set()
    for item in raw_items:
        insight_id = ""
        if isinstance(item, dict):
            insight_id = str(item.get("id", ""))
        elif isinstance(item, str):
            insight_id = item
        if insight_id and insight_id not in seen:
            seen.add(insight_id)
            ids.append(insight_id)
    return ids


def _trace_bundle_from_test_record(
    project_dir: Path,
    record: dict,
    translation_history: object,
) -> dict[str, object]:
    bundle: dict[str, object] = {"project": project_dir.name}
    for key in (
        "callable_id",
        "source",
        "project",
        "selected_insights",
        "test_output",
        "test_stats",
        "post_patch_test_output",
        "post_patch_test_stats",
        "test_improved",
        "c_context",
        "c_code",
        "rust_context",
        "callable_context",
        "generated_rust",
        "compiler_errors",
        "failing_assertions",
    ):
        if key in record:
            bundle[key] = record[key]
    translation_trace = _translation_trace_for_record(record, translation_history)
    if translation_trace:
        bundle.update(translation_trace)
    return bundle


def _compile_trace_records_from_translation_history(
    project_dir: Path,
    translation_history: object,
) -> list[dict]:
    if not isinstance(translation_history, dict):
        return []
    records: list[dict] = []
    for entry_key, entry in translation_history.items():
        if not isinstance(entry, dict):
            continue
        attempts = entry.get("compilation_attempts")
        if not isinstance(attempts, list):
            continue
        callable_id = str(entry.get("callable_id") or entry_key)
        entry_insight_ids = _insight_ids_from_record(entry)
        for attempt_index, attempt in enumerate(attempts):
            if not isinstance(attempt, dict) or attempt.get("compilation_success"):
                continue
            output = str(attempt.get("compilation_output") or "").strip()
            compiler_errors = _compiler_errors_from_compile_attempt(attempt)
            evidence_text = output or "\n".join(compiler_errors)
            if not evidence_text.strip():
                continue
            selected_insights = attempt.get("selected_insights")
            if not selected_insights and entry_insight_ids:
                selected_insights = [{"id": insight_id} for insight_id in entry_insight_ids]
            record = {
                "source": "translation_history",
                "project": project_dir.name,
                "callable_id": callable_id,
                "callable_label": entry.get("callable_label") or entry.get("module_name") or callable_id,
                "selected_insights": selected_insights or [],
                "c_context": _c_program_from_translation_entry(entry, attempt),
                "test_output": evidence_text,
                "test_stats": {"failed": 1, "passed": 0},
                "compiler_errors": compiler_errors,
                "compilation_attempt": attempt,
            }
            generated_rust = attempt.get("code") or attempt.get("fixed_code") or entry.get("final_code")
            if isinstance(generated_rust, str) and generated_rust.strip():
                record["generated_rust"] = generated_rust
            replay_attempt = _later_successful_compile_attempt(attempts, attempt_index)
            if replay_attempt is not None:
                record["replay_compilation_attempt"] = replay_attempt
                record["replay_compilation_output"] = str(
                    replay_attempt.get("compilation_output") or ""
                )
            if "c_binding" in entry:
                record["c_binding"] = entry["c_binding"]
            if isinstance(attempts, list):
                record["compilation_attempts"] = attempts
            records.append(record)
    return records


def _compile_challenges_from_translation_history(
    project_dir: Path,
    translation_history: object,
    known_ids: set[str],
) -> list[ChallengeRecord]:
    challenges: list[ChallengeRecord] = []
    for record in _compile_trace_records_from_translation_history(project_dir, translation_history):
        evidence_text = str(record.get("test_output") or "").strip()
        if not evidence_text:
            continue
        compiler_errors = [
            str(item)
            for item in record.get("compiler_errors", [])
            if str(item).strip()
        ]
        for insight_id in _insight_ids_from_record(record):
            if insight_id not in known_ids:
                continue
            challenges.append(
                ChallengeRecord(
                    insight_id=insight_id,
                    c_program=str(record.get("c_context") or ""),
                    rust_interface=str(
                        record.get("callable_label")
                        or record.get("callable_id")
                        or project_dir.name
                    ),
                    harness=evidence_text[:2000],
                    oracle="\n".join(compiler_errors or [evidence_text])[:2000],
                    evidence=ExecutionEvidence(
                        compiles=False,
                        tests_pass=False,
                        output=evidence_text,
                    ),
                    replay_evidence=_replay_evidence_from_compile_trace_record(record),
                    trace_bundle=dict(record),
                )
            )
    return challenges


def _llm_call_challenges_from_project(
    project_dir: Path,
    known_ids: set[str],
) -> list[ChallengeRecord]:
    challenges: list[ChallengeRecord] = []
    for call in _load_jsonl(project_dir / "llm_calls.jsonl"):
        if not isinstance(call, dict):
            continue
        output_issue = str(call.get("output_issue") or "").strip()
        if not output_issue:
            continue
        evidence_text = _llm_call_evidence_text(call, output_issue)
        if not evidence_text:
            continue
        for insight_id in _insight_ids_from_record(call):
            if insight_id not in known_ids:
                continue
            challenges.append(
                ChallengeRecord(
                    insight_id=insight_id,
                    c_program=str(call.get("prompt") or ""),
                    rust_interface=str(
                        call.get("callable_label")
                        or call.get("callable_id")
                        or call.get("name")
                        or project_dir.name
                    ),
                    harness=str(call.get("raw_text") or call.get("extracted_code") or "")[:2000],
                    oracle=output_issue[:2000],
                    evidence=ExecutionEvidence(
                        compiles=False,
                        tests_pass=False,
                        output=evidence_text,
                    ),
                    trace_bundle=_trace_bundle_from_llm_call(project_dir, call, output_issue),
                )
            )
    return challenges


def _llm_call_evidence_text(call: dict, output_issue: str) -> str:
    parts = [f"output_issue: {output_issue}"]
    phase = str(call.get("phase") or "").strip()
    if phase:
        parts.append(f"phase: {phase}")
    raw_text = str(call.get("raw_text") or "").strip()
    if raw_text:
        parts.append(_truncate(raw_text, 500))
    return "\n".join(parts).strip()


def _trace_bundle_from_llm_call(
    project_dir: Path,
    call: dict,
    output_issue: str,
) -> dict[str, object]:
    bundle: dict[str, object] = {
        "source": "llm_calls",
        "project": project_dir.name,
        "output_issue": output_issue,
    }
    for key in (
        "callable_id",
        "callable_label",
        "module",
        "kind",
        "type_name",
        "name",
        "phase",
        "prompt",
        "raw_text",
        "extracted_code",
        "insights",
        "dependencies",
        "alignment",
    ):
        if key in call:
            bundle[key] = call[key]
    return bundle


def _later_successful_compile_attempt(
    attempts: list,
    failed_attempt_index: int,
) -> dict | None:
    for attempt in attempts[failed_attempt_index + 1 :]:
        if isinstance(attempt, dict) and attempt.get("compilation_success"):
            return attempt
    return None


def _replay_evidence_from_compile_trace_record(record: dict) -> ExecutionEvidence | None:
    replay_attempt = record.get("replay_compilation_attempt")
    if not isinstance(replay_attempt, dict):
        return None
    output = str(
        record.get("replay_compilation_output")
        or replay_attempt.get("compilation_output")
        or ""
    )
    return ExecutionEvidence(
        compiles=bool(replay_attempt.get("compilation_success")),
        tests_pass=False,
        output=output,
    )


def _c_program_from_translation_entry(entry: dict, attempt: dict) -> str:
    for source in (attempt, entry):
        for key in ("c_context", "c_code"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value
    binding = entry.get("c_binding")
    if isinstance(binding, dict):
        functions = binding.get("functions")
        if isinstance(functions, list):
            chunks: list[str] = []
            for function in functions:
                if not isinstance(function, dict):
                    continue
                code = function.get("code")
                if isinstance(code, str) and code.strip():
                    chunks.append(code.strip())
                else:
                    name = str(function.get("name") or "").strip()
                    if name:
                        chunks.append(name)
            if chunks:
                return "\n\n".join(chunks)
    return str(entry.get("callable_label") or entry.get("callable_id") or "")


def _compiler_errors_from_compile_attempt(attempt: dict) -> list[str]:
    raw_errors = attempt.get("errors")
    errors: list[str] = []
    if isinstance(raw_errors, list):
        for item in raw_errors:
            if isinstance(item, dict):
                text = " ".join(
                    str(item.get(key) or "").strip()
                    for key in ("message", "excerpt")
                    if item.get(key)
                ).strip()
            else:
                text = str(item).strip()
            if text:
                errors.append(text)
    if errors:
        return errors
    output = str(attempt.get("compilation_output") or "").strip()
    return [output] if output else []


def _translation_trace_for_record(record: dict, translation_history: object) -> dict[str, object]:
    if not isinstance(translation_history, dict) or not translation_history:
        return {}
    callable_id = str(record.get("callable_id", ""))
    selected_entry = None
    if callable_id:
        selected_entry = translation_history.get(callable_id)
        if selected_entry is None:
            for key, value in translation_history.items():
                if str(key).endswith(callable_id):
                    selected_entry = value
                    break
    if selected_entry is None and len(translation_history) == 1:
        selected_entry = next(iter(translation_history.values()))
    if not isinstance(selected_entry, dict):
        return {}
    trace: dict[str, object] = {}
    if selected_entry.get("final_code"):
        trace["generated_rust"] = selected_entry["final_code"]
    attempts = selected_entry.get("compilation_attempts")
    if isinstance(attempts, list):
        trace["compilation_attempts"] = attempts
        compiler_errors = [
            str(item.get("compilation_output", ""))
            for item in attempts
            if isinstance(item, dict) and not item.get("compilation_success")
        ]
        if compiler_errors:
            trace["compiler_errors"] = compiler_errors
    return trace


def _evidence_from_test_record(record: dict) -> ExecutionEvidence:
    stats = record.get("test_stats") or {}
    failed = int(stats.get("failed") or 0)
    total = int(stats.get("total") or failed or stats.get("passed") or 0)
    return ExecutionEvidence(
        compiles=not _looks_like_compile_failure(str(record.get("test_output", ""))),
        tests_pass=total > 0 and failed == 0,
        output=str(record.get("test_output", "")),
    )


def _replay_evidence_from_test_record(record: dict) -> ExecutionEvidence | None:
    if "post_patch_test_stats" not in record and "post_patch_test_output" not in record:
        if record.get("test_improved"):
            return ExecutionEvidence(compiles=True, tests_pass=True, output="test_improved")
        return None
    stats = record.get("post_patch_test_stats") or {}
    failed = int(stats.get("failed") or 0)
    total = int(stats.get("total") or failed or stats.get("passed") or 0)
    tests_pass = (total > 0 and failed == 0) or bool(record.get("test_improved"))
    return ExecutionEvidence(
        compiles=not _looks_like_compile_failure(str(record.get("post_patch_test_output", ""))),
        tests_pass=tests_pass,
        output=str(record.get("post_patch_test_output", "")),
    )


def _looks_like_compile_failure(output: str) -> bool:
    lowered = output.lower()
    return "could not compile" in lowered or "error:" in lowered and "test result" not in lowered


_CONTRADICTION_NEGATORS = (
    "do not",
    "don't",
    "dont",
    "must not",
    "should not",
    "cannot",
    "can't",
    "cant",
    "never",
    "avoid",
)

_CONTRADICTION_ACTION_ALIASES = {
    "recomputes": "recompute",
    "recomputed": "recompute",
    "recomputing": "recompute",
    "recalculates": "recalculate",
    "recalculated": "recalculate",
    "recalculating": "recalculate",
    "derives": "derive",
    "derived": "derive",
    "deriving": "derive",
    "infers": "infer",
    "inferred": "infer",
    "inferring": "infer",
    "trims": "trim",
    "trimmed": "trim",
    "trimming": "trim",
    "strips": "strip",
    "stripped": "strip",
    "stripping": "strip",
    "truncates": "truncate",
    "truncated": "truncate",
    "truncating": "truncate",
    "drops": "drop",
    "dropped": "drop",
    "dropping": "drop",
    "removes": "remove",
    "removed": "remove",
    "removing": "remove",
    "discards": "discard",
    "discarded": "discard",
    "discarding": "discard",
    "ignores": "ignore",
    "ignored": "ignore",
    "ignoring": "ignore",
    "overwrites": "overwrite",
    "overwritten": "overwrite",
    "overwriting": "overwrite",
    "copies": "copy",
    "copied": "copy",
    "copying": "copy",
    "allocates": "allocate",
    "allocated": "allocate",
    "allocating": "allocate",
    "frees": "free",
    "freed": "free",
    "freeing": "free",
    "panics": "panic",
    "panicked": "panic",
    "panicking": "panic",
    "unwraps": "unwrap",
    "unwrapped": "unwrap",
    "unwrapping": "unwrap",
    "assumes": "assume",
    "assumed": "assume",
    "assuming": "assume",
}

_CONTRADICTION_ACTIONS = {
    "recompute",
    "recalculate",
    "derive",
    "infer",
    "trim",
    "strip",
    "truncate",
    "drop",
    "remove",
    "discard",
    "ignore",
    "overwrite",
    "copy",
    "allocate",
    "free",
    "panic",
    "unwrap",
    "assume",
}

_CONTRADICTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "when",
    "while",
    "with",
}

_CONTRADICTION_DIRECTIVE_MARKERS = (
    "suggest",
    "suggests",
    "must",
    "should",
    "need",
    "needs",
    "ensure",
    "ensures",
    "preserve",
    "preserves",
    "keep",
    "keeps",
    "use",
    "uses",
)

_FAILURE_ONLY_MARKERS = (
    "assertion failed",
    "panicked",
    "test failed",
    "mismatch",
    "changed from",
    "error:",
)


def _contradiction_rules_from_record(record: InsightRecord) -> dict[str, set[tuple[str, ...]]]:
    text = _normalize_record_text(record)
    return {"forbidden": _negated_action_signatures(text)}


def _negated_action_signatures(text: str) -> set[tuple[str, ...]]:
    signatures: set[tuple[str, ...]] = set()
    for negator in _CONTRADICTION_NEGATORS:
        pattern = rf"\b{re.escape(negator)}\s+([^.;:\n]{{1,120}})"
        for match in re.finditer(pattern, text):
            phrase = re.split(r"\b(?:because|unless|when|while|but|and)\b", match.group(1))[0]
            signature = _action_signature(phrase)
            if signature:
                signatures.add(signature)
    return signatures


def _rules_contradict_text(forbidden_signatures: set[tuple[str, ...]], text: str) -> bool:
    if not forbidden_signatures:
        return False
    for segment in re.split(r"[.;\n]", text):
        normalized_segment = _normalize(segment)
        if not normalized_segment.strip():
            continue
        if _segment_has_negation(normalized_segment):
            continue
        if _looks_like_failure_only_segment(normalized_segment):
            continue
        segment_tokens = set(_canonical_action_token(token) for token in _tokens(normalized_segment))
        for signature in forbidden_signatures:
            if _action_signature_matches(signature, segment_tokens):
                return True
    return False


def _segment_has_negation(text: str) -> bool:
    return any(f" {negator} " in f" {text} " for negator in _CONTRADICTION_NEGATORS) or bool(
        re.search(r"\bnot\b", text)
    )


def _looks_like_failure_only_segment(text: str) -> bool:
    if not any(marker in text for marker in _FAILURE_ONLY_MARKERS):
        return False
    return not any(marker in text for marker in _CONTRADICTION_DIRECTIVE_MARKERS)


def _action_signature(phrase: str) -> tuple[str, ...]:
    signature_tokens: list[str] = []
    for token in _tokens(_normalize(phrase)):
        canonical = _canonical_action_token(token)
        if len(canonical) < 3 or canonical in _CONTRADICTION_STOPWORDS:
            continue
        if canonical in signature_tokens:
            continue
        signature_tokens.append(canonical)
        if len(signature_tokens) >= 5:
            break
    if not any(token in _CONTRADICTION_ACTIONS for token in signature_tokens):
        return ()
    if len(signature_tokens) < 2:
        return ()
    return tuple(signature_tokens)


def _action_signature_matches(signature: tuple[str, ...], segment_tokens: set[str]) -> bool:
    actions = {token for token in signature if token in _CONTRADICTION_ACTIONS}
    domain_tokens = {token for token in signature if token not in _CONTRADICTION_ACTIONS}
    if not actions or not actions.issubset(segment_tokens):
        return False
    if not domain_tokens:
        return True
    required_domain_matches = min(2, len(domain_tokens))
    return len(domain_tokens & segment_tokens) >= required_domain_matches


def _canonical_action_token(token: str) -> str:
    return _CONTRADICTION_ACTION_ALIASES.get(token, token)


def _normalize_record_text(record: InsightRecord) -> str:
    return _normalize(
        " ".join(
            [
                record.trigger,
                record.goal,
                record.constraint,
                record.risk,
                record.guidance,
                " ".join(record.tags),
                " ".join(record.keywords),
            ]
        )
    )


def _candidate_fingerprint(candidate: InsightRecord) -> str:
    return hashlib.sha256(
        "\n".join([candidate.trigger, candidate.constraint, candidate.risk]).encode("utf-8")
    ).hexdigest()


def _keyword_candidates(text: str) -> list[str]:
    keywords: list[str] = []
    seen: set[str] = set()
    for token in _tokens(_normalize(text)):
        if len(token) < 4 or token in seen:
            continue
        seen.add(token)
        keywords.append(token)
    return keywords


def _failure_signature(text: str) -> str:
    normalized = _normalize(text)
    normalized = normalized.replace("account for failing evidence:", " ")
    normalized = normalized.replace("without this insight translations may reproduce:", " ")
    normalized = normalized.replace("assertion failed:", " ")
    return re.sub(r"\s+", " ", normalized).strip()


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", _normalize(text)).strip("-")
    return slug or "unknown"


def _truncate(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."
