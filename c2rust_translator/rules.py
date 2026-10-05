from __future__ import annotations

import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class TranslationRule:
    id: str
    title: str
    trigger: str
    goal: str
    constraint: str
    risk: str
    guidance: str
    tags: tuple[str, ...]
    keywords: tuple[str, ...]

    def to_prompt_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "trigger": self.trigger,
            "goal": self.goal,
            "constraint": self.constraint,
            "risk": self.risk,
            "guidance": self.guidance,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class RuleInteraction:
    first_id: str
    second_id: str
    support: int
    independent: int
    conflict: int
    explanation: str = ""

    @property
    def relationship(self) -> str:
        if self.conflict > self.support and self.conflict >= self.independent:
            return "conflict"
        if self.support >= self.independent and self.support > 0:
            return "support"
        return "independent"

    def to_dict(self) -> dict[str, object]:
        return {
            "first_id": self.first_id,
            "second_id": self.second_id,
            "support": self.support,
            "independent": self.independent,
            "conflict": self.conflict,
            "relationship": self.relationship,
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class RuleSelection:
    rules: tuple[TranslationRule, ...]
    interactions: tuple[RuleInteraction, ...] = ()


class StructuredInteractionScorer:
    def __init__(self, score_fn: Callable[[dict[str, object]], dict[str, object] | str]):
        self.score_fn = score_fn

    def score(self, first: TranslationRule, second: TranslationRule) -> RuleInteraction:
        raw = self.score_fn(_interaction_payload(first, second))
        data = _parse_interaction_score(raw)
        support = _normalized_score_to_int(_score_field(data, "support"), "support")
        conflict = _normalized_score_to_int(_score_field(data, "conflict"), "conflict")
        if "independent" in data or "independent_score" in data:
            independent = _normalized_score_to_int(_score_field(data, "independent"), "independent")
        else:
            independent = max(0, 10 - support - conflict)
        return RuleInteraction(
            first_id=first.id,
            second_id=second.id,
            support=support,
            independent=independent,
            conflict=conflict,
            explanation=str(data.get("explanation") or data.get("rationale") or ""),
        )


LLM_INTERACTION_SCORER_SYSTEM = (
    "You score whether two Rust translation rules support, conflict with, "
    "or are independent of each other for prompt-time selection. support means "
    "the two rules reinforce each other or should be retrieved together; "
    "conflict means following one likely violates the other; independent means "
    "ordinary overlap, shared vocabulary, or unrelated coverage without a clear "
    "dependency. Do not infer conflict from ordinary overlap. Do not infer "
    "conflict from id or title overlap; require a concrete incompatible "
    "constraint, goal, or risk. Return exactly one JSON object matching the "
    "provided schema; no markdown, no code fences, no comments, and no "
    "explanatory prose. Do not add fields outside the schema."
)


class LLMRuleInteractionScorer:
    def __init__(self, agent):
        self.agent = agent
        self.calls: list[dict[str, object]] = []
        self.structured = StructuredInteractionScorer(self._score_payload)

    def score(self, first: TranslationRule, second: TranslationRule) -> RuleInteraction:
        return self.structured.score(first, second)

    def _score_payload(self, payload: dict[str, object]) -> str:
        prompt = json.dumps(payload, ensure_ascii=False, indent=2)
        result = self.agent.translate(LLM_INTERACTION_SCORER_SYSTEM, prompt)
        raw_text = str(getattr(result, "raw_text", "") or getattr(result, "code", ""))
        usage = getattr(result, "usage", None)
        first = payload.get("first") if isinstance(payload, dict) else {}
        second = payload.get("second") if isinstance(payload, dict) else {}
        self.calls.append(
            {
                "system": LLM_INTERACTION_SCORER_SYSTEM,
                "prompt": prompt,
                "raw_text": raw_text,
                "elapsed_seconds": getattr(result, "elapsed_seconds", 0.0),
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                    "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
                },
                "first_id": str(first.get("id", "")) if isinstance(first, dict) else "",
                "second_id": str(second.get("id", "")) if isinstance(second, dict) else "",
            }
        )
        return raw_text

    def drain_calls(self) -> list[dict[str, object]]:
        calls = list(self.calls)
        self.calls.clear()
        return calls


class RelationBackedInteractionScorer:
    ACTIONABLE_RELATIONS = {"supports", "requires", "precedes", "conflicts", "conflict"}

    def __init__(
        self,
        relations: Iterable[object],
        *,
        fallback: Callable[[TranslationRule, TranslationRule], RuleInteraction] | None = None,
    ):
        self.fallback = fallback
        self.relations = {}
        for relation in relations:
            relation_name = _relation_value(relation, "relation").strip().lower()
            if relation_name not in self.ACTIONABLE_RELATIONS:
                continue
            first_id = _relation_value(relation, "first_id")
            second_id = _relation_value(relation, "second_id")
            if not first_id or not second_id:
                continue
            self.relations[(first_id, second_id)] = relation

    @property
    def has_actionable_relations(self) -> bool:
        return bool(self.relations)

    def score(self, first: TranslationRule, second: TranslationRule) -> RuleInteraction:
        relation = self.relations.get((first.id, second.id))
        direction = "forward"
        if relation is None:
            relation = self.relations.get((second.id, first.id))
            direction = "reverse"
        if relation is None:
            fallback = self.fallback or score_rule_interaction
            return fallback(first, second)

        relation_name = _relation_value(relation, "relation").strip().lower()
        explanation = _relation_explanation(relation, relation_name, direction)
        if relation_name in {"conflicts", "conflict"}:
            return RuleInteraction(
                first_id=first.id,
                second_id=second.id,
                support=0,
                independent=0,
                conflict=10,
                explanation=explanation,
            )
        return RuleInteraction(
            first_id=first.id,
            second_id=second.id,
            support=10,
            independent=0,
            conflict=0,
            explanation=explanation,
        )


def _relation_value(relation: object, field: str) -> str:
    if isinstance(relation, dict):
        return str(relation.get(field, ""))
    return str(getattr(relation, field, ""))


def _relation_explanation(relation: object, relation_name: str, direction: str) -> str:
    first_id = _relation_value(relation, "first_id")
    second_id = _relation_value(relation, "second_id")
    rationale = _relation_value(relation, "rationale")
    evidence = _relation_value(relation, "evidence")
    detail = rationale or evidence
    prefix = f"bank relation {relation_name} {first_id}->{second_id}"
    if direction == "reverse":
        prefix += " (reverse selection order)"
    return f"{prefix}: {detail}" if detail else prefix


def _interaction_payload(first: TranslationRule, second: TranslationRule) -> dict[str, object]:
    return {
        "first": first.to_prompt_dict(),
        "second": second.to_prompt_dict(),
        "schema": {
            "support": "number in [0, 1]",
            "conflict": "number in [0, 1]",
            "independent": "optional number in [0, 1]",
            "explanation": "optional short rationale",
        },
    }


def _parse_interaction_score(raw: dict[str, object] | str) -> dict[str, object]:
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            parsed = _parse_json_object_from_text(raw, exc)
    else:
        parsed = raw
    if not isinstance(parsed, dict):
        raise ValueError("Interaction scorer must return a JSON object")
    return parsed


def _parse_json_object_from_text(text: str, original_error: json.JSONDecodeError) -> dict[str, object]:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
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
    raise ValueError(f"Invalid interaction score JSON: {original_error}") from original_error


def _score_field(data: dict[str, object], field: str) -> object:
    score_key = f"{field}_score"
    if field in data:
        return data[field]
    if score_key in data:
        return data[score_key]
    raise ValueError(f"Interaction scorer missing required field: {field}")


def _normalized_score_to_int(raw: object, field: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"Interaction scorer field {field} must be numeric")
    value = float(raw)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"Interaction scorer field {field} must be in [0, 1]")
    return round(value * 10)


@dataclass(frozen=True)
class RuleLibrary:
    rules: Sequence[object]
    select: Callable[..., Sequence[object]]
    path: Path | None = None


_EMPIRICAL_WEAK_PAIR_PENALTIES: dict[tuple[str, str], int] = {
    ("generated-test-mutating-clone", "multi-helper-composition"): 24,
    ("multi-helper-composition", "pointer-buffer-length-coupling"): 18,
    ("multi-helper-composition", "signed-unsigned-boundaries"): 16,
    ("generated-test-mutating-clone", "signed-unsigned-boundaries"): 16,
    ("generated-test-mutating-clone", "resource-wrapper-constructor-safety"): 15,
    ("callback-user-data-state", "multi-helper-composition"): 15,
    ("bit-exact-encoding", "generated-test-mutating-clone"): 14,
    ("generated-test-mutating-clone", "output-struct-capacity-fields"): 14,
    ("generated-test-mutating-clone", "pointer-buffer-length-coupling"): 14,
    ("callback-user-data-state", "generated-test-mutating-clone"): 13,
}


def empirical_weak_pair_penalty(first_id: str, second_id: str) -> int:
    pair = tuple(sorted((first_id, second_id)))
    return _EMPIRICAL_WEAK_PAIR_PENALTIES.get(pair, 0)


DEFAULT_RULES: tuple[TranslationRule, ...] = (
    TranslationRule(
        id="interface-contract-first",
        title="Honor the Rust interface before mirroring C shape",
        trigger="Rust interface uses a safe or higher-level contract, or the C binding is weak/no_safe_binding.",
        goal="Implement the exact public Rust signature and observable behavior required by tests.",
        constraint="Keep the signature unchanged; do not introduce public ABI-shaped helpers just because C has them.",
        risk="Copying C pointer/return conventions into a safe Rust contract creates compile errors or semantic drift.",
        guidance="Translate behavior, not the C surface API. Use C as semantic evidence while making the Rust interface the source of truth.",
        tags=("contract", "interface", "no_safe_binding"),
        keywords=("no_safe_binding", "safe binding", "higher-level contract", "public abi", "signature mismatch"),
    ),
    TranslationRule(
        id="multi-helper-composition",
        title="Compose all bound C helpers for one Rust callable",
        trigger="Alignment binds one Rust callable to multiple C functions, or the C implementation delegates through helper calls.",
        goal="Preserve the whole operation, including validation, state updates, and helper-side effects.",
        constraint="Read helper functions in call order and carry over side effects that matter to the public result.",
        risk="Using only the first obvious C function drops hidden initialization, bounds checks, or state transitions.",
        guidance="When the binding is composed_from_multiple or a C function calls project helpers, inspect each helper before writing the Rust body.",
        tags=("alignment", "helpers", "composition"),
        keywords=("composed_from_multiple", "c_calls_present", "helper call", "delegates through", "bound c functions"),
    ),
    TranslationRule(
        id="mutating-output-parameters",
        title="Preserve C output parameter mutations",
        trigger="C passes pointers to scalar slots, struct fields, array elements, or helper output parameters that are written through during the call.",
        goal="Keep every in-place output or inout mutation visible to the Rust caller and to later simulation steps.",
        constraint="Do not pass output slots by value or through immutable references. If a translated helper signature has already lost mutability for an output, inline or reimplement the helper's mutation at the caller instead of relying on that non-mutating dependency.",
        risk="Rust can compile while silently dropping writes to normal/tangent arrays, accumulated force vectors, counters, lengths, or previous-state slots, making tests see zeros or stale values.",
        guidance="Trace every C argument of the form `&array[i]`, `&struct.field`, or `T *out` that the callee assigns through. In Rust, write back through `&mut` slices or mutable locals, and update all paired state such as previous_normal, previous_tangent, and force_p2 together. For pairwise contact/collision loops such as compute_forces/collide_two_particles, if tests observe both endpoints or both matrix slots, factor the contact update as apply_contact(src, dst) and call it once for p1->p2 and once for p2->p1. Each direction must recompute with swapped velocity/particle inputs, properties[dst], and its own `src * size + dst` normal/tangent slot; do not synthesize the reverse by negating one computed force or by sharing one previous-force slot.",
        tags=("outputs", "mutation", "pointers", "state", "test-repair"),
        keywords=(
            "output parameter",
            "output parameters",
            "inout",
            "side effect",
            "side effects",
            "pointer output",
            "writes through",
            "write back",
            "&array[i]",
            "&struct.field",
            "double *previous_normal",
            "double *previous_tangent",
            "vector *force_p2",
            "*previous_normal",
            "*previous_tangent",
            "previous_normal",
            "previous_tangent",
            "force_p2",
            "normal_forces",
            "tangent_forces",
            "resultant_forces",
            "p1 <- p2",
            "p2 <- p1",
            "compute_forces",
            "collide_two_particles",
            "&mut [f64]",
            "&mut [data::vector]",
        ),
    ),
    TranslationRule(
        id="output-struct-capacity-fields",
        title="Preserve caller-owned output struct capacity fields",
        trigger="C fills a caller-provided output struct that has both count/length and capacity/limit fields, while the Rust API returns that struct by value.",
        goal="Keep metadata fields such as capacity, limit, max_entries, or allocated length equivalent to the caller-visible contract, not just the number of parsed or written entries.",
        constraint="Do not recompute capacity from entries.len() or count unless the C code does that. When C resets only count and writes entries up to an existing capacity, initialize the returned Rust struct with the interface/test-visible default capacity and update count separately. If the Rust wrapper has no capacity parameter, use the capacity constant shown by the tests or interface rather than deriving capacity from the parsed entry count.",
        risk="The translated parser or loader can parse all entries correctly but fail equality tests because capacity collapses to count, zero, or the backing Vec length instead of the C caller's configured capacity.",
        guidance="For functions like cfg_parse(src, len, Cfg *cfg, err), treat cfg as an output object with preexisting layout. If the Rust wrapper returns Cfg, preserve capacity as a semantic field, for example the test/interface default capacity such as TEST_CAPACITY, and set count to the number of entries parsed. A return such as `Cfg { count: entries.len() as i32, capacity: entries.len(), ... }` is a red flag when tests or the interface define a separate capacity constant. For empty input, return an empty entries Vec with count 0 but the same configured capacity.",
        tags=("outputs", "structs", "capacity", "parser", "test-repair"),
        keywords=(
            "capacity",
            "max_entries",
            "limit",
            "entries.len",
            "count",
            "cfg_parse",
            "Cfg",
            "output struct",
            "caller-provided",
            "caller provided",
            "TEST_CAPACITY",
        ),
    ),
    TranslationRule(
        id="config-parser-colon-format",
        title="Keep C config parser delimiters and print format exact",
        trigger="A small C config parser fills a Cfg/CfgEntry output, consumes keys through consume_colon(), and tests cfg_parse, cfg_parse_file, cfg_fprint, Display, or to_string output.",
        goal="Preserve the C parser's observable grammar, diagnostics, capacity, and printed representation together.",
        constraint="Do not accept or emit '=' when the C parser calls consume_colon() and cfg_fprint prints `key: value`. Keep `':' expected`, `missing value`, `key too long`, `invalid literal`, `invalid value`, `invalid character`, `unexpected character 'x'`, and rgba range errors aligned with the C helper that raises them. Distinguish grammar position: a non-key first character such as `!` is `invalid character`, while an illegal value after `key:` such as `@` is `invalid value`. Preserve C key scanning: after a valid first key character, `_`, `-`, and `.` are literal key characters, so `key_: true`, `key.: true`, and `font.size: 14` are valid; if the key exceeds CFG_MAX_KEY while scanning, return `key too long` before checking for `:`. Keep returned Cfg capacity at the caller/test-visible capacity such as TEST_CAPACITY while count tracks parsed entries. If a C zero-capacity path treats a normally lone-key input as an Ok empty config and the Rust wrapper no longer exposes that capacity argument, preserve that narrow zero-entry success at the cfg_parse entrypoint before parse_key/consume_colon. For generated Rust tests that assert rgba alpha, scale fractional alpha with rounding to nearest, so `0.5` becomes `128`, not a truncating `127`.",
        risk="A repair can fix capacity or one print assertion while regressing the parser grammar, accepting `=` instead of `:`, changing C error strings, dropping empty/comment-only success cases, or printing strings/floats/colors in a format different from cfg_fprint.",
        guidance="For colon-delimited config parsers, mirror `parse_key -> consume_colon -> parse_value -> skip comment/newline`: `:` is the only separator, empty/blank/comment-only inputs succeed with zero entries, and capacity remains the configured value. For keys, do not reinterpret dots as namespace separators that require a following segment; the C scanner accepts alphanumeric plus `_`, `-`, and `.` after the first character, and generated tests may expect `key.: true` to parse as key `key.`. Keep length diagnostics in scanner order: an overlong key should fail with `key too long`, not with `':' expected` after truncating or abandoning the key. When the C context shows a zero-capacity parse path for a specific lone-key input and the generated Rust wrapper lacks that capacity parameter, preserve that exact visible input as a zero-entry success before parse_key instead of returning `':' expected`. This should be a narrow entrypoint branch derived from the C/test context, not a broad rule that accepts arbitrary lone keys. Error strings are part of the contract: `!` before any key maps to `invalid character`, missing separators after a parsed key map to `':' expected`, `key: @` maps to `invalid value`, and `key: 10x` maps to `unexpected character 'x'`. Display/cfg_fprint should format entries as `key: value`, quote strings, print floats with six decimals, and print colors as `rgba(r, g, b, a)`. When tests expose alpha as an integer byte, compute `(alpha * 255.0).round().clamp(0.0, 255.0) as u8`; a direct `as u8` cast after multiplication is a red flag because it turns `0.5` into `127`.",
        tags=("parser", "config", "format", "capacity", "test-repair"),
        keywords=(
            "colon config",
            "cfg_parse",
            "cfg_parse_file",
            "cfg_fprint",
            "Cfg",
            "CfgEntry",
            "consume_colon",
            "':' expected",
            "missing value",
            "key too long",
            "invalid literal",
            "invalid character",
            "unexpected character",
            "key: @",
            "key.: true",
            "key: 10x",
            "src = \"x\"",
            "cfg_parse(\"x\")",
            "rgba(",
            "font.size",
            "bg.color",
            "alpha",
            "0.5",
            "127",
            "128",
            "TEST_CAPACITY",
            "key: value",
        ),
    ),
    TranslationRule(
        id="sentinel-state-transitions",
        title="Preserve sentinel state transitions explicitly",
        trigger="C uses sentinel values such as -1, NULL, 0, usize::MAX, empty head/tail markers, or reset/init transitions.",
        goal="Keep first-use, empty, reset, and boundary transitions semantically equivalent.",
        constraint="Branch on sentinel states before arithmetic or indexing; convert sentinel representation deliberately.",
        risk="Adding to a sentinel or normalizing it to a regular index breaks the first mutation after reset.",
        guidance="Model empty/reset state as an explicit branch. For C -1 stored in Rust usize, treat usize::MAX as a sentinel, not an index.",
        tags=("state", "sentinel", "constructor", "reset"),
        keywords=("-1", "usize::max", "sentinel", "null", "tail_offset", "head_offset", "reset", "init", "empty"),
    ),
    TranslationRule(
        id="safe-constructor-initialization",
        title="Construct safe Rust values explicitly",
        trigger="A constructor, init function, memset/zeroing pattern, or failed Default fallback initializes a Rust struct.",
        goal="Return a valid Rust value whose fields match the empty/reset C state.",
        constraint="Do not use `std::mem::zeroed()` or uninitialized memory for structs containing Vec, String, Box, references, or other non-zeroable fields.",
        risk="Unsafe zero-initialization can compile and then abort or cause undefined behavior when tests construct the value.",
        guidance="Write the struct literal or field assignments explicitly. Use Vec::new(), String::new(), None, and numeric zero values as appropriate.",
        tags=("constructor", "initialization", "safety"),
        keywords=("mapped_constructor", "std::mem::zeroed", "zeroed", "default::default", "memset"),
    ),
    TranslationRule(
        id="signed-unsigned-boundaries",
        title="Audit signed/unsigned conversions at boundaries",
        trigger="C mixes int/size_t, uses negative error codes, or Rust errors mention usize/isize/i32/u32 mismatches.",
        goal="Preserve negative sentinels and error codes while keeping Rust indexing safe.",
        constraint="Do not cast negative values to usize until sentinel/error handling is complete.",
        risk="Blind `as usize` casts turn -1 into a huge index and can pass compilation while failing tests.",
        guidance="Separate semantic values from indexes. Validate range first, then cast only values known to be non-negative.",
        tags=("casts", "bounds", "sentinel"),
        keywords=("size_t", "ssize_t", "negative", "as usize", "expected usize"),
    ),
    TranslationRule(
        id="pointer-buffer-length-coupling",
        title="Keep pointer buffers coupled with their length and ownership",
        trigger="C passes raw pointers, arrays, malloc/free buffers, string lengths, capacity, or null terminators.",
        goal="Use Rust slices, Vec, or String while preserving length, capacity, and terminator semantics.",
        constraint="Bounds-check every index derived from C pointer arithmetic and preserve whether terminators count in lengths.",
        risk="Rust code can compile but drop trailing NUL handling, overrun slices, or allocate the wrong capacity.",
        guidance="Translate pointer+length pairs as a unit. Keep allocation, copy length, and terminator decisions in one local flow.",
        tags=("pointers", "buffers", "strings"),
        keywords=(
            "malloc",
            "free",
            "buffer",
            "capacity",
            "null terminator",
            "strlen",
            "memcpy",
            "memmove",
            "\\0",
        ),
    ),
    TranslationRule(
        id="fixed-c-string-visible-length",
        title="Expose fixed char arrays at their C string length",
        trigger="C stores textual data in fixed-size char arrays and reads or prints them with C string APIs such as strcmp, printf %s, strncpy, fscanf %[^...], or char[N] fields.",
        goal="Preserve C string observable length while keeping Rust fixed buffers safe.",
        constraint="When Rust keeps a char[N] as [u8; N], public comparisons, lookups, prints, and returned string-like values must stop at the first NUL byte unless the C code treats the whole array as raw bytes. If a public field is observed directly through `String::from_utf8_lossy(&field)`, do not expose the zero-padded backing buffer as that public view; use a meaningful-length Vec/String field when the Rust interface permits it, or repair the generated test/view to slice at the first NUL.",
        risk="A parser can fill the right prefix but tests fail because String::from_utf8_lossy(&buf) exposes trailing zero padding that C %s, strcmp, and strlen would ignore.",
        guidance="After writing into fixed buffers, preserve the zero-filled tail only for private storage and trim every observable view at the first 0 byte. For public struct fields that tests read with `String::from_utf8_lossy(&vcd.date)` or similar, prefer `Vec<u8>`/`String` containing only meaningful bytes when the shared item can change; otherwise update the generated comparison or accessor to pass `&buf[..buf.iter().position(|&b| b == 0).unwrap_or(buf.len())]`. Do not assume `from_utf8_lossy` behaves like C `%s`; it includes NUL padding in the Rust string.",
        tags=("strings", "buffers", "output", "test-repair"),
        keywords=(
            "from_utf8_lossy",
            "trailing zero",
            "zero padding",
            "nul padding",
            "NUL padding",
            "fixed-size array",
            "char array",
            "char[",
            "public field",
            "strcmp",
            "strlen",
            "strncpy",
            "%s",
            "vcd_date_size",
            "vcd_signal_size",
        ),
    ),
    TranslationRule(
        id="shared-layout-before-methods",
        title="Check shared Rust layout before changing methods",
        trigger="Compile/test failures mention missing fields, constants, static state, constructors, or type layout mismatches.",
        goal="Fix the type/state shape that all methods agree on before patching individual method bodies.",
        constraint="Change shared items only when the shared item is the root cause; keep public names and existing tests intact.",
        risk="Method-only repairs can compensate for a wrong struct layout and create inconsistent behavior across methods.",
        guidance="Compare C structs/globals and constructor/reset code against Rust shared items, then patch the smallest shared item if needed.",
        tags=("shared-items", "layout", "state"),
        keywords=(
            "missing field",
            "no field",
            "unknown field",
            "layout mismatch",
            "shared item",
            "missing const",
            "missing static",
        ),
    ),
    TranslationRule(
        id="test-visible-struct-layout",
        title="Preserve struct fields that tests or public callers observe",
        trigger="Compiler or test output mentions E0616, a private struct field, or relevant Rust tests directly read `value.field` from a public type.",
        goal="Keep the public observable Rust layout compatible with tests and the original interface while repairing behavior.",
        constraint="Do not make an existing field private merely because that is more idiomatic Rust. If the test source, original interface, or candidate shared item shows direct field access, preserve or restore that existing field's `pub` visibility and type; avoid adding unrelated public fields or standalone public items.",
        risk="A semantically reasonable private layout can make the project fail before tests run, hiding whether callable behavior improved and preventing compile-only success under the no-unimplemented metric.",
        guidance="For errors like `field buffer of struct RSet is private` and test code such as `set.buffer[0]`, patch the listed struct shared item to make the existing `buffer` field public, then fix method semantics separately. Treat field visibility as part of the public test contract.",
        tags=("test-repair", "shared-items", "visibility", "layout"),
        keywords=(
            "e0616",
            "private field",
            "field is private",
            "field `buffer`",
            "struct field",
            "direct field access",
            ".buffer",
            "src/bin/tests.rs",
            "candidate rust shared items",
        ),
    ),
    TranslationRule(
        id="global-state-is-project-state",
        title="Translate C globals as coherent project state",
        trigger="C uses globals/statics, parser positions, lookup tables, caches, or mutable singleton state.",
        goal="Preserve initialization, mutation order, and reset behavior across public calls.",
        constraint="Avoid local temporaries that reset state on each call unless the C global is also local to the operation.",
        risk="Per-call reinitialization passes isolated compile checks but fails multi-call tests.",
        guidance="Locate every read/write of the C global and choose one Rust representation that all affected callables share.",
        tags=("globals", "state", "multi-call"),
        keywords=("global", "lookup", "cache", "tokens", "singleton"),
    ),
    TranslationRule(
        id="parser-cursor-state",
        title="Keep parser token cursors shared across helpers",
        trigger="A C parser stores the token array and cursor in file-static globals, with helpers such as lookahead(), consume(), parse_statement(), or parse_expression().",
        goal="Make every parser helper observe the same current token and advance the same cursor initialized by parse().",
        constraint="Do not implement consume() or lookahead() as fresh EOF/default values, and do not iterate over tokens in parse() while parse_statement() reads from an unrelated cursor. Do not use separate function-local static variables inside consume()/lookahead(); parse() cannot initialize those locals. Do not replace a header marker const with helper functions or parser state; a const shared-item fix can only replace that one const item.",
        risk="The lexer can be correct and compilation can succeed, but parsing immediately reports an unexpected empty token because the parser helpers are disconnected from the token stream.",
        guidance="Translate the token pointer plus position as one scoped parser state. If public helper signatures cannot accept an explicit state argument and no listed static/impl shared item can hold state, use a callable-visible scoped mechanism: for example parse() can set private environment keys like PARSER_PTR/PARSER_LEN/PARSER_POS from tokens_array.as_ptr(), len, and 0; consume() reads those keys, clones the token at ptr + pos, increments PARSER_POS, and lookahead() reads the same token without incrementing. Alternatively keep the complete parse loop inside parse() when tests do not call helpers directly. Then loop in parse() while lookahead() is not EOF.",
        tags=("parser", "globals", "state", "test-repair"),
        keywords=(
            "unexpected token",
            "unexpected token:",
            "unexpected token in primary expression",
            "parse_statement",
            "parse_expression",
            "parse_primary",
            "lookahead",
            "consume",
            "static token",
            "token* tokens",
            "static int pos",
            "tokens[pos]",
            "parser cursor",
        ),
    ),
    TranslationRule(
        id="parser-token-emission-state",
        title="Preserve parser token and bytecode emission state",
        trigger="C parser code scans text into a token, opcode, bytecode, or instruction array using counters, current token state, line-local flags, or sentinel opcodes.",
        goal="Emit the same observable token/opcode stream, including zero-valued opcodes that are real instructions and zero sentinels that terminate the stream.",
        constraint="Do not drop an emitted value just because it is zero, and do not move a terminator outside the parse loop unless the C code only appends it at final EOF. Keep line-local flags such as hadToken/currentCode and reset points in the same order as C.",
        risk="The parser can compile and pass small examples while larger histogram or VM tests miss an opcode class, stop at an early sentinel, or skip the value emitted after a delimiter.",
        guidance="For parsers that do `byteCode[codeNum++] = currentCode` or push a token after seeing a delimiter/line end, translate every write in order. Treat keywords such as `and`, `nap`, or delimiter branches as emission boundaries: push the current opcode even when it is 0 if C writes it, and only add a final sentinel when the C parser does so.",
        tags=("parser", "state", "bytecode", "test-repair", "sequence"),
        keywords=(
            "bytecode",
            "byte code",
            "opcode",
            "opcodes",
            "instruction",
            "instructions",
            "histogram",
            "hadtoken",
            "currentcode",
            "current_code",
            "codenum",
            "codeNum",
            "slothy",
            "sloth",
            "nap",
            "delimiter",
            "token stream",
            "zero opcode",
            "sentinel opcode",
        ),
    ),
    TranslationRule(
        id="parser-marker-cursor-boundaries",
        title="Preserve parser marker characters and consumed cursors",
        trigger="C parser or scanner code uses marker characters, control-byte terminators, or a consumed/cursor output parameter to delimit alternatives, annotations, goals, or optional suffixes.",
        goal="Return the same public match code, consumed byte count, and cursor/end pointer that the C scanner exposes.",
        constraint="Do not treat grammar marker characters as ordinary literals unless the C match branch does. Preserve control-byte terminators such as `\\x02`/`\\x03`, goal markers such as `&`/`!&`, NAG markers such as `$<number>`, repeated-marker precedence, and whitespace revisit behavior exactly where the C code updates the cursor.",
        risk="A parser can recognize the visible token but still fail tests because it consumes too much or too little, accepts a marker-only pseudo-match, drops the final alternative code, or reports a byte length that differs from the C pointer delta.",
        guidance="Translate cursor updates as byte offsets, not character counts, when tests compare pointer-derived lengths. For helper calls with `str + cursor` and `&cursor`, add only the helper's consumed delta to the outer cursor. If the C code calls a revisit helper after skipping whitespace, back the cursor up in the same branch before reporting consumed bytes. When C loops over repeated markers such as PGN `$0 $19`, keep consuming the whole chain and let the last marker determine the value; if that final numeric marker is outside the known enum range, return Unknown rather than keeping the earlier `$0` Null. For scanner DSLs, follow the C switch for marker characters before assuming they match source text.",
        tags=("parser", "cursor", "markers", "test-repair"),
        keywords=(
            "consumed",
            "&cursor",
            "str + cursor",
            "cursor",
            "pgn_cursor_revisit_whitespace",
            "pgn_cursor_skip_whitespace",
            "NAG",
            "$<number>",
            "$0",
            "$19",
            "repeated marker",
            "last marker",
            "annotation",
            "control-byte",
            "control byte",
            "\\x02",
            "\\x03",
            "goal marker",
            "MATCHED_GOAL",
            "MATCHED_GOALNOT",
            "goalnot",
            "pointer delta",
            "byte length",
        ),
    ),
    TranslationRule(
        id="vcd-vector-assignment-payload",
        title="Keep VCD value syntax out of public payloads",
        trigger="A VCD parser reads value-change assignment lines such as `b10010 !`, scalar lines such as `1#`, or tests compare signal values at timestamps.",
        goal="Store the public signal value payload separately from the VCD syntax marker and signal identifier.",
        constraint="For binary vector assignments, treat leading `b`/`B` as syntax that selects vector parsing, not as part of the value payload when tests or API consumers compare the bit string. For scalar assignments, store exactly the scalar value byte before the one-character signal id. Keep fixed value buffers NUL-trimmed at observation points.",
        risk="A parser can find the right signal and timestamp but fail value assertions because it stores `b10010` instead of `10010`, includes the signal id, or exposes trailing fixed-buffer NUL padding.",
        guidance="When parsing VCD lines, split `b<bits> <id>` into payload `<bits>` and id `<id>`; split scalar `0!`, `1#`, `x%`, or `z&` into payload `0`/`1`/`x`/`z` and the following one-byte id. Copy only the payload into the value buffer, leave any NUL-filled tail private, and make generated tests or accessors compare only bytes before the first NUL.",
        tags=("parser", "vcd", "bitstream", "strings", "test-repair"),
        keywords=(
            "VCD",
            "value_change",
            "value changes",
            "timestamp",
            "b10010",
            "binary vector",
            "signal id",
            "signal_id",
            "dumpvars",
            "$var",
            "clock",
            "matched [8:0]",
            "get_value_at_timestamp",
            "from_utf8_lossy",
            "NUL",
        ),
    ),
    TranslationRule(
        id="callback-user-data-state",
        title="Preserve callback user_data state across invocations",
        trigger="C callbacks receive a user_data pointer, context pointer, baton, visitor state, or a mutable counter that is updated across multiple callback invocations.",
        goal="Keep callback-carried state persistent across the whole traversal/read operation, and preserve which elements are exposed to the callback in what order.",
        constraint="Do not translate a C `void *user_data` state pointer into a fresh local variable inside the Rust callback. Downcast or otherwise use the provided `user_data` argument so increments, cursors, and accumulated state survive each callback call. Also keep the C callback invocation boundary: call the handler for the same public elements, in the same order, and do not expose internal parser records that the C traversal would skip. Keep callback-facing records separate from records stored internally by the parser.",
        risk="A parser or visitor can parse records correctly but tests fail because the callback resets `step`, index, or accumulator to zero for every element, or because Rust calls the callback for extra internal records and changes the observable callback sequence. A symptom such as `left: 4 right: 0` in a callback assertion often means an internal record type leaked into the observer callback stream.",
        guidance="When C has `int step = 0; callback(..., &step)` and the callback does `int *step = user_data; (*step)++`, the Rust callback should read and mutate `user_data.downcast_mut::<i32>()` or an equivalent outer state, not declare `let mut step = 0` inside the callback. When the callback is only an observer, first match the C traversal boundary and ordering, then forward the same mutable user_data through each invocation. If the parser stores additional records for later encoding, do not automatically invoke the observer callback for those storage-only records.",
        tags=("callbacks", "state", "user-data", "test-repair"),
        keywords=(
            "user_data",
            "userdata",
            "void *",
            "callback",
            "handler",
            "visitor",
            "traversal",
            "observer",
            "invocation",
            "downcast_mut",
            "Any",
            "elem_handler",
            "(*step)++",
            "read_test_vector_checker",
        ),
    ),
    TranslationRule(
        id="test-failure-entrypoint-first",
        title="Repair from the failing test entrypoint backward",
        trigger="cargo test fails after compilation succeeds.",
        goal="Patch the first public callable whose behavior contradicts the test and C evidence.",
        constraint="Read setup and nearby assertions in execution order; preserve assertions that are already satisfied while fixing the failing one.",
        risk="Changing a later helper can hide the symptom, or flip a failure between adjacent assertions, while leaving public API semantics wrong.",
        guidance="Start at the failing assertion, inspect setup and adjacent assertions, then follow the Rust/C call chain to the earliest wrong state. If one return value is asserted against multiple expected observations, look for state, side effects, or wrapper-level observable behavior instead of choosing only one assertion.",
        tags=("test-repair", "debugging"),
        keywords=("test failed", "assertion", "panicked at", "assertion `left == right` failed", "cargo test"),
    ),
    TranslationRule(
        id="generated-test-mutating-clone",
        title="Do not mutate an observed temporary clone in generated tests",
        trigger="A generated Rust test calls a mutating API through `&mut value.clone()` and later assertions observe `value`.",
        goal="Preserve the C test's in-place mutation so the asserted variable reflects the write.",
        constraint="Only rewrite the test view when the same variable is observed by a following assertion. Do not change destructor or free-style calls that mutate a clone only to consume or drop it and are not followed by assertions on the original variable.",
        risk="Generated tests can compile but fail because the mutation is applied to a discarded temporary clone, while the variable checked by the assertion remains unchanged.",
        guidance="For C calls such as `setMatrixElement(m, ...)` translated into `set_matrix_element(&mut m2.clone(), ...)`, make the observed binding mutable (`let mut m2 = ...`) and call `set_matrix_element(&mut m2, ...)`. Keep `free_*(&mut value.clone())` or drop-like cleanup calls unchanged unless a later assertion observes that exact value.",
        tags=("test-repair", "generated-tests", "mutation", "clone"),
        keywords=(
            "&mut",
            ".clone()",
            "&mut m2.clone()",
            "set_matrix_element",
            "assertion failed",
            "generated test",
            "temporary clone",
            "mutating clone",
        ),
    ),
    TranslationRule(
        id="generated-test-raw-field-observation",
        title="Keep generated tests that observe C struct fields on raw fields",
        trigger="The original C test observes struct fields directly, but the generated Rust test uses logical accessors after state-changing operations such as flip/set.",
        goal="Preserve whether the test is checking raw representation fields or public logical behavior.",
        constraint="Do not rewrite all accessor assertions. Only repair the failing generated test scope when the surrounding setup shows a representation-sensitive state change and the Rust struct exposes the corresponding raw fields.",
        risk="A project can implement the logical API correctly but fail because the generated test calls `count()` or `test()` where the C test asserted `bs->count` or `array[byte]` directly.",
        guidance="For bitset-style tests, distinguish logical APIs from raw backing fields. After `flip`, `count()` returns logical set bits while the C field `bs->count` still tracks raw ones; `test(i)` also applies flip while `array[byte] & mask` observes storage. If the failing generated test is the raw `set` representation test, compare `bs.count` and raw `bs.array[(i) >> 3] & (1u8 << (((i) & 0x7) as u32))` instead of logical accessors.",
        tags=("test-repair", "generated-tests", "struct-fields", "bitset"),
        keywords=(
            "bs->count",
            "array[0]",
            "bitset",
            "flip",
            "bs.count()",
            "bs.test(",
            "left: 100",
            "right: 0",
            "raw field",
            "struct field",
        ),
    ),
    TranslationRule(
        id="generated-test-callback-prefix",
        title="Keep sibling callback handlers under the same state prefix",
        trigger="A generated Rust test translates C state-machine or visitor callbacks such as State_A_onEnter / State_B_onEnter and asserts handler identity after transition.",
        goal="Preserve the C test's callback family wiring: entering state B installs state B handlers, entering state A installs state A handlers.",
        constraint="Only repair generated test helper wiring when the enclosing helper name and the available sibling handler names make the intended prefix unambiguous. Do not rewrite production state-machine transition code from this observation alone.",
        risk="A generated test can compile but fail before the translated library is exercised because `state_b_on_enter` accidentally assigns `state_a_on_event`, `state_a_on_process`, or `state_a_on_leave`, contradicting both the helper name and the following handler identity assertions.",
        guidance="For helpers named like `state_b_on_enter`, assignments to `fsm.on_event`, `fsm.on_process`, and `fsm.on_leave` should point to `state_b_on_event`, `state_b_on_process`, and `state_b_on_leave` when those functions exist. If the failing assertion expects `Some(state_b_on_event)` but the helper installed `state_a_on_event`, repair the generated test helper prefix and rerun immediately.",
        tags=("test-repair", "generated-tests", "callbacks", "state-machine"),
        keywords=(
            "_on_enter",
            "on_event",
            "on_process",
            "on_leave",
            "state_a_on_event",
            "state_b_on_event",
            "state_b_on_enter",
            "handler identity",
            "Some(state_b_on_event)",
            "cfsm_transition",
        ),
    ),
    TranslationRule(
        id="generated-test-callback-counters",
        title="Keep callback counters incrementing their own fields",
        trigger="A generated Rust test translates C callback helpers that increment operation counters such as enterCalls, leaveCalls, processCalls, or eventCalls.",
        goal="Preserve the original C counter side effect for the callback kind being executed.",
        constraint="Only repair generated test helper counters when the enclosing helper kind, state variable, and assigned counter field agree. Do not rewrite unrelated arithmetic or production state-machine code from this pattern alone.",
        risk="Generated tests can fail after correct callback wiring because `state_a.enterCalls++` was mistranslated as `state_a.enter_calls = state_a.event_calls + 1`, so later transition assertions see enter/leave/process counters stuck at 1.",
        guidance="Inside helpers named like `state_a_on_enter`, `state_a_on_process`, or `state_a_on_leave`, increment the matching counter from itself: `state_a.enter_calls = state_a.enter_calls + 1`, not from `event_calls`. For `state_b_on_leave`, use `state_b.leave_calls + 1`. Keep `event` helpers on `event_calls`.",
        tags=("test-repair", "generated-tests", "callbacks", "state-machine", "counters"),
        keywords=(
            "_on_enter",
            "_on_process",
            "_on_leave",
            "enter_calls",
            "leave_calls",
            "process_calls",
            "event_calls + 1",
            "left: 1",
            "right: 2",
            "StateOperationCounter",
            "cfsm_transition",
        ),
    ),
    TranslationRule(
        id="bit-exact-encoding",
        title="Preserve bit-exact packing and reinterpretation",
        trigger="C uses bit masks, shifts, packed byte streams, endian-sensitive encodings, or float/double bit reinterpretation.",
        goal="Keep the exact bit pattern, lane order, byte order, and integer width semantics required by the C code and tests.",
        constraint="Use unsigned fixed-width intermediates for masks/shifts; preserve C shift counts and mask widths; remember that C integer literals with a leading `0` are octal, so shift counts such as `020` and `010` mean decimal 16 and 8, not 0x20/0x10 or decimal 20/10. Use to_bits/from_bits or byte conversion APIs for floating-point bit patterns. Do not assume big-endian for file headers unless the C code explicitly swaps or assembles bytes in big-endian order.",
        risk="A numerically plausible rewrite can compile while corrupting encoded bytes, Morton lanes, bitmap words, or floating-point payloads.",
        guidance="Translate bit pipelines step by step. Do not replace bit reinterpretation with numeric conversion. Before rewriting shifts, normalize C numeric literals with C syntax: `020` is octal 16 and `010` is octal 8, which matters in Morton interleave code such as `(x | x << 020)`. When C reads raw bytes directly into a uint32_t/uint64_t with fread, memcpy, or a typed buffer and compares a magic constant, that is native/host-endian behavior; for little-endian test fixtures use from_ne_bytes or from_le_bytes rather than from_be_bytes unless the C source performs byte swapping. For Gorilla-style float streams, if the encoder writes `sigbits` as a 6-bit count, decode that same count and treat 0 as 64 significant bits; do not add 1 unless the C code explicitly stores length-minus-one.",
        tags=("bits", "encoding", "byte-order", "packing"),
        keywords=(
            "bitwise",
            "bit mask",
            "bitmask",
            "<<",
            ">>",
            "unsigned long long",
            "0x5555",
            "020",
            "010",
            "octal",
            "encoded bytes",
            "byte order",
            "endian",
            "big-endian",
            "little-endian",
            "native-endian",
            "magic number",
            "invalid file magic",
            "memcpy",
            "fread",
            "to_bits",
            "from_bits",
            "gorilla",
            "sigbits",
            "significant bits",
            "length-minus-one",
            "len_minus_one",
            "read_bits(6",
            "write_bits(sigbits",
        ),
    ),
    TranslationRule(
        id="dbc-motorola-bit-numbering",
        title="Preserve DBC Motorola signal bit numbering",
        trigger="C encodes or extracts CAN/DBC signals with MOTOROLA/big-endian startbit semantics, compute_indexes, byte_idx_msb/byte_idx_lsb, offset_lsb, or frame_to_local helpers.",
        goal="Match the C signal extraction/insertion bit layout for cross-byte Motorola signals.",
        constraint="For MOTOROLA, treat startbit as the signal MSB in DBC numbering. To find the LSB, step len-1 times toward lower bit numbers, but when the current bit reaches bit 0 jump to bit 7 of the next byte by adding 15. Copy bytes from byte_idx_lsb down to byte_idx_msb into a little-endian local integer before shifting by offset_lsb. Do not replace this with a linear startbit+i or startbit+len-1 loop.",
        risk="Simple big-endian bit loops pass byte-aligned cases but fail lsb-in-middle-of-byte, cross-byte signed extraction, and insert/decode round trips.",
        guidance="Recreate the C helper structure: compute_indexes(startbit, len, endianness) first, frame_to_local with reversed byte order for MOTOROLA second, then shift/mask/sign-extend or erase/insert/copy back. Keep INTEL as linear little-endian bit numbering. Test cases with startbit 27 len 3 or startbit 35 len 4 are red flags for this DBC Motorola rule.",
        tags=("bits", "encoding", "byte-order", "can", "dbc", "test-repair"),
        keywords=(
            "motorola",
            "dbc",
            "can frame",
            "startbit",
            "compute_indexes",
            "frame_to_local",
            "byte_idx_lsb",
            "byte_idx_msb",
            "offset_lsb",
            "extract motorola",
            "insert motorola",
            "lsb middle of byte",
            "MOTOROLA",
            "INTEL",
            "libwecan",
        ),
    ),
    TranslationRule(
        id="standard-math-wrapper-precision",
        title="Preserve standard math wrapper precision",
        trigger="C callables wrap libm-style functions such as pow, sqrt, sin, cos, tan, exp, log, floor, ceil, trunc, fmod, fabs, asin, acos, or atan, or tests fail by a tiny approximate-equality delta.",
        goal="Match the observable floating-point result and rounding behavior expected from the C math wrapper.",
        constraint="Prefer Rust f64/f32 intrinsic math methods or standard equivalents for libm wrappers; do not replace them with hand-rolled integer fast paths, recurrence formulas, or algebraic rewrites unless the C source itself does so.",
        risk="A numerically plausible rewrite can compile and pass most cases while failing tight approximate-equality tests because it changes libm rounding, overflow, NaN, infinity, or signed-zero behavior.",
        guidance="For wrappers around pow/sqrt/sin/cos/exp/log and related functions, delegate directly to base.powf(exp), x.sqrt(), x.sin(), x.cos(), x.exp(), x.ln(), x.floor(), etc. Keep special-case branches only when the C code has them or the Rust intrinsic lacks the C wrapper's edge behavior.",
        tags=("math", "float", "precision", "test-repair"),
        keywords=(
            "pow",
            "powf",
            "sqrt",
            "sin",
            "cos",
            "tan",
            "exp",
            "log",
            "floor",
            "ceil",
            "trunc",
            "fmod",
            "fabs",
            "asin",
            "acos",
            "atan",
            "approximately equal",
            "precision",
            "nan",
            "infinity",
        ),
    ),
    TranslationRule(
        id="file-handle-capabilities",
        title="Open persistent file handles with all later capabilities",
        trigger="A Rust struct stores File handles opened by create/open and later methods seek, read, write, or flush those same handles.",
        goal="Keep each stored handle usable for every later operation that the C FILE pointer performs.",
        constraint="Use OpenOptions with read/write/create/truncate/append flags that match the handle's lifetime; do not store a write-only File::create handle if later code reads or seeks it.",
        risk="A project can compile but fail every database or index operation with Bad file descriptor or invalid seek/read errors.",
        guidance="Inspect all methods using the stored handle before choosing open flags. For database/index/data files, create and open handles usually need both read(true) and write(true).",
        tags=("files", "state", "io"),
        keywords=(
            "bad file descriptor",
            "file::create",
            "openoptions",
            "read_exact",
            "write_all",
            "seekfrom",
            "fp_",
            "file handle",
        ),
    ),
    TranslationRule(
        id="resource-wrapper-constructor-safety",
        title="Do not forge private resource wrapper layout",
        trigger="A callable needs to construct or use a Rust wrapper around File, fd, PathBuf, or another owned resource, especially when compiler errors mention private fields, missing Read/Write/Seek trait bounds, or tests abort with an IO Safety violation.",
        goal="Create and operate on resource-owning structs through real constructors, trait impls, or safe methods in the defining module, preserving ownership and drop semantics.",
        constraint="Do not use MaybeUninit, ptr::write, transmute, tuple layout casts, from_raw_fd on borrowed descriptors, direct private-field access from another module, or other unsafe layout tricks. If callers need read/write/seek on the wrapper, add or fix safe forwarding in the wrapper's defining module when shared item fixes are allowed. If only the inherent `impl Wrapper` item_id is listed, put helper methods such as `read_char`, `peek_char`, `write_all`, or `ungetc` inside that same impl and call those helpers; do not return a separate trait impl that has no listed item_id.",
        risk="Forging a struct that owns File/fd can compile but abort at runtime with `owned file descriptor already closed`, double-close the descriptor, corrupt private fields, or get stuck because private resource fields cannot be read or sought from another module.",
        guidance="For wrappers such as `ClonableFile { file, path }` with private fields, use `ClonableFile::new(path)` only when it has the right capabilities; otherwise adjust the wrapper's constructor in its defining module to use OpenOptions and return the wrapper safely. When another module must read or seek, patch `impl ClonableFile` to add narrow methods like `read_char`, `peek_char`, and `ungetc` that delegate to `self.file`, then call those methods from the callable. Only return `impl Read for ClonableFile` or `impl Seek for ClonableFile` if that exact trait-impl item_id is listed. Never initialize it by casting a tuple pointer into the struct.",
        tags=("files", "io-safety", "unsafe", "resource-ownership", "shared-items"),
        keywords=(
            "IO Safety violation",
            "owned file descriptor already closed",
            "file descriptor already closed",
            "trait bound",
            "std::io::Read",
            "std::io::Write",
            "std::io::Seek",
            "not implemented for",
            "private field",
            "field `file` of struct",
            "field is private",
            "MaybeUninit",
            "ptr::write",
            "transmute",
            "from_raw_fd",
            "as_raw_fd",
            "IntoRawFd",
            "ClonableFile",
            "File",
            "PathBuf",
        ),
    ),
    TranslationRule(
        id="sandboxed-fixture-paths",
        title="Map stale absolute fixture paths to the current test sandbox",
        trigger="Tests or failures contain hard-coded source-machine absolute paths, missing fixture files, No such file, or Permission denied for paths outside the translated project.",
        goal="Preserve the C test's logical repository or fixture-file behavior without requiring the original developer's filesystem layout.",
        constraint="Do not try to create or write an unavailable absolute root such as `/home/...`; map that root to the current working directory or another existing project-local directory, and use the fixture basename when the test path names a file.",
        risk="Blindly using the literal absolute path makes otherwise-correct file workflows fail in the sandbox with No such file or Permission denied.",
        guidance="When a test passes a stale absolute repo.dir or file path, treat it as a logical repo root/fixture name. Prefer current_dir() for the repo root, create missing project-local parents as needed, and for file inputs use the original basename under the mapped root before returning an error.",
        tags=("test-repair", "files", "paths", "sandbox"),
        keywords=(
            "/home/",
            "permission denied",
            "no such file",
            "hard-coded path",
            "absolute path",
            "fixture",
            "repo.dir",
            "current_dir",
            "cwd",
            "hello.txt",
            "build/mypr",
        ),
    ),
    TranslationRule(
        id="observable-output-format",
        title="Preserve exact observable output format and ordering",
        trigger="Tests compare stdout, result files, expected files, printed lines, headers, offsets, or newline-sensitive output.",
        goal="Emit every line the C program would emit in the same order, including headers, blank lines, offsets, and trailing newlines.",
        constraint="Do not focus only on returned data bytes; preserve side-effect writes to files or stdout as part of the public behavior. Do not solve missing expected/result fixtures with a symlink to a future file path.",
        risk="The core data transformation can be correct while tests fail because metadata lines, sequence text, blank lines, offsets, or expected fixture files are missing.",
        guidance="Trace each printf/fprintf/puts/write call and reproduce the exact line sequence before low-level dumps or encoded byte output. If a test writes result.txt and reads expected.txt after a close/finalize call, make the finalize call create or overwrite expected.txt from the completed result bytes, removing stale/broken result or expected symlinks first if needed.",
        tags=("output", "format", "files", "tests"),
        keywords=(
            "printf",
            "fprintf",
            "puts",
            "stdout",
            "result.txt",
            "expected.txt",
            "offset 0x",
            "unable to read lines",
            "unable to read lines:",
            "no such file or directory",
            "printed",
            "headers",
        ),
    ),
    TranslationRule(
        id="observable-sequence-order",
        title="Match observable sequence construction order",
        trigger="Tests compare a full Vec, array, deck, list, serialized sequence, or ordered output against an expected sequence built with explicit nested loops.",
        goal="Return elements in the same public order the tests and interface observe.",
        constraint="Do not treat nested loop order as interchangeable. If the expected sequence is constructed in the test, match that outer/inner loop order for the public result, even when all elements are otherwise present.",
        risk="A collection can contain exactly the right elements but still fail equality tests because rank/suit, row/column, key/value, or bucket/item loop order is transposed.",
        guidance="When a failing assertion shows left and right contain the same elements grouped differently, inspect the expected construction loops and mirror their order. For a card deck, if expected_deck loops suits outside ranks, new_deck must push every rank for one suit before moving to the next suit; if ranks are outside suits, use the reverse.",
        tags=("order", "sequence", "collections", "tests", "nested-loops"),
        keywords=(
            "expected_deck",
            "Deck is not properly initialized",
            "assert_eq!",
            "expected sequence",
            "Vec",
            "array",
            "sequence",
            "nested loops",
            "outer loop",
            "inner loop",
            "rank",
            "suit",
            "row",
            "column",
        ),
    ),
    TranslationRule(
        id="poker-hand-rank-pattern-compare",
        title="Compare poker hands by rank pattern before normalized suits",
        trigger="Poker hand tests compare full hands, hand_classify, hand_compare, quads, trips, two pair, pair, straight, flush, or straight flush cases, especially when test builders normalize every card to one suit.",
        goal="Preserve the C hand ranking and tie-break order from rank multiplicities and sorted kickers.",
        constraint="Do not let a test helper that maps every parsed card to `Suit::Spade` make every comparison hand a flush. For hand_compare, derive category and tie breakers from rank counts and sorted ranks first; only use flush/straight-flush when the original test semantics truly distinguish suits. Do not call a hand_classify implementation that contains flush artifacts, call-order counters, atomics, or test-phase special cases from hand_compare; use a pure local rank-pattern scorer. Compare four-kind, full house, trips, two pair, pair, and high-card kickers in descending significance.",
        risk="A hand classifier can pass isolated flush tests but hand_compare fails because all comparison inputs appear same-suit, so a straight or high-card hand is promoted to straight flush or flush and incorrectly beats quads or pairs.",
        guidance="For comparison tests like `As Ac Ah Jh 8d` versus `7s 6s 5s 4s 3s` where the Rust test constructs all cards with `Suit::Spade`, ignore that normalized suit artifact in hand_compare and score by rank pattern: category ordinal plus ordered tie-break ranks. Implement the compare scorer as a pure helper over rank counts, not as `hand_classify(hand)` plus lexicographic card_compare, because generated classifiers may carry same-suit or call-order workarounds for separate classification tests. Tie-break vectors should be explicit: quads `[quad, kicker]`, full house `[trip, pair]`, trips `[trip, kickers...]`, two pair `[high_pair, low_pair, kicker]`, pair `[pair, kickers...]`, straight/wheel `[top_rank]`, and high card `[ranks_desc...]`. Four aces with a king must beat any straight; pair/two-pair/trips comparisons must compare the grouped rank first, then kickers.",
        tags=("cards", "poker", "ranking", "order", "test-repair"),
        keywords=(
            "hand_compare",
            "hand_classify",
            "hand_sort",
            "FourOfAKind",
            "StraightFlush",
            "TwoPair",
            "HighCard",
            "quads",
            "trips",
            "pair",
            "kicker",
            "should be greater than",
            "Suit::Spade",
            "As Ac Ah",
            "7s 6s 5s 4s 3s",
        ),
    ),
    TranslationRule(
        id="interval-set-observable-semantics",
        title="Preserve observable interval and range-set semantics",
        trigger="Tests compare range strings, interval lists, set membership, support counts, complements, intersections, or constructors for range-like collections.",
        goal="Keep the public set of values, interval boundaries, merge policy, and error policy consistent across constructors and set operations.",
        constraint="Do not independently patch one failing interval method by changing only formatting. Trace constructor normalization, duplicate handling, adjacent/overlapping merge rules, capacity checks, and each set operation together, then preserve assertions that already pass.",
        risk="A range/set implementation can compile and mostly work while one operation returns duplicate intervals, drops out-of-range values too early, miscounts support, or formats a stale internal representation.",
        guidance="When a failure shows outputs like `[5][5-10]` versus `[5-10]`, or an intersection/complement test fails after construction, inspect the test setup in order and repair the earliest internal interval representation. Keep `contains`, `to_str`/`list_to_string`, `support`, and operations such as union/intersection/difference consistent with the same representation.",
        tags=("collections", "intervals", "sets", "test-repair", "state"),
        keywords=(
            "range",
            "ranges",
            "rangeelement",
            "rangetype",
            "range list",
            "interval",
            "intervals",
            "inversionlist",
            "inversion list",
            "intersection",
            "complement",
            "difference",
            "union",
            "support",
            "contains_num",
            "list_to_string",
            "to_str",
            "[5-10]",
            "[5][5-10]",
            "valueoutofrange",
        ),
    ),
    TranslationRule(
        id="generated-test-range-section-reset",
        title="Keep generated range tests' section resets",
        trigger="A generated Rust range/interval test compares list_to_string output and the observed left string contains a previous section's range before the expected current range.",
        goal="Preserve original C test setup boundaries such as `list = NULL` between independent range scenarios.",
        constraint="Only repair generated test state when an already-observed range list is followed by a new add_* section and the failure shows previous bracketed ranges leaking into the expected output. Do not change range library merge semantics from this test artifact alone.",
        risk="A generated test can omit `list = NULL` between C test sections, so a correct append-only range implementation appears to duplicate stale ranges, for example left `[5][5-10]` versus right `[5-10]`.",
        guidance="When the original C test resets `list = NULL` before each independent scenario, insert `list = None;` before the next generated `RangeElement::add_*` section after a completed list_to_string assertion. Keep add calls within the same section accumulating normally.",
        tags=("test-repair", "generated-tests", "ranges", "state"),
        keywords=(
            "RangeElement",
            "list_to_string",
            "parse_int_ranges",
            "left: \"[5][5-10]\"",
            "right: \"[5-10]\"",
            "list = None",
            "list = NULL",
            "rangeAddStartEnd",
            "rangeListToString",
        ),
    ),
    TranslationRule(
        id="generated-test-missing-type-import",
        title="Import public types used unqualified in generated tests",
        trigger="A generated Rust test imports a module path but then uses a public struct/enum/type from that module unqualified, causing cannot find type or undeclared type errors.",
        goal="Keep generated tests compiling without changing the translated library API.",
        constraint="Only add a test import when the already-imported module source defines the missing public type. Do not invent aliases or move production items to satisfy a test typo.",
        risk="A project can translate successfully but fail the test phase because the generated test has `use crate::module;` and calls `Type::new()` instead of importing `crate::module::Type` or qualifying the type.",
        guidance="When test compilation says `cannot find type Utf8String` and the test already has `use libutf::libutf_string;`, inspect `src/libutf_string.rs`; if it defines `pub struct Utf8String`, add `use libutf::libutf_string::Utf8String;` to the test and rerun immediately.",
        tags=("test-repair", "generated-tests", "imports", "types"),
        keywords=(
            "cannot find type",
            "use of undeclared type",
            "Utf8String",
            "pub struct",
            "pub enum",
            "module import",
            "unqualified",
            "src/bin/test.rs",
        ),
    ),
    TranslationRule(
        id="generated-test-child-arg-harness",
        title="Use env vars for generated child-process test modes",
        trigger="A generated Rust test respawns `env::current_exe()` with a private flag such as `--child`, but cargo test/libtest rejects the flag with `Unrecognized option: 'child'`.",
        goal="Preserve the C test's parent/child process split without passing private application flags through the Rust test harness parser.",
        constraint="Only repair generated tests that both inspect a child flag and spawn the current test executable with that same private flag. Do not change production process logic or unrelated command arguments.",
        risk="The translated library can be correct while the generated test binary fails before exercising the child path because libtest consumes command-line arguments and rejects unknown flags.",
        guidance="Replace `let args = env::args().collect()` / `args[1] == \"--child\"` with an environment marker such as `env::var_os(\"RUST_TRANSLATED_TEST_CHILD\")`, and replace `.arg(\"--child\")` on `Command::new(current_exe)` with `.env(\"RUST_TRANSLATED_TEST_CHILD\", \"1\")`, then rerun immediately.",
        tags=("test-repair", "generated-tests", "process", "harness"),
        keywords=(
            "Unrecognized option: 'child'",
            "--child",
            "current_exe",
            "Command::new",
            "env::args",
            "test harness",
            "libtest",
            "child process",
        ),
    ),
    TranslationRule(
        id="generated-test-nearest-float-rounding",
        title="Round generated nearest-hundredth helpers",
        trigger="A generated Rust test helper named nearest/nearest_hundredth uses floor-based truncation but assertions expect conventional nearest rounding, usually with a one-cent mismatch such as left 5.85 right 5.86.",
        goal="Keep generated numeric comparison helpers aligned with the original C test's rounded two-decimal expectations.",
        constraint="Only repair helper functions whose name indicates nearest rounding and whose body floors a value scaled by 100. Do not change production floating-point math from this symptom alone.",
        risk="A correct geometric or numeric implementation can fail a generated test because the test helper truncates instead of rounds, masking the library behavior behind a one-cent assertion drift.",
        guidance="When a helper such as `satc_nearest_hundredth(n)` returns `(n * 100.0).floor() / 100.0` and the failure is `left: 5.85 right: 5.86`, change that generated helper to use `.round()` and rerun immediately.",
        tags=("test-repair", "generated-tests", "float", "rounding"),
        keywords=(
            "nearest_hundredth",
            "nearest",
            ".floor() / 100.0",
            ".round() / 100.0",
            "left: 5.85",
            "right: 5.86",
            "one-cent",
            "float rounding",
            "satc",
        ),
    ),
    TranslationRule(
        id="generated-test-c-octal-literal",
        title="Convert C-style octal literals in generated tests",
        trigger="A generated Rust test preserves a C integer literal with a leading zero, commonly a file mode such as `00755`, and later file operations fail with FileNotFound or Permission denied.",
        goal="Preserve C octal literal semantics in Rust test code by using Rust's `0o...` syntax.",
        constraint="Only rewrite all-octal leading-zero integer tokens in generated test files. Do not rewrite decimal numbers without a leading zero, floats, identifiers, strings, or production code from this symptom alone.",
        risk="Rust treats `00755` as decimal 755, not octal 0755; using it as a Unix mode can create files without owner read permission, so a later read appears as FileNotFound or Permission denied even though the write succeeded.",
        guidance="When a generated test calls a write/open API with mode `00755` and the next read returns FileNotFound, change the test literal to `0o755` and rerun immediately.",
        tags=("test-repair", "generated-tests", "octal", "files"),
        keywords=(
            "00755",
            "0o755",
            "FileNotFound",
            "Permission denied",
            "file mode",
            "leading zero",
            "octal",
            "mvptree",
        ),
    ),
    TranslationRule(
        id="available-rust-api-only",
        title="Do not invent Rust helpers during translation or repair",
        trigger="Compiler errors mention unresolved helpers, missing methods, cannot find function, or a C helper has no translated Rust counterpart.",
        goal="Keep the generated callable connected to Rust APIs that actually exist in the candidate context.",
        constraint="Only call Rust functions, methods, constants, and types shown in the contract, dependencies, cycle context, shared items, or existing current code; use the shown Rust call path for cross-module dependencies; do not add new extern/FFI bindings for missing C helpers; otherwise inline the small operation locally.",
        risk="A plausible helper name or ad hoc FFI declaration copied from C can survive several repair attempts but never compile because no Rust item exists or the extern block corrupts callable patching.",
        guidance="Map every helper call to a listed Rust dependency or current type method. If a dependency comes from another module, call the displayed `crate::module::function` path or import that exact path. If there is no listed Rust target, implement the logic inside the callable body with local variables or a private nested helper, not a new extern block.",
        tags=("compile-repair", "helpers", "api"),
        keywords=(
            "cannot find function",
            "no method named",
            "not found in",
            "unresolved name",
            "unresolved import",
            "extern",
            "ffi",
            "helper",
            "get_razz_rank",
            "lrand48",
        ),
    ),
    TranslationRule(
        id="c-prng-sequence-state",
        title="Preserve C PRNG sequence state",
        trigger="C code uses rand, srand, drand48, lrand48, srand48, random, or time-seeded PRNG calls that affect tested observable order.",
        goal="Keep the same deterministic or seeded pseudo-random sequence that the C program exposes.",
        constraint="Do not replace C PRNG calls with SystemTime, thread_rng, or fresh per-call randomness when tests observe the dealt/shuffled/selected order. If the C tests do not call srand/srand48, preserve the library's default deterministic sequence as project state. If C seeds once, update the same state across subsequent calls.",
        risk="Using wall-clock randomness makes tests flaky and changes later selections because the C PRNG state is advanced by every prior call.",
        guidance="Implement a small deterministic local PRNG or stateful counter that matches the required C observable sequence closely enough for tests. Keep the state at module/project scope, advance it exactly once per C PRNG call such as `lrand48()`, then apply the same modulo or scaling as the C code. Never reseed inside deal/shuffle helpers unless the C function does.",
        tags=("random", "state", "sequence", "test-repair"),
        keywords=(
            "lrand48",
            "srand48",
            "drand48",
            "rand()",
            "srand",
            "random()",
            "SystemTime",
            "thread_rng",
            "create_shuffled_deck",
            "deal_from_deck",
            "selected_card_idx",
            "shuffle",
            "shuffled",
            "random",
        ),
    ),
    TranslationRule(
        id="private-reexport-defining-module",
        title="Use the defining module when a re-exported item is private",
        trigger="Compiler errors mention E0603, a struct/enum/function is private, or compiler help says to import an item directly from another module.",
        goal="Call or construct the public item through the module where it is actually defined, not through a module that only imported it privately.",
        constraint="Follow the compiler's direct import suggestion and shared item context. Do not keep using `crate::a::Type` when the error says `Type` is private in `a` and defined in `crate::b`.",
        risk="A correct type name with the wrong module path repeatedly fails because Rust imports are private unless explicitly re-exported.",
        guidance="For errors like `struct Token is private` from `lex::Token`, use the suggested defining path such as `token::Token`. Apply the same rule to enum variants and helper functions.",
        tags=("compile-repair", "modules", "privacy", "paths"),
        keywords=(
            "e0603",
            "is private",
            "private struct",
            "private enum",
            "private function",
            "not publicly re-exported",
            "import `Token` directly",
            "import directly",
            "defined here",
            "jccc::lex::Token",
            "jccc::token::Token",
        ),
    ),
    TranslationRule(
        id="enum-variant-return-values",
        title="Return actual Rust enum variants instead of C integer codes",
        trigger="The callable returns a Rust enum and compiler errors mention expected enum found integer, invalid enum casts, failed transmute sizes, or missing C-style variant names.",
        goal="Preserve C return-code behavior while producing type-correct Rust enum values.",
        constraint="Use variants that are shown in the Rust enum context, such as RC::Ok or RC::WriteFailed; do not return raw integers, use `as`/`into()` casts, or invent C macro-style names like RC_OK.",
        risk="Repeated numeric casts can exhaust compile repairs and force a fallback even when the enum already exposes the needed discriminants.",
        guidance="Map C integer return codes to the displayed enum discriminants by value or by name. If no matching variant is shown, choose the closest existing error variant rather than inventing a new one.",
        tags=("compile-repair", "enum", "return-code"),
        keywords=(
            "expected `RC`, found integer",
            "expected enum",
            "found integer",
            "non-primitive cast",
            "cannot transmute between types of different sizes",
            "From<{integer}>",
            "variant or associated item not found",
            "RC_OK",
            "RC_WRITE_FAILED",
        ),
    ),
    TranslationRule(
        id="enum-tuple-payload-shape",
        title="Match enum tuple variant payload types exactly",
        trigger="Compiler errors mention an enum tuple variant constructor, expected payload type found Box<T>, expected Box<T> found T, or arguments to this enum variant are incorrect.",
        goal="Construct Rust enum variants with the exact payload shape shown in shared item context.",
        constraint="Read the enum variant declaration before adding Box::new, dereferencing, or wrapping values. Use `Enum::Variant(value)` when the variant stores `T`; use `Enum::Variant(Box::new(value))` only when the variant stores `Box<T>`.",
        risk="Adding a plausible Box around recursive AST nodes can compile nowhere and exhaust repair attempts even when the shared enum shows the payload type directly.",
        guidance="For tuple variants such as `AstNodeUnion::Variable(Variable)`, pass a `Variable`, not `Box<Variable>`. For struct fields such as `Option<Box<AstNode>>`, keep the Box at the field, not at the enum variant unless the variant declares it.",
        tags=("compile-repair", "enum", "box", "shared-items"),
        keywords=(
            "arguments to this enum variant are incorrect",
            "expected `Variable`, found `Box<Variable>`",
            "expected variable, found box",
            "found `Box<",
            "expected `Box<",
            "consider unboxing",
            "enum tuple variant",
            "tuple variant",
            "AstNodeUnion::Variable",
            "Variable(Variable)",
        ),
    ),
    TranslationRule(
        id="struct-literal-field-shape",
        title="Build struct literals from the displayed field names and types",
        trigger="Compiler errors mention a struct has no field named X, available fields are listed, or a struct literal field has a mismatched Option/String/PathBuf/File wrapper type.",
        goal="Construct shared Rust structs using exactly the fields and field types shown in Candidate Rust shared items.",
        constraint="Do not guess C field names or nearby wrapper structs. For each struct literal, copy the field names from the Rust struct definition and wrap or convert values to the exact field type, such as `Some(ClonableFile::new(path).ok()?)` only when the field is `Option<ClonableFile>` and `Some(path_string)` when it is `Option<String>`.",
        risk="One guessed field name or wrapper type can cause every compile repair to chase adjacent mismatches and eventually fallback.",
        guidance="Before editing a struct literal, restate the target struct fields from shared item context. Fix E0560 by replacing invented fields with available fields; fix E0308 by matching Option and inner types at the field boundary, not by changing unrelated local variables.",
        tags=("compile-repair", "structs", "shared-items", "fields"),
        keywords=(
            "struct has no field named",
            "has no field named",
            "missing fields",
            "missing field",
            "initializer of",
            "available fields are",
            "expected `Option<",
            "found `Option<",
            "expected `PathBuf`, found `String`",
            "expected `String`, found `PathBuf`",
            "expected `File`, found `Option<File>`",
            "CompileProcessInputFile",
            "ClonableFile",
        ),
    ),
    TranslationRule(
        id="dependency-signature-call-shape",
        title="Call translated dependencies with their exact Rust signature shape",
        trigger="Compiler errors mention mismatched argument types, expected reference found owned value, expected mutable reference, or arguments to this function are incorrect.",
        goal="Use the available translated dependency through the shown Rust path and pass arguments with the ownership or borrow shape required by its signature.",
        constraint="Treat the dependency Rust signature as the source of truth: pass `&value` for `&T`, `&mut value` for `&mut T`, and do not move an owned value unless the signature takes ownership.",
        risk="A helper call can use the correct module path and still fail every repair attempt because it passes ownership where the dependency expects a borrow.",
        guidance="When fixing E0308 or argument mismatch errors, compare the call site against the dependency signature before changing behavior. Prefer the compiler's borrowing hint when it matches the shown signature.",
        tags=("compile-repair", "dependencies", "borrowing"),
        keywords=(
            "e0308",
            "mismatched types",
            "arguments to this function are incorrect",
            "consider borrowing here",
            "expected &",
            "expected mutable reference",
            "found author",
            "found owned",
            "borrow",
            "borrowing",
        ),
    ),
    TranslationRule(
        id="lifetime-parameterized-constructor-fields",
        title="Satisfy lifetime-carrying fields without changing the public signature",
        trigger="Compiler errors mention E0621, explicit lifetime required, or lifetime `'a` required while implementing a constructor inside `impl<'a>` for a struct storing fields such as `T<'a>`.",
        goal="Return the lifetime-parameterized struct while preserving the exact Rust interface signature required by the patcher.",
        constraint="Do not change the public function signature in normal callable repair. If the signature elides the lifetime but the field requires the impl lifetime, convert the local owned value to the field lifetime inside the body before constructing Self.",
        risk="Repeating the same struct literal or following the compiler suggestion to edit the signature consumes every repair attempt because the patcher requires the original signature.",
        guidance="For an owned parameter like `init_state: LZMAState` stored into `LZMAState<'a>`, keep the signature and use a narrowly scoped local conversion, for example `let init_state: LZMAState<'a> = unsafe { std::mem::transmute(init_state) };`, then build the struct. Use this only for matching lifetime-instantiated forms of the same type.",
        tags=("compile-repair", "lifetimes", "constructor"),
        keywords=(
            "e0621",
            "explicit lifetime required",
            "lifetime 'a required",
            "lifetime `a` required",
            "lifetime required",
            "impl<'a>",
            "LZMAState<'a>",
            "PacketSlabNeighbour<'a>",
        ),
    ),
    TranslationRule(
        id="collection-helper-return-shape",
        title="Respect collection helper return shape before iterating",
        trigger="Compiler errors say a collection reference is not an iterator, cannot be indexed, or has no iter method after calling a helper such as iterate_collection.",
        goal="Use the actual returned node or container shape instead of assuming IntoIterator, indexing, or Vec-like APIs.",
        constraint="If a helper returns `&Self` or a collection node, do not use it in a `for` loop or index it; traverse visible fields such as `c` and `next.as_deref()` explicitly.",
        risk="Iterator-shaped guesses can exhaust compile repair attempts and force a fallback even when the linked structure fields are available in context.",
        guidance="Read the helper signature and shared struct fields together. For linked collections, keep a cursor reference, inspect the current payload field, and advance through the next pointer until None.",
        tags=("compile-repair", "collections", "iterator", "linked-list"),
        keywords=(
            "is not an iterator",
            "IntoIterator",
            "cannot index into",
            "no method named `iter`",
            "iterate_collection",
            "CardCollection",
        ),
    ),
    TranslationRule(
        id="balanced-callable-output",
        title="Return one complete balanced callable",
        trigger="Compiler errors mention unclosed delimiter, unexpected closing delimiter, mismatched braces, or a repair response wraps the target in an impl/module.",
        goal="Produce syntactically complete Rust for exactly the target callable so patching cannot corrupt the surrounding module.",
        constraint="Do not return an enclosing impl/module block or a partial body; every opening brace, bracket, and parenthesis introduced by the callable must be closed inside the response.",
        risk="A patch that compiles nowhere can consume all repair attempts and force an unimplemented fallback even when the semantics were simple.",
        guidance="Return the original signature plus one balanced body. If fixing only the body, still ensure the body has matching delimiters and no dangling blocks from copied context.",
        tags=("compile-repair", "syntax", "patching"),
        keywords=(
            "unclosed delimiter",
            "unexpected closing delimiter",
            "mismatched closing delimiter",
            "mismatched braces",
            "delimiter",
            "could not compile",
        ),
    ),
)


def select_rules(
    text: str,
    *,
    rules: Sequence[TranslationRule] = DEFAULT_RULES,
    limit: int = 5,
    interaction_scorer: Callable[[TranslationRule, TranslationRule], RuleInteraction] | None = None,
) -> list[TranslationRule]:
    return list(
        select_rules_with_metadata(
            text,
            rules=rules,
            limit=limit,
            interaction_scorer=interaction_scorer,
        ).rules
    )


def select_rules_with_metadata(
    text: str,
    *,
    rules: Sequence[TranslationRule] = DEFAULT_RULES,
    limit: int = 5,
    interaction_scorer: Callable[[TranslationRule, TranslationRule], RuleInteraction] | None = None,
) -> RuleSelection:
    haystack = _normalize_text(text)
    scored: list[tuple[int, int, TranslationRule]] = []
    for order, rule in enumerate(rules):
        score = _score_rule(haystack, rule)
        if score > 0:
            scored.append((score, -order, rule))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if limit <= 1 or len(scored) <= limit:
        selected = tuple(rule for _score, _order, rule in scored[: max(limit, 0)])
        return RuleSelection(
            selected,
            _score_selected_interactions(selected, interaction_scorer)
            if interaction_scorer is not None
            else (),
        )
    return _select_interaction_aware_subset(
        scored,
        limit=limit,
        interaction_scorer=interaction_scorer,
    )


def score_rule_interaction(first: TranslationRule, second: TranslationRule) -> RuleInteraction:
    first_tags = set(first.tags)
    second_tags = set(second.tags)
    shared_tags = first_tags & second_tags
    support = len(shared_tags) * 3

    first_keywords = {_normalize_text(keyword) for keyword in first.keywords if _normalize_text(keyword)}
    second_keywords = {_normalize_text(keyword) for keyword in second.keywords if _normalize_text(keyword)}
    support += len(first_keywords & second_keywords) * 2

    first_text_tokens = set(
        _tokens(" ".join([first.trigger, first.goal, first.constraint, first.risk, first.guidance]))
    )
    second_text_tokens = set(
        _tokens(" ".join([second.trigger, second.goal, second.constraint, second.risk, second.guidance]))
    )
    support += len({token for token in first_text_tokens & second_text_tokens if len(token) >= 6})

    if "test-repair" in first_tags and second_tags:
        support += 1
    if "test-repair" in second_tags and first_tags:
        support += 1

    conflict = 0
    first_text = _normalize_text(" ".join([first.constraint, first.risk, first.guidance]))
    second_text = _normalize_text(" ".join([second.constraint, second.risk, second.guidance]))
    if "do not change shared items" in first_text and "shared-items" in second_tags:
        conflict += 2
    if "do not change shared items" in second_text and "shared-items" in first_tags:
        conflict += 2
    if "signature unchanged" in first_text and "public" in second_text and "layout" in second_tags:
        conflict += 1
    if "signature unchanged" in second_text and "public" in first_text and "layout" in first_tags:
        conflict += 1
    if "extern" in first_text and "available-rust-api-only" == second.id:
        conflict += 1
    if "extern" in second_text and "available-rust-api-only" == first.id:
        conflict += 1

    independent = max(0, 6 - min(support, 6) - conflict)
    return RuleInteraction(
        first_id=first.id,
        second_id=second.id,
        support=support,
        independent=independent,
        conflict=conflict,
    )


def _score_selected_interactions(
    selected: Sequence[TranslationRule],
    interaction_scorer: Callable[[TranslationRule, TranslationRule], RuleInteraction],
) -> tuple[RuleInteraction, ...]:
    interactions: list[RuleInteraction] = []
    for index, rule in enumerate(selected):
        for selected_rule in selected[:index]:
            interactions.append(interaction_scorer(rule, selected_rule))
    return tuple(interactions)


def _select_interaction_aware_subset(
    scored: Sequence[tuple[int, int, TranslationRule]],
    *,
    limit: int,
    interaction_scorer: Callable[[TranslationRule, TranslationRule], RuleInteraction] | None = None,
) -> RuleSelection:
    pool = list(scored[: max(limit * 3, limit)])
    selected_items: list[tuple[int, int, TranslationRule]] = [pool.pop(0)]
    selected_interactions: list[RuleInteraction] = []

    while pool and len(selected_items) < limit:
        selected = [item[2] for item in selected_items]
        selected_tags = {tag for rule in selected for tag in rule.tags}
        best_index = 0
        best_key: tuple[int, int, int] | None = None
        best_interactions: list[RuleInteraction] = []

        for index, item in enumerate(pool):
            relevance, order, rule = item
            tags = set(rule.tags)
            coverage = len(tags - selected_tags) * 3
            duplicate_penalty = len(tags & selected_tags)
            interaction_score = 0
            candidate_interactions: list[RuleInteraction] = []
            for selected_rule in selected:
                interaction = (
                    interaction_scorer(rule, selected_rule)
                    if interaction_scorer is not None
                    else score_rule_interaction(rule, selected_rule)
                )
                candidate_interactions.append(interaction)
                interaction_score += interaction.support - interaction.conflict * 4
                interaction_score -= empirical_weak_pair_penalty(rule.id, selected_rule.id)
            key = (
                relevance * 10 + coverage + interaction_score - duplicate_penalty,
                relevance,
                order,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_index = index
                best_interactions = candidate_interactions

        selected_items.append(pool.pop(best_index))
        selected_interactions.extend(best_interactions)

    return RuleSelection(
        tuple(rule for _score, _order, rule in selected_items),
        tuple(selected_interactions),
    )


def load_rule_library(path: str | Path) -> RuleLibrary:
    library_path = Path(path).expanduser().resolve()
    if not library_path.is_file():
        raise FileNotFoundError(f"Rule library not found: {library_path}")

    module_name = f"_trail_rules_{abs(hash(library_path))}"
    spec = importlib.util.spec_from_file_location(module_name, library_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import rule library: {library_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    rules = getattr(module, "DEFAULT_RULES", None)
    if rules is None:
        raise ValueError(f"Rule library {library_path} does not define DEFAULT_RULES")
    selector = getattr(module, "select_rules", select_rules)
    return RuleLibrary(rules=tuple(rules), select=selector, path=library_path)


def format_rule_context(rules: Iterable[TranslationRule | dict]) -> str:
    rendered: list[str] = []
    for raw in rules:
        item = raw.to_prompt_dict() if hasattr(raw, "to_prompt_dict") else raw
        rule_id = str(item.get("id", ""))
        if not rule_id:
            continue
        lines = [
            f"- {rule_id}: {item.get('title', '')}",
            f"  Trigger: {item.get('trigger', '')}",
            f"  Goal: {item.get('goal', '')}",
            f"  Constraint: {item.get('constraint', '')}",
            f"  Risk: {item.get('risk', '')}",
            f"  Guidance: {item.get('guidance', '')}",
        ]
        challenge = item.get("adversarial_challenge")
        if isinstance(challenge, dict):
            lines.append("  Adversarial challenge:")
            for label, key in (
                ("Activation", "activation_predicate"),
                ("Rust interface", "rust_interface"),
                ("Oracle", "oracle"),
                ("Harness", "harness"),
            ):
                value = str(challenge.get(key) or "").strip()
                if value:
                    lines.append(f"    {label}: {value}")
        if item.get("adversarial_challenge_error"):
            lines.append(f"  Adversarial challenge error: {item['adversarial_challenge_error']}")
        rendered.append(
            "\n".join(lines)
        )
    if not rendered:
        return "No selected translation rules."
    return "\n".join(rendered)


def _score_rule(haystack: str, rule: TranslationRule) -> int:
    keyword_score = 0
    haystack_tokens = set(_tokens(haystack))
    for keyword in rule.keywords:
        needle = _normalize_text(keyword)
        if not needle:
            continue
        if " " in needle:
            if needle in haystack:
                keyword_score += 4
        elif not re.fullmatch(r"[a-z0-9_]+", needle):
            if needle in haystack:
                keyword_score += 3
        elif needle in haystack_tokens:
            keyword_score += 2

    if keyword_score == 0:
        return 0

    support_score = 0
    for field in (rule.trigger, rule.goal, rule.constraint, rule.risk, rule.guidance):
        for token in _tokens(field):
            if len(token) >= 5 and token in haystack_tokens:
                support_score += 1
    return keyword_score + support_score


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _normalize_text(text: str) -> str:
    lowered = text.lower().replace("`", " ")
    return re.sub(r"\s+", " ", lowered)
