from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    from .prompts import ALIGNMENT_SYSTEM_PROMPT, build_alignment_prompt
except ImportError:
    from prompts import ALIGNMENT_SYSTEM_PROMPT, build_alignment_prompt


RELATIONSHIPS = {
    "direct",
    "composed_from_multiple",
    "split_from_one",
    "semantic_rewrite",
    "no_safe_binding",
}


@dataclass(frozen=True)
class CBinding:
    functions: tuple[Any, ...]
    relationship: str
    confidence: float
    rationale: str
    source: str

    def first_function(self):
        return self.functions[0] if self.functions else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "relationship": self.relationship,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "source": self.source,
            "functions": [
                {
                    "name": getattr(function, "name", ""),
                    "file_path": getattr(function, "file_path", ""),
                    "id": AlignmentPlanner.c_function_id(function),
                }
                for function in self.functions
            ],
        }


@dataclass
class AlignmentResult:
    bindings: dict[tuple, CBinding]
    prompts: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)


class AlignmentPlanner:
    CANDIDATE_CODE_TOTAL_CHARS = 80000
    CANDIDATE_CODE_PER_FUNCTION_CHARS = 4000
    ALIGNMENT_MAX_OUTPUT_TOKENS = 4096

    def __init__(self, *, agent, metadata: dict, confidence_threshold: float = 0.65):
        self.agent = agent
        self.metadata = metadata
        self.confidence_threshold = confidence_threshold

    def plan_module(self, callables, c_functions, rule_candidates) -> AlignmentResult:
        stats = self._empty_stats()
        callables_by_label = {self._callable_label(item): item for item in callables}
        functions_by_id = {self.c_function_id(function): function for function in c_functions}
        prompt = self._build_basic_prompt(callables, c_functions, rule_candidates)
        stats["alignment_llm_calls"] += 1
        prompt_record: dict[str, Any] = {
            "prompt": prompt,
            "raw_response": "",
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            },
        }
        try:
            result = self._translate_alignment(prompt)
        except Exception as exc:
            stats["alignment_llm_errors"] += 1
            prompt_record["error"] = str(exc)
            accepted: dict[tuple, CBinding] = {}
        else:
            if result.usage.prompt_tokens is not None:
                stats["prompt_tokens"] += result.usage.prompt_tokens
            if result.usage.completion_tokens is not None:
                stats["completion_tokens"] += result.usage.completion_tokens
            if result.usage.total_tokens is not None:
                stats["total_tokens"] += result.usage.total_tokens

            prompt_record["raw_response"] = result.raw_text
            prompt_record["usage"] = {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
            }
            data = self._extract_json(result.raw_text)
            if data is None:
                stats["malformed_alignment_responses"] += 1
                accepted = {}
            else:
                accepted = self._validate_bindings(data, callables_by_label, functions_by_id, stats)

        for item in callables:
            item_id = self._callable_id(item)
            if item_id not in accepted:
                accepted[item_id] = self._fallback_binding(item, rule_candidates)
                if accepted[item_id].source == "rule_fallback":
                    stats["rule_fallback_callables"] += 1
                else:
                    stats["empty_fallback_callables"] += 1

        return AlignmentResult(
            bindings=accepted,
            prompts=[prompt_record],
            stats=stats,
        )

    def _translate_alignment(self, prompt: str):
        old_max_tokens = getattr(self.agent, "max_tokens", None)
        should_restore = isinstance(old_max_tokens, int) and old_max_tokens > self.ALIGNMENT_MAX_OUTPUT_TOKENS
        if should_restore:
            self.agent.max_tokens = self.ALIGNMENT_MAX_OUTPUT_TOKENS
        try:
            return self.agent.translate(ALIGNMENT_SYSTEM_PROMPT, prompt)
        finally:
            if should_restore:
                self.agent.max_tokens = old_max_tokens

    def _validate_bindings(
        self,
        data: dict[str, Any],
        callables_by_label: dict[str, Any],
        functions_by_id: dict[str, Any],
        stats: dict[str, int],
    ) -> dict[tuple, CBinding]:
        raw_bindings = data.get("bindings")
        if raw_bindings is None:
            raw_bindings = self._bindings_from_shorthand(data)
        if not isinstance(raw_bindings, list):
            stats["malformed_alignment_responses"] += 1
            return {}

        accepted: dict[tuple, CBinding] = {}
        for raw in raw_bindings:
            if not isinstance(raw, dict):
                stats["rejected_bindings"] += 1
                continue

            label = raw.get("rust_callable")
            item = callables_by_label.get(label)
            relationship = raw.get("relationship")
            confidence = raw.get("confidence")
            c_function_ids = raw.get("c_functions", [])
            rationale = raw.get("rationale", "")

            if item is None or relationship not in RELATIONSHIPS:
                stats["rejected_bindings"] += 1
                continue
            if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
                stats["rejected_bindings"] += 1
                continue
            if float(confidence) < self.confidence_threshold:
                stats["rejected_bindings"] += 1
                continue
            if not isinstance(c_function_ids, list):
                stats["rejected_bindings"] += 1
                continue

            functions = []
            for function_id in c_function_ids:
                function = functions_by_id.get(function_id)
                if function is None:
                    stats["rejected_bindings"] += 1
                    functions = []
                    break
                functions.append(function)
            else:
                item_id = self._callable_id(item)
                accepted[item_id] = CBinding(
                    functions=tuple(functions),
                    relationship=relationship,
                    confidence=float(confidence),
                    rationale=str(rationale),
                    source="llm",
                )
                stats["llm_bound_callables"] += 1
        return accepted

    def _bindings_from_shorthand(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for rust_callable, c_functions in data.items():
            if isinstance(c_functions, str):
                function_ids = [c_functions]
            elif isinstance(c_functions, list) and all(isinstance(item, str) for item in c_functions):
                function_ids = c_functions
            else:
                continue
            out.append(
                {
                    "rust_callable": rust_callable,
                    "c_functions": function_ids,
                    "relationship": "direct" if len(function_ids) <= 1 else "composed_from_multiple",
                    "confidence": 1.0,
                    "rationale": "Model returned shorthand callable-to-function mapping.",
                }
            )
        return out

    def _fallback_binding(self, item, rule_candidates) -> CBinding:
        candidates = rule_candidates.get(self._callable_label(item), [])
        if len(candidates) == 1:
            return CBinding(
                functions=(candidates[0],),
                relationship="direct",
                confidence=1.0,
                rationale="Deterministic rule fallback selected one candidate.",
                source="rule_fallback",
            )
        return CBinding(
            functions=(),
            relationship="no_safe_binding",
            confidence=0.0,
            rationale="No validated alignment binding.",
            source="empty_fallback",
        )

    def _extract_json(self, text: str) -> Optional[dict[str, Any]]:
        json_blocks = re.findall(r"```json\s*\n(.*?)\n\s*```", text, re.DOTALL)
        candidates = json_blocks or [text]
        for candidate in candidates:
            try:
                data = json.loads(candidate.strip())
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
        return None

    def _build_basic_prompt(self, callables, c_functions, rule_candidates) -> str:
        callable_payload = [
            {
                "id": self._callable_label(item),
                "signature": item.signature,
                "stub": item.stub,
                "param_count": item.param_count,
            }
            for item in callables
        ]
        remaining_code_chars = self.CANDIDATE_CODE_TOTAL_CHARS
        function_payload = []
        for function in c_functions:
            code = str(getattr(function, "code", "") or "")
            code, remaining_code_chars, truncated = self._alignment_code_snippet(code, remaining_code_chars)
            payload = {
                "id": self.c_function_id(function),
                "name": getattr(function, "name", ""),
                "file_path": getattr(function, "file_path", ""),
                "param_count": getattr(function, "param_count", 0),
                "calls": list(getattr(function, "calls", ())),
                "code": code,
            }
            if truncated:
                payload["code_truncated"] = True
            function_payload.append(payload)
        inventory = [
            {
                "id": self.c_function_id(function),
                "rule_candidate_for": [
                    label for label, functions in rule_candidates.items() if function in functions
                ],
            }
            for function in c_functions
        ]
        module = callables[0].module if callables else ""
        return build_alignment_prompt(
            module=module,
            callables=callable_payload,
            candidate_functions=function_payload,
            type_context=[],
            function_inventory=inventory,
        )

    def _alignment_code_snippet(self, code: str, remaining_code_chars: int) -> tuple[str, int, bool]:
        if not code:
            return "", remaining_code_chars, False
        if remaining_code_chars <= 0:
            return "", 0, True

        limit = min(self.CANDIDATE_CODE_PER_FUNCTION_CHARS, remaining_code_chars)
        if len(code) <= limit:
            return code, remaining_code_chars - len(code), False

        marker = "\n/* ... alignment candidate code truncated ... */"
        body_limit = max(0, limit - len(marker))
        return code[:body_limit].rstrip() + marker, remaining_code_chars - limit, True

    @staticmethod
    def c_function_id(function) -> str:
        return f"{getattr(function, 'file_path', '')}::{getattr(function, 'name', '')}"

    def _c_function_id(self, function) -> str:
        return self.c_function_id(function)

    def _callable_id(self, item) -> tuple:
        name = item.name if item.kind == "fn" or not item.type_name else f"{item.type_name}::{item.name}"
        return (item.module, item.kind, name)

    def _callable_label(self, item) -> str:
        return "::".join(str(part) for part in self._callable_id(item))

    def _empty_stats(self) -> dict[str, int]:
        return {
            "alignment_llm_calls": 0,
            "llm_bound_callables": 0,
            "rule_fallback_callables": 0,
            "empty_fallback_callables": 0,
            "rejected_bindings": 0,
            "malformed_alignment_responses": 0,
            "alignment_llm_errors": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
