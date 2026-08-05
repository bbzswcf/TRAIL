import unittest
from dataclasses import dataclass

from c2rust_translator.alignment_planner import AlignmentPlanner, CBinding
from c2rust_translator.prompts import build_translate_prompt, format_alignment_block
from c2rust_translator.rbench_interface import Callable
from c2rust_translator.translation_agent import TranslationResult, Usage


@dataclass(frozen=True)
class FakeCFunction:
    name: str
    file_path: str
    code: str
    param_count: int = 0
    calls: tuple[str, ...] = ()
    name_key: str = ""
    name_tokens: tuple[str, ...] = ()
    file_key: str = ""
    original_file: str | None = None
    original_key: str = ""


class FakeAgent:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def translate(self, system, prompt):
        self.calls.append({"system": system, "prompt": prompt, "max_tokens": getattr(self, "max_tokens", None)})
        return TranslationResult(
            code=self.text,
            raw_text=self.text,
            elapsed_seconds=0.01,
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )


class RaisingAgent:
    def __init__(self, exc):
        self.exc = exc
        self.calls = []

    def translate(self, system, prompt):
        self.calls.append({"system": system, "prompt": prompt})
        raise self.exc


def callable(name, *, module="buffer", kind="fn", type_name=None):
    return Callable(
        module=module,
        kind=kind,
        name=name,
        type_name=type_name,
        file_name=f"{module}.rs",
        signature=f"pub fn {name}()",
        signature_key=f"pub fn {name}()",
        param_count=0,
        stub=f"pub fn {name}() {{ unimplemented!() }}",
    )


class AlignmentPlannerTest(unittest.TestCase):
    def test_accepts_multiple_c_functions_for_one_rust_callable(self):
        rust_item = callable("push", kind="method", type_name="Buffer")
        c_put = FakeCFunction(name="buffer_put", file_path="buffer.c", code="void buffer_put() {}")
        c_advance = FakeCFunction(name="advance", file_path="buffer.c", code="void advance() {}")
        agent = FakeAgent(
            """
            ```json
            {
              "bindings": [
                {
                  "rust_callable": "buffer::method::Buffer::push",
                  "c_functions": ["buffer.c::buffer_put", "buffer.c::advance"],
                  "relationship": "composed_from_multiple",
                  "confidence": 0.91,
                  "rationale": "push inserts and advances the cursor"
                }
              ]
            }
            ```
            """
        )

        planner = AlignmentPlanner(agent=agent, metadata={"files": {}}, confidence_threshold=0.65)
        result = planner.plan_module([rust_item], [c_put, c_advance], {"buffer::method::Buffer::push": [c_put]})

        binding = result.bindings[("buffer", "method", "Buffer::push")]
        self.assertEqual(binding.relationship, "composed_from_multiple")
        self.assertEqual([fn.name for fn in binding.functions], ["buffer_put", "advance"])
        self.assertEqual(binding.source, "llm")
        self.assertEqual(result.stats["llm_bound_callables"], 1)

    def test_low_confidence_and_unknown_c_functions_fall_back(self):
        rust_item = callable("decode", kind="method", type_name="Amp")
        c_decode = FakeCFunction(name="amp_decode", file_path="amp.c", code="int amp_decode() {}")
        agent = FakeAgent(
            """
            {
              "bindings": [
                {
                  "rust_callable": "buffer::method::Amp::decode",
                  "c_functions": ["amp.c::missing"],
                  "relationship": "direct",
                  "confidence": 0.20,
                  "rationale": "weak guess"
                }
              ]
            }
            """
        )

        planner = AlignmentPlanner(agent=agent, metadata={"files": {}}, confidence_threshold=0.65)
        result = planner.plan_module([rust_item], [c_decode], {"buffer::method::Amp::decode": [c_decode]})

        binding = result.bindings[("buffer", "method", "Amp::decode")]
        self.assertEqual(binding.relationship, "direct")
        self.assertEqual([fn.name for fn in binding.functions], ["amp_decode"])
        self.assertEqual(binding.source, "rule_fallback")
        self.assertGreaterEqual(result.stats["rejected_bindings"], 1)

    def test_malformed_response_uses_empty_fallback_when_no_rule_candidate_exists(self):
        rust_item = callable("new", kind="method", type_name="Buffer")
        agent = FakeAgent("not json")

        planner = AlignmentPlanner(agent=agent, metadata={"files": {}}, confidence_threshold=0.65)
        result = planner.plan_module([rust_item], [], {"buffer::method::Buffer::new": []})

        binding = result.bindings[("buffer", "method", "Buffer::new")]
        self.assertEqual(binding.relationship, "no_safe_binding")
        self.assertEqual(binding.functions, ())
        self.assertEqual(binding.source, "empty_fallback")
        self.assertEqual(result.stats["malformed_alignment_responses"], 1)

    def test_alignment_llm_error_uses_rule_fallback(self):
        rust_item = callable("decode", kind="method", type_name="Amp")
        c_decode = FakeCFunction(name="amp_decode", file_path="amp.c", code="int amp_decode() {}")
        agent = RaisingAgent(RuntimeError("Translation failed after non-retryable HTTP error: 400"))

        planner = AlignmentPlanner(agent=agent, metadata={"files": {}}, confidence_threshold=0.65)
        result = planner.plan_module([rust_item], [c_decode], {"buffer::method::Amp::decode": [c_decode]})

        binding = result.bindings[("buffer", "method", "Amp::decode")]
        self.assertEqual(binding.relationship, "direct")
        self.assertEqual([fn.name for fn in binding.functions], ["amp_decode"])
        self.assertEqual(binding.source, "rule_fallback")
        self.assertEqual(result.stats["alignment_llm_calls"], 1)
        self.assertEqual(result.stats["alignment_llm_errors"], 1)
        self.assertEqual(result.stats["rule_fallback_callables"], 1)
        self.assertIn("Translation failed after non-retryable HTTP error", result.prompts[0]["error"])

    def test_alignment_prompt_truncates_large_candidate_code(self):
        rust_item = callable("decode", kind="method", type_name="Amp")
        long_code = "int amp_decode(void) {\n" + "  value += 1;\n" * 5000 + "}\n"
        c_decode = FakeCFunction(name="amp_decode", file_path="amp.c", code=long_code)
        agent = FakeAgent('{"bindings": []}')

        planner = AlignmentPlanner(agent=agent, metadata={"files": {}}, confidence_threshold=0.65)
        result = planner.plan_module([rust_item], [c_decode], {"buffer::method::Amp::decode": [c_decode]})

        prompt = agent.calls[0]["prompt"]
        self.assertIn("alignment candidate code truncated", prompt)
        self.assertIn('"code_truncated": true', prompt)
        self.assertLess(len(prompt), len(long_code))
        self.assertEqual(result.bindings[("buffer", "method", "Amp::decode")].source, "rule_fallback")

    def test_alignment_call_uses_small_output_budget_and_restores_agent(self):
        rust_item = callable("decode", kind="method", type_name="Amp")
        c_decode = FakeCFunction(name="amp_decode", file_path="amp.c", code="int amp_decode() {}")
        agent = FakeAgent('{"bindings": []}')
        agent.max_tokens = 128000

        planner = AlignmentPlanner(agent=agent, metadata={"files": {}}, confidence_threshold=0.65)
        planner.plan_module([rust_item], [c_decode], {"buffer::method::Amp::decode": [c_decode]})

        self.assertEqual(agent.calls[0]["max_tokens"], 4096)
        self.assertEqual(agent.max_tokens, 128000)

    def test_accepts_shorthand_mapping_object_from_model(self):
        rust_item = callable("leftpad")
        c_leftpad = FakeCFunction(name="leftpad", file_path="src/leftpad.i", code="size_t leftpad() {}")
        agent = FakeAgent('{"buffer::fn::leftpad": "src/leftpad.i::leftpad"}')

        planner = AlignmentPlanner(agent=agent, metadata={"files": {}}, confidence_threshold=0.65)
        result = planner.plan_module([rust_item], [c_leftpad], {"buffer::fn::leftpad": []})

        binding = result.bindings[("buffer", "fn", "leftpad")]
        self.assertEqual(binding.relationship, "direct")
        self.assertEqual(binding.source, "llm")
        self.assertEqual(binding.functions, (c_leftpad,))
        self.assertEqual(result.stats["llm_bound_callables"], 1)


class AlignmentPromptTest(unittest.TestCase):
    def test_format_alignment_block_includes_relationship_rationale_and_each_c_function(self):
        binding = CBinding(
            functions=(
                FakeCFunction(name="a", file_path="x.c", code="int a(void) { return 1; }", calls=("b",)),
                FakeCFunction(name="b", file_path="x.c", code="int b(void) { return 2; }"),
            ),
            relationship="composed_from_multiple",
            confidence=0.9,
            rationale="a delegates to b",
            source="llm",
        )

        text = format_alignment_block(binding)

        self.assertIn("composed_from_multiple", text)
        self.assertIn("a delegates to b", text)
        self.assertIn("x.c::a", text)
        self.assertIn("x.c::b", text)
        self.assertIn("Calls: b", text)

    def test_translate_prompt_uses_alignment_block_instead_of_single_c_code(self):
        binding = CBinding(
            functions=(),
            relationship="no_safe_binding",
            confidence=0.0,
            rationale="no validated binding",
            source="empty_fallback",
        )

        prompt = build_translate_prompt(
            contract="pub fn new() -> Self { unimplemented!() }",
            alignment=binding,
            deps=[],
            file_context=[],
        )

        self.assertIn("C alignment", prompt)
        self.assertIn("no_safe_binding", prompt)
        self.assertNotIn("Relevant C code:", prompt)


if __name__ == "__main__":
    unittest.main()
