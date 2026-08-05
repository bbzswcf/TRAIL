import json

try:
    from .insights import format_insight_context
except ImportError:
    from insights import format_insight_context


SYSTEM_PROMPT = """You are an expert in translating C code to idiomatic Rust code.

Core contract:
- Preserve the observable C behavior while producing idiomatic, simple, safe Rust when the public interface allows it.
- Respect the supplied public Rust interface exactly: keep the requested function, method, type, visibility, module path, argument ownership, reference mutability, lifetimes, return type, and public field shape unchanged unless the user prompt explicitly asks for a different artifact.
- Return exactly the artifact requested by the user prompt, such as one Rust callable, one complete replacement snippet, one JSON object, or one suffix block.
- Return raw content only: no markdown fences, no markdown bullets around code, no prose explanation, and no surrounding commentary unless the user prompt explicitly requests it.

Repository constraints:
- Use only the C code, Rust contract, shared items, translated dependencies, imports, fixtures, and project context provided in the prompt.
- Do not invent Rust helpers, traits, modules, fixtures, files, feature flags, or public APIs that are not present in the provided context or requested by the prompt.
- Do not add new extern declarations, FFI bindings, unsafe shims, path aliases, or placeholder stubs for missing C helpers unless the prompt explicitly declares that interface.
- Treat provided translation insights as constraints. Apply them when they match the current callable, and do not ignore them in favor of a superficially simpler rewrite.
- Prefer existing translated dependency paths and signatures exactly as provided; pass owned values, shared references, and mutable references according to those signatures.
- Do not assume Clone, Copy, Default, From, Into, serde, iterator traits, or other helper impls exist unless the provided Rust context shows them.
"""

ALIGNMENT_SYSTEM_PROMPT = """You align Rust interface callables to candidate C functions.
Return strict JSON only. Do not translate code."""


def format_deps(deps: list[dict]) -> str:
    if not deps:
        return "No translated dependencies available."

    parts: list[str] = []
    for dep in deps:
        label = dep["label"]
        if dep.get("c_name"):
            label = f"{label} <= {dep['c_name']}"
        parts.append(f"- {label}")
        if dep.get("rust_path"):
            parts.append(f"  Rust call path: `{dep['rust_path']}`")
        if dep.get("signature"):
            parts.append("  Rust signature:")
            parts.append(f"```rust\n{dep['signature']}\n```")
        parts.append(f"```rust\n{dep['rust_code']}\n```")
    return "\n".join(parts)


def format_cycle_context(items: list[dict]) -> str:
    if not items:
        return "No cyclic peers."

    parts: list[str] = []
    for item in items:
        parts.append(f"- {item['label']}")
        parts.append(f"  Contract:\n```rust\n{item['contract']}\n```")
        if item.get("c_code"):
            parts.append(f"  C code:\n```c\n{item['c_code']}\n```")
    return "\n".join(parts)


def format_file_context(items: list[dict]) -> str:
    if not items:
        return "No extra C file context."

    parts: list[str] = []
    for item in items:
        parts.append(f"- {item['path']}")
        parts.append(f"```c\n{item['code']}\n```")
    return "\n".join(parts)


def build_alignment_prompt(
    *,
    module: str,
    callables: list[dict],
    candidate_functions: list[dict],
    type_context: list[dict] | None = None,
    function_inventory: list[dict] | None = None,
) -> str:
    return f"""Align Rust interface callables for module `{module}` to candidate C functions.

Return JSON only with this shape:
```json
{{
  "bindings": [
    {{
      "rust_callable": "module::kind::name",
      "c_functions": ["file.c::function_name"],
      "relationship": "direct",
      "confidence": 0.91,
      "rationale": "short reason"
    }}
  ]
}}
```

Allowed relationships:
- direct
- composed_from_multiple
- split_from_one
- semantic_rewrite
- no_safe_binding

Rules:
1. Use only Rust callable ids and C function ids listed below.
2. A Rust callable may use zero, one, or many C functions.
3. A C function may be reused by multiple Rust callables.
4. Return no Rust implementation code.
5. Prefer no_safe_binding over a weak name-only guess.

Rust callables:
```json
{json.dumps(callables, indent=2, ensure_ascii=False)}
```

Candidate C functions:
```json
{json.dumps(candidate_functions, indent=2, ensure_ascii=False)}
```

Relevant C types and globals:
```json
{json.dumps(type_context or [], indent=2, ensure_ascii=False)}
```

Project function inventory:
```json
{json.dumps(function_inventory or [], indent=2, ensure_ascii=False)}
```
"""


def format_alignment_block(binding) -> str:
    parts = [
        f"Relationship: {binding.relationship}",
        f"Confidence: {binding.confidence:.2f}",
        f"Source: {binding.source}",
        f"Rationale: {binding.rationale}",
    ]

    if not binding.functions:
        parts.append("No validated C function binding.")
        return "\n".join(parts)

    parts.append("Bound C functions:")
    for function in binding.functions:
        function_id = f"{getattr(function, 'file_path', '')}::{getattr(function, 'name', '')}"
        parts.append(f"- {function_id}")
        calls = getattr(function, "calls", ())
        if calls:
            parts.append(f"  Calls: {', '.join(calls)}")
        code = getattr(function, "code", "")
        if code:
            parts.append(f"```c\n{code}\n```")
    return "\n".join(parts)


def build_translate_prompt(
    *,
    contract: str,
    alignment,
    deps: list[dict],
    cycle_context: list[dict] | None = None,
    callable_category: str = "direct_function",
    file_context: list[dict] | None = None,
    shared_item_context: list[dict] | None = None,
    allow_shared_item_fixes: bool = False,
    insight_context: list[dict] | None = None,
) -> str:
    shared_rules = ""
    return_shape = "Return only the Rust callable code."
    if allow_shared_item_fixes:
        shared_rules = """
Shared item repair is enabled because this project has already accumulated more than 10 callable translation failures.
Use it only when the current compile/translation problem requires changing shared Rust state, type layout, constants, statics, impl blocks, or trait delegation.
"""
        return_shape = """Return strict JSON only with this shape:
```json
{
  "shared_item_fixes": [
    {
      "item_id": "module::kind::name",
      "rust_code": "complete replacement Rust shared item"
    }
  ],
  "rust_code": "complete Rust callable with the original public signature"
}
```

Use only item_id values listed in Candidate Rust shared items.
"""

    return f"""Implement this Rust interface callable.

Contract:
```rust
{contract}
```

Rules:
1. Keep the public signature exactly unchanged.
2. You may only change the callable body and private local details.
3. Do not add new `pub` items.
4. Prefer the standard library unless the project already needs something else.
5. Use the C alignment according to its relationship. If there is no safe binding, use the file context and contract to build the smallest faithful wrapper or implementation you can justify.
6. Do not change shared Rust items unless shared item repair is explicitly enabled below.
7. Do not call Rust helpers that are not shown in this prompt's contract, dependencies, cycle context, shared items, or current type context; inline the logic locally if no Rust helper exists.
8. When a dependency lists `Rust call path`, call exactly that path instead of a bare function name or guessed module path. Match the shown Rust signature exactly: pass `&value` for `&T`, `&mut value` for `&mut T`, and move owned values only when the signature takes ownership.
9. Do not declare new `extern` blocks or FFI bindings inside the callable; if a C library helper is unavailable in Rust, use existing Rust APIs or inline a local Rust implementation.
10. Do not call `.clone()` or assume `Clone` / `Copy` for candidate structs or enums unless the shown shared item already derives or implements it. If ownership duplication is necessary, move the value only when allowed, rebuild it field by field, or use an existing project deep-copy helper shown in context.
11. Return exactly one complete callable with balanced delimiters; do not wrap it in an enclosing `impl` or `mod` block.

{shared_rules}

Callable category:
{callable_category}

C alignment:
{format_alignment_block(alignment)}

Already translated Rust dependencies:
{format_deps(deps)}

Other callables in the same call cycle:
{format_cycle_context(cycle_context or [])}

Relevant C file context:
{format_file_context(file_context or [])}

Translation insights:
{format_insight_context(insight_context or [])}

Candidate Rust shared items:
```json
{json.dumps(shared_item_context or [], indent=2, ensure_ascii=False)}
```

{return_shape}
"""


def build_fix_prompt(
    *,
    contract: str,
    rust_code: str,
    errors: str,
    output: str,
    alignment=None,
    deps: list[dict] | None = None,
    cycle_context: list[dict] | None = None,
    callable_category: str = "direct_function",
    file_context: list[dict] | None = None,
    shared_item_context: list[dict] | None = None,
    insight_context: list[dict] | None = None,
) -> str:
    alignment_text = format_alignment_block(alignment) if alignment is not None else "No C alignment context."
    return f"""Fix this Rust callable so it compiles.

Contract:
```rust
{contract}
```

Current Rust code:
```rust
{rust_code}
```

Structured compiler errors:
```json
{errors}
```

Compiler output tail:
```
{output}
```

Callable category:
{callable_category}

C alignment:
{alignment_text}

Already translated Rust dependencies:
{format_deps(deps or [])}

Other callables in the same call cycle:
{format_cycle_context(cycle_context or [])}

Relevant C file context:
{format_file_context(file_context or [])}

Translation insights:
{format_insight_context(insight_context or [])}

Candidate Rust shared items:
```json
{json.dumps(shared_item_context or [], indent=2, ensure_ascii=False)}
```

Rules:
1. Keep the public signature exactly unchanged.
2. Only change the callable body and private local details.
3. Do not add new `pub` items.
4. Treat Candidate Rust shared items as read-only context in this repair mode.
5. Do not call Rust helpers that are not shown in this prompt's contract, dependencies, cycle context, shared items, or current code; inline the logic locally if no Rust helper exists.
6. When a dependency lists `Rust call path`, call exactly that path instead of a bare function name or guessed module path. Match the shown Rust signature exactly: pass `&value` for `&T`, `&mut value` for `&mut T`, and move owned values only when the signature takes ownership.
7. Do not declare new `extern` blocks or FFI bindings inside the callable; if a C library helper is unavailable in Rust, use existing Rust APIs or inline a local Rust implementation.
8. Do not call `.clone()` or assume `Clone` / `Copy` for candidate structs or enums unless the shown shared item already derives or implements it. If ownership duplication is necessary, move the value only when allowed, rebuild it field by field, or use an existing project deep-copy helper shown in context.
9. Return exactly one complete callable with balanced delimiters; do not wrap it in an enclosing `impl` or `mod` block.

Return only the fixed Rust callable code.
"""


def build_shared_compile_fix_prompt(
    *,
    contract: str,
    rust_code: str,
    errors: str,
    output: str,
    alignment,
    deps: list[dict],
    cycle_context: list[dict] | None = None,
    callable_category: str = "direct_function",
    file_context: list[dict] | None = None,
    shared_item_context: list[dict] | None = None,
    insight_context: list[dict] | None = None,
) -> str:
    return f"""Fix this Rust callable and, only if required, related shared Rust items so the project compiles.

Shared item repair is enabled because the current compiler errors may require changing shared Rust items such as type layout, constants, statics, impl blocks, or trait delegation.

Contract:
```rust
{contract}
```

Current Rust code:
```rust
{rust_code}
```

Structured compiler errors:
```json
{errors}
```

Compiler output tail:
```
{output}
```

Callable category:
{callable_category}

C alignment:
{format_alignment_block(alignment)}

Already translated Rust dependencies:
{format_deps(deps)}

Other callables in the same call cycle:
{format_cycle_context(cycle_context or [])}

Relevant C file context:
{format_file_context(file_context or [])}

Translation insights:
{format_insight_context(insight_context or [])}

Candidate Rust shared items:
```json
{json.dumps(shared_item_context or [], indent=2, ensure_ascii=False)}
```

Return strict JSON only with this shape:
```json
{{
  "shared_item_fixes": [
    {{
      "item_id": "module::kind::name",
      "rust_code": "complete replacement Rust shared item"
    }}
  ],
  "rust_code": "complete Rust callable with the original public signature"
}}
```

Rules:
1. Keep the public callable signature exactly unchanged.
2. Use only item_id values listed in Candidate Rust shared items.
3. Prefer fixing only the callable body when shared state, type layout, or a shared impl/trait is not the cause.
4. Return the smallest patch set needed to compile.
5. Do not call Rust helpers that are not shown in this prompt's contract, dependencies, cycle context, shared items, or current code; inline the logic locally if no Rust helper exists.
6. When a dependency lists `Rust call path`, call exactly that path instead of a bare function name or guessed module path. Match the shown Rust signature exactly: pass `&value` for `&T`, `&mut value` for `&mut T`, and move owned values only when the signature takes ownership.
7. Do not declare new `extern` blocks or FFI bindings inside the callable; if a C library helper is unavailable in Rust, use existing Rust APIs or inline a local Rust implementation.
8. Do not call `.clone()` or assume `Clone` / `Copy` for candidate structs or enums unless the shown shared item already derives or implements it. If ownership duplication is necessary, move the value only when allowed, rebuild it field by field, or use an existing project deep-copy helper shown in context.
9. Return exactly one complete callable with balanced delimiters; do not wrap it in an enclosing `impl` or `mod` block.
10. Each shared_item_fix replaces exactly one listed item_id. If the listed item is an inherent `impl Type`, the replacement must be one complete `impl Type {{ ... }}` block. Do not put extra trait impls, structs, constants, imports, or unrelated items into that replacement; if a trait impl item_id is not listed, add narrow forwarding methods to the listed inherent impl and call those methods from the callable.
"""


def build_test_fix_prompt(
    *,
    project_name: str,
    test_output: str,
    test_stats: dict,
    test_context: list[dict],
    callable_context: list[dict],
    shared_item_context: list[dict] | None = None,
    insight_context: list[dict] | None = None,
) -> str:
    return f"""Fix this translated Rust project so its tests pass.

Project: {project_name}

Test stats:
```json
{json.dumps(test_stats, indent=2, ensure_ascii=False)}
```

Failing test output:
```
{test_output}
```

Relevant test source:
```json
{json.dumps(test_context, indent=2, ensure_ascii=False)}
```

Candidate Rust callables and original C context:
```json
{json.dumps(callable_context, indent=2, ensure_ascii=False)}
```

Candidate Rust shared items:
```json
{json.dumps(shared_item_context or [], indent=2, ensure_ascii=False)}
```

Translation insights:
{format_insight_context(insight_context or [])}

Return strict JSON only with this shape:
```json
{{
  "shared_item_fixes": [
    {{
      "item_id": "module::kind::name",
      "rust_code": "complete replacement Rust shared item"
    }}
  ],
  "fixes": [
    {{
      "callable_id": "module::kind::name",
      "rust_code": "complete Rust callable with the original public signature"
    }}
  ],
  "test_source_fixes": [
    {{
      "path": "src/bin/generated_test.rs",
      "find": "exact existing Rust test source text",
      "replace": "replacement Rust test source text"
    }}
  ]
}}
```

Rules:
1. Use only callable_id values listed in the candidate context.
2. Use only item_id values listed in Candidate Rust shared items.
3. Keep each public callable signature exactly unchanged.
4. Prefer callable-only fixes. Change shared items only when shared state, type layout, constants, statics, impl blocks, or trait delegation are the likely cause.
5. Do not add new standalone `pub` items. You may preserve or restore `pub` visibility on an existing struct field when the relevant tests, original interface, or candidate shared item directly observe that field.
6. Return the smallest set of callable and shared item fixes needed for the failing tests.
7. Treat existing public interface bodies in the candidate context as suspect too; they may be partial skeleton code, not trusted reference implementations.
8. Analyze the relevant test source in execution order before the failing assertion. If the first failure happens immediately after construction or initialization, compare constructor/reset sentinel state against the C binding before changing later methods.
9. When C uses `-1` as a sentinel and Rust represents it as `usize::MAX`, do not add to that sentinel and do not silently normalize it to `0`. Add an explicit sentinel branch that preserves the C state transition; for example, the first write after a reset should start at index `0`, set head to `0`, set tail to `length - 1`, and set data size to `length`.
10. Do not call `.clone()` or assume `Clone`/`Copy` for candidate structs or enums unless the shown shared item already implements it. If ownership duplication is necessary, move the value, rebuild it field by field, or include the minimal shared item fix that makes the trait implementation explicit.
11. If relevant tests contain hard-coded absolute paths from another machine or an unavailable fixture root, do not require that exact root to exist. Preserve the logical repo root or fixture-file behavior by using the current working directory or a project-local path, and use the original file basename when mapping file inputs.
12. Each shared_item_fix replaces exactly one listed item_id. A `const` item_id replacement must be one complete `const` item, a `static` item_id replacement must be one complete `static` item, and an `impl Type` item_id replacement must be one complete `impl Type {{ ... }}` block. Do not put extra helper functions, structs, statics, imports, trait impls, or unrelated items into a shared_item_fix; if no listed shared item can hold the helper, keep the logic inside the listed callable fixes.
13. Use test_source_fixes only for generated Rust test harness code listed in Relevant test source. The find text must be an exact snippet that appears once in that file. Do not edit production source through test_source_fixes.
14. Do not return `unimplemented!()`, `todo!()`, `panic!("not implemented")`, or placeholder bodies as a fix. A runtime placeholder panic is an implementation failure; replace it with real logic from the C context and tests.
15. If the failure is an equality diff isolated to one public field, make a narrow field/constant repair first. Do not rewrite an already passing parser, scanner, or formatter unless the failing output shows that control flow or parsing semantics are also wrong.
16. When constructing a struct from a `Vec`, `String`, or other moved owner and also storing derived metadata such as length/count/capacity, compute the metadata before moving the owner into the struct literal.
17. For helpers that take `&File` and read or seek the shared cursor, use a mutable binding such as `mut file: &File` or `let mut file = file;` and call `Read`/`Seek` on that binding. Do not pass `&mut file` when it creates `&mut &File` borrow errors, and do not clone/reopen the file just to satisfy mutability.
18. Avoid `Vec::resize(value)` when the element type is not `Clone`; use `resize_with`, `while vec.len() <= idx {{ vec.push(...) }}`, or direct `push` construction instead.
19. If a declaration lookup expects a name with a bracketed range or dimension suffix, keep that suffix in the stored public name while keeping the compact identifier code separate for assignments.
20. For probabilistic membership failures, distinguish generated upper-bound assertions from implementation underfill: a large negative observed-minus-expected rate should keep the one-sided test shape, but may still require repairing bit-count, hash-count, upgrade, or insertion logic.
21. If an equality diff still differs only in `capacity`, edit the live capacity initializer or return literal that produced the left-side value. Do not rewrite parser branches that already pass.
22. For line-oriented assignment/value-change records, trim whitespace around the value field and the compact identifier token before checking identifier length or looking up the declared record.
23. For table-backed probabilistic filters, clamp requested zero or tiny capacity to the source minimum before computing derived metadata; do not return an inert object with zero hash/bit metadata if later add/test operations are valid.
24. When a capacity equality diff shows concrete left and right values, treat the right-side value as the test oracle. Do not replace one wrong capacity with a guessed power-of-two, reserve size, or common library default.
25. In scanner/parser translations, whitespace in C-style format strings skips arbitrary whitespace; do not translate `\n`, `\t`, or spaces in those formats into a required literal byte unless the source matched a non-whitespace token.
26. For probabilistic filters that time out after many successful progress lines, preserve the source constructor scale: source minimum capacity, raw bit/hash computation, rounded table size, adjusted stored hash count, staged growth threshold, and clear-without-metadata-reset semantics.
27. If a parser fixture expects success with an empty collection or zero count, preserve source loop guards that can leave the output empty before a complete record is committed. Return the expected empty structure for that fixture instead of forcing a separator or missing-value error.
28. For declarations with public lookup names, read the visible name until the declaration terminator or record end. Preserve internal spaces and bracketed range/dimension suffixes in the public name, and keep compact assignment identifiers in a separate mapping.
29. For staged probabilistic filters, the constructor may compute metadata but should not allocate the final bit table or make `upgraded` true immediately unless the source does. First add should grow the staged bucket table, and only later growth should upgrade to bits at the source threshold.
"""


def build_retry_prompt(*, original_prompt: str, previous_output: str, issue: str) -> str:
    if issue == "incomplete_output":
        issue_text = "The previous response was truncated before the callable body finished."
    elif issue == "forbidden_ffi_declaration":
        issue_text = "The previous response declared a new extern/FFI binding inside the callable."
    else:
        issue_text = "The previous response could not be patched back into the target callable."

    tail = previous_output[-2000:] if len(previous_output) > 2000 else previous_output

    if issue == "incomplete_output":
        return f"""{original_prompt}

Previous response problem:
{issue_text}

Current callable tail:
```rust
{tail}
```

Continue the same callable exactly from where the previous response stopped.
Do not repeat the function signature or any already emitted prefix.
Do not restart the callable from the beginning.
Do not add explanation or code fences.
Return only the missing Rust code suffix.
"""

    return f"""{original_prompt}

Previous response problem:
{issue_text}

Previous response tail:
```rust
{tail}
```

Return the complete callable again from the beginning.
Keep the public signature exactly unchanged.
Do not declare new extern blocks or FFI bindings; use existing Rust APIs or inline local Rust logic.
Make sure the callable body is complete and all braces are balanced.
Return only Rust callable code.
"""
