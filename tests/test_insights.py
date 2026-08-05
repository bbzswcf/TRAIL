import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from c2rust_translator.adversarial_insights import InsightBank
from c2rust_translator.alignment_planner import CBinding
from c2rust_translator.insights import (
    DEFAULT_INSIGHTS,
    LLM_INTERACTION_SCORER_SYSTEM,
    LLMInsightInteractionScorer,
    StructuredInteractionScorer,
    TranslationInsight,
    empirical_weak_pair_penalty,
    format_insight_context,
    load_insight_library as _load_insight_library,
    score_insight_interaction,
    select_insights,
    select_insights_with_metadata,
)
from c2rust_translator.prompts import (
    SYSTEM_PROMPT,
    build_fix_prompt,
    build_retry_prompt,
    build_test_fix_prompt,
    build_translate_prompt,
    format_deps,
)


def load_insight_library(path):
    snapshot_path = Path(path)
    if str(path).startswith("snapshots/insights/") and not snapshot_path.exists():
        raise unittest.SkipTest(f"local Python insight snapshot not present: {path}")
    return _load_insight_library(path)


class InsightSelectionTest(unittest.TestCase):
    def test_final_json_insight_banks_load(self):
        expected_counts = {
            "snapshots/insights/gpt_final_insight_bank.json": 99,
            "snapshots/insights/deepseek_final_insight_bank.json": 49,
            "snapshots/insights/kimi_final_insight_bank.json": 228,
        }

        for path, expected_count in expected_counts.items():
            with self.subTest(path=path):
                bank = InsightBank.load(path)
                self.assertEqual(len(bank.records), expected_count)

    def test_load_insight_library_uses_snapshot_selector(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "variant.py"
            path.write_text(
                """
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class TranslationInsight:
    id: str
    title: str
    trigger: str
    goal: str
    constraint: str
    risk: str
    guidance: str
    tags: tuple[str, ...]
    keywords: tuple[str, ...]

    def to_prompt_dict(self):
        return {"id": self.id, "title": self.title, "tags": list(self.tags)}

DEFAULT_INSIGHTS = (
    TranslationInsight(
        id="variant-only",
        title="Variant Only",
        trigger="",
        goal="",
        constraint="",
        risk="",
        guidance="",
        tags=("variant",),
        keywords=("variant_marker",),
    ),
)

def select_insights(text, insights=DEFAULT_INSIGHTS, limit=5):
    return list(insights)[:limit]
""",
                encoding="utf-8",
            )

            library = load_insight_library(path)

        self.assertEqual([insight.id for insight in library.insights], ["variant-only"])
        self.assertEqual([insight.id for insight in library.select("anything", limit=1)], ["variant-only"])

    def test_format_insight_context_accepts_snapshot_insight_objects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "variant.py"
            path.write_text(
                """
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class TranslationInsight:
    id: str
    title: str
    trigger: str
    goal: str
    constraint: str
    risk: str
    guidance: str
    tags: tuple[str, ...]
    keywords: tuple[str, ...]

    def to_prompt_dict(self):
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

DEFAULT_INSIGHTS = (
    TranslationInsight(
        id="snapshot-only",
        title="Snapshot Only",
        trigger="snapshot trigger",
        goal="snapshot goal",
        constraint="snapshot constraint",
        risk="snapshot risk",
        guidance="snapshot guidance",
        tags=("snapshot",),
        keywords=("snapshot_marker",),
    ),
)

def select_insights(text, insights=DEFAULT_INSIGHTS, limit=5):
    return list(insights)[:limit]
""",
                encoding="utf-8",
            )

            library = load_insight_library(path)

        rendered = format_insight_context(library.select("snapshot_marker", limit=1))

        self.assertIn("snapshot-only", rendered)
        self.assertIn("snapshot guidance", rendered)

    def test_evoc2rust_prompt_insights_load_as_replacement_library(self):
        library = load_insight_library("snapshots/insights/evoc2rust_prompt_insights.py")

        insight_ids = [insight.id for insight in library.insights]

        self.assertEqual(
            insight_ids,
            [
                "evo-c-type-layout-semantics",
                "evo-c-operator-semantics",
                "evo-macro-function-disambiguation",
                "evo-cast-boundaries",
                "evo-c-truthiness-control-flow",
                "evo-struct-index-temporaries",
                "evo-initialization-before-use",
                "evo-macro-expansion-special-cases",
                "evo-compile-repair-locality",
            ],
        )

    def test_evoc2rust_prompt_insights_select_prompt_rules_for_c_translation_cues(self):
        library = load_insight_library("snapshots/insights/evoc2rust_prompt_insights.py")

        selected = library.select(
            """
            Source C has MY_MACRO(value), an uppercase K256[i] global array,
            sizeof(int), &state.field, p++, (uint32_t *)buffer, switch(kind),
            and a non-comparison condition `if (flags)`.
            """,
            limit=5,
        )
        selected_ids = {insight.id for insight in selected}

        self.assertIn("evo-macro-function-disambiguation", selected_ids)
        self.assertIn("evo-c-operator-semantics", selected_ids)
        self.assertIn("evo-cast-boundaries", selected_ids)
        self.assertIn("evo-c-truthiness-control-flow", selected_ids)

    def test_evoc2rust_prompt_insights_do_not_leak_baseline_runtime_helpers(self):
        library = load_insight_library("snapshots/insights/evoc2rust_prompt_insights.py")

        rendered = format_insight_context(library.insights)

        banned_runtime_terms = (
            "Ptr<",
            "VoidPtr",
            "Array<",
            "Global<",
            "FilePtr",
            "FuncPtr",
            "c_ref!",
            "c_sizeof!",
            "c_sizeofval!",
            "c_for!",
            "c_do!",
            "c_switch!",
            ".cast()",
            ".as_bool()",
            "suffix_plus_plus",
            "prefix_plus_plus",
        )
        for term in banned_runtime_terms:
            with self.subTest(term=term):
                self.assertNotIn(term, rendered)

    def test_generic_full_repro_focused_insights_do_not_leak_project_test_or_model_names(self):
        leak_terms = (
            "deepseek",
            "ali",
            "flash",
            "kimi",
            "gpt",
            "math-library-in-c",
            "matrix_multiplication",
            "simplexml",
            "mdb",
            "libbeaufort",
            "geofence",
            "ltre",
            "remimu",
            "simple_lang",
            "gorilla-paper-encode",
            "graph-recogniser",
            "linear-algebra-c",
            "bigint",
            "c-aces",
            "coroutine",
            "cset",
            "simple-sparsehash",
            "lambda-calculus-eval",
            "mvptree",
            "impcheck",
            "clhash",
            "bitset",
            "c-blind-rsa-signatures",
            "inversion_list",
            "merkle-tree-c",
            "castom_test_exp",
            "scastom_test_pow",
            "test_cset_iteration",
            "test_xml",
            "test_",
            "fasthamming",
            "fast_hamming",
            "simple_config",
            "simple-config",
            "blt",
            "libvcd",
            "ted",
            "morton",
            "fleur",
            "clhash",
            "clog",
            "vaultsync",
            "bostree",
            "blind-rsa",
            "dict_deserialize",
            "test5",
            "dict_str",
            "libwecan",
            "razz",
            "roaring-bitmap",
            "libutf",
            "utf8",
            "inversion_list",
            "inversion-list",
        )
        snapshot_paths = (
            Path("snapshots/insights/generic_default_full_repro.py"),
            Path("snapshots/insights/generic_single_full_repro.py"),
            Path("snapshots/insights/generic_multi_full_repro.py"),
            Path("snapshots/insights/generic_kimi_multi_full_repro.py"),
            Path("snapshots/insights/generic_kimi_multi_full_repro_v37.py"),
            Path("snapshots/insights/generic_kimi_multi_full_repro_v38.py"),
            Path("snapshots/insights/generic_kimi_multi_full_repro_v39.py"),
            Path("snapshots/insights/generic_kimi_multi_full_repro_v40.py"),
            Path("snapshots/insights/generic_kimi_multi_full_repro_v41.py"),
            Path("snapshots/insights/generic_kimi_multi_full_repro_v42.py"),
            Path("snapshots/insights/generic_kimi_multi_full_repro_v43.py"),
            Path("snapshots/insights/generic_kimi_multi_full_repro_v44.py"),
            Path("snapshots/insights/generic_kimi_multi_full_repro_v45.py"),
            Path("snapshots/insights/generic_kimi_multi_full_repro_v46.py"),
            Path("snapshots/insights/generic_kimi_multi_full_repro_v47.py"),
            Path("snapshots/insights/generic_kimi_multi_full_repro_v48.py"),
            Path("snapshots/insights/generic_kimi_multi_full_repro_v49.py"),
            Path("snapshots/insights/generic_kimi_multi_full_repro_v50.py"),
            Path("snapshots/insights/generic_kimi_multi_full_repro_v51.py"),
            Path("snapshots/insights/generic_kimi_multi_full_repro_v52.py"),
            Path("snapshots/insights/generic_kimi_multi_full_repro_v53.py"),
            Path("snapshots/insights/generic_kimi_multi_full_repro_v54.py"),
            Path("snapshots/insights/generic_kimi_multi_full_repro_v55.py"),
            Path("snapshots/insights/generic_kimi_multi_full_repro_v56.py"),
            Path("snapshots/insights/generic_kimi_multi_full_repro_v57.py"),
            Path("snapshots/insights/generic_kimi_multi_full_repro_v58.py"),
            Path("snapshots/insights/generic_kimi_multi_full_repro_v59.py"),
            Path("snapshots/insights/generic_kimi_multi_full_repro_v60.py"),
        )

        for path in snapshot_paths:
            library = load_insight_library(path)
            module = library.select.__globals__
            focused_insights = module.get("FOCUSED_REPRO_INSIGHTS", ())
            extra_insights = module.get("EXTRA_REPRO_INSIGHTS", ())
            for insight in (*focused_insights, *extra_insights):
                fields = insight.to_prompt_dict()
                fields["keywords"] = list(insight.keywords)
                rendered = repr(fields).lower()
                for term in leak_terms:
                    if term == "test_" or "-" in term or "_" in term:
                        leaked = term in rendered
                    else:
                        leaked = re.search(rf"\b{re.escape(term)}\b", rendered) is not None
                    self.assertFalse(leaked, f"{term!r} leaked in {path}:{insight.id}")

    def test_generic_full_repro_selectors_use_pattern_context(self):
        default_library = load_insight_library("snapshots/insights/generic_default_full_repro.py")
        single_library = load_insight_library("snapshots/insights/generic_single_full_repro.py")
        multi_library = load_insight_library("snapshots/insights/generic_multi_full_repro.py")
        kimi_multi_library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro.py")

        default_ids = [insight.id for insight in default_library.select("powf exp approximate equality", limit=2)]
        tag_parser_ids = [
            insight.id
            for insight in default_library.select(
                "parser_get_next_token state_translate BEGIN_OPEN_TAG BEGIN_CLOSE_TAG END_TAG tag_stack",
                limit=2,
            )
        ]
        single_ids = [insight.id for insight in single_library.select("parse_regex parse_term unimplemented", limit=1)]
        multi_ids = [insight.id for insight in multi_library.select("product >> 128 u128 carryless high limb", limit=2)]
        wide_shift_ids = [
            insight.id
            for insight in multi_library.select(
                "this arithmetic operation will overflow: attempt to shift right by 128",
                limit=2,
            )
        ]
        wide_xor_ids = [
            insight.id
            for insight in multi_library.select(
                "assertion failed: xor difference changed after same bit flip; wrapping_mul hash mix",
                limit=2,
            )
        ]
        wide_wrapper_ids = [
            insight.id
            for insight in multi_library.select(
                "hasher method uses stored random_data key and should call top-level hash bytes",
                limit=2,
            )
        ]
        wide_bounds_ids = [
            insight.id
            for insight in multi_library.select(
                "index out of bounds: the len is 133 but the index is 133 when reading expanded key",
                limit=2,
            )
        ]
        comparator_ids = [
            insight.id
            for insight in multi_library.select(
                "intersect requires a compare function default bytes comparator contains failed transmute_copy larger than Src",
                limit=2,
            )
        ]
        comparator_panic_ids = [
            insight.id
            for insight in multi_library.select(
                "test_default_bytes_comparator panicked because add requires a compare function while custom_comparator passes",
                limit=2,
            )
        ]
        comparator_guidance = next(
            insight.guidance
            for insight in multi_library.insights
            if insight.id == "collection-comparator-byte-equality"
        )
        parser_entrypoint_ids = [
            insight.id
            for insight in multi_library.select(
                "public parser entrypoint builds ParseContext then calls parse_regex parse_term parse_factor unimplemented",
                limit=3,
            )
        ]
        automaton_remap_ids = [
            insight.id
            for insight in multi_library.select(
                "minimize the DFA by merging indistinguishable states with old_to_new and new_states representative mapping",
                limit=3,
            )
        ]
        bucket_copy_ids = [
            insight.id
            for insight in multi_library.select(
                "error cannot move out of index of Vec bucket because elem has generic type T and does not implement the Copy trait",
                limit=3,
            )
        ]
        shared_mut_accessor_ids = [
            insight.id
            for insight in multi_library.select(
                "pub fn values_ref(&self) -> &mut Vec<Value> error invalid_reference_casting casting &T to &mut T is undefined behavior",
                limit=3,
            )
        ]
        bucket_sentinel_ids = [
            insight.id
            for insight in multi_library.select(
                "Tests timeout after 120 seconds in add; buckets use linear probe, pi == -1 empty slot, but new buckets are zeroed",
                limit=3,
            )
        ]
        proof_subset_ids = [
            insight.id
            for insight in multi_library.select(
                "proof verify uses proof.indices tree node positions, proof.lemmas sibling nodes, leaves_count and leaf offset",
                limit=3,
            )
        ]
        range_feistel_ids = [
            insight.id
            for insight in multi_library.select(
                "format preserving Feistel range permutation with sqrt halves, Speck round keys, encrypt decrypt overflow at left * base",
                limit=3,
            )
        ]
        range_speck_ids = [
            insight.id
            for insight in multi_library.select(
                "new(range: u64, rounds: u64, seed: u64) uses speck_expand then encrypt decrypt Feistel halves with ceil sqrt",
                limit=5,
            )
        ]
        gap_buffer_ids = [
            insight.id
            for insight in multi_library.select(
                "GapBuffer has gap_len gap_loc str_len move_gap split char_at and panics with range start index out of range on empty string",
                limit=3,
            )
        ]
        tree_roundtrip_ids = [
            insight.id
            for insight in multi_library.select(
                "write tree to file then read tree from file with HEADER_SIZE, node type markers, active flag, bytelength, write_node and read_node",
                limit=3,
            )
        ]
        date_sentinel_ids = [
            insight.id
            for insight in multi_library.select(
                "date parser tm_mon tm_mday tm_year tm_usec mktime tm_to_time_t MONTH_DAYS index out of bounds usize::MAX for missing date time-only input",
                limit=3,
            )
        ]
        alpha_convert_ids = [
            insight.id
            for insight in multi_library.select(
                "alpha_convert lambda fresh variable name should not use _fresh; needs monotonic counter numeric suffix",
                limit=3,
            )
        ]
        definition_mutation_ids = [
            insight.id
            for insight in multi_library.select(
                "expand_definitions sees AstNodeType::DEFINITION and term expanded but immutable reference does not mutate the observed node in place",
                limit=3,
            )
        ]
        config_parser_ids = [
            insight.id
            for insight in multi_library.select(
                "cfg_parse consumes consume_colon and cfg_fprint prints key: value; key too long and rgba alpha 0.5 should round to 128",
                limit=3,
            )
        ]
        open_address_payload_ids = [
            insight.id
            for insight in multi_library.select(
                "open address hash_table uses buckets with payload void* last_found compute_hash wrapping_mul and deleted tombstone slots",
                limit=3,
            )
        ]
        contact_force_ids = [
            insight.id
            for insight in multi_library.select(
                "contact loop uses p1_idx p2_idx particles_size normal_forces tangent_forces resultant_forces and must not update only one direction",
                limit=3,
            )
        ]
        kimi_config_ids = [
            insight.id
            for insight in kimi_multi_library.select(
                "config parser consume colon key: then reports ':' expected but expected missing value after delimiter",
                limit=3,
            )
        ]
        kimi_byte_view_ids = [
            insight.id
            for insight in kimi_multi_library.select(
                "generated test cannot find module bytemuck bytes_of undeclared crate; use to_ne_bytes key bytes",
                limit=3,
            )
        ]
        kimi_fixture_ids = [
            insight.id
            for insight in kimi_multi_library.select(
                "read back roundtrip fixture fails with No such file before reopen; writer should flush relative path",
                limit=3,
            )
        ]
        kimi_range_ids = [
            insight.id
            for insight in kimi_multi_library.select(
                "intersection setup unwraps value out of range for member outside universe support in derived set",
                limit=3,
            )
        ]
        kimi_interleave_ids = [
            insight.id
            for insight in kimi_multi_library.select(
                "z-order interleave spread bits result is transposed between even bits and odd bits coordinate lanes",
                limit=3,
            )
        ]
        kimi_mask_interleave_ids = [
            insight.id
            for insight in kimi_multi_library.select(
                "function takes hi and lo lanes; xu = lo, yu = hi; masks 0x5555555555555555 0x3333333333333333 0x0000FFFF0000FFFF 0x00FF00FF00FF00FF with C shifts 020 and 010",
                limit=3,
            )
        ]
        kimi_decode_interleave_ids = [
            insight.id
            for insight in kimi_multi_library.select(
                "generated harness uses a single-lane deinterleave helper result for both original coordinate lanes; decode both lanes from unshifted and one-bit-shifted z-order code",
                limit=3,
            )
        ]
        kimi_bitmap_ids = [
            insight.id
            for insight in kimi_multi_library.select(
                "bitmap fill updates cardinality but contains returns false for bit-vector array container membership",
                limit=3,
            )
        ]
        kimi_string_ids = [
            insight.id
            for insight in kimi_multi_library.select(
                "substring byte offset byte length UTF-16 UTF-8 conversion uses char index instead of bytes",
                limit=3,
            )
        ]
        kimi_utf_ids = [
            insight.id
            for insight in kimi_multi_library.select(
                "UTF-8 validation panics at char boundary and mishandles surrogate overlong lossy invalid sequence",
                limit=3,
            )
        ]
        kimi_tree_ids = [
            insight.id
            for insight in kimi_multi_library.select(
                "tree remove delete leaves manual node count different from order statistics left_child_count right_child_count rank select",
                limit=3,
            )
        ]
        kimi_crypto_ids = [
            insight.id
            for insight in kimi_multi_library.select(
                "crypto rsa keypair blind sign verify returns -1 failure stub despite openssl padding backend",
                limit=3,
            )
        ]
        kimi_string_key_ids = [
            insight.id
            for insight in kimi_multi_library.select(
                "deserialize snapshot header rejects string key with key type conflict because pointer sized serialized slot key size differs",
                limit=3,
            )
        ]
        kimi_motorola_ids = [
            insight.id
            for insight in kimi_multi_library.select(
                "CAN signal Motorola frame_to_local startbit decode float negative value needs reverse copy little-endian bytes",
                limit=3,
            )
        ]
        kimi_circular_ids = [
            insight.id
            for insight in kimi_multi_library.select(
                "linked list circular collection iterate next is none before it wraps back to head after append",
                limit=3,
            )
        ]
        kimi_open_address_ids = [
            insight.id
            for insight in kimi_multi_library.select(
                "open address robin hood buckets pi bucket_size remove contains set algebra default byte comparator",
                limit=3,
            )
        ]
        kimi_sorted_ids = [
            insight.id
            for insight in kimi_multi_library.select(
                "sorted order lexicographic forall first next prev allprefixed ceil floor vacant entry traversal",
                limit=3,
            )
        ]
        kimi_cursor_ids = [
            insight.id
            for insight in kimi_multi_library.select(
                "timestamp value-change assignment line shared cursor buffered reader try_clone read ahead loses signal id payload bits",
                limit=3,
            )
        ]
        kimi_decl_ids = [
            insight.id
            for insight in kimi_multi_library.select(
                "$var declaration signal name bracket suffix $end parts[3] split whitespace after id token",
                limit=3,
            )
        ]
        kimi_gap_ids = [
            insight.id
            for insight in kimi_multi_library.select(
                "gap buffer gap_len gap_loc str_len split backspace get_string exposes stale tail instead of visible text",
                limit=3,
            )
        ]

        self.assertIn("stdlib-float-wrapper-precision", default_ids)
        self.assertEqual("token-state-parser-tag-events", tag_parser_ids[0])
        self.assertEqual(["recursive-descent-entrypoint-complete"], single_ids)
        self.assertIn("wide-carryless-product-limbs", multi_ids)
        self.assertIn("wide-carryless-product-limbs", wide_shift_ids)
        self.assertIn("wide-carryless-product-limbs", wide_xor_ids)
        self.assertIn("wide-carryless-product-limbs", wide_wrapper_ids)
        self.assertIn("wide-carryless-product-limbs", wide_bounds_ids)
        self.assertIn("collection-comparator-byte-equality", comparator_ids)
        self.assertEqual("collection-comparator-byte-equality", comparator_panic_ids[0])
        self.assertIn("preserve every existing public method", comparator_guidance)
        self.assertIn("parser-entrypoint-local-helper-closure", parser_entrypoint_ids)
        self.assertIn("recursive-descent-entrypoint-complete", parser_entrypoint_ids)
        self.assertIn("automaton-minimize-complete-remap", automaton_remap_ids)
        self.assertIn("bucket-value-copy-without-clone", bucket_copy_ids)
        self.assertIn("shared-self-mut-field-accessor", shared_mut_accessor_ids)
        self.assertIn("open-address-bucket-sentinel-state", bucket_sentinel_ids)
        self.assertIn("proof-indexed-leaf-subset-rebuild", proof_subset_ids)
        self.assertIn("range-feistel-modular-halves", range_feistel_ids)
        self.assertIn("range-feistel-modular-halves", range_speck_ids)
        self.assertNotIn("unsigned-prng-wrapping-key-expansion", range_speck_ids)
        self.assertNotIn("bit-exact-encoding", range_speck_ids)
        self.assertIn("gap-buffer-allocated-length-invariant", gap_buffer_ids)
        self.assertIn("serialized-tree-roundtrip-layout", tree_roundtrip_ids)
        self.assertIn("date-field-sentinel-normalization", date_sentinel_ids)
        self.assertIn("alpha-conversion-monotonic-suffix", alpha_convert_ids)
        self.assertIn("definition-expansion-in-place", definition_mutation_ids)
        self.assertIn("config-parser-colon-format", config_parser_ids)
        self.assertIn("open-address-hash-payload-wrapping", open_address_payload_ids)
        self.assertEqual("bidirectional-contact-force-pair-update", contact_force_ids[0])
        self.assertIn("pairwise-contact-output-slots", contact_force_ids)
        self.assertEqual("config-missing-value-after-delimiter", kimi_config_ids[0])
        self.assertEqual("generated-byte-view-without-extra-crate", kimi_byte_view_ids[0])
        self.assertEqual("fixture-roundtrip-before-read", kimi_fixture_ids[0])
        self.assertEqual("range-intersection-out-of-universe-inputs", kimi_range_ids[0])
        self.assertEqual("bit-interleave-coordinate-order", kimi_interleave_ids[0])
        self.assertEqual("bit-interleave-coordinate-order", kimi_mask_interleave_ids[0])
        self.assertEqual("bit-interleave-coordinate-order", kimi_decode_interleave_ids[0])
        self.assertEqual("bitmap-fill-membership-before-counts", kimi_bitmap_ids[0])
        self.assertEqual("byte-indexed-string-slices", kimi_string_ids[0])
        self.assertEqual("utf-8-validation-boundaries", kimi_utf_ids[0])
        self.assertEqual("order-stat-tree-remove-counts", kimi_tree_ids[0])
        self.assertEqual("crypto-wrapper-real-backend", kimi_crypto_ids[0])
        self.assertEqual("serialized-string-key-slot-layout", kimi_string_key_ids[0])
        self.assertEqual("motorola-signal-local-byte-copy", kimi_motorola_ids[0])
        self.assertEqual("circular-linked-collection-invariant", kimi_circular_ids[0])
        self.assertEqual("open-address-remove-and-equality", kimi_open_address_ids[0])
        self.assertEqual("sorted-key-traversal-store", kimi_sorted_ids[0])
        self.assertEqual("shared-cursor-single-line-assignment", kimi_cursor_ids[0])
        self.assertEqual("multi-token-declaration-name", kimi_decl_ids[0])
        self.assertEqual("gap-buffer-split-live-tail", kimi_gap_ids[0])

    def test_kimi_v19_generic_selector_prioritizes_latest_gate_patterns(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v19.py")

        alpha_ids = [
            insight.id
            for insight in library.select(
                "rgba alpha parser stores byte; input rgba(255, 255, 255, 0.5) left a: 127 right a: 128 from 255.0 scaling",
                limit=3,
            )
        ]
        staged_ids = [
            insight.id
            for insight in library.select(
                "bad probability hits=1001 probability membership upgraded bits buckets grow constructor",
                limit=3,
            )
        ]

        self.assertEqual("normalized-alpha-rounds-to-byte", alpha_ids[0])
        self.assertIn("staged-filter-constructor-leaves-bit-table-absent", staged_ids)

    def test_kimi_v20_generic_selector_prioritizes_blank_eof_delimiter_pattern(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v20.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "parser after key skips whitespace then end-of-input; left ':' expected right missing value delimiter colon blank cursor",
                limit=3,
            )
        ]

        self.assertEqual("delimiter-eof-after-blank-is-missing-value", selected_ids[0])

    def test_kimi_v21_generic_selector_prioritizes_trailing_decimal_float(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v21.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "parser success case is_ok fails for numeric value 1. after scanner rewinds decimal point instead of float fraction",
                limit=3,
            )
        ]

        self.assertEqual("trailing-decimal-point-is-float", selected_ids[0])

    def test_kimi_v22_generic_selector_prioritizes_value_blank_eof(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v22.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "value parser consumes whitespace until eof; left invalid value right missing value blank cursor",
                limit=3,
            )
        ]

        self.assertEqual("value-eof-after-blank-is-missing-value", selected_ids[0])

    def test_kimi_v23_generic_selector_prioritizes_value_blank_eof_without_v22_chain(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v23.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "value parser consumes whitespace until eof; left invalid value right missing value blank cursor",
                limit=3,
            )
        ]

        self.assertEqual("value-eof-after-blank-is-missing-value", selected_ids[0])

    def test_kimi_v24_generic_selector_prioritizes_local_byte_blank_delimiter_eof(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v24.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "local byte parser after key skips whitespace then eof; left ':' expected right missing value delimiter cursor",
                limit=3,
            )
        ]

        self.assertEqual("local-byte-delimiter-blank-eof-is-missing-value", selected_ids[0])

    def test_kimi_v25_generic_selector_prioritizes_local_helper_blank_delimiter_eof(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v25.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "local helper parser after key calls skip whitespace then eof; left ':' expected right missing value delimiter cursor",
                limit=3,
            )
        ]

        self.assertEqual("local-helper-delimiter-blank-eof-is-missing-value", selected_ids[0])

    def test_kimi_v26_generic_selector_prioritizes_unclosed_quote_error(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v26.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "quoted string parser reaches eof; left invalid literal right closing quote expected",
                limit=3,
            )
        ]

        self.assertEqual("quoted-literal-eof-reports-closing-delimiter", selected_ids[0])

    def test_kimi_v27_generic_selector_prioritizes_quoted_literal_length_cap(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v27.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "quoted string literal exceeds maximum value length; cfg.is_err assertion failed value too long",
                limit=3,
            )
        ]

        self.assertEqual("quoted-literal-length-cap-before-return", selected_ids[0])

    def test_kimi_v28_generic_selector_prioritizes_value_suffix_error(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v28.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "value token has suffix byte; left ':' expected right unexpected character suffix",
                limit=3,
            )
        ]

        self.assertEqual("value-token-suffix-is-unexpected-character", selected_ids[0])

    def test_kimi_v29_generic_selector_prioritizes_unknown_alpha_literal_error(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v29.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "alphabetic value token after keyword checks left invalid value right invalid literal",
                limit=3,
            )
        ]

        self.assertEqual("unknown-alpha-value-token-is-invalid-literal", selected_ids[0])

    def test_kimi_v30_generic_selector_prioritizes_empty_component_eof(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v30.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "component list reaches eof with empty component; left invalid value right comma expected",
                limit=3,
            )
        ]

        self.assertEqual("empty-component-eof-keeps-delimiter-error", selected_ids[0])

    def test_kimi_v31_generic_selector_prioritizes_function_like_open_delimiter(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v31.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "function-like value keyword without opening delimiter left invalid literal right '(' expected",
                limit=3,
            )
        ]

        self.assertEqual("function-like-value-keyword-needs-opening-delimiter", selected_ids[0])

    def test_kimi_v32_generic_selector_prioritizes_component_range(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v32.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "component byte cast lost range check; expected integers in range and alpha normalized value",
                limit=3,
            )
        ]

        self.assertEqual("component-list-range-before-byte-cast", selected_ids[0])

    def test_kimi_v33_generic_selector_prioritizes_numeric_too_large(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v33.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "numeric integer overflow and float precision left invalid value right number too large",
                limit=3,
            )
        ]

        self.assertEqual("numeric-token-overflow-is-number-too-large", selected_ids[0])

    def test_kimi_v34_generic_selector_prioritizes_sign_only_component(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v34.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "component numeric token is sign only; left invalid value right number expected",
                limit=3,
            )
        ]

        self.assertEqual("component-sign-only-is-number-expected", selected_ids[0])

    def test_kimi_v35_generic_selector_prioritizes_local_char_blank_delimiter_eof(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v35.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "local char parser after key skips whitespace then eof; left ':' expected right missing value delimiter cursor",
                limit=3,
            )
        ]

        self.assertEqual("local-char-delimiter-blank-eof-is-missing-value", selected_ids[0])

    def test_kimi_v36_generic_selector_prioritizes_local_char_blank_value_terminator(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v36.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "local char parser after delimiter skips whitespace then newline eof; left invalid value right missing value cursor",
                limit=3,
            )
        ]

        self.assertEqual("local-char-value-terminator-after-blank-is-missing-value", selected_ids[0])

    def test_kimi_v37_generic_selector_prioritizes_non_leaky_full_gap_patterns(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v37.py")

        noncopy_ids = [
            insight.id
            for insight in library.select(
                "constructor builds fixed-size array of structs but E0277 says element type does not implement Copy; use array from_fn",
                limit=4,
            )
        ]
        matrix_ids = [
            insight.id
            for insight in library.select(
                "generated matrix multiply test calls method 1 with rows columns mismatch left -1 right 0; should fill result dimensions",
                limit=4,
            )
        ]
        filter_ids = [
            insight.id
            for insight in library.select(
                "probabilistic filter constructor tiny capacity add then membership test false; grow power of two table threshold",
                limit=4,
            )
        ]
        parser_ids = [
            insight.id
            for insight in library.select(
                "recursive descent parser parse entrypoint lookahead consume share token cursor; parse loop reaches eof",
                limit=4,
            )
        ]
        coroutine_ids = [
            insight.id
            for insight in library.select(
                "cooperative coroutine resume yield unimplemented; avoid ucontext makecontext swapcontext and model single-shot state machine",
                limit=4,
            )
        ]

        self.assertEqual("noncopy-array-from-fn-constructor", noncopy_ids[0])
        self.assertEqual("matrix-generated-dimension-contract", matrix_ids[0])
        self.assertEqual("probabilistic-filter-power2-capacity", filter_ids[0])
        self.assertEqual("recursive-parser-shared-cursor-entrypoint", parser_ids[0])
        self.assertEqual("stackless-routine-single-shot-state", coroutine_ids[0])

    def test_kimi_v38_generic_selector_prioritizes_load_factor_hash_resize(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v38.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "hash table insert says table fillness is above the load factor and needs to be resized; "
                "resize is not implemented and reinsert must preserve existing key data entries without adding fallback fields",
                limit=4,
            )
        ]

        self.assertEqual("load-factor-hash-table-resize-reinsert", selected_ids[0])

    def test_kimi_v39_generic_selector_prioritizes_parser_cursor_without_undeclared_globals(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v39.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "recursive parser repair compile failed: cannot find value POS in this scope and cannot find value TOKENS; "
                "parse entrypoint should keep token cursor state in local closures instead of undeclared globals",
                limit=4,
            )
        ]

        self.assertEqual("parser-cursor-without-undeclared-globals", selected_ids[0])

    def test_kimi_v40_generic_selector_prioritizes_hash_table_helper_placeholders(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v40.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "hash table insert path panicked at private helper get_first_entry with not implemented; "
                "resize/refill/recreate and bucket entry helper placeholders remain in the table impl",
                limit=4,
            )
        ]

        self.assertEqual("hash-table-helper-placeholders-in-impl", selected_ids[0])

    def test_kimi_v41_generic_selector_filters_broad_test_repair_noise(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v41.py")

        contact_ids = [
            insight.id
            for insight in library.select(
                "test repair after cargo test panic in contact loop: p1_idx p2_idx particles_size "
                "normal_forces tangent_forces resultant_forces; generated test has &mut particle.clone() "
                "but no File private field, no openssl rsa, no callback user_data, no output capacity struct",
                limit=5,
            )
        ]
        matrix_ids = [
            insight.id
            for insight in library.select(
                "test repair after cargo test panic in matrix multiply dispatcher: method 1 rows columns "
                "dimension mismatch left -1 right 0; generated harness expects return 0 and result dimensions",
                limit=5,
            )
        ]
        coroutine_ids = [
            insight.id
            for insight in library.select(
                "test repair after cargo test panic in cooperative routine resume yield running status "
                "ready suspend dead; ucontext makecontext swapcontext trampoline remains unimplemented",
                limit=5,
            )
        ]
        parser_ids = [
            insight.id
            for insight in library.select(
                "test repair compile failed in recursive parser: cannot find value POS in this scope, "
                "cannot find value TOKENS; parse entrypoint lookahead consume should keep local cursor closures",
                limit=5,
            )
        ]

        self.assertEqual("bidirectional-contact-force-pair-update", contact_ids[0])
        self.assertNotIn("resource-wrapper-constructor-safety", contact_ids)
        self.assertNotIn("config-parser-colon-format", contact_ids)
        self.assertEqual("matrix-generated-dimension-contract", matrix_ids[0])
        self.assertNotIn("crypto-wrapper-real-backend", matrix_ids)
        self.assertEqual("stackless-routine-single-shot-state", coroutine_ids[0])
        self.assertNotIn("output-struct-capacity-fields", coroutine_ids)
        self.assertEqual("parser-cursor-without-undeclared-globals", parser_ids[0])
        self.assertNotIn("callback-user-data-state", parser_ids)

    def test_kimi_v42_generic_selector_prioritizes_pending_refill_entry(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v42.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "hash table cycle has been detected and refill/recreate reinserts old entries; "
                "assertion failed cur_size should match after recreate left 3 right 4 because the "
                "displaced pending key value is outside the bucket arrays during rebuild",
                limit=5,
            )
        ]

        self.assertEqual("hash-refill-preserves-displaced-pending-entry", selected_ids[0])

    def test_kimi_v43_generic_selector_prioritizes_borrowed_hash_probe(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v43.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "hash lookup query(&self, key: &str) failed with borrowed data escapes outside of method; "
                "the helper requires a static key, while the open addressing probe also mixes u32 and usize indexes",
                limit=5,
            )
        ]

        self.assertEqual("hash-borrowed-key-local-byte-probe", selected_ids[0])

    def test_kimi_v43_generic_selector_prioritizes_parse_entrypoint_without_new_globals(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v43.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "parser panic at consume unimplemented; repair cannot find PARSER_TOKENS PARSER_POS PARSER_LEN "
                "and local thread storage inside one function is not visible to consume or lookahead; parse entrypoint "
                "should be self-contained with a local cursor and recursive descent helpers",
                limit=5,
            )
        ]

        self.assertEqual("parser-entrypoint-local-recursive-descent", selected_ids[0])

    def test_kimi_v44_generic_selector_prioritizes_private_hash_helper_without_ref_cast(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v44.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "hash table repair still has resize refill private helper placeholders, then compile failed with "
                "invalid_reference_casting: casting &T to &mut T in get_first_entry(&self, key: &str) -> &mut entry; "
                "replace the private helper shape or use indexes instead of unsafe reference casts",
                limit=5,
            )
        ]

        self.assertEqual("hash-private-mut-helper-without-ref-cast", selected_ids[0])

    def test_kimi_v44_generic_selector_prioritizes_borrowed_token_expression_loop(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v44.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "parser entrypoint local cursor repair prints Expected ';' at end of statement after input with "
                "5 + 3 - 2 because expression parsing consumed only the primary; compile repair reports E0507 "
                "cannot move out of op_token.token_type behind a shared reference and suggests clone",
                limit=5,
            )
        ]

        self.assertEqual("parser-borrowed-token-expression-operator-clone", selected_ids[0])

    def test_kimi_v45_generic_selector_prioritizes_zero_offset_option_storage(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v45.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "command line option parser default callback writes parsed int float bool values through "
                "a data pointer plus offsetof; first field offset is 0, but assertion left 0 right 42 "
                "because offset zero was treated as no destination and skipped the store",
                limit=5,
            )
        ]

        self.assertEqual("option-offset-zero-is-valid-field", selected_ids[0])

    def test_kimi_v46_generic_selector_prioritizes_pivoted_determinant_product(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v46.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "determinant of a 4 by 4 matrix rounds to the wrong value after pivoted LU decomposition; "
                "row swaps update swap_count, but the result recursively computes determinant of constructed "
                "L and U matrices instead of returning signed product of the eliminated diagonal pivots",
                limit=5,
            )
        ]

        self.assertEqual("determinant-pivot-elimination-diagonal-product", selected_ids[0])

    def test_kimi_v47_generic_selector_prioritizes_option_offset_from_source_metadata(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v47.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "option parser source metadata shows struct field offset usize matching original offsetof, "
                "callback receives value plus data pointer and option, type flags include int long float double bool, "
                "default storage should write through data.add(offset) for every valid destination field",
                limit=5,
            )
        ]

        self.assertEqual("option-offset-zero-is-valid-field", selected_ids[0])

    def test_kimi_v48_generic_selector_prioritizes_empty_wrapper_layout(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v48.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "helper contract buffer_read over Candidate Rust shared item current_rust pub struct Buffer {} "
                "while C file context uses buffer->data buffer->len buffer->rindex; compile repair reports "
                "no field data len rindex on type Buffer after inventing C storage fields",
                limit=5,
            )
        ]

        self.assertEqual("skeletal-rust-wrapper-layout-is-authoritative", selected_ids[0])

    def test_kimi_v48_generic_selector_prioritizes_empty_wrapper_from_source_only_context(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v48.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "contract buffer_read over a buffer helper has no translated dependencies; "
                "the bound C code reads buffer->data[buffer->rindex], checks buffer->rindex >= buffer->len, "
                "and increments buffer->rindex before returning the byte",
                limit=5,
            )
        ]

        self.assertEqual("skeletal-rust-wrapper-layout-is-authoritative", selected_ids[0])

    def test_kimi_v48_generic_selector_prioritizes_empty_wrapper_constructor_context(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v48.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "constructor contract buffer_create has no translated dependencies; "
                "the bound C code allocates a buffer then assigns buf->data, buf->len, and buf->msize, "
                "but the Rust wrapper may be skeletal and should not invent C storage fields",
                limit=5,
            )
        ]

        self.assertEqual("skeletal-rust-wrapper-layout-is-authoritative", selected_ids[0])

    def test_kimi_v48_generic_selector_prioritizes_empty_wrapper_binding_context(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v48.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "mapped_constructor buffer_create helpers/buffer.i struct buffer *buffer_create() { "
                "struct buffer *buf = calloc(sizeof(struct buffer), 1); "
                "buf->data = calloc(2000, 1); buf->len = 0; buf->msize = 2000; return buf; }",
                limit=5,
            )
        ]

        self.assertEqual("skeletal-rust-wrapper-layout-is-authoritative", selected_ids[0])

    def test_kimi_v48_generic_selector_prioritizes_empty_wrapper_helper_binding_context(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v48.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "mapped_function buffer_extend helpers/buffer.i void buffer_extend(struct buffer *buffer, int size) { "
                "buffer->msize += size; buffer->data = realloc(buffer->data, buffer->msize); }",
                limit=5,
            )
        ]

        self.assertEqual("skeletal-rust-wrapper-layout-is-authoritative", selected_ids[0])

    def test_kimi_v48_generic_selector_prioritizes_empty_wrapper_pointer_binding_context(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v48.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "mapped_function buffer_ptr helpers/buffer.i void *buffer_ptr(struct buffer *buffer) { "
                "return buffer->data; }",
                limit=5,
            )
        ]

        self.assertEqual("skeletal-rust-wrapper-layout-is-authoritative", selected_ids[0])

    def test_kimi_v48_generic_selector_prioritizes_empty_wrapper_free_binding_context(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v48.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "mapped_function buffer_free helpers/buffer.i void buffer_free(struct buffer *buffer) { "
                "free(buffer->data); free(buffer); }",
                limit=5,
            )
        ]

        self.assertEqual("skeletal-rust-wrapper-layout-is-authoritative", selected_ids[0])

    def test_kimi_v48_generic_selector_prioritizes_empty_wrapper_write_binding_context(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v48.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "mapped_function buffer_write helpers/buffer.i void buffer_write(struct buffer *buffer, char c) { "
                "buffer_need(buffer, 1); buffer->data[buffer->len] = c; buffer->len++; }",
                limit=5,
            )
        ]

        self.assertEqual("skeletal-rust-wrapper-layout-is-authoritative", selected_ids[0])

    def test_kimi_v48_generic_selector_prioritizes_empty_vector_constructor_binding_context(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v48.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "mapped_constructor vector_create helpers/vector.i struct vector *vector_create(size_t esize) { "
                "struct vector *vec = calloc(sizeof(struct vector), 1); vec->esize = esize; "
                "vec->data = calloc(esize, VECTOR_ELEMENT_INCREMENT); vec->saves = vector_create(sizeof(struct vector)); }",
                limit=5,
            )
        ]

        self.assertEqual("skeletal-rust-wrapper-layout-is-authoritative", selected_ids[0])

    def test_kimi_v48_generic_selector_prioritizes_empty_vector_helper_binding_context(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v48.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "mapped_function vector_peek helpers/vector.i void *vector_peek(struct vector *vector) { "
                "if (vector->pindex >= vector->count) return 0; return vector->data + vector->pindex * vector->esize; }",
                limit=5,
            )
        ]

        self.assertEqual("skeletal-rust-wrapper-layout-is-authoritative", selected_ids[0])

    def test_kimi_v48_generic_selector_prioritizes_file_compiler_entrypoint(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v48.py")

        selected_ids = [
            insight.id
            for insight in library.select(
                "public compile_file entrypoint panicked at not implemented but tests assert the returned status is "
                "COMPILER_FILE_COMPILED_OK; C flow creates a compile_process, creates a lex_process, runs lex, "
                "passes tokens back to the compiler, runs parse, and returns COMPILER_FAILED_WITH_ERRORS on failures",
                limit=5,
            )
        ]

        self.assertEqual("compiler-file-entrypoint-compose-visible-stages", selected_ids[0])

    def test_kimi_v49_generic_selector_prioritizes_empty_vector_accessor(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v49.py")

        selected = library.select(
            "Callable category mapped_function; Bound C functions helpers/vector.i::vector_at "
            "void *vector_at(struct vector *vector, int index) { "
            "return vector->data + (index * vector->esize); } "
            "Candidate Rust shared items current_rust pub struct Vector {}",
            limit=5,
        )
        selected_ids = [insight.id for insight in selected]
        rendered = format_insight_context(selected)

        self.assertEqual("skeletal-rust-wrapper-layout-is-authoritative", selected_ids[0])
        self.assertIn("constructors should return `Wrapper {}`", rendered)
        self.assertIn("accessors should return `None`", rendered)

    def test_kimi_v49_generic_selector_prioritizes_private_field_construction(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v49.py")

        selected = library.select(
            "compile failed outside defining module with error[E0451]: fields file and path of "
            "struct ClonableFile are private while building Some(ClonableFile { file, path }); "
            "Candidate Rust shared items show pub struct ClonableFile { file: File, path: PathBuf } "
            "and impl ClonableFile { pub fn new(path: impl Into<PathBuf>) -> io::Result<Self> }",
            limit=5,
        )
        selected_ids = [insight.id for insight in selected]
        rendered = format_insight_context(selected)

        self.assertEqual("private-field-construction-uses-visible-api", selected_ids[0])
        self.assertIn("Do not use a struct literal for a type whose fields are private", rendered)

    def test_kimi_v50_generic_selector_prioritizes_entrypoint_dependency_shapes(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v50.py")

        selected = library.select(
            "public compile_file entrypoint composes compile_process_create and lex_process_create, "
            "but compile repair reports E0308 mismatched types: expected CompileProcess found Option<_> "
            "from `let Some(mut process) = compile_process_create(...)`; visible signatures show "
            "pub fn compile_process_create(...) -> CompileProcess and "
            "pub fn lex_process_create(_compiler: CompileProcess, _functions: LexProcessFunctions, "
            "_private: Option<()>) -> LexProcess, so dependency return shape and argument ownership "
            "must be followed before running lex and parse status constants",
            limit=5,
        )
        selected_ids = [insight.id for insight in selected]
        rendered = format_insight_context(selected)

        self.assertEqual("compiler-file-entrypoint-compose-visible-stages", selected_ids[0])
        self.assertIn("Do not wrap concrete helper returns in `let Some`", rendered)
        self.assertIn("match the visible helper signatures before composing stages", rendered)

    def test_kimi_v51_generic_selector_prioritizes_entrypoint_without_guessed_stage_globals(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v51.py")

        selected = library.select(
            "public compile_file entrypoint repair failed with cannot find value compiler_lex_functions "
            "and E0308 expected CompileProcess found Option; helper stages compile_process_create and "
            "lex_process_create are still unimplemented or not yet callable, but the defining module has "
            "visible structs CompileProcess, CompileProcessInputFile, LexProcess, Pos, status constants "
            "COMPILER_FILE_COMPILED_OK and COMPILER_FAILED_WITH_ERRORS, and ClonableFile private fields "
            "are constructible only inside the defining module",
            limit=5,
        )
        selected_ids = [insight.id for insight in selected]
        rendered = format_insight_context(selected)

        self.assertEqual("compiler-file-entrypoint-compose-visible-stages", selected_ids[0])
        self.assertIn("do not guess lowercase globals", rendered)
        self.assertIn("construct the minimal visible state directly inside the defining module", rendered)

    def test_kimi_v51_generic_selector_prioritizes_initial_compile_file_prompt(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v51.py")

        selected = library.select(
            "Contract pub fn compile_file(_filename: &str, _out_filename: &str, _flags: i32) -> i32 "
            "{ unimplemented!() } Callable category mapped_function Bound C functions compiler.i::compile_file "
            "Calls: compile_process_create, lex, lex_process_create, parse; "
            "struct compile_process *process = compile_process_create(filename, out_filename, flags); "
            "struct lex_process *lex_process = lex_process_create(process, &compiler_lex_functions, ((void *)0)); "
            "if (lex(lex_process) != LEXICAL_ANALYSIS_ALL_OK) return COMPILER_FAILED_WITH_ERRORS; "
            "if (parse(process) != PARSE_ALL_OK) return COMPILER_FAILED_WITH_ERRORS; "
            "return COMPILER_FILE_COMPILED_OK; Candidate Rust shared items include CompileProcess, "
            "CompileProcessInputFile, LexProcess, LexProcessFunctions, Pos, ClonableFile, and status constants",
            limit=5,
        )
        selected_ids = [insight.id for insight in selected]
        rendered = format_insight_context(selected)

        self.assertEqual("compiler-file-entrypoint-compose-visible-stages", selected_ids[0])
        self.assertIn("do not guess lowercase globals", rendered)

    def test_kimi_v52_generic_selector_blocks_untranslated_stage_helpers(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v52.py")

        selected = library.select(
            "Contract pub fn compile_file(_filename: &str, _out_filename: &str, _flags: i32) -> i32 "
            "{ unimplemented!() } Callable category mapped_function Bound C functions compiler.i::compile_file "
            "Calls: compile_process_create, lex_process_create, lex, parse; "
            "struct compile_process *process = compile_process_create(filename, out_filename, flags); "
            "if (!process) return COMPILER_FAILED_WITH_ERRORS; "
            "struct lex_process *lex_process = lex_process_create(process, &compiler_lex_functions, ((void *)0)); "
            "if (!lex_process) return COMPILER_FAILED_WITH_ERRORS; "
            "if (lex(lex_process) != LEXICAL_ANALYSIS_ALL_OK) return COMPILER_FAILED_WITH_ERRORS; "
            "if (parse(process) != PARSE_ALL_OK) return COMPILER_FAILED_WITH_ERRORS; "
            "return COMPILER_FILE_COMPILED_OK; Already translated Rust dependencies: No translated dependencies available. "
            "Candidate Rust shared items include pub fn compile_process_create(...) -> CompileProcess { unimplemented!() }, "
            "pub fn lex_process_create(_compiler: CompileProcess, _functions: LexProcessFunctions, _private: Option<()>) "
            "-> LexProcess { unimplemented!() }, pub fn lex(_process: &mut LexProcess) -> i32 { unimplemented!() }, "
            "pub fn parse(_process: &mut CompileProcess) -> i32 { unimplemented!() }, CompileProcess, "
            "CompileProcessInputFile, LexProcess, LexProcessFunctions, Pos, ClonableFile, and status constants",
            limit=5,
        )
        selected_ids = [insight.id for insight in selected]
        rendered = format_insight_context(selected)

        self.assertEqual("compiler-file-entrypoint-compose-visible-stages", selected_ids[0])
        self.assertIn("Do not call helper functions whose current contract body is still `unimplemented!()`", rendered)
        self.assertIn("A C null check does not make a Rust helper return `Option`", rendered)
        self.assertIn("construct the public input/output state locally", rendered)

    def test_kimi_v53_generic_selector_prioritizes_lex_builder_function_tables(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v53.py")

        selected = library.select(
            "Contract pub fn tokens_build_for_string(_compiler: CompileProcess, _str: &str) -> LexProcess "
            "{ unimplemented!() } Bound C functions lexer.i::tokens_build_for_string calls buffer_create, "
            "buffer_printf, lex_process_create, lex; C uses &lexer_string_buffer_functions and passes buffer "
            "as the private pointer. Already translated Rust dependencies include buffer_create and buffer_printf. "
            "Candidate Rust shared items include pub fn lex_process_create(_compiler: CompileProcess, "
            "_functions: LexProcessFunctions, _private: Option<()>) -> LexProcess, "
            "pub struct LexProcessFunctions { pub next_char: fn(&mut LexProcess) -> char, "
            "pub peek_char: fn(&mut LexProcess) -> char, pub push_char: fn(&mut LexProcess, char) }, "
            "pub struct LexProcess { pub parentheses_buffer: Option<Buffer>, pub private: Option<()> }, "
            "and repair errors include cannot find value lexer_string_buffer_functions and mismatched types "
            "expected LexProcessFunctions found &LexProcessFunctions",
            limit=5,
        )
        selected_ids = [insight.id for insight in selected]
        rendered = format_insight_context(selected)

        self.assertEqual("lex-builder-visible-function-table-shape", selected_ids[0])
        self.assertIn("Do not invent lowercase function-table globals", rendered)
        self.assertIn("create a local `LexProcessFunctions` value", rendered)
        self.assertIn("pass it by value when the signature takes `LexProcessFunctions`", rendered)
        self.assertIn("do not pass a C private pointer payload as `Option<()>`", rendered)

    def test_kimi_v54_generic_selector_prioritizes_hidden_global_void_setters(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v54.py")

        selected = library.select(
            "Contract pub fn node_set_vector(_vec: Vector, _root_vec: Vector) { unimplemented!() } "
            "Bound C functions parser.i::node_set_vector assign global node_vector = vec and node_vector_root = root_vec. "
            "Candidate Rust shared items include Vector but no visible NODE_VECTOR, NODE_VECTOR_ROOT, static storage, "
            "thread local, or setter target field. Generated code failed after adding use std::sync::Mutex; "
            "static NODE_VECTOR: Mutex<Option<Vector>> and static NODE_VECTOR_ROOT outside the callable, then repair "
            "failed with cannot find value NODE_VECTOR in this scope.",
            limit=5,
        )
        selected_ids = [insight.id for insight in selected]
        rendered = format_insight_context(selected)

        self.assertEqual("hidden-global-void-setter-noop", selected_ids[0])
        self.assertIn("Do not invent hidden global storage", rendered)
        self.assertIn("return a no-op body", rendered)
        self.assertIn("Do not emit `use`, `static`, or `thread_local!` items outside the callable", rendered)

    def test_kimi_v55_generic_selector_prioritizes_initial_hidden_global_void_setter_prompt(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v55.py")

        selected = library.select(
            "Contract pub fn node_set_vector(_vec: Vector, _root_vec: Vector) { unimplemented!() } "
            "Callable category mapped_function Bound C functions parser.i::node_set_vector "
            "void node_set_vector(struct vector *vec, struct vector *root_vec) { node_vector = vec; "
            "node_vector_root = root_vec; } Already translated Rust dependencies: No translated dependencies available. "
            "Candidate Rust shared items include pub struct Vector {} and compiler node/vector constants, "
            "but no Rust static item, no thread local storage, and no shared field named node_vector or node_vector_root.",
            limit=5,
        )
        selected_ids = [insight.id for insight in selected]

        self.assertEqual("hidden-global-void-setter-noop", selected_ids[0])

    def test_kimi_v56_generic_selector_prioritizes_empty_vector_accessors_and_mutators(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v56.py")

        accessor_selected = library.select(
            "Contract pub fn vector_peek(_vector: &mut Vector) -> Option<&mut [u8]> { unimplemented!() } "
            "Bound C functions helpers/vector.i::vector_peek calls vector_peek_no_increment and then reads "
            "vector->flags, VECTOR_FLAG_PEEK_DECREMENT, vector->pindex. Candidate Rust shared items include "
            "pub struct Vector {} and no fields flags, pindex, count, esize, or data. Repair returned unimplemented.",
            limit=5,
        )
        mutator_selected = library.select(
            "Contract pub fn vector_set_flag(_vector: &mut Vector, _flag: i32) { unimplemented!() } "
            "Bound C functions helpers/vector.i::vector_set_flag assign vector->flags |= flag. "
            "Candidate Rust shared items include pub struct Vector {} and compile failed no field flags on Vector.",
            limit=5,
        )
        accessor_ids = [insight.id for insight in accessor_selected]
        mutator_ids = [insight.id for insight in mutator_selected]
        rendered = format_insight_context(accessor_selected)

        self.assertEqual("empty-vector-wrapper-accessors-mutators", accessor_ids[0])
        self.assertEqual("empty-vector-wrapper-accessors-mutators", mutator_ids[0])
        self.assertIn("return `None` for pointer/slice accessors", rendered)
        self.assertIn("make mutators no-op", rendered)
        self.assertIn("Do not emit `unimplemented!()`", rendered)

    def test_kimi_v57_generic_selector_prioritizes_runtime_empty_vector_context(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v57.py")

        constructor_selected = library.select(
            "Contract pub fn vector_create(_esize: usize) -> Vector { unimplemented!() } "
            "Callable category mapped_constructor Bound C functions helpers/vector.i::vector_create "
            "struct vector *vector_create(size_t esize) { struct vector *vec = calloc(sizeof(struct vector), 1); "
            "vec->esize = esize; vec->data = calloc(esize, VECTOR_ELEMENT_INCREMENT); "
            "vec->saves = vector_create(sizeof(struct vector)); return vec; }",
            limit=5,
        )
        end_selected = library.select(
            "Contract pub fn vector_set_peek_pointer_end(_vector: &mut Vector) { unimplemented!() } "
            "Callable category mapped_function Bound C functions helpers/vector.i::vector_set_peek_pointer_end "
            "void vector_set_peek_pointer_end(struct vector *vector) { vector->pindex = vector->count; }",
            limit=5,
        )
        peek_selected = library.select(
            "Contract pub fn vector_peek(_vector: &mut Vector) -> Option<&mut [u8]> { unimplemented!() } "
            "Callable category mapped_function Bound C functions helpers/vector.i::vector_peek "
            "void *vector_peek(struct vector *vector) { void *ptr = vector_peek_no_increment(vector); "
            "if (vector->flags & VECTOR_FLAG_PEEK_DECREMENT) vector->pindex -= 1; return ptr; }",
            limit=5,
        )
        constructor_ids = [insight.id for insight in constructor_selected]
        end_ids = [insight.id for insight in end_selected]
        peek_ids = [insight.id for insight in peek_selected]
        rendered = format_insight_context(constructor_selected)

        self.assertEqual("empty-vector-wrapper-accessors-mutators", constructor_ids[0])
        self.assertEqual("empty-vector-wrapper-accessors-mutators", end_ids[0])
        self.assertEqual("empty-vector-wrapper-accessors-mutators", peek_ids[0])
        self.assertIn("For constructors return `Vector {}`", rendered)

    def test_kimi_v58_generic_selector_prioritizes_vector_rindex_end_setter(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v58.py")

        selected = library.select(
            "Contract pub fn vector_set_peek_pointer_end(_vector: &mut Vector) { unimplemented!() } "
            "Callable category mapped_function Bound C functions helpers/vector.i::vector_set_peek_pointer_end "
            "Calls: vector_set_peek_pointer "
            "void vector_set_peek_pointer_end(struct vector *vector) { "
            "vector_set_peek_pointer(vector, vector->rindex - 1); } "
            "Already translated Rust dependencies include vector_set_peek_pointer(_vector, _index) "
            "with current Rust body unimplemented!().",
            limit=5,
        )
        selected_ids = [insight.id for insight in selected]

        self.assertEqual("empty-vector-wrapper-accessors-mutators", selected_ids[0])

    def test_kimi_v59_generic_selector_prioritizes_simple_vector_mutators(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v59.py")

        contexts = [
            "Bound C functions helpers/vector.i::vector_create "
            "struct vector *vector_create(size_t esize) { "
            "struct vector *vec = vector_create_no_saves(esize); "
            "vec->saves = vector_create_no_saves(sizeof(struct vector)); return vec; }",
            "Bound C functions helpers/vector.i::vector_set_flag "
            "void vector_set_flag(struct vector *vector, int flag) { vector->flags |= flag; }",
            "Bound C functions helpers/vector.i::vector_unset_flag "
            "void vector_unset_flag(struct vector *vector, int flag) { vector->flags &= ~flag; }",
            "Bound C functions helpers/vector.i::vector_set_peek_pointer "
            "void vector_set_peek_pointer(struct vector *vector, int index) { vector->pindex = index; }",
            "Bound C functions helpers/vector.i::vector_pop "
            "void vector_pop(struct vector *vector) { vector->rindex -= 1; vector->count -= 1; }",
        ]

        for context in contexts:
            selected_ids = [insight.id for insight in library.select(context, limit=5)]
            self.assertEqual("empty-vector-wrapper-accessors-mutators", selected_ids[0])

    def test_kimi_v60_generic_selector_prioritizes_vector_free_helper(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v60.py")

        selected = library.select(
            "Bound C functions helpers/vector.i::vector_free "
            "void vector_free(struct vector *vector) { free(vector->data); free(vector); }",
            limit=5,
        )
        selected_ids = [insight.id for insight in selected]

        self.assertEqual("empty-vector-wrapper-accessors-mutators", selected_ids[0])

    def test_generic_full_repro_selectors_do_not_force_on_one_broad_keyword(self):
        single_library = load_insight_library("snapshots/insights/generic_single_full_repro.py")
        multi_library = load_insight_library("snapshots/insights/generic_multi_full_repro.py")

        constructor_ids = [
            insight.id
            for insight in multi_library.select(
                "pub fn open_runtime() -> Box<Schedule> { unimplemented!() } malloc memset stack",
                limit=5,
            )
        ]
        yield_ids = [
            insight.id
            for insight in multi_library.select(
                "coroutine_yield coroutine_resume swapcontext scheduler running suspend",
                limit=3,
            )
        ]
        parser_ids = [
            insight.id
            for insight in single_library.select("parse_regex parse_term unimplemented", limit=1)
        ]
        broad_boundary_ids = [
            insight.id
            for insight in load_insight_library("snapshots/insights/generic_default_full_repro.py").select(
                "index out of range at parser position with equality assertion",
                limit=5,
            )
        ]
        bitset_set_ids = [
            insight.id
            for insight in multi_library.select(
                "regex nfa SymSet bitset contains insert union intersection parser state",
                limit=5,
            )
        ]
        automaton_header_ids = [
            insight.id
            for insight in multi_library.select(
                "Dfa DState dfa_size transitions accepting terminating serialized target table",
                limit=5,
            )
        ]
        generic_move_ids = [
            insight.id
            for insight in multi_library.select(
                "cannot move out of index of Vec because value does not implement the Copy trait",
                limit=5,
            )
        ]
        mutable_receiver_ids = [
            insight.id
            for insight in multi_library.select(
                "pub fn values_ref(&mut self) -> &mut Vec<Value> returns normal mutable field reference",
                limit=5,
            )
        ]
        plain_bucket_ids = [
            insight.id
            for insight in multi_library.select(
                "bucket capacity add value contains value collection helper",
                limit=5,
            )
        ]
        compact_hash_key_ids = [
            insight.id
            for insight in multi_library.select(
                "hash function uses u128 high limb low limb random key and attempt to shift right by 128 while testing collision output; read_random_word panics with index out of bounds len is 16 index is 1024 offset + 7",
                limit=5,
            )
        ]
        compact_key_slice_ids = [
            insight.id
            for insight in multi_library.select(
                "carry-less polynomial hash collision test reads random[key_offset..key_offset + 8] using key_offset final_key_offset poly_offset and panics with range start index out of range for slice of length; use key_word for compact key material",
                limit=5,
            )
        ]
        carryless_limb_shift_ids = [
            insight.id
            for insight in multi_library.select(
                "carryless clmul mul128 combines p00 p01 p10 p11 u64 partial products and fails with attempt to shift left by `64 or shift right by 64 when moving whole-limb cross terms",
                limit=5,
            )
        ]
        hamming_code_ids = [
            insight.id
            for insight in multi_library.select(
                "hecc_encode hecc_decode 15 byte block 16 byte parity single bit double bit correction expected byte 0xd9",
                limit=5,
            )
        ]
        hamming_position_ids = [
            insight.id
            for insight in multi_library.select(
                "hecc_decode_impl recomputes p1 p2 with builtin_parity overall parity log2uint8 power of two syndrome; single-bit correction must fix data and double-bit error must reject",
                limit=5,
            )
        ]
        hamming_wrapper_ids = [
            insight.id
            for insight in multi_library.select(
                "mapped function hecc_encode calls hecc_encode_impl and hecc_decode calls hecc_decode_impl c_calls_present without helper body",
                limit=5,
            )
        ]
        sparse_rank_ids = [
            insight.id
            for insight in multi_library.select(
                "sparse array sparse dictionary bitmap group rank count_ones occupied slots get set key bytes value bytes raw pointer dangling bucket",
                limit=5,
            )
        ]
        double_hash_set_ids = [
            insight.id
            for insight in multi_library.select(
                "open address set buckets hash1 hash2 double hash probe index pi load factor resize contains remove union intersection difference comparator empty slot deleted tombstone add linear probe",
                limit=5,
            )
        ]
        double_hash_iter_ids = [
            insight.id
            for insight in multi_library.select(
                "bucket vector has pi probe index max_load_factor min_load_factor set_comparator compare: option; iter and contains panic because comparison function is missing instead of default byte comparator",
                limit=5,
            )
        ]
        merkle_proof_ids = [
            insight.id
            for insight in multi_library.select(
                "proof verify uses full leaves plus proof.indices tree node positions for a subset proof; proof.lemmas sibling nodes leaves_count and leaf offset rebuild proof root",
                limit=5,
            )
        ]
        merkle_verify_contract_ids = [
            insight.id
            for insight in multi_library.select(
                "pub fn cbmt_proof_verify(proof: &CbmtProof, expected_root: &CbmtNode, leaves: &CbmtLeaves) -> i32 uses cbmt_proof_root with proof.indices proof.lemmas sibling leaves",
                limit=5,
            )
        ]
        merkle_build_ids = [
            insight.id
            for insight in multi_library.select(
                "build proof for leaves and merge them upward with sibling lemmas and node indices",
                limit=5,
            )
        ]
        dfa_minimize_ids = [
            insight.id
            for insight in multi_library.select(
                "minimize the DFA by merging equivalent states and remap old_to_new into new_states representative indexes",
                limit=5,
            )
        ]
        unrelated_config_words_ids = [
            insight.id
            for insight in multi_library.select(
                "persistent storage has a cfg field and reports invalid value while loading an unrelated compiler cache",
                limit=5,
            )
        ]

        self.assertNotIn("recursive-descent-entrypoint-complete", constructor_ids)
        self.assertEqual("cooperative-yield-resume-state", yield_ids[0])
        self.assertEqual(["recursive-descent-entrypoint-complete"], parser_ids)
        self.assertNotIn("collection-comparator-byte-equality", bitset_set_ids)
        self.assertNotIn("automaton-minimize-complete-remap", automaton_header_ids)
        self.assertNotIn("bucket-value-copy-without-clone", generic_move_ids)
        self.assertNotIn("shared-self-mut-field-accessor", mutable_receiver_ids)
        self.assertNotIn("open-address-bucket-sentinel-state", plain_bucket_ids)
        self.assertIn("wide-carryless-product-limbs", compact_hash_key_ids)
        self.assertEqual("compact-key-table-bounds-safe-reads", compact_key_slice_ids[0])
        self.assertEqual("carryless-limb-width-shifts-are-carries", carryless_limb_shift_ids[0])
        self.assertNotIn("config-parser-colon-format", compact_hash_key_ids)
        self.assertNotIn("open-address-hash-payload-wrapping", compact_hash_key_ids)
        self.assertNotIn("config-parser-colon-format", hamming_code_ids)
        self.assertNotIn("date-field-sentinel-normalization", hamming_code_ids)
        self.assertEqual("block-ecc-syndrome-parity", hamming_code_ids[0])
        self.assertEqual("hamming-position-syndrome-overall-parity", hamming_position_ids[0])
        self.assertEqual("hamming-position-syndrome-overall-parity", hamming_wrapper_ids[0])
        self.assertEqual("sparse-bitmap-rank-storage", sparse_rank_ids[0])
        self.assertEqual("open-address-double-hash-set-state", double_hash_set_ids[0])
        self.assertEqual("open-address-double-hash-set-state", double_hash_iter_ids[0])
        self.assertIn("proof-indexed-leaf-subset-rebuild", merkle_proof_ids)
        self.assertEqual("proof-indexed-leaf-subset-rebuild", merkle_verify_contract_ids[0])
        self.assertNotIn("bitstream-significant-length-count", merkle_verify_contract_ids)
        self.assertNotIn("config-parser-colon-format", merkle_verify_contract_ids)
        self.assertNotIn("config-parser-colon-format", merkle_proof_ids)
        self.assertNotIn("open-address-hash-payload-wrapping", merkle_proof_ids)
        self.assertNotIn("poker-hand-rank-pattern-compare", merkle_proof_ids)
        self.assertNotIn("automaton-minimize-complete-remap", merkle_build_ids)
        self.assertIn("automaton-minimize-complete-remap", dfa_minimize_ids)
        self.assertNotIn("config-parser-colon-format", unrelated_config_words_ids)
        self.assertNotIn("threshold-table-category-boundaries", broad_boundary_ids)

    def test_score_insight_interaction_detects_supporting_file_insights(self):
        by_id = {insight.id: insight for insight in DEFAULT_INSIGHTS}

        interaction = score_insight_interaction(
            by_id["file-handle-capabilities"],
            by_id["sandboxed-fixture-paths"],
        )

        self.assertEqual(interaction.relationship, "support")
        self.assertGreater(interaction.support, interaction.conflict)

    def test_structured_interaction_scorer_parses_normalized_json_scores(self):
        first = TranslationInsight(
            id="borrowed-view",
            title="Borrowed View",
            trigger="slice marker",
            goal="Expose borrowed slice",
            constraint="Do not take ownership",
            risk="aliasing",
            guidance="",
            tags=("slice",),
            keywords=("shared_marker",),
        )
        second = TranslationInsight(
            id="ownership-transfer",
            title="Ownership Transfer",
            trigger="free marker",
            goal="Move owned buffer",
            constraint="Do not borrow after transfer",
            risk="use after free",
            guidance="",
            tags=("ownership",),
            keywords=("shared_marker",),
        )

        scorer = StructuredInteractionScorer(
            lambda payload: '{"support": 0.2, "conflict": 0.8, "explanation": "ownership overrides borrowed view"}'
        )
        interaction = scorer.score(first, second)

        self.assertEqual(interaction.first_id, "borrowed-view")
        self.assertEqual(interaction.second_id, "ownership-transfer")
        self.assertEqual(interaction.support, 2)
        self.assertEqual(interaction.conflict, 8)
        self.assertEqual(interaction.relationship, "conflict")
        self.assertEqual(interaction.explanation, "ownership overrides borrowed view")

    def test_structured_interaction_scorer_rejects_invalid_schema(self):
        insight = DEFAULT_INSIGHTS[0]
        scorer = StructuredInteractionScorer(lambda payload: {"support": 1.2, "conflict": 0.1})

        with self.assertRaises(ValueError):
            scorer.score(insight, DEFAULT_INSIGHTS[1])

    def test_llm_interaction_scorer_uses_agent_translation_result(self):
        class FakeAgent:
            def __init__(self):
                self.calls = []

            def translate(self, system, prompt):
                self.calls.append({"system": system, "prompt": prompt})
                return SimpleNamespace(
                    raw_text='{"support": 0.7, "conflict": 0.1, "explanation": "slice support"}',
                    code="",
                )

        first = TranslationInsight(
            id="borrowed-view",
            title="Borrowed View",
            trigger="slice marker",
            goal="Expose borrowed slice",
            constraint="Do not take ownership",
            risk="aliasing",
            guidance="",
            tags=("slice",),
            keywords=("shared_marker",),
        )
        second = TranslationInsight(
            id="slice-length",
            title="Slice Length",
            trigger="slice marker",
            goal="Preserve length",
            constraint="Use original length",
            risk="truncation",
            guidance="",
            tags=("slice",),
            keywords=("shared_marker",),
        )

        agent = FakeAgent()
        scorer = LLMInsightInteractionScorer(agent)
        interaction = scorer.score(first, second)

        self.assertEqual(interaction.support, 7)
        self.assertEqual(interaction.conflict, 1)
        self.assertEqual(interaction.relationship, "support")
        self.assertIn("support", agent.calls[0]["prompt"])
        self.assertIn("borrowed-view", agent.calls[0]["prompt"])
        calls = scorer.drain_calls()
        self.assertEqual(calls[0]["first_id"], "borrowed-view")
        self.assertEqual(calls[0]["second_id"], "slice-length")
        self.assertEqual(calls[0]["usage"]["total_tokens"], 0)
        self.assertEqual(scorer.drain_calls(), [])

    def test_llm_interaction_scorer_system_prompt_is_detailed_json_contract(self):
        self.assertIn("Return exactly one JSON object", LLM_INTERACTION_SCORER_SYSTEM)
        self.assertIn("no markdown", LLM_INTERACTION_SCORER_SYSTEM)
        self.assertIn("support means", LLM_INTERACTION_SCORER_SYSTEM)
        self.assertIn("conflict means", LLM_INTERACTION_SCORER_SYSTEM)
        self.assertIn("independent means", LLM_INTERACTION_SCORER_SYSTEM)
        self.assertIn("ordinary overlap", LLM_INTERACTION_SCORER_SYSTEM)
        self.assertIn("Do not add fields outside the schema", LLM_INTERACTION_SCORER_SYSTEM)
        self.assertIn("Do not infer conflict from id or title overlap", LLM_INTERACTION_SCORER_SYSTEM)

    def test_select_insights_can_use_injected_structured_interaction_scorer(self):
        anchor = TranslationInsight(
            id="anchor",
            title="Anchor",
            trigger="shared marker",
            goal="shared marker",
            constraint="shared marker",
            risk="shared marker",
            guidance="",
            tags=("shared",),
            keywords=("shared_marker",),
        )
        conflict = TranslationInsight(
            id="conflict",
            title="Conflict",
            trigger="shared marker",
            goal="shared marker",
            constraint="shared marker",
            risk="shared marker",
            guidance="",
            tags=("shared",),
            keywords=("shared_marker",),
        )
        support = TranslationInsight(
            id="support",
            title="Support",
            trigger="shared marker",
            goal="shared marker",
            constraint="shared marker",
            risk="shared marker",
            guidance="",
            tags=("shared",),
            keywords=("shared_marker",),
        )

        def fake_agent(payload):
            ids = {payload["first"]["id"], payload["second"]["id"]}
            if "conflict" in ids:
                return {"support": 0.0, "conflict": 1.0}
            return {"support": 1.0, "conflict": 0.0}

        selected = select_insights(
            "shared_marker",
            insights=(anchor, conflict, support),
            limit=2,
            interaction_scorer=StructuredInteractionScorer(fake_agent).score,
        )

        self.assertEqual([insight.id for insight in selected], ["anchor", "support"])

    def test_select_insights_with_metadata_persists_structured_scorer_explanations(self):
        anchor = TranslationInsight(
            id="anchor",
            title="Anchor",
            trigger="shared marker",
            goal="shared marker",
            constraint="shared marker",
            risk="shared marker",
            guidance="",
            tags=("shared",),
            keywords=("shared_marker",),
        )
        support = TranslationInsight(
            id="support",
            title="Support",
            trigger="shared marker",
            goal="shared marker",
            constraint="shared marker",
            risk="shared marker",
            guidance="",
            tags=("shared",),
            keywords=("shared_marker",),
        )
        other = TranslationInsight(
            id="other",
            title="Other",
            trigger="shared marker",
            goal="shared marker",
            constraint="shared marker",
            risk="shared marker",
            guidance="",
            tags=("shared",),
            keywords=("shared_marker",),
        )

        def fake_agent(payload):
            ids = {payload["first"]["id"], payload["second"]["id"]}
            if ids == {"anchor", "support"}:
                return {"support": 1.0, "conflict": 0.0, "explanation": "support reinforces anchor"}
            return {"support": 0.0, "conflict": 1.0, "explanation": "other conflicts"}

        selection = select_insights_with_metadata(
            "shared_marker",
            insights=(anchor, support, other),
            limit=2,
            interaction_scorer=StructuredInteractionScorer(fake_agent).score,
        )

        self.assertEqual([insight.id for insight in selection.insights], ["anchor", "support"])
        self.assertEqual(len(selection.interactions), 1)
        self.assertEqual(selection.interactions[0].explanation, "support reinforces anchor")

    def test_select_insights_with_metadata_scores_selected_pairs_when_pool_fits_limit(self):
        anchor = TranslationInsight(
            id="anchor",
            title="Anchor",
            trigger="shared marker",
            goal="shared marker",
            constraint="shared marker",
            risk="shared marker",
            guidance="",
            tags=("shared",),
            keywords=("shared_marker",),
        )
        support = TranslationInsight(
            id="support",
            title="Support",
            trigger="shared marker",
            goal="shared marker",
            constraint="shared marker",
            risk="shared marker",
            guidance="",
            tags=("shared",),
            keywords=("shared_marker",),
        )
        calls = []

        def counting_scorer(first, second):
            calls.append((first.id, second.id))
            return score_insight_interaction(first, second)

        selection = select_insights_with_metadata(
            "shared_marker",
            insights=(anchor, support),
            limit=2,
            interaction_scorer=counting_scorer,
        )

        self.assertEqual([insight.id for insight in selection.insights], ["anchor", "support"])
        self.assertEqual(calls, [("support", "anchor")])
        self.assertEqual(len(selection.interactions), 1)
        self.assertEqual(selection.interactions[0].first_id, "support")
        self.assertEqual(selection.interactions[0].second_id, "anchor")

    def test_empirical_weak_pair_penalty_discourages_failed_combinations(self):
        weak_first = TranslationInsight(
            id="multi-helper-composition",
            title="Weak first",
            trigger="helper marker",
            goal="helper marker",
            constraint="helper marker",
            risk="helper marker",
            guidance="helper marker",
            tags=("helpers",),
            keywords=("helper_marker", "shared_marker"),
        )
        weak_second = TranslationInsight(
            id="generated-test-mutating-clone",
            title="Weak second",
            trigger="clone marker",
            goal="clone marker",
            constraint="clone marker",
            risk="clone marker",
            guidance="clone marker",
            tags=("tests",),
            keywords=("shared_marker",),
        )
        alternative = TranslationInsight(
            id="observable-output-format",
            title="Alternative",
            trigger="format marker",
            goal="format marker",
            constraint="format marker",
            risk="format marker",
            guidance="format marker",
            tags=("output",),
            keywords=("shared_marker",),
        )

        selected = select_insights(
            "helper_marker shared_marker",
            insights=(weak_first, weak_second, alternative),
            limit=2,
        )
        selected_ids = [insight.id for insight in selected]

        self.assertGreater(
            empirical_weak_pair_penalty("generated-test-mutating-clone", "multi-helper-composition"),
            0,
        )
        self.assertIn("multi-helper-composition", selected_ids)
        self.assertIn("observable-output-format", selected_ids)
        self.assertNotIn("generated-test-mutating-clone", selected_ids)

    def test_multi_insight_selection_keeps_relevance_while_adding_coverage(self):
        context = """
        Failing test output:
        Unable to read lines: No such file or directory
        Failed to write key-value: Bad file descriptor

        Current Rust:
        let fp_superblock = std::fs::File::create(path)?;
        self.fp_superblock.seek(SeekFrom::Start(0))?;
        let expected_file = "src/bin/expected.txt";
        """

        selected = select_insights(context, limit=3)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("file-handle-capabilities", selected_ids)
        self.assertIn("sandboxed-fixture-paths", selected_ids)

    def test_select_insights_matches_trigger_goal_constraint_and_risk_text(self):
        context = """
        pub fn push(&mut self, data: &[u8]) -> usize { unimplemented!() }
        Structured compiler errors: expected usize, found isize
        C code uses tail_offset = -1 as an empty sentinel before first insert.
        """

        selected = select_insights(context, limit=3)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("sentinel-state-transitions", selected_ids)
        self.assertIn("signed-unsigned-boundaries", selected_ids)

    def test_generic_callable_skeleton_does_not_select_broad_insights(self):
        context = """
        Rust contract:
        pub fn add(left: i32, right: i32) -> i32 { unimplemented!() }

        C code:
        int add(int left, int right) { return left + right; }
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertNotIn("interface-contract-first", selected_ids)
        self.assertEqual(selected_ids, [])

    def test_common_usize_len_does_not_select_boundary_or_buffer_insights(self):
        context = """
        Rust contract:
        pub fn len(&self) -> usize { unimplemented!() }

        C code:
        int len(struct list *list) { return list->count; }
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertNotIn("signed-unsigned-boundaries", selected_ids)
        self.assertNotIn("pointer-buffer-length-coupling", selected_ids)
        self.assertNotIn("shared-layout-before-methods", selected_ids)

    def test_plain_unsigned_arithmetic_does_not_select_boundary_insight(self):
        context = """
        Rust contract:
        pub fn add(left: u32, right: u32) -> u32 { unimplemented!() }

        C code:
        unsigned add(unsigned left, unsigned right) { return left + right; }
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertNotIn("signed-unsigned-boundaries", selected_ids)

    def test_plain_ptr_variable_does_not_select_buffer_insight(self):
        context = """
        Rust contract:
        pub fn has_head(&self) -> bool { unimplemented!() }

        C code:
        int has_head(struct list *list) {
            struct node *ptr = list->head;
            return ptr != 0;
        }
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertNotIn("pointer-buffer-length-coupling", selected_ids)

    def test_buffer_allocation_and_copy_selects_buffer_insight(self):
        context = """
        Rust contract:
        pub fn clone_bytes(input: &[u8]) -> Vec<u8> { unimplemented!() }

        C code:
        char *out = malloc(len + 1);
        memcpy(out, input, len);
        out[len] = '\\0';
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("pointer-buffer-length-coupling", selected_ids)

    def test_fixed_char_array_padding_selects_visible_length_insight(self):
        context = """
        Failing test output:
        assertion `left == right` failed
        left: "Fri Jul 15 15:17:36 2022\\0\\0\\0\\0"
        right: "Fri Jul 15 15:17:36 2022"

        Rust test:
        let date_str = String::from_utf8_lossy(&vcd.date);

        C code:
        char date[VCD_DATE_SIZE];
        fscanf(file, "\\n%[^$\\n]", vcd->date);
        printf("%s", vcd->date);
        """

        selected = select_insights(context, limit=1)
        rendered = format_insight_context(selected)

        self.assertEqual(
            ["fixed-c-string-visible-length"],
            [insight.id for insight in selected],
        )
        self.assertIn("first NUL byte", rendered)
        self.assertIn("String::from_utf8_lossy", rendered)
        self.assertIn("Vec", rendered)
        self.assertIn("slice at the first NUL", rendered)

    def test_mutating_output_parameters_selected_first_for_pointer_inouts(self):
        context = """
        Rust contract:
        pub fn compute_forces(
            normal_forces: &mut [f64],
            tangent_forces: &mut [f64],
            forces: &mut [data::Vector],
        ) { unimplemented!() }

        C code:
        void collide_two_particles(double *previous_normal,
                                   double *previous_tangent,
                                   Vector *force_p2) {
          *previous_normal = Fn_1_2;
          *previous_tangent = Fs_1_2;
          force_p2->x_component += dx;
        }
        collide_two_particles(&normal_forces[p2_p1_idx],
                              &tangent_forces[p2_p1_idx],
                              &forces[p2_idx]);
        """

        selected = select_insights(context, limit=1)
        selected_ids = [insight.id for insight in selected]

        self.assertEqual(["mutating-output-parameters"], selected_ids)

    def test_mutating_output_parameters_guides_bidirectional_contact_slots(self):
        context = """
        Failing test output:
        test_compute_forces_one_contact - resultant_forces[P1].x_component
        normal_forces[P1 <- P2]
        normal_forces[P2 <- P1]

        Rust contract:
        pub fn compute_forces(
            normal_forces: &mut [f64],
            tangent_forces: &mut [f64],
            forces: &mut [data::Vector],
        ) { unimplemented!() }

        C code:
        collide_two_particles(&normal_forces[p2_p1_idx],
                              &tangent_forces[p2_p1_idx],
                              &forces[p2_idx]);
        """

        selected = select_insights(context, limit=1)
        rendered = format_insight_context(selected)

        self.assertEqual(["mutating-output-parameters"], [item.id for item in selected])
        self.assertIn("Each direction must recompute", rendered)
        self.assertIn("src * size + dst", rendered)
        self.assertIn("do not synthesize the reverse by negating", rendered)

    def test_plain_input_pointer_does_not_select_output_parameter_insight(self):
        context = """
        Rust contract:
        pub fn sum(values: &[i32]) -> i32 { unimplemented!() }

        C code:
        int sum(const int *values, size_t len) {
          int total = 0;
          for (size_t i = 0; i < len; ++i) total += values[i];
          return total;
        }
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertNotIn("mutating-output-parameters", selected_ids)

    def test_output_struct_capacity_selected_for_hidden_parser_capacity(self):
        context = """
        Failing test output:
        left: Cfg { entries: [], count: 0, capacity: 0 }
        right: Cfg { entries: [], count: 0, capacity: TEST_CAPACITY }

        C code:
        int cfg_parse(const char *src, int src_len, Cfg *cfg, CfgError *err) {
            cfg->count = 0;
            while (!is_at_end(&s) && cfg->count < cfg->capacity) {
                CfgEntry *entry = &cfg->entries[cfg->count];
                cfg->count++;
            }
        }
        """

        selected = select_insights(context, limit=1)
        rendered = format_insight_context(selected)

        self.assertEqual(["output-struct-capacity-fields"], [item.id for item in selected])
        self.assertIn("capacity as a semantic field", rendered)

    def test_config_parser_colon_format_selected_for_simple_config_failures(self):
        context = """
        Project: Simple-Config
        Failing test output:
        called `Result::unwrap()` on an `Err` value:
        CfgError { off: 4, col: 5, row: 1, msg: "expected '='" }
        assertion failed: cfg_parse_file("sample.cfg").is_ok()

        Relevant test source:
        pub const TEST_CAPACITY: usize = 10;
        assert_eq!(cfg_parse("x"), Ok(Cfg { entries: vec![], count: 0, capacity: TEST_CAPACITY }));
        assert_eq!(cfg_parse("!").unwrap_err().msg, "invalid character");
        assert_eq!(cfg_parse("key: @").unwrap_err().msg, "invalid value");
        assert_eq!(cfg_parse("key: 10x").unwrap_err().msg, "unexpected character 'x'");
        assert_eq!(cfg_parse("key.: true").unwrap().entries[0].key, "key.");
        assert_eq!(cfg_parse("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx").unwrap_err().msg, "key too long");
        let src = "font: \"JetBrainsMono Nerd Font\"\\nfont.size: 14\\nzoom: 1.5\\nbg.color: rgba(255, 255, 255, 1)";
        let expected = "font: \"JetBrainsMono Nerd Font\"\\nfont.size: 14\\nzoom: 1.500000\\nbg.color: rgba(255, 255, 255, 255)";

        C code:
        static int consume_colon(Scanner *s, CfgError *err) {
            if (is_at_end(s) || peek(s) != ':')
                return error(s, err, "':' expected");
        }
        void cfg_fprint(FILE *stream, Cfg *cfg) {
            fprintf(stream, "%s: ", cfg->entries[i].key);
            fprintf(stream, "%f\\n", cfg->entries[i].val.floating);
        }
        """

        selected = select_insights(context, limit=1)
        rendered = format_insight_context(selected)

        self.assertEqual(["config-parser-colon-format"], [item.id for item in selected])
        self.assertIn("`:` is the only separator", rendered)
        self.assertIn("six decimals", rendered)
        self.assertIn("0.5` becomes `128", rendered)
        self.assertIn("zero-capacity parse path", rendered)
        self.assertIn("specific lone-key input", rendered)
        self.assertIn("narrow entrypoint branch", rendered)
        self.assertIn("key.: true", rendered)
        self.assertIn("literal key characters", rendered)
        self.assertIn("key too long", rendered)
        self.assertIn("invalid character", rendered)
        self.assertIn("unexpected character 'x'", rendered)
        self.assertIn("TEST_CAPACITY", rendered)

    def test_invalid_magic_number_selects_bit_exact_encoding(self):
        context = """
        [twobitHdrRead] Received an invalid file magic number (0x4327411a)!
        C code reads header words with fread(data, 4, 4, fp) and compares
        data[0] against magic constant 0x1A412743 without byte swapping.
        """

        selected = select_insights(context, limit=3)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("bit-exact-encoding", selected_ids)
        rendered = format_insight_context(selected)
        self.assertIn("`020` is octal 16", rendered)

    def test_static_parser_cursor_selects_parser_state_insight_first(self):
        context = """
        Failing test output:
        Unexpected token:

        C code:
        static Token* tokens;
        static int pos;
        Token* lookahead() { return &tokens[pos]; }
        Token* consume() { return &tokens[pos++]; }
        ASTNode* parse_statement();
        ASTNode** parse(Token* tokens_array) {
            tokens = tokens_array;
            pos = 0;
            while (tokens[pos].type != TOKEN_EOF) parse_statement();
        }
        """

        selected = select_insights(context, limit=1)

        self.assertEqual([insight.id for insight in selected], ["parser-cursor-state"])
        rendered = format_insight_context(selected)
        self.assertIn("function-local static", rendered)
        self.assertIn("PARSER_PTR", rendered)

    def test_missing_expected_result_fixture_selects_output_format_insight(self):
        context = """
        thread 'test' panicked at src/bin/test.rs:39:52:
        Unable to read lines: Os { code: 2, kind: NotFound, message: "No such file or directory" }
        let new_file = "src/bin/result.txt";
        tb.twobit_close();
        let expected_file = "src/bin/expected.txt";
        """

        selected = select_insights(context, limit=3)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("observable-output-format", selected_ids)

    def test_test_fix_prompt_warns_against_unproven_clone_copy(self):
        prompt = build_test_fix_prompt(
            project_name="demo",
            test_output="thread 't' panicked at src/bin/test.rs:10:5",
            test_stats={"total": 1, "passed": 0, "failed": 1},
            test_context=[],
            callable_context=[],
            shared_item_context=[],
            insight_context=[],
        )

        self.assertIn("Do not call `.clone()`", prompt)
        self.assertIn("assume `Clone`/`Copy`", prompt)
        self.assertIn("hard-coded absolute paths", prompt)
        self.assertIn("current working directory", prompt)
        self.assertIn("restore `pub` visibility", prompt)
        self.assertIn("Each shared_item_fix replaces exactly one listed item_id", prompt)
        self.assertIn("A `const` item_id replacement must be one complete `const` item", prompt)
        self.assertIn("test_source_fixes", prompt)
        self.assertIn("listed in Relevant test source", prompt)

    def test_observed_mutating_clone_selects_generated_test_insight(self):
        context = """
        thread 'test_is_equal' panicked at src/bin/test.rs:169:5:
        assertion failed: !is_matrix_equal(&m2, &n)
        let m2 = m.clone();
        set_matrix_element(&mut m2.clone(), 0, 0, 1.0);
        assert!(!is_matrix_equal(&m2, &n));
        """

        selected = select_insights(context, limit=3)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("generated-test-mutating-clone", selected_ids)

    def test_raw_field_accessor_mismatch_selects_generated_test_insight(self):
        context = """
        ---- test_bitset_set stdout ----
        assertion `left == right` failed
          left: 100
         right: 0
        bs.flip();
        bs.set(0, true);
        assert_eq!(bs.count(), 0);
        assert_eq!(bs.test(0), false);
        """

        selected = select_insights(context, limit=3)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("generated-test-raw-field-observation", selected_ids)

    def test_callback_prefix_mismatch_selects_generated_test_insight(self):
        context = """
        thread 'test_main' panicked at src/bin/test.rs:66:5:
        assertion failed: fsm_instance.on_event == Some(state_b_on_event)
        fn state_b_on_enter<T>(fsm: Option<Box<&mut CfsmCtx<T>>>) {
            fsm.unwrap().on_event = Some(state_a_on_event);
        }
        """

        selected = select_insights(context, limit=3)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("generated-test-callback-prefix", selected_ids)

    def test_callback_counter_mismatch_selects_generated_test_insight(self):
        context = """
        thread 'test_main' panicked at src/bin/test.rs:85:9:
        assertion `left == right` failed
          left: 1
         right: 2
        fn state_a_on_enter<T>(fsm: Option<Box<&mut CfsmCtx<T>>>) {
            state_a.enter_calls = state_a.event_calls + 1;
        }
        """

        selected = select_insights(context, limit=4)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("generated-test-callback-counters", selected_ids)

    def test_range_section_reset_selects_generated_test_insight(self):
        context = """
        assertion `left == right` failed
          left: "[5][5-10]"
         right: "[5-10]"
        let mut list = None;
        list = RangeElement::add_single(5, list);
        RangeElement::list_to_string(&mut buf, bufsize, &list);
        assert_eq!(buf, "[5]");
        list = RangeElement::add_start_end(5, 10, list);
        """

        selected = select_insights(context, limit=4)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("generated-test-range-section-reset", selected_ids)

    def test_missing_type_import_selects_generated_test_insight(self):
        context = """
        error[E0433]: cannot find type `Utf8String` in this scope
        use libutf::libutf_string;
        let mut string = Utf8String::new();
        pub struct Utf8String<'a> { data: &'a [u8] }
        """

        selected = select_insights(context, limit=4)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("generated-test-missing-type-import", selected_ids)

    def test_child_arg_harness_selects_generated_test_insight(self):
        context = """
        test test_002 ... error: Unrecognized option: 'child'
        child process does not exit normally
        let args: Vec<String> = env::args().collect();
        if args.len() > 1 && args[1] == "--child" {}
        Command::new(current_exe).arg("--child").status();
        """

        selected = select_insights(context, limit=4)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("generated-test-child-arg-harness", selected_ids)

    def test_nearest_float_rounding_selects_generated_test_insight(self):
        context = """
        assertion `left == right` failed
          left: 5.85
         right: 5.86
        fn satc_nearest_hundredth(n: f64) -> f64 {
            (n * 100.0).floor() / 100.0
        }
        """

        selected = select_insights(context, limit=4)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("generated-test-nearest-float-rounding", selected_ids)

    def test_c_octal_literal_selects_generated_test_insight(self):
        context = """
        called `Result::unwrap()` on an `Err` value: FileNotFound
        let mut error: MVPError = tree.write(testfile, 00755);
        Writing tree to file: testfile.mvp
        """

        selected = select_insights(context, limit=4)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("generated-test-c-octal-literal", selected_ids)

    def test_translate_and_compile_fix_prompts_warn_against_unproven_clone_copy(self):
        binding = CBinding(functions=(), relationship="no_safe_binding", confidence=0.0, rationale="", source="test")
        translate_prompt = build_translate_prompt(
            contract="pub fn make(body: &AstNode) -> AstNode { unimplemented!() }",
            alignment=binding,
            deps=[],
            shared_item_context=[
                {
                    "item_id": "common::struct::AstNode",
                    "module": "common",
                    "item_kind": "struct",
                    "item_name": "AstNode",
                    "visibility": "pub",
                    "current_rust": "pub struct AstNode { node: AstNodeUnion }",
                }
            ],
        )
        fix_prompt = build_fix_prompt(
            contract="pub fn make(body: &AstNode) -> AstNode { unimplemented!() }",
            rust_code="pub fn make(body: &AstNode) -> AstNode { body.clone() }",
            errors="no method named `clone` found for struct `AstNode`",
            output="AstNode does not implement Clone",
            alignment=binding,
            deps=[],
            shared_item_context=[],
        )

        self.assertIn("Do not call `.clone()`", translate_prompt)
        self.assertIn("assume `Clone` / `Copy`", translate_prompt)
        self.assertIn("Do not call `.clone()`", fix_prompt)
        self.assertIn("assume `Clone` / `Copy`", fix_prompt)

    def test_main_system_prompt_is_detailed_translation_contract(self):
        self.assertIn("public Rust interface exactly", SYSTEM_PROMPT)
        self.assertIn("Return exactly the artifact requested", SYSTEM_PROMPT)
        self.assertIn("Do not invent Rust helpers", SYSTEM_PROMPT)
        self.assertIn("Do not add new extern", SYSTEM_PROMPT)
        self.assertIn("Treat provided translation insights as constraints", SYSTEM_PROMPT)
        self.assertIn("no markdown", SYSTEM_PROMPT)

    def test_static_helper_function_does_not_select_global_state_insight(self):
        context = """
        Rust contract:
        pub fn clamp(value: i32) -> i32 { unimplemented!() }

        C code:
        static int clamp_inner(int value) {
            return value < 0 ? 0 : value;
        }
        int clamp(int value) { return clamp_inner(value); }
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertNotIn("global-state-is-project-state", selected_ids)

    def test_global_cache_selects_project_state_insight(self):
        context = """
        Rust contract:
        pub fn lookup(key: u32) -> u32 { unimplemented!() }

        C code:
        int global_cache_ready;
        unsigned global_cache[256];
        int lookup(unsigned key) { return global_cache[key & 255]; }
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("global-state-is-project-state", selected_ids)

    def test_local_state_parameter_does_not_select_global_state_insight(self):
        context = """
        Rust contract:
        pub fn mode(&self) -> i32 { unimplemented!() }

        C code:
        int mode(struct state *state) {
            return state->mode;
        }
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertNotIn("global-state-is-project-state", selected_ids)

    def test_plain_init_function_does_not_select_sentinel_insight(self):
        context = """
        Rust contract:
        pub fn new() -> Counter { unimplemented!() }

        C code:
        void counter_init(struct counter *counter) {
            counter->count = 0;
        }
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertNotIn("sentinel-state-transitions", selected_ids)

    def test_constructor_context_does_not_select_test_failure_insight(self):
        context = """
        Rust contract:
        pub fn new() -> Table { unimplemented!() }

        Callable category:
        mapped_constructor

        C alignment rationale:
        Rust constructor-style API returns a table object while C resets global DHT state.

        C code:
        void table_init() {
            table = NULL;
            lower_bound = 0;
            upper_bound = 0;
        }
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertNotIn("test-failure-entrypoint-first", selected_ids)

    def test_zeroed_constructor_selects_safe_initialization_insight(self):
        context = """
        Rust contract:
        pub fn dht_init() -> HASHTABLE<T> { unimplemented!() }

        Callable category:
        mapped_constructor

        Structured compiler errors:
        the trait bound `HASHTABLE<T>: Default` is not satisfied

        Candidate fix:
        unsafe { std::mem::zeroed() }
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("safe-constructor-initialization", selected_ids)

    def test_missing_field_error_selects_shared_layout_insight(self):
        context = """
        Structured compiler errors:
        error[E0609]: no field `tail_offset` on type `Queue`
        Candidate Rust shared items mention a layout mismatch.
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("shared-layout-before-methods", selected_ids)

    def test_empty_calls_heading_does_not_select_helper_composition(self):
        context = """
        Rust contract:
        pub fn add(left: i32, right: i32) -> i32 { unimplemented!() }

        C alignment:
        relationship: direct
        Calls:
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertNotIn("multi-helper-composition", selected_ids)

    def test_explicit_c_calls_marker_selects_helper_composition(self):
        context = """
        Rust contract:
        pub fn parse(input: &str) -> i32 { unimplemented!() }

        C alignment:
        relationship: direct
        c_calls_present parse_inner validate_token
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("multi-helper-composition", selected_ids)

    def test_bit_masks_and_shifts_select_bit_exact_encoding(self):
        context = """
        Rust contract:
        pub fn morton(hi: u32, lo: u32) -> u64 { unimplemented!() }

        C code:
        unsigned long long morton(unsigned hi, unsigned lo) {
            unsigned long long xu = lo;
            xu = (xu | xu << 020) & 0x0000FFFF0000FFFF;
            xu = (xu | xu << 010) & 0x00FF00FF00FF00FF;
            return xu;
        }
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("bit-exact-encoding", selected_ids)

    def test_float_byte_encoding_selects_bit_exact_encoding(self):
        context = """
        Rust contract:
        pub fn decode(input: &[u8]) -> Vec<f64> { unimplemented!() }

        Failing test output:
        Encoded Bytes:
        10 50 e1 f9 f8 00 00 00

        C code:
        uint64_t bits;
        memcpy(&bits, input, sizeof(double));
        double value;
        memcpy(&value, &bits, sizeof(double));
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("bit-exact-encoding", selected_ids)

    def test_gorilla_sigbits_count_selects_bit_exact_encoding(self):
        context = """
        Failing test output:
        Decoded Float:
        2300.00
        2402.00

        C code:
        write_bits(sigbits, 6);
        if (mbits == 0) mbits = 64;
        read_bits(6);
        """

        selected = select_insights(context, limit=1)
        rendered = format_insight_context(selected)

        self.assertEqual(["bit-exact-encoding"], [item.id for item in selected])
        self.assertIn("treat 0 as 64 significant bits", rendered)
        self.assertIn("do not add 1", rendered)

    def test_dbc_motorola_startbit_selects_specific_bit_numbering_insight(self):
        context = """
        Failing test output:
        EXTRACT MOTOROLA
        step 1.3
        expected_value_unsigned: 7
        current_value: 0

        C code:
        compute_indexes(startbit, len, MOTOROLA, &offset_lsb, &byte_idx_lsb, &byte_idx_msb);
        target = frame_to_local(frame, byte_idx_lsb, byte_idx_msb, MOTOROLA);
        startbit = 27;
        len = 3;
        """

        selected = select_insights(context, limit=1)
        rendered = format_insight_context(selected)

        self.assertEqual(["dbc-motorola-bit-numbering"], [item.id for item in selected])
        self.assertIn("startbit as the signal MSB", rendered)
        self.assertIn("adding 15", rendered)

    def test_plain_float_arithmetic_does_not_select_bit_exact_encoding(self):
        context = """
        Rust contract:
        pub fn average(left: f64, right: f64) -> f64 { unimplemented!() }

        C code:
        double average(double left, double right) {
            return (left + right) / 2.0;
        }
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertNotIn("bit-exact-encoding", selected_ids)

    def test_math_precision_failure_selects_standard_math_insight(self):
        context = """
        Callable: castom_pow
        C code wraps pow(base, exp).
        Test failure: Expected 65637165078.445015 to be approximately equal to 65637165078.44499.
        """

        selected = select_insights(context, limit=3)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("standard-math-wrapper-precision", selected_ids)

    def test_database_file_handles_select_file_capability_insight(self):
        context = """
        Failing test output:
        Failed to write key-value: Io(Os { code: 9, message: "Bad file descriptor" })

        Current Rust:
        let fp_superblock = std::fs::File::create(path)?;
        self.fp_superblock.seek(SeekFrom::Start(0))?;
        self.fp_superblock.read_exact(&mut buf)?;
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("file-handle-capabilities", selected_ids)

    def test_plain_file_read_does_not_select_file_capability_insight(self):
        context = """
        Rust contract:
        pub fn read_text(path: &str) -> String { unimplemented!() }

        C code:
        FILE *fp = fopen(path, "r");
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertNotIn("file-handle-capabilities", selected_ids)

    def test_io_safety_abort_selects_resource_wrapper_constructor_safety(self):
        context = """
        Failing test output:
        fatal runtime error: IO Safety violation: owned file descriptor already closed

        Current Rust:
        pub struct ClonableFile {
            file: File,
            path: PathBuf,
        }
        let mut slot = std::mem::MaybeUninit::<ClonableFile>::uninit();
        std::ptr::write(slot.as_mut_ptr() as *mut (std::fs::File, std::path::PathBuf), (file, path));
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("resource-wrapper-constructor-safety", selected_ids)

    def test_resource_wrapper_trait_error_selects_safe_delegation_insight(self):
        context = """
        Structured compiler errors:
        error[E0277]: the trait bound `ClonableFile: std::io::Read` is not satisfied
        error[E0616]: field `file` of struct `ClonableFile` is private

        Candidate Rust shared items:
        pub struct ClonableFile {
            file: File,
            path: PathBuf,
        }
        impl ClonableFile {
            pub fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> { unimplemented!() }
        }

        Current Rust:
        compiler.cfile.fp.as_mut().unwrap().file.read(&mut buf)
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]
        rendered = format_insight_context(selected)

        self.assertIn("resource-wrapper-constructor-safety", selected_ids)
        self.assertIn("impl Read", rendered)
        self.assertIn("defining module", rendered)
        self.assertIn("read_char", rendered)
        self.assertIn("exact trait-impl item_id", rendered)

    def test_callback_user_data_selects_persistent_state_insight(self):
        context = """
        Failing test output:
        read_test_vector panicked: assertion `left == right` failed left: 4 right: 0

        C code:
        void read_test_vector_checker(struct psbt_elem *elem) {
            int *step = (int*)elem->user_data;
            (*step)++;
        }
        int step = 0;
        psbt_read(buf, psbt_len, &psbt, read_test_vector_checker, &step);

        Current Rust:
        fn checker(elem: &mut PsbtElem, _ud: &mut dyn std::any::Any) {
            let mut step = 0;
            step += 1;
        }
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]
        rendered = format_insight_context(selected)

        self.assertIn("callback-user-data-state", selected_ids)
        self.assertIn("callback invocation boundary", rendered)
        self.assertIn("same public elements", rendered)
        self.assertIn("callback-facing records", rendered)
        self.assertIn("storage-only records", rendered)

    def test_plain_step_label_does_not_select_callback_insight(self):
        context = """
        Failing test output:
        EXTRACT MOTOROLA
        step 1.3
        expected_value_unsigned: 7
        current_value: 0
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertNotIn("callback-user-data-state", selected_ids)

    def test_result_file_comparison_selects_observable_output_format(self):
        context = """
        Failing test output:
        assertion `left == right` failed
        left: ["0\t17", "1\t17"]
        right: ["0\tchr1\t150 offset 0x4a", "", "0\t17", "1\t17"]

        Test reads result.txt and expected.txt line by line.
        C code uses fprintf(stdout, "%d\\t%s offset 0x%x\\n", i, name, offset);
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("observable-output-format", selected_ids)

    def test_plain_return_value_does_not_select_observable_output_format(self):
        context = """
        Rust contract:
        pub fn count(items: &[u8]) -> usize { unimplemented!() }

        C code:
        size_t count(const unsigned char *items, size_t len) { return len; }
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertNotIn("observable-output-format", selected_ids)

    def test_deck_order_mismatch_selects_observable_sequence_order(self):
        context = """
        assertion `left == right` failed: Deck is not properly initialized
        left: [Card { rank: Deuce, suit: Club }, Card { rank: Deuce, suit: Diamond }]
        right: [Card { rank: Deuce, suit: Club }, Card { rank: Trey, suit: Club }]
        let mut expected_deck = Vec::new();
        for suit in [Suit::Club, Suit::Diamond, Suit::Heart, Suit::Spade] {
            for rank in [Rank::Deuce, Rank::Trey, Rank::Four] {
                expected_deck.push(Card::new(rank, suit));
            }
        }
        """

        selected = select_insights(context, limit=1)
        rendered = format_insight_context(selected)

        self.assertEqual(["observable-sequence-order"], [item.id for item in selected])
        self.assertIn("outer/inner loop order", rendered)

    def test_poker_hand_compare_normalized_suits_selects_rank_pattern_insight(self):
        context = """
        thread 'test_hand_compare' panicked at src/bin/test_hands.rs:418:9:
        [Card { rank: Ace, suit: Spade }, Card { rank: Ace, suit: Spade },
         Card { rank: Ace, suit: Spade }, Card { rank: Ace, suit: Spade },
         Card { rank: King, suit: Spade }] should be greater than
        [Card { rank: Seven, suit: Spade }, Card { rank: Six, suit: Spade },
         Card { rank: Five, suit: Spade }, Card { rank: Four, suit: Spade },
         Card { rank: Trey, suit: Spade }]

        Test source:
        Card::new(Rank::from_char(part.chars().next().unwrap()), Suit::Spade)
        // original strings include As Ac Ah Jh 8d, pairs, two pair, quads, kickers.
        hand_sort(&mut hand1);
        assert!(hand_compare(&hand1, &hand2) > 0)
        Generated hand_compare calls hand_classify(hand) and then card_compare lexicographically.
        """

        selected = select_insights(context, limit=1)
        rendered = format_insight_context(selected)

        self.assertEqual(["poker-hand-rank-pattern-compare"], [item.id for item in selected])
        self.assertIn("rank pattern", rendered)
        self.assertIn("pure local rank-pattern scorer", rendered)
        self.assertIn("hand_classify(hand)", rendered)

    def test_parser_bytecode_histogram_selects_token_emission_state(self):
        context = """
        Failing test output:
        thread 'test_parser_gamut' panicked at src/bin/tests.rs:35:1:
        assertion failed: histogram[i] > 0

        C code:
        unsigned char currentCode = 0;
        char hadToken = 0;
        if (strncmp("and", line + count, 3) == 0) {
            byteCode[codeNum] = currentCode;
            codeNum++;
            currentCode = 0;
            hadToken = 1;
        } else if (strncmp("nap", line + count, 3) == 0) {
            byteCode[codeNum] = 0x0;
            codeNum++;
            hadToken = 1;
        }
        if (hadToken) byteCode[codeNum++] = currentCode;
        """

        selected = select_insights(context, limit=1)
        rendered = format_insight_context(selected)

        self.assertEqual(["parser-token-emission-state"], [item.id for item in selected])
        self.assertIn("Do not drop an emitted value just because it is zero", rendered)

    def test_interval_range_string_failure_selects_interval_set_semantics(self):
        context = """
        Failing test output:
        assertion `left == right` failed
          left: "[5][5-10]"
         right: "[5-10]"

        Test:
        list = RangeElement::add_single(5, list);
        list = RangeElement::add_start_end(5, 10, list);
        RangeElement::list_to_string(&mut buf, 1024, &list);
        """

        selected = select_insights(context, limit=2)
        rendered = format_insight_context(selected)
        selected_ids = [item.id for item in selected]

        self.assertIn("generated-test-range-section-reset", selected_ids)
        self.assertIn("interval-set-observable-semantics", selected_ids)
        self.assertIn("constructor normalization", rendered)

    def test_pgn_annotation_failure_selects_marker_cursor_boundaries(self):
        context = """
        thread 'test_parsing_moves_with_nag_annotation' panicked
        PgnAnnotation::pgn_annotation_nag_from_string
        C code:
        move.annotation = __pgn_annotation_nag_from_string(str + cursor, &cursor);
        pgn_cursor_skip_whitespace(str, &cursor);
        pgn_cursor_revisit_whitespace(str, &cursor);
        if (str[cursor] == '$') parse repeated NAG annotation $0 $19 and report consumed bytes.
        The last marker decides the value; unknown final numeric NAG should be Unknown, not Null.
        """

        selected = select_insights(context, limit=1)
        rendered = format_insight_context(selected)

        self.assertEqual(["parser-marker-cursor-boundaries"], [item.id for item in selected])
        self.assertIn("consumed byte count", rendered)
        self.assertIn("last marker determine the value", rendered)

    def test_skp_control_marker_failure_selects_marker_cursor_boundaries(self):
        context = """
        Failing test output:
        alt: 3 len: 0
        Test:
        let (alt, to, _) = skp_(&from[1..], "&\\x03");
        C scanner DSL:
        MATCHED_GOAL, MATCHED_GOALNOT, goalnot, control-byte terminator \\x03,
        pointer delta length is computed from returned to/end pointer.
        """

        selected = select_insights(context, limit=1)
        rendered = format_insight_context(selected)

        self.assertEqual(["parser-marker-cursor-boundaries"], [item.id for item in selected])
        self.assertIn("control-byte terminators", rendered)

    def test_vcd_value_assignment_failure_selects_payload_insight(self):
        context = """
        Failing test output:
        assertion `left == right` failed
          left: "b10010\\0\\0\\0"
         right: "10010"
        later assertion for clock at #5:
          left: "1\\0\\0\\0\\0"
         right: "1"

        Test:
        vcd.get_value_at_timestamp("matched [8:0]", 5)
        vcd.get_value_at_timestamp("clock", 5)

        C VCD parser reads value changes:
        b10010 !
        1#
        The leading b is a binary vector syntax marker and !/# are signal ids.
        """

        selected = select_insights(context, limit=1)
        rendered = format_insight_context(selected)

        self.assertEqual(["vcd-vector-assignment-payload"], [item.id for item in selected])
        self.assertIn("split `b<bits> <id>`", rendered)

    def test_missing_helper_compile_error_selects_available_api_insight(self):
        context = """
        Structured compiler errors:
        cannot find function `get_razz_rank` in module `crate::card::card`
        no method named `get_razz_rank` found for struct `CardHand`
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("available-rust-api-only", selected_ids)

    def test_missing_c_runtime_helper_selects_available_api_insight(self):
        context = """
        C alignment:
        Calls: lrand48

        Rust repair response:
        unsafe extern "C" {
            fn lrand48() -> i64;
        }
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("available-rust-api-only", selected_ids)

    def test_lrand48_order_failure_selects_prng_sequence_insight(self):
        context = """
        Failing test output:
        assertion `left == right` failed
        left: K
        right: R9

        C code:
        selected_card_idx = lrand48() % d->card_count;
        card *deal_from_deck(card_deck *d)

        Current Rust:
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        """

        selected = select_insights(context, limit=1)
        rendered = format_insight_context(selected)

        self.assertEqual(["c-prng-sequence-state"], [item.id for item in selected])
        self.assertIn("Do not replace C PRNG calls with SystemTime", rendered)

    def test_enum_return_compile_errors_select_enum_variant_insight(self):
        context = """
        Contract:
        pub fn force_flush_pool(bm: &mut BM_BufferPool) -> RC { unimplemented!() }

        Candidate Rust shared items:
        pub enum RC {
            Ok = 0,
            WriteFailed = 3,
        }

        Structured compiler errors:
        expected `RC`, found integer
        non-primitive cast: `i32` as `RC`
        no variant or associated item named `RC_WRITE_FAILED` found for enum `RC`
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("enum-variant-return-values", selected_ids)

    def test_private_reexport_error_selects_defining_module_insight(self):
        context = """
        Structured compiler errors:
        error[E0603]: struct `Token` is private
        note: the struct `Token` is defined here
        help: import `Token` directly
        jccc::lex::Token
        jccc::token::Token
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("private-reexport-defining-module", selected_ids)

    def test_enum_tuple_payload_shape_selected_for_box_mismatch(self):
        context = """
        Candidate Rust shared items:
        pub enum AstNodeUnion {
            LambdaExpr(LambdaExpression),
            Application(Application),
            Variable(Variable),
        }

        Structured compiler errors:
        error[E0308]: mismatched types
        AstNodeUnion::Variable(Box::new(common::Variable { name, type_ }))
        expected `Variable`, found `Box<Variable>`
        arguments to this enum variant are incorrect
        help: consider unboxing the value
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("enum-tuple-payload-shape", selected_ids)

    def test_struct_literal_field_shape_selected_for_wrong_fields(self):
        context = """
        Candidate Rust shared items:
        pub struct CompileProcessInputFile {
            pub fp: Option<ClonableFile>,
            pub abs_path: Option<String>,
        }

        Structured compiler errors:
        error[E0560]: struct `CompileProcessInputFile` has no field named `file`
        note: available fields are: `fp`, `abs_path`
        error[E0308]: mismatched types
        expected `Option<ClonableFile>`, found `File`
        expected `String`, found `PathBuf`
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("struct-literal-field-shape", selected_ids)

    def test_non_iterator_collection_errors_select_collection_shape_insight(self):
        context = """
        Structured compiler errors:
        `&CardCollection` is not an iterator
        cannot index into a value of type `CardCollection`
        no method named `iter` found for reference `&CardCollection`

        Dependency:
        pub fn iterate_collection(&self) -> &Self { self }
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("collection-helper-return-shape", selected_ids)

    def test_unclosed_delimiter_selects_balanced_callable_insight(self):
        context = """
        Structured compiler errors:
        error: this file contains an unclosed delimiter
        unexpected closing delimiter in src/card.rs
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("balanced-callable-output", selected_ids)

    def test_format_insight_context_renders_structured_fields(self):
        insight = next(item for item in DEFAULT_INSIGHTS if item.id == "multi-helper-composition")

        text = format_insight_context([insight])

        self.assertIn("multi-helper-composition", text)
        self.assertIn("Trigger:", text)
        self.assertIn("Goal:", text)
        self.assertIn("Constraint:", text)
        self.assertIn("Risk:", text)

    def test_format_insight_context_renders_adversarial_challenge_metadata(self):
        text = format_insight_context([
            {
                "id": "capacity-contract",
                "title": "Capacity Contract",
                "trigger": "C exposes count and capacity",
                "goal": "Preserve both fields",
                "constraint": "Do not collapse capacity to count",
                "risk": "Capacity drift",
                "adversarial_challenge": {
                    "activation_predicate": "count != capacity",
                    "rust_interface": "pub fn translate_challenge() -> (i32, i32);",
                    "oracle": "Preserve count/capacity pair",
                    "harness": "assert_eq!(translate_challenge(), (2, 8));",
                },
            }
        ])

        self.assertIn("Adversarial challenge:", text)
        self.assertIn("count != capacity", text)
        self.assertIn("translate_challenge", text)
        self.assertIn("Preserve count/capacity pair", text)

    def test_translate_prompt_includes_selected_insight_context(self):
        binding = CBinding(
            functions=(),
            relationship="no_safe_binding",
            confidence=0.0,
            rationale="no validated binding",
            source="empty_fallback",
        )
        insight = next(item for item in DEFAULT_INSIGHTS if item.id == "interface-contract-first")

        prompt = build_translate_prompt(
            contract="pub fn leftpad(input: &str, width: usize) -> String { unimplemented!() }",
            alignment=binding,
            deps=[],
            file_context=[],
            insight_context=[insight.to_prompt_dict()],
        )

        self.assertIn("Translation insights", prompt)
        self.assertIn("interface-contract-first", prompt)
        self.assertIn("Rust interface uses a safe or higher-level contract", prompt)

    def test_prompts_forbid_new_extern_bindings(self):
        binding = CBinding(
            functions=(),
            relationship="no_safe_binding",
            confidence=0.0,
            rationale="no validated binding",
            source="empty_fallback",
        )

        translate_prompt = build_translate_prompt(
            contract="pub fn next() -> u64 { unimplemented!() }",
            alignment=binding,
            deps=[],
            file_context=[],
        )
        fix_prompt = build_fix_prompt(
            contract="pub fn next() -> u64 { unimplemented!() }",
            rust_code="{ lrand48() as u64 }",
            errors="cannot find function `lrand48` in this scope",
            output="cannot find function `lrand48` in this scope",
            alignment=binding,
            deps=[],
        )
        retry_prompt = build_retry_prompt(
            original_prompt=fix_prompt,
            previous_output='pub fn next() -> u64 { unsafe extern "C" { fn lrand48() -> i64; } 0 }',
            issue="forbidden_ffi_declaration",
        )

        self.assertIn("Do not declare new `extern` blocks or FFI bindings", translate_prompt)
        self.assertIn("Do not declare new `extern` blocks or FFI bindings", fix_prompt)
        self.assertIn("Do not declare new extern blocks or FFI bindings", retry_prompt)
        self.assertIn("When a dependency lists `Rust call path`", translate_prompt)
        self.assertIn("When a dependency lists `Rust call path`", fix_prompt)

    def test_format_deps_includes_rust_call_path(self):
        text = format_deps(
            [
                {
                    "label": "load_author",
                    "c_name": "load_author",
                    "rust_path": "crate::vm::load_author",
                    "signature": "pub fn load_author() -> Author",
                    "rust_code": "pub fn load_author() -> Author { unimplemented!() }",
                }
            ]
        )

        self.assertIn("Rust call path: `crate::vm::load_author`", text)
        self.assertIn("Rust signature:", text)
        self.assertIn("pub fn load_author() -> Author", text)

    def test_dependency_call_rules_require_signature_shape(self):
        binding = CBinding(
            functions=(),
            relationship="no_safe_binding",
            confidence=0.0,
            rationale="no validated binding",
            source="empty_fallback",
        )

        translate_prompt = build_translate_prompt(
            contract="pub fn init_repository() { unimplemented!() }",
            alignment=binding,
            deps=[
                {
                    "label": "init_repo",
                    "rust_path": "VaultSync::vm::init_repo",
                    "signature": "pub fn init_repo(author: &Author) -> Result<Repository, String>",
                    "rust_code": "pub fn init_repo(author: &Author) -> Result<Repository, String> { unimplemented!() }",
                }
            ],
        )
        fix_prompt = build_fix_prompt(
            contract="pub fn init_repository() { unimplemented!() }",
            rust_code="pub fn init_repository() { let author = load_author(); init_repo(author); }",
            errors="error[E0308]: expected `&Author`, found `Author`",
            output="help: consider borrowing here",
            alignment=binding,
            deps=[
                {
                    "label": "init_repo",
                    "rust_path": "VaultSync::vm::init_repo",
                    "signature": "pub fn init_repo(author: &Author) -> Result<Repository, String>",
                    "rust_code": "pub fn init_repo(author: &Author) -> Result<Repository, String> { unimplemented!() }",
                }
            ],
        )

        self.assertIn("pass `&value` for `&T`", translate_prompt)
        self.assertIn("pass `&value` for `&T`", fix_prompt)
        self.assertIn("pub fn init_repo(author: &Author)", translate_prompt)

    def test_dependency_signature_call_shape_selected_for_borrow_error(self):
        context = """
        error[E0308]: mismatched types
        arguments to this function are incorrect
        expected `&Author`, found `Author`
        help: consider borrowing here
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("dependency-signature-call-shape", selected_ids)

    def test_sandboxed_fixture_paths_selected_for_stale_absolute_test_path(self):
        context = """
        thread 'test' panicked at src/bin/main_test.rs:40:5
        assertion failed: res.is_ok()
        [ERROR]: Can not create a tracking dir (Permission denied (os error 13))
        Path::new("/home/elhalili/workspace/VaultSync/build/mypr/hello.txt")
        repo.dir = "/home/elhalili/workspace/VaultSync/build/mypr"
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("sandboxed-fixture-paths", selected_ids)

    def test_test_visible_struct_layout_selected_for_private_field_test_compile_error(self):
        context = """
        error[E0616]: field `buffer` of struct `RSet` is private
          --> src/bin/tests.rs:26:20
        assert_eq!(set.buffer[0], 3);
        Candidate Rust shared items:
        pub struct RSet { buffer: Vec<u16>, size: usize }
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("test-visible-struct-layout", selected_ids)

    def test_lifetime_constructor_insight_selected_for_e0621(self):
        context = """
        error[E0621]: explicit lifetime required in the type of `init_state`
        impl<'a> PacketSlabNeighbour<'a>
        pub struct PacketSlabNeighbour<'a> { init_state: LZMAState<'a> }
        help: add explicit lifetime `'a` to the type of `init_state`
        """

        selected = select_insights(context, limit=5)
        selected_ids = [insight.id for insight in selected]

        self.assertIn("lifetime-parameterized-constructor-fields", selected_ids)

    def test_kimi_v61_prioritizes_runtime_open_address_bucket_failure(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v61.py")

        context = """
        Test output: insertion and membership tests panic at table.rs:376
        index out of bounds: the len is 0 but the index is 0.
        Relevant source has let mut idx = hash % self.bucket_size;
        loop { match self.buckets[idx].pi { -1 | -2 => insert } }
        Candidate Rust callables include add(value), remove(value), contains(&value),
        iter(), set_comparator(compare), union, intersection, difference.
        Candidate Rust shared items include
        pub struct BucketValue<T> { pi: i32, elem: T }
        pub struct Table<T> {
            buckets: Vec<BucketValue<T>>,
            max_load_factor: f64,
            min_load_factor: f64,
            bucket_size: usize,
            compare: Option<fn(&T, &T) -> bool>,
        }
        """

        selected_ids = [insight.id for insight in library.select(context, limit=6)]

        self.assertEqual("open-address-double-hash-set-state", selected_ids[0])
        self.assertEqual("open-address-bucket-sentinel-state", selected_ids[1])

    def test_kimi_v62_prioritizes_hamming_sidecar_runtime_failure(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v62.py")

        context = """
        Test output: encode/decode roundtrip panicked with Decoding should succeed.
        The encoded 16 byte block differs only in the final check byte; single bit
        correction should still succeed and double bit corruption should reject.
        Candidate Rust callables include hecc_encode, hecc_decode, hecc_encode_impl,
        hecc_decode_impl, and log2uint8. The source processes fixed 15 byte data
        blocks plus one parity byte/check byte, recomputes syndrome and overall parity,
        and maps non-power-of-two syndrome positions back to data bits.
        """

        selected_ids = [insight.id for insight in library.select(context, limit=6)]

        self.assertEqual("sidecar-ecc-check-byte-contract", selected_ids[0])
        self.assertEqual("block-ecc-syndrome-parity", selected_ids[1])
        self.assertEqual("hamming-position-syndrome-overall-parity", selected_ids[2])

    def test_kimi_v63_prioritizes_tree_roundtrip_runtime_failure(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v63.py")

        context = """
        Test output: Adding 100 points to tree. Adding 10 cluster points to tree.
        Writing tree to file: testfile.dat, then read back through the public reader.
        The assertion failed because the write/read roundtrip did not return expected success.
        Candidate callables include tree.write, tree.read, add points, write_node,
        read_node, write_datapoint, read_datapoint, header parsing, datatype bytes,
        root node, leaf node, and recursive node hierarchy serialization.
        """

        selected_ids = [insight.id for insight in library.select(context, limit=6)]

        self.assertEqual("serialized-tree-roundtrip-layout", selected_ids[0])
        self.assertEqual("tree-add-preserve-existing-root", selected_ids[1])
        self.assertEqual("roundtrip-file-permission-before-layout", selected_ids[2])

    def test_kimi_v64_prioritizes_gapbuffer_runtime_failure(self):
        library = load_insight_library("snapshots/insights/generic_kimi_multi_full_repro_v64.py")

        context = """
        Test output:
        Testing GapBuffer
        Test 4 GapBufferMoveGap, TextBufferInsert and Backspace
        Test 5 Split
        assertion `left == right` failed
          left: "cccaaabbbaaaaaaaaaaaaaaaaaddd"
         right: "cccaaa"
        The split and backspace path exposes stale tail bytes instead of visible text.
        """

        selected_ids = [insight.id for insight in library.select(context, limit=6)]

        self.assertEqual("gap-buffer-allocated-length-invariant", selected_ids[0])
        self.assertEqual("gap-buffer-split-live-tail", selected_ids[1])


if __name__ == "__main__":
    unittest.main()
