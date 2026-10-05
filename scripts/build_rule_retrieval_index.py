#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable, Sequence

from c2rust_translator.adversarial_rules import (
    RuleBank,
    RetrievalEngine,
    _bank_fingerprint,
    _embedding_index_metadata_path,
)
from c2rust_translator.rules import DEFAULT_RULES, load_rule_library

EMBEDDING_DEPENDENCY_MODULES = ("faiss", "sentence_transformers")


def embedding_dependency_status(
    *,
    find_spec: Callable[[str], object | None] | None = None,
) -> dict:
    finder = find_spec or importlib.util.find_spec
    modules = {
        name: "available" if finder(name) is not None else "missing"
        for name in EMBEDDING_DEPENDENCY_MODULES
    }
    missing = [name for name, status in modules.items() if status == "missing"]
    install_hint = ""
    if missing:
        install_hint = (
            "Install FAISS/BGE retrieval dependencies in the active uv environment, "
            "for example: uv pip install faiss-cpu sentence-transformers"
        )
    return {
        "available": not missing,
        "modules": modules,
        "missing": missing,
        "install_hint": install_hint,
    }


def build_index_manifest(
    bank: RuleBank,
    index_cache_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    embedding_model=None,
    faiss_module=None,
    find_spec: Callable[[str], object | None] | None = None,
) -> dict:
    index_path = Path(index_cache_path)
    engine = RetrievalEngine(
        bank,
        prefer_embeddings=True,
        index_cache_path=index_path,
        embedding_model=embedding_model,
        faiss_module=faiss_module,
        find_spec=find_spec,
    )
    metadata_path = _embedding_index_metadata_path(index_path)
    manifest = {
        "backend": engine.status.backend,
        "reason": engine.status.reason,
        "cache_status": engine.cache_status,
        "embedding_dependencies": embedding_dependency_status(find_spec=find_spec),
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
        "index_exists": index_path.exists(),
        "metadata_exists": metadata_path.exists(),
        "record_count": len(bank.records),
        "record_ids": [record.id for record in bank.records],
        "bank_fingerprint": _bank_fingerprint(bank),
    }
    if manifest_path is not None:
        output_path = Path(manifest_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_bank(
    path: str | Path | None,
    *,
    rules_path: str | Path | None = None,
) -> RuleBank:
    if path:
        return RuleBank.load(path)
    if rules_path:
        library = load_rule_library(rules_path)
        return RuleBank.from_legacy(library.rules)
    return RuleBank.from_legacy(DEFAULT_RULES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a FAISS/BGE retrieval index cache for a JSON bank or local Python rule snapshot.",
    )
    parser.add_argument(
        "--bank",
        default="",
        help="JSON rule bank to index. Defaults to the built-in rule records.",
    )
    parser.add_argument(
        "--rules-path",
        default="",
        help=(
            "Local Python rule snapshot exporting DEFAULT_RULES to index. "
            "Use this for local snapshot-based indexes; mutually exclusive with --bank."
        ),
    )
    parser.add_argument(
        "--output-index",
        required=True,
        help="Path to write the FAISS index cache, for example snapshots/rules/retrieval.faiss.",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Optional manifest JSON path. Defaults to <output-index>.manifest.json.",
    )
    parser.add_argument(
        "--require-embedding",
        action="store_true",
        help="Exit nonzero if FAISS/BGE dependencies are unavailable and keyword fallback is used.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.bank and args.rules_path:
        parser.error("--bank and --rules-path are mutually exclusive")
    bank = load_bank(args.bank or None, rules_path=args.rules_path or None)
    index_path = Path(args.output_index)
    manifest_path = Path(args.manifest) if args.manifest else Path(str(index_path) + ".manifest.json")
    manifest = build_index_manifest(bank, index_path, manifest_path=manifest_path)
    print(json.dumps(manifest, indent=2))
    if args.require_embedding and manifest["backend"] != "faiss-bge":
        print(manifest["reason"], file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
