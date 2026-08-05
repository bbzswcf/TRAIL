from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from c2rust_translator.artifacts import resolve_project_root, resolve_run_root


DEFAULT_EXCLUDED_PROJECTS = {"fslib", "libfor", "tisp"}
UNIMPLEMENTED_RE = re.compile(r"\b(unimplemented!|todo!)\s*\(")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize full-dataset experiment compile/test pass rates."
    )
    parser.add_argument("runs", nargs="+", help="Run ids under output/runs/ to summarize; legacy output/<run_id> is also supported.")
    parser.add_argument(
        "--baseline",
        default="",
        help="Run id used as the delta baseline. Defaults to the first run.",
    )
    parser.add_argument(
        "--patch",
        action="append",
        default=[],
        metavar="[TARGET_RUN:]PROJECT=PATCH_RUN",
        help=(
            "Use PATCH_RUN for PROJECT when summarizing TARGET_RUN. "
            "If TARGET_RUN is omitted, the first non-baseline run is patched. "
            "Repeat for multiple projects."
        ),
    )
    parser.add_argument(
        "--project",
        action="append",
        default=[],
        help="Restrict metrics to this project. Repeat for multiple projects.",
    )
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="Include fslib/libfor/tisp instead of excluding experiment outliers.",
    )
    parser.add_argument(
        "--all-projects",
        action="store_true",
        help="Use metadata projects instead of the common project set across runs.",
    )
    parser.add_argument(
        "--quality-table",
        action="store_true",
        help="Also print SafeRate/Lint Pass/Idiom Penalty fields from recorded artifacts.",
    )
    return parser


def parse_patches(values: list[str], default_target_run: str) -> dict[tuple[str, str], str]:
    patches: dict[tuple[str, str], str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(
                f"Invalid --patch value {value!r}; expected [TARGET_RUN:]PROJECT=PATCH_RUN"
            )
        target, patch_run_id = value.split("=", 1)
        if ":" in target:
            target_run_id, project = target.split(":", 1)
            target_run_id = target_run_id.strip()
        else:
            target_run_id = default_target_run
            project = target
        project = project.strip()
        patch_run_id = patch_run_id.strip()
        if not target_run_id or not project or not patch_run_id:
            raise SystemExit(
                f"Invalid --patch value {value!r}; expected [TARGET_RUN:]PROJECT=PATCH_RUN"
            )
        patches[(target_run_id, project)] = patch_run_id
    return patches


def metadata_projects(root: Path, include_excluded: bool) -> set[str]:
    projects = {
        path.stem
        for path in (root / "data" / "c_metadata").glob("*.json")
        if not path.stem.endswith("_dep_graph")
    }
    if not include_excluded:
        projects -= DEFAULT_EXCLUDED_PROJECTS
    return projects


def reported_projects(root: Path, run_id: str, include_excluded: bool) -> set[str]:
    projects = {
        path.parent.name
        for path in resolve_run_root(root, run_id).glob("*/translation_report.json")
    }
    if not include_excluded:
        projects -= DEFAULT_EXCLUDED_PROJECTS
    return projects


def has_noninterface_unimplemented(project_root: Path) -> bool:
    src_root = project_root / "src"
    if not src_root.exists():
        return False
    for path in src_root.rglob("*.rs"):
        rel = str(path.relative_to(src_root)).lower()
        if "interface" in rel or "interfaces" in rel:
            continue
        if UNIMPLEMENTED_RE.search(path.read_text(encoding="utf-8", errors="ignore")):
            return True
    return False


def load_report(project_root: Path) -> dict:
    return json.loads((project_root / "translation_report.json").read_text())


UNSAFE_EXTERN_BLOCK_RE = re.compile(
    r'\bunsafe\s+extern\s+(?:"[^"]+"\s*)?\{[^{}]*\}',
    re.DOTALL,
)
UNSAFE_IMPLEMENTATION_RE = re.compile(r"\bunsafe\s*(?:\{|fn\b|impl\b|trait\b)")


def load_translation_history(project_root: Path) -> dict:
    path = project_root / "translation_history.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _contains_unsafe_implementation(code: str) -> bool:
    without_extern_decls = UNSAFE_EXTERN_BLOCK_RE.sub("", code)
    return UNSAFE_IMPLEMENTATION_RE.search(without_extern_decls) is not None


def callable_safety_metrics(project_root: Path) -> tuple[int, int]:
    history = load_translation_history(project_root)
    safe_callables = 0
    total_callables = 0
    for item in history.values():
        if not isinstance(item, dict):
            continue
        final_code = item.get("final_code")
        if not isinstance(final_code, str) or not final_code.strip():
            continue
        total_callables += 1
        safe_callables += int(not _contains_unsafe_implementation(final_code))
    return safe_callables, total_callables


def load_quality_metrics(project_root: Path) -> dict:
    path = project_root / "quality_metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def project_metrics(root: Path, run_id: str, project: str) -> tuple[bool, bool, bool]:
    project_root = resolve_project_root(root, run_id, project)
    report = load_report(project_root)
    stats = report.get("statistics") or {}
    compile_no_unimplemented = (
        stats.get("total_modules") == stats.get("translated_modules")
        and stats.get("failed_modules") == 0
        and not has_noninterface_unimplemented(project_root)
    )
    test_pass = bool(report.get("success"))
    compile_pass = test_pass or (compile_no_unimplemented and not test_pass)
    return compile_pass, test_pass, compile_no_unimplemented


def project_quality_metrics(root: Path, run_id: str, project: str) -> dict:
    project_root = resolve_project_root(root, run_id, project)
    report = load_report(project_root)
    compile_pass, test_pass, compile_no_unimplemented = project_metrics(root, run_id, project)
    safe_callables, total_callables = callable_safety_metrics(project_root)
    quality = load_quality_metrics(project_root)
    config = report.get("config") or {}

    lint_pass = quality.get("lint_pass", quality.get("clippy_pass"))
    idiom_penalty = quality.get("idiom_penalty")
    has_idiom_penalty = isinstance(idiom_penalty, (int, float)) and not isinstance(idiom_penalty, bool)
    return {
        "compile_pass": compile_pass,
        "test_pass": test_pass,
        "compile_no_unimplemented": compile_no_unimplemented,
        "safe_callables": safe_callables,
        "total_callables": total_callables,
        "lint_pass": lint_pass if isinstance(lint_pass, bool) else None,
        "idiom_penalty": idiom_penalty if has_idiom_penalty else None,
        "profile": config.get("experiment_ablation_label") or config.get("experiment_profile") or "unknown",
        "model": config.get("model_display_name") or "unknown",
    }


def pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.0%"
    return f"{numerator / denominator * 100:.1f}%"


def quality_pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return pct(numerator, denominator)


def _single_label(labels: set[str]) -> str:
    cleaned = sorted(label for label in labels if label and label != "unknown")
    if not cleaned:
        return "unknown"
    if len(cleaned) == 1:
        return cleaned[0]
    return "|".join(cleaned)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    args = build_parser().parse_args()
    baseline = args.baseline or args.runs[0]

    if baseline not in args.runs:
        args.runs.insert(0, baseline)
    default_patch_target = next((run_id for run_id in args.runs if run_id != baseline), baseline)
    patches = parse_patches(args.patch, default_patch_target)

    if args.project:
        projects = set(args.project)
    elif args.all_projects:
        projects = metadata_projects(root, args.include_excluded)
    else:
        projects = metadata_projects(root, args.include_excluded)
        for run_id in args.runs:
            projects &= reported_projects(root, run_id, args.include_excluded)

    rows: dict[str, tuple[int, int, int, int, list[str]]] = {}
    quality_rows: dict[str, dict[str, object]] = {}
    for run_id in args.runs:
        compile_count = 0
        test_count = 0
        compile_no_unimplemented_count = 0
        missing: list[str] = []
        quality_summary = {
            "safe_callables": 0,
            "total_callables": 0,
            "lint_pass": 0,
            "lint_total": 0,
            "idiom_penalties": [],
            "profiles": set(),
            "models": set(),
        }
        for project in sorted(projects):
            actual_run = patches.get((run_id, project), run_id)
            report_path = resolve_project_root(root, actual_run, project) / "translation_report.json"
            if not report_path.exists():
                missing.append(project)
                continue
            compile_pass, test_pass, compile_no_unimplemented = project_metrics(
                root, actual_run, project
            )
            compile_count += int(compile_pass)
            test_count += int(test_pass)
            compile_no_unimplemented_count += int(compile_no_unimplemented)
            if args.quality_table:
                quality_metrics = project_quality_metrics(root, actual_run, project)
                quality_summary["safe_callables"] += quality_metrics["safe_callables"]
                quality_summary["total_callables"] += quality_metrics["total_callables"]
                if quality_metrics["lint_pass"] is not None:
                    quality_summary["lint_total"] += 1
                    quality_summary["lint_pass"] += int(quality_metrics["lint_pass"])
                if quality_metrics["idiom_penalty"] is not None:
                    quality_summary["idiom_penalties"].append(quality_metrics["idiom_penalty"])
                quality_summary["profiles"].add(quality_metrics["profile"])
                quality_summary["models"].add(quality_metrics["model"])
        rows[run_id] = (
            compile_count,
            test_count,
            compile_no_unimplemented_count,
            len(projects),
            missing,
        )
        quality_rows[run_id] = quality_summary

    base_compile, base_test, _, base_total, _ = rows[baseline]
    print(f"projects={len(projects)}")
    for run_id in args.runs:
        compile_count, test_count, compile_no_unimplemented_count, total, missing = rows[run_id]
        compile_delta = compile_count - base_compile
        test_delta = test_count - base_test
        print(
            f"{run_id}: "
            f"compile={compile_count}/{total} ({pct(compile_count, total)}) "
            f"test={test_count}/{total} ({pct(test_count, total)}) "
            f"compile_no_unimplemented={compile_no_unimplemented_count}/{total} "
            f"delta_compile={compile_delta} ({pct(compile_delta, base_total)}) "
            f"delta_test={test_delta} ({pct(test_delta, base_total)})"
        )
        if args.quality_table:
            quality_summary = quality_rows[run_id]
            safe_callables = int(quality_summary["safe_callables"])
            total_callables = int(quality_summary["total_callables"])
            lint_pass = int(quality_summary["lint_pass"])
            lint_total = int(quality_summary["lint_total"])
            idiom_penalties = quality_summary["idiom_penalties"]
            idiom_penalty = (
                f"{sum(idiom_penalties) / len(idiom_penalties):.2f}" if idiom_penalties else "n/a"
            )
            lint_text = f"{lint_pass}/{lint_total}" if lint_total else "n/a"
            print(
                "  quality_table "
                f"profile={_single_label(quality_summary['profiles'])} "
                f"model={_single_label(quality_summary['models'])} "
                f"safe_callables={safe_callables}/{total_callables} "
                f"SafeRate={quality_pct(safe_callables, total_callables)} "
                f"lint_pass={lint_text} "
                f"idiom_penalty={idiom_penalty}"
            )
        if missing:
            print(f"  missing={', '.join(missing)}")


if __name__ == "__main__":
    main()
