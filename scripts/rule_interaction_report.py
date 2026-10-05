from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Iterable

from c2rust_translator.artifacts import resolve_run_root
from c2rust_translator.rules import DEFAULT_RULES, empirical_weak_pair_penalty, score_rule_interaction


DEFAULT_EXCLUDED_PROJECTS = {"fslib", "libfor", "tisp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find weak multi-rule combinations from historical translation artifacts."
    )
    parser.add_argument("runs", nargs="+", help="Run ids under output/runs/ to analyze; legacy output/<run_id> is also supported.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum pair rows to print.")
    parser.add_argument(
        "--min-failed-projects",
        type=int,
        default=1,
        help="Only print pairs seen in at least this many failed projects.",
    )
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="Include fslib/libfor/tisp in the interaction report.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    return parser


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return default


def project_success(project_dir: Path) -> bool:
    report = load_json(project_dir / "translation_report.json", {})
    return bool(report.get("success"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def rule_ids(record: dict[str, Any]) -> list[str]:
    raw_items = record.get("rules") or record.get("selected_rules") or []
    ids: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        rule_id = ""
        if isinstance(item, dict):
            rule_id = str(item.get("id", ""))
        elif isinstance(item, str):
            rule_id = item
        if rule_id and rule_id not in seen:
            seen.add(rule_id)
            ids.append(rule_id)
    return ids


def iter_project_records(project_dir: Path) -> Iterable[dict[str, Any]]:
    yield from iter_jsonl(project_dir / "llm_calls.jsonl")

    test_history = load_json(project_dir / "test_fix_history.json", [])
    if isinstance(test_history, list):
        for record in test_history:
            if isinstance(record, dict):
                record = dict(record)
                record.setdefault("phase", "test_fix")
                yield record

    translation_history = load_json(project_dir / "translation_history.json", [])
    if isinstance(translation_history, list):
        for record in translation_history:
            if isinstance(record, dict):
                record = dict(record)
                record.setdefault("phase", "translation_history")
                yield record


def analyze_run(root: Path, run_id: str, include_excluded: bool) -> dict[str, Any]:
    run_root = resolve_run_root(root, run_id)
    id_to_rule = {rule.id: rule for rule in DEFAULT_RULES}
    pair_stats: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "failed_projects": set(),
            "passed_projects": set(),
            "failed_records": 0,
            "passed_records": 0,
            "phases": Counter(),
        }
    )
    projects = 0
    failed_projects = 0

    for project_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        project = project_dir.name
        if not include_excluded and project in DEFAULT_EXCLUDED_PROJECTS:
            continue
        if not (project_dir / "translation_report.json").exists():
            continue
        projects += 1
        success = project_success(project_dir)
        failed_projects += int(not success)

        for record in iter_project_records(project_dir):
            ids = [rule_id for rule_id in rule_ids(record) if rule_id in id_to_rule]
            if len(ids) < 2:
                continue
            phase = str(record.get("phase") or "unknown")
            for pair in combinations(sorted(ids), 2):
                stats = pair_stats[pair]
                stats["phases"][phase] += 1
                if success:
                    stats["passed_projects"].add(project)
                    stats["passed_records"] += 1
                else:
                    stats["failed_projects"].add(project)
                    stats["failed_records"] += 1

    rows: list[dict[str, Any]] = []
    for pair, stats in pair_stats.items():
        first, second = pair
        interaction = score_rule_interaction(id_to_rule[first], id_to_rule[second])
        empirical_penalty = empirical_weak_pair_penalty(first, second)
        failed_project_count = len(stats["failed_projects"])
        passed_project_count = len(stats["passed_projects"])
        weak_score = (
            failed_project_count * 10
            + stats["failed_records"]
            - passed_project_count * 3
            + interaction.conflict * 5
            - min(interaction.support, 10)
        )
        rows.append(
            {
                "first_id": first,
                "second_id": second,
                "weak_score": weak_score,
                "failed_projects": failed_project_count,
                "passed_projects": passed_project_count,
                "failed_records": stats["failed_records"],
                "passed_records": stats["passed_records"],
                "phases": dict(stats["phases"].most_common()),
                "empirical_penalty": empirical_penalty,
                "interaction": interaction.to_dict(),
            }
        )

    rows.sort(
        key=lambda row: (
            row["weak_score"],
            row["failed_projects"],
            row["failed_records"],
            -row["passed_projects"],
            row["first_id"],
            row["second_id"],
        ),
        reverse=True,
    )
    return {
        "run_id": run_id,
        "projects": projects,
        "failed_projects": failed_projects,
        "pairs": rows,
    }


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    args = build_parser().parse_args()
    reports = [analyze_run(root, run_id, args.include_excluded) for run_id in args.runs]

    for report in reports:
        report["pairs"] = [
            row
            for row in report["pairs"]
            if row["failed_projects"] >= args.min_failed_projects
        ][: max(args.limit, 0)]

    if args.json:
        print(json.dumps(reports, indent=2, sort_keys=True))
        return

    for report in reports:
        print(
            f"RUN {report['run_id']} projects={report['projects']} "
            f"failed_projects={report['failed_projects']}"
        )
        if not report["pairs"]:
            print("  no multi-rule weak pairs matched the filters")
            continue
        for row in report["pairs"]:
            interaction = row["interaction"]
            phases = ", ".join(f"{phase}:{count}" for phase, count in row["phases"].items())
            print(
                "  "
                f"{row['first_id']} + {row['second_id']} "
                f"weak_score={row['weak_score']} "
                f"failed_projects={row['failed_projects']} "
                f"failed_records={row['failed_records']} "
                f"passed_projects={row['passed_projects']} "
                f"relationship={interaction['relationship']} "
                f"support={interaction['support']} "
                f"conflict={interaction['conflict']} "
                f"empirical_penalty={row['empirical_penalty']} "
                f"phases={phases}"
            )


if __name__ == "__main__":
    main()
