from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import contextlib
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import TextIO

from c2rust_translator.translation_agent import DEFAULT_RESPONSES_API_URL, DEFAULT_RESPONSES_MODEL
from c2rust_translator.artifacts import default_run_root, resolve_full_dataset_log_path, runs_root
from c2rust_translator.experiment_profiles import profile_choices, resolve_experiment_profile
from c2rust_translator.translation_runner import (
    DEFAULT_CHAT_BASE_URL,
    DEFAULT_ADVERSARIAL_INSIGHT_OUTPUT_BANK,
    DEFAULT_ADVERSARIAL_INSIGHT_REPORT,
    _env_float,
    _env_int,
    _optional_env_int,
    discover_available_projects,
    load_env_file,
    online_adversarial_insights_enabled,
    parallel_online_adversarial_updates_supported,
    project_config,
    run_adversarial_insight_update,
    resolve_run_id,
)
from c2rust_translator.translator import Translator


_C_CODE_EXTENSIONS = {".c", ".h"}
_C_CODE_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "build",
    "bld",
    "target",
    "CMakeFiles",
    "__pycache__",
}
EXPERIMENT_EXCLUDED_PROJECTS = {"fslib", "libfor", "tisp"}


class Tee:
    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full translation dataset with detailed logging.")
    parser.add_argument(
        "--log-path",
        default="",
        help=(
            "Path to the combined stdout/stderr log file. Defaults to logs/runs/<run_id>.log; "
            "a bare relative filename or old-style logs/<name>.log is also written under logs/runs/."
        ),
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Remove existing output/runs/* entries before running; legacy output/<run_id>/ directories are preserved.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help=(
            "Subdirectory name under output/runs/. Defaults to a fresh timestamp (YYYYmmdd-HHMMSS); "
            "all projects in this batch share the same run id, so artifacts land in "
            "output/runs/<run_id>/<project>/. Pass an explicit value (e.g. when using --resume-from-log) "
            "to keep writing into the prior run's directory."
        ),
    )
    parser.add_argument(
        "--append-log",
        action="store_true",
        help="Append to the existing log file instead of overwriting it.",
    )
    parser.add_argument(
        "--resume-from-log",
        action="store_true",
        help="Skip projects already marked with PROJECT_DONE in the log file.",
    )
    parser.add_argument(
        "--rerun-project",
        action="append",
        default=[],
        help="Project to rerun even if it already appears as completed in the log. Repeat this option to add more.",
    )
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="Include fslib/libfor/tisp in the full-dataset queue.",
    )
    parser.add_argument(
        "--max-fix-attempts",
        type=int,
        default=5,
        help="Max compile-fix attempts per callable.",
    )
    parser.add_argument(
        "--max-test-fix-attempts",
        type=int,
        default=3,
        help="Max project-level test fix attempts after cargo test failures.",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Do not stop immediately when a module translation fails inside a project.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL", DEFAULT_RESPONSES_MODEL),
        help="LLM model name (env: LLM_MODEL).",
    )
    parser.add_argument(
        "--llm-protocol",
        choices=["openai-responses", "openai-chat", "google-genai"],
        default=os.getenv("LLM_PROTOCOL", "openai-responses"),
        help="LLM protocol adapter to use (env: LLM_PROTOCOL).",
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("LLM_API_URL", DEFAULT_RESPONSES_API_URL),
        help="Responses or Google GenAI API URL when --llm-protocol uses api-url (env: LLM_API_URL).",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BASE_URL", DEFAULT_CHAT_BASE_URL),
        help="Chat Completions API base URL when --llm-protocol=openai-chat (env: LLM_BASE_URL).",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY", ""),
        help="LLM API key. No default; export OPENAI_API_KEY/LLM_API_KEY or pass --api-key explicitly.",
    )
    parser.add_argument(
        "--session-id",
        default=os.getenv("LLM_SESSION_ID", ""),
        help="Session id used in the Responses extra header (env: LLM_SESSION_ID).",
    )
    parser.add_argument(
        "--logid",
        default=os.getenv("LLM_LOGID") or os.getenv("X_TT_LOGID", ""),
        help="X-TT-logid header value for Responses requests (env: LLM_LOGID or X_TT_LOGID).",
    )
    parser.add_argument(
        "--min-request-interval",
        type=float,
        default=_env_float("LLM_MIN_REQUEST_INTERVAL", 0.0),
        help="Minimum seconds between LLM request starts within each project worker (env: LLM_MIN_REQUEST_INTERVAL).",
    )
    parser.add_argument(
        "--rate-limit-retry-delay",
        type=float,
        default=_env_float("LLM_RATE_LIMIT_RETRY_DELAY", 60.0),
        help="Minimum retry sleep seconds after HTTP 429 responses (env: LLM_RATE_LIMIT_RETRY_DELAY).",
    )
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=_env_int("LLM_TIMEOUT", 300),
        help="Per-request LLM timeout in seconds (env: LLM_TIMEOUT).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=_env_int("TRANSLATION_JOBS", 1),
        help="Number of projects to translate concurrently (env: TRANSLATION_JOBS).",
    )
    parser.add_argument(
        "--alignment-mode",
        choices=["llm", "rule"],
        default=os.getenv("ALIGNMENT_MODE", "llm"),
        help="C/Rust alignment mode before translation (env: ALIGNMENT_MODE).",
    )
    parser.add_argument(
        "--alignment-confidence-threshold",
        type=float,
        default=_env_float("ALIGNMENT_CONFIDENCE_THRESHOLD", 0.65),
        help="Minimum LLM alignment confidence accepted as a safe binding.",
    )
    parser.add_argument(
        "--insights-mode",
        choices=["default", "off"],
        default=os.getenv("INSIGHTS_MODE", "default"),
        help="Use structured translation insights in LLM prompts, or disable them for ablation (env: INSIGHTS_MODE).",
    )
    parser.add_argument(
        "--insight-limit",
        type=int,
        default=_env_int("INSIGHT_LIMIT", 5),
        help="Maximum selected translation insights to include per prompt (env: INSIGHT_LIMIT).",
    )
    parser.add_argument(
        "--insights-path",
        default=os.getenv("INSIGHTS_PATH", ""),
        help="Optional Python file exporting DEFAULT_INSIGHTS and select_insights for this run (env: INSIGHTS_PATH).",
    )
    parser.add_argument(
        "--insights-json-bank",
        default=os.getenv("INSIGHTS_JSON_BANK", ""),
        help=(
            "Optional JSON insight bank for prompt-time insight retrieval "
            "(env: INSIGHTS_JSON_BANK). When set, it overrides profile-selected final banks "
            "and Python insight snapshots for prompt injection."
        ),
    )
    parser.add_argument(
        "--insight-retrieval-backend",
        choices=["auto", "keyword", "embedding"],
        default=os.getenv("INSIGHT_RETRIEVAL_BACKEND", "auto"),
        help=(
            "Retrieval backend preference for --insights-json-bank. "
            "'auto' and 'embedding' try FAISS/BGE with keyword fallback; 'keyword' disables embeddings "
            "(env: INSIGHT_RETRIEVAL_BACKEND)."
        ),
    )
    parser.add_argument(
        "--insight-retrieval-index-cache",
        default=os.getenv("INSIGHT_RETRIEVAL_INDEX_CACHE", ""),
        help=(
            "Optional FAISS index cache path for JSON bank retrieval. "
            "Defaults to <run_root>/insight_retrieval.faiss when JSON bank retrieval uses auto/embedding "
            "(env: INSIGHT_RETRIEVAL_INDEX_CACHE)."
        ),
    )
    parser.add_argument(
        "--insight-interaction-scorer",
        choices=["heuristic", "live"],
        default=os.getenv("INSIGHT_INTERACTION_SCORER", "heuristic"),
        help=(
            "Support/conflict scorer for multi-insight selection. "
            "'heuristic' uses local scores; 'live' asks the configured LLM for structured JSON scores "
            "(env: INSIGHT_INTERACTION_SCORER)."
        ),
    )
    parser.add_argument(
        "--experiment-profile",
        choices=profile_choices(),
        default=os.getenv("EXPERIMENT_PROFILE", ""),
        help=(
            "Named experiment profile. When set, applies the model-matched final JSON insight bank and "
            "structured ablation metadata (env: EXPERIMENT_PROFILE)."
        ),
    )
    parser.add_argument(
        "--adversarial-rounds",
        type=int,
        default=None,
        help=(
            "Adversarial refinement round count K to record for this run. "
            "Defaults to the selected experiment profile's value."
        ),
    )
    parser.add_argument(
        "--adversarial-insight-bank",
        default=os.getenv("ADVERSARIAL_INSIGHT_BANK", ""),
        help=(
            "Optional JSON insight bank for adversarial insight learning. When insights/profile "
            "mining are enabled, this bank is also the prompt retrieval bank and default mutable "
            "online update bank (env: ADVERSARIAL_INSIGHT_BANK)."
        ),
    )
    parser.add_argument(
        "--adversarial-insight-output-bank",
        default=os.getenv("ADVERSARIAL_INSIGHT_OUTPUT_BANK", ""),
        help=(
            "Output JSON insight bank for post-run adversarial insight learning. "
            "Defaults to the active prompt bank for online same-bank updates, otherwise "
            f"<run_root>/{DEFAULT_ADVERSARIAL_INSIGHT_OUTPUT_BANK}."
        ),
    )
    parser.add_argument(
        "--adversarial-insight-report",
        default=os.getenv("ADVERSARIAL_INSIGHT_REPORT", ""),
        help=(
            "Output JSON report for post-run adversarial insight learning. "
            f"Defaults to <run_root>/{DEFAULT_ADVERSARIAL_INSIGHT_REPORT}."
        ),
    )
    parser.add_argument(
        "--online-adversarial-insights",
        action="store_true",
        default=os.getenv("ONLINE_ADVERSARIAL_INSIGHTS", "").lower() in {"1", "true", "yes"},
        help=(
            "Run adversarial insight learning after each sequential project. "
            "When --adversarial-insight-bank is omitted, this uses --insights-json-bank as the mutable bank "
            "(env: ONLINE_ADVERSARIAL_INSIGHTS)."
        ),
    )
    parser.add_argument(
        "--adversarial-admit-trace-candidates",
        action="store_true",
        default=os.getenv("ADVERSARIAL_ADMIT_TRACE_CANDIDATES", "").lower() in {"1", "true", "yes"},
        help="Validate trace-mined candidate insights during adversarial bank updates.",
    )
    parser.add_argument(
        "--adversarial-run-synthetic-challenges",
        action="store_true",
        default=os.getenv("ADVERSARIAL_RUN_SYNTHETIC_CHALLENGES", "").lower() in {"1", "true", "yes"},
        help="Execute deterministic single-insight synthetic challenges during adversarial bank updates.",
    )
    parser.add_argument(
        "--adversarial-run-compositional-challenges",
        action="store_true",
        default=os.getenv("ADVERSARIAL_RUN_COMPOSITIONAL_CHALLENGES", "").lower() in {"1", "true", "yes"},
        help="Execute pairwise compositional challenges during adversarial bank updates.",
    )
    parser.add_argument(
        "--adversarial-admit-composition-failure-candidates",
        action="store_true",
        default=os.getenv("ADVERSARIAL_ADMIT_COMPOSITION_FAILURE_CANDIDATES", "").lower()
        in {"1", "true", "yes"},
        help="Turn compiling but test-failing compositional challenge evidence into validated candidate insights.",
    )
    parser.add_argument(
        "--adversarial-live-structured-agents",
        action="store_true",
        default=os.getenv("ADVERSARIAL_LIVE_STRUCTURED_AGENTS", "").lower() in {"1", "true", "yes"},
        help="Use the configured LLM provider for adversarial Challenger/Refiner structured agents.",
    )
    parser.add_argument(
        "--adversarial-live-trace-synthesis",
        action="store_true",
        default=os.getenv("ADVERSARIAL_LIVE_TRACE_SYNTHESIS", "").lower() in {"1", "true", "yes"},
        help="Use the configured LLM provider to synthesize trace-mined candidate insights.",
    )
    parser.add_argument(
        "--adversarial-live-trace-contradiction-review",
        action="store_true",
        default=os.getenv("ADVERSARIAL_LIVE_TRACE_CONTRADICTION_REVIEW", "").lower()
        in {"1", "true", "yes"},
        help="Use the configured LLM provider to reject trace candidates that contradict the active bank.",
    )
    parser.add_argument(
        "--adversarial-live-composition-relation-review",
        action="store_true",
        default=os.getenv("ADVERSARIAL_LIVE_COMPOSITION_RELATION_REVIEW", "").lower()
        in {"1", "true", "yes"},
        help="Use the configured LLM provider to infer supports/requires/precedes composition relations.",
    )
    parser.add_argument(
        "--adversarial-live-semantic-oracle-validation",
        action="store_true",
        default=os.getenv("ADVERSARIAL_LIVE_SEMANTIC_ORACLE_VALIDATION", "").lower()
        in {"1", "true", "yes"},
        help="Use the configured LLM provider to validate challenge oracle/harness semantics.",
    )
    parser.add_argument(
        "--adversarial-live-activation-validation",
        action="store_true",
        default=os.getenv("ADVERSARIAL_LIVE_ACTIVATION_VALIDATION", "").lower() in {"1", "true", "yes"},
        help="Use the configured LLM provider to validate whether challenges activate their target insights.",
    )
    parser.add_argument(
        "--adversarial-live-translator-replay",
        action="store_true",
        default=os.getenv("ADVERSARIAL_LIVE_TRANSLATOR_REPLAY", "").lower() in {"1", "true", "yes"},
        help="Use the configured LLM provider through Translator.translate_adversarial_challenge for fresh trace replay.",
    )
    parser.add_argument(
        "--adversarial-pre-translation-challenges",
        action="store_true",
        default=os.getenv("ADVERSARIAL_PRE_TRANSLATION_CHALLENGES", "").lower()
        in {"1", "true", "yes"},
        help=(
            "Attach pre-translation adversarial challenge metadata to selected prompt insights without "
            "requiring callable-level online adversarial updates."
        ),
    )
    parser.add_argument(
        "--adversarial-pre-translation-refinement",
        action="store_true",
        default=os.getenv("ADVERSARIAL_PRE_TRANSLATION_REFINEMENT", "").lower() in {"1", "true", "yes"},
        help=(
            "Use evidence-gated challenge replay to refine selected insights locally before building "
            "translation/fix prompts. By default this only updates the prompt-local insight; combine with "
            "--adversarial-pre-translation-refinement-writeback to persist accepted updates."
        ),
    )
    parser.add_argument(
        "--adversarial-pre-translation-refinement-writeback",
        action="store_true",
        default=os.getenv("ADVERSARIAL_PRE_TRANSLATION_REFINEMENT_WRITEBACK", "").lower()
        in {"1", "true", "yes"},
        help=(
            "When pre-translation refinement accepts an update, persist it to the active JSON prompt bank "
            "and reload prompt retrieval before later selections."
        ),
    )
    parser.add_argument(
        "--adversarial-pre-translation-refinement-rounds",
        type=int,
        default=_optional_env_int("ADVERSARIAL_PRE_TRANSLATION_REFINEMENT_ROUNDS"),
        help=(
            "Maximum accepted prompt-local pre-translation refinement rounds per selected insight. "
            "Defaults to 1 for custom runs and to the profile K for profiles that enable "
            "pre-translation refinement "
            "(env: ADVERSARIAL_PRE_TRANSLATION_REFINEMENT_ROUNDS)."
        ),
    )
    return parser


def experiment_log_metadata(args: argparse.Namespace) -> dict[str, object]:
    raw_profile = getattr(args, "experiment_profile", "") or ""
    if not raw_profile:
        return {
            "experiment_profile": "custom",
            "experiment_ablation_label": "custom",
            "insight_stage": "custom" if getattr(args, "insights_mode", "default") != "off" else "off",
            "adversarial_rounds": max(getattr(args, "adversarial_rounds", 0) or 0, 0),
        }

    profile = resolve_experiment_profile(raw_profile)
    raw_rounds = getattr(args, "adversarial_rounds", None)
    return {
        "experiment_profile": profile.name,
        "experiment_ablation_label": profile.display_label,
        "insight_stage": profile.insight_stage,
        "adversarial_rounds": (
            profile.default_adversarial_rounds if raw_rounds is None else max(raw_rounds, 0)
        ),
    }


def clear_runs_dir(runs_dir: Path) -> None:
    runs_dir.mkdir(parents=True, exist_ok=True)
    for child in runs_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def completed_projects_from_log(log_path: Path) -> list[str]:
    if not log_path.exists():
        return []
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return re.findall(r"^PROJECT_DONE\s+(\S+)\s+elapsed=", text, re.MULTILINE)


def _count_lines(path: Path) -> int:
    data = path.read_bytes()
    if not data:
        return 0
    return data.count(b"\n") + (0 if data.endswith(b"\n") else 1)


def c_project_code_lines(root: Path, project: str) -> int:
    project_dir = root / "data" / "project" / "CBench" / project
    if not project_dir.exists():
        return 0

    total = 0
    for dirpath, dirnames, filenames in os.walk(project_dir):
        dirnames[:] = sorted(
            dirname
            for dirname in dirnames
            if not dirname.startswith(".") and dirname not in _C_CODE_SKIP_DIRS
        )
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            if path.suffix.lower() not in _C_CODE_EXTENSIONS:
                continue
            total += _count_lines(path)
    return total


def order_projects_by_c_code_size(root: Path, projects: list[str]) -> list[str]:
    return sorted(projects, key=lambda project: (-c_project_code_lines(root, project), project.casefold()))


def selected_projects(root: Path, args: argparse.Namespace, log_path: Path) -> list[str]:
    include_excluded = bool(getattr(args, "include_excluded", False))
    projects = [
        project
        for project in discover_available_projects(root)
        if include_excluded or project not in EXPERIMENT_EXCLUDED_PROJECTS
    ]
    if not projects:
        return []

    completed = set(completed_projects_from_log(log_path)) if args.resume_from_log else set()
    rerun = [project for project in args.rerun_project if project]

    chosen: list[str] = []
    seen: set[str] = set()

    for project in rerun:
        if project not in projects or project in seen:
            continue
        seen.add(project)
        chosen.append(project)

    for project in order_projects_by_c_code_size(root, projects):
        if project in seen:
            continue
        if project in completed:
            continue
        seen.add(project)
        chosen.append(project)

    return chosen


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    load_env_file(root / ".env")
    args = build_parser().parse_args()
    run_id = resolve_run_id(args.run_id)
    log_path = resolve_full_dataset_log_path(root, run_id, args.log_path).resolve()
    runs_dir = runs_root(root)
    run_root = default_run_root(root, run_id)

    if args.clear_output:
        clear_runs_dir(runs_dir)

    projects = selected_projects(root, args, log_path)
    if not projects:
        raise SystemExit("No projects selected to run")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    log_mode = "a" if args.append_log else "w"

    with log_path.open(log_mode, encoding="utf-8") as log_file:
        tee_stdout = Tee(sys.stdout, log_file)
        tee_stderr = Tee(sys.stderr, log_file)
        with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
            if args.append_log and log_path.stat().st_size > 0:
                print("", flush=True)
                print("=" * 80, flush=True)
                print(f"RESUME_RUN started_at={time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
                print("=" * 80, flush=True)
            print(f"TOTAL_PROJECTS {len(projects)}", flush=True)
            print(f"LOG_PATH {log_path}", flush=True)
            print(f"RUN_ID {run_id}", flush=True)
            print(f"RUN_ROOT {run_root}", flush=True)
            print("PROJECT_MODE stop_on_first_failure=True", flush=True)
            print(f"PROJECT_JOBS {max(args.jobs, 1)}", flush=True)
            print("PROJECT_ORDER c_code_lines_desc", flush=True)
            if args.include_excluded:
                print("EXCLUDED_PROJECTS none", flush=True)
                print(
                    f"INCLUDED_EXPERIMENT_OUTLIERS {', '.join(sorted(EXPERIMENT_EXCLUDED_PROJECTS))}",
                    flush=True,
                )
            else:
                print(f"EXCLUDED_PROJECTS {', '.join(sorted(EXPERIMENT_EXCLUDED_PROJECTS))}", flush=True)
            profile_metadata = experiment_log_metadata(args)
            print(f"EXPERIMENT_PROFILE {profile_metadata['experiment_profile']}", flush=True)
            print(f"EXPERIMENT_ABLATION_LABEL {profile_metadata['experiment_ablation_label']}", flush=True)
            print(f"ADVERSARIAL_ROUNDS {profile_metadata['adversarial_rounds']}", flush=True)
            print(f"INSIGHT_STAGE {profile_metadata['insight_stage']}", flush=True)
            if args.resume_from_log:
                print("RESUME_FROM_LOG True", flush=True)
            if args.rerun_project:
                print(f"RERUN_PROJECTS {', '.join(args.rerun_project)}", flush=True)

            def run_one(index: int, project_name: str) -> tuple[str, float]:
                print(f"RUN_PROJECT {index}/{len(projects)} {project_name}", flush=True)
                project_started = time.time()
                try:
                    config = project_config(root, run_root, project_name, args)
                    config["stop_on_first_failure"] = True
                    Translator(config).run()
                except Exception as exc:
                    print(f"PROJECT_CRASH {project_name}: {exc}", flush=True)
                elapsed = time.time() - project_started
                print(f"PROJECT_DONE {project_name} elapsed={elapsed:.2f}", flush=True)
                return project_name, elapsed

            jobs = max(args.jobs, 1)
            if jobs == 1 or len(projects) <= 1:
                for index, project_name in enumerate(projects, 1):
                    run_one(index, project_name)
                    if online_adversarial_insights_enabled(args):
                        run_adversarial_insight_update(root, run_root, args)
            else:
                if online_adversarial_insights_enabled(args) and not parallel_online_adversarial_updates_supported(args):
                    raise SystemExit(
                        "--online-adversarial-insights with --jobs > 1 requires a "
                        "project-local active bank, --insights-json-bank, or "
                        "--adversarial-insight-bank"
                    )
                with ThreadPoolExecutor(max_workers=min(jobs, len(projects))) as executor:
                    futures = [
                        executor.submit(run_one, index, project_name)
                        for index, project_name in enumerate(projects, 1)
                    ]
                    for future in as_completed(futures):
                        future.result()
            if not online_adversarial_insights_enabled(args):
                run_adversarial_insight_update(root, run_root, args)
            print(f"ALL_DONE elapsed={time.time() - started:.2f}", flush=True)


if __name__ == "__main__":
    main()
