"""
翻译任务运行入口（批量）。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import difflib
import os
import threading
import time
from pathlib import Path
from typing import Iterable, Optional, Type

try:
    from .adversarial_insights import (
        InsightBank,
        LLMActivationValidator,
        LLMCompositionRelationReviewer,
        LLMSemanticOracleValidator,
        LLMStructuredChallengerAgent,
        LLMStructuredRefinerAgent,
        LLMTraceContradictionReviewer,
        LLMTraceInsightSynthesizer,
        TranslatorChallengeReplayAdapter,
        run_offline_adversarial_learning,
    )
    from .artifacts import default_run_root, project_root
    from .experiment_profiles import (
        model_default_insight_bank_path,
        model_display_name,
        profile_choices,
        resolve_experiment_profile,
    )
    from .insights import DEFAULT_INSIGHTS, load_insight_library
    from .translation_agent import DEFAULT_RESPONSES_API_URL, DEFAULT_RESPONSES_MODEL, create_agent
    from .translator import Translator
except ImportError:
    from adversarial_insights import (
        InsightBank,
        LLMActivationValidator,
        LLMCompositionRelationReviewer,
        LLMSemanticOracleValidator,
        LLMStructuredChallengerAgent,
        LLMStructuredRefinerAgent,
        LLMTraceContradictionReviewer,
        LLMTraceInsightSynthesizer,
        TranslatorChallengeReplayAdapter,
        run_offline_adversarial_learning,
    )
    from artifacts import default_run_root, project_root
    from experiment_profiles import (
        model_default_insight_bank_path,
        model_display_name,
        profile_choices,
        resolve_experiment_profile,
    )
    from insights import DEFAULT_INSIGHTS, load_insight_library
    from translation_agent import DEFAULT_RESPONSES_API_URL, DEFAULT_RESPONSES_MODEL, create_agent
    from translator import Translator


RUN_ID_TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"
DEFAULT_CHAT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_GOOGLE_GENAI_API_URL = "https://generativelanguage.googleapis.com"
DEFAULT_ADVERSARIAL_INSIGHT_OUTPUT_BANK = "adversarial_insights.updated.json"
DEFAULT_ADVERSARIAL_INSIGHT_REPORT = "adversarial_insight_report.json"
DEFAULT_ACTIVE_INSIGHT_BANK = "insights.active.json"
DEFAULT_PRE_TRANSLATION_REFINEMENT_ROUNDS = 1
_ACTIVE_INSIGHT_BANK_LOCK = threading.Lock()

SINGLE_INSIGHT_ADVERSARIAL_OPTION_KEYS = {
    "admit_trace_candidates",
    "run_synthetic_challenges",
    "live_structured_agents",
    "live_trace_synthesis",
    "live_trace_contradiction_review",
    "live_semantic_oracle_validation",
    "live_activation_validation",
    "live_translator_replay",
    "pre_translation_challenges",
    "pre_translation_refinement",
    "pre_translation_refinement_writeback",
}

COMPOSITIONAL_ADVERSARIAL_OPTION_KEYS = {
    "run_compositional_challenges",
    "admit_composition_failure_candidates",
    "live_composition_relation_review",
}

MODEL_CONFIG_OVERRIDES = {
    "ali-deepseek-v4-flash": {
        "protocol": "openai-chat",
        "max_tokens": 128000,
        "chat_extra_body": {"enable_thinking": True},
        "chat_token_parameter": "max_tokens",
    },
    "kimi-k2.5": {
        "protocol": "openai-chat",
        "max_tokens": 128000,
        "chat_extra_body": None,
        "chat_token_parameter": "max_completion_tokens",
    },
    "gemini-3.1-p-priority": {
        "protocol": "google-genai",
        "api_url": DEFAULT_GOOGLE_GENAI_API_URL,
        "max_tokens": 65536,
        "google_api_version": "v1",
    },
    "gemini-3.1-p": {
        "protocol": "google-genai",
        "api_url": DEFAULT_GOOGLE_GENAI_API_URL,
        "max_tokens": 65536,
        "google_api_version": "v1",
    },
}


def load_env_file(path: Path, *, override: bool = False) -> None:
    """Load simple KEY=VALUE lines from a .env file into os.environ."""
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _optional_env_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _api_key_default() -> str:
    return os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY", "")


def resolve_run_id(run_id: Optional[str]) -> str:
    """Return a non-empty run id; default to current local timestamp."""
    if run_id:
        return run_id
    return time.strftime(RUN_ID_TIMESTAMP_FORMAT)


DEFAULT_PROJECTS = [
    "leftpad",
    "morton",
    "geofence",
    "amp",
    "ulidgen",
    "NandC",
    "avalanche",
    "libqueue",
    "Phills_DHT",
    "bitset",
    "chtrie",
    "gfc",
    "csyncmers",
    "file2str",
    "fft",
    "42-Kocaeli-Printf",
    "Math-Library-in-C",
    "utf8",
    "FastHamming",
    "libvcd",
    "murmurhash_c",
    "coroutine",
    "totp",
    "vec",
    "hydra",
    "CircularBuffer",
    "libwecan",
    "ljmm",
    "rhbloom",
    "rect_pack_h",
    "btree-map",
    "c-string",
    "fs_c",
    "2DPartInt",
    "libbase122",
    "aes128-SIMD",
    "hamta",
    "kd3",
    "libbeaufort",
    "Simple-Sparsehash",
    "SimpleXML",
    "quadtree",
    "clog",
    "Graph-recogniser",
    "Holdem-Odds",
    "SlothLang",
    "gorilla-paper-encode",
    "bhshell",
    "roaring-bitmap",
    "clhash",
    "bostree",
    "matrix_multiplication",
    "lib2bit",
    "cset",
    "carrays",
    "rbtree-lab",
    "rubiksolver",
    "simple_lang",
    "bigint",
    "cJSON",
    "satc",
    "inversion_list",
    "blt",
    "Simple-Config",
    "libtinyfseq",
    "approxidate",
    "ted",
    "c-aces",
    "merkle-tree-c",
    "dict",
    "fleur",
    "c-blind-rsa-signatures",
    "cissy",
    "Linear-Algebra-C",
    "emlang",
    "expr",
    "mvptree",
    "VaultSync",
    "razz_simulation",
    "Remimu",
    "libpgn",
    "LTRE",
    "lambda-calculus-eval",
    "jccc",
    "libpsbt",
    "mdb",
    "Genetic-neural-network-for-simple-control",
    "impcheck",
    "cfsm",
    "worsp",
    "libutf",
    "skp",
    "kairoCompiler",
    "fslib",
    "XOpt",
    "recordManager",
    "libm17",
    "tisp",
    "Megalania",
    "libfor",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run C->Rust translation for one or more projects.")
    parser.add_argument("projects", nargs="*", help="Project names in data/c_metadata")
    parser.add_argument(
        "-p",
        "--project",
        dest="project_options",
        action="append",
        default=[],
        help="Project name to run; repeat this option to run multiple projects",
    )
    parser.add_argument(
        "--projects",
        dest="project_csv",
        help="Comma-separated project names, for example: leftpad,morton,file2str",
    )
    parser.add_argument(
        "--list-projects",
        action="store_true",
        help="List available projects discovered from data/c_metadata and exit",
    )
    parser.add_argument("--limit", type=int, default=3, help="Default projects count when no positional projects are passed")
    parser.add_argument("--max-fix-attempts", type=int, default=5, help="Max compilation fix attempts per module")
    parser.add_argument("--max-test-fix-attempts", type=int, default=3, help="Max project-level test fix attempts after cargo test failures")
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Do not stop immediately when a module translation fails",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL", DEFAULT_RESPONSES_MODEL),
        help="LLM model name (env: LLM_MODEL)",
    )
    parser.add_argument(
        "--llm-protocol",
        choices=["openai-responses", "openai-chat", "google-genai"],
        default=os.getenv("LLM_PROTOCOL", "openai-responses"),
        help="LLM protocol adapter to use (env: LLM_PROTOCOL)",
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("LLM_API_URL", DEFAULT_RESPONSES_API_URL),
        help="Responses or Google GenAI API URL when --llm-protocol uses api-url (env: LLM_API_URL)",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BASE_URL", DEFAULT_CHAT_BASE_URL),
        help="Chat Completions API base URL when --llm-protocol=openai-chat (env: LLM_BASE_URL)",
    )
    parser.add_argument(
        "--api-key",
        default=_api_key_default(),
        help="LLM API key. No default; export OPENAI_API_KEY/LLM_API_KEY or pass --api-key explicitly.",
    )
    parser.add_argument(
        "--session-id",
        default=os.getenv("LLM_SESSION_ID", ""),
        help="Session id used in the Responses extra header (env: LLM_SESSION_ID)",
    )
    parser.add_argument(
        "--logid",
        default=os.getenv("LLM_LOGID") or os.getenv("X_TT_LOGID", ""),
        help="X-TT-logid header value for Responses requests (env: LLM_LOGID or X_TT_LOGID)",
    )
    parser.add_argument(
        "--min-request-interval",
        type=float,
        default=_env_float("LLM_MIN_REQUEST_INTERVAL", 0.0),
        help="Minimum seconds between LLM request starts within each project worker (env: LLM_MIN_REQUEST_INTERVAL)",
    )
    parser.add_argument(
        "--rate-limit-retry-delay",
        type=float,
        default=_env_float("LLM_RATE_LIMIT_RETRY_DELAY", 60.0),
        help="Minimum retry sleep seconds after HTTP 429 responses (env: LLM_RATE_LIMIT_RETRY_DELAY)",
    )
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=_env_int("LLM_TIMEOUT", 300),
        help="Per-request LLM timeout in seconds (env: LLM_TIMEOUT)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=_env_int("TRANSLATION_JOBS", 1),
        help="Number of projects to translate concurrently (env: TRANSLATION_JOBS)",
    )
    parser.add_argument(
        "--alignment-mode",
        choices=["llm", "rule"],
        default=os.getenv("ALIGNMENT_MODE", "llm"),
        help="C/Rust alignment mode before translation (env: ALIGNMENT_MODE)",
    )
    parser.add_argument(
        "--alignment-confidence-threshold",
        type=float,
        default=_env_float("ALIGNMENT_CONFIDENCE_THRESHOLD", 0.65),
        help="Minimum LLM alignment confidence accepted as a safe binding",
    )
    parser.add_argument(
        "--insights-mode",
        choices=["default", "off"],
        default=os.getenv("INSIGHTS_MODE", "default"),
        help="Use structured translation insights in LLM prompts, or disable them for ablation (env: INSIGHTS_MODE)",
    )
    parser.add_argument(
        "--insight-limit",
        type=int,
        default=_env_int("INSIGHT_LIMIT", 5),
        help="Maximum selected translation insights to include per prompt (env: INSIGHT_LIMIT)",
    )
    parser.add_argument(
        "--insights-path",
        default=os.getenv("INSIGHTS_PATH", ""),
        help="Optional Python file exporting DEFAULT_INSIGHTS and select_insights for this run (env: INSIGHTS_PATH)",
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
    parser.add_argument(
        "--run-id",
        default="",
        help=(
            "Subdirectory name under output/runs/. Defaults to a fresh timestamp (YYYYmmdd-HHMMSS) "
            "so each run writes to output/runs/<run_id>/<project>/ and never overwrites previous runs. "
            "Pass an explicit value to reuse / target an existing run directory."
        ),
    )
    return parser


def discover_available_projects(root_dir: Path) -> list[str]:
    metadata_dir = root_dir / "data" / "c_metadata"
    if not metadata_dir.exists():
        return []

    projects = []
    for metadata_file in sorted(metadata_dir.glob("*.json")):
        project_name = metadata_file.stem
        if project_name.endswith("_dep_graph"):
            continue
        projects.append(project_name)
    return projects


def _split_project_csv(project_csv: Optional[str]) -> list[str]:
    if not project_csv:
        return []
    return [item.strip() for item in project_csv.split(",") if item.strip()]


def _dedupe_projects(projects: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    seen = set()
    for project_name in projects:
        if not project_name or project_name in seen:
            continue
        seen.add(project_name)
        deduped.append(project_name)
    return deduped


def _select_projects(
    cli_projects: Iterable[str],
    option_projects: Iterable[str],
    csv_projects: Iterable[str],
    limit: int,
    available_projects: Iterable[str],
) -> list[str]:
    requested_projects = _dedupe_projects([*cli_projects, *option_projects, *csv_projects])
    if requested_projects:
        return requested_projects

    available_project_list = list(available_projects)
    if available_project_list:
        available_project_set = set(available_project_list)
        default_projects = [project for project in DEFAULT_PROJECTS if project in available_project_set]
        if default_projects:
            return default_projects[: max(limit, 0)]
        return available_project_list[: max(limit, 0)]

    return DEFAULT_PROJECTS[: max(limit, 0)]


def _validate_projects(
    parser: argparse.ArgumentParser,
    projects: Iterable[str],
    available_projects: Iterable[str],
) -> None:
    available_project_list = list(available_projects)
    available_project_set = set(available_project_list)
    unknown_projects = [project for project in projects if project not in available_project_set]
    if not unknown_projects:
        return

    unknown_parts = []
    for project_name in unknown_projects:
        matches = difflib.get_close_matches(project_name, available_project_list, n=3)
        if matches:
            unknown_parts.append(f"{project_name} (did you mean: {', '.join(matches)})")
        else:
            unknown_parts.append(project_name)

    parser.error(
        "Unknown project(s): "
        + "; ".join(unknown_parts)
        + ". Use --list-projects to inspect available project names."
    )


def build_llm_config(args: argparse.Namespace) -> dict:
    config = {
        "api_key": args.api_key,
        "api_url": args.api_url,
        "base_url": args.base_url,
        "model_name": args.model,
        "protocol": args.llm_protocol,
        "temperature": 1.0,
        "max_retries": 20,
        "retry_initial_delay": 4,
        "retry_max_delay": 60,
        "max_tokens": 128000,
        "reasoning_effort": "medium",
        "include": ["reasoning.encrypted_content"],
        "min_request_interval": getattr(args, "min_request_interval", 0.0),
        "rate_limit_retry_delay": getattr(args, "rate_limit_retry_delay", 60.0),
        "timeout": getattr(args, "llm_timeout", 300),
        "session_id": args.session_id,
        "logid": args.logid,
    }
    config.update(MODEL_CONFIG_OVERRIDES.get(args.model, {}))
    return config


def _resolve_insights_path(root_dir: Path, raw_path: str) -> str:
    if not raw_path:
        return ""
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = root_dir / path
    return str(path.resolve())


def _resolve_optional_path(root_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = root_dir / path
    return path.resolve()


def adversarial_insight_update_paths(
    root_dir: Path,
    run_root: Path,
    args: argparse.Namespace,
    *,
    insights_json_bank_path: str = "",
) -> tuple[Path | None, Path | None, Path | None]:
    raw_bank = getattr(args, "adversarial_insight_bank", "") or ""
    online_uses_prompt_bank = False
    if not raw_bank and online_adversarial_insights_enabled(args):
        raw_bank = insights_json_bank_path or getattr(args, "insights_json_bank", "") or ""
        online_uses_prompt_bank = bool(raw_bank)
    if not raw_bank:
        return None, None, None

    bank_path = _resolve_optional_path(root_dir, raw_bank)
    if (
        online_adversarial_insights_enabled(args)
        and insights_json_bank_path
        and bank_path.resolve() == Path(insights_json_bank_path).expanduser().resolve()
    ):
        online_uses_prompt_bank = True
    raw_output_bank = getattr(args, "adversarial_insight_output_bank", "") or ""
    raw_report = getattr(args, "adversarial_insight_report", "") or ""
    output_bank_path = (
        _resolve_optional_path(root_dir, raw_output_bank)
        if raw_output_bank
        else bank_path
        if online_uses_prompt_bank
        else run_root / DEFAULT_ADVERSARIAL_INSIGHT_OUTPUT_BANK
    )
    report_path = (
        _resolve_optional_path(root_dir, raw_report)
        if raw_report
        else run_root / DEFAULT_ADVERSARIAL_INSIGHT_REPORT
    )
    return bank_path, output_bank_path, report_path


def ensure_active_json_insight_bank(
    root_dir: Path,
    run_root: Path,
    *,
    insights_path: str,
    source_json_bank_path: str = "",
) -> str:
    bank_path = run_root / DEFAULT_ACTIVE_INSIGHT_BANK
    with _ACTIVE_INSIGHT_BANK_LOCK:
        if bank_path.exists():
            return str(bank_path)

        if source_json_bank_path:
            source_bank = InsightBank.load(_resolve_insights_path(root_dir, source_json_bank_path))
            source_bank.save(bank_path)
        elif insights_path:
            library = load_insight_library(_resolve_insights_path(root_dir, insights_path))
            source_insights = library.insights
            InsightBank.from_legacy(source_insights).save(bank_path)
        else:
            source_insights = DEFAULT_INSIGHTS
            InsightBank.from_legacy(source_insights).save(bank_path)
    return str(bank_path)


def active_json_insight_bank_path(
    root_dir: Path,
    run_root: Path,
    args: argparse.Namespace,
    *,
    project_name: str = "",
) -> str:
    raw_bank = getattr(args, "insights_json_bank", "") or ""
    if raw_bank:
        return _resolve_insights_path(root_dir, raw_bank)
    raw_adversarial_bank = getattr(args, "adversarial_insight_bank", "") or ""
    if raw_adversarial_bank and _explicit_adversarial_bank_primary_store_enabled(args):
        return _resolve_insights_path(root_dir, raw_adversarial_bank)

    raw_profile = getattr(args, "experiment_profile", "") or ""
    profile = resolve_experiment_profile(raw_profile) if raw_profile else None
    insights_enabled = getattr(args, "insights_mode", "default") != "off"
    if profile is not None:
        insights_enabled = insights_enabled and profile.insights_enabled
    if not insights_enabled:
        return ""

    raw_insights_path = getattr(args, "insights_path", "") or ""
    profile_insights_path = profile.default_insights_path if profile else ""
    effective_insights_path = raw_insights_path or profile_insights_path
    source_json_bank_path = ""
    if not raw_insights_path and profile is not None:
        source_json_bank_path = model_default_insight_bank_path(str(getattr(args, "model", "") or ""))
    bank_root = (
        project_root(run_root, project_name)
        if project_name and project_local_online_bank_enabled(args)
        else run_root
    )
    return ensure_active_json_insight_bank(
        root_dir,
        bank_root,
        insights_path=effective_insights_path,
        source_json_bank_path=source_json_bank_path,
    )


def adversarial_learning_option_metadata(args: argparse.Namespace) -> dict:
    profile_defaults = profile_adversarial_learning_defaults(args)
    flags = {
        "admit_trace_candidates": profile_defaults["admit_trace_candidates"]
        or bool(getattr(args, "adversarial_admit_trace_candidates", False)),
        "run_synthetic_challenges": profile_defaults["run_synthetic_challenges"]
        or bool(getattr(args, "adversarial_run_synthetic_challenges", False)),
        "run_compositional_challenges": profile_defaults["run_compositional_challenges"]
        or bool(getattr(args, "adversarial_run_compositional_challenges", False)),
        "admit_composition_failure_candidates": profile_defaults[
            "admit_composition_failure_candidates"
        ]
        or bool(getattr(args, "adversarial_admit_composition_failure_candidates", False)),
        "live_structured_agents": profile_defaults["live_structured_agents"]
        or bool(getattr(args, "adversarial_live_structured_agents", False)),
        "live_trace_synthesis": profile_defaults["live_trace_synthesis"]
        or bool(getattr(args, "adversarial_live_trace_synthesis", False)),
        "live_trace_contradiction_review": profile_defaults["live_trace_contradiction_review"]
        or bool(getattr(args, "adversarial_live_trace_contradiction_review", False)),
        "live_composition_relation_review": profile_defaults["live_composition_relation_review"]
        or bool(getattr(args, "adversarial_live_composition_relation_review", False)),
        "live_semantic_oracle_validation": profile_defaults["live_semantic_oracle_validation"]
        or bool(getattr(args, "adversarial_live_semantic_oracle_validation", False)),
        "live_activation_validation": profile_defaults["live_activation_validation"]
        or bool(getattr(args, "adversarial_live_activation_validation", False)),
        "live_translator_replay": profile_defaults["live_translator_replay"]
        or bool(getattr(args, "adversarial_live_translator_replay", False)),
        "pre_translation_challenges": profile_defaults["pre_translation_challenges"]
        or bool(getattr(args, "adversarial_pre_translation_challenges", False)),
        "pre_translation_refinement": profile_defaults["pre_translation_refinement"]
        or bool(getattr(args, "adversarial_pre_translation_refinement", False)),
        "pre_translation_refinement_writeback": profile_defaults[
            "pre_translation_refinement_writeback"
        ]
        or bool(getattr(args, "adversarial_pre_translation_refinement_writeback", False)),
    }
    if (
        flags["pre_translation_refinement"]
        and not flags["pre_translation_refinement_writeback"]
        and _pre_translation_refinement_writeback_defaults_allowed(args)
    ):
        flags["pre_translation_refinement_writeback"] = True
    gated = apply_profile_adversarial_gates(args, flags)
    gated["pre_translation_refinement_rounds"] = pre_translation_refinement_round_count(
        args,
        enabled=bool(gated.get("pre_translation_refinement", False)),
    )
    return gated


def profile_adversarial_learning_defaults(args: argparse.Namespace) -> dict[str, bool]:
    defaults = {
        "admit_trace_candidates": False,
        "run_synthetic_challenges": False,
        "run_compositional_challenges": False,
        "admit_composition_failure_candidates": False,
        "live_structured_agents": False,
        "live_trace_synthesis": False,
        "live_trace_contradiction_review": False,
        "live_composition_relation_review": False,
        "live_semantic_oracle_validation": False,
        "live_activation_validation": False,
        "live_translator_replay": False,
        "pre_translation_challenges": False,
        "pre_translation_refinement": False,
        "pre_translation_refinement_writeback": False,
    }
    raw_profile = getattr(args, "experiment_profile", "") or ""
    if not raw_profile:
        return defaults

    profile = resolve_experiment_profile(raw_profile)
    if not profile.insights_enabled or not profile.insight_mining:
        return defaults

    if profile.single_insight_adversarial_learning:
        for key in SINGLE_INSIGHT_ADVERSARIAL_OPTION_KEYS:
            defaults[key] = True
        if (
            _effective_jobs(args) == 1
            or project_local_online_bank_enabled(args)
            or _shared_prompt_bank_online_enabled(args)
            or _explicit_adversarial_bank_primary_store_enabled(args)
        ):
            defaults["pre_translation_refinement_writeback"] = True

    if profile.compositional_insight_adversarial_learning:
        for key in COMPOSITIONAL_ADVERSARIAL_OPTION_KEYS:
            defaults[key] = True

    return defaults


def pre_translation_refinement_round_count(
    args: argparse.Namespace,
    *,
    enabled: bool,
) -> int:
    if not enabled:
        return 0
    raw_rounds = getattr(args, "adversarial_pre_translation_refinement_rounds", None)
    if raw_rounds is not None:
        try:
            return max(int(raw_rounds), 1)
        except (TypeError, ValueError):
            return DEFAULT_PRE_TRANSLATION_REFINEMENT_ROUNDS
    raw_profile = getattr(args, "experiment_profile", "") or ""
    if raw_profile:
        profile = resolve_experiment_profile(raw_profile)
        if (
            profile.insights_enabled
            and profile.insight_mining
            and profile.single_insight_adversarial_learning
        ):
            return max(profile.default_adversarial_rounds, 1)
    return DEFAULT_PRE_TRANSLATION_REFINEMENT_ROUNDS


def online_adversarial_insights_enabled(args: argparse.Namespace) -> bool:
    if bool(getattr(args, "online_adversarial_insights", False)):
        return True
    if _explicit_adversarial_bank_primary_store_enabled(args):
        return True
    if _custom_online_defaults_allowed(args):
        return True
    if not _profile_online_defaults_allowed(args):
        return False
    if _effective_jobs(args) == 1:
        return True
    return project_local_online_bank_enabled(args) or _shared_prompt_bank_online_enabled(args)


def project_local_online_bank_enabled(args: argparse.Namespace) -> bool:
    if _effective_jobs(args) <= 1:
        return False
    if bool(getattr(args, "insights_json_bank", "") or ""):
        return False
    if bool(getattr(args, "adversarial_insight_bank", "") or ""):
        return False
    return (
        _profile_online_defaults_allowed(args)
        or _custom_parallel_online_defaults_allowed(args)
    )


def parallel_online_adversarial_updates_supported(args: argparse.Namespace) -> bool:
    return (
        project_local_online_bank_enabled(args)
        or _shared_prompt_bank_online_enabled(args)
        or _explicit_parallel_adversarial_bank_online_enabled(args)
    )


def _custom_online_defaults_allowed(args: argparse.Namespace) -> bool:
    if _effective_jobs(args) == 1:
        return _custom_online_base_defaults_allowed(args)
    return (
        _custom_parallel_online_defaults_allowed(args)
        or _custom_parallel_shared_prompt_bank_online_defaults_allowed(args)
    )


def _custom_parallel_online_defaults_allowed(args: argparse.Namespace) -> bool:
    return _custom_online_base_defaults_allowed(args) and not bool(
        getattr(args, "insights_json_bank", "") or ""
    )


def _custom_parallel_shared_prompt_bank_online_defaults_allowed(args: argparse.Namespace) -> bool:
    return _custom_online_base_defaults_allowed(args) and bool(
        getattr(args, "insights_json_bank", "") or ""
    )


def _custom_online_base_defaults_allowed(args: argparse.Namespace) -> bool:
    if not hasattr(args, "insights_mode"):
        return False
    if getattr(args, "experiment_profile", "") or "":
        return False
    if bool(getattr(args, "adversarial_insight_bank", "") or ""):
        return False
    return getattr(args, "insights_mode", "default") != "off"


def _shared_prompt_bank_online_enabled(args: argparse.Namespace) -> bool:
    return (
        _effective_jobs(args) > 1
        and (
            bool(getattr(args, "online_adversarial_insights", False))
            or _profile_online_defaults_allowed(args)
            or _custom_parallel_shared_prompt_bank_online_defaults_allowed(args)
        )
        and bool(getattr(args, "insights_json_bank", "") or "")
        and not bool(getattr(args, "adversarial_insight_bank", "") or "")
    )


def _pre_translation_refinement_writeback_defaults_allowed(args: argparse.Namespace) -> bool:
    if getattr(args, "insights_mode", "default") == "off":
        return False
    return online_adversarial_insights_enabled(args)


def _explicit_parallel_adversarial_bank_online_enabled(args: argparse.Namespace) -> bool:
    return (
        _effective_jobs(args) > 1
        and (
            bool(getattr(args, "online_adversarial_insights", False))
            or _explicit_adversarial_bank_primary_store_enabled(args)
        )
        and bool(getattr(args, "adversarial_insight_bank", "") or "")
    )


def _explicit_adversarial_bank_primary_store_enabled(args: argparse.Namespace) -> bool:
    if not bool(getattr(args, "adversarial_insight_bank", "") or ""):
        return False
    if not hasattr(args, "insights_mode"):
        return False
    if getattr(args, "insights_mode", "default") == "off":
        return False
    raw_profile = getattr(args, "experiment_profile", "") or ""
    if not raw_profile:
        return True

    profile = resolve_experiment_profile(raw_profile)
    return (
        profile.insights_enabled
        and profile.insight_mining
        and profile.single_insight_adversarial_learning
    )


def _profile_online_defaults_allowed(args: argparse.Namespace) -> bool:
    raw_profile = getattr(args, "experiment_profile", "") or ""
    if not raw_profile:
        return False

    profile = resolve_experiment_profile(raw_profile)
    return (
        profile.insights_enabled
        and profile.insight_mining
        and profile.single_insight_adversarial_learning
    )


def _effective_jobs(args: argparse.Namespace) -> int:
    try:
        return max(int(getattr(args, "jobs", 1) or 1), 1)
    except (TypeError, ValueError):
        return 1


def apply_profile_adversarial_gates(args: argparse.Namespace, flags: dict[str, bool]) -> dict[str, bool]:
    gated = dict(flags)
    raw_profile = getattr(args, "experiment_profile", "") or ""
    if not raw_profile:
        return gated

    profile = resolve_experiment_profile(raw_profile)
    if not profile.insights_enabled or not profile.insight_mining:
        for key in gated:
            gated[key] = False
        return gated

    if not profile.single_insight_adversarial_learning:
        for key in SINGLE_INSIGHT_ADVERSARIAL_OPTION_KEYS | COMPOSITIONAL_ADVERSARIAL_OPTION_KEYS:
            gated[key] = False
        return gated

    if not profile.compositional_insight_adversarial_learning:
        for key in COMPOSITIONAL_ADVERSARIAL_OPTION_KEYS:
            gated[key] = False

    return gated


def project_config(root_dir: Path, run_root: Path, project_name: str, args: argparse.Namespace) -> dict:
    raw_profile = getattr(args, "experiment_profile", "") or ""
    profile = resolve_experiment_profile(raw_profile) if raw_profile else None
    raw_insights_path = getattr(args, "insights_path", "")
    profile_insights_path = profile.default_insights_path if profile else ""
    effective_insights_path = raw_insights_path or profile_insights_path

    insights_enabled = getattr(args, "insights_mode", "default") != "off"
    if profile is not None:
        insights_enabled = insights_enabled and profile.insights_enabled
    insight_interaction_scorer = getattr(args, "insight_interaction_scorer", "heuristic") or "heuristic"
    if (
        insight_interaction_scorer == "heuristic"
        and profile is not None
        and insights_enabled
        and profile.insight_mining
        and profile.compositional_insight_adversarial_learning
    ):
        insight_interaction_scorer = "live"

    raw_rounds = getattr(args, "adversarial_rounds", None)
    if raw_rounds is None:
        adversarial_rounds = profile.default_adversarial_rounds if profile else 0
    else:
        adversarial_rounds = max(raw_rounds, 0)

    insights_json_bank_path = active_json_insight_bank_path(
        root_dir,
        run_root,
        args,
        project_name=project_name,
    )
    project_local_online_bank = bool(project_local_online_bank_enabled(args) and insights_json_bank_path)
    shared_prompt_online_bank = bool(
        _shared_prompt_bank_online_enabled(args) and insights_json_bank_path
    )
    explicit_parallel_adversarial_bank = _explicit_parallel_adversarial_bank_online_enabled(args)
    adversarial_update_run_root = (
        Path(insights_json_bank_path).parent
        if project_local_online_bank
        else project_root(run_root, project_name)
        if shared_prompt_online_bank or explicit_parallel_adversarial_bank
        else run_root
    )
    adversarial_bank, adversarial_output_bank, adversarial_report = adversarial_insight_update_paths(
        root_dir,
        adversarial_update_run_root,
        args,
        insights_json_bank_path=insights_json_bank_path,
    )
    retrieval_backend = getattr(args, "insight_retrieval_backend", "auto") or "auto"
    retrieval_cache_root = (
        Path(insights_json_bank_path).parent
        if project_local_online_bank
        else run_root
    )
    retrieval_index_cache_path = insight_retrieval_index_cache_path(
        root_dir,
        retrieval_cache_root,
        raw_cache_path=getattr(args, "insight_retrieval_index_cache", "") or "",
        insights_json_bank_path=insights_json_bank_path,
        backend_preference=retrieval_backend,
    )

    return {
        "metadata_path": str(root_dir / f"data/c_metadata/{project_name}.json"),
        "rbench_path": str(root_dir / f"data/project/RBench/{project_name}"),
        "output_dir": str(project_root(run_root, project_name)),
        "llm_config": build_llm_config(args),
        "max_fix_attempts": args.max_fix_attempts,
        "max_test_fix_attempts": args.max_test_fix_attempts,
        "stop_on_first_failure": not args.continue_on_failure,
        "alignment_mode": getattr(args, "alignment_mode", "llm"),
        "alignment_confidence_threshold": getattr(args, "alignment_confidence_threshold", 0.65),
        "insights_enabled": insights_enabled,
        "insight_limit": max(getattr(args, "insight_limit", 5), 0),
        "insights_path": _resolve_insights_path(root_dir, effective_insights_path),
        "insights_json_bank_path": insights_json_bank_path,
        "insight_interaction_scorer": insight_interaction_scorer,
        "insight_retrieval_backend_preference": retrieval_backend,
        "insight_retrieval_index_cache_path": str(retrieval_index_cache_path),
        "experiment_profile": profile.name if profile else "custom",
        "experiment_ablation_label": profile.display_label if profile else "custom",
        "insight_stage": profile.insight_stage if profile else ("off" if not insights_enabled else "custom"),
        "insight_mining": profile.insight_mining if profile else insights_enabled,
        "single_insight_adversarial_learning": (
            profile.single_insight_adversarial_learning if profile else False
        ),
        "compositional_insight_adversarial_learning": (
            profile.compositional_insight_adversarial_learning if profile else False
        ),
        "adversarial_rounds": adversarial_rounds,
        "online_adversarial_insights": online_adversarial_insights_enabled(args),
        "adversarial_insight_bank_path": str(adversarial_bank) if adversarial_bank else "",
        "adversarial_insight_output_bank_path": (
            str(adversarial_output_bank) if adversarial_output_bank else ""
        ),
        "adversarial_insight_report_path": str(adversarial_report) if adversarial_report else "",
        "adversarial_learning_options": adversarial_learning_option_metadata(args),
        "model_display_name": model_display_name(getattr(args, "model", "")),
    }


def insight_retrieval_index_cache_path(
    root_dir: Path,
    run_root: Path,
    *,
    raw_cache_path: str,
    insights_json_bank_path: str,
    backend_preference: str,
) -> Path | str:
    if not insights_json_bank_path:
        return ""
    if raw_cache_path:
        return _resolve_insights_path(root_dir, raw_cache_path)
    if backend_preference == "keyword":
        return ""
    return run_root / "insight_retrieval.faiss"


def adversarial_learning_options(args: argparse.Namespace) -> dict:
    flags = adversarial_learning_option_metadata(args)
    live_agent = None

    def agent():
        nonlocal live_agent
        if live_agent is None:
            live_agent = create_agent(build_llm_config(args))
        return live_agent

    challenger_agent = None
    refiner_agent = None
    if flags["live_structured_agents"]:
        challenger_agent = LLMStructuredChallengerAgent(agent())
        refiner_agent = LLMStructuredRefinerAgent(agent())

    replay_translate_fn = None
    if flags["live_translator_replay"]:
        def translator_factory(config):
            translator = Translator.__new__(Translator)
            translator.config = config
            translator.agent = agent()
            translator.llm_calls = []
            return translator

        replay_translate_fn = TranslatorChallengeReplayAdapter(translator_factory)

    return {
        "admit_trace_candidates": flags["admit_trace_candidates"],
        "admit_composition_failure_candidates": flags["admit_composition_failure_candidates"],
        "run_synthetic_challenges": flags["run_synthetic_challenges"],
        "run_compositional_challenges": flags["run_compositional_challenges"],
        "pre_translation_challenges": flags["pre_translation_challenges"],
        "pre_translation_refinement": flags["pre_translation_refinement"],
        "pre_translation_refinement_rounds": flags["pre_translation_refinement_rounds"],
        "pre_translation_refinement_writeback": flags["pre_translation_refinement_writeback"],
        "challenger_agent": challenger_agent,
        "refiner_agent": refiner_agent,
        "oracle_validator": (
            LLMSemanticOracleValidator(agent())
            if flags["live_semantic_oracle_validation"]
            else None
        ),
        "activation_validator": (
            LLMActivationValidator(agent())
            if flags["live_activation_validation"]
            else None
        ),
        "trace_synthesizer": (
            LLMTraceInsightSynthesizer(agent())
            if flags["live_trace_synthesis"]
            else None
        ),
        "trace_contradiction_reviewer": (
            LLMTraceContradictionReviewer(agent())
            if flags["live_trace_contradiction_review"]
            else None
        ),
        "composition_relation_reviewer": (
            LLMCompositionRelationReviewer(agent())
            if flags["live_composition_relation_review"]
            else None
        ),
        "replay_translate_fn": replay_translate_fn,
    }


def run_adversarial_insight_update(
    root_dir: Path,
    run_root: Path,
    args: argparse.Namespace,
) -> dict | None:
    insights_json_bank_path = (
        active_json_insight_bank_path(root_dir, run_root, args)
        if online_adversarial_insights_enabled(args)
        else ""
    )
    bank_path, output_bank_path, report_path = adversarial_insight_update_paths(
        root_dir,
        run_root,
        args,
        insights_json_bank_path=insights_json_bank_path,
    )
    if bank_path is None or output_bank_path is None or report_path is None:
        return None

    print("Running post-run adversarial insight learning...")
    print(f"Input bank: {bank_path}")
    print(f"Output bank: {output_bank_path}")
    print(f"Report: {report_path}")
    result = run_offline_adversarial_learning(
        run_root,
        bank_path,
        output_bank_path,
        report_path,
        **adversarial_learning_options(args),
    )
    print(
        "Adversarial insight learning done: "
        f"challenges={result.get('challenges', 0)} "
        f"accepted_updates={result.get('accepted_updates', 0)}"
    )
    return result


def _run_project(
    project_name: str,
    *,
    index: int,
    total: int,
    root_dir: Path,
    run_root: Path,
    args: argparse.Namespace,
    translator_cls: Type,
) -> None:
    config = project_config(root_dir, run_root, project_name, args)
    print(f"Start translation of {project_name} project ({index}/{total})...")
    print(f"Using model: {config['llm_config']['model_name']}")
    print(f"Using protocol: {config['llm_config']['protocol']}")
    print(f"Output directory: {config['output_dir']}")
    print()

    translator = translator_cls(config)
    translator.run()


def run_project_batch(
    projects: list[str],
    *,
    root_dir: Path,
    run_root: Path,
    args: argparse.Namespace,
    translator_cls: Type,
) -> None:
    total = len(projects)
    jobs = max(args.jobs, 1)
    if jobs == 1 or total <= 1:
        for index, project_name in enumerate(projects, 1):
            _run_project(
                project_name,
                index=index,
                total=total,
                root_dir=root_dir,
                run_root=run_root,
                args=args,
                translator_cls=translator_cls,
            )
            if online_adversarial_insights_enabled(args):
                run_adversarial_insight_update(root_dir, run_root, args)
        return

    if online_adversarial_insights_enabled(args) and not parallel_online_adversarial_updates_supported(args):
        raise ValueError(
            "--online-adversarial-insights with --jobs > 1 requires a project-local "
            "active bank, --insights-json-bank, or --adversarial-insight-bank"
        )

    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=min(jobs, total)) as executor:
        futures = {
            executor.submit(
                _run_project,
                project_name,
                index=index,
                total=total,
                root_dir=root_dir,
                run_root=run_root,
                args=args,
                translator_cls=translator_cls,
            ): project_name
            for index, project_name in enumerate(projects, 1)
        }
        for future in as_completed(futures):
            project_name = futures[future]
            try:
                future.result()
            except Exception as exc:
                errors.append(f"{project_name}: {exc}")
                print(f"Project crashed: {project_name}: {exc}")

    if errors:
        raise RuntimeError("One or more projects crashed: " + "; ".join(errors))


def main(translator_cls: Optional[Type] = None) -> None:
    root_dir = Path(__file__).parent.parent
    load_env_file(root_dir / ".env")
    parser = build_arg_parser()
    args = parser.parse_args()

    if translator_cls is None:
        try:
            from .translator import Translator
        except ImportError:
            from translator import Translator
        translator_cls = Translator

    available_projects = discover_available_projects(root_dir)

    if args.list_projects:
        if available_projects:
            for project_name in available_projects:
                print(project_name)
        else:
            print(f"No metadata projects discovered under {root_dir / 'data' / 'c_metadata'}")
        return

    projects = _select_projects(
        args.projects,
        args.project_options,
        _split_project_csv(args.project_csv),
        args.limit,
        available_projects,
    )
    if available_projects:
        _validate_projects(parser, projects, available_projects)
    if not projects:
        parser.error("No projects selected. Pass a project name or use --list-projects.")

    run_id = resolve_run_id(args.run_id)
    run_root = default_run_root(root_dir, run_id)

    print(f"Selected {len(projects)} project(s): {', '.join(projects)}")
    print(f"Run id: {run_id}")
    print(f"Run root: {run_root}")
    print(f"Project jobs: {max(args.jobs, 1)}")
    print()

    run_project_batch(
        projects,
        root_dir=root_dir,
        run_root=run_root,
        args=args,
        translator_cls=translator_cls,
    )
    if not online_adversarial_insights_enabled(args):
        run_adversarial_insight_update(root_dir, run_root, args)


if __name__ == "__main__":
    main()
