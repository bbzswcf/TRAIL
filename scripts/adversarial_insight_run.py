from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from c2rust_translator.adversarial_insights import (
    LLMActivationValidator,
    LLMCompositionRelationReviewer,
    LLMSemanticOracleValidator,
    LLMTraceContradictionReviewer,
    LLMTraceInsightSynthesizer,
    LLMStructuredChallengerAgent,
    LLMStructuredRefinerAgent,
    StructuredChallengerAgent,
    StructuredRefinerAgent,
    TranslatorChallengeReplayAdapter,
    run_offline_adversarial_learning,
)
from c2rust_translator.experiment_profiles import profile_choices, resolve_experiment_profile
from c2rust_translator.translation_agent import (
    DEFAULT_RESPONSES_API_URL,
    DEFAULT_RESPONSES_MODEL,
    create_agent,
)
from c2rust_translator.translator import Translator

DEFAULT_CHAT_BASE_URL = "https://api.openai.com/v1"

SINGLE_INSIGHT_PROFILE_FLAG_KEYS = {
    "admit_trace_candidates",
    "run_synthetic_challenges",
    "live_structured_agents",
    "live_trace_synthesis",
    "live_trace_contradiction_review",
    "live_semantic_oracle_validation",
    "live_activation_validation",
    "live_translator_replay",
}

COMPOSITIONAL_PROFILE_FLAG_KEYS = {
    "run_compositional_challenges",
    "admit_composition_failure_candidates",
    "live_composition_relation_review",
}

ADVERSARIAL_FLAG_KEYS = SINGLE_INSIGHT_PROFILE_FLAG_KEYS | COMPOSITIONAL_PROFILE_FLAG_KEYS


def _api_key_default() -> str:
    return os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY", "")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run opt-in offline adversarial insight learning over existing translation artifacts."
    )
    parser.add_argument("run_root", help="Run root containing per-project translation artifacts.")
    parser.add_argument("--bank", required=True, help="Input JSON insight bank path.")
    parser.add_argument("--output-bank", required=True, help="Output JSON insight bank path.")
    parser.add_argument("--report", required=True, help="Output evolution report JSON path.")
    parser.add_argument(
        "--experiment-profile",
        choices=profile_choices(),
        default=os.getenv("EXPERIMENT_PROFILE", ""),
        help=(
            "Apply named experiment profile defaults/gates to offline adversarial learning "
            "(env: EXPERIMENT_PROFILE)."
        ),
    )
    parser.add_argument(
        "--admit-trace-candidates",
        action="store_true",
        help="Validate trace-mined candidate insights and append admitted candidates to the output bank.",
    )
    parser.add_argument(
        "--run-synthetic-challenges",
        action="store_true",
        help="Execute deterministic single-insight synthetic challenges and record tool/sanity evidence.",
    )
    parser.add_argument(
        "--run-compositional-challenges",
        action="store_true",
        help="Execute deterministic compositional insight challenges and record tool/sanity evidence.",
    )
    parser.add_argument(
        "--admit-composition-failure-candidates",
        action="store_true",
        help=(
            "When a compositional challenge compiles, passes sanity, and fails its tests, "
            "turn that failure into a candidate insight and validate it through the single-insight gate."
        ),
    )
    parser.add_argument(
        "--live-composition-relation-review",
        action="store_true",
        default=os.getenv("LIVE_COMPOSITION_RELATION_REVIEW", "").lower() in {"1", "true", "yes"},
        help=(
            "Use the configured LLM provider to infer supports/requires/precedes relations "
            "for passed compositional challenges (env: LIVE_COMPOSITION_RELATION_REVIEW)."
        ),
    )
    parser.add_argument(
        "--live-semantic-oracle-validation",
        action="store_true",
        default=os.getenv("LIVE_SEMANTIC_ORACLE_VALIDATION", "").lower() in {"1", "true", "yes"},
        help=(
            "Use the configured LLM provider to validate generated challenge oracle/harness semantics "
            "during sanity checks (env: LIVE_SEMANTIC_ORACLE_VALIDATION)."
        ),
    )
    parser.add_argument(
        "--live-activation-validation",
        action="store_true",
        default=os.getenv("LIVE_ACTIVATION_VALIDATION", "").lower() in {"1", "true", "yes"},
        help=(
            "Use the configured LLM provider to validate whether generated challenges activate "
            "their target insights during sanity checks (env: LIVE_ACTIVATION_VALIDATION)."
        ),
    )
    parser.add_argument(
        "--live-translator-replay",
        action="store_true",
        default=os.getenv("LIVE_TRANSLATOR_REPLAY", "").lower() in {"1", "true", "yes"},
        help=(
            "Use the configured LLM provider through Translator.translate_adversarial_challenge "
            "for fresh before/after replay evidence on trace updates (env: LIVE_TRANSLATOR_REPLAY)."
        ),
    )
    parser.add_argument(
        "--structured-challenger-responses",
        help=(
            "JSON/JSONL file containing structured Challenger schema responses keyed by insight_id. "
            "When set, synthetic and trace-candidate challenge generation use these responses."
        ),
    )
    parser.add_argument(
        "--structured-refiner-responses",
        help=(
            "JSON/JSONL file containing structured Refiner schema responses keyed by insight_id. "
            "When set, trace challenge refinement uses these responses before evidence gating."
        ),
    )
    parser.add_argument(
        "--live-structured-agents",
        action="store_true",
        default=os.getenv("LIVE_STRUCTURED_ADVERSARIAL_AGENTS", "").lower() in {"1", "true", "yes"},
        help=(
            "Use the configured LLM provider for structured Challenger and Refiner calls "
            "(env: LIVE_STRUCTURED_ADVERSARIAL_AGENTS)."
        ),
    )
    parser.add_argument(
        "--live-trace-synthesis",
        action="store_true",
        default=os.getenv("LIVE_TRACE_SYNTHESIS", "").lower() in {"1", "true", "yes"},
        help=(
            "Use the configured LLM provider to synthesize trace-mined candidate insights "
            "when --admit-trace-candidates is enabled (env: LIVE_TRACE_SYNTHESIS)."
        ),
    )
    parser.add_argument(
        "--live-trace-contradiction-review",
        action="store_true",
        default=os.getenv("LIVE_TRACE_CONTRADICTION_REVIEW", "").lower() in {"1", "true", "yes"},
        help=(
            "Use the configured LLM provider to semantically reject trace-mined candidates that "
            "contradict the existing bank when --admit-trace-candidates is enabled "
            "(env: LIVE_TRACE_CONTRADICTION_REVIEW)."
        ),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL", DEFAULT_RESPONSES_MODEL),
        help="LLM model name for --live-structured-agents (env: LLM_MODEL).",
    )
    parser.add_argument(
        "--llm-protocol",
        choices=["openai-responses", "openai-chat", "google-genai"],
        default=os.getenv("LLM_PROTOCOL", "openai-responses"),
        help="LLM protocol adapter for --live-structured-agents (env: LLM_PROTOCOL).",
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("LLM_API_URL", DEFAULT_RESPONSES_API_URL),
        help="Responses or Google GenAI API URL for --live-structured-agents (env: LLM_API_URL).",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BASE_URL", DEFAULT_CHAT_BASE_URL),
        help="Chat Completions API base URL for --live-structured-agents (env: LLM_BASE_URL).",
    )
    parser.add_argument(
        "--api-key",
        default=_api_key_default(),
        help="LLM API key for --live-structured-agents (env: OPENAI_API_KEY or LLM_API_KEY).",
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
        help="Minimum seconds between LLM request starts (env: LLM_MIN_REQUEST_INTERVAL).",
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
    return parser


def adversarial_learning_flags(args) -> dict[str, bool]:
    flags = {key: bool(getattr(args, key, False)) for key in ADVERSARIAL_FLAG_KEYS}
    raw_profile = getattr(args, "experiment_profile", "") or ""
    if not raw_profile:
        return flags

    profile = resolve_experiment_profile(raw_profile)
    profile_defaults = {key: False for key in ADVERSARIAL_FLAG_KEYS}
    if profile.insights_enabled and profile.insight_mining:
        if profile.single_insight_adversarial_learning:
            for key in SINGLE_INSIGHT_PROFILE_FLAG_KEYS:
                profile_defaults[key] = True
        if profile.compositional_insight_adversarial_learning:
            for key in COMPOSITIONAL_PROFILE_FLAG_KEYS:
                profile_defaults[key] = True

    for key, enabled in profile_defaults.items():
        flags[key] = flags[key] or enabled

    if not profile.insights_enabled or not profile.insight_mining:
        for key in flags:
            flags[key] = False
        return flags

    if not profile.single_insight_adversarial_learning:
        for key in ADVERSARIAL_FLAG_KEYS:
            flags[key] = False
        return flags

    if not profile.compositional_insight_adversarial_learning:
        for key in COMPOSITIONAL_PROFILE_FLAG_KEYS:
            flags[key] = False

    return flags


def _apply_adversarial_learning_flags(args) -> None:
    for key, enabled in adversarial_learning_flags(args).items():
        setattr(args, key, enabled)


def _structured_agents_from_args(args) -> tuple[StructuredChallengerAgent | None, StructuredRefinerAgent | None]:
    if getattr(args, "live_structured_agents", False):
        if args.structured_challenger_responses or args.structured_refiner_responses:
            raise ValueError("--live-structured-agents cannot be combined with structured response files")
        agent = create_agent(_llm_config_from_args(args))
        return LLMStructuredChallengerAgent(agent), LLMStructuredRefinerAgent(agent)
    return (
        _structured_challenger_agent(args.structured_challenger_responses),
        _structured_refiner_agent(args.structured_refiner_responses),
    )


def _trace_synthesizer_from_args(args):
    if not getattr(args, "live_trace_synthesis", False):
        return None
    return LLMTraceInsightSynthesizer(create_agent(_llm_config_from_args(args)))


def _trace_contradiction_reviewer_from_args(args):
    if not getattr(args, "live_trace_contradiction_review", False):
        return None
    return LLMTraceContradictionReviewer(create_agent(_llm_config_from_args(args)))


def _composition_relation_reviewer_from_args(args):
    if not getattr(args, "live_composition_relation_review", False):
        return None
    return LLMCompositionRelationReviewer(create_agent(_llm_config_from_args(args)))


def _oracle_validator_from_args(args):
    if not getattr(args, "live_semantic_oracle_validation", False):
        return None
    return LLMSemanticOracleValidator(create_agent(_llm_config_from_args(args)))


def _activation_validator_from_args(args):
    if not getattr(args, "live_activation_validation", False):
        return None
    return LLMActivationValidator(create_agent(_llm_config_from_args(args)))


def _replay_translate_fn_from_args(args):
    if not getattr(args, "live_translator_replay", False):
        return None
    agent = create_agent(_llm_config_from_args(args))

    def translator_factory(config):
        translator = Translator.__new__(Translator)
        translator.config = config
        translator.agent = agent
        translator.llm_calls = []
        return translator

    return TranslatorChallengeReplayAdapter(translator_factory)


def _llm_config_from_args(args) -> dict:
    return {
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
        "min_request_interval": args.min_request_interval,
        "rate_limit_retry_delay": args.rate_limit_retry_delay,
        "timeout": args.llm_timeout,
        "session_id": args.session_id,
        "logid": args.logid,
    }


def _structured_challenger_agent(path: str | None) -> StructuredChallengerAgent | None:
    if not path:
        return None
    return StructuredChallengerAgent(_structured_response_lookup(path))


def _structured_refiner_agent(path: str | None) -> StructuredRefinerAgent | None:
    if not path:
        return None
    return StructuredRefinerAgent(_structured_response_lookup(path))


def _structured_response_lookup(path: str):
    responses_by_id = _load_structured_responses(path)

    def lookup(payload: dict[str, object]) -> dict[str, object]:
        insight = payload.get("insight")
        insight_id = str(insight.get("id", "") if isinstance(insight, dict) else "")
        responses = responses_by_id.get(insight_id, [])
        if not responses:
            raise ValueError(f"No structured response available for insight id: {insight_id}")
        return responses.pop(0)

    return lookup


def _load_structured_responses(path: str) -> dict[str, list[dict[str, object]]]:
    response_path = Path(path)
    if response_path.suffix == ".jsonl":
        raw_items = [
            json.loads(line)
            for line in response_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        raw = json.loads(response_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "responses" in raw:
            raw_items = raw["responses"]
        elif isinstance(raw, dict) and "insight_id" in raw:
            raw_items = [raw]
        elif isinstance(raw, list):
            raw_items = raw
        else:
            raise ValueError("Structured response file must contain a response object or response list")
    responses_by_id: dict[str, list[dict[str, object]]] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            raise ValueError("Structured responses must be JSON objects")
        insight_id = str(item.get("insight_id", ""))
        if not insight_id:
            raise ValueError("Structured responses must include insight_id")
        responses_by_id.setdefault(insight_id, []).append(item)
    return responses_by_id


def main() -> None:
    args = build_parser().parse_args()
    _apply_adversarial_learning_flags(args)
    challenger_agent, refiner_agent = _structured_agents_from_args(args)
    trace_synthesizer = _trace_synthesizer_from_args(args)
    trace_contradiction_reviewer = _trace_contradiction_reviewer_from_args(args)
    composition_relation_reviewer = _composition_relation_reviewer_from_args(args)
    oracle_validator = _oracle_validator_from_args(args)
    activation_validator = _activation_validator_from_args(args)
    replay_translate_fn = _replay_translate_fn_from_args(args)
    result = run_offline_adversarial_learning(
        args.run_root,
        args.bank,
        args.output_bank,
        args.report,
        admit_trace_candidates=args.admit_trace_candidates,
        admit_composition_failure_candidates=args.admit_composition_failure_candidates,
        run_synthetic_challenges=args.run_synthetic_challenges,
        run_compositional_challenges=args.run_compositional_challenges,
        challenger_agent=challenger_agent,
        refiner_agent=refiner_agent,
        oracle_validator=oracle_validator,
        activation_validator=activation_validator,
        trace_synthesizer=trace_synthesizer,
        trace_contradiction_reviewer=trace_contradiction_reviewer,
        composition_relation_reviewer=composition_relation_reviewer,
        replay_translate_fn=replay_translate_fn,
    )
    print(
        f"challenges={result['challenges']} "
        f"accepted_updates={result['accepted_updates']} "
        f"admitted_candidates={result['admitted_candidates']} "
        f"relation_updates={result['relation_updates']} "
        f"synthetic_challenges={result['synthetic_challenges']} "
        f"compositional_challenges={result['compositional_challenges']} "
        f"output_bank={result['output_bank']} "
        f"report={args.report}"
    )


if __name__ == "__main__":
    main()
