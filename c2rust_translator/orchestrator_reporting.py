"""
翻译流程报告与历史记录落盘工具。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple


ADVERSARIAL_LEARNING_OPTION_DEFAULTS = {
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
    "pre_translation_refinement_rounds": 0,
    "pre_translation_refinement_writeback": False,
}


def _normalize_adversarial_learning_options(options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw_options = options or {}
    normalized: Dict[str, Any] = {}
    for key, default in ADVERSARIAL_LEARNING_OPTION_DEFAULTS.items():
        raw_value = raw_options.get(key, default)
        if key == "pre_translation_refinement_rounds":
            try:
                normalized[key] = max(int(raw_value), 0)
            except (TypeError, ValueError):
                normalized[key] = int(default)
        else:
            normalized[key] = bool(raw_value)
    return normalized


def generate_translation_report(
    *,
    project_name: str,
    metadata_path: str,
    output_dir: Path,
    max_fix_attempts: int,
    dep_analysis: Optional[Dict[str, Any]],
    translated_modules: Set[Tuple[str, str, str]],
    translation_log: list,
    translation_history: Dict[Tuple[str, str, str], Dict[str, Any]],
    success: bool,
    alignment_stats: Optional[Dict[str, Any]] = None,
    max_test_fix_attempts: int = 3,
    rules_enabled: bool = True,
    rule_limit: int = 5,
    rules_path: str = "",
    rules_json_bank_path: str = "",
    rule_interaction_scorer: str = "",
    rule_retrieval_backend_preference: str = "",
    rule_retrieval_index_cache_path: str = "",
    rule_retrieval_backend: str = "",
    rule_retrieval_status: str = "",
    rule_retrieval_cache_status: str = "",
    experiment_profile: str = "custom",
    experiment_ablation_label: str = "custom",
    rule_stage: str = "custom",
    rule_mining: bool = True,
    single_rule_adversarial_learning: bool = False,
    compositional_rule_adversarial_learning: bool = False,
    adversarial_rounds: int = 0,
    online_adversarial_rules: bool = False,
    adversarial_rule_bank_path: str = "",
    adversarial_rule_output_bank_path: str = "",
    adversarial_rule_report_path: str = "",
    adversarial_learning_options: Optional[Dict[str, Any]] = None,
    model_display_name: str = "",
    test_fix_history: Optional[list] = None,
    error: Optional[str] = None,
) -> Path:
    """生成翻译报告并落盘。"""
    total_modules = dep_analysis["total_modules"] if dep_analysis else 0
    translated_count = len(translated_modules)

    report = {
        "project_name": project_name,
        "success": success,
        "config": {
            "metadata_path": str(metadata_path),
            "output_dir": str(output_dir),
            "max_fix_attempts": max_fix_attempts,
            "max_test_fix_attempts": max_test_fix_attempts,
            "rules_enabled": rules_enabled,
            "rule_limit": rule_limit,
            "rules_path": rules_path,
            "rules_json_bank_path": rules_json_bank_path,
            "rule_interaction_scorer": rule_interaction_scorer,
            "rule_retrieval_backend_preference": rule_retrieval_backend_preference,
            "rule_retrieval_index_cache_path": rule_retrieval_index_cache_path,
            "rule_retrieval_backend": rule_retrieval_backend,
            "rule_retrieval_status": rule_retrieval_status,
            "rule_retrieval_cache_status": rule_retrieval_cache_status,
            "experiment_profile": experiment_profile,
            "experiment_ablation_label": experiment_ablation_label,
            "rule_stage": rule_stage,
            "rule_mining": rule_mining,
            "single_rule_adversarial_learning": single_rule_adversarial_learning,
            "compositional_rule_adversarial_learning": compositional_rule_adversarial_learning,
            "adversarial_rounds": adversarial_rounds,
            "online_adversarial_rules": online_adversarial_rules,
            "adversarial_rule_bank_path": adversarial_rule_bank_path,
            "adversarial_rule_output_bank_path": adversarial_rule_output_bank_path,
            "adversarial_rule_report_path": adversarial_rule_report_path,
            "adversarial_learning_options": _normalize_adversarial_learning_options(
                adversarial_learning_options
            ),
            "model_display_name": model_display_name,
        },
        "statistics": {
            "total_modules": total_modules,
            "translated_modules": translated_count,
            "failed_modules": max(total_modules - translated_count, 0),
        },
        "alignment_statistics": alignment_stats or {},
        "test_fix_attempts": test_fix_history or [],
        "llm_statistics": _token_totals(translation_history, alignment_stats, test_fix_history),
        "adversarial_refinement_statistics": _adversarial_refinement_totals(
            translation_history,
            test_fix_history,
        ),
        "translation_log": translation_log,
    }

    if error:
        report["error"] = error

    report_path = output_dir / "translation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  Translation report saved to: {report_path}")
    return report_path


def _token_totals(
    translation_history: Dict[Tuple[str, str, str], Dict[str, Any]],
    alignment_stats: Optional[Dict[str, Any]] = None,
    test_fix_history: Optional[list] = None,
) -> Dict[str, Any]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    llm_calls = 0
    llm_seconds = 0.0

    if alignment_stats:
        llm_calls += alignment_stats.get("alignment_llm_calls") or 0
        llm_seconds += alignment_stats.get("elapsed_seconds") or 0.0
        prompt_tokens += alignment_stats.get("prompt_tokens") or 0
        completion_tokens += alignment_stats.get("completion_tokens") or 0
        total_tokens += alignment_stats.get("total_tokens") or 0

    for history in translation_history.values():
        recorded_calls = history.get("llm_calls") or []
        if recorded_calls:
            for call in recorded_calls:
                usage = call.get("usage") or {}
                llm_calls += 1
                llm_seconds += call.get("elapsed_seconds") or 0.0
                prompt_tokens += usage.get("prompt_tokens") or 0
                completion_tokens += usage.get("completion_tokens") or 0
                total_tokens += usage.get("total_tokens") or 0
            continue

        initial = history.get("initial_translation_stats") or {}
        if initial:
            llm_calls += 1
            llm_seconds += initial.get("elapsed_seconds") or 0.0
            prompt_tokens += initial.get("prompt_tokens") or 0
            completion_tokens += initial.get("completion_tokens") or 0
            total_tokens += initial.get("total_tokens") or 0

        for attempt in history.get("compilation_attempts", []):
            stats = attempt.get("llm_stats") or {}
            if not stats:
                continue
            llm_calls += 1
            llm_seconds += stats.get("elapsed_seconds") or 0.0
            prompt_tokens += stats.get("prompt_tokens") or 0
            completion_tokens += stats.get("completion_tokens") or 0
            total_tokens += stats.get("total_tokens") or 0

    for attempt in test_fix_history or []:
        stats = attempt.get("llm_stats") or {}
        if not stats:
            continue
        llm_calls += 1
        llm_seconds += stats.get("elapsed_seconds") or 0.0
        prompt_tokens += stats.get("prompt_tokens") or 0
        completion_tokens += stats.get("completion_tokens") or 0
        total_tokens += stats.get("total_tokens") or 0

    return {
        "llm_calls": llm_calls,
        "llm_seconds": llm_seconds,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _adversarial_refinement_totals(
    translation_history: Dict[Tuple[str, str, str], Dict[str, Any]],
    test_fix_history: Optional[list] = None,
) -> Dict[str, Any]:
    total = 0
    accepted = 0
    writebacks = 0
    writeback_errors = 0
    writeback_paths: set[str] = set()

    for item in _iter_selected_rules(translation_history, test_fix_history):
        refinement = item.get("pre_translation_refinement") if isinstance(item, dict) else None
        if not isinstance(refinement, dict):
            continue
        total += 1
        if bool(refinement.get("accepted", False)):
            accepted += 1
        writeback_status = str(refinement.get("writeback", "") or "")
        if writeback_status == "updated":
            writebacks += 1
            path = str(refinement.get("writeback_path", "") or "")
            if path:
                writeback_paths.add(path)
        elif writeback_status == "error":
            writeback_errors += 1

    return {
        "pre_translation_refinements": total,
        "accepted_pre_translation_refinements": accepted,
        "pre_translation_refinement_writebacks": writebacks,
        "pre_translation_refinement_writeback_errors": writeback_errors,
        "pre_translation_refinement_writeback_paths": sorted(writeback_paths),
    }


def _iter_selected_rules(
    translation_history: Dict[Tuple[str, str, str], Dict[str, Any]],
    test_fix_history: Optional[list] = None,
):
    for history in translation_history.values():
        for item in history.get("selected_rules") or []:
            yield item
        for attempt in history.get("compilation_attempts", []) or []:
            for item in attempt.get("selected_rules") or []:
                yield item

    for attempt in test_fix_history or []:
        for item in attempt.get("selected_rules") or []:
            yield item


def save_translation_history_artifacts(
    *,
    project_name: str,
    output_dir: Path,
    translation_history: Dict[Tuple[str, str, str], Dict[str, Any]],
) -> Tuple[Path, Path, Path]:
    """保存完整历史、prompt工件和文本摘要。"""
    # 转换历史记录为可序列化的格式
    serializable_history: Dict[str, Dict[str, Any]] = {}
    for module_id, history in translation_history.items():
        key = f"{module_id[0]}::{module_id[1]}::{module_id[2]}"
        normalized_history = dict(history)
        # 保证prompt字段在JSON中始终存在，便于稳定消费
        normalized_history.setdefault("system_prompt", None)
        normalized_history.setdefault("initial_translation_prompt", None)
        serializable_history[key] = normalized_history

    history_path = output_dir / "translation_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(serializable_history, f, indent=2, ensure_ascii=False)
    print(f"  Translation history saved to: {history_path}")

    prompts_artifact = {}
    for key, history in serializable_history.items():
        prompts_artifact[key] = {
            "system_prompt": history.get("system_prompt"),
            "initial_translation_prompt": history.get("initial_translation_prompt"),
            "llm_calls": [
                {
                    "sequence": call.get("sequence"),
                    "phase": call.get("phase"),
                    "callable_id": call.get("callable_id"),
                    "prompt": call.get("prompt"),
                }
                for call in history.get("llm_calls", [])
            ],
        }
    prompts_path = output_dir / "translation_prompts.json"
    with open(prompts_path, "w", encoding="utf-8") as f:
        json.dump(prompts_artifact, f, indent=2, ensure_ascii=False)
    print(f"  Translation prompts saved to: {prompts_path}")

    summary_path = output_dir / "translation_history_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Translation History Summary for {project_name}\n")
        f.write(f"{'=' * 80}\n\n")

        for module_id, history in translation_history.items():
            file_path, module_type, module_name = module_id
            f.write(f"Module: {module_name} ({module_type})\n")
            f.write(f"File: {file_path}\n")
            f.write(f"Success: {history.get('success', False)}\n")
            f.write(f"Duration: {history.get('duration', 0):.2f}s\n")
            if history.get("used_unimplemented_fallback"):
                fallback = history.get("fallback") or {}
                f.write(f"Unimplemented fallback: {fallback.get('status', 'unknown')}\n")
            if history.get("llm_calls"):
                f.write(f"LLM calls recorded: {len(history['llm_calls'])}\n")

            if history.get("error"):
                f.write(f"Error: {history['error']}\n")

            initial_prompt = history.get("initial_translation_prompt")
            system_prompt = history.get("system_prompt")
            f.write(f"Initial prompt recorded: {bool(initial_prompt)}\n")
            if initial_prompt:
                f.write(f"Initial prompt chars: {len(initial_prompt)}\n")
            f.write(f"System prompt recorded: {bool(system_prompt)}\n")
            initial_stats = history.get("initial_translation_stats") or {}
            if initial_stats:
                f.write(
                    "Initial translation LLM stats: "
                    f"{initial_stats.get('elapsed_seconds', 0):.2f}s, "
                    f"prompt={initial_stats.get('prompt_tokens')}, "
                    f"completion={initial_stats.get('completion_tokens')}, "
                    f"total={initial_stats.get('total_tokens')}\n"
                )

            attempts = history.get("compilation_attempts", [])
            if attempts:
                f.write(f"Compilation attempts: {len(attempts)}\n")
                for i, attempt in enumerate(attempts, 1):
                    f.write(f"  Attempt {i}: {attempt.get('status', 'unknown')}\n")
                    if attempt.get("errors"):
                        f.write(f"    Errors: {len(attempt['errors'])}\n")
                    llm_stats = attempt.get("llm_stats") or {}
                    if llm_stats:
                        f.write(
                            "    LLM stats: "
                            f"{llm_stats.get('elapsed_seconds', 0):.2f}s, "
                            f"prompt={llm_stats.get('prompt_tokens')}, "
                            f"completion={llm_stats.get('completion_tokens')}, "
                            f"total={llm_stats.get('total_tokens')}\n"
                        )

            f.write(f"\n{'-' * 80}\n\n")

    print(f"  Translation history summary saved to: {summary_path}")
    return history_path, prompts_path, summary_path
