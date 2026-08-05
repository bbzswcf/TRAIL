from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentProfile:
    name: str
    display_label: str
    insights_enabled: bool
    default_insights_path: str
    insight_stage: str
    insight_mining: bool
    single_insight_adversarial_learning: bool
    compositional_insight_adversarial_learning: bool
    default_adversarial_rounds: int


_PROFILES: dict[str, ExperimentProfile] = {
    "full": ExperimentProfile(
        name="full",
        display_label="TRAIL",
        insights_enabled=True,
        default_insights_path="",
        insight_stage="multi",
        insight_mining=True,
        single_insight_adversarial_learning=True,
        compositional_insight_adversarial_learning=True,
        default_adversarial_rounds=5,
    ),
    "without_compositional": ExperimentProfile(
        name="without_compositional",
        display_label="w/o Compositional Insight Adversarial Learning",
        insights_enabled=True,
        default_insights_path="",
        insight_stage="single",
        insight_mining=True,
        single_insight_adversarial_learning=True,
        compositional_insight_adversarial_learning=False,
        default_adversarial_rounds=5,
    ),
    "without_single": ExperimentProfile(
        name="without_single",
        display_label="w/o Single Insight Adversarial Learning",
        insights_enabled=True,
        default_insights_path="",
        insight_stage="default",
        insight_mining=True,
        single_insight_adversarial_learning=False,
        compositional_insight_adversarial_learning=False,
        default_adversarial_rounds=0,
    ),
    "without_insight_mining": ExperimentProfile(
        name="without_insight_mining",
        display_label="w/o Insight Mining",
        insights_enabled=False,
        default_insights_path="",
        insight_stage="off",
        insight_mining=False,
        single_insight_adversarial_learning=False,
        compositional_insight_adversarial_learning=False,
        default_adversarial_rounds=0,
    ),
}

_ALIASES: dict[str, str] = {
    "": "full",
    "trail": "full",
    "multi": "full",
    "no-compositional": "without_compositional",
    "without-compositional": "without_compositional",
    "w/o-compositional": "without_compositional",
    "single": "without_compositional",
    "no-single": "without_single",
    "without-single": "without_single",
    "w/o-single": "without_single",
    "default": "without_single",
    "mined": "without_single",
    "no-mining": "without_insight_mining",
    "without-mining": "without_insight_mining",
    "without-insight-mining": "without_insight_mining",
    "w/o-insight-mining": "without_insight_mining",
    "baseline": "without_insight_mining",
    "off": "without_insight_mining",
}

_DISPLAY_NAMES: dict[str, str] = {
    "gpt-5.4-mini-2026-03-17": "GPT-5.4-mini",
    "gpt-5.4-mini": "GPT-5.4-mini",
    "kimi-k2.5": "Kimi-K2.5",
    "ali-deepseek-v4-flash": "DeepSeek-V4-Flash",
    "deepseek-v4-flash": "DeepSeek-V4-Flash",
}

_MODEL_DEFAULT_INSIGHT_BANKS: dict[str, str] = {
    "gpt-5.4-mini-2026-03-17": "snapshots/insights/gpt_final_insight_bank.json",
    "gpt-5.4-mini": "snapshots/insights/gpt_final_insight_bank.json",
    "kimi-k2.5": "snapshots/insights/kimi_final_insight_bank.json",
    "ali-deepseek-v4-flash": "snapshots/insights/deepseek_final_insight_bank.json",
    "deepseek-v4-flash": "snapshots/insights/deepseek_final_insight_bank.json",
}


def resolve_experiment_profile(name: str | None) -> ExperimentProfile:
    key = (name or "").strip().lower().replace("_", "-")
    canonical = _ALIASES.get(key, key.replace("-", "_"))
    try:
        return _PROFILES[canonical]
    except KeyError as exc:
        choices = ", ".join(profile_choices())
        raise ValueError(f"Unknown experiment profile {name!r}. Choose one of: {choices}") from exc


def profile_choices() -> list[str]:
    canonical = [name.replace("_", "-") for name in _PROFILES]
    aliases = sorted(alias for alias in _ALIASES if alias)
    return sorted(set([*canonical, *aliases]))


def model_display_name(model: str) -> str:
    normalized = model.strip().lower()
    return _DISPLAY_NAMES.get(normalized, model)


def model_default_insight_bank_path(model: str) -> str:
    normalized = model.strip().lower()
    return _MODEL_DEFAULT_INSIGHT_BANKS.get(normalized, "")
