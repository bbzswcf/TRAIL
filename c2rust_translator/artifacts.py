"""Shared filesystem layout helpers for experiment artifacts."""

from __future__ import annotations

from pathlib import Path


def output_root(repo_root: Path) -> Path:
    return repo_root / "output"


def runs_root(repo_root: Path) -> Path:
    return output_root(repo_root) / "runs"


def logs_root(repo_root: Path) -> Path:
    return repo_root / "logs"


def run_logs_root(repo_root: Path) -> Path:
    return logs_root(repo_root) / "runs"


def default_run_root(repo_root: Path, run_id: str) -> Path:
    return runs_root(repo_root) / run_id


def project_root(run_root: Path, project: str) -> Path:
    return run_root / project


def default_project_root(repo_root: Path, run_id: str, project: str) -> Path:
    return project_root(default_run_root(repo_root, run_id), project)


def legacy_run_root(repo_root: Path, run_id: str) -> Path:
    return output_root(repo_root) / run_id


def legacy_project_root(repo_root: Path, run_id: str, project: str) -> Path:
    return project_root(legacy_run_root(repo_root, run_id), project)


def resolve_run_root(repo_root: Path, run_id: str) -> Path:
    """Return the existing run root for a run id, falling back to the new layout."""
    current = default_run_root(repo_root, run_id)
    if current.exists():
        return current
    legacy = legacy_run_root(repo_root, run_id)
    if legacy.exists():
        return legacy
    return current


def resolve_project_root(repo_root: Path, run_id: str, project: str) -> Path:
    """Return a project's artifact directory, using legacy layout only for existing runs."""
    return project_root(resolve_run_root(repo_root, run_id), project)


def default_full_dataset_log_path(repo_root: Path, run_id: str) -> Path:
    return run_logs_root(repo_root) / f"{run_id}.log"


def resolve_full_dataset_log_path(repo_root: Path, run_id: str, raw_path: str = "") -> Path:
    """Resolve full-run logs into the structured logs/runs tree by default.

    A bare relative filename such as ``trial.log`` and the old-style
    ``logs/trial.log`` spelling are both treated as run logs and written under
    ``logs/runs/``. Relative paths with another explicit parent keep their
    parent so callers can still opt into a specific local location.
    """
    if not raw_path:
        return default_full_dataset_log_path(repo_root, run_id)

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    if len(path.parts) == 1:
        return run_logs_root(repo_root) / path
    if len(path.parts) == 2 and path.parts[0] == "logs":
        return run_logs_root(repo_root) / path.name
    return repo_root / path
