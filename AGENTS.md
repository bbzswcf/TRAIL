# AGENTS.md

## Runbook

- Use `uv run` for Python scripts and modules. See `README.md` for common commands.
- New translation output is generated under `output/runs/<run_id>/<project>/`,
  where `<run_id>` defaults to a fresh local timestamp (`YYYYmmdd-HHMMSS`).
  Historical `output/<run_id>/<project>/` directories remain readable by
  metrics and repair scripts. Pass `--run-id` to reuse an existing run id.
- Full concurrent runs should use `scripts/full_dataset_run.py --jobs 15`;
  non-rerun projects are scheduled by descending C project code size so larger
  projects start first.

## Architecture

- See `README.md` for the repository file tree.
- See `TRANSLATION_WORKFLOW.md` for module internals and output artifacts.

## Repository Structure

- Treat `AGENTS.md` as the repository agent guide; do not add duplicate
  `Agent.md`, `agent.md`, or similarly named instruction files.
- Keep the top-level layout intentional and small:
  - `c2rust_translator/`: importable Python package and translation pipeline modules.
  - `scripts/`: runnable preprocessing, experiment, metrics, and repair scripts.
  - `tests/`: unit tests for package modules and scripts.
  - `data/`: benchmark inputs and derived metadata.
  - `snapshots/`: versioned rule/config snapshots.
  - `output/`: generated translation artifacts; keep ignored and do not commit run outputs.
  - `logs/`: local run logs and audits; keep ignored and do not commit generated logs.
  - `docs/`: design notes and execution plans.
  - `reference_baseline/`: external baseline submodules or vendored comparison repos.
- Do not introduce a new top-level directory unless the existing directories
  above are a poor fit. If a new top-level directory is necessary, update both
  `README.md` and this section in the same commit.
- When moving, adding, or deleting files, keep related files colocated with
  their owner module and update imports, docs, tests, and ignore rules in the
  same change.
- Keep local caches and generated files out of repository status:
  `__pycache__/`, `.pytest_cache/`, `*.egg-info/`, `run_pids/`, `logs/`,
  `output/`, and local manuscript workspaces should remain ignored or untracked.

## Workflow Invariants

- Keep names, modules, and APIs direct and minimal.
- Avoid compatibility wrappers or extra abstractions unless they are clearly
  necessary.
- Default translator behavior is project-local fail-fast: if one callable
  cannot be translated/fixed, stop that project and move to the next project in
  batch runs. Override with `--continue-on-failure` when needed.
- `cargo check` runs after callable translation; `cargo test` runs after a
  project finishes translation.

## Editing Notes

- Prefer modifying existing files over adding new layers.
- Keep file names, class names, and function names simple and literal.
- After each intentional repository change, create a git commit unless the user
  explicitly asks not to commit yet.
- Do not commit generated logs, run outputs, PID files, or transient caches
  unless the user explicitly asks for a selected artifact.
- If you change the translation flow, update `TRANSLATION_WORKFLOW.md` and
  `README.md` when user-facing behavior changes.
