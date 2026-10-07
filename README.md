<div align="center">

# Translator vs. Challenger: Adversarial Agentic Self-Improvement for C-to-Rust Translation

</div>

TRAIL is an adversarial agentic learning framework for project-level C-to-Rust
translation with single-rule and compositional rule adversarial learning.

## Project Directory Structure

```text
.
├── c2rust_translator/                  # Main translation package
│   ├── translator.py                   # Translation and repair orchestration
│   ├── translation_runner.py           # Single-project and multi-project CLI
│   ├── alignment_planner.py            # C/Rust callable binding planner
│   ├── rules.py                     # Built-in structured translation rules
│   ├── adversarial_rules.py         # JSON rule banks and adversarial updates
│   ├── rbench_interface.py             # Rust interface scanning
│   ├── rust_skeleton_generator.py      # Output crate generation
│   ├── compilation_validator.py        # cargo check wrapper
│   ├── test_validator.py               # cargo test wrapper
│   ├── orchestrator_reporting.py       # Report and history serialization
│   └── prompts.py                      # Translation and repair prompts
├── scripts/                            # Preprocessing, experiment, metrics, and repair tools
├── tests/                              # Unit and regression tests
├── data/                               # Benchmark inputs and extracted C metadata
└── snapshots/rules/                 # Final JSON rule banks for GPT, DeepSeek, and Kimi
```

## Introduction

TRAIL translates C projects into Rust by pairing a translator agent with
challenger-style adversarial learning over reusable single-rule and
compositional rule records. Individual rules capture focused translation
patterns or failure modes, while rule compositions combine multiple
records when a callable requires coordinated reasoning across C semantics, Rust
interfaces, ownership, and validation feedback. The pipeline consists of four
stages:

- It first preprocesses benchmark projects and extracts C metadata for
  functions, structs, enums, typedefs, globals, and build context.
- It scans Rust interface crates, binds public Rust callables to relevant C
  functions or file context, and creates a Rust output skeleton.
- It incrementally translates callables with single-rule retrieval and
  compositional rule guidance, runs `cargo check` after each callable, and
  repairs compile failures.
- It runs project-level tests, applies bounded test repair, records translation
  traces, and updates JSON rule banks through adversarial challenge
  workflows that mine stronger individual rules and rule compositions.

## Getting Started

### Prerequisites

1. Clone the repository:

   ```bash
   git clone https://anonymous.4open.science/r/anonymous-artifact-B2C7213dvhw3 TRAIL
   cd TRAIL
   ```

2. Install system dependencies:

   - Python `>=3.10`
   - Rust toolchain with `cargo`
   - C/C++ toolchain with `gcc` or `clang`
   - `clang-format`

### Installation

Install Python dependencies with `uv`:

```bash
uv sync
```

Configure LLM credentials through environment variables or a local `.env` file.
Common variables include `OPENAI_API_KEY`, `LLM_API_URL`, `LLM_MODEL`,
`LLM_SESSION_ID`, and `LLM_LOGID`.
The checked-in defaults use the public OpenAI and Google API endpoints; override
them through `LLM_API_URL` or `LLM_BASE_URL` when using another provider.

## Usage

### 1. Metadata Extraction

Run preprocessing and C metadata extraction before translation:

```bash
uv run scripts/preprocess.py
uv run scripts/c_metadata.py
```

List available projects:

```bash
uv run -m c2rust_translator.translation_runner --list-projects
```

### 2. Translation

Translate a single project:

```bash
uv run -m c2rust_translator.translation_runner leftpad
```

Translate multiple projects concurrently:

```bash
uv run -m c2rust_translator.translation_runner --jobs 4 leftpad morton geofence
```

Generated outputs are written under `output/runs/<run_id>/<project>/`. Pass
`--run-id` to reuse an existing run directory.

### 3. Adversarial Rule Update

Run the adversarial rule update workflow over an existing output tree:

```bash
uv run python scripts/adversarial_rule_run.py output/runs/<run_id> \
  --bank path/to/rule_bank.json \
  --output-bank output/runs/<run_id>/adversarial_rules.updated.json \
  --report output/runs/<run_id>/adversarial_rule_report.json
```

### 4. Rule Banks

The repository publishes final rule banks in the JSON bank format used by
the experiment workflow:

| Model | Bank |
| --- | --- |
| GPT | `snapshots/rules/gpt_final_rule_bank.json` |
| DeepSeek | `snapshots/rules/deepseek_final_rule_bank.json` |
| Kimi | `snapshots/rules/kimi_final_rule_bank.json` |

Named experiment profiles copy the matching model bank into the run directory
as `rules.active.json` before prompt retrieval or online adversarial updates.
Use `--rules-json-bank` to run from a specific bank:

```bash
uv run -m c2rust_translator.translation_runner leftpad \
  --rules-json-bank snapshots/rules/gpt_final_rule_bank.json
```

If no profile or JSON bank is selected, the runner falls back to the built-in
structured records in `c2rust_translator/rules.py`. Local Python snapshot
files can still be used with `--rules-path`, but they are ignored by git.

### 5. Full Experiments

Run a full concurrent experiment:

```bash
uv run python scripts/full_dataset_run.py --jobs 15 --experiment-profile full
```

Useful options:

- `--experiment-profile full|no-compositional|no-single|no-mining`
- `--rules-mode default|off`
- `--rule-limit N`
- `--rule-retrieval-backend auto|keyword|embedding`
- `--max-fix-attempts N`
- `--max-test-fix-attempts N`
- `--continue-on-failure`

### 6. Datasets

TRAIL uses benchmark datasets for project-level C-to-Rust translation:

- CRUST-Bench is included as `data/project/CRUST_bench.zip`. The extracted
  dataset is organized under `data/project/CBench/` for C benchmark projects
  and `data/project/RBench/` for the corresponding Rust interface crates. The
  archive contains source artifacts only; nested VCS metadata and compiled
  build products are excluded.
- The SmartC2Rust benchmark dataset is not vendored in this repository. It is
  available at: https://github.com/momo-trip/SmartC2Rust/tree/main/benchmark

### 7. Metrics

Summarize compile/test status for a run:

```bash
uv run python scripts/experiment_metrics.py <run_id>
```

Rerun project-level test repair for selected projects from an existing run:

```bash
uv run python scripts/test_repair_existing_run.py <run_id> --projects <name>
```
