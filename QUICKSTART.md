# Quickstart

These commands audit the released PhysDox software and data without calling any
hosted model API.

## 1. Unpack Archives

Use one working directory containing both submitted archives:

```bash
mkdir physdox_repro
cd physdox_repro
unzip /path/to/3919_PhysDox_Benchmarking_LLMs_Software_camera_ready.zip -d software
unzip /path/to/3919_PhysDox_Benchmarking_LLMs_Data.zip -d data_archive
cd software
```

The expected layout is:

```text
physdox_repro/
├── software/
│   ├── src/
│   ├── scripts/
│   ├── pyproject.toml
│   └── uv.lock
└── data_archive/
    ├── benchmark_data/
    └── experiment_outputs/
```

## 2. Install

```bash
uv sync
```

The package requires Python 3.11 or newer. The submitted `.python-version`
selects Python 3.13 when that interpreter is available.

## 3. Check CLI Entrypoints

```bash
uv run benchmark-two-stage-eval --help
uv run python scripts/audit_experiments.py --help
uv run python scripts/audit_response_format.py --help
uv run python scripts/export_main_results_table.py --help
```

## 4. Audit Released Outputs

```bash
uv run python scripts/audit_experiments.py \
  --root ../data_archive/experiment_outputs/experiments/main_20260510_main

uv run python scripts/audit_response_format.py \
  --root ../data_archive/experiment_outputs/experiments/main_20260510_main
```

The released data archive contains the per-item outputs needed for these audits.
The response-format audit intentionally returns status 1 when it reports known
individual self-consistency sample parsing failures. In the submitted outputs,
9 of 149,574 individual samples are unparseable across four Gold runs; the
self-consistency wrappers and their aggregate predictions remain complete.

## 5. Export Paper-Facing Tables

```bash
uv run python scripts/export_main_results_table.py \
  --root ../data_archive/experiment_outputs/experiments/main_20260510_main \
  --output ./reproduced_tables
```

## 6. Optional Full Model Rerun

Full reruns call hosted LLM APIs and require provider credentials. Create a local
`.env` file or set environment variables for the provider configuration. Example:

```bash
LLM_API_KEY=...
LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
LLM_MODEL_NAME=your-model-name

uv run benchmark-two-stage-eval \
  --data ../data_archive/benchmark_data/data/gold/gold_0510_final_full.json \
  --model custom \
  --strategy zero-shot \
  --concurrency 8 \
  --rpm 60 \
  --retries 5
```

Model-provider behavior, model availability, and API versions can change over
time, so the released per-item outputs are the authoritative artifacts for
auditing the submitted paper results.
