# Quickstart

## 1. Install

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/phish-tech/PhysDox.git
cd PhysDox
uv sync
```

## 2. Check your setup without an API key

```bash
uv run python examples/mock_llm_server.py &      # or run it in a second terminal
export LLM_API_KEY=dummy LLM_BASE_URL=http://127.0.0.1:8765/v1 LLM_MODEL_NAME=mock
uv run physdox-eval --data gold_dev --model custom --stage1-only
```

The mock model answers at random. The scores mean nothing, but a result table means the data download,
prompting, parsing, and metrics all work. PowerShell users: `$env:LLM_API_KEY="dummy"`, and so on.

## 3. Evaluate a real model

Any OpenAI-compatible endpoint works. Put the credentials in the environment or in a local `.env`
file (git-ignored).

```bash
export LLM_API_KEY=sk-...  LLM_BASE_URL=https://your-endpoint/v1  LLM_MODEL_NAME=your-model

uv run physdox-eval --data gold_dev --model custom --stage1-only
uv run physdox-eval --data silver --model custom --max-items 500 --sample-seed 0
uv run physdox-eval --data silver --model custom --strategy cot --concurrency 8 --rpm 60
```

- `--data`: `silver`, `gold_dev`, `gold` (gated, set `HF_TOKEN`), or a JSON path/URL. Files are cached in `~/.cache/physdox`.
- `--strategy`: `zero-shot`, `cot`, `self-cons`, `tool-aug`.
- Without `--stage1-only`, true fatal items also go through Stage 2 diagnosis.
- Output goes to `experiments/<run>/` (or `--output-dir`). To make an interrupted run resumable, pass
  `--output-dir` and repeat the same command. If your provider returns 429 or 502 errors, lower `--concurrency` and `--rpm`.

## 4. Audit the released outputs

```bash
hf download phishphish/PhysDox --repo-type dataset --local-dir ../physdox_data
uv run python scripts/audit_experiments.py \
  --root ../physdox_data/experiment_outputs/experiments/main_20260510_main
uv run python scripts/audit_response_format.py \
  --root ../physdox_data/experiment_outputs/experiments/main_20260510_main
```

The public repository holds the per-item outputs of the 24 Silver runs. The 24 Gold runs' per-item files
contain Gold labels and are in the gated repository `phishphish/PhysDox-Gold`. Without them the audit shows
0 items for the Gold runs. After access is granted:

```bash
hf download phishphish/PhysDox-Gold --repo-type dataset --local-dir ../physdox_gold
cp -r ../physdox_gold/experiment_outputs/* ../physdox_data/experiment_outputs/
```

The response-format audit exits with status 1 on purpose. It reports 9 unparseable individual votes among
149,574 self-consistency samples across four Gold runs. The aggregate predictions of those runs are complete.

## 5. Tables and figures

The ready-made paper tables are in `physdox_data/experiment_outputs/experiments/main_20260510_main/tables/`.
To rebuild them from raw outputs (needs the Gold per-item files):

```bash
uv run python scripts/export_main_results_table.py \
  --root ../physdox_data/experiment_outputs/experiments/main_20260510_main \
  --output ./reproduced_tables
```

To redraw the README charts from the public tables:

```bash
uv run --extra plots python scripts/make_result_figures.py
```

Provider behavior changes over time, so the released per-item outputs are the reference for the paper numbers.
