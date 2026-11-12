# Quickstart

These commands audit the released PhysDox outputs without calling any hosted model API.

## 1. Install

The package needs Python 3.11 or newer.

```bash
git clone https://github.com/phish-tech/PhysDox.git
cd PhysDox
uv sync
```

## 2. Download the data

```bash
huggingface-cli download phishphish/PhysDox --repo-type dataset --local-dir ../physdox_data
```

This gives the Silver set, the Gold dev subset, all aggregate results, and the per-item
outputs of the 24 Silver runs. The per-item outputs of the 24 Gold runs contain Gold labels
and are in the gated repository `phishphish/PhysDox-Gold`. After access is granted:

```bash
huggingface-cli download phishphish/PhysDox-Gold --repo-type dataset --local-dir ../physdox_gold
cp -r ../physdox_gold/experiment_outputs/* ../physdox_data/experiment_outputs/
```

Layout after downloading:

```text
physdox_data/
  silver/physdox_silver_5000.json
  gold_dev/physdox_gold_dev_100.json
  experiment_outputs/experiments/main_20260510_main/
    silver5k_twostage_<model>_<strategy>/     # summary, confusion matrices, per-item outputs
    gold683_twostage_<model>_<strategy>/      # summary and confusion matrices (per-item: gated)
    tables/
```

## 3. Audit the released outputs

The experiment audit expects per-item files for all 48 runs, so it needs the Gold outputs too.
Without Gold access, audit only the Silver runs by pointing `--root` at a copy that has only the
`silver5k_*` directories.

```bash
uv run python scripts/audit_experiments.py \
  --root ../physdox_data/experiment_outputs/experiments/main_20260510_main
uv run python scripts/audit_response_format.py \
  --root ../physdox_data/experiment_outputs/experiments/main_20260510_main
```

The response-format audit exits with status 1 on purpose. It reports 9 unparseable individual
votes among 149,574 self-consistency samples across four Gold runs. The aggregate predictions of
those runs are complete.

## 4. Export the paper tables

```bash
uv run python scripts/export_main_results_table.py \
  --root ../physdox_data/experiment_outputs/experiments/main_20260510_main \
  --output ./reproduced_tables
```

## 5. Evaluate your own model

Any OpenAI-compatible endpoint works. Set the credentials in a local `.env` file, which is git-ignored.

```bash
LLM_API_KEY=...
LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
LLM_MODEL_NAME=your-model-name

uv run benchmark-two-stage-eval \
  --data ../physdox_data/silver/physdox_silver_5000.json \
  --model custom \
  --strategy zero-shot \
  --concurrency 8 \
  --rpm 60 \
  --retries 5
```

Strategies are `zero-shot`, `cot`, `self-cons`, and `tool-aug`. Provider behavior changes over
time, so the released per-item outputs are the reference for the paper numbers.
