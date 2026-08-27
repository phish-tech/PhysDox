# Reproducibility Notes

## What Can Be Reproduced Without API Keys

The separate data archive contains:

- Silver 5k benchmark data.
- Gold 683 benchmark data.
- Main experiment summaries for 48 paper-facing runs.
- Confusion matrices and three-class analyses for all 48 runs.
- Per-item model outputs for all 48 runs, totaling 136,392 rows.
- Exported result tables.

With the software archive and data archive together, reviewers can audit output
completeness, response formatting, metric aggregation, and table exports without
calling any hosted model API.

## What Requires API Keys

Full model reruns require credentials and access for the evaluated hosted models.
The exact access environment used for the submitted experiments may not be
available to all reviewers. Provider-side model revisions can also affect exact
rerun reproducibility.

For this reason, the data archive preserves per-item model outputs and compact
run summaries. These are the authoritative artifacts for reproducing the
paper-facing numbers.

The strict response-format audit reports 9 unparseable individual votes among
149,574 self-consistency samples across four Gold runs and exits with status 1.
Those raw samples are preserved for auditability. The enclosing self-consistency
records contain aggregate predictions, and all 48 paper-facing runs are
complete according to the experiment audit.

## Relevant Commands

```bash
uv run python scripts/audit_experiments.py \
  --root ../data_archive/experiment_outputs/experiments/main_20260510_main

uv run python scripts/audit_response_format.py \
  --root ../data_archive/experiment_outputs/experiments/main_20260510_main

uv run python scripts/export_main_results_table.py \
  --root ../data_archive/experiment_outputs/experiments/main_20260510_main \
  --output ./reproduced_tables
```

## Expected Archive Relationship

The software archive does not include benchmark data or model outputs. This is
intentional: the EMNLP submission system provides separate software and data
uploads. The software archive should be used with the paired data archive:

```text
3919_PhysDox_Benchmarking_LLMs_Data.zip
```

## Privacy And Exclusions

The archives exclude API keys, `.env` files, local checkpoints, local repair
logs, source partner workbooks, and editable review spreadsheets. Public
benchmark JSON files and paper-facing outputs are included in the data archive.
