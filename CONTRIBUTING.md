# Contributing

This repository is intended to support a paper-facing benchmark release.
Contributions should keep the benchmark reproducible and avoid committing local
runtime artifacts.

## Development

Install dependencies:

```bash
uv sync
```

Run release checks before opening a pull request:

```bash
uv run python scripts/audit_release.py
uv run python scripts/audit_experiments.py --root experiments/main_20260510_main
```

## Data and Artifacts

Do not commit API keys, `.env`, raw provider responses, run logs, checkpoints,
review spreadsheets, or intermediate partner workbooks. Large raw outputs should
be published through the external artifact bundle documented in
`docs/release_manifest.md`.

## Pull Requests

For code changes, include a short description, the commands you ran, and any
dataset or result files affected. For benchmark data changes, include the
generation or filtering procedure and updated manifests.
