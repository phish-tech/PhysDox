# PhysDox Software Archive

This camera-ready archive contains the software for evaluating PhysDox, a
benchmark for physical feasibility auditing of physiological sensing protocols.

The software archive is intentionally separated from the data archive:

- Software archive: Python package, command-line entrypoints, audit scripts,
  table exporters, and reproducibility documentation.
- Data archive: Gold/Silver datasets, main experiment summaries, confusion
  matrices, result tables, and per-item model outputs.

Use both archives together to reproduce the paper-facing audits. Full hosted-LLM
reruns require provider credentials and model access, but auditing the released
outputs does not require API keys.

## Contents

- `src/hallucination_benchmark/`: benchmark evaluation package.
- `scripts/`: audit and table export utilities.
- `pyproject.toml`, `uv.lock`: Python package metadata and dependency lock.
- `QUICKSTART.md`: minimal commands for installing and auditing.
- `REPRODUCIBILITY.md`: data/output availability and rerun notes.
- `software_manifest.json`: archive file list and expected paired data archive.

The manifest records the size and SHA-256 digest of every included file. The
archive intentionally excludes credentials, runtime caches, checkpoints, source
workbooks, and editable review files.

## Fast Check

After unpacking this software archive:

```bash
uv sync
uv run benchmark-two-stage-eval --help
uv run python scripts/audit_experiments.py --help
uv run python scripts/audit_response_format.py --help
```

After unpacking the separate data archive next to this software archive, see
`QUICKSTART.md` for audit commands over the released outputs.
