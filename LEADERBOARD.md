# Leaderboard

Stage-1 macro-F1 (%) for `valid` / `minor_issue` / `fatal_violation`, best of four strategies per model.
Numbers come from the EMNLP 2026 paper runs (`main_20260510_main`).

| Model | Best strategy (Gold) | Gold S1 macro-F1 | Silver S1 macro-F1 |
|---|---|---:|---:|
| DeepSeek V4 Flash | Self-Cons | 53.0 | 52.9 |
| Claude Opus 4.7 | Tool-Aug | 52.6 | 61.4 |
| MiMo v2.5 | Tool-Aug | 47.4 | 62.0 |
| MiMo v2.5 Pro | Zero-Shot | 45.6 | 62.4 |
| DeepSeek V4 Pro | Self-Cons | 45.5 | 56.4 |
| GPT-5.5 | Self-Cons | 38.5 | 43.9 |

The Silver column uses each model's best Silver strategy, which can differ from its best Gold strategy.

## Add your model

1. Run `benchmark-two-stage-eval` on Silver (public) and on Gold if you have access.
2. Open a pull request that adds a row above and attaches your `summary.json`.
3. Report the model name, version or date, and the inference settings.
