# STATE — S6E10 improvement loop

This file is the single source of truth for the model-improvement loop.
**Re-read after every round.** Update with: current champion, current GATE,
the best blend, the last 10 experiments, and the next 3 ideas.

---

## Mission

Kaggle S6E10, binary `satisfaction`, ROC AUC. Goal OOF AUC ≥ **0.9605**
(stretch 0.9610). Ceiling 0.9620. Anything > 0.963 is a leakage alarm.

## Hardware

Apple Silicon (arm64), **8 CPU cores**, **16 GB RAM**, **no GPU**. A
plain LightGBM 5-fold on full data ≈ 76 s. CatBoost on CPU is 5-10× slower.

## Harness (frozen)

- `folds.csv` — StratifiedKFold(5, shuffle=True, random_state=42), saved
  to disk. Never change after creation.
- `harness.py` — single source of truth for IO, folds, OOF/test artifacts.
- `_tests/test_harness.py` — 7 pytest tests, ~4 s on full data.
- `experiments.csv` — append-only.

## Champion

| round | champion | OOF AUC | blend AUC | GATE |
|---|---|---|---|---|
| 0 | (none yet — first run) | — | — | — |

## Best blend

(none yet)

## Last 10 experiments

(see `experiments.csv`)

## Per-segment AUC (after a NEW CHAMPION)

(populated when a new champion is found)

## Top features by gain (after a NEW CHAMPION)

(populated when a new champion is found)

## Next 3 ideas

1. **exp_B_route**: Flight Distance group features (counts, per-source
   counts, mean/std of ratings+Age, per-source category share). Computed
   on train+test without labels. → `lgbm_full` candidate.
2. **exp_C_te**: target encoding of Flight Distance + 4-way (Flight
   Distance × Class × Type of Travel × Customer Type) smoothed target.
3. **exp_E_lgbm_xgb**: model variety — XGBoost with native categoricals
   on the route-enriched features (gives the blend a non-LGBM member).

## Open questions

(see `QUESTIONS.md`)

## Stop condition

- 8 REJECTED rounds in a row, OR
- 60 rounds total.

We stop only after every item in the EXPERIMENT QUEUE (A→I) has been
tried or explicitly skipped.
