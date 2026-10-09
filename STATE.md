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
| 0 (sanity) | `lgbm_raw` (raw features only) | **0.958810** | — | **0.000064** |
| 1 (B) | TBD | TBD | TBD | 0.000064 |

## STEP 0 diagnostics

- **Sanity `lgbm_raw`** (full 5-fold): OOF AUC = **0.958810**, runtime
  1.3 min, per-fold = [0.9589, 0.9578, 0.9595, 0.9588, 0.9592].
  Matches the user's expected ≈ 0.9588 — harness is correct.
- **Noise check** (3 seeds, 42 / 53 / 64): OOF AUCs = 0.958810,
  0.958873, 0.958834. Stdev = 0.000032. **GATE = 0.000064**.
- **Adversarial validation** AUC = **0.4995** — train and test are
  indistinguishable, folds are perfectly trustworthy. No correction
  needed for OOF→public-LB gap beyond the usual 0.0003-0.0006.

## Best blend

(none yet — only 1 OOF; nested-CV stacker with 1 OOF is undefined)

## Last 10 experiments

(see `experiments.csv`)

### Per-segment AUC at sanity baseline

(0.9588 OOF — overall; per-segment analysis deferred until we have a
non-trivial model)

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
