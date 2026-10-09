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
- `_tests/test_harness.py` — 8 pytest tests, ~4 s on full data.
- `experiments.csv` — append-only.

## Champion

| round | champion | OOF AUC | Δ vs prev | blend AUC | GATE |
|---|---|---|---|---|---|
| 0 (sanity) | `lgbm_raw` (raw features only) | 0.958810 | — | — | 0.000064 |
| 1 (B) | `lgbm_full` (route features) | **0.960421** | **+0.001611** | — | 0.000064 |
| 2 (C) | `lgbm_full` (still champion) | 0.960421 | 0 | **0.960685** (blend) | 0.000064 |

We're at **0.960685** in the blend, which clears the 0.9605 goal. Single
model is at 0.960421 (just below 0.9605); the next model-variety round
(E) should close the gap on its own.

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

| members | C | OOF AUC | notes |
|---|---|---|---|
| `lgbm_raw` + `lgbm_full` | 0.2 | 0.960555 | 2-OOF stack |
| `lgbm_raw` + `lgbm_full` + `lgbm_full_te` | **0.005** | **0.960685** | 3-OOF stack (current best) |

The TE model is slightly worse on its own but adds diversity; the
blend gains +0.000130 by including it.

## Per-segment AUC at champion (`lgbm_full`)

### 1D

| segment | n | AUC |
|---|---|---|
| Type of Travel = Business | 497,441 | 0.9571 |
| Type of Travel = Personal  | 202,194 | **0.8319** |
| Class = Business           | 342,212 | 0.9402 |
| Class = Eco                | 327,404 | 0.9043 |
| Class = Eco Plus           |  30,019 | 0.9320 |
| Customer Type = Loyal      | 576,990 | 0.9609 |
| Customer Type = disloyal   | 122,645 | **0.9240** |
| Age 50-85 (oldest)         | 161,837 | 0.9624 (strongest) |
| Age 6-27 (youngest)        | 176,127 | 0.9409 |

### 3D (Class × Type × Customer) — 5 weakest

| Class | Type | Customer | n | AUC |
|---|---|---|---|---|
| Business | Personal | Loyal     | 3,939  | **0.8148** |
| Eco      | Personal | Loyal     | 185,691 | 0.8323 |
| Eco Plus | Personal | Loyal     | 12,459  | 0.8328 |
| Eco Plus | Business | disloyal  | 2,574  | 0.8685 |
| Eco      | Business | disloyal  | 76,070 | 0.8944 |

**Insight**: "Personal Travel" is the single biggest weakness, regardless
of class. Personal-Travel customers are 4x more likely to be unsatisfied
but the model only achieves 0.83 AUC there. The weakest *large* segment
is Eco × Personal × Loyal (n=185k, AUC=0.83).

## Top 15 features by gain (`lgbm_full`)

| feature | gain |
|---|---|
| Online boarding       | 2,071,879 |
| Inflight wifi service | 590,036 |
| Type of Travel        | 481,930 |
| Class                 | 294,952 |
| Inflight entertainment| 240,610 |
| Customer Type         | 182,734 |
| Checkin service       | 102,176 |
| Baggage handling      | 82,987 |
| Ease of Online booking| 80,358 |
| On-board service      | 79,213 |
| Seat comfort          | 77,510 |
| Cleanliness           | 46,916 |
| Age                   | 43,402 |
| Leg room service      | 35,563 |
| Gate location         | 31,804 |

Note: route features did NOT make the top 15. They support the model but
the raw ratings dominate. **Online boarding** is by far the strongest
predictor (3.5× the second-place).

## Last 10 experiments

(see `experiments.csv`)

## Next 3 ideas

1. **exp_D_small**: small interaction features (log delays, rating
   aggregates, Type×Class×Customer cross). The TE didn't add signal at
   the 4-way level because the route features already capture much of
   the same info. Small hand-engineered features might add NEW signal.
2. **exp_E_xgb**: XGBoost on the route-enriched features. Adds a
   non-LGBM member to the blend (highest-leverage single change for
   ensemble diversity). CPU-bound: ~3-5 min per fold.
3. **exp_E_cat**: CatBoost on the route-enriched features. CPU is
   5-10× slower than LGBM; will run a depth-4 variant first.

## Open questions

(see `QUESTIONS.md`)

## Stop condition

- 8 REJECTED rounds in a row, OR
- 60 rounds total.

We stop only after every item in the EXPERIMENT QUEUE (A→I) has been
tried or explicitly skipped.
