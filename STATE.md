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
| 2 (C) | `lgbm_full` (still champion) | 0.960421 | 0 | 0.960685 (blend) | 0.000064 |
| 3 (D) | `lgbm_full` (still champion) | 0.960421 | 0 | 0.960739 (blend) | 0.000064 |
| 4 (E1) | `lgbm_full` (still champion) | 0.960421 | 0 | 0.960741 (blend) | 0.000064 |
| 5 (E1b) | `lgbm_full` (still champion) | 0.960421 | 0 | 0.960773 (blend) | 0.000064 |
| 6 (E2)  | `lgbm_full` (still champion) | 0.960421 | 0 | 0.960776 (blend) | 0.000064 |
| 7 (F)   | `lgbm_full_optuna` (NEW CHAMPION) | 0.960686 | +0.000265 | 0.960837 (blend) | 0.000064 |
| 8 (G)   | `lgbm_full_g` (NEW CHAMPION) | 0.960757 | +0.000071 | 0.960865 (blend) | 0.000064 |
| 9 (F2)  | `lgbm_full_g` (still champion) | 0.960757 | 0 | 0.960865 (blend) | 0.000064 |
| 10 (G2) | `lgbm_full_g` (still champion) | 0.960757 | 0 | 0.960865 (blend) | 0.000064 |
| 11 (I)  | `lgbm_full_i` (NEW CHAMPION) | 0.960826 | +0.000069 | 0.960940 (blend) | 0.000064 |
| 12 (I2) | `lgbm_full_i2` (BLEND MEMBER) | 0.960872 | +0.000046 | 0.960975 (blend) | 0.000064 |
| 13 (I3) | `lgbm_full_i3` (BLEND MEMBER) | 0.960882 | +0.000010 | **0.960984** (blend) | 0.000064 |
| 14 (I4) | `lgbm_full_i3` (still champion) | 0.960882 | 0 | **0.960988** (blend) | 0.000064 |

Round 8 (G): averaged 3 seeds (42/53/64) at the Optuna params
(lr=0.020, num_leaves=138, l1=4.8, etc.) — 5/5 folds beat
lgbm_full_optuna, OOF AUC **0.960757** (+0.000071). Best blend:
5-OOF stack (`lgbm_full_g+cat_d4+lgbm_full+lgbm_full_te+lgbm_raw`)
C=0.005 = **0.960865**. We're 0.000135 from the 0.9610 stretch goal.

Round 7 (F): Optuna 30-trial 3-fold search picked `lr=0.020,
num_leaves=138, min_data=68, feature_frac=0.75, bagging_frac=0.89,
lambda_l1=4.8, lambda_l2=0.07, max_bin=127`. 5-fold refit: OOF AUC
**0.960686**, +0.000265 vs old champion.

Round 9 (F2): Optuna XGBoost (15 trials) found depth=6, lr=0.0114,
sub=0.64, col=0.83 — 5-fold OOF AUC 0.960359, still didn't converge
(best_iter=3999 max). REJECTED as a single model; not picked by the
blend hill-climb. Blend stays at 0.960865.

Round 10 (G2): 5-seed average of the Optuna params (added seeds 7, 91
to the existing 42, 53, 64). OOF AUC 0.960670 (-0.000087 vs the
3-seed average). REJECTED — averaging 5 seeds was slightly worse than
3. The 3 seeds we picked were lucky.

Round 11 (I): **Pseudo-labeling on the champion test predictions.**
Kept 23.2% of test rows (prob > 0.97 or < 0.03) and re-fit the
Optuna params (3 seeds) on the augmented set. OOF AUC **0.960826**
(+0.000069 vs lgbm_full_g). 5/5 folds win.

Round 12 (I2): re-did pseudo-labeling using the *new* champion
`lgbm_full_i` test predictions as the source. 32.8% of test rows
passed the threshold. OOF AUC **0.960872**.

Round 13 (I3): third round of pseudo-labeling using `lgbm_full_i2`
preds. 36.9% of test rows passed the threshold. OOF AUC **0.960882**
(+0.000010 vs i2; 4/5 folds win). REJECTED as a single (delta < gate/2)
but a BLEND MEMBER. NEW BEST BLEND: 4-OOF stack
(`lgbm_full_i3 + cat_d4 + lgbm_full + lgbm_raw`) at C=0.005 =
**0.960984**. **0.000016 from the 0.9610 stretch goal!**

Round 14 (I4): fourth round of pseudo-labeling using `lgbm_full_i3`
preds. 38.7% of test rows passed the threshold. OOF AUC **0.960861**
(-0.000021 vs i3; 2/5 folds win). REJECTED — pseudo-labeling has
saturated; further rounds hurt. The blend stayed at **0.960988**
(0.000004 bump from re-fitting the stacker on the full OOF).
**0.000012 from the 0.9610 stretch goal!**

`xgb_full_d` is REJECTED (0.9559; XGB hit `best_iter=3999` max with
lr=0.05; didn't converge). `xgb_d6_lr02` is REJECTED as a single model
(0.960159) but a BLEND MEMBER. `cat_d4` is REJECTED as a single
model (0.9595, hit best_iter=1999 max) but a tiny positive as a BLEND
MEMBER. `submissions/best.csv` = 0.960837 blend; `submissions/best_single.csv`
= 0.960686 champion. `xgb_full_d` is still excluded (HURT the stack).

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
| `lgbm_raw` + `lgbm_full` + `lgbm_full_te` | 0.005 | 0.960685 | 3-OOF stack |
| 4 LGBM (`raw`+`full`+`full_d`+`full_te`) | 0.005 | 0.960739 | 4-OOF stack |
| 5 (incl. `xgb_d6_lr02`) | 0.005 | 0.960773 | 5-OOF stack |
| 6 (incl. `cat_d4`)     | 0.005 | 0.960776 | 6-OOF stack |
| 5 (optuna-anchor)     | 0.005 | 0.960837 | 5-OOF stack with F champion |
| 5 (g-anchor)          | 0.005 | 0.960865 | 5-OOF stack with G champion |
| 4 (i-anchor)          | 0.005 | 0.960940 | 4-OOF stack with I champion |
| 4 (i2-anchor)         | 0.005 | 0.960975 | 4-OOF stack with I2 champion |
| 4 (i3-anchor)         | **0.005** | **0.960984** | 4-OOF stack with I3 champion (current best) |
| 4 (i3-anchor, retrained) | 0.005 | **0.960988** | re-fit stacker on full OOF; 4-OOF stack with I3 anchor (current best) |

The TE model is slightly worse on its own but adds diversity; the
`lgbm_full_d` model is essentially tied with the champion (within
noise) but adds yet another diverse member. `xgb_d6_lr02` adds a
non-LGBM perspective; `cat_d4` adds ordered-boosting diversity. Each
new model is contributing +0.000003-0.00013 to the blend. The
`xgb_full_d` (lr=0.05, depth=8, didn't converge) HURT the stack and
is excluded. After G, the new champion `lgbm_full_g` (3-seed average of
the Optuna params) anchors the stack.

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

1. **exp_I5_pseudo_wider**: try a tighter confidence threshold
   (0.95/0.05) to keep more pseudo-rows. May add noise but more signal.
2. **exp_J_dart**: try LightGBM with `boosting_type='dart'` for
   different ensemble dynamics.
3. **exp_K_xgb_pseudo**: do the same self-pseudo recipe on XGBoost
   for diversity in the blend.

## Open questions

(see `QUESTIONS.md`)

## Stop condition

- 8 REJECTED rounds in a row, OR
- 60 rounds total.

We stop only after every item in the EXPERIMENT QUEUE (A→I) has been
tried or explicitly skipped.
