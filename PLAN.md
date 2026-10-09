# PLAN — S6E10 experiment queue

The user's EXPERIMENT QUEUE, in order. Each item becomes one experiment
in the round-by-round loop. Skip = explicit note in `experiments.csv`
+ `QUESTIONS.md`.

| # | idea | hypothesis | status |
|---|---|---|---|
| A | Original data: add the public "Airline Passenger Satisfaction" dataset as extra training rows. Map columns/labels. Use in training folds only. | The extra ~130 k rows may improve the LGBM fit on the minority class. | **SKIP** — no original CSV found on disk or Kaggle input. |
| B | Flight Distance group features: count, per-source count, mean/std of ratings+Age, per-source category share. | "Same route" passengers behave similarly, but the model has no route context. Adding route aggregates gives it that signal. | NEXT |
| C | Target encoding of Flight Distance + 4-way (Flight Distance × Class × Type of Travel × Customer Type), nested inside each training fold, smoothed (m=20..50). | Smoothing buys ~0.0001-0.0002 OOF AUC; nested fold-fitting kills leakage. | queued |
| D | Small features: ratings as categoricals, count of 0 ratings, mean/std/min/max of ratings, log1p of delays, arrival−departure delay, missing flag for Arrival Delay, type×class×customer. | Hand-engineered interactions give GBDTs a faster route to a good fit. | queued |
| E | Model variety: LightGBM, XGBoost, CatBoost with native categoricals (and sklearn-MLP fallback since no pytabkit/CUDA locally). | Diversity matters in the ensemble. Inter-model correlation is already ≥ 0.98 in `s6e10_solution.py`, so adding CatBoost + XGB is the cheapest diversity gain. | queued |
| F | Optuna: ≤ 40 trials per model on a 30 % sample, then confirm on full data. | Search spaces are well-defined in the user mission; default LGBM_BEST was set without search. | queued |
| G | Final fits: lower learning rate with early stopping, averaged over 3-5 seeds. | Lower LR + more trees reduces variance; seed averaging removes seed noise. | queued |
| H | Blend method: hill-climbing on OOF ranks. | Rank-blend is more robust to scale shifts between models. | queued |
| I | Pseudo-labelling with very confident test predictions (e.g., prob > 0.99 or < 0.01). | Add ~1-5 k high-confidence test rows to training; may help on noisy edges. | queued |

## New ideas (populated after the queue empties)

(populated by 5 ideas at a time from weakest segments / feature
importances / most-wrong rows)
