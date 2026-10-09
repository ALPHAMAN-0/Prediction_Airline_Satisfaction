# Latest run results

Snapshot of the most recent pipeline run (committed for tracking). The numbers
below are from a **SMOKE run** (5% sample, 2 folds) — the full-data run is the
target for Kaggle submission.

| member | OOF AUC | type |
|---|---|---|
| `hill`         | **0.965440** | ensemble (Caruana hill-climb, logit space) |
| `rank`         | 0.965356 | ensemble (rank-average of top-3) |
| `stack`        | 0.965321 | ensemble (logistic stack, C tuned) |
| `lgbm_full`    | 0.964923 | model (LGBM, Optuna-tuned, 40 trials) |
| `lgbm_full_et_bag` | 0.964850 | model (LGBM extra_trees, 3-seed bag) |
| `lgbm_full_et` | 0.964706 | model (LGBM extra_trees) |
| `xgb_full`     | 0.962517 | model (XGBoost GPU) |
| `cat_depth10`  | 0.962328 | model (CatBoost GPU, depth=10) |
| `lgbm_raw`     | 0.953293 | baseline (raw features only) |

**Submission written:** `output/submission.csv` = `hill` (best OOF AUC).
**Safe fallback:** `output/submission_safe.csv` = rank-avg of top-3 base learners.

## Notes

- Best OOF AUC = **0.96544** (hill ensemble, on 5% sample).
- Inter-model correlation ≥ 0.98 for all GBDT members; ensemble gains come
  mostly from `hill`/`stack` re-weighting rather than diversity.
- Missing from this run (would diversify the ensemble):
  `realmlp_full` (set `RUN_RMLP=1`)
  `cat_all_cat` (set `RUN_CAT_CAT=1`)
- Next step for Kaggle submission: re-run with `SMOKE=0`, `RUN_RMLP=1`,
  `RUN_CAT_CAT=1` on a T4 × 2 GPU.
