# Kaggle Playground S6E10 – Predicting Airline Satisfaction

End-to-end pipeline for the binary classification competition. Single notebook, runs
on Kaggle with GPU T4 × 2, internet ON, ~12 h budget.

## Files

| file | purpose |
|---|---|
| `s6e10_solution.py`  | Jupytext percent-format source (the truth) |
| `s6e10_solution.ipynb` | The Kaggle-ready notebook (paired with the .py via jupytext) |
| `playground-series-dataCSV/` | local CSVs (train.csv, test.csv, sample_submission.csv) |

## How to run on Kaggle

1. **Add inputs.** Attach:
   - the **competition data** (Playground Series S6E10 — *Predicting Airline Satisfaction*) — the auto-mounted dataset already has `train.csv`, `test.csv`, `sample_submission.csv`.
   - (Optional) the **original public dataset** "Airline Passenger Satisfaction"
     (`teejmahal20/airline-passenger-satisfaction`) so the label-safe block has ~130 k
     external rows. The pipeline auto-detects it.
2. **Accelerator.** GPU T4 × 2 (or any single GPU ≥ 11 GB).
3. **Internet.** ON (needed only if you install `pytabkit`).
4. Open `s6e10_solution.ipynb` and **Run All**. To resume a crashed run, just
   Run All again — done models are skipped.

Toggle models / sizes via env vars at the top of the CONFIG cell:

```python
os.environ["SMOKE"]     = "0"   # 1 -> 5% sample, 2 folds (for quick tests)
os.environ["N_FOLDS"]   = "5"
os.environ["SEED"]      = "42"
os.environ["TUNE_LGBM"] = "1"   # 40-trial Optuna; set 0 to skip and use defaults
os.environ["RUN_RMLP"]  = "1"   # pytabkit RealMLP (GPU)
```

Local sanity check (uses 5% sample, 2 folds, CPU-only is fine for LGBM/XGB/CatBoost
fallback):

```bash
DATA_DIR=./playground-series-dataCSV OUT_DIR=./output SMOKE=1 N_FOLDS=2 \
  RUN_RMLP=0 USE_GPU=0 .venv/bin/python s6e10_solution.py
```

## What the pipeline does

| Step | What | Approx. minutes on Kaggle T4 × 2 |
|---|---|---|
| **A** Load & check | Read 21 features, fix target, build fixed StratifiedKFold(5,42) | < 1 |
| **B** Baseline LGBM (raw features only) | `lgbm_raw` | 1 – 2 |
| **C** Route features | Flight-Distance aggregates, residuals, per-route category shares, smoothed target rates from the original dataset, `orig_xgb` prior | 5 – 10 |
| **D** Target encoding (3 blocks) | `sklearn.TargetEncoder`, conditional (rating × segment), frequency | 1 – 3 |
| **E** LGBM tuned on FULL features | `lgbm_full`, `lgbm_full_et` (40 Optuna trials) | 10 – 20 |
| **F.1** XGBoost (GPU, hist) | `xgb_full` | 5 – 10 |
| **F.2** CatBoost (GPU) × 2 variants | `cat_depth10`, `cat_all_cat` | 15 – 30 |
| **F.3** RealMLP (pytabkit, GPU) | `realmlp_full` | 60 – 90 |
| **G** Ensemble | logistic stack (C tuned), rank-average top-3, Caruana hill-climb | < 5 |
| **H** Submission | best ensemble → `submission.csv`, safe rank-avg → `submission_safe.csv` | < 1 |

Total target runtime ≈ 2 – 3 h. Worst case ≈ 4 h.

## Outputs

After a successful run, `/kaggle/working/` contains:

```
submission.csv              # best ensemble's test predictions
submission_safe.csv         # rank-average of top-3 individual members (safer)
oof_<name>.npy              # OOF predictions per model
test_<name>.npy             # averaged fold predictions per model
results.csv                 # (name, oof_auc, fold_std, minutes) summary
```

## Expected OOF AUC (full data, 5 folds)

The pipeline is tuned to roughly hit:

| model | expected OOF AUC |
|---|---|
| `lgbm_raw` | ≈ 0.9588 |
| `lgbm_full` | ≈ 0.9600 – 0.9610 |
| `lgbm_full_et` | ≈ 0.9602 – 0.9612 |
| `xgb_full` | ≈ 0.9590 |
| `cat_depth10` | ≈ 0.9590 |
| `cat_all_cat` | ≈ 0.9585 – 0.9600 |
| `realmlp_full` | ≈ 0.9600 – 0.9615 |
| stack / hill ensemble | ≈ 0.9610 – 0.9625 |

If `lgbm_full` < 0.9570 in smoke, debug the route block before the full run.

## Notes & pitfalls

- pandas `.map` on a category column keeps the category dtype — wrap with
  `.astype(float32)` where you need a numeric column.
- RealMLP expects numeric input. We hand it the FULL feature block which is
  fully numeric at that point.
- Resume-safety: every model short-circuits if its `oof_<name>.npy` is already
  present, so a crashed session can be resumed without recomputation.
- We do **not** append original-data rows to the training fold (known to hurt).
