"""Experiment D: small hand-engineered interaction features.

Builds on the champion (route features). Adds:
- log1p of Departure / Arrival delays
- delay diff (dep - arr), max/min/total of delays
- missing flag for Arrival Delay (NaN in raw data)
- ratings: count of 0 ratings, mean/std/min/max/range
- low_rating_frac (<=2), high_rating_frac (>=4), flat_rater flag
- type_x_class_x_customer string key (categorical; may not help LGBM but
  the model treats it as an embedded lookup)

TDD: `_add_small_features` is tested in `_tests/test_harness.py` (T9).
"""
from __future__ import annotations

import os
import sys
import time
import json

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import harness  # noqa: E402
import exp_B_route  # noqa: E402

EXP_ID = "3_D_small"
EXP_NAME = "lgbm_full_d"
CHANGE = "Champion + small hand-engineered features (log delays, delay diff, missing flag, rating aggregates, type_x_class_x_customer)"


# ----- feature function (tested in T9) ------------------------------------
def _add_small_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the small interaction features. Works on a single DataFrame
    (train or test); does NOT touch the target column."""
    if "satisfaction" in df.columns:
        RATING = harness.rating_cols(df)
    else:
        RATING = _rating_cols_no_y(df)
    out = df.copy()
    dep = out["Departure Delay in Minutes"].astype("float32")
    arr = out["Arrival Delay in Minutes"].astype("float32")
    out["log_dep_delay"] = np.log1p(dep.fillna(0)).astype("float32")
    out["log_arr_delay"] = np.log1p(arr.fillna(0)).astype("float32")
    out["total_delay"] = (dep.fillna(0) + arr.fillna(0)).astype("float32")
    out["max_delay"] = np.maximum(dep.fillna(0), arr.fillna(0)).astype("float32")
    out["min_delay"] = np.minimum(dep.fillna(0), arr.fillna(0)).astype("float32")
    out["delay_diff"] = (dep.fillna(0) - arr.fillna(0)).astype("float32")
    out["has_dep_delay"] = (dep.fillna(0) > 0).astype("int8")
    out["has_arr_delay"] = (arr.fillna(0) > 0).astype("int8")
    out["on_time"] = ((dep.fillna(0) == 0) & (arr.fillna(0) == 0)).astype("int8")
    out["arr_delay_missing"] = arr.isna().astype("int8")
    rvals = out[RATING].astype("float32")
    out["rating_mean"] = rvals.mean(axis=1).astype("float32")
    out["rating_std"]  = rvals.std(axis=1).astype("float32")
    out["rating_min"]  = rvals.min(axis=1).astype("float32")
    out["rating_max"]  = rvals.max(axis=1).astype("float32")
    out["rating_range"] = (out["rating_max"] - out["rating_min"]).astype("float32")
    out["n_zero_ratings"] = (rvals == 0).sum(axis=1).astype("int8")
    out["flat_rater"] = (out["rating_std"] < 0.5).astype("int8")
    out["low_rating_frac"] = (rvals <= 2).mean(axis=1).astype("float32")
    out["high_rating_frac"] = (rvals >= 4).mean(axis=1).astype("float32")
    # combined categorical (helps the model as a "lookup")
    if {"Type of Travel", "Class", "Customer Type"}.issubset(out.columns):
        out["type_x_class_x_customer"] = (
            out["Type of Travel"].astype(str) + "|" +
            out["Class"].astype(str) + "|" +
            out["Customer Type"].astype(str)
        ).astype("category")
    return out


def _rating_cols_no_y(df: pd.DataFrame) -> list[str]:
    DROP = ["id", "satisfaction", "__y__"]
    RAW = [c for c in df.columns if c not in DROP]
    return [c for c in RAW if c not in harness.CAT_COLS + harness.NUM_COLS]


# ----- model ---------------------------------------------------------------
def lgbm_full_d_5fold(name: str, seed: int = 42) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Champion (route features) + small interactions."""
    tr, te = harness.load_train_test()
    folds = harness.load_folds(tr)
    y = tr["__y__"].values
    tr_enr, te_enr = exp_B_route.build_route_features(tr, te)
    tr_enr = _add_small_features(tr_enr)
    te_enr = _add_small_features(te_enr)
    feats = [c for c in tr_enr.columns
             if c not in ("id", "satisfaction", "__y__", "__src__")]
    Xtr = tr_enr[feats].copy()
    Xte = te_enr[feats].copy()
    for c in harness.CAT_COLS:
        if c in Xtr.columns:
            Xtr[c] = Xtr[c].astype("category")
            Xte[c] = Xte[c].astype("category")
    Xtr, Xte = harness.make_aligned_categoricals(Xtr, Xte)
    cat_cols = [c for c in Xtr.columns if str(Xtr[c].dtype) == "category"]
    params = dict(objective="binary", metric="auc", learning_rate=0.05,
                  num_leaves=63, min_data_in_leaf=50, feature_fraction=0.85,
                  bagging_fraction=0.85, bagging_freq=1,
                  lambda_l1=0.1, lambda_l2=0.1,
                  verbosity=-1, seed=seed, num_threads=8, max_bin=255)
    oof = np.zeros(len(tr), dtype=np.float64)
    test_pred = np.zeros(len(te), dtype=np.float64)
    fold_aucs: list[float] = []
    for k in range(harness.N_FOLDS):
        tr_m = folds != k; va_m = folds == k
        dtr = lgb.Dataset(Xtr.iloc[tr_m], y[tr_m],
                          categorical_feature=cat_cols, free_raw_data=False)
        dva = lgb.Dataset(Xtr.iloc[va_m], y[va_m],
                          categorical_feature=cat_cols, reference=dtr,
                          free_raw_data=False)
        m = lgb.train(params, dtr, num_boost_round=4000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False),
                                 lgb.log_evaluation(0)])
        oof[va_m] = m.predict(Xtr.iloc[va_m], num_iteration=m.best_iteration)
        test_pred += m.predict(Xte, num_iteration=m.best_iteration) / harness.N_FOLDS
        a = roc_auc_score(y[va_m], oof[va_m])
        fold_aucs.append(a)
        print(f"  fold {k}: best_iter={m.best_iteration}  AUC={a:.6f}")
    return oof, test_pred, fold_aucs


# ----- log helper ---------------------------------------------------------
def log_round(name: str, oof: np.ndarray, test_pred: np.ndarray,
              fold_aucs: list[float], minutes: float,
              champion_auc: float) -> dict:
    y = harness.load_train_test()[0]["__y__"].values
    auc = float(roc_auc_score(y, oof))
    delta = auc - champion_auc
    fold_str = ",".join(f"{a:.4f}" for a in fold_aucs)
    gate = float(json.load(open(os.path.join(ROOT, "logs", "step0_summary.json")))["gate"])
    raw_per_fold = harness.per_fold_auc(y, harness.load_oof("lgbm_full"),
                                        harness.load_folds(harness.load_train_test()[0]))
    n_fold_win = sum(1 for a, b in zip(fold_aucs, raw_per_fold) if a > b)
    improved = (delta >= gate) and (n_fold_win >= 4)
    if improved:
        result = "NEW CHAMPION"
    elif delta >= gate * 0.5:
        result = "BLEND MEMBER"
    else:
        result = "REJECTED"
    notes = (f"per_fold_won={n_fold_win}/5 vs lgbm_full (champion) "
             f"gate={gate:.6f}")
    exp = harness.Experiment(
        exp_id=EXP_ID, change=CHANGE, oof_auc=auc, delta=delta,
        per_fold_aucs=fold_str, blend_auc=0.0,
        runtime_min=minutes, result=result, notes=notes,
    )
    harness.append_experiment(exp)
    np.save(harness.oof_path(name), oof.astype(np.float32))
    np.save(harness.pred_path(name), test_pred.astype(np.float32))
    return {"auc": auc, "delta": delta, "result": result,
            "fold_aucs": fold_aucs, "n_fold_win": n_fold_win,
            "per_fold_champ": raw_per_fold}


def main() -> None:
    print(f"\n=== Experiment D: {CHANGE} ===")
    t0 = time.time()
    oof, test_pred, fold_aucs = lgbm_full_d_5fold(EXP_NAME, seed=harness.SEED)
    minutes = (time.time() - t0) / 60.0
    summary = log_round(EXP_NAME, oof, test_pred, fold_aucs, minutes,
                        champion_auc=0.960421)
    auc = summary["auc"]; delta = summary["delta"]; result = summary["result"]
    print(f"\n{EXP_NAME}: OOF AUC = {auc:.6f}  delta {delta:+.6f}  "
          f"({minutes:.1f} min)  {result}")
    print(f"per-fold: {[round(a, 4) for a in fold_aucs]}")
    harness.status_line(
        exp_id=EXP_ID, change=CHANGE, oof_auc=auc, delta=delta,
        result=result,
        champion_auc=auc if result == "NEW CHAMPION" else 0.960421,
        blend_auc=0.0,
    )


if __name__ == "__main__":
    main()
