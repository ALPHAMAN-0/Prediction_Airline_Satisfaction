"""Experiment E (XGBoost): same features as the champion (route + small
interactions) but trained with XGBoost on CPU with native categoricals.

Hypothesis: XGBoost's different splitting strategy gives the blend a
non-LGBM member, which is the highest-leverage single change for
ensemble diversity.

Saves to oof/xgb_full_d.npy, preds/xgb_full_d.npy.
"""
from __future__ import annotations

import os
import sys
import time
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import harness  # noqa: E402
import exp_B_route  # noqa: E402
import exp_D_small  # noqa: E402

EXP_ID = "4_E1_xgb"
EXP_NAME = "xgb_full_d"
CHANGE = "XGBoost on champion features (route+small), depth=8, lr=0.05"


def xgb_full_d_5fold(name: str, max_depth: int = 8, learning_rate: float = 0.05,
                     seed: int = 42) -> tuple[np.ndarray, np.ndarray, list[float]]:
    tr, te = harness.load_train_test()
    folds = harness.load_folds(tr)
    y = tr["__y__"].values
    tr_enr, te_enr = exp_B_route.build_route_features(tr, te)
    tr_enr = exp_D_small._add_small_features(tr_enr)
    te_enr = exp_D_small._add_small_features(te_enr)
    feats = [c for c in tr_enr.columns
             if c not in ("id", "satisfaction", "__y__", "__src__")]
    Xtr = tr_enr[feats].copy()
    Xte = te_enr[feats].copy()
    for c in harness.CAT_COLS:
        if c in Xtr.columns:
            Xtr[c] = Xtr[c].astype("category")
            Xte[c] = Xte[c].astype("category")
    Xtr, Xte = harness.make_aligned_categoricals(Xtr, Xte)
    params = dict(objective="binary:logistic", eval_metric="auc",
                  tree_method="hist", device="cpu",
                  max_depth=max_depth, learning_rate=learning_rate,
                  subsample=0.8, colsample_bytree=0.8,
                  reg_alpha=0.1, reg_lambda=0.5,
                  seed=seed, verbosity=0, enable_categorical=True,
                  early_stopping_rounds=200)
    oof = np.zeros(len(tr), dtype=np.float64)
    test_pred = np.zeros(len(te), dtype=np.float64)
    fold_aucs: list[float] = []
    for k in range(harness.N_FOLDS):
        tr_m = folds != k; va_m = folds == k
        dtr = xgb.DMatrix(Xtr.iloc[tr_m], label=y[tr_m], enable_categorical=True)
        dva = xgb.DMatrix(Xtr.iloc[va_m], label=y[va_m], enable_categorical=True)
        dte = xgb.DMatrix(Xte, enable_categorical=True)
        booster = xgb.train(params, dtr, num_boost_round=4000,
                            evals=[(dva, "val")], verbose_eval=0)
        bi = getattr(booster, "best_iteration", None)
        if bi is None or bi < 0:
            bi = booster.num_boosted_rounds() - 1
        oof[va_m] = booster.predict(dva, iteration_range=(0, bi + 1))
        test_pred += booster.predict(dte, iteration_range=(0, bi + 1)) / harness.N_FOLDS
        a = roc_auc_score(y[va_m], oof[va_m])
        fold_aucs.append(a)
        print(f"  fold {k}: best_iter={bi}  AUC={a:.6f}")
    return oof, test_pred, fold_aucs


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
    print(f"\n=== Experiment E1 (XGBoost): {CHANGE} ===")
    t0 = time.time()
    oof, test_pred, fold_aucs = xgb_full_d_5fold(EXP_NAME)
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
