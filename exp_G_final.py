"""Experiment G: re-fit the NEW champion (Optuna params) with 3 seeds averaged.

After round 7 (F), the new champion is `lgbm_full_optuna` with params:
  lr=0.020, num_leaves=138, min_data=68, feature_frac=0.75,
  bagging_frac=0.89, lambda_l1=4.8, lambda_l2=0.07, max_bin=127

Hypothesis: averaging over 3 seeds at the same params should reduce
variance and give a small bump (typically +0.0001-0.0003 on similar
Kaggle tasks).

Saves to oof/lgbm_full_g.npy / preds/lgbm_full_g.npy.
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
import exp_D_small  # noqa: E402

EXP_ID = "8_G_final"
EXP_NAME = "lgbm_full_g"
CHANGE = "NEW champion (Optuna params) + 3 seeds averaged; lr=0.020, num_leaves=138"
SEEDS = [42, 53, 64]
CHAMPION_PARAMS = dict(
    learning_rate=0.020340260398623772,
    num_leaves=138,
    min_data_in_leaf=68,
    feature_fraction=0.7483052441171618,
    bagging_fraction=0.8870395218311953,
    bagging_freq=1,
    lambda_l1=4.806655024757077,
    lambda_l2=0.06748507222716818,
    max_bin=127,
)


def lgbm_full_g_5fold(seeds: list[int] = SEEDS) -> tuple[np.ndarray, np.ndarray, list[float]]:
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
    cat_cols = [c for c in Xtr.columns if str(Xtr[c].dtype) == "category"]
    oof = np.zeros(len(tr), dtype=np.float64)
    test_pred = np.zeros(len(te), dtype=np.float64)
    fold_aucs: list[float] = []
    for k in range(harness.N_FOLDS):
        tr_m = folds != k; va_m = folds == k
        oof_k = np.zeros(int(va_m.sum()))
        test_k = np.zeros(len(Xte))
        for s in seeds:
            params = dict(objective="binary", metric="auc",
                          verbosity=-1, seed=s, num_threads=8,
                          **CHAMPION_PARAMS)
            dtr = lgb.Dataset(Xtr.iloc[tr_m], y[tr_m],
                              categorical_feature=cat_cols, free_raw_data=False)
            dva = lgb.Dataset(Xtr.iloc[va_m], y[va_m],
                              categorical_feature=cat_cols, reference=dtr,
                              free_raw_data=False)
            m = lgb.train(params, dtr, num_boost_round=6000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(300, verbose=False),
                                     lgb.log_evaluation(0)])
            oof_k += m.predict(Xtr.iloc[va_m], num_iteration=m.best_iteration) / len(seeds)
            test_k += m.predict(Xte, num_iteration=m.best_iteration) / len(seeds)
        oof[va_m] = oof_k
        test_pred += test_k / harness.N_FOLDS
        a = roc_auc_score(y[va_m], oof[va_m])
        fold_aucs.append(a)
        print(f"  fold {k}: AUC={a:.6f}  (avg of {len(seeds)} seeds)")
    return oof, test_pred, fold_aucs


def main() -> None:
    print(f"\n=== Experiment G: {CHANGE} ===")
    t0 = time.time()
    oof, test_pred, fold_aucs = lgbm_full_g_5fold()
    minutes = (time.time() - t0) / 60.0
    y = harness.load_train_test()[0]["__y__"].values
    auc = float(roc_auc_score(y, oof))
    delta = auc - 0.960686   # new champion
    fold_str = ",".join(f"{a:.4f}" for a in fold_aucs)
    gate = float(json.load(open(os.path.join(ROOT, "logs", "step0_summary.json")))["gate"])
    raw_per_fold = harness.per_fold_auc(y, harness.load_oof("lgbm_full_optuna"),
                                        harness.load_folds(harness.load_train_test()[0]))
    n_fold_win = sum(1 for a, b in zip(fold_aucs, raw_per_fold) if a > b)
    if (delta >= gate) and (n_fold_win >= 4):
        result = "NEW CHAMPION"
    elif delta >= gate * 0.5:
        result = "BLEND MEMBER"
    else:
        result = "REJECTED"
    notes = (f"per_fold_won={n_fold_win}/5 vs lgbm_full_optuna (champion) "
             f"gate={gate:.6f}")
    exp = harness.Experiment(
        exp_id=EXP_ID, change=CHANGE, oof_auc=auc, delta=delta,
        per_fold_aucs=fold_str, blend_auc=0.0,
        runtime_min=minutes, result=result, notes=notes,
    )
    harness.append_experiment(exp)
    np.save(harness.oof_path(EXP_NAME), oof.astype(np.float32))
    np.save(harness.pred_path(EXP_NAME), test_pred.astype(np.float32))
    print(f"\n{EXP_NAME}: OOF AUC = {auc:.6f}  delta {delta:+.6f}  "
          f"({minutes:.1f} min)  {result}")
    print(f"per-fold: {[round(a, 4) for a in fold_aucs]}")
    harness.status_line(
        exp_id=EXP_ID, change=CHANGE, oof_auc=auc, delta=delta,
        result=result,
        champion_auc=auc if result == "NEW CHAMPION" else 0.960686,
        blend_auc=0.0,
    )


if __name__ == "__main__":
    main()
