"""Experiment I4: fourth round of pseudo-labeling (using lgbm_full_i3 preds).

Same recipe as I/I2/I3, source = lgbm_full_i3.
Saves to oof/lgbm_full_i4.npy / preds/lgbm_full_i4.npy.
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
import exp_I_pseudo  # noqa: E402

EXP_ID = "14_I4_pseudo_i3"
EXP_NAME = "lgbm_full_i4"
CHANGE = "Pseudo-labeling round 4: source = lgbm_full_i3 test preds"
SEEDS = [42, 53, 64]


def main() -> None:
    print(f"\n=== Experiment I4: {CHANGE} ===")
    t0 = time.time()
    Xtr, Xte, y, folds, cat_cols, feats = exp_I_pseudo.build_features()

    champ_test = harness.load_pred("lgbm_full_i3")
    HIGH, LOW = exp_I_pseudo.HIGH, exp_I_pseudo.LOW
    n_test = len(champ_test)
    n_high = int((champ_test >= HIGH).sum())
    n_low = int((champ_test <= LOW).sum())
    n_kept = n_high + n_low
    print(f"  test rows: {n_test} | prob>={HIGH}: {n_high} | prob<={LOW}: {n_low} "
          f"| kept: {n_kept} ({100*n_kept/n_test:.1f}%)")

    keep_mask = (champ_test >= HIGH) | (champ_test <= LOW)
    pseudo_X = Xte[keep_mask].reset_index(drop=True)
    for c in Xtr.columns:
        if c in pseudo_X.columns:
            pseudo_X[c] = pseudo_X[c].astype(Xtr[c].dtype)
    pseudo_y = (champ_test[keep_mask] >= 0.5).astype(np.int8)
    pseudo_w = np.where(champ_test[keep_mask] >= HIGH,
                        champ_test[keep_mask],
                        1.0 - champ_test[keep_mask]).astype(np.float32)
    pseudo_w = np.minimum(pseudo_w, 1.0).astype(np.float32)
    print(f"  pseudo: pos={int(pseudo_y.sum())}  neg={int((1-pseudo_y).sum())}")

    oof = np.zeros(len(Xtr), dtype=np.float64)
    test_pred = np.zeros(len(Xte), dtype=np.float64)
    fold_aucs: list[float] = []
    params_base = exp_I_pseudo.CHAMPION_PARAMS
    for k in range(harness.N_FOLDS):
        tr_m = folds != k
        va_m = folds == k
        X_aug = pd.concat([Xtr.iloc[tr_m], pseudo_X], ignore_index=True)
        y_aug = np.concatenate([y[tr_m], pseudo_y])
        w_aug = np.concatenate([np.ones(int(tr_m.sum()), dtype=np.float32), pseudo_w])
        oof_k = np.zeros(int(va_m.sum()))
        test_k = np.zeros(len(Xte))
        for s in SEEDS:
            params = dict(objective="binary", metric="auc",
                          verbosity=-1, seed=s, num_threads=8,
                          **params_base)
            dtr = lgb.Dataset(X_aug, y_aug, weight=w_aug,
                              categorical_feature=cat_cols, free_raw_data=False)
            dva = lgb.Dataset(Xtr.iloc[va_m], y[va_m],
                              categorical_feature=cat_cols, reference=dtr,
                              free_raw_data=False)
            m = lgb.train(params, dtr, num_boost_round=6000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(300, verbose=False),
                                     lgb.log_evaluation(0)])
            oof_k += m.predict(Xtr.iloc[va_m], num_iteration=m.best_iteration) / len(SEEDS)
            test_k += m.predict(Xte, num_iteration=m.best_iteration) / len(SEEDS)
        oof[va_m] = oof_k
        test_pred += test_k / harness.N_FOLDS
        a = roc_auc_score(y[va_m], oof[va_m])
        fold_aucs.append(a)
        print(f"  fold {k}: AUC={a:.6f}")
    auc = float(roc_auc_score(y, oof))
    delta = auc - 0.960882   # current champion (lgbm_full_i3)
    fold_str = ",".join(f"{a:.4f}" for a in fold_aucs)
    gate = float(json.load(open(os.path.join(ROOT, "logs", "step0_summary.json")))["gate"])
    raw_per_fold = harness.per_fold_auc(y, harness.load_oof("lgbm_full_i3"), folds)
    n_fold_win = sum(1 for a, b in zip(fold_aucs, raw_per_fold) if a > b)
    if (delta >= gate) and (n_fold_win >= 4):
        result = "NEW CHAMPION"
    elif delta >= gate * 0.5:
        result = "BLEND MEMBER"
    else:
        result = "REJECTED"
    minutes = (time.time() - t0) / 60.0
    notes = (f"pseudo n={n_kept} ({100*n_kept/n_test:.1f}%) | "
             f"per_fold_won={n_fold_win}/5 vs lgbm_full_i3 gate={gate:.6f}")
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
        champion_auc=auc if result == "NEW CHAMPION" else 0.960882,
        blend_auc=0.0,
    )


if __name__ == "__main__":
    main()
