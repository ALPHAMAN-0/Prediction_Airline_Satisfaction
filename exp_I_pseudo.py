"""Experiment I: pseudo-labeling with confident test predictions.

Approach:
1. Use the current best test predictions (lgbm_full_g on test) as
   pseudo-labels. Take rows with prob > 0.97 or prob < 0.03 — about
   20-30% of test should qualify on a well-calibrated classifier.
2. Add those rows to the training set with their pseudo-labels.
3. Re-fit the champion (Optuna params, 3 seeds) on the augmented set.
4. The held-out fold is still the original train fold (we never
   pseudo-label validation rows).

Risk: if the original model is miscalibrated for the test distribution,
pseudo-labels will be wrong and the round will HURT. The adversarial
validation AUC was 0.4995 (folds are trustworthy), so the test
distribution is close to train. We expect this to be safe.

Saves to oof/lgbm_full_i.npy / preds/lgbm_full_i.npy.
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

EXP_ID = "11_I_pseudo"
EXP_NAME = "lgbm_full_i"
CHANGE = "Pseudo-labeling: add test rows with prob>0.97 or <0.03 (~20-30% of test) and re-fit champion"
HIGH = 0.97
LOW = 0.03
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


def build_features() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[str], list[str]]:
    """Return (Xtr, Xte, y, folds, cat_cols, feats) using the same recipe
    as the champion."""
    tr, te = harness.load_train_test()
    folds = harness.load_folds(tr)
    y = tr["__y__"].values
    tr_enr, te_enr = exp_B_route.build_route_features(tr, te)
    tr_enr = exp_D_small._add_small_features(tr_enr)
    te_enr = exp_D_small._add_small_features(te_enr)
    PRIOR = float(y.mean())
    from exp_C_te import _te_4way_kfold, _te_flight_dist_kfold
    oof_4, full_4 = _te_4way_kfold(tr_enr, folds, PRIOR)
    tr_enr["te_4way"] = oof_4
    te_keys = (te_enr["Class"].astype(str) + "|" + te_enr["Type of Travel"].astype(str) +
               "|" + te_enr["Customer Type"].astype(str) + "|" + te_enr["Flight Distance"].astype(str))
    tr_enr["te_4way"] = oof_4
    te_enr["te_4way"] = te_keys.map(full_4).fillna(PRIOR).astype(np.float32).values
    oof_fd, full_fd = _te_flight_dist_kfold(tr_enr, folds, PRIOR)
    tr_enr["te_flight_dist"] = oof_fd
    te_enr["te_flight_dist"] = te_enr["Flight Distance"].astype(str).map(full_fd).fillna(PRIOR).astype(np.float32).values
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
    return Xtr, Xte, y, folds, cat_cols, feats


def main() -> None:
    print(f"\n=== Experiment I: {CHANGE} ===")
    t0 = time.time()
    Xtr, Xte, y, folds, cat_cols, feats = build_features()

    # 1. Get the test predictions from the current champion
    champ_test = harness.load_pred("lgbm_full_g")
    n_test = len(champ_test)
    n_high = int((champ_test >= HIGH).sum())
    n_low = int((champ_test <= LOW).sum())
    n_kept = n_high + n_low
    print(f"  test rows: {n_test} | prob>={HIGH}: {n_high} | prob<={LOW}: {n_low} "
          f"| kept: {n_kept} ({100*n_kept/n_test:.1f}%)")

    # 2. Build pseudo-labeled rows
    keep_mask = (champ_test >= HIGH) | (champ_test <= LOW)
    pseudo_X = Xte[keep_mask].reset_index(drop=True)
    # After reset_index, the category dtype can be lost. Cast each column
    # to match Xtr's dtype so the eventual pd.concat preserves it.
    for c in Xtr.columns:
        if c in pseudo_X.columns:
            pseudo_X[c] = pseudo_X[c].astype(Xtr[c].dtype)
    pseudo_y = (champ_test[keep_mask] >= 0.5).astype(np.int8)
    pseudo_w = np.where(champ_test[keep_mask] >= HIGH,
                        champ_test[keep_mask],
                        1.0 - champ_test[keep_mask]).astype(np.float32)
    # Cap weights to 1.0 (don't overshoot the real-label weight)
    pseudo_w = np.minimum(pseudo_w, 1.0).astype(np.float32)
    print(f"  pseudo: pos={int(pseudo_y.sum())}  neg={int((1-pseudo_y).sum())}")

    # 3. 5-fold CV: in each fold, append pseudo to the train set only
    oof = np.zeros(len(Xtr), dtype=np.float64)
    test_pred = np.zeros(len(Xte), dtype=np.float64)
    fold_aucs: list[float] = []
    for k in range(harness.N_FOLDS):
        tr_m = folds != k
        va_m = folds == k
        # Augment train with pseudo
        X_aug = pd.concat([Xtr.iloc[tr_m], pseudo_X], ignore_index=True)
        y_aug = np.concatenate([y[tr_m], pseudo_y])
        w_aug = np.concatenate([np.ones(int(tr_m.sum()), dtype=np.float32), pseudo_w])
        oof_k = np.zeros(int(va_m.sum()))
        test_k = np.zeros(len(Xte))
        for s in SEEDS:
            params = dict(objective="binary", metric="auc",
                          verbosity=-1, seed=s, num_threads=8,
                          **CHAMPION_PARAMS)
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
    delta = auc - 0.960757   # current champion
    fold_str = ",".join(f"{a:.4f}" for a in fold_aucs)
    gate = float(json.load(open(os.path.join(ROOT, "logs", "step0_summary.json")))["gate"])
    raw_per_fold = harness.per_fold_auc(y, harness.load_oof("lgbm_full_g"),
                                        harness.load_folds(harness.load_train_test()[0]))
    n_fold_win = sum(1 for a, b in zip(fold_aucs, raw_per_fold) if a > b)
    if (delta >= gate) and (n_fold_win >= 4):
        result = "NEW CHAMPION"
    elif delta >= gate * 0.5:
        result = "BLEND MEMBER"
    else:
        result = "REJECTED"
    minutes = (time.time() - t0) / 60.0
    notes = (f"pseudo n={n_kept} ({100*n_kept/n_test:.1f}%) | "
             f"per_fold_won={n_fold_win}/5 vs lgbm_full_g gate={gate:.6f}")
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
        champion_auc=auc if result == "NEW CHAMPION" else 0.960757,
        blend_auc=0.0,
    )


if __name__ == "__main__":
    main()
