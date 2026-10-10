"""Experiment J: LightGBM with boosting_type='dart' for diversity.

DART (Dropouts meet Multiple Additive Regression Trees) drops a random
subset of trees at each iteration, making it behave like an ensemble of
ensembles. Often weaker than gbdt on a single model but adds diversity
in the blend.

Saves to oof/lgbm_dart.npy / preds/lgbm_dart.npy.
"""
from __future__ import annotations

import os
import sys
import time
import json

import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import harness  # noqa: E402
import exp_I_pseudo  # noqa: E402

EXP_ID = "15_J_dart"
EXP_NAME = "lgbm_dart"
CHANGE = "LightGBM boosting_type='dart' at Optuna params (3 seeds)"
SEEDS = [42, 53, 64]

# DART needs different params — drop_rate, skip_drop, etc.
# We use the champion's lr/leaves/etc as a base, but add DART-specific
# knobs. DART does NOT use early stopping (no best_iteration),
# so we set a fixed number of rounds.
DART_PARAMS = dict(
    boosting_type="dart",
    learning_rate=0.05,            # higher LR for DART (it's slower)
    num_leaves=138,
    min_data_in_leaf=68,
    feature_fraction=0.75,
    bagging_fraction=0.89,
    bagging_freq=1,
    lambda_l1=4.8,
    lambda_l2=0.07,
    max_bin=127,
    drop_rate=0.1,
    skip_drop=0.5,
    max_drop=50,
    num_boost_round=1500,          # DART fixed
)


def main() -> None:
    print(f"\n=== Experiment J: {CHANGE} ===")
    t0 = time.time()
    Xtr, Xte, y, folds, cat_cols, feats = exp_I_pseudo.build_features()

    oof = np.zeros(len(Xtr), dtype=np.float64)
    test_pred = np.zeros(len(Xte), dtype=np.float64)
    fold_aucs: list[float] = []
    for k in range(harness.N_FOLDS):
        tr_m = folds != k
        va_m = folds == k
        oof_k = np.zeros(int(va_m.sum()))
        test_k = np.zeros(len(Xte))
        for s in SEEDS:
            params = dict(objective="binary", metric="auc",
                          verbosity=-1, seed=s, num_threads=8,
                          **DART_PARAMS)
            dtr = lgb.Dataset(Xtr.iloc[tr_m], y[tr_m],
                              categorical_feature=cat_cols, free_raw_data=False)
            dva = lgb.Dataset(Xtr.iloc[va_m], y[va_m],
                              categorical_feature=cat_cols, reference=dtr,
                              free_raw_data=False)
            m = lgb.train(params, dtr,
                          num_boost_round=DART_PARAMS["num_boost_round"],
                          valid_sets=[dva],
                          callbacks=[lgb.log_evaluation(0)])
            oof_k += m.predict(Xtr.iloc[va_m]) / len(SEEDS)
            test_k += m.predict(Xte) / len(SEEDS)
        oof[va_m] = oof_k
        test_pred += test_k / harness.N_FOLDS
        a = roc_auc_score(y[va_m], oof[va_m])
        fold_aucs.append(a)
        print(f"  fold {k}: AUC={a:.6f}")
    auc = float(roc_auc_score(y, oof))
    delta = auc - 0.960882   # current champion
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
    notes = (f"dart 1500 rounds no early-stop | "
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
