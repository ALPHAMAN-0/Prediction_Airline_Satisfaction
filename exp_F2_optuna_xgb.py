"""Experiment F2: Optuna search for XGBoost (15 trials, 3-fold).

The first XGBoost experiment (E1) used depth=8 lr=0.05 and didn't
converge (best_iter=3999 max). E1b at depth=6 lr=0.02 also didn't
converge. Optuna should find a stable config — likely shallower
(depth=3-5) and lower LR (0.01-0.05).

Budget: 15 trials × ~3 min each = ~45 min. 3-fold for speed.

Saves to oof/xgb_optuna.npy / preds/xgb_optuna.npy.
"""
from __future__ import annotations

import os
import sys
import time
import json

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import harness  # noqa: E402
import exp_B_route  # noqa: E402
import exp_D_small  # noqa: E402

EXP_ID = "9_F2_optuna_xgb"
EXP_NAME = "xgb_optuna"
CHANGE = "Optuna XGBoost (15 trials, 3-fold) for depth/lr/colsample — fix didn't-converge"
N_TRIALS = 15
N_FOLDS_TRIAL = 3
NUM_BOOST = 4000


def _prep_data():
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
    return Xtr, Xte, y, folds


def _cv_eval(Xtr, y, folds, fold_ids, params, num_boost=NUM_BOOST) -> float:
    aucs = []
    for k in fold_ids:
        tr_m = folds != k
        va_m = folds == k
        dtr = xgb.DMatrix(Xtr.iloc[tr_m], label=y[tr_m], enable_categorical=True)
        dva = xgb.DMatrix(Xtr.iloc[va_m], label=y[va_m], enable_categorical=True)
        booster = xgb.train(params, dtr, num_boost_round=num_boost,
                            evals=[(dva, "val")], verbose_eval=0)
        bi = getattr(booster, "best_iteration", None)
        if bi is None or bi < 0:
            bi = booster.num_boosted_rounds() - 1
        p = booster.predict(dva, iteration_range=(0, bi + 1))
        a = roc_auc_score(y[va_m], p)
        aucs.append(a)
    return float(np.mean(aucs))


def main() -> None:
    print(f"\n=== Experiment F2: {CHANGE} ===")
    t0 = time.time()
    Xtr, Xte, y, folds = _prep_data()
    trial_fold_ids = [0, 1, 2]

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            objective="binary:logistic", eval_metric="auc",
            tree_method="hist", device="cpu",
            max_depth=trial.suggest_int("max_depth", 3, 7),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
            seed=harness.SEED, verbosity=0, enable_categorical=True,
        )
        return _cv_eval(Xtr, y, folds, trial_fold_ids, params)

    sampler = optuna.samplers.TPESampler(seed=harness.SEED, multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    print(f"Optuna best 3-fold mean AUC: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")

    # Final 5-fold refit
    best_params = dict(
        objective="binary:logistic", eval_metric="auc",
        tree_method="hist", device="cpu",
        verbosity=0, enable_categorical=True,
        early_stopping_rounds=200,
        **study.best_params,
    )
    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(Xte), dtype=np.float64)
    fold_aucs: list[float] = []
    for k in range(harness.N_FOLDS):
        tr_m = folds != k
        va_m = folds == k
        dtr = xgb.DMatrix(Xtr.iloc[tr_m], label=y[tr_m], enable_categorical=True)
        dva = xgb.DMatrix(Xtr.iloc[va_m], label=y[va_m], enable_categorical=True)
        dte = xgb.DMatrix(Xte, enable_categorical=True)
        booster = xgb.train(best_params, dtr, num_boost_round=4000,
                            evals=[(dva, "val")], verbose_eval=0)
        bi = getattr(booster, "best_iteration", None)
        if bi is None or bi < 0:
            bi = booster.num_boosted_rounds() - 1
        oof[va_m] = booster.predict(dva, iteration_range=(0, bi + 1))
        test_pred += booster.predict(dte, iteration_range=(0, bi + 1)) / harness.N_FOLDS
        a = roc_auc_score(y[va_m], oof[va_m])
        fold_aucs.append(a)
        print(f"  fold {k}: best_iter={bi}  AUC={a:.6f}")
    auc = float(roc_auc_score(y, oof))
    delta = auc - 0.960757   # current champion
    fold_str = ",".join(f"{a:.4f}" for a in fold_aucs)
    gate = float(json.load(open(os.path.join(ROOT, "logs", "step0_summary.json")))["gate"])
    raw_per_fold = harness.per_fold_auc(y, harness.load_oof("lgbm_full_g"), folds)
    n_fold_win = sum(1 for a, b in zip(fold_aucs, raw_per_fold) if a > b)
    if (delta >= gate) and (n_fold_win >= 4):
        result = "NEW CHAMPION"
    elif delta >= gate * 0.5:
        result = "BLEND MEMBER"
    else:
        result = "REJECTED"
    minutes = (time.time() - t0) / 60.0
    notes = (f"optuna_3fold_best={study.best_value:.6f} | per_fold_won="
             f"{n_fold_win}/5 vs lgbm_full_g (champion) gate={gate:.6f}")
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
