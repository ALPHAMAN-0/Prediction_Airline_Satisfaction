"""Experiment F: Optuna search on the LGBM champion hyperparameters.

We use 3-fold CV (faster) inside Optuna and re-fit the best params on the
full 5-fold for the final OOF. The 3 folds are the first 3 of the 5 frozen
folds (a strict subset of the held-out 5-fold structure), so AUCs are
comparable across trials but somewhat noisier than the 5-fold number.

Budget: <= 40 trials, ~1-2 min each on 3-fold = ~40-80 min total. We cap
at 30 trials to stay safe.

Saves the final 5-fold OOF/test to oof/lgbm_full_optuna.npy / preds/.
"""
from __future__ import annotations

import os
import sys
import time
import json

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import harness  # noqa: E402
import exp_B_route  # noqa: E402
import exp_D_small  # noqa: E402

EXP_ID = "7_F_optuna"
EXP_NAME = "lgbm_full_optuna"
CHANGE = "Optuna search (30 trials, 3-fold) over LGBM hyperparameters; final 5-fold refit"
N_TRIALS = 30
N_FOLDS_TRIAL = 3   # 3-fold inside Optuna
NUM_BOOST = 3000    # cap per trial


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
    cat_cols = [c for c in Xtr.columns if str(Xtr[c].dtype) == "category"]
    return Xtr, Xte, y, folds, cat_cols


def _cv_eval(Xtr, y, folds, fold_ids, params, cat_cols, num_boost=NUM_BOOST):
    """3-fold CV. `folds` is the full 5-fold assignment; `fold_ids` is the
    list of fold indices to iterate (e.g. [0,1,2])."""
    aucs = []
    for k in fold_ids:
        tr_m = folds != k
        va_m = folds == k
        dtr = lgb.Dataset(Xtr.iloc[tr_m], y[tr_m],
                          categorical_feature=cat_cols, free_raw_data=False)
        dva = lgb.Dataset(Xtr.iloc[va_m], y[va_m],
                          categorical_feature=cat_cols, reference=dtr,
                          free_raw_data=False)
        m = lgb.train(params, dtr, num_boost_round=num_boost, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(150, verbose=False),
                                 lgb.log_evaluation(0)])
        p = m.predict(Xtr.iloc[va_m], num_iteration=m.best_iteration)
        a = roc_auc_score(y[va_m], p)
        aucs.append(a)
    return float(np.mean(aucs))


def main() -> None:
    print(f"\n=== Experiment F: {CHANGE} ===")
    t0 = time.time()
    Xtr, Xte, y, folds, cat_cols = _prep_data()
    trial_fold_ids = [0, 1, 2]   # 3-fold inside Optuna

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            objective="binary", metric="auc",
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.10, log=True),
            num_leaves=trial.suggest_int("num_leaves", 31, 255),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 20, 200),
            feature_fraction=trial.suggest_float("feature_fraction", 0.6, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.6, 1.0),
            bagging_freq=trial.suggest_int("bagging_freq", 0, 5),
            lambda_l1=trial.suggest_float("lambda_l1", 1e-3, 5.0, log=True),
            lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 5.0, log=True),
            max_bin=trial.suggest_categorical("max_bin", [127, 255, 511]),
            verbosity=-1, seed=harness.SEED, num_threads=8,
        )
        return _cv_eval(Xtr, y, folds, trial_fold_ids, params, cat_cols)

    sampler = optuna.samplers.TPESampler(seed=harness.SEED, multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    print(f"Optuna best 3-fold mean AUC: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")

    # Final 5-fold refit with the best params
    best_params = dict(
        objective="binary", metric="auc",
        verbosity=-1, seed=harness.SEED, num_threads=8,
        **study.best_params,
    )
    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(Xte), dtype=np.float64)
    fold_aucs: list[float] = []
    for k in range(harness.N_FOLDS):
        tr_m = folds != k
        va_m = folds == k
        dtr = lgb.Dataset(Xtr.iloc[tr_m], y[tr_m],
                          categorical_feature=cat_cols, free_raw_data=False)
        dva = lgb.Dataset(Xtr.iloc[va_m], y[va_m],
                          categorical_feature=cat_cols, reference=dtr,
                          free_raw_data=False)
        m = lgb.train(best_params, dtr, num_boost_round=4000,
                      valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False),
                                 lgb.log_evaluation(0)])
        oof[va_m] = m.predict(Xtr.iloc[va_m], num_iteration=m.best_iteration)
        test_pred += m.predict(Xte, num_iteration=m.best_iteration) / harness.N_FOLDS
        a = roc_auc_score(y[va_m], oof[va_m])
        fold_aucs.append(a)
        print(f"  fold {k}: best_iter={m.best_iteration}  AUC={a:.6f}")
    auc = float(roc_auc_score(y, oof))
    delta = auc - 0.960421   # current champion
    fold_str = ",".join(f"{a:.4f}" for a in fold_aucs)
    gate = float(json.load(open(os.path.join(ROOT, "logs", "step0_summary.json")))["gate"])
    raw_per_fold = harness.per_fold_auc(y, harness.load_oof("lgbm_full"), folds)
    n_fold_win = sum(1 for a, b in zip(fold_aucs, raw_per_fold) if a > b)
    if (delta >= gate) and (n_fold_win >= 4):
        result = "NEW CHAMPION"
    elif delta >= gate * 0.5:
        result = "BLEND MEMBER"
    else:
        result = "REJECTED"
    minutes = (time.time() - t0) / 60.0
    notes = (f"optuna_3fold_best={study.best_value:.6f} | per_fold_won="
             f"{n_fold_win}/5 vs lgbm_full (champion) gate={gate:.6f}")
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
        champion_auc=auc if result == "NEW CHAMPION" else 0.960421,
        blend_auc=0.0,
    )


if __name__ == "__main__":
    main()
