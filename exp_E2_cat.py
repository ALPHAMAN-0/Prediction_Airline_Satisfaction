"""Experiment E2: CatBoost on the route-enriched features.

CatBoost handles categoricals natively (ordered TS) which gives the
ensemble a structurally different member from LGBM/XGB. CPU-only on
Apple Silicon; 5-10x slower than LGBM, so we start with depth=4 and
~2000 iters.

Hypothesis: even if a single AUC is below the LGBM champion, the
diversity from CatBoost's ordered boosting + native cat handling
should help the blend.

Saves to oof/cat_d4.npy, preds/cat_d4.npy.
"""
from __future__ import annotations

import os
import sys
import time
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import harness  # noqa: E402
import exp_B_route  # noqa: E402
import exp_D_small  # noqa: E402

EXP_ID = "6_E2_cat"
EXP_NAME = "cat_d4"
CHANGE = "CatBoost depth=4, lr=0.05, 2000 iters on route-enriched features"


def cat_d_5fold(name: str, depth: int = 4, learning_rate: float = 0.05,
                iterations: int = 2000, seed: int = 42
                ) -> tuple[np.ndarray, np.ndarray, list[float]]:
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
    # CatBoost wants categoricals as strings (no NaN, no pandas category dtype)
    cat_cols = [c for c in Xtr.columns if c in harness.CAT_COLS
                or str(Xtr[c].dtype) == "category"]
    for c in cat_cols:
        Xtr[c] = Xtr[c].astype(str).fillna("nan")
        Xte[c] = Xte[c].astype(str).fillna("nan")
    # ensure column order is identical (CatBoost is positional for cat_features)
    Xtr = Xtr[feats]
    Xte = Xte[feats]
    cat_idx = [feats.index(c) for c in cat_cols if c in feats]
    oof = np.zeros(len(tr), dtype=np.float64)
    test_pred = np.zeros(len(te), dtype=np.float64)
    fold_aucs: list[float] = []
    for k in range(harness.N_FOLDS):
        tr_m = folds != k
        va_m = folds == k
        train_pool = Pool(Xtr.iloc[tr_m], y[tr_m], cat_features=cat_idx)
        val_pool = Pool(Xtr.iloc[va_m], y[va_m], cat_features=cat_idx)
        model = CatBoostClassifier(
            iterations=iterations, learning_rate=learning_rate, depth=depth,
            l2_leaf_reg=3.0, random_seed=seed, verbose=0,
            eval_metric="AUC", loss_function="Logloss",
            early_stopping_rounds=200, task_type="CPU",
            thread_count=8,
        )
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        oof[va_m] = model.predict_proba(val_pool)[:, 1]
        test_pool = Pool(Xte, cat_features=cat_idx)
        test_pred += model.predict_proba(test_pool)[:, 1] / harness.N_FOLDS
        a = roc_auc_score(y[va_m], oof[va_m])
        fold_aucs.append(a)
        bi = model.get_best_iteration()
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
    print(f"\n=== Experiment E2: {CHANGE} ===")
    t0 = time.time()
    oof, test_pred, fold_aucs = cat_d_5fold(EXP_NAME, depth=4, learning_rate=0.05,
                                            iterations=2000, seed=harness.SEED)
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
