"""Experiment C: Target encoding of Flight Distance and 4-way (Class, Type
of Travel, Customer Type, Flight Distance), nested inside each training
fold, smoothed (m=20-50).

Hypothesis: the 4-way key directly targets the 0.83-AUC "Personal Travel"
weakness. A smoothed mean target for that key gives the model a strong
prior, especially for the long tail of rare keys.

TDD: `_te_4way_kfold` is tested in `_tests/test_harness.py` (T8).
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

EXP_ID = "2_C_te"
EXP_NAME = "lgbm_full_te"
CHANGE = "Add nested 4-way TE (Class,Type,Cust,FlightDist), smooth=20, plus route+FD TE"


# ----- target encoding (tested in T8) -------------------------------------
def _te_4way_kfold(df: pd.DataFrame, folds: np.ndarray,
                   PRIOR: float, SMOOTH: float = 20,
                   cols: tuple[str, ...] = ("Class", "Type of Travel",
                                            "Customer Type", "Flight Distance")
                   ) -> tuple[np.ndarray, pd.Series]:
    """Nested 5-fold smoothed target encoding for a multi-column key.

    Returns (oof_array, full_rate_series) where `oof_array` is the OOF
    encoding for each row of `df` (length N), and `full_rate_series` is
    the smoothed per-key mean target computed on ALL rows (used to encode
    the test set).
    """
    keys = df[list(cols)].astype(str).agg("|".join, axis=1).values
    y = df["__y__"].values
    n = len(df)
    oof = np.full(n, PRIOR, dtype=np.float64)
    for k in range(int(folds.max()) + 1):
        tr = folds != k; va = folds == k
        sk = pd.DataFrame({"k": keys[tr], "y": y[tr]}).groupby("k",
                                                               observed=True)["y"].agg(["sum","size"])
        r = (sk["sum"] + SMOOTH * PRIOR) / (sk["size"] + SMOOTH)
        mp = r.to_dict()
        oof[va] = pd.Series(keys[va]).map(mp).fillna(PRIOR).values
    # full encoding on the full table
    g_full = pd.DataFrame({"k": keys, "y": y}).groupby("k",
                                                       observed=True)["y"].agg(["sum","size"])
    rate_full = (g_full["sum"] + SMOOTH * PRIOR) / (g_full["size"] + SMOOTH)
    return oof, rate_full


def _te_flight_dist_kfold(df: pd.DataFrame, folds: np.ndarray,
                          PRIOR: float, SMOOTH: float = 20) -> tuple[np.ndarray, pd.Series]:
    """Per-Flight Distance smoothed target encoding (single key)."""
    return _te_4way_kfold(df, folds, PRIOR=PRIOR, SMOOTH=SMOOTH,
                          cols=("Flight Distance",))


def _attach_te(tr: pd.DataFrame, te: pd.DataFrame, folds: np.ndarray,
               PRIOR: float, SMOOTH: float = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add TE columns to train and test (per-row OOF for train, full mean for test)."""
    oof4, full4 = _te_4way_kfold(tr, folds, PRIOR=PRIOR, SMOOTH=SMOOTH)
    tr = tr.copy()
    te = te.copy()
    tr["te_4way"] = oof4.astype("float32")
    # encode test by mapping its key
    keys_te = te[["Class", "Type of Travel", "Customer Type",
                  "Flight Distance"]].astype(str).agg("|".join, axis=1).values
    te["te_4way"] = pd.Series(keys_te).map(full4.to_dict()).fillna(PRIOR).astype("float32").values

    # also a 1D Flight Distance TE
    oof1, full1 = _te_flight_dist_kfold(tr, folds, PRIOR=PRIOR, SMOOTH=SMOOTH)
    tr["te_fd"] = oof1.astype("float32")
    keys_te1 = te["Flight Distance"].astype(str).values
    te["te_fd"] = pd.Series(keys_te1).map(full1.to_dict()).fillna(PRIOR).astype("float32").values
    return tr, te


# ----- model ---------------------------------------------------------------
def lgbm_full_te_5fold(name: str, seed: int = 42) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Champion features (route aggregates) + nested-CV target encoding."""
    tr, te = harness.load_train_test()
    folds = harness.load_folds(tr)
    y = tr["__y__"].values
    # route features
    tr_enr, te_enr = exp_B_route.build_route_features(tr, te)
    # attach target encoding
    PRIOR = float(y.mean())
    tr_enr, te_enr = _attach_te(tr_enr, te_enr, folds, PRIOR=PRIOR, SMOOTH=20)
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
        exp_id=EXP_ID,
        change=CHANGE,
        oof_auc=auc,
        delta=delta,
        per_fold_aucs=fold_str,
        blend_auc=0.0,
        runtime_min=minutes,
        result=result,
        notes=notes,
    )
    harness.append_experiment(exp)
    np.save(harness.oof_path(name), oof.astype(np.float32))
    np.save(harness.pred_path(name), test_pred.astype(np.float32))
    return {
        "auc": auc, "delta": delta, "result": result, "fold_aucs": fold_aucs,
        "n_fold_win": n_fold_win, "per_fold_champ": raw_per_fold,
    }


def main() -> None:
    print(f"\n=== Experiment C: {CHANGE} ===")
    t0 = time.time()
    oof, test_pred, fold_aucs = lgbm_full_te_5fold(EXP_NAME, seed=harness.SEED)
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
