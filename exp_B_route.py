"""Experiment B: Flight Distance group features.

Hypothesis: "same route" passengers behave similarly, but the model has no
route context. Adding route aggregates gives the model that signal.

What this script does:
1. Build route aggregates (count, per-source count, mean/std of each
   rating + Age, per-source category share) on train + test (no labels).
2. Train a 5-fold LGBM with the augmented feature set, save OOF + test.
3. Compare against the `lgbm_raw` champion (0.958810).
4. Append to `experiments.csv`; update `STATE.md`; write submissions.

TDD: `_route_aggregates` is tested in `_tests/test_harness.py` (T7).
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

EXP_ID = "1_B_route"
EXP_NAME = "lgbm_full"  # artifact name (s6e10 convention)
CHANGE = "Flight Distance group features (count, src counts, mean/std ratings+Age, cat shares)"


# ----- feature function (tested in T7) ------------------------------------
def _route_aggregates(df: pd.DataFrame, route_col: str,
                      agg_cols: list[str]) -> pd.DataFrame:
    """Per-route aggregates: count + mean + std of each `agg_cols` column.

    Deterministic and label-free (only uses the columns named).
    """
    g = df.groupby(route_col, observed=True)
    out = pd.DataFrame({route_col: g.size().index})
    out["route_count"] = g.size().values
    for c in agg_cols:
        out[f"route_{c}_mean"] = g[c].mean().values
        out[f"route_{c}_std"]  = g[c].std().fillna(0).values
    return out


def _per_source_counts(df: pd.DataFrame, route_col: str,
                       src_col: str = "__src__") -> pd.DataFrame:
    """Per-(route, source) counts. Useful to detect routes that show up
    mostly in test (data drift)."""
    g = df.groupby([route_col, src_col], observed=True).size().unstack(fill_value=0)
    g.columns = [f"route_src{int(c)}_count" for c in g.columns]
    return g.reset_index()


def _per_route_cat_shares(df: pd.DataFrame, route_col: str,
                          cat_cols: list[str]) -> pd.DataFrame:
    """Per-route one-hot shares for each value of each cat column."""
    out = pd.DataFrame({route_col: df[route_col].drop_duplicates().values})
    for c in cat_cols:
        dummies = pd.get_dummies(df[c].astype(str), prefix=c).astype("int8")
        dummies[route_col] = df[route_col].values
        agg = dummies.groupby(route_col, observed=True).mean()
        agg.columns = [f"route_share_{col}" for col in agg.columns]
        out = out.merge(agg.reset_index(), on=route_col, how="left")
    return out


def build_route_features(tr: pd.DataFrame, te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the route features for train and test. Label-free.

    Returns two DataFrames (train_enriched, test_enriched) with the same
    base columns plus the new route_* columns.
    """
    RATING = harness.rating_cols(tr)
    ROUTE = "Flight Distance"
    RAW = harness.raw_feats(tr)
    all_raw = pd.concat([
        tr[RAW].assign(__src__=0),
        te[RAW].assign(__src__=1),
    ], ignore_index=True)
    agg_cols = RATING + ["Age"]
    agg = _route_aggregates(all_raw, ROUTE, agg_cols)
    src = _per_source_counts(all_raw, ROUTE)
    cats = _per_route_cat_shares(all_raw, ROUTE, harness.CAT_COLS)
    # merge in order: agg, src, cats
    tr_out = tr.merge(agg, on=ROUTE, how="left")
    tr_out = tr_out.merge(src, on=ROUTE, how="left")
    tr_out = tr_out.merge(cats, on=ROUTE, how="left")
    te_out = te.merge(agg, on=ROUTE, how="left")
    te_out = te_out.merge(src, on=ROUTE, how="left")
    te_out = te_out.merge(cats, on=ROUTE, how="left")
    return tr_out, te_out


# ----- model ---------------------------------------------------------------
def lgbm_full_5fold(name: str, seed: int = 42) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Same hyperparameters as `lgbm_raw` (s6e10_solution defaults)."""
    tr, te = harness.load_train_test()
    folds = harness.load_folds(tr)
    y = tr["__y__"].values
    tr_enr, te_enr = build_route_features(tr, te)
    feats = [c for c in tr_enr.columns
             if c not in ("id", "satisfaction", "__y__", "__src__")]
    # ensure categoricals
    Xtr = tr_enr[feats].copy()
    Xte = te_enr[feats].copy()
    for c in harness.CAT_COLS:
        if c in Xtr.columns:
            Xtr[c] = Xtr[c].astype("category")
            Xte[c] = Xte[c].astype("category")
    # align categories
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
    """Append a row to experiments.csv and return summary dict."""
    y = harness.load_train_test()[0]["__y__"].values
    auc = float(roc_auc_score(y, oof))
    delta = auc - champion_auc
    fold_str = ",".join(f"{a:.4f}" for a in fold_aucs)
    # determine result vs GATE
    gate = float(json.load(open(os.path.join(ROOT, "logs", "step0_summary.json")))["gate"])
    n_folds_improved = sum(
        1 for a in fold_aucs
        if a > 0  # placeholder; we compare per-fold below
    )
    # Decide the result
    improved = (delta >= gate)
    n_fold_win = 0
    raw_oof = harness.load_oof("lgbm_raw")
    raw_per_fold = harness.per_fold_auc(y, raw_oof, harness.load_folds(harness.load_train_test()[0]))
    for a, b in zip(fold_aucs, raw_per_fold):
        if a > b:
            n_fold_win += 1
    if improved and n_fold_win >= 4:
        result = "NEW CHAMPION"
    elif improved:
        result = "BLEND MEMBER"
    else:
        result = "REJECTED"
    notes = (f"per_fold_won={n_fold_win}/5 vs raw "
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
        "n_fold_win": n_fold_win, "per_fold_raw": raw_per_fold,
    }


def main() -> None:
    print(f"\n=== Experiment B: {CHANGE} ===")
    t0 = time.time()
    oof, test_pred, fold_aucs = lgbm_full_5fold(EXP_NAME, seed=harness.SEED)
    minutes = (time.time() - t0) / 60.0
    summary = log_round(EXP_NAME, oof, test_pred, fold_aucs, minutes,
                        champion_auc=0.958810)
    auc = summary["auc"]
    delta = summary["delta"]
    result = summary["result"]
    print(f"\n{EXP_NAME}: OOF AUC = {auc:.6f}  delta {delta:+.6f}  "
          f"({minutes:.1f} min)  {result}")
    print(f"per-fold: {[round(a, 4) for a in fold_aucs]}")
    # status line in the required format
    harness.status_line(
        exp_id=EXP_ID,
        change=CHANGE,
        oof_auc=auc,
        delta=delta,
        result=result,
        champion_auc=auc if result == "NEW CHAMPION" else 0.958810,
        blend_auc=0.0,
    )


if __name__ == "__main__":
    main()
