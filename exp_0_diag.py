"""STEP 0 diagnostics: sanity check, noise check, adversarial validation.

Run once at the start of the loop. After this script:
- oof/lgbm_raw.npy and preds/lgbm_raw.npy exist (5-fold raw LGBM)
- GATE is computed and recorded in STATE.md
- adversarial validation AUC is recorded
"""
from __future__ import annotations

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import harness  # noqa: E402


def cast_cats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in harness.CAT_COLS:
        if c in out.columns:
            out[c] = out[c].astype("category")
    return out


def lgbm_raw_5fold(seed: int = 42) -> tuple[np.ndarray, np.ndarray, list[float]]:
    tr, te = harness.load_train_test()
    folds = harness.load_folds(tr)
    y = tr["__y__"].values
    RAW = harness.raw_feats(tr)
    Xtr = cast_cats(tr[RAW])
    Xte = cast_cats(te[RAW])
    # align categories across train/test
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


def adversarial_validation() -> float:
    """Train LGBM to tell train from test. AUC near 0.5 == folds are
    trustworthy. AUC > 0.7 is a red flag."""
    tr, te = harness.load_train_test()
    RAW = harness.raw_feats(tr)
    Xtr = cast_cats(tr[RAW])
    Xte = cast_cats(te[RAW])
    Xtr, Xte = harness.make_aligned_categoricals(Xtr, Xte)
    X = pd.concat([Xtr, Xte], ignore_index=True)
    y_adv = np.concatenate([np.zeros(len(Xtr)), np.ones(len(Xte))]).astype(int)
    cat_cols = [c for c in X.columns if str(X[c].dtype) == "category"]
    skf = __import__("sklearn.model_selection", fromlist=["StratifiedKFold"]).StratifiedKFold(
        n_splits=5, shuffle=True, random_state=harness.SEED)
    oof = np.zeros(len(X))
    params = dict(objective="binary", metric="auc", learning_rate=0.05,
                  num_leaves=63, min_data_in_leaf=50, feature_fraction=0.85,
                  bagging_fraction=0.85, bagging_freq=1, verbosity=-1,
                  seed=harness.SEED, num_threads=8)
    for tr_m, va_m in skf.split(X, y_adv):
        dtr = lgb.Dataset(X.iloc[tr_m], y_adv[tr_m],
                          categorical_feature=cat_cols, free_raw_data=False)
        dva = lgb.Dataset(X.iloc[va_m], y_adv[va_m],
                          categorical_feature=cat_cols, reference=dtr,
                          free_raw_data=False)
        m = lgb.train(params, dtr, num_boost_round=400, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(0)])
        oof[va_m] = m.predict(X.iloc[va_m], num_iteration=m.best_iteration)
    auc = roc_auc_score(y_adv, oof)
    return float(auc)


def noise_check(seeds: list[int]) -> tuple[list[float], float]:
    """Run lgbm_raw with different seeds; report AUCs and stdev."""
    aucs = []
    for s in seeds:
        oof, _, fa = lgbm_raw_5fold(seed=s)
        a = roc_auc_score(*_y_for_oof(oof))
        aucs.append(a)
        print(f"  seed {s}: OOF AUC = {a:.6f}  per-fold = {[round(x, 4) for x in fa]}")
    return aucs, float(np.std(aucs, ddof=1))


def _y_for_oof(oof: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tr, _ = harness.load_train_test()
    return tr["__y__"].values, oof


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-noise", action="store_true",
                    help="Skip the 3-seed noise check (saves ~5 min).")
    ap.add_argument("--skip-adv", action="store_true",
                    help="Skip adversarial validation.")
    args = ap.parse_args()

    print("\n=== STEP 0: Sanity check (lgbm_raw, full 5-fold) ===")
    t0 = time.time()
    oof, test_pred, fold_aucs = lgbm_raw_5fold(seed=harness.SEED)
    minutes = (time.time() - t0) / 60.0
    y = _y_for_oof(oof)[0]
    auc = float(roc_auc_score(y, oof))
    print(f"\nlgbm_raw: OOF AUC = {auc:.6f}  fold_std = {np.std(fold_aucs):.5f}  "
          f"({minutes:.1f} min)")
    np.save(harness.oof_path("lgbm_raw"), oof.astype(np.float32))
    np.save(harness.pred_path("lgbm_raw"), test_pred.astype(np.float32))

    # noise check
    gate = 0.00005
    if not args.skip_noise:
        print("\n=== STEP 0: Noise check (3 seeds) ===")
        _, stdev = noise_check([harness.SEED, harness.SEED + 11, harness.SEED + 22])
        gate = max(0.00005, 2.0 * stdev)
        print(f"\nStdev of OOF AUC across seeds: {stdev:.6f}  GATE = {gate:.6f}")
    else:
        print("\n=== STEP 0: Noise check SKIPPED (GATE = default 0.00005) ===")

    # adversarial validation
    adv_auc = float("nan")
    if not args.skip_adv:
        print("\n=== STEP 0: Adversarial validation ===")
        adv_auc = adversarial_validation()
        print(f"adversarial AUC = {adv_auc:.4f}  (near 0.5 == folds trustworthy)")

    # write a small summary file so STATE can read it
    summary = {
        "lgbm_raw_oof_auc": auc,
        "lgbm_raw_runtime_min": minutes,
        "gate": gate,
        "adv_auc": adv_auc,
    }
    with open(os.path.join(ROOT, "logs", "step0_summary.json"), "w") as f:
        import json
        json.dump(summary, f, indent=2)
    print("\nWrote logs/step0_summary.json")


if __name__ == "__main__":
    main()
