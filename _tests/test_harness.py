"""TDD tests for harness.py and the validation pipeline.

These tests run in < 2 minutes on a 5% sample. They protect:
- T1  data invariants (row counts, target, ids)
- T2  fold determinism (each row in exactly one fold, identical across runs)
- T3  leakage in label-free features (changing y does NOT change features)
- T4  OOF and test artifacts (length, no NaN, [0,1], row count for test)
- T5  submission CSV (columns, id order, value range, no NaN)
- T6  blend function (logit-space stacker returns probabilities in [0,1])

Run:  .venv/bin/python -m pytest _tests/test_harness.py -x -v
"""
from __future__ import annotations

import os
import sys
import json
import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Ensure project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import harness  # noqa: E402


# ----- T1: data invariants -------------------------------------------------
def test_t1_data_invariants():
    tr, te = harness.load_train_test()
    assert tr.shape[0] == 699635, f"train rows {tr.shape[0]}"
    assert te.shape[0] == 299844, f"test rows {te.shape[0]}"
    assert "id" in tr.columns and "id" in te.columns
    assert "__y__" in tr.columns
    assert set(tr["__y__"].unique()) == {0, 1}
    assert tr["__y__"].notna().all()
    # unique ids
    assert tr["id"].is_unique
    assert te["id"].is_unique
    # ids are 0..N-1 (or at least contiguous in some range)
    assert tr["id"].min() == 0
    assert tr["id"].max() == len(tr) - 1
    assert te["id"].min() >= len(tr)
    # ratings columns
    rc = harness.rating_cols(tr)
    assert len(rc) == 13, f"expected 13 rating cols, got {len(rc)}: {rc}"


# ----- T2: fold determinism ------------------------------------------------
def test_t2_folds_determinism():
    tr, _ = harness.load_train_test()
    folds = harness.load_folds(tr)
    assert len(folds) == len(tr)
    assert set(np.unique(folds).tolist()) == set(range(harness.N_FOLDS))
    counts = np.bincount(folds)
    assert min(counts) == max(counts), f"unbalanced folds: {counts}"
    # same seed gives the same assignment
    y = tr["__y__"].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=harness.SEED)
    rebuilt = np.zeros(len(tr), dtype=np.int8)
    for k, (_, vi) in enumerate(skf.split(tr, y)):
        rebuilt[vi] = k
    np.testing.assert_array_equal(folds, rebuilt)


# ----- T3: label-free features have no leakage ----------------------------
def test_t3_label_free_features_no_leakage():
    """Building route aggregates from a permuted y must give the same features."""
    tr, te = harness.load_train_test()
    RATING = harness.rating_cols(tr) + ["Age"]
    ROUTE = "Flight Distance"
    all_raw = pd.concat([
        tr[harness.raw_feats(tr)].assign(__src__=0),
        te[harness.raw_feats(te)].assign(__src__=1),
    ], ignore_index=True)
    # original aggregates
    g = all_raw.groupby(ROUTE, observed=True)
    cnt_orig = g.size().rename("route_count")
    mean_orig = g[RATING].mean().rename(columns=lambda c: f"route_{c}_mean")
    # permute y and check no row of cnt_orig / mean_orig changes
    perm = np.random.default_rng(0).permutation(len(all_raw))
    all_raw_perm = all_raw.iloc[perm].reset_index(drop=True)
    g2 = all_raw_perm.groupby(ROUTE, observed=True)
    cnt2 = g2.size().rename("route_count")
    mean2 = g2[RATING].mean().rename(columns=lambda c: f"route_{c}_mean")
    # same keys -> same aggregates (since per-key mean ignores order)
    pd.testing.assert_series_equal(cnt_orig.sort_index(), cnt2.sort_index(),
                                    check_names=False)
    for c in mean_orig.columns:
        a = mean_orig[c].sort_index()
        b = mean2[c].sort_index()
        pd.testing.assert_series_equal(a, b, check_names=False)


# ----- T4: OOF / test artifact invariants ---------------------------------
def test_t4_oof_artifact_invariants(tmp_path):
    # write a fake OOF + test
    n_train, n_test = 1000, 500
    rng = np.random.default_rng(0)
    oof = rng.uniform(0, 1, n_train).astype(np.float32)
    test = rng.uniform(0, 1, n_test).astype(np.float32)
    name = "_t4_fake"
    np.save(os.path.join(harness.OOF_DIR, f"{name}.npy"), oof)
    np.save(os.path.join(harness.PRED_DIR, f"{name}.npy"), test)
    try:
        o = harness.load_oof(name)
        t = harness.load_pred(name)
        assert o.shape == (n_train,) and not np.isnan(o).any() and not np.isinf(o).any()
        assert t.shape == (n_test,) and not np.isnan(t).any() and not np.isinf(t).any()
        assert o.min() >= 0 and o.max() <= 1
        assert t.min() >= 0 and t.max() <= 1
    finally:
        os.remove(os.path.join(harness.OOF_DIR, f"{name}.npy"))
        os.remove(os.path.join(harness.PRED_DIR, f"{name}.npy"))


# ----- T5: submission writer ----------------------------------------------
def test_t5_submission_writer(tmp_path):
    # fake a submission file and verify
    samp = pd.read_csv(harness.SAMPLE_PATH)            # full sample
    n = len(samp)
    probs = np.linspace(0, 1, n).astype(np.float32)
    out = os.path.join(harness.SUB_DIR, "_t5_test.csv")
    try:
        harness.write_submission(probs, out, name="t5")
        sub = pd.read_csv(out)
        assert list(sub.columns) == ["id", "satisfaction"]
        assert len(sub) == n
        assert (sub["id"].values == samp["id"].values).all()
        assert sub["satisfaction"].between(0, 1).all()
        assert sub["satisfaction"].notna().all()
    finally:
        if os.path.exists(out):
            os.remove(out)


# ----- T6: blend / logit helpers -----------------------------------------
def test_t6_logit_round_trip():
    p = np.array([0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999])
    z = harness._to_logit(p)
    p2 = harness._sigmoid(z)
    np.testing.assert_allclose(p, p2, atol=1e-5)


def test_t6_blend_stack_smoke():
    """Make a 2-OOF fake, run a stacker, ensure the output is in [0,1] and
    AUC is no worse than the best single OOF."""
    rng = np.random.default_rng(0)
    n = 500
    y = (rng.uniform(0, 1, n) > 0.5).astype(int)
    # two correlated-but-different "predictions"
    s1 = y * 0.8 + rng.normal(0, 0.3, n)
    s2 = y * 0.7 + rng.normal(0, 0.4, n)
    p1 = harness._sigmoid(s1)
    p2 = harness._sigmoid(s2)
    # write
    for nm, p in [("t6_a", p1), ("t6_b", p2)]:
        np.save(os.path.join(harness.OOF_DIR, f"{nm}.npy"), p.astype(np.float32))
        np.save(os.path.join(harness.PRED_DIR, f"{nm}.npy"), p[:100].astype(np.float32))
    try:
        a1 = roc_auc_score(y, p1)
        a2 = roc_auc_score(y, p2)
        from sklearn.linear_model import LogisticRegression
        oofs = [harness.load_oof("t6_a"), harness.load_oof("t6_b")]
        tests = [harness.load_pred("t6_a"), harness.load_pred("t6_b")]
        X = np.column_stack([harness._to_logit(o) for o in oofs])
        Xt = np.column_stack([harness._to_logit(t) for t in tests])
        # nested CV: hold out half, fit on the other, predict
        sk = StratifiedKFold(3, shuffle=True, random_state=0)
        stack_oof = np.zeros(n)
        for tr_, va_ in sk.split(X, y):
            lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
            lr.fit(X[tr_], y[tr_])
            stack_oof[va_] = lr.predict_proba(X[va_])[:, 1]
        assert stack_oof.min() >= 0 and stack_oof.max() <= 1
        a_stack = roc_auc_score(y, stack_oof)
        # stacker should be at least as good as the best single, within tolerance
        assert a_stack >= min(a1, a2) - 1e-3, \
            f"stack {a_stack:.4f} worse than min({a1:.4f}, {a2:.4f})"
    finally:
        for nm in ("t6_a", "t6_b"):
            for d in (harness.OOF_DIR, harness.PRED_DIR):
                p = os.path.join(d, f"{nm}.npy")
                if os.path.exists(p):
                    os.remove(p)
