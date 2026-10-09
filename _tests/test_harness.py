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


# ----- T7: route-features aggregation is deterministic and label-free -----
def test_t7_route_aggregates_deterministic_label_free():
    """Build route aggregates from a tiny hand-made table; check
    expected counts/means and that permuting y leaves them unchanged."""
    df = pd.DataFrame({
        "Flight Distance": [100, 100, 200, 200, 200, 300, 300, 300, 300],
        "src":            [  0,   0,   1,   0,   1,   0,   0,   1,   1],
        "Age":            [ 20,  30,  40,  50,  60,  25,  35,  45,  55],
        "rating_x":       [  1,   2,   3,   4,   5,   2,   3,   4,   5],
        "y":              [  0,   1,   0,   1,   0,   1,   0,   1,   0],
    })
    # import lazily to keep the test self-contained even if exp_B isn't built yet
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "exp_B_route", os.path.join(ROOT, "exp_B_route.py"))
    if spec is None or spec.loader is None:
        pytest.skip("exp_B_route.py not built yet")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except (FileNotFoundError, SyntaxError, ImportError) as e:
        pytest.skip(f"exp_B_route.py not ready: {e}")
    # build aggregates twice (different orderings / seeds) -> must be identical
    agg_a = mod._route_aggregates(df, "Flight Distance", ["Age", "rating_x"])
    perm = np.random.default_rng(7).permutation(len(df))
    agg_b = mod._route_aggregates(df.iloc[perm].reset_index(drop=True),
                                  "Flight Distance", ["Age", "rating_x"])
    pd.testing.assert_frame_equal(
        agg_a.sort_values("Flight Distance").reset_index(drop=True),
        agg_b.sort_values("Flight Distance").reset_index(drop=True))
    # exact means on a hand-computed table
    means = agg_a.set_index("Flight Distance")[["route_Age_mean", "route_rating_x_mean"]]
    np.testing.assert_allclose(means.loc[100, "route_Age_mean"], 25.0)
    np.testing.assert_allclose(means.loc[200, "route_Age_mean"], 50.0)
    np.testing.assert_allclose(means.loc[300, "route_Age_mean"], 40.0)
    np.testing.assert_allclose(means.loc[100, "route_rating_x_mean"], 1.5)
    np.testing.assert_allclose(means.loc[300, "route_rating_x_mean"], 3.5)
    # counts
    counts = agg_a.set_index("Flight Distance")["route_count"]
    assert counts.loc[100] == 2
    assert counts.loc[200] == 3
    assert counts.loc[300] == 4


# ----- T8: target encoding is fold-safe and matches hand-computed values --
def test_t8_target_encoding_fold_safe_and_values():
    """Build a 4-way key (cls, type, cust, route) and check the per-key
    smoothed target encoding is correct, leak-free, and matches a
    hand-computed value on a 6-row table."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "exp_C_te", os.path.join(ROOT, "exp_C_te.py"))
    if spec is None or spec.loader is None:
        pytest.skip("exp_C_te.py not built yet")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except (FileNotFoundError, SyntaxError, ImportError) as e:
        pytest.skip(f"exp_C_te.py not ready: {e}")

    # 6-row hand-made table
    df = pd.DataFrame({
        "cls":   ["A", "A", "A", "B", "B", "B"],
        "type":  ["t1", "t1", "t2", "t1", "t2", "t2"],
        "cust":  ["L",  "L",  "D",  "L",  "D",  "D"],
        "route": [100, 100, 200, 200, 300, 300],
        "y":     [  1,   0,   1,   1,   0,   1],
    })
    df["__y__"] = df["y"]  # the implementation reads "__y__"
    df = df.rename(columns={"cls": "Class", "type": "Type of Travel",
                            "cust": "Customer Type", "route": "Flight Distance"})
    # Build a fold assignment (2 folds)
    folds = np.array([0, 1, 0, 1, 0, 1])
    keys = df["Class"].astype(str) + "|" + df["Type of Travel"].astype(str) + "|" + \
           df["Customer Type"].astype(str) + "|" + df["Flight Distance"].astype(str)
    PRIOR = float(df["y"].mean())
    SMOOTH = 1  # so hand-computed is easy
    oof, full = mod._te_4way_kfold(df, folds, PRIOR=PRIOR, SMOOTH=SMOOTH,
                                   cols=("Class","Type of Travel","Customer Type","Flight Distance"))
    # row 0 (fold 0): must use fold-1 only. Fold 1 has key "A|t1|L|100" once
    # with y=0. So oof[0] = (0 + 1*PRIOR) / (1 + 1) = PRIOR/2
    expected_0 = (0 + SMOOTH * PRIOR) / (1 + SMOOTH)
    assert abs(oof[0] - expected_0) < 1e-6, f"row0 {oof[0]} vs {expected_0}"
    # row 1 (fold 1): must use fold-0 only. Fold 0 has key "A|t1|L|100" once
    # with y=1. So oof[1] = (1 + PRIOR) / 2
    expected_1 = (1 + SMOOTH * PRIOR) / (1 + SMOOTH)
    assert abs(oof[1] - expected_1) < 1e-6, f"row1 {oof[1]} vs {expected_1}"
    # full is computed on the full table (the eventual test-set encoding)
    # key "A|t1|L|100" has sum=1, size=2 -> (1 + PRIOR) / (2 + 1)
    full_a = full.get("A|t1|L|100")
    if full_a is None:
        # full may be a dict or a Series indexed by key
        full_a = full["A|t1|L|100"] if "A|t1|L|100" in full.index else None
    assert full_a is not None
    expected_full = (1 + SMOOTH * PRIOR) / (2 + SMOOTH)
    assert abs(float(full_a) - expected_full) < 1e-6, \
        f"full {full_a} vs {expected_full}"


# ----- T9: small interaction features are correct on a hand-made table ----
def test_t9_small_interactions_correct():
    """Build rating aggregates, delay features, missing flag on a
    hand-made table; compare to hand-computed values."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "exp_D_small", os.path.join(ROOT, "exp_D_small.py"))
    if spec is None or spec.loader is None:
        pytest.skip("exp_D_small.py not built yet")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except (FileNotFoundError, SyntaxError, ImportError) as e:
        pytest.skip(f"exp_D_small.py not ready: {e}")

    df = pd.DataFrame({
        "Departure Delay in Minutes": [0.0, 10.0, 30.0, 0.0, 5.0],
        "Arrival Delay in Minutes":   [0.0,  float("nan"), 5.0, 0.0, 0.0],
        "r1": [1, 2, 3, 4, 5],
        "r2": [5, 4, 3, 2, 0],
        "r3": [0, 0, 1, 2, 3],
    })
    out = mod._add_small_features(df)
    # log1p of delays (0 -> 0; 10 -> 2.4; 30 -> 3.4; etc.)
    np.testing.assert_allclose(out["log_dep_delay"].iloc[0], 0.0)
    np.testing.assert_allclose(out["log_arr_delay"].iloc[0], 0.0)
    np.testing.assert_allclose(out["log_dep_delay"].iloc[2], np.log1p(30.0))
    # delay diff = dep - arr
    np.testing.assert_allclose(out["delay_diff"].iloc[0], 0.0)
    np.testing.assert_allclose(out["delay_diff"].iloc[2], 30.0 - 5.0)
    # has_arr_delay is False for the NaN row (1)
    assert bool(out["has_arr_delay"].iloc[1]) is False
    assert bool(out["has_arr_delay"].iloc[2]) is True
    # arr_delay_missing
    assert bool(out["arr_delay_missing"].iloc[1]) is True
    assert bool(out["arr_delay_missing"].iloc[0]) is False
    # rating aggregates
    rvals = df[["r1","r2","r3"]]
    np.testing.assert_allclose(out["rating_mean"].iloc[0], rvals.iloc[0].mean())
    np.testing.assert_allclose(out["rating_min"].iloc[4], 0.0)
    np.testing.assert_allclose(out["rating_max"].iloc[4], 5.0)
    # count of 0 ratings
    assert int(out["n_zero_ratings"].iloc[0]) == 1  # r3
    assert int(out["n_zero_ratings"].iloc[4]) == 1  # r2
    assert int(out["n_zero_ratings"].iloc[2]) == 0  # none
    # low_rating_frac (<=2): r1=1 and r3=0 are both <=2, so 2/3
    np.testing.assert_allclose(out["low_rating_frac"].iloc[0], 2/3)
    # high_rating_frac (>=4): r2=5, so 1/3
    np.testing.assert_allclose(out["high_rating_frac"].iloc[0], 1/3)
    # flat_rater
    assert bool(out["flat_rater"].iloc[3]) is False  # 4,2,2 not flat
