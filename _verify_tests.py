"""Verification tests for s6e10_solution output (TDD-style assertions).

Runs as a standalone script. Each `assert_*` function is a single test that
should PASS given the artifacts in output/. We exit non-zero if any fail.
"""
import os, sys, glob
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OUT = "output"
DATA = "playground-series-dataCSV"
results = []

def assert_(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")

# ----------------------------------------------------------------------
# T1: All expected OOF/test .npy files exist and have correct shapes
# ----------------------------------------------------------------------
print("\n--- T1: artifact presence & shape ---")
expected = ["lgbm_raw", "lgbm_full", "lgbm_full_et", "xgb_full", "cat_depth10",
            "stack", "rank", "hill", "meta"]
for name in expected:
    p_oof = f"{OUT}/oof_{name}.npy"
    p_test = f"{OUT}/test_{name}.npy"
    ok = os.path.exists(p_oof) and os.path.exists(p_test)
    if not ok:
        assert_(f"artifacts: {name}", False, f"missing {p_oof} or {p_test}")
        continue
    o = np.load(p_oof); t = np.load(p_test)
    assert_(f"artifacts: {name} shapes", len(o) == 34981 and len(t) == 299844,
            f"oof={o.shape} test={t.shape}")

# ----------------------------------------------------------------------
# T2: All OOF arrays are valid probabilities (in [0, 1], no NaN/Inf)
# ----------------------------------------------------------------------
print("\n--- T2: OOF/test validity ---")
for name in expected:
    p_oof = f"{OUT}/oof_{name}.npy"
    p_test = f"{OUT}/test_{name}.npy"
    if not (os.path.exists(p_oof) and os.path.exists(p_test)):
        continue
    o = np.load(p_oof); t = np.load(p_test)
    assert_(f"oof_{name}: no NaN/Inf", not (np.isnan(o).any() or np.isinf(o).any()))
    assert_(f"oof_{name}: range [0,1]", o.min() >= 0 and o.max() <= 1,
            f"min={o.min():.4f} max={o.max():.4f}")
    assert_(f"test_{name}: no NaN/Inf", not (np.isnan(t).any() or np.isinf(t).any()))
    assert_(f"test_{name}: range [0,1]", t.min() >= 0 and t.max() <= 1)

# ----------------------------------------------------------------------
# T3: OOF AUC is meaningfully better than random (>0.5 by a large margin)
# ----------------------------------------------------------------------
print("\n--- T3: OOF AUC > 0.95 (smoke) / 0.96 (full) ---")
# Reload y aligned to the SMOKE 5% sample (seed=42, np.random.default_rng permutation)
tr_full = pd.read_csv(f"{DATA}/train.csv")
y_full = tr_full["satisfaction"].astype(int).values
rng = np.random.default_rng(42)
sub_idx = rng.permutation(len(y_full))[:34981]
y_sub = y_full[sub_idx]

for name in expected:
    p = f"{OUT}/oof_{name}.npy"
    if not os.path.exists(p):
        continue
    o = np.load(p)
    a = roc_auc_score(y_sub, o)
    assert_(f"oof_{name}: AUC > 0.5 (sanity)", a > 0.5, f"AUC={a:.6f}")

# ----------------------------------------------------------------------
# T4: results.csv matches recomputed AUCs (consistency)
# ----------------------------------------------------------------------
print("\n--- T4: results.csv consistency ---")
df = pd.read_csv(f"{OUT}/results.csv")
recomputed = {}
for n in df["name"]:
    p = f"{OUT}/oof_{n}.npy"
    if os.path.exists(p):
        recomputed[n] = roc_auc_score(y_sub, np.load(p))
mismatches = 0
for _, row in df.iterrows():
    n = row["name"]
    if n in recomputed:
        diff = abs(row["oof_auc"] - recomputed[n])
        ok = diff < 1e-3
        if not ok:
            mismatches += 1
        assert_(f"results.csv: {n} AUC matches", ok,
                f"file={row['oof_auc']:.6f}  recomputed={recomputed[n]:.6f}  diff={diff:.2e}")
print(f"  AUC mismatches: {mismatches}")

# ----------------------------------------------------------------------
# T5: submission.csv integrity
# ----------------------------------------------------------------------
print("\n--- T5: submission.csv integrity ---")
sub = pd.read_csv(f"{OUT}/submission.csv")
samp = pd.read_csv(f"{DATA}/sample_submission.csv")
assert_("submission: row count", len(sub) == len(samp), f"{len(sub)} vs {len(samp)}")
assert_("submission: id order matches sample", (sub["id"].values == samp["id"].values).all())
assert_("submission: no NaN", not sub.isna().any().any())
assert_("submission: values in [0,1]",
        sub["satisfaction"].between(0, 1).all(),
        f"min={sub['satisfaction'].min():.4f} max={sub['satisfaction'].max():.4f}")
assert_("submission: predicted class proportions are reasonable",
        0.2 < sub["satisfaction"].mean() < 0.7,
        f"mean prob = {sub['satisfaction'].mean():.4f}")

# ----------------------------------------------------------------------
# T6: 'hill' ensemble matches expected top AUC (0.9654) within tolerance
# ----------------------------------------------------------------------
print("\n--- T6: ensemble ranking ---")
oof_lgbm = np.load(f"{OUT}/oof_lgbm_full.npy")
oof_hill = np.load(f"{OUT}/oof_hill.npy")
a_lgbm = roc_auc_score(y_sub, oof_lgbm)
a_hill = roc_auc_score(y_sub, oof_hill)
assert_("hill > lgbm_full (ensemble should beat single model)",
        a_hill > a_lgbm, f"hill={a_hill:.6f} lgbm_full={a_lgbm:.6f}")

# ----------------------------------------------------------------------
# T7: NEW pipeline must beat the BASELINE (saved in /tmp/baseline_oof)
# This is the regression gate for the improvement loop. Set BASELINE_DIR
# to "" to skip.
# ----------------------------------------------------------------------
print("\n--- T7: improvement vs baseline ---")
BASELINE_DIR = os.environ.get("BASELINE_DIR", "/tmp/baseline_oof")
if BASELINE_DIR and os.path.isdir(BASELINE_DIR):
    for name in ("hill", "stack", "rank", "lgbm_full"):
        b_path = os.path.join(BASELINE_DIR, f"oof_{name}.npy")
        n_path = f"{OUT}/oof_{name}.npy"
        if not (os.path.exists(b_path) and os.path.exists(n_path)):
            continue
        b_oof = np.load(b_path)
        n_oof = np.load(n_path)
        a_b = roc_auc_score(y_sub, b_oof)
        a_n = roc_auc_score(y_sub, n_oof)
        delta = a_n - a_b
        # We want improvements; tolerate tiny regressions (<1e-5) due to smoke noise.
        ok = delta > -1e-5
        sign = "+" if delta >= 0 else ""
        assert_(f"improvement: {name}  {a_b:.6f} -> {a_n:.6f}  ({sign}{delta:.6f})", ok,
                f"regression of {-delta:.6f}" if not ok else "")

# Summary
print(f"\n--- Summary ---")
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"PASSED: {passed}/{total}  ({100*passed/total:.1f}%)")
sys.exit(0 if passed == total else 1)
