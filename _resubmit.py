"""Apply the new robust submission logic to the existing output/ artifacts.

Reads oof_{stack,rank,hill}.npy and test_{stack,rank,hill}.npy, applies the
ENSEMBLE_PICK_THRESHOLD rule, and writes output/submission.csv.

Does NOT change submission_safe.csv.
"""
import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

OUT = "output"
DATA = "playground-series-dataCSV"
ENSEMBLE_PICK_THRESHOLD = 0.0005

# Reload labels aligned to the SMOKE 5% sample
tr = pd.read_csv(f"{DATA}/train.csv")
y = tr["satisfaction"].astype(int).values
rng = np.random.default_rng(42)
y = y[rng.permutation(len(y))[:34981]]

# Reload ensembles
candidates = []
for name in ("stack", "rank", "hill"):
    p_oof  = f"{OUT}/oof_{name}.npy"
    p_test = f"{OUT}/test_{name}.npy"
    if not (os.path.exists(p_oof) and os.path.exists(p_test)):
        print(f"  skipping {name}: file missing")
        continue
    o = np.load(p_oof); t = np.load(p_test)
    candidates.append((name, roc_auc_score(y, o), o, t))

if not candidates:
    raise SystemExit("no ensemble OOFs found")

candidates.sort(key=lambda x: -x[1])
print("Final candidates:", [(c[0], round(c[1], 6)) for c in candidates])

best_name, best_auc, best_oof, best_test = candidates[0]
spread = candidates[0][1] - candidates[-1][1]

if spread > ENSEMBLE_PICK_THRESHOLD:
    print(f"  spread={spread:.6f} > {ENSEMBLE_PICK_THRESHOLD}: using single best ({best_name})")
    final_test = best_test
    final_name = best_name
    final_auc = best_auc
else:
    print(f"  spread={spread:.6f} <= {ENSEMBLE_PICK_THRESHOLD}: rank-averaging {len(candidates)} ensembles")
    def _rank01(a):
        return np.argsort(np.argsort(a)) / (len(a) - 1)
    all_oof  = np.column_stack([c[2] for c in candidates])
    all_test = np.column_stack([c[3] for c in candidates])
    final_test = np.mean([_rank01(all_test[:, j]) for j in range(all_test.shape[1])], axis=0)
    final_oof  = np.mean([_rank01(all_oof[:, j])  for j in range(all_oof.shape[1])],  axis=0)
    final_auc = roc_auc_score(y, final_oof)
    final_name = "ensemble_rank_avg"

# Build submission
test_ids = pd.read_csv(f"{DATA}/sample_submission.csv")["id"].values
sub = pd.DataFrame({"id": test_ids, "satisfaction": final_test})

# Validate
assert len(sub) == 299844
assert (sub["id"].values == test_ids).all()
assert sub["satisfaction"].notna().all()
assert sub["satisfaction"].between(0, 1).all()

# Save
backup = f"{OUT}/submission_pick-best.csv"
if not os.path.exists(backup):
    pd.read_csv(f"{OUT}/submission.csv").to_csv(backup, index=False)
    print(f"backed up old submission to {backup}")

sub.to_csv(f"{OUT}/submission.csv", index=False)
print(
    f"wrote output/submission.csv: shape={sub.shape}  "
    f"min={sub['satisfaction'].min():.4f}  max={sub['satisfaction'].max():.4f}  "
    f"mean={sub['satisfaction'].mean():.4f}"
)
print(f"  strategy: {final_name}  OOF AUC = {final_auc:.6f}")
print(f"  prior submission (hill alone) OOF AUC = {candidates[0][1]:.6f}")
