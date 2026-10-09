"""Frozen validation harness for the S6E10 improvement loop.

Single source of truth for data loading, fold ids, OOF/test artifacts, and
state-file updates. Every experiment script imports from this module. Once
created, the harness, folds, and metric definitions are NEVER modified to
inflate a score.
"""
from __future__ import annotations

import os
import time
import glob
import json
import hashlib
import warnings
from dataclasses import dataclass, field, asdict
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ----- paths ---------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "playground-series-dataCSV")
OUT_DIR  = ROOT  # oof/, preds/, logs/, submissions/ live here
OOF_DIR   = os.path.join(ROOT, "oof")
PRED_DIR  = os.path.join(ROOT, "preds")
LOG_DIR   = os.path.join(ROOT, "logs")
SUB_DIR   = os.path.join(ROOT, "submissions")
FOLD_FILE = os.path.join(ROOT, "folds.csv")
EXP_FILE  = os.path.join(ROOT, "experiments.csv")
STATE_FILE = os.path.join(ROOT, "STATE.md")
PLAN_FILE  = os.path.join(ROOT, "PLAN.md")
QUESTIONS_FILE = os.path.join(ROOT, "QUESTIONS.md")

CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]
NUM_COLS = ["Age", "Flight Distance",
            "Departure Delay in Minutes", "Arrival Delay in Minutes"]
N_FOLDS = 5
SEED = 42
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH  = os.path.join(DATA_DIR, "test.csv")
SAMPLE_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

os.makedirs(OOF_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SUB_DIR, exist_ok=True)


# ----- data ----------------------------------------------------------------
def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load competition train/test as pandas DataFrames.

    Adds `__y__` to train (binary 0/1). Preserves original id order.
    """
    tr = pd.read_csv(TRAIN_PATH)
    te = pd.read_csv(TEST_PATH)
    tr.columns = [c.strip() for c in tr.columns]
    te.columns = [c.strip() for c in te.columns]
    if tr["satisfaction"].dtype == bool:
        tr["__y__"] = tr["satisfaction"].astype(np.int8)
    else:
        # string labels (defensive)
        m = {"satisfied": 1, "neutral or dissatisfied": 0,
             "True": 1, "False": 0, "yes": 1, "no": 0,
             "1": 1, "0": 0}
        tr["__y__"] = tr["satisfaction"].astype(str).str.strip().map(m).astype(np.int8)
    return tr, te


def load_folds(tr: pd.DataFrame) -> np.ndarray:
    """Load the per-row fold id (0..4) for the given train frame.

    Folds must match folds.csv exactly. Test: every id appears, every fold
    has the same number of rows.
    """
    df = pd.read_csv(FOLD_FILE)
    # merge on id to be tolerant of train reordering
    merged = tr[["id"]].merge(df, on="id", how="left")
    assert merged["fold"].notna().all(), "fold file missing some ids"
    assert sorted(merged["fold"].unique()) == list(range(N_FOLDS))
    counts = merged["fold"].value_counts().sort_index().values
    assert min(counts) == max(counts), f"unbalanced folds: {counts}"
    return merged["fold"].astype(np.int8).values


def rating_cols(tr: pd.DataFrame) -> list[str]:
    DROP = ["id", "satisfaction", "__y__"]
    RAW = [c for c in tr.columns if c not in DROP]
    return [c for c in RAW if c not in CAT_COLS + NUM_COLS]


def raw_feats(tr: pd.DataFrame) -> list[str]:
    DROP = ["id", "satisfaction", "__y__"]
    return [c for c in tr.columns if c not in DROP]


# ----- OOF / test artifacts ------------------------------------------------
def oof_path(name: str) -> str:
    return os.path.join(OOF_DIR, f"{name}.npy")


def pred_path(name: str) -> str:
    return os.path.join(PRED_DIR, f"{name}.npy")


def have(name: str) -> bool:
    return os.path.exists(oof_path(name)) and os.path.exists(pred_path(name))


def load_oof(name: str) -> np.ndarray:
    return np.load(oof_path(name))


def load_pred(name: str) -> np.ndarray:
    return np.load(pred_path(name))


def list_oofs() -> list[str]:
    return sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(OOF_DIR, "*.npy")))


# ----- metrics -------------------------------------------------------------
def per_fold_auc(y: np.ndarray, oof: np.ndarray, folds: np.ndarray) -> list[float]:
    aucs = []
    for k in range(N_FOLDS):
        m = folds == k
        if m.sum() == 0:
            continue
        aucs.append(float(roc_auc_score(y[m], oof[m])))
    return aucs


def overall_auc(y: np.ndarray, oof: np.ndarray) -> float:
    return float(roc_auc_score(y, oof))


# ----- experiment table ----------------------------------------------------
@dataclass
class Experiment:
    exp_id: str
    change: str
    oof_auc: float
    delta: float
    per_fold_aucs: str   # "0.9591,0.9590,0.9592,0.9591,0.9590"
    blend_auc: float
    runtime_min: float
    result: str          # NEW CHAMPION / BLEND MEMBER / REJECTED
    notes: str = ""

    def to_csv_row(self) -> dict:
        return asdict(self)


def append_experiment(exp: Experiment) -> None:
    header_needed = not os.path.exists(EXP_FILE)
    df = pd.DataFrame([exp.to_csv_row()])
    df.to_csv(EXP_FILE, mode="a", header=header_needed, index=False)


def load_experiments() -> pd.DataFrame:
    if not os.path.exists(EXP_FILE):
        return pd.DataFrame()
    return pd.read_csv(EXP_FILE)


# ----- submissions ---------------------------------------------------------
def write_submission(probs: np.ndarray, path: str, name: str = "") -> None:
    """Write a Kaggle submission file with the required columns/order."""
    samp = pd.read_csv(SAMPLE_PATH)
    sub = pd.DataFrame({"id": samp["id"].values, "satisfaction": probs})
    assert len(sub) == len(samp), f"row count {len(sub)} vs {len(samp)}"
    assert (sub["id"].values == samp["id"].values).all(), "id order mismatch"
    assert sub["satisfaction"].notna().all(), "NaN in submission"
    assert sub["satisfaction"].between(0, 1).all(), "values outside [0,1]"
    sub.to_csv(path, index=False)
    if name:
        print(f"  wrote {path}  ({name})  min={sub['satisfaction'].min():.4f} "
              f"max={sub['satisfaction'].max():.4f}  mean={sub['satisfaction'].mean():.4f}")


# ----- log line ------------------------------------------------------------
def status_line(exp_id: str, change: str, oof_auc: float, delta: float,
                result: str, champion_auc: float, blend_auc: float) -> None:
    print(f"{exp_id:>8} | {change[:55]:<55} | OOF AUC {oof_auc:.6f} | "
          f"delta {delta:+.6f} | {result:<13} | champion {champion_auc:.6f} | "
          f"blend {blend_auc:.6f}")


# ----- fold-aware test prediction -----------------------------------------
def predict_test_avg(model_predict_fn: Callable[[pd.DataFrame, int], np.ndarray],
                     Xte: pd.DataFrame, folds: np.ndarray) -> np.ndarray:
    """Average test predictions across the 5 fold models.

    `model_predict_fn` should return a 1-D array of probabilities aligned
    with Xte. This is a thin wrapper; the fold-loop itself lives in
    experiment scripts.
    """
    return model_predict_fn(Xte, 0)  # placeholder; not used directly


# ----- small utilities -----------------------------------------------------
def _to_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def make_aligned_categoricals(*dfs: pd.DataFrame) -> list[pd.DataFrame]:
    """Cast CAT_COLS to category with the union of categories across dfs."""
    out = []
    for df in dfs:
        d = df.copy()
        for c in CAT_COLS:
            if c in d.columns:
                d[c] = d[c].astype("category")
        out.append(d)
    cats = {}
    for c in CAT_COLS:
        seen = []
        for d in out:
            if c in d.columns:
                seen.append(d[c].astype("category"))
        if seen:
            cats[c] = pd.api.types.union_categoricals(seen).categories
    for d in out:
        for c, cc in cats.items():
            if c in d.columns:
                d[c] = d[c].astype(pd.CategoricalDtype(categories=cc))
    return out
