# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
# ---

# %% [markdown]
# # Kaggle Playground S6E10 – Predicting Airline Satisfaction
#
# Pipeline: load → baseline LGBM → route features → target encoding → LGBM tune →
# XGBoost, CatBoost, RealMLP → ensemble → submission.
#
# All targets fit on the training fold only. OOF AUC is the only selection metric.

# %% [markdown]
# ## CONFIG

# %%
import os, time, gc, glob, warnings, random
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---- user-tunable config --------------------------------------------------
SMOKE        = bool(int(os.environ.get("SMOKE", "0")))   # 1 -> 5% sample, 2 folds
N_FOLDS      = int(os.environ.get("N_FOLDS", "5"))
SEED         = int(os.environ.get("SEED",  "42"))
RUN_LGBM     = bool(int(os.environ.get("RUN_LGBM",     "1")))
RUN_LGBM_ET  = bool(int(os.environ.get("RUN_LGBM_ET",  "1")))
RUN_XGB      = bool(int(os.environ.get("RUN_XGB",      "1")))
RUN_CAT      = bool(int(os.environ.get("RUN_CAT",      "1")))
RUN_CAT_CAT  = bool(int(os.environ.get("RUN_CAT_CAT",  "1")))  # all-cats variant
RUN_RMLP     = bool(int(os.environ.get("RUN_RMLP",     "1")))  # RealMLP via pytabkit
RUN_ORIG     = bool(int(os.environ.get("RUN_ORIG",     "1")))  # train on original data
TUNE_LGBM    = bool(int(os.environ.get("TUNE_LGBM",    "1")))

OUT_DIR      = os.environ.get(
    "OUT_DIR", "/kaggle/working" if os.path.isdir("/kaggle") else "./output"
)
DATA_DIR     = os.environ.get(
    "DATA_DIR", "/kaggle/input" if os.path.isdir("/kaggle/input") else "./data"
)

os.makedirs(OUT_DIR, exist_ok=True)
print("CONFIG:", dict(SMOKE=SMOKE, N_FOLDS=N_FOLDS, SEED=SEED,
                      RUN_LGBM=RUN_LGBM, RUN_LGBM_ET=RUN_LGBM_ET,
                      RUN_XGB=RUN_XGB, RUN_CAT=RUN_CAT, RUN_CAT_CAT=RUN_CAT_CAT,
                      RUN_RMLP=RUN_RMLP, RUN_ORIG=RUN_ORIG, TUNE_LGBM=TUNE_LGBM,
                      OUT_DIR=OUT_DIR, DATA_DIR=DATA_DIR))

# %% [markdown]
# ## Environment check

# %%
try:
    import psutil
    print(f"CPU count: {psutil.cpu_count()}, RAM: {psutil.virtual_memory().total/1e9:.1f} GB")
except ImportError:
    import os as _os
    print(f"CPU count: {_os.cpu_count()}")
try:
    import torch
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(),
          torch.cuda.device_count() if torch.cuda.is_available() else 0)
except (ImportError, RuntimeError) as e:
    print("no torch:", e)
try:
    import subprocess as sp
    print(sp.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"],
                 capture_output=True, text=True, timeout=10).stdout)
except (ImportError, FileNotFoundError, OSError):
    # nvidia-smi missing or not on PATH -> just skip the GPU-name banner.
    pass

# %% [markdown]
# ## Helpers: paths, IO, results

# %%
import contextlib, io
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

def _find_csvs():
    """Locate competition and original CSVs under DATA_DIR, fall back to ./data."""
    candidates = []
    for root in [DATA_DIR, os.path.join(DATA_DIR, "playground-series-dataCSV"),
                 os.path.join(DATA_DIR, "playground-series-s6e10"),
                 "./data", "."]:
        if not os.path.isdir(root):
            continue
        for p in glob.glob(os.path.join(root, "**", "train.csv"), recursive=True):
            candidates.append(("train", p))
        for p in glob.glob(os.path.join(root, "**", "test.csv"), recursive=True):
            candidates.append(("test", p))
        for p in glob.glob(os.path.join(root, "**", "sample_submission.csv"), recursive=True):
            candidates.append(("sub", p))
    # The original "airline-passenger-satisfaction" CSVs have 'satisfaction' cells
    # containing the string "satisfied" / "neutral or dissatisfied" (not True/False).
    for root in [DATA_DIR, "./data"]:
        if not os.path.isdir(root):
            continue
        for p in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True):
            try:
                hdr = pd.read_csv(p, nrows=2)
            except (pd.errors.ParserError, OSError, UnicodeDecodeError):
                continue
            if "satisfaction" not in hdr.columns:
                continue
            # look at first non-header value of satisfaction
            try:
                v = pd.read_csv(p, usecols=["satisfaction"], nrows=5)["satisfaction"].iloc[0]
            except (pd.errors.ParserError, OSError, UnicodeDecodeError, KeyError, ValueError):
                continue
            if isinstance(v, str):
                candidates.append(("orig", p))
    # dedup keeping last (deepest) path per role
    out = {}
    for role, p in candidates:
        out.setdefault(role, []).append(p)
    return {k: v[-1] if v else None for k, v in out.items()}

PATHS = _find_csvs()
print("PATHS:", PATHS)

# %% [markdown]
# ## Step A — Load & align

# %%
def _read_train_test():
    train = pd.read_csv(PATHS["train"])
    test  = pd.read_csv(PATHS["test"])
    return train, test

def _read_original():
    """The original Kaggle dataset: 'satisfied' / 'neutral or dissatisfied'."""
    if not PATHS.get("orig"):
        return None
    df = pd.read_csv(PATHS["orig"])
    df.columns = [c.strip() for c in df.columns]
    # Drop id/Unnamed columns
    for c in [c for c in df.columns if c.lower() in ("id", "unnamed: 0")]:
        df = df.drop(columns=c)
    if "satisfaction" in df.columns:
        df = df[df["satisfaction"].astype(str).str.contains("satisf", na=False)]
    return df.reset_index(drop=True)

def _normalize(df):
    # strip whitespace in column names
    df.columns = [c.strip() for c in df.columns]
    return df

train_raw, test_raw = _read_train_test()
train_raw = _normalize(train_raw); test_raw = _normalize(test_raw)
orig_raw  = _read_original()
if orig_raw is not None:
    orig_raw = _normalize(orig_raw)

# Encode target
def _to_int_y(s):
    if s.dtype == bool:
        return s.astype(int)
    if s.dtype == object:
        m = {"True": 1, "False": 0,
             "satisfied": 1, "neutral or dissatisfied": 0,
             "yes": 1, "no": 0, "1": 1, "0": 0}
        return s.astype(str).str.strip().map(m).astype("int8")
    return (s > 0).astype("int8")

train_raw["__y__"] = _to_int_y(train_raw["satisfaction"])
if orig_raw is not None and "satisfaction" in orig_raw.columns:
    orig_raw["__y__"] = _to_int_y(orig_raw["satisfaction"])

CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]
NUM_COLS = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
RATING_COLS = [c for c in train_raw.columns
               if c not in (["id", "satisfaction", "__y__"] + CAT_COLS + NUM_COLS)]
print(
    f"train={train_raw.shape}  test={test_raw.shape}  "
    f"orig={None if orig_raw is None else orig_raw.shape}"
)
print(f"CAT_COLS={CAT_COLS}\nNUM_COLS={NUM_COLS}\nRATING_COLS={RATING_COLS}")
print(f"positive rate train = {train_raw['__y__'].mean():.4f}")

# %% [markdown]
# ## Step A.1 — Optional SMOKE downsample

# %%
if SMOKE:
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(train_raw))[: max(1, int(0.05 * len(train_raw)))]
    train_raw = train_raw.iloc[idx].reset_index(drop=True)
    print("SMOKE downsampled train to:", train_raw.shape)

# %% [markdown]
# ## Step A.2 — Fixed StratifiedKFold

# %%
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
FOLDS = np.zeros(len(train_raw), dtype=np.int8)
for k, (_, vi) in enumerate(skf.split(train_raw, train_raw["__y__"])):
    FOLDS[vi] = k
print("Fold sizes:", np.bincount(FOLDS))

# %% [markdown]
# ## Step A.3 — Feature columns

# %%
DROP = ["id", "satisfaction", "__y__"]
RAW_FEATS = [c for c in train_raw.columns if c not in DROP]
print(f"#raw features = {len(RAW_FEATS)}")

# %%
# A small global table we'll grow as we add features.
results_rows = []
def log_result(name, oof, test_pred, minutes, n_folds):
    auc = roc_auc_score(train_raw["__y__"].values, oof)
    fold_aucs = []
    for k in range(n_folds):
        m = FOLDS == k
        if m.sum() == 0:
            continue
        fold_aucs.append(roc_auc_score(train_raw["__y__"].values[m], oof[m]))
    fold_std = float(np.std(fold_aucs))
    np.save(os.path.join(OUT_DIR, f"oof_{name}.npy"), oof.astype(np.float32))
    np.save(os.path.join(OUT_DIR, f"test_{name}.npy"), test_pred.astype(np.float32))
    results_rows.append((name, auc, fold_std, minutes))
    print(f"[{name:>22s}] OOF AUC = {auc:.6f}  fold std = {fold_std:.5f}  ({minutes:.1f} min)")
    return auc

def write_results():
    df = pd.DataFrame(results_rows, columns=["name", "oof_auc", "fold_std", "minutes"])
    df = df.sort_values("oof_auc", ascending=False).reset_index(drop=True)
    df.to_csv(os.path.join(OUT_DIR, "results.csv"), index=False)
    print(df.to_string(index=False))
    return df

# %% [markdown]
# ## Step B — Baseline LightGBM (raw features)

# %%
def _cast_cats(df):
    df = df.copy()
    for c in CAT_COLS:
        df[c] = df[c].astype("category")
    return df

def _x_for_lgbm(df):
    return _cast_cats(df[RAW_FEATS])

# Check if baseline output already exists (resume)
def _have(name):
    return (os.path.exists(os.path.join(OUT_DIR, f"oof_{name}.npy"))
            and os.path.exists(os.path.join(OUT_DIR, f"test_{name}.npy")))

import lightgbm as lgb

def run_lgbm(name, params, n_estimators=4000, early_stopping=200, use_extra_features=None):
    if _have(name):
        oof = np.load(os.path.join(OUT_DIR, f"oof_{name}.npy"))
        tp  = np.load(os.path.join(OUT_DIR, f"test_{name}.npy"))
        log_result(name, oof, tp, 0.0, N_FOLDS)
        return oof, tp
    t0 = time.time()
    oof = np.zeros(len(train_raw), dtype=np.float64)
    test_pred = np.zeros(len(test_raw),  dtype=np.float64)
    n_used = 0
    cat_features = [c for c in CAT_COLS if c in RAW_FEATS] + [
        c for c in RAW_FEATS
        if c not in CAT_COLS and str(train_raw[c].dtype) == "category"
    ]
    for k in range(N_FOLDS):
        tr = FOLDS != k; va = FOLDS == k
        Xtr = train_raw.loc[tr, RAW_FEATS] if use_extra_features is None else pd.concat(
            [train_raw.loc[tr, RAW_FEATS]] + use_extra_features(tr), axis=1)
        Xva = train_raw.loc[va, RAW_FEATS] if use_extra_features is None else pd.concat(
            [train_raw.loc[va, RAW_FEATS]] + use_extra_features(va), axis=1)
        Xte = test_raw[RAW_FEATS] if use_extra_features is None else pd.concat(
            [test_raw[RAW_FEATS]] + use_extra_features(None, True), axis=1)
        for c in CAT_COLS:
            if c in Xtr.columns:
                cats = pd.api.types.union_categoricals([Xtr[c].astype("category"),
                                                         Xva[c].astype("category"),
                                                         Xte[c].astype("category")]).categories
                Xtr[c] = Xtr[c].astype(pd.CategoricalDtype(categories=cats))
                Xva[c] = Xva[c].astype(pd.CategoricalDtype(categories=cats))
                Xte[c] = Xte[c].astype(pd.CategoricalDtype(categories=cats))
        cat_cols = [c for c in Xtr.columns if str(Xtr[c].dtype) == "category"]
        dtr = lgb.Dataset(
            Xtr, train_raw.loc[tr, "__y__"],
            categorical_feature=cat_cols, free_raw_data=False,
        )
        dva = lgb.Dataset(
            Xva, train_raw.loc[va, "__y__"],
            categorical_feature=cat_cols, reference=dtr, free_raw_data=False,
        )
        model = lgb.train(params, dtr, num_boost_round=n_estimators,
                          valid_sets=[dva], valid_names=["val"],
                          callbacks=[lgb.early_stopping(early_stopping, verbose=False),
                                     lgb.log_evaluation(0)])
        oof[va] = model.predict(Xva, num_iteration=model.best_iteration)
        test_pred += model.predict(Xte, num_iteration=model.best_iteration) / N_FOLDS
        n_used += 1
    minutes = (time.time() - t0) / 60.0
    log_result(name, oof, test_pred, minutes, N_FOLDS)
    return oof, test_pred

# %%
if RUN_LGBM:
    p = dict(objective="binary", metric="auc", learning_rate=0.05,
             num_leaves=63, min_data_in_leaf=50, feature_fraction=0.85,
             bagging_fraction=0.85, bagging_freq=1, lambda_l1=0.1, lambda_l2=0.1,
             verbosity=-1, seed=SEED, num_threads=-1, max_bin=255)
    run_lgbm("lgbm_raw", p)

# %% [markdown]
# ## Step C — Route features (group by Flight Distance)

# %%
# Strategy: compute aggregates on the full (train + test + original FEATURES) set,
# but never on labels.  Then build a "label-safe" block using only the original
# dataset's labels (its rows are disjoint from the competition data).

ALL_RAW = pd.concat([
    train_raw[RAW_FEATS].assign(__src__=0),
    test_raw[RAW_FEATS].assign(__src__=1),
], ignore_index=True) if orig_raw is None else pd.concat([
    train_raw[RAW_FEATS].assign(__src__=0),
    test_raw[RAW_FEATS].assign(__src__=1),
    orig_raw[RAW_FEATS].assign(__src__=2),
], ignore_index=True)

ROUTE = "Flight Distance"
RATINGS_FOR_AGG = RATING_COLS + ["Age"]

# route_count and per-source counts
def _route_counts():
    g = ALL_RAW.groupby(ROUTE, observed=True)
    cnt = g.size().rename("route_count")
    src = g["__src__"].value_counts().unstack(fill_value=0)
    src.columns = [f"route_src{int(c)}_count" for c in src.columns]
    return pd.concat([cnt, src], axis=1).reset_index()

# per-route mean/std of ratings + Age, and residual = value - route mean
def _route_stats():
    g = ALL_RAW.groupby(ROUTE, observed=True)
    out = pd.DataFrame({ROUTE: g.size().index})
    for c in RATINGS_FOR_AGG:
        m = g[c].mean().rename(f"route_{c}_mean").values
        s = g[c].std().rename(f"route_{c}_std").values
        out[f"route_{c}_mean"] = m
        out[f"route_{c}_std"]  = s
    return out

# per-route category share
def _route_cats():
    g = ALL_RAW.groupby(ROUTE, observed=True)
    out = pd.DataFrame({ROUTE: g.size().index})
    for c in CAT_COLS:
        dummies = pd.get_dummies(ALL_RAW[c].astype(str), prefix=c).astype("int8")
        dummies[ROUTE] = ALL_RAW[ROUTE].values
        agg = dummies.groupby(ROUTE, observed=True).mean()
        agg.columns = [f"route_share_{col}" for col in agg.columns]
        out = out.merge(agg.reset_index(), on=ROUTE, how="left")
    return out

ROUTE_COUNTS  = _route_counts()
ROUTE_STATS   = _route_stats()
ROUTE_CATS    = _route_cats()

# %%
def _attach_route(df):
    out = df.merge(ROUTE_COUNTS, on=ROUTE, how="left")
    out = out.merge(ROUTE_STATS,  on=ROUTE, how="left")
    out = out.merge(ROUTE_CATS,   on=ROUTE, how="left")
    # residual = value - route mean  (only for ratings + Age)
    for c in RATINGS_FOR_AGG:
        out[f"res_{c}"] = out[c].astype("float32") - out[f"route_{c}_mean"].astype("float32")
    return out

# Attach once, then use everywhere downstream.
train_feats = _attach_route(train_raw[RAW_FEATS])
test_feats  = _attach_route(test_raw[RAW_FEATS])
ROUTE_FEATURE_NAMES = [c for c in train_feats.columns if c.startswith(("route_", "res_"))]
print(f"#route features = {len(ROUTE_FEATURE_NAMES)}")

# %% [markdown]
# ### C.1 — Label-safe block from original dataset

# %%
def _build_label_safe():
    """Smoothed target rate per Flight Distance and per value of each other column,
    computed from the ORIGINAL dataset only (its rows are disjoint from competition data)."""
    out = {}
    if orig_raw is None:
        return out
    y = orig_raw["__y__"].values
    PRIOR = float(y.mean())
    SMOOTH = 20
    # per Flight Distance
    g = orig_raw.groupby(ROUTE, observed=True)
    cnt = g.size(); sm = g["__y__"].sum()
    rate = ((sm + SMOOTH * PRIOR) / (cnt + SMOOTH)).rename("route_y_smooth").reset_index()
    cnt  = cnt.rename("route_y_count").reset_index()
    out["route_y"] = rate.merge(cnt, on=ROUTE, how="outer")
    # per value of each other column
    for c in [c for c in RAW_FEATS if c != ROUTE]:
        df = pd.DataFrame({c: orig_raw[c].astype(str).values, "__y__": y})
        agg = df.groupby(c, observed=True)["__y__"].agg(["sum", "size"])
        rate = (agg["sum"] + SMOOTH * PRIOR) / (agg["size"] + SMOOTH)
        out[c] = pd.DataFrame({c: agg.index.astype(str), f"{c}_y_smooth": rate.values,
                                f"{c}_y_count": agg["size"].values})
    return out

LABEL_SAFE = _build_label_safe()

def _attach_label_safe(df):
    out = df.copy()
    # route_y
    if "route_y" in LABEL_SAFE:
        out = out.merge(LABEL_SAFE["route_y"], on=ROUTE, how="left")
    for c in [c for c in RAW_FEATS if c != ROUTE]:
        if c in LABEL_SAFE:
            tmp = LABEL_SAFE[c]
            key = out[c].astype(str)
            mp = dict(zip(tmp[c].values, tmp[f"{c}_y_smooth"].values))
            out[f"{c}_y_smooth"] = key.map(mp).astype("float32")
            mp2 = dict(zip(tmp[c].values, tmp[f"{c}_y_count"].values))
            out[f"{c}_y_count"]  = key.map(mp2).astype("float32")
    return out

# %% [markdown]
# ### C.2 — orig_proba from a model trained on the original data

# %%
ORIG_OOF_NAME = "orig_xgb"
ORIG_TEST_NAME = "orig_xgb"
# Resume-safety: skip if we already have the feature file in either the new
# (feat_orig_proba_*.npy) or legacy (oof_orig_proba.npy) naming.
_orig_proba_present = (
    os.path.exists(os.path.join(OUT_DIR, "feat_orig_proba_train.npy"))
    or os.path.exists(os.path.join(OUT_DIR, "oof_orig_proba.npy"))
)
if RUN_ORIG and orig_raw is not None and not _orig_proba_present and not _have("orig_xgb"):
    try:
        import xgboost as xgb
        X = _cast_cats(orig_raw[RAW_FEATS])
        for c in CAT_COLS:
            if c in X.columns:
                cats = X[c].cat.categories
                X[c] = X[c].astype(pd.CategoricalDtype(categories=cats))
        y = orig_raw["__y__"].values
        dtr = xgb.DMatrix(X, label=y, enable_categorical=True)
        params = dict(objective="binary:logistic", eval_metric="auc",
                      tree_method="hist", device="cpu",
                      max_depth=8, learning_rate=0.05, seed=SEED, verbosity=0)
        bst = xgb.train(params, dtr, num_boost_round=600)
        # predictions for competition train + test (align categories)
        for df in (train_raw, test_raw):
            for c in CAT_COLS:
                if c in df.columns:
                    df[c] = df[c].astype("category")
        # ensure same categories
        for c in CAT_COLS:
            if c in train_raw.columns:
                cats = X[c].cat.categories
                train_raw[c] = train_raw[c].astype(pd.CategoricalDtype(categories=cats))
                test_raw[c]  = test_raw[c].astype(pd.CategoricalDtype(categories=cats))
        dtest_train = xgb.DMatrix(_cast_cats(train_raw[RAW_FEATS]), enable_categorical=True)
        dtest_test  = xgb.DMatrix(_cast_cats(test_raw[RAW_FEATS]),  enable_categorical=True)
        orig_proba_train = bst.predict(dtest_train)
        orig_proba_test  = bst.predict(dtest_test)
        # Save as features so we can merge later. NOTE: these are NOT OOF predictions;
        # the model that produced them was trained on the original (label-safe) dataset
        # and the prediction was made on the full competition train+test. The filename
        # prefix `feat_` makes that explicit so no one tries to compute an OOF AUC on it.
        np.save(os.path.join(OUT_DIR, "feat_orig_proba_train.npy"),
                orig_proba_train.astype(np.float32))
        np.save(os.path.join(OUT_DIR, "feat_orig_proba_test.npy"),
                orig_proba_test.astype(np.float32))
        # Log
        # NOTE: this prediction was made with a model that SAW the labels used here.
        # It's a feature, not an OOF. We still want it in results.csv for visibility.
        # We do not compute a real OOF AUC for this one.
        results_rows.append(("orig_proba", float("nan"), float("nan"), 0.5))
        print("[orig_proba] saved train/test predictions (feature, no OOF).")
    except (ImportError, ValueError, RuntimeError) as e:
        print("orig_proba step failed:", e)
        RUN_ORIG = False

# %% [markdown]
# ## Step D — Target encoding block (ablation)

# %%
from sklearn.preprocessing import TargetEncoder

def _target_encode_block():
    """Per-fold target encoding using sklearn TargetEncoder (cross-fitted)."""
    cols = [c for c in RAW_FEATS]  # cast to string
    enc_train = pd.DataFrame(index=train_raw.index)
    enc_test  = pd.DataFrame(index=test_raw.index)
    for k in range(N_FOLDS):
        tr = FOLDS != k; va = FOLDS == k
        te = TargetEncoder(target_type="binary", smooth="auto",
                           cv=5, shuffle=True, random_state=SEED)
        Xtr_s = train_raw.loc[tr, cols].astype(str)
        Xva_s = train_raw.loc[va, cols].astype(str)
        Xte_s = test_raw[cols].astype(str)
        te.fit(Xtr_s, train_raw.loc[tr, "__y__"].values)
        enc_train.loc[va, [f"te_{c}" for c in cols]] = te.transform(Xva_s)
        enc_test[[f"te_{c}" for c in cols]] = te.transform(Xte_s)
    return enc_train.astype("float32"), enc_test.astype("float32")

# [Change 1] Per-Flight Distance x segment KFold-encoded smoothed target.
# Each row gets the OOF mean target rate of OTHER-FOLD rows sharing the same
# (Flight Distance, segment) value, smoothed by the global prior. The test set
# gets the mean computed on ALL train rows per key.
def _route_x_segment_kfold(segs=("Class", "Type of Travel", "Customer Type"),
                           smooth=20):
    PRIOR = float(train_raw["__y__"].mean())
    feats_tr = pd.DataFrame(index=train_raw.index)
    feats_te = pd.DataFrame(index=test_raw.index)
    y = train_raw["__y__"].values
    for seg in segs:
        df = pd.DataFrame({
            "route": train_raw["Flight Distance"].values,
            "seg":   train_raw[seg].astype(str).values,
            "y":     y,
        })
        df["key"] = df["route"].astype(str) + "|" + df["seg"]
        # global table per key
        g = df.groupby("key", observed=True)["y"].agg(["sum", "size"])
        rate_full = (g["sum"] + smooth * PRIOR) / (g["size"] + smooth)
        rate_full_dict = rate_full.to_dict()
        size_full_dict = g["size"].to_dict()
        # OOF per fold
        oof = np.zeros(len(df), dtype=np.float64)
        for k in range(N_FOLDS):
            tr = FOLDS != k; va = FOLDS == k
            # compute mean over train rows per key
            tr_keys = df.loc[tr, "key"].values
            tr_y    = df.loc[tr, "y"].values
            sk = pd.DataFrame({"key": tr_keys, "y": tr_y}).groupby("key",
                            observed=True)["y"].agg(["sum", "size"])
            r = (sk["sum"] + smooth * PRIOR) / (sk["size"] + smooth)
            mp = r.to_dict()
            keys_va = df.loc[va, "key"].values
            oof[va] = pd.Series(keys_va).map(mp).fillna(PRIOR).values
        feats_tr[f"route_x_{seg}_y"] = oof.astype("float32")
        # test = full rate per key
        keys_te = (test_raw["Flight Distance"].astype(str).values + "|"
                   + test_raw[seg].astype(str).values)
        feats_te[f"route_x_{seg}_y"] = (
            pd.Series(keys_te).map(rate_full_dict).fillna(PRIOR).astype("float32")
        )
    return feats_tr, feats_te

ROUTE_X_SEG_TR, ROUTE_X_SEG_TE = _route_x_segment_kfold()
print("ROUTE_X_SEG shape:", ROUTE_X_SEG_TR.shape, ROUTE_X_SEG_TE.shape)

# Conditional encodings: each rating x segment, smooth=20
def _conditional_enc():
    segs = ["Type of Travel", "Customer Type", "Class"]
    feats_tr = pd.DataFrame(index=train_raw.index)
    feats_te = pd.DataFrame(index=test_raw.index)
    PRIOR = float(train_raw["__y__"].mean()); SMOOTH = 20
    # We need groupby strings of (seg, value)
    if orig_raw is not None:
        all_tr = pd.concat([train_raw.assign(__src__="c"),
                            orig_raw.assign(__src__="o")], ignore_index=True)
    else:
        all_tr = train_raw.assign(__src__="c")
    # Rating x segment means, encoded per row
    for seg in segs:
        for c in RATING_COLS + ["Age", "Flight Distance"]:
            df = all_tr[[seg, c, "__y__"]].copy()
            df[c] = df[c].astype(str)
            df[seg] = df[seg].astype(str)
            df["k"] = df[seg] + "|" + df[c]
            g = df.groupby("k", observed=True)["__y__"].agg(["sum", "size"])
            rate = (g["sum"] + SMOOTH * PRIOR) / (g["size"] + SMOOTH)
            mp = rate.to_dict()
            key_tr = train_raw[seg].astype(str) + "|" + train_raw[c].astype(str)
            key_te = test_raw[seg].astype(str)  + "|" + test_raw[c].astype(str)
            feats_tr[f"cond_{seg}_{c}"] = key_tr.map(mp).astype("float32")
            feats_te[f"cond_{seg}_{c}"] = key_te.map(mp).astype("float32")
    return feats_tr, feats_te

# Frequency encoding
def _freq_enc():
    full = pd.concat([train_raw[RAW_FEATS], test_raw[RAW_FEATS]], ignore_index=True)
    n = len(full)
    out_tr = pd.DataFrame(index=train_raw.index)
    out_te = pd.DataFrame(index=test_raw.index)
    for c in RAW_FEATS:
        cnt = full[c].astype(str).value_counts()
        v = np.log1p(1e6 * cnt / n).to_dict()
        out_tr[f"freq_{c}"] = train_raw[c].astype(str).map(v).astype("float32")
        out_te[f"freq_{c}"] = test_raw[c].astype(str).map(v).astype("float32")
    return out_tr, out_te

TE_TRAIN, TE_TEST   = _target_encode_block()
COND_TR,  COND_TEST = _conditional_enc()
FREQ_TR,  FREQ_TEST = _freq_enc()
print("TE  :", TE_TRAIN.shape, TE_TEST.shape)
print("COND:", COND_TR.shape, COND_TEST.shape)
print("FREQ:", FREQ_TR.shape, FREQ_TEST.shape)

# %%
def _full_features(df_feats, df):
    """Concatenate: df_feats (= base+route) + label-safe + orig_proba + te + cond + freq."""
    out = pd.concat([df_feats.reset_index(drop=True),
                     df[["__y__"]].reset_index(drop=True) if "__y__" in df.columns else
                     pd.DataFrame(index=range(len(df)))], axis=1)
    return out

# Build full feature blocks once
def build_full():
    # base + route
    base_tr = train_feats.copy()
    base_te = test_feats.copy()
    # label safe
    base_tr = _attach_label_safe(base_tr)
    base_te = _attach_label_safe(base_te)
    # orig_proba (if available)
    if os.path.exists(os.path.join(OUT_DIR, "feat_orig_proba_train.npy")):
        base_tr["orig_proba"] = np.load(
            os.path.join(OUT_DIR, "feat_orig_proba_train.npy")
        ).astype("float32")
        base_te["orig_proba"] = np.load(
            os.path.join(OUT_DIR, "feat_orig_proba_test.npy")
        ).astype("float32")
    # te / cond / freq / route_x_seg
    base_tr = pd.concat([base_tr.reset_index(drop=True),
                         TE_TRAIN.reset_index(drop=True),
                         COND_TR.reset_index(drop=True),
                         FREQ_TR.reset_index(drop=True),
                         ROUTE_X_SEG_TR.reset_index(drop=True)], axis=1)
    base_te = pd.concat([base_te.reset_index(drop=True),
                         TE_TEST.reset_index(drop=True),
                         COND_TEST.reset_index(drop=True),
                         FREQ_TEST.reset_index(drop=True),
                         ROUTE_X_SEG_TE.reset_index(drop=True)], axis=1)
    return base_tr, base_te

FULL_TR, FULL_TE = build_full()

# [Change 5] Drop redundant route-mean columns (we keep the residual which
# encodes the same information in a way that's already centred on the target).
# This trims the feature space and lets LGBM/XGB/CatBoost focus on signal.
_drop_cols = [c for c in FULL_TR.columns
              if c.startswith("route_") and c.endswith("_mean")
              and not c.startswith("route_y_")]
if _drop_cols:
    FULL_TR = FULL_TR.drop(columns=_drop_cols)
    FULL_TE = FULL_TE.drop(columns=_drop_cols)
    print(f"Dropped {len(_drop_cols)} redundant route-mean columns")

print("FULL features:", FULL_TR.shape, FULL_TE.shape)

ALL_FEATURE_NAMES = list(FULL_TR.columns)
print("#all features =", len(ALL_FEATURE_NAMES))

# %% [markdown]
# ## Step E — LightGBM tuning (Optuna, small budget)

# %%
def lgb_oof(params, n_estimators=4000, early_stopping=200):
    oof = np.zeros(len(FULL_TR), dtype=np.float64)
    for k in range(N_FOLDS):
        tr = FOLDS != k; va = FOLDS == k
        Xtr = FULL_TR.iloc[tr].copy(); Xva = FULL_TR.iloc[va].copy()
        ytr = train_raw.loc[tr, "__y__"].values
        yva = train_raw.loc[va, "__y__"].values
        for c in CAT_COLS:
            if c in Xtr.columns:
                cats = pd.api.types.union_categoricals([Xtr[c].astype("category"),
                                                         Xva[c].astype("category")]).categories
                Xtr[c] = Xtr[c].astype(pd.CategoricalDtype(categories=cats))
                Xva[c] = Xva[c].astype(pd.CategoricalDtype(categories=cats))
        cat_cols = [c for c in Xtr.columns if str(Xtr[c].dtype) == "category"]
        dtr = lgb.Dataset(Xtr, ytr, categorical_feature=cat_cols, free_raw_data=False)
        dva = lgb.Dataset(
            Xva, yva,
            categorical_feature=cat_cols,
            reference=dtr, free_raw_data=False,
        )
        m = lgb.train(params, dtr, num_boost_round=n_estimators, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(early_stopping, verbose=False),
                                 lgb.log_evaluation(0)])
        oof[va] = m.predict(Xva, num_iteration=m.best_iteration)
    return oof

# %%
if TUNE_LGBM:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # Use a sub-sample (10%) and 2 folds for tuning to keep budget tight.
    rng = np.random.default_rng(SEED)
    tune_idx = rng.permutation(len(FULL_TR))[: max(1, int(0.10 * len(FULL_TR)))]
    Xt = FULL_TR.iloc[tune_idx].reset_index(drop=True)
    yt = train_raw["__y__"].values[tune_idx]
    skf_t = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
    folds_t = np.zeros(len(tune_idx), dtype=np.int8)
    for k, (_, vi) in enumerate(skf_t.split(Xt, yt)):
        folds_t[vi] = k
    def objective(trial):
        p = dict(objective="binary", metric="auc", verbosity=-1, seed=SEED,
                 num_threads=-1, learning_rate=trial.suggest_float("lr", 0.02, 0.06, log=True),
                 num_leaves=trial.suggest_int("num_leaves", 31, 127),
                 min_data_in_leaf=trial.suggest_int("min_data", 20, 200),
                 feature_fraction=trial.suggest_float("ff", 0.6, 1.0),
                 bagging_fraction=trial.suggest_float("bf", 0.6, 1.0),
                 bagging_freq=1, lambda_l1=trial.suggest_float("l1", 0.0, 5.0),
                 lambda_l2=trial.suggest_float("l2", 0.0, 5.0),
                 max_bin=trial.suggest_categorical("max_bin", [255, 511, 1023]))
        oof = np.zeros(len(Xt), dtype=np.float64)
        for k in range(2):
            tr = folds_t != k; va = folds_t == k
            Xtr = Xt.iloc[tr].copy(); Xva = Xt.iloc[va].copy()
            for c in CAT_COLS:
                if c in Xtr.columns:
                    cats = pd.api.types.union_categoricals(
                        [Xtr[c].astype("category"), Xva[c].astype("category")]).categories
                    Xtr[c] = Xtr[c].astype(pd.CategoricalDtype(categories=cats))
                    Xva[c] = Xva[c].astype(pd.CategoricalDtype(categories=cats))
            cat_cols = [c for c in Xtr.columns if str(Xtr[c].dtype) == "category"]
            dtr = lgb.Dataset(Xtr, yt[tr], categorical_feature=cat_cols, free_raw_data=False)
            dva = lgb.Dataset(
                Xva, yt[va],
                categorical_feature=cat_cols,
                reference=dtr, free_raw_data=False,
            )
            m = lgb.train(p, dtr, num_boost_round=2000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(100, verbose=False),
                                     lgb.log_evaluation(0)])
            oof[va] = m.predict(Xva, num_iteration=m.best_iteration)
        return roc_auc_score(yt, oof)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=40, show_progress_bar=False)
    print("Best LGBM params:", study.best_params, "value:", study.best_value)

# %% [markdown]
# ## Step E.1 — Train tuned LightGBM (full features)

# %%
# Default tuned params (used if TUNE_LGBM=False or in SMOKE)
LGBM_BEST = dict(objective="binary", metric="auc", verbosity=-1, seed=SEED,
                 num_threads=-1, learning_rate=0.03,
                 num_leaves=67, min_data_in_leaf=80, feature_fraction=0.8,
                 bagging_fraction=0.8, bagging_freq=1,
                 lambda_l1=0.5, lambda_l2=0.5, max_bin=1023)

if TUNE_LGBM:
    try:
        if "study" in dir() and len(study.trials) > 0:
            bp = study.best_params
            LGBM_BEST.update(dict(learning_rate=bp["lr"], num_leaves=bp["num_leaves"],
                                  min_data_in_leaf=bp["min_data"], feature_fraction=bp["ff"],
                                  bagging_fraction=bp["bf"], lambda_l1=bp["l1"],
                                  lambda_l2=bp["l2"], max_bin=bp["max_bin"]))
    except (AttributeError, KeyError, TypeError) as e:
        # optuna study may not exist (TUNE_LGBM was off) or bp may be missing keys.
        # Either way, just keep the default LGBM_BEST.
        print("optuna overwrite skipped:", e)

# Use a helper that reuses FULL_TR/TE
def run_lgbm_full(name, params, n_estimators=4000, early_stopping=200):
    if _have(name):
        oof = np.load(os.path.join(OUT_DIR, f"oof_{name}.npy"))
        tp  = np.load(os.path.join(OUT_DIR, f"test_{name}.npy"))
        log_result(name, oof, tp, 0.0, N_FOLDS)
        return oof, tp
    t0 = time.time()
    oof = np.zeros(len(FULL_TR), dtype=np.float64)
    test_pred = np.zeros(len(FULL_TE), dtype=np.float64)
    for k in range(N_FOLDS):
        tr = FOLDS != k; va = FOLDS == k
        Xtr = FULL_TR.iloc[tr].copy(); Xva = FULL_TR.iloc[va].copy()
        Xte = FULL_TE.copy()
        ytr = train_raw.loc[tr, "__y__"].values
        yva = train_raw.loc[va, "__y__"].values
        for c in CAT_COLS:
            if c in Xtr.columns:
                cats = pd.api.types.union_categoricals(
                    [Xtr[c].astype("category"), Xva[c].astype("category"),
                     Xte[c].astype("category")]).categories
                Xtr[c] = Xtr[c].astype(pd.CategoricalDtype(categories=cats))
                Xva[c] = Xva[c].astype(pd.CategoricalDtype(categories=cats))
                Xte[c] = Xte[c].astype(pd.CategoricalDtype(categories=cats))
        cat_cols = [c for c in Xtr.columns if str(Xtr[c].dtype) == "category"]
        dtr = lgb.Dataset(Xtr, ytr, categorical_feature=cat_cols, free_raw_data=False)
        dva = lgb.Dataset(
            Xva, yva,
            categorical_feature=cat_cols,
            reference=dtr, free_raw_data=False,
        )
        m = lgb.train(params, dtr, num_boost_round=n_estimators,
                      valid_sets=[dva], valid_names=["val"],
                      callbacks=[lgb.early_stopping(early_stopping, verbose=False),
                                 lgb.log_evaluation(0)])
        oof[va] = m.predict(Xva, num_iteration=m.best_iteration)
        test_pred += m.predict(Xte, num_iteration=m.best_iteration) / N_FOLDS
    minutes = (time.time() - t0) / 60.0
    log_result(name, oof, test_pred, minutes, N_FOLDS)
    return oof, test_pred

if RUN_LGBM:
    run_lgbm_full("lgbm_full", LGBM_BEST)

# %%
if RUN_LGBM_ET:
    p_et = dict(LGBM_BEST); p_et.update(extra_trees=True)
    run_lgbm_full("lgbm_full_et", p_et)

# [Change 3] 3-seed bag of lgbm_full_et (extra_trees), averaged in logit space.
# This adds 2 more diverse members to the ensemble without doubling the model
# cost. Each member is a separate OOF/test pair. Logit-space averaging is
# typically +0.0001-0.0003 AUC over probability-space averaging.
def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

if RUN_LGBM_ET and os.environ.get("RUN_LGBM_BAG", "1") == "1":
    p_et_bag = dict(LGBM_BEST); p_et_bag.update(extra_trees=True)
    oof_logit_acc = None; test_logit_acc = None
    for s in (SEED + 101, SEED + 202, SEED + 303):
        p = dict(p_et_bag); p["seed"] = s
        p["feature_fraction_seed"] = s
        p["bagging_seed"] = s
        name = f"lgbm_full_et_s{s % 1000}"
        o, t = run_lgbm_full(name, p)
        oof_logit_acc  = _logit(o) if oof_logit_acc  is None else oof_logit_acc  + _logit(o)
        test_logit_acc = _logit(t) if test_logit_acc is None else test_logit_acc + _logit(t)
    oof_bag  = _sigmoid(oof_logit_acc  / 3.0)
    test_bag = _sigmoid(test_logit_acc / 3.0)
    log_result("lgbm_full_et_bag", oof_bag, test_bag, 0.0, N_FOLDS)

# %% [markdown]
# ## Step F.1 — XGBoost (GPU)

# %%
if RUN_XGB:
    import xgboost as xgb
    if _have("xgb_full"):
        oof = np.load(os.path.join(OUT_DIR, "oof_xgb_full.npy"))
        tp  = np.load(os.path.join(OUT_DIR, "test_xgb_full.npy"))
        log_result("xgb_full", oof, tp, 0.0, N_FOLDS)
    else:
        t0 = time.time()
        oof = np.zeros(len(FULL_TR), dtype=np.float64)
        tp  = np.zeros(len(FULL_TE), dtype=np.float64)
        device = "cuda" if (os.environ.get("USE_GPU", "1") == "1") else "cpu"
        try:
            # quick GPU availability check
            xgb.DMatrix(np.zeros((2, 2))).slice([0, 1])
        except (xgb.core.XGBoostError, RuntimeError):
            # No CUDA-capable XGBoost build available -> fall back to CPU.
            device = "cpu"
        params = dict(objective="binary:logistic", eval_metric="auc",
                      tree_method="hist", device=device,
                      max_depth=8, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.5,
                      seed=SEED, verbosity=0, enable_categorical=True,
                      early_stopping_rounds=200)
        for k in range(N_FOLDS):
            tr = FOLDS != k; va = FOLDS == k
            Xtr = FULL_TR.iloc[tr].copy(); Xva = FULL_TR.iloc[va].copy(); Xte = FULL_TE.copy()
            ytr = train_raw.loc[tr, "__y__"].values
            yva = train_raw.loc[va, "__y__"].values
            for c in CAT_COLS:
                if c in Xtr.columns:
                    cats = pd.api.types.union_categoricals(
                        [Xtr[c].astype("category"), Xva[c].astype("category"),
                         Xte[c].astype("category")]).categories
                    Xtr[c] = Xtr[c].astype(pd.CategoricalDtype(categories=cats))
                    Xva[c] = Xva[c].astype(pd.CategoricalDtype(categories=cats))
                    Xte[c] = Xte[c].astype(pd.CategoricalDtype(categories=cats))
            dtr = xgb.DMatrix(Xtr, label=ytr, enable_categorical=True)
            dva = xgb.DMatrix(Xva, label=yva, enable_categorical=True)
            dte = xgb.DMatrix(Xte, enable_categorical=True)
            booster = xgb.train(params, dtr, num_boost_round=4000,
                                evals=[(dva, "val")], verbose_eval=0)
            bi = getattr(booster, "best_iteration", None)
            if bi is None or bi < 0:
                bi = booster.num_boosted_rounds() - 1
            oof[va] = booster.predict(dva, iteration_range=(0, bi + 1))
            tp += booster.predict(dte, iteration_range=(0, bi + 1)) / N_FOLDS
        minutes = (time.time() - t0) / 60.0
        log_result("xgb_full", oof, tp, minutes, N_FOLDS)
        del booster; gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass

# %% [markdown]
# ## Step F.2 — CatBoost (GPU, two variants)

# %%
def _cat_fit(name, cat_features, depth, grow="Depthwise", iters=4000, lr=0.05,
             loss="Logloss", od_type="Iter", od_wait=200):
    if _have(name):
        oof = np.load(os.path.join(OUT_DIR, f"oof_{name}.npy"))
        tp  = np.load(os.path.join(OUT_DIR, f"test_{name}.npy"))
        log_result(name, oof, tp, 0.0, N_FOLDS)
        return oof, tp
    from catboost import CatBoostClassifier, Pool
    t0 = time.time()
    oof = np.zeros(len(FULL_TR), dtype=np.float64)
    tp  = np.zeros(len(FULL_TE), dtype=np.float64)
    # Pick task type: GPU by default, CPU fallback if CUDA unavailable
    task_type = "GPU"
    if os.environ.get("USE_GPU", "1") == "0":
        task_type = "CPU"
    else:
        try:
            import torch
            if not torch.cuda.is_available():
                task_type = "CPU"
        except (ImportError, RuntimeError):
            # torch not installed, or CUDA driver missing -> fall back to CPU.
            task_type = "CPU"
    for k in range(N_FOLDS):
        tr = FOLDS != k; va = FOLDS == k
        Xtr = FULL_TR.iloc[tr].copy(); Xva = FULL_TR.iloc[va].copy(); Xte = FULL_TE.copy()
        ytr = train_raw.loc[tr, "__y__"].values
        yva = train_raw.loc[va, "__y__"].values
        for c in cat_features:
            if c in Xtr.columns:
                Xtr[c] = Xtr[c].astype(str)
                Xva[c] = Xva[c].astype(str)
                Xte[c] = Xte[c].astype(str)
        kw = dict(iterations=iters, learning_rate=lr, depth=depth,
                  grow_policy=grow, loss_function=loss, eval_metric=loss,
                  od_type=od_type, od_wait=od_wait,
                  random_seed=SEED, verbose=False, allow_writing_files=False)
        if task_type == "GPU":
            kw.update(task_type="GPU", devices="0")
        else:
            kw.update(task_type="CPU", thread_count=-1)
        model = CatBoostClassifier(**kw)
        model.fit(Pool(Xtr, ytr, cat_features=cat_features),
                  eval_set=Pool(Xva, yva, cat_features=cat_features), use_best_model=True)
        oof[va] = model.predict_proba(Xva)[:, 1]
        tp += model.predict_proba(Xte)[:, 1] / N_FOLDS
    minutes = (time.time() - t0) / 60.0
    log_result(name, oof, tp, minutes, N_FOLDS)
    del model; gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass
    return oof, tp

# %%
if RUN_CAT:
    _cat_fit("cat_depth10", CAT_COLS, depth=10, grow="Depthwise", iters=4000, lr=0.05)

# %%
if RUN_CAT_CAT:
    cat_all = list(FULL_TR.columns)  # all features as categorical
    _cat_fit("cat_all_cat", cat_all, depth=8, grow="Depthwise", iters=3000, lr=0.05,
             loss="Logloss", od_type="Iter", od_wait=200)

# %% [markdown]
# ## Step F.3 — RealMLP (pytabkit)

# %%
if RUN_RMLP:
    try:
        import pytabkit
        from pytabkit import RealMLP_TD_Classifier
        print("pytabkit:", pytabkit.__version__)
        if _have("realmlp_full"):
            oof = np.load(os.path.join(OUT_DIR, "oof_realmlp_full.npy"))
            tp  = np.load(os.path.join(OUT_DIR, "test_realmlp_full.npy"))
            log_result("realmlp_full", oof, tp, 0.0, N_FOLDS)
        else:
            t0 = time.time()
            oof = np.zeros(len(FULL_TR), dtype=np.float64)
            tp  = np.zeros(len(FULL_TE), dtype=np.float64)
            y = train_raw["__y__"].values
            for k in range(N_FOLDS):
                tr = FOLDS != k; va = FOLDS == k
                Xtr = FULL_TR.iloc[tr].values.astype("float32")
                Xva = FULL_TR.iloc[va].values.astype("float32")
                Xte = FULL_TE.values.astype("float32")
                # [Change 4] tuned RealMLP: bigger hidden, longer epochs, slightly higher lr.
                clf = RealMLP_TD_Classifier(device="cuda", n_epochs=80, batch_size=4096,
                                            lr=2e-3, hidden_sizes=[512, 512, 512],
                                            random_state=SEED + k)
                clf.fit(Xtr, y[tr])
                oof[va] = clf.predict_proba(Xva)[:, 1]
                tp += clf.predict_proba(Xte)[:, 1] / N_FOLDS
                del clf; gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except (ImportError, RuntimeError):
                    pass
            minutes = (time.time() - t0) / 60.0
            log_result("realmlp_full", oof, tp, minutes, N_FOLDS)
    except (ImportError, RuntimeError, ValueError) as e:
        # RealMLP is optional. Failures here (e.g. pytabkit missing, CUDA OOM)
        # should disable the rest of the RealMLP block, not crash the pipeline.
        print("RealMLP step failed:", e)
        RUN_RMLP = False

# %% [markdown]
# ## Step G — Ensemble (logistic stack, rank-avg, hill-climb)

# %%
# Load all available OOF/test arrays
def _load_all(prefix=""):
    items = []
    for p in sorted(glob.glob(os.path.join(OUT_DIR, f"{prefix}oof_*.npy"))):
        name = os.path.basename(p)[len("oof_"):-4]
        if name in ("orig_proba",):
            continue
        test_p = os.path.join(OUT_DIR, f"test_{name}.npy")
        if not os.path.exists(test_p):
            continue
        items.append((name, np.load(p).astype(np.float64),
                      np.load(test_p).astype(np.float64)))
    return items

ENTRIES = _load_all()
print("members:", [n for n, _, _ in ENTRIES])

# %%
def _to_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

# Nested-CV logistic stack
def _stack_lr(oofs, tests, y, C=0.02, n_repeats=3):
    n = len(y)
    from sklearn.linear_model import LogisticRegression
    oof_stack = np.zeros(n)
    weights = np.zeros(len(oofs))
    for rep in range(n_repeats):
        skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + rep)
        oof_rep = np.zeros(n)
        for tr, va in skf2.split(np.zeros(n), y):
            X = np.column_stack([_to_logit(o[va]) for o in oofs])
            Xtr = np.column_stack([_to_logit(o[tr]) for o in oofs])
            lr = LogisticRegression(C=C, solver="lbfgs", max_iter=200)
            lr.fit(Xtr, y[tr])
            oof_rep[va] = lr.predict_proba(X)[:, 1]
        oof_stack += oof_rep / n_repeats
    # Final weights
    Xall = np.column_stack([_to_logit(o) for o in oofs])
    lr = LogisticRegression(C=C, solver="lbfgs", max_iter=400)
    lr.fit(Xall, y)
    weights = lr.coef_[0]
    test_stack = np.zeros(len(tests[0]))
    Xt = np.column_stack([_to_logit(t) for t in tests])
    test_stack = lr.predict_proba(Xt)[:, 1]
    return oof_stack, test_stack, weights

# %%
y = train_raw["__y__"].values
NAMES = [n for n, _, _ in ENTRIES]
OOFS  = [o for _, o, _ in ENTRIES]
TESTS = [t for _, _, t in ENTRIES]

if len(OOFS) >= 2:
    # tune C
    best = None
    for C in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
        os_, ts_, w_ = _stack_lr(OOFS, TESTS, y, C=C, n_repeats=2)
        a = roc_auc_score(y, os_)
        print(f"stack C={C}  OOF AUC = {a:.6f}  weights = {dict(zip(NAMES, np.round(w_, 2)))}")
        if best is None or a > best[0]:
            best = (a, C, os_, ts_, w_)
    a, C, os_, ts_, w_ = best
    print(f"Best stack C={C} OOF AUC = {a:.6f}")
    np.save(os.path.join(OUT_DIR, "oof_stack.npy"), os_.astype(np.float32))
    np.save(os.path.join(OUT_DIR, "test_stack.npy"), ts_.astype(np.float32))
    results_rows.append(("stack", a, 0.0, 0.0))
else:
    print("Not enough OOFs for stacking yet.")

# Rank-average of the top-3 individual members
if len(OOFS) >= 3:
    aucs = [roc_auc_score(y, o) for o in OOFS]
    order = np.argsort(aucs)[::-1]
    top3 = order[:3]
    os_rank = np.mean(
        [np.argsort(np.argsort(OOFS[i])) / (len(OOFS[i]) - 1) for i in top3],
        axis=0,
    )
    ts_rank = np.mean(
        [np.argsort(np.argsort(TESTS[i])) / (len(TESTS[i]) - 1) for i in top3],
        axis=0,
    )
    a_rank = roc_auc_score(y, os_rank)
    print(f"rank-avg top-3 OOF AUC = {a_rank:.6f}")
    np.save(os.path.join(OUT_DIR, "oof_rank.npy"), os_rank.astype(np.float32))
    np.save(os.path.join(OUT_DIR, "test_rank.npy"), ts_rank.astype(np.float32))
    results_rows.append(("rank", a_rank, 0.0, 0.0))

# Greedy hill-climb in LOGIT space (Caruana-style, [Change 2]).
# Blending in logit space usually beats probability-space for AUC by ~0.0003.
def _hill_climb(oofs, tests, y, max_steps=200, tol=1e-7):
    oofs_l = [_to_logit(o) for o in oofs]
    tests_l = [_to_logit(t) for t in tests]
    n = len(oofs_l)
    cur = np.zeros(len(y)); cur_t = np.zeros(len(tests_l[0]))
    weights = np.zeros(n); best_auc = 0.0
    for step in range(max_steps):
        improved = False
        for i in range(n):
            for w in (1.0, 2.0, 3.0, 5.0, 0.5):
                cand = (cur * weights.sum() + oofs_l[i] * w) / (weights.sum() + w)
                a = roc_auc_score(y, cand)
                if a > best_auc + tol:
                    best_auc = a; weights[i] += w; cur = cand; improved = True
        if not improved: break
    if weights.sum() > 0:
        cur_t = np.zeros(len(tests_l[0]))
        for i, w in enumerate(weights):
            if w > 0:
                cur_t += tests_l[i] * w
        cur_t /= weights.sum()
        cur_t = 1.0 / (1.0 + np.exp(-cur_t))   # back to probability
    cur_p = 1.0 / (1.0 + np.exp(-cur))
    return cur_p, cur_t, best_auc, weights

if len(OOFS) >= 2:
    os_h, ts_h, auc_h, w_h = _hill_climb(OOFS, TESTS, y)
    print(f"hill-climb OOF AUC = {auc_h:.6f}  weights = {dict(zip(NAMES, np.round(w_h, 2)))}")
    np.save(os.path.join(OUT_DIR, "oof_hill.npy"), os_h.astype(np.float32))
    np.save(os.path.join(OUT_DIR, "test_hill.npy"), ts_h.astype(np.float32))
    results_rows.append(("hill", auc_h, 0.0, 0.0))

# %% [markdown]
# ## Step H — Submission

# %%
def _write_submission(probs, path):
    sub = pd.DataFrame({"id": test_raw["id"].values, "satisfaction": probs})
    # validation
    samp = pd.read_csv(PATHS["sub"])
    assert len(sub) == len(samp), f"row count mismatch {len(sub)} vs {len(samp)}"
    assert (sub["id"].values == samp["id"].values).all(), "id order mismatch"
    assert sub["satisfaction"].notna().all(), "NaN in submission"
    assert sub["satisfaction"].between(0, 1).all(), "values outside [0,1]"
    sub.to_csv(path, index=False)
    print(
        f"wrote {path}: {sub.shape}  "
        f"(min={sub['satisfaction'].min():.4f}, max={sub['satisfaction'].max():.4f})"
    )

# Pick the submission strategy. The 3 ensembles (stack, rank, hill) often
# differ by < 0.0002 in OOF AUC -- below the fold-std noise floor (~0.0003).
# In that regime, picking the "best" one is overfitting to OOF noise and is
# not robust on the hidden test set. We use a simple rule:
#   - if the spread between the best and worst ensemble is > ENSEMBLE_PICK_THRESHOLD
#     (i.e. a real signal), use the single best one
#   - otherwise, use the rank-average of all 3 ensembles (more robust)
ENSEMBLE_PICK_THRESHOLD = 0.0005
candidates = []
for name, fname in (("stack", "stack"), ("rank_top3", "rank"), ("hill_climb", "hill")):
    p = os.path.join(OUT_DIR, f"oof_{fname}.npy")
    if not os.path.exists(p):
        continue
    o = np.load(p)
    candidates.append((
        name,
        roc_auc_score(y, o),
        o,
        np.load(os.path.join(OUT_DIR, f"test_{fname}.npy")),
    ))

if not candidates:
    print("WARNING: no ensemble OOFs found; falling back to last base learner.")
    best_test = TESTS[-1]
    best_auc = float("nan")
    best_name = "fallback"
else:
    candidates.sort(key=lambda x: -x[1])
    print(
        "Final candidates:",
        [(c[0], round(c[1], 6)) for c in candidates],
    )
    best_name, best_auc, best_oof, best_test = candidates[0]
    spread = candidates[0][1] - candidates[-1][1]
    if spread > ENSEMBLE_PICK_THRESHOLD:
        # Real signal: pick the best ensemble.
        print(
            f"  spread={spread:.6f} > {ENSEMBLE_PICK_THRESHOLD}: "
            f"using single best ({best_name})"
        )
    else:
        # Noise regime: rank-average all 3 ensembles for robustness.
        print(
            f"  spread={spread:.6f} <= {ENSEMBLE_PICK_THRESHOLD}: "
            f"rank-averaging {len(candidates)} ensembles for robustness"
        )
        all_oof  = np.column_stack([c[2] for c in candidates])
        all_test = np.column_stack([c[3] for c in candidates])
        # logit-space mean is more robust than rank-mean for binary AUC,
        # but rank-mean is rank-invariant so safe under any monotonic transform.
        def _rank01(a):
            return np.argsort(np.argsort(a)) / (len(a) - 1)
        best_test = np.mean(
            [_rank01(all_test[:, j]) for j in range(all_test.shape[1])],
            axis=0,
        )
        # OOF AUC for logging
        best_oof_rank = np.mean(
            [_rank01(all_oof[:, j]) for j in range(all_oof.shape[1])],
            axis=0,
        )
        best_auc = roc_auc_score(y, best_oof_rank)
        best_name = "ensemble_rank_avg"
    _write_submission(best_test, os.path.join(OUT_DIR, "submission.csv"))

# safe alternative: rank-avg of top-3 base learners (most diverse single-models)
if len(OOFS) >= 3:
    aucs = [roc_auc_score(y, o) for o in OOFS]
    order = np.argsort(aucs)[::-1][:3]
    ts_rank_safe = np.mean(
        [np.argsort(np.argsort(TESTS[i])) / (len(TESTS[i]) - 1) for i in order],
        axis=0,
    )
    _write_submission(ts_rank_safe, os.path.join(OUT_DIR, "submission_safe.csv"))

# %% [markdown]
# ## Final results

# %%
df_results = write_results()
print("Best OOF AUC:", df_results.iloc[0]["oof_auc"] if len(df_results) else "n/a")
