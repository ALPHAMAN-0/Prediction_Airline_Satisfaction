"""Experiment E1b: XGBoost with shallower trees (depth=6) and lower LR
(lr=0.02). The first XGB hit best_iter=3999 (max) at lr=0.05 with
depth=8 — it didn't converge. Going shallower + slower.

Saves to oof/xgb_d6_lr02.npy, preds/xgb_d6_lr02.npy.
"""
from __future__ import annotations

import os
import sys
import time
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import harness  # noqa: E402
import exp_B_route  # noqa: E402
import exp_D_small  # noqa: E402
import exp_E1_xgb  # noqa: E402

EXP_ID = "5_E1b_xgb"
EXP_NAME = "xgb_d6_lr02"
CHANGE = "XGBoost depth=6, lr=0.02, more iterations (was: depth=8, lr=0.05, hit best_iter=3999)"


def main() -> None:
    print(f"\n=== Experiment E1b: {CHANGE} ===")
    t0 = time.time()
    oof, test_pred, fold_aucs = exp_E1_xgb.xgb_full_d_5fold(
        EXP_NAME, max_depth=6, learning_rate=0.02, seed=harness.SEED)
    minutes = (time.time() - t0) / 60.0
    summary = exp_E1_xgb.log_round(EXP_NAME, oof, test_pred, fold_aucs, minutes,
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
