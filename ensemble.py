"""
XGBoost baseline for the CSI500 stock-selection competition.

Pipeline
--------
1. Load data/prices.parquet
2. Build features + 5-day forward target (features.py)
3. Train XGBoost on all but the last `EMBARGO_DAYS` training rows
4. Validate on those held-out rows (reports rank IC as sanity check)
5. Predict on the most recent date
6. Build a portfolio: top-K names, score-weighted with the 10% cap

Usage
-----
  python baseline_xgboost.py                       # predict from latest data
  python baseline_xgboost.py --as-of 20260503      # predict as of a given date
  python baseline_xgboost.py --top-k 50 --out submissions/week1.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

from features_benchmark import (
#from features_baseline import (
    FEATURE_COLUMNS, TARGET_COLUMN, FORWARD_HORIZON,
    build_features, training_frame, prediction_frame,
)

from ridge_model import RIDGE_CONFIG, train_ridge_model

DATA_DIR = Path(__file__).parent / "data"
VAL_DAYS = 10               # number of trading days in the validation window
EMBARGO_DAYS = 5            # gap between train end and val start (>= FORWARD_HORIZON
                            # so training targets don't reach into val dates)
MIN_STOCKS = 30             # rule: portfolio must hold >= 30 names
MAX_WEIGHT = 0.10           # rule: per-stock weight cap
DEFAULT_TOP_K = 50          # baseline picks top-50 by predicted score

VARIANTS = {
    "xgb_aggressive_top30": {
        "model_type": "xgb",
        "top_k": 30,
        "params": {
            "n_estimators": 400,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
            "min_child_weight": 10,
            "reg_lambda": 1.0,
        },
    },
    "xgb_stable_top50": {
        "model_type": "xgb",
        "top_k": 50,
        "params": {
            "n_estimators": 600,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "min_child_weight": 15,
            "reg_lambda": 2.0,
        },
    },
    "xgb_conservative_top75": {
        "model_type": "xgb",
        "top_k": 75,
        "params": {
            "n_estimators": 700,
            "max_depth": 3,
            "learning_rate": 0.02,
            "subsample": 0.90,
            "colsample_bytree": 0.65,
            "min_child_weight": 20,
            "reg_lambda": 4.0,
        },
    },
}
VARIANTS.update(RIDGE_CONFIG)

ENSEMBLE_WEIGHTS = {
    "xgb_aggressive_top30": 0.25,
    "xgb_stable_top50": 0.20,
    "xgb_conservative_top75": 0.20,
    "ridge": 0.35,
}

def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame, config: dict):
    model_type = config.get("model_type", "xgb")
    params = config["params"]

    if model_type == "xgb":
        model = xgb.XGBRegressor(
            **params,
            tree_method="hist",
            n_jobs=-1,
            early_stopping_rounds=30,
            random_state=42,
        )
        model.fit(
            train_df[FEATURE_COLUMNS],
            train_df[TARGET_COLUMN],
            eval_set=[(val_df[FEATURE_COLUMNS], val_df[TARGET_COLUMN])],
            verbose=False,
        )
        return model
    if model_type == "ridge":
        model = train_ridge_model(
            train_df=train_df,
            feature_columns=FEATURE_COLUMNS,
            target_column=TARGET_COLUMN,
            alpha=params["alpha"],
        )
        return model
    raise ValueError(f"Unsupported model_type: {model_type}")

def rank_ic(y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray) -> float:
    """Daily cross-sectional Spearman correlation, averaged over dates."""
    ics = []
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < 20:
            continue
        rho, _ = spearmanr(y_true[mask], y_pred[mask])
        if not np.isnan(rho):
            ics.append(rho)
    return float(np.mean(ics)) if ics else float("nan")


def build_portfolio(scores: pd.Series, top_k: int = DEFAULT_TOP_K) -> pd.Series:
    """Top-K names, weight proportional to (rank) then capped at MAX_WEIGHT.

    We use rank-weights rather than score-weights so pathological score scales
    do not produce a single dominant name.  After capping at 10% we redistribute
    spillover to uncapped names and iterate until feasible.
    """
    if top_k < MIN_STOCKS:
        raise ValueError(f"top_k must be >= {MIN_STOCKS} (rule)")
    chosen = scores.sort_values(ascending=False).head(top_k).copy()

    # Rank-based weights (best stock gets largest weight, then normalize).
    ranks = np.arange(top_k, 0, -1, dtype=float)
    w = pd.Series(ranks / ranks.sum(), index=chosen.index)

    # Iteratively cap at MAX_WEIGHT and redistribute to uncapped names.
    for _ in range(50):
        over = w > MAX_WEIGHT
        if not over.any():
            break
        excess = (w[over] - MAX_WEIGHT).sum()
        w[over] = MAX_WEIGHT
        free = ~over
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()

    assert abs(w.sum() - 1.0) < 1e-6, f"weights sum to {w.sum()}"
    assert (w <= MAX_WEIGHT + 1e-9).all(), "cap violated"
    assert (w > 0).sum() >= MIN_STOCKS, "too few names"
    return w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--as-of", default=None, help="YYYYMMDD; defaults to latest date in data")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--out", default="submission.csv")
    args = p.parse_args()

    print(f">> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)
    print(f"   {len(prices):,} rows, {prices['stock_code'].nunique()} stocks, "
          f"dates {prices['date'].min().date()} to {prices['date'].max().date()}")

    print(">> Building features")
    panel = build_features(prices)
    # Bound training data so backtesting with --as-of doesn't leak future rows.
    # Training uses features from date t with target = close(t+FORWARD_HORIZON),
    # so we cap training dates at as_of - FORWARD_HORIZON trading days.
    as_of_ts = pd.Timestamp(args.as_of) if args.as_of else panel["date"].max()
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of_ts)))
    cutoff_idx = max(0, as_of_idx - FORWARD_HORIZON)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])
    train_pool = training_frame(panel, max_date=train_cutoff)

    # Time-based split with embargo:
    #   [ ... train ... | embargo (discarded) | val (last VAL_DAYS) ]
    # The embargo prevents training labels (5-day forward) from reaching into
    # dates whose prices also feed the validation features.
    all_dates = np.sort(train_pool["date"].unique())
    if len(all_dates) < VAL_DAYS + EMBARGO_DAYS + 20:
        raise RuntimeError("Not enough dates to train; download more history.")
    val_start = pd.Timestamp(all_dates[-VAL_DAYS])
    train_end = pd.Timestamp(all_dates[-(VAL_DAYS + EMBARGO_DAYS + 1)])
    train_df = train_pool[train_pool["date"] <= train_end]
    val_df = train_pool[train_pool["date"] >= val_start]
    print(f"   train: {len(train_df):,} rows up to {train_end.date()}")
    print(f"   embargo: {EMBARGO_DAYS} trading days (discarded)")
    print(f"   val:   {len(val_df):,} rows from {val_start.date()}")

    pred_df = prediction_frame(panel, as_of=args.as_of)
    pred_df["stock_code"] = pred_df["stock_code"].astype(str).str.zfill(6)

    if pred_df.empty:
        raise RuntimeError(f"No prediction rows for as_of={args.as_of}")

    pred_date = pred_df["date"].iloc[0]
    print(f">> Predicting as of {pred_date.date()}, {len(pred_df)} stocks")

    score_table = pd.DataFrame({"stock_code": pred_df["stock_code"]})

    for name, config in VARIANTS.items():
        print(f">> Training {name}")
        model = train_model(train_df, val_df, config)

        raw_scores = model.predict(pred_df[FEATURE_COLUMNS])
        score_table[name] = raw_scores

        if TARGET_COLUMN in val_df.columns:
            val_pred = model.predict(val_df[FEATURE_COLUMNS])
            ic = rank_ic(
                val_df[TARGET_COLUMN].to_numpy(),
                val_pred,
                val_df["date"].to_numpy(),
            )
            print(f"   validation rank IC: {ic:.4f}")
    
    score_table = score_table.set_index("stock_code")

    ensemble_score = pd.Series(0.0, index=score_table.index)

    for name, weight in ENSEMBLE_WEIGHTS.items():
        s = score_table[name]
        s = (s - s.mean()) / (s.std() + 1e-12)
        ensemble_score += weight * s

    weights = build_portfolio(ensemble_score, top_k=args.top_k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out = pd.DataFrame({
        "stock_code": weights.index,
        "weight": weights.values,
    })

    out.to_csv(out_path, index=False)

    print(f">> Wrote {len(out)} names to {out_path}")
    print(
        f"   weight summary: min={out['weight'].min():.4f} "
        f"max={out['weight'].max():.4f} "
        f"sum={out['weight'].sum():.6f}"
    )


if __name__ == "__main__":
    main()
