# 1. Train XGBoost on past data
# 2. Skip embargo days (avoid leakage)
# 3. Test on next 5 trading days
# 4. Build portfolio
# 5. measure:
#    - rank IC
#    - portfolio return
#    - excess return
#    - win rate

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

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

from features_baseline import (
#from features_liquidity import (
#from features_benchmark import (
    FEATURE_COLUMNS, TARGET_COLUMN, FORWARD_HORIZON,
    build_features, training_frame, #prediction_frame,
)

RESULTS_DIR = Path(__file__).parent / "experiments"
DATA_DIR = Path(__file__).parent / "data"

VAL_DAYS = 5
EMBARGO_DAYS = 6
N_SPLITS = 6
STEP_DAYS = 20

MIN_STOCKS = 30             # rule: portfolio must hold >= 30 names
MAX_WEIGHT = 0.10           # rule: per-stock weight cap
DEFAULT_TOP_K = 50          # baseline picks top-50 by predicted score

def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        #baseline
        # n_estimators=400,
        # max_depth=5,
        # learning_rate=0.05,
        # subsample=0.8,
        # colsample_bytree=0.8,
        # min_child_weight=10,
        # reg_lambda=1.0,
        # tree_method="hist",
        # n_jobs=-1,
        # early_stopping_rounds=30,

        # tuned
        n_estimators=400,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.65,
        min_child_weight=10,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=-1,
        early_stopping_rounds=30,
        random_state=42,
    )

    model.fit(
        train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN],
        eval_set=[(val_df[FEATURE_COLUMNS], val_df[TARGET_COLUMN])],
        verbose=False,
    )

    return model

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
        raise ValueError(f"top_k must be >= {MIN_STOCKS}")
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

    w = w / w.sum()

    assert abs(w.sum() - 1.0) < 1e-6, f"weights sum to {w.sum()}"
    assert (w <= MAX_WEIGHT + 1e-9).all(), "cap violated"
    assert (w > 0).sum() >= MIN_STOCKS, "too few names"

    return w

def make_splits(dates: np.ndarray) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Returns list of:
        train_end, val_start, val_end

    Each split:
        train <= train_end
        embargo gap
        val_start through val_end
    """
    dates = np.sort(pd.to_datetime(dates))
    splits = []

    latest_val_end_idx = len(dates) - 1 - FORWARD_HORIZON

    for i in range(N_SPLITS):
        val_end_idx = latest_val_end_idx - i * STEP_DAYS
        val_start_idx = val_end_idx - VAL_DAYS + 1
        train_end_idx = val_start_idx - EMBARGO_DAYS - 1

        if train_end_idx < 120:
            break

        train_end = pd.Timestamp(dates[train_end_idx])
        val_start = pd.Timestamp(dates[val_start_idx])
        val_end = pd.Timestamp(dates[val_end_idx])

        splits.append((train_end, val_start, val_end))

    return list(reversed(splits))


def evaluate_split(
    panel: pd.DataFrame,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
) -> dict:
    train_df = training_frame(panel, max_date=train_end)
    val_df = training_frame(panel, min_date=val_start, max_date=val_end)

    model = train_model(train_df, val_df)

    val_pred = model.predict(val_df[FEATURE_COLUMNS])
    ic = rank_ic(
        val_df[TARGET_COLUMN].to_numpy(),
        val_pred,
        val_df["date"].to_numpy(),
    )

    # Portfolio-style backtest inside validation window
    daily_results = []

    for d in sorted(val_df["date"].unique()):
        day_df = val_df[val_df["date"] == d].copy()
        day_df["score"] = model.predict(day_df[FEATURE_COLUMNS])

        weights = build_portfolio(day_df.set_index("stock_code")["score"], top_k=DEFAULT_TOP_K)

        realized = day_df.set_index("stock_code")[TARGET_COLUMN]
        port_ret = float((weights * realized.loc[weights.index]).sum())
        equal_ret = float(realized.mean())
        excess_ret = port_ret - equal_ret

        daily_results.append(
            {
                "date": pd.Timestamp(d),
                "portfolio_ret": port_ret,
                "equal_weight_ret": equal_ret,
                "excess_vs_equal": excess_ret,
            }
        )

    daily = pd.DataFrame(daily_results)

    return {
        "train_end": pd.Timestamp(train_end).date(),
        "val_start": pd.Timestamp(val_start).date(),
        "val_end": pd.Timestamp(val_end).date(),
        "rank_ic": ic,
        "portfolio_ret_mean": daily["portfolio_ret"].mean(),
        "equal_weight_ret_mean": daily["equal_weight_ret"].mean(),
        "excess_vs_equal_mean": daily["excess_vs_equal"].mean(),
        "win_rate_vs_equal": (daily["excess_vs_equal"] > 0).mean(),
        "n_train": len(train_df),
        "n_val": len(val_df),
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    prices_path = DATA_DIR / "prices.parquet"

    print(f">> Loading {prices_path}")
    prices = pd.read_parquet(prices_path)

    print(
        f"   {len(prices):,} rows, "
        f"{prices['stock_code'].nunique()} stocks, "
        f"{prices['date'].min()} to {prices['date'].max()}"
    )

    print(">> Building features")
    panel = build_features(prices)

    all_dates = np.sort(panel["date"].unique())
    splits = make_splits(all_dates)

    print(f">> Running {len(splits)} CV splits")
    
    results = []

    for k, (train_end, val_start, val_end) in enumerate(splits, start=1):
        print(
            f"\nSplit {k}: train <= {pd.Timestamp(train_end).date()}, "
            f"val {pd.Timestamp(val_start).date()} to {pd.Timestamp(val_end).date()}"
        )
        res = evaluate_split(panel, train_end, val_start, val_end)
        results.append(res)

        print(
            f"   rank IC={res['rank_ic']:.4f}, "
            f"excess={res['excess_vs_equal_mean']:.4%}, "
            f"win_rate={res['win_rate_vs_equal']:.2f}"
        )

    out = pd.DataFrame(results)
    print("\n=== CV Summary ===")
    print(out)

    print("\n=== Averages ===")
    print(out[[
        "rank_ic",
        "portfolio_ret_mean",
        "equal_weight_ret_mean",
        "excess_vs_equal_mean",
        "win_rate_vs_equal",
    ]].mean())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = RESULTS_DIR / f"cv_results_{timestamp}.csv"

    out.to_csv(outpath, index=False)
    print(f"\n>> Wrote {outpath}")

if __name__ == "__main__":
    main()