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
Rolling CV for CSI500 XGBoost model

1. Train only on past data
2. Use an embargo before prediction/evaluation
3. Predict once as of the trading day before the validation window
4. Build one portfolio
5. Hold that portfolio through the validation window
6. Score portfolio return minus CSI500 benchmark return
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

#from features_baseline import (
#from features_liquidity import (
from features_benchmark import (
#from features_cross_sectional import (
    FEATURE_COLUMNS, TARGET_COLUMN, FORWARD_HORIZON,
    build_features, training_frame, #prediction_frame,
)

RESULTS_DIR = Path(__file__).parent / "experiments"
DATA_DIR = Path(__file__).parent / "data"

VAL_DAYS = 10 #3 #week 1 live eval is 3 days may 6-8 
EMBARGO_DAYS = 5
N_SPLITS = 6
STEP_DAYS = 20

MIN_STOCKS = 30             # rule: portfolio must hold >= 30 names
MAX_WEIGHT = 0.10           # rule: per-stock weight cap

DEFAULT_TOP_K = 30   
#DEFAULT_TOP_K = 40   
#DEFAULT_TOP_K = 50          # baseline picks top-50 by predicted score
#DEFAULT_TOP_K = 75  
#DEFAULT_TOP_K = 100   

def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        # baseline
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=-1,
        early_stopping_rounds=30,

        # tuned
        # n_estimators=400,
        # max_depth=4,
        # learning_rate=0.02,
        # subsample=0.8,
        # colsample_bytree=0.65,
        # min_child_weight=10,
        # reg_lambda=1.0,
        # tree_method="hist",
        # n_jobs=-1,
        # early_stopping_rounds=30,

        # only good when valday=3
        # n_estimators=400,
        # max_depth=6,
        # learning_rate=0.04,
        # subsample=0.8,
        # colsample_bytree=0.7,
        # min_child_weight=10,
        # reg_lambda=1.0,
        # tree_method="hist",
        # n_jobs=-1,
        # early_stopping_rounds=30,
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

def realized_stock_returns(
    prices: pd.DataFrame,
    weights: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    """
    Match score_submission.py convention:

    entry = close on last trading day before start
            or open on start if prior close unavailable

    exit = close on end
           or last available close inside window if suspended/missing
    """
    rets = {}

    for code in weights.index:
        df = prices[prices["stock_code"] == code].sort_values("date")
        in_window = df[(df["date"] >= start) & (df["date"] <= end)]

        if in_window.empty:
            rets[code] = 0.0
            continue

        before = df[df["date"] < start]

        if not before.empty:
            entry = before["close"].iloc[-1]
        else:
            entry = in_window["open"].iloc[0]

        exit_ = in_window["close"].iloc[-1]

        if entry <= 0 or pd.isna(entry) or pd.isna(exit_):
            rets[code] = 0.0
        else:
            rets[code] = float(exit_ / entry - 1.0)

    return pd.Series(rets)

def realized_index_return(
    index_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> float:
    """CSI500 benchmark return using same convention as score_submission.py."""
    idx_window = index_df[
        (index_df["date"] >= start) & (index_df["date"] <= end)
    ].sort_values("date")

    if idx_window.empty:
        raise RuntimeError(f"No index data in [{start.date()}, {end.date()}]")

    idx_before = index_df[index_df["date"] < start]

    if not idx_before.empty:
        entry = idx_before["close"].iloc[-1]
    else:
        entry = idx_window["open"].iloc[0]

    exit_ = idx_window["close"].iloc[-1]

    return float(exit_ / entry - 1.0)

def make_splits(dates: np.ndarray) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
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

    latest_val_end_idx = len(dates) - 1 #- FORWARD_HORIZON

    for i in range(N_SPLITS):
        val_end_idx = latest_val_end_idx - i * STEP_DAYS
        val_start_idx = val_end_idx - VAL_DAYS + 1
        as_of_idx = val_start_idx - 1

        # Ensure no target leakage into eval period.
        train_end_idx = as_of_idx - FORWARD_HORIZON - EMBARGO_DAYS

        if train_end_idx < 120:
            break

        train_end = pd.Timestamp(dates[train_end_idx])
        as_of = pd.Timestamp(dates[as_of_idx])
        val_start = pd.Timestamp(dates[val_start_idx])
        val_end = pd.Timestamp(dates[val_end_idx])

        splits.append((train_end, as_of, val_start, val_end))

    return list(reversed(splits))


def evaluate_split(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    train_end: pd.Timestamp,
    as_of: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
) -> dict:
    train_pool = training_frame(panel, max_date=train_end)

    all_train_dates = np.sort(train_pool["date"].unique())

    if len(all_train_dates) < 30:
        raise RuntimeError("Not enough training dates.")

    # Internal early-stopping validation window.
    # This is before as_of and before the live validation window.
    es_days = min(10, max(3, len(all_train_dates) // 5))
    es_start = pd.Timestamp(all_train_dates[-es_days])

    fit_train_df = train_pool[train_pool["date"] < es_start]
    es_df = train_pool[train_pool["date"] >= es_start]
    
    if fit_train_df.empty or es_df.empty:
        raise RuntimeError("Empty train or early-stopping validation frame.")

    model = train_model(fit_train_df, es_df)

    pred_df = panel[panel["date"] == as_of].dropna(subset=FEATURE_COLUMNS).copy()

    if pred_df.empty:
        raise RuntimeError(f"No prediction rows for as_of={as_of.date()}")

    pred_df["score"] = model.predict(pred_df[FEATURE_COLUMNS])

    weights = build_portfolio(
        pred_df.set_index("stock_code")["score"],
        top_k=DEFAULT_TOP_K,
    )

    stock_rets = realized_stock_returns(prices, weights, val_start, val_end)
    portfolio_return = float((weights * stock_rets.loc[weights.index]).sum())

    benchmark_return = realized_index_return(index_df, val_start, val_end)
    excess_return = portfolio_return - benchmark_return

    # Optional sanity IC on as_of target if target exists.
    pred_with_target = pred_df.dropna(subset=[TARGET_COLUMN])

    if pred_with_target.empty:
        ic = float("nan")
    else:
        ic = rank_ic(
            pred_with_target[TARGET_COLUMN].to_numpy(),
            pred_with_target["score"].to_numpy(),
            pred_with_target["date"].to_numpy(),
        )

    return {
        "train_end": train_end.date(),
        "as_of": as_of.date(),
        "val_start": val_start.date(),
        "val_end": val_end.date(),
        "rank_ic_asof": ic,
        "portfolio_return": portfolio_return,
        "benchmark_return": benchmark_return,
        "excess_return": excess_return,
        "top_k": len(weights),
        "max_weight": weights.max(),
        "min_weight": weights.min(),
        "n_train": len(fit_train_df),
        "n_es": len(es_df),
        "n_pred": len(pred_df),
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    prices_path = DATA_DIR / "prices.parquet"
    index_path = DATA_DIR / "index.parquet"

    print(f">> Loading {prices_path}")
    prices = pd.read_parquet(prices_path)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)

    print(f">> Loading {index_path}")
    index_df = pd.read_parquet(index_path)
    index_df["date"] = pd.to_datetime(index_df["date"])

    print(
        f"   {len(prices):,} rows, "
        f"{prices['stock_code'].nunique()} stocks, "
        f"{prices['date'].min().date()} to {prices['date'].max().date()}"
    )

    print(">> Building features")
    panel = build_features(prices)
    panel["stock_code"] = panel["stock_code"].astype(str).str.zfill(6)

    all_dates = np.sort(panel["date"].unique())
    splits = make_splits(all_dates)

    print(f">> Running {len(splits)} CV splits")
    
    results = []

    for k, (train_end, as_of, val_start, val_end) in enumerate(splits, start=1):
        print(
            f"\nSplit {k}: "
            f"train <= {train_end.date()}, "
            f"as_of={as_of.date()}, "
            f"eval {val_start.date()} to {val_end.date()}"
        )
        res = evaluate_split(
            panel=panel,
            prices=prices,
            index_df=index_df,
            train_end=train_end,
            as_of=as_of,
            val_start=val_start,
            val_end=val_end,
        )
        results.append(res)

        print(
            f"   IC={res['rank_ic_asof']:.4f}, "
            f"portfolio={res['portfolio_return']:.4%}, "
            f"benchmark={res['benchmark_return']:.4%}, "
            f"excess={res['excess_return']:.4%}"
        )

    out = pd.DataFrame(results)
    print("\n=== CV Summary ===")
    print(out)

    print("\n=== Averages ===")
    print(out[[
        "rank_ic_asof",
        "portfolio_return",
        "benchmark_return",
        "excess_return",
    ]].mean())

    print("\n=== Win Rate vs CSI500 ===")
    print((out["excess_return"] > 0).mean())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = RESULTS_DIR / f"cv_real_scorer_style_{timestamp}.csv"

    out.to_csv(outpath, index=False)
    print(f"\n>> Wrote {outpath}")

if __name__ == "__main__":
    main()