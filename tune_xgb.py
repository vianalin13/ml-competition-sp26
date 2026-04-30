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

Tune XGBoost hyperparameters for the CSI500 stock-selection competition.

Run:
    python tune_xgb.py

Outputs:
    experiments/xgb_tuning_results_<timestamp>.csv 
    experiments/xgb_param_rankings_<timestamp>.csv
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

from features import (
    FEATURE_COLUMNS, TARGET_COLUMN, FORWARD_HORIZON,
    build_features, training_frame, 
    #prediction_frame, 
    #this files j for experimenting so we won't predict or build a portfolio, faster to remove this for now
)

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "experiments"

VAL_DAYS = 10               # number of trading days in the validation window
EMBARGO_DAYS = 5            # gap between train end and val start (>= FORWARD_HORIZON
                            # so training targets don't reach into val dates)


# -------configs--------
BASE_CONFIG =    {
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "reg_lambda": 1.0,
}

def make_configs():
    configs = []

    #baseline
    configs.append({"name": "baseline", **BASE_CONFIG})

    #one param at a time experiments
    for value in [200, 300, 500, 600]:
        configs.append({"name": f"n_estimators_{value}", **BASE_CONFIG, "n_estimators": value})

    for value in [3, 4, 6, 7]:
        configs.append({"name": f"max_depth_{value}", **BASE_CONFIG, "max_depth": value})

    for value in [0.03, 0.04, 0.06, 0.07]:
        configs.append({"name": f"learning_rate_{value}", **BASE_CONFIG, "learning_rate": value})
    
    for value in [0.6, 0.7, 0.9, 1.0]:
        configs.append({"name": f"subsample_{value}", **BASE_CONFIG, "subsample": value})
    
    for value in [0.6, 0.7, 0.9, 1.0]:
        configs.append({"name": f"colsample_bytree_{value}", **BASE_CONFIG, "colsample_bytree": value})
    
    for value in [5, 7, 15, 20]:
        configs.append({"name": f"min_child_weight_{value}", **BASE_CONFIG, "min_child_weight": value})
    
    for value in [0.5, 0.8 , 2.0, 5.0]:
        configs.append({"name": f"reg_lambda_{value}", **BASE_CONFIG, "reg_lambda": value})

    return configs


def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame, params) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        **params,
        tree_method="hist",
        n_jobs=-1,
        early_stopping_rounds=30,
        random_state=42, #same data, same params -> deterministic results, so we can compare across runs
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

def build_param_ranking(results_df: pd.DataFrame) -> pd.DataFrame:
    param_groups = [
        "n_estimators", 
        "max_depth",
        "learning_rate", 
        "subsample",
        "colsample_bytree", 
        "min_child_weight", 
        "reg_lambda",
    ]

    summary_rows = []

    for param in param_groups:
        group_df = results_df[["name", "ic", param]].copy()
        group_df = group_df[
            group_df["name"].eq("baseline") 
            | group_df["name"].str.startswith(param)
        ]

        if group_df.empty: 
            continue

        group_df = group_df.sort_values("ic", ascending=False)

        print(f"\n=== {param} ranking ===")
        print(group_df.to_string(index=False))

        for rank, (_, row) in enumerate(group_df.iterrows(), start=1):
            summary_rows.append({
                "param": param, 
                "rank": rank, 
                "config_name": row["name"],
                "value": row[param],
                "ic": row["ic"],
            })
        
    return pd.DataFrame(summary_rows)

def main():

    RESULTS_DIR.mkdir(exist_ok=True)
    
    #remove arg parser, just use latest data same split logic, loop configs
    #don't need to pass arg every time

    print(f">> Loading prices")
    prices = pd.read_parquet(DATA_DIR / "prices.parquet")

    print(f"{len(prices):,} rows, "
          f"{prices['stock_code'].nunique()} stocks, "
          f"dates {prices['date'].min().date()} to {prices['date'].max().date()}"
    )

    print(">> Building features")
    panel = build_features(prices)
    # Bound training data so backtesting with --as-of doesn't leak future rows.
    # Training uses features from date t with target = close(t+FORWARD_HORIZON),
    # so we cap training dates at as_of - FORWARD_HORIZON trading days.
    as_of_ts = panel["date"].max()
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

    configs = make_configs()
    results = []

    print(f"\n>> Running {len(configs)} XGBoost configs experiments")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_out_path = RESULTS_DIR / f"xgb_tuning_results_{timestamp}.csv"
    ranking_out_path = RESULTS_DIR / f"xgb_param_rankings_{timestamp}.csv"

    total_start = time.time()
    for i, config in enumerate(configs, start=1):
        name = config["name"]
        params = {k: v for k, v in config.items() if k != "name"}

        print(f"\n[{i}/{len(configs)}] {name}")
        start_time = time.time()

        model = train_model(train_df, val_df, params)

        val_pred = model.predict(val_df[FEATURE_COLUMNS])
        ic = rank_ic(
            val_df[TARGET_COLUMN].to_numpy(),
            val_pred,
            val_df["date"].to_numpy()
        )

        elapsed = time.time() - start_time

        print(f" IC: {ic:.4f}  time: {elapsed:.1f}s")

        results.append({
            "name": name,
            "ic": ic,
            "elapsed_time": elapsed,
            **params,
        })

        results_df = pd.DataFrame(results).sort_values("ic", ascending=False)
        results_df.to_csv(raw_out_path, index=False)
        

    
    print("\n=== Final Results ===")
    results_df = pd.DataFrame(results).sort_values("ic", ascending=False)
    print(results_df)

    print("\n=== Best values by parameter group ===")
    ranking_df = build_param_ranking(results_df)
    ranking_df.to_csv(ranking_out_path, index=False)

    print(f"Saved results to {raw_out_path}")
    print(f"Saved parameter rankings to {ranking_out_path}")

    total_time = time.time() - total_start
    print(f"\n>> Total runtime: {total_time/60:.2f} minutes ({total_time:.1f}s)")

if __name__ == "__main__":
    main()
