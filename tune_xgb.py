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

from features_baseline import (
    FEATURE_COLUMNS, TARGET_COLUMN, FORWARD_HORIZON,
    build_features, training_frame, 
    #prediction_frame, 
    #this files j for experimenting so we won't predict or build a portfolio, faster to remove this for now
)

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "experiments"

VAL_DAYS = 10               # number of trading days in the validation window
EMBARGO_DAYS = 6            # gap between train end and val start (>= FORWARD_HORIZON
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
    #second run: don't need to change, no randomness
    #for value in [200, 300, 500, 600]:
    #    configs.append({"name": f"n_estimators_{value}", **BASE_CONFIG, "n_estimators": value})

    #second run: 3 is best, done tuning this one 
    # for value in [3, 4, 6, 7]:
    #     configs.append({"name": f"max_depth_{value}", **BASE_CONFIG, "max_depth": value})

    # #second run: 0.03 0.04 0.05 ranked so lower is better, trying to go lower
    # #lower numbers might be from not enough trees rather than actual bad results
    # for value in [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040]:
    #     configs.append({"name": f"learning_rate_{value}", **BASE_CONFIG, "learning_rate": value})
    
    # #second run: 0.7 best, go smaller 0.6-0.8
    # #third run: 0.65 and 0.7 tied so i cld try the values in between 0.66-0.69 but that might j be noise so i will just keep 0.65 and 0.7 for now
    # for value in [0.60, 0.65, 0.70, 0.75, 0.80]:
    #     configs.append({"name": f"subsample_{value}", **BASE_CONFIG, "subsample": value})
    
    # for value in [0.60, 0.65, 0.70, 0.75, 0.80]:
    #     configs.append({"name": f"colsample_bytree_{value}", **BASE_CONFIG, "colsample_bytree": value})
    
    # #second run: 15 best but ranked 15 20 5 so test all around these
    # #third run" 25 was best so j gna add some extras above 25 
    # for value in [10, 20, 25, 27, 30]:
    #     configs.append({"name": f"min_child_weight_{value}", **BASE_CONFIG, "min_child_weight": value})
    
    # #second run: weaker run, 2.0 was good 1.0 - 5.0 
    # for value in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
    #     configs.append({"name": f"reg_lambda_{value}", **BASE_CONFIG, "reg_lambda": value})

    # combo experiments - best individual params combined
    combo_configs = [
        {
            "name": "combo_ic_focused",
            "n_estimators": 400,
            "max_depth": 4,           # 0.116340, good IC with lower std than 3
            "learning_rate": 0.02,    # best IC + lowest std
            "subsample": 0.7,
            "colsample_bytree": 0.65, # same as 0.7 but faster
            "min_child_weight": 10,   # baseline
            "reg_lambda": 1.0,        # baseline
        },
        {
            "name": "combo_stable_focused",
            "n_estimators": 400,
            "max_depth": 5,           # 0.110256, most stable overall
            "learning_rate": 0.02,    # best IC + lowest std
            "subsample": 0.7,
            "colsample_bytree": 0.65,
            "min_child_weight": 20,   # most stable (std 0.044817)
            "reg_lambda": 1.0,
        },
        {
            "name": "combo_balanced",
            "n_estimators": 400,
            "max_depth": 4,
            "learning_rate": 0.02,
            "subsample": 0.7,
            "colsample_bytree": 0.65,
            "min_child_weight": 20,   # try stable MCW with IC-focused depth
            "reg_lambda": 1.0,
        },
        {
            "name": "combo_high_lr",
            "n_estimators": 400,
            "max_depth": 5,
            "learning_rate": 0.03,    # next best after 0.02
            "subsample": 0.7,
            "colsample_bytree": 0.65,
            "min_child_weight": 20,
            "reg_lambda": 1.0,
        },
        # === Tier 1: All top 3 individual winners combined ===
        {
            "name": "combo_all_winners",
            "n_estimators": 400,
            "max_depth": 5,           # baseline, preserve stability
            "learning_rate": 0.02,    # best IC + lowest std
            "subsample": 0.7,         # better than 0.8
            "colsample_bytree": 0.65, # tied best IC, faster than 0.7
            "min_child_weight": 10,   # baseline
            "reg_lambda": 1.0,        # baseline
        },
        
        # === Tier 2: Colsample + LR interactions (top 2 performers) ===
        {
            "name": "combo_colsample_lr_deep",
            "n_estimators": 400,
            "max_depth": 3,           # test if high colsample needs shallower
            "learning_rate": 0.02,
            "subsample": 0.8,         # keep baseline
            "colsample_bytree": 0.65,
            "min_child_weight": 10,
            "reg_lambda": 1.0,
        },
        {
            "name": "combo_colsample_lr_deeper",
            "n_estimators": 400,
            "max_depth": 6,           # test if high colsample needs deeper
            "learning_rate": 0.02,
            "subsample": 0.8,
            "colsample_bytree": 0.65,
            "min_child_weight": 10,
            "reg_lambda": 1.0,
        },
        
        # === Tier 3: Test if high colsample needs high subsample ===
        {
            "name": "combo_high_sampling",  # hypothesis: more data = more colsample benefit
            "n_estimators": 400,
            "max_depth": 5,
            "learning_rate": 0.02,
            "subsample": 0.75,        # higher than 0.7
            "colsample_bytree": 0.65,
            "min_child_weight": 10,
            "reg_lambda": 1.0,
        },
        {
            "name": "combo_high_subsample_aggressive",
            "n_estimators": 400,
            "max_depth": 5,
            "learning_rate": 0.02,
            "subsample": 0.8,         # back to baseline, but test with colsample
            "colsample_bytree": 0.65,
            "min_child_weight": 10,
            "reg_lambda": 1.0,
        },
        
        # === Tier 4: Stabilization - lower reg_lambda had lower std ===
        {
            "name": "combo_stable_lr",
            "n_estimators": 400,
            "max_depth": 5,
            "learning_rate": 0.02,
            "subsample": 0.7,
            "colsample_bytree": 0.65,
            "min_child_weight": 10,
            "reg_lambda": 2.0,        # lower std configs used 2.0-2.5
        },
        
        # === Tier 5: Aggressive - colsample 0.7 (not 0.65) ===
        {
            "name": "combo_max_colsample",
            "n_estimators": 400,
            "max_depth": 5,
            "learning_rate": 0.02,
            "subsample": 0.7,
            "colsample_bytree": 0.7,  # tied best IC, no speed penalty if already run
            "min_child_weight": 10,
            "reg_lambda": 1.0,
        },
        
        # === Tier 6: Test lower LR with colsample ===
        {
            "name": "combo_lower_lr",
            "n_estimators": 400,
            "max_depth": 5,
            "learning_rate": 0.01,    # next tier down from 0.02
            "subsample": 0.7,
            "colsample_bytree": 0.65,
            "min_child_weight": 10,
            "reg_lambda": 1.0,
        },
        
        # === Final tuning round: locked subsample 0.8, colsample 0.65, lr 0.02 ===
        # Test max_depth finer range
        {
            "name": "final_depth_2",
            "n_estimators": 400,
            "max_depth": 2,
            "learning_rate": 0.02,
            "subsample": 0.8,
            "colsample_bytree": 0.65,
            "min_child_weight": 10,
            "reg_lambda": 1.0,
        },
        {
            "name": "final_depth_4",
            "n_estimators": 400,
            "max_depth": 4,
            "learning_rate": 0.02,
            "subsample": 0.8,
            "colsample_bytree": 0.65,
            "min_child_weight": 10,
            "reg_lambda": 1.0,
        },
        
        # Test min_child_weight
        {
            "name": "final_mcw_5",
            "n_estimators": 400,
            "max_depth": 3,
            "learning_rate": 0.02,
            "subsample": 0.8,
            "colsample_bytree": 0.65,
            "min_child_weight": 5,
            "reg_lambda": 1.0,
        },
        {
            "name": "final_mcw_15",
            "n_estimators": 400,
            "max_depth": 3,
            "learning_rate": 0.02,
            "subsample": 0.8,
            "colsample_bytree": 0.65,
            "min_child_weight": 15,
            "reg_lambda": 1.0,
        },
        
        # Test reg_lambda
        {
            "name": "final_reg_0.5",
            "n_estimators": 400,
            "max_depth": 3,
            "learning_rate": 0.02,
            "subsample": 0.8,
            "colsample_bytree": 0.65,
            "min_child_weight": 10,
            "reg_lambda": 0.5,
        },
        {
            "name": "final_reg_1.5",
            "n_estimators": 400,
            "max_depth": 3,
            "learning_rate": 0.02,
            "subsample": 0.8,
            "colsample_bytree": 0.65,
            "min_child_weight": 10,
            "reg_lambda": 1.5,
        },
        
        # Best combo from depth tests (guessing depth 3 is better)
        {
            "name": "final_locked_depth3",
            "n_estimators": 400,
            "max_depth": 3,
            "learning_rate": 0.02,
            "subsample": 0.8,
            "colsample_bytree": 0.65,
            "min_child_weight": 10,
            "reg_lambda": 1.0,
        },

        
    ]
    
    for combo in combo_configs:
        configs.append(combo)

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

def make_rolling_splits(train_pool, n_splits=3):
    all_dates = np.sort(train_pool["date"].unique())

    split_points = np.linspace(
        60, 
        len(all_dates) - VAL_DAYS,
        n_splits, 
        dtype=int,
    )

    splits = []

    for i, val_start_idx in enumerate(split_points, start=1):
        val_end_idx = val_start_idx + VAL_DAYS
        train_end_idx = val_start_idx - EMBARGO_DAYS - 1

        train_dates = all_dates[:train_end_idx+1]
        val_dates = all_dates[val_start_idx:val_end_idx]

        train_df = train_pool[train_pool["date"].isin(train_dates)]
        val_df = train_pool[train_pool["date"].isin(val_dates)]

        splits.append((i, train_df, val_df))

    return splits


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
    #all_dates = np.sort(train_pool["date"].unique())
    #if len(all_dates) < VAL_DAYS + EMBARGO_DAYS + 20:
    #    raise RuntimeError("Not enough dates to train; download more history.")
    
    #val_start = pd.Timestamp(all_dates[-VAL_DAYS])
    #train_end = pd.Timestamp(all_dates[-(VAL_DAYS + EMBARGO_DAYS + 1)])

    #train_df = train_pool[train_pool["date"] <= train_end]
    #val_df = train_pool[train_pool["date"] >= val_start]

    # print(f"   train: {len(train_df):,} rows up to {train_end.date()}")
    # print(f"   embargo: {EMBARGO_DAYS} trading days (discarded)")
    # print(f"   val:   {len(val_df):,} rows from {val_start.date()}")
    
    print(">> Creating 3 splits")
    splits = make_rolling_splits(train_pool, 3)
    for i, train_df, val_df in splits:
        print(f"    split {i}: train={len(train_df):,}, val={len(val_df):,}")

    configs = make_configs()
    results = []

    print(f"\n>> Running {len(configs)} COMBO experiments (EMBARGO_DAYS={EMBARGO_DAYS})")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_out_path = RESULTS_DIR / f"xgb_tuning_results_{timestamp}.csv"
    #ranking_out_path = RESULTS_DIR / f"xgb_param_rankings_{timestamp}.csv"

    for config in configs:
        name = config["name"]
        params = {k: v for k, v in config.items() if k != "name"}

        print(f"\n---{name}---")
        split_ics = []

        start_time = time.time()

        for split_id, train_df, val_df in splits:
            model = train_model(train_df, val_df, params)

            val_pred = model.predict(val_df[FEATURE_COLUMNS])
            ic = rank_ic(
                val_df[TARGET_COLUMN].to_numpy(),
                val_pred,
                val_df["date"].to_numpy()
            )
            split_ics.append(ic)

        avg_ic = np.mean(split_ics)
        std_ic = np.std(split_ics)
        elapsed = time.time() - start_time

        print(f"IC avg: {avg_ic:.4f} | std: {std_ic:.4f} | splits: {[round(x,4) for x in split_ics]}")

        results.append({
            "name": name,
            "ic": avg_ic,
            "ic_std": std_ic,
            "split1_ic": split_ics[0],
            "split2_ic": split_ics[1],
            "split3_ic": split_ics[2],
            "elapsed_time": elapsed,
            **params,
        })
        pd.DataFrame(results).sort_values("ic", ascending=False).to_csv(raw_out_path, index=False)

    
    print("\n=== Final Results ===")
    df = pd.DataFrame(results).sort_values("ic", ascending=False)
    print(df[["name", "ic", "ic_std", "split1_ic", "split2_ic", "split3_ic", "elapsed_time"]].to_string(index=False))

    print(f"\nSaved to {raw_out_path}")




    

if __name__ == "__main__":
    main()
