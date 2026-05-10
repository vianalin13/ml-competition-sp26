"""
XGBoost ensemble CV for the CSI500 competition.

What this does
--------------
1. Trains 3 intentionally different XGBoost variants.
2. Scores each variant individually using scorer-style rolling CV.
3. Builds an equal-weight ensemble from standardized model scores (before weighting), and scores it using the same CV windows.
4. Builds a weighted ensemble from standardized model scores, with weights chosen based on prior experience and intuition about the risk/return tradeoff of each variant.
5. Scores the ensemble using the same CV windows.
6. Writes detailed and summary CSVs under ./experiments.

Run
---
python cv_xgb_ensemble.py
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

from features_benchmark import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    FORWARD_HORIZON,
    build_features,
    training_frame,
)
from ridge_model import RIDGE_CONFIG, train_ridge_model

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "experiments"

VAL_DAYS = 10
EMBARGO_DAYS = 5
N_SPLITS = 6
STEP_DAYS = 20

MIN_STOCKS = 30
MAX_WEIGHT = 0.10
ENSEMBLE_TOP_K = 50



# keep these simple first. After this works, add more variants.
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

def build_portfolio(scores: pd.Series, top_k: int) -> pd.Series:
    """Top-K rank-weighted long-only portfolio with 10% cap."""
    if top_k < MIN_STOCKS:
        raise ValueError(f"top_k must be >= {MIN_STOCKS}")

    chosen = scores.sort_values(ascending=False).head(top_k)
    ranks = np.arange(top_k, 0, -1, dtype=float)
    w = pd.Series(ranks / ranks.sum(), index=chosen.index)

    # Cap at 10% and redistribute excess to uncapped names.
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
    assert (w <= MAX_WEIGHT + 1e-9).all(), "max weight cap violated"
    assert (w > 0).sum() >= MIN_STOCKS, "too few names"

    return w


def realized_stock_returns(
    prices: pd.DataFrame,
    weights: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    """Match score_submission.py return convention."""
    rets = {}

    for code in weights.index:
        df = prices[prices["stock_code"] == code].sort_values("date")
        in_window = df[(df["date"] >= start) & (df["date"] <= end)]

        if in_window.empty:
            rets[code] = 0.0
            continue

        before = df[df["date"] < start]
        entry = before["close"].iloc[-1] if not before.empty else in_window["open"].iloc[0]
        exit_ = in_window["close"].iloc[-1]

        if entry <= 0 or pd.isna(entry) or pd.isna(exit_):
            rets[code] = 0.0
        else:
            rets[code] = float(exit_ / entry - 1.0)

    return pd.Series(rets)


def realized_index_return(index_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """CSI500 benchmark return using same convention as score_submission.py."""
    idx_window = index_df[(index_df["date"] >= start) & (index_df["date"] <= end)].sort_values("date")

    if idx_window.empty:
        raise RuntimeError(f"No index data from {start.date()} to {end.date()}")

    idx_before = index_df[index_df["date"] < start]
    entry = idx_before["close"].iloc[-1] if not idx_before.empty else idx_window["open"].iloc[0]
    exit_ = idx_window["close"].iloc[-1]

    return float(exit_ / entry - 1.0)


def make_splits(dates: np.ndarray):
    """
    Returns rolling CV splits:
        train <= train_end
        predict on as_of
        evaluate val_start through val_end
    """
    dates = np.sort(pd.to_datetime(dates))
    splits = []

    latest_val_end_idx = len(dates) - 1

    for i in range(N_SPLITS):
        val_end_idx = latest_val_end_idx - i * STEP_DAYS
        val_start_idx = val_end_idx - VAL_DAYS + 1
        as_of_idx = val_start_idx - 1
        train_end_idx = as_of_idx - FORWARD_HORIZON - EMBARGO_DAYS

        if train_end_idx < 120:
            break

        splits.append(
            (
                pd.Timestamp(dates[train_end_idx]),
                pd.Timestamp(dates[as_of_idx]),
                pd.Timestamp(dates[val_start_idx]),
                pd.Timestamp(dates[val_end_idx]),
            )
        )

    return list(reversed(splits))


def evaluate_scores(
    scores: pd.Series,
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    top_k: int,
):
    weights = build_portfolio(scores, top_k=top_k)

    stock_rets = realized_stock_returns(prices, weights, val_start, val_end)
    portfolio_return = float((weights * stock_rets.loc[weights.index]).sum())
    benchmark_return = realized_index_return(index_df, val_start, val_end)
    excess_return = portfolio_return - benchmark_return

    return portfolio_return, benchmark_return, excess_return, weights


def evaluate_split(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    train_end: pd.Timestamp,
    as_of: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
) -> list[dict]:
    train_pool = training_frame(panel, max_date=train_end)
    all_train_dates = np.sort(train_pool["date"].unique())

    if len(all_train_dates) < 30:
        raise RuntimeError("Not enough training dates")

    # Internal early-stopping validation set, before the actual CV eval window.
    es_days = min(10, max(3, len(all_train_dates) // 5))
    es_start = pd.Timestamp(all_train_dates[-es_days])

    fit_train_df = train_pool[train_pool["date"] < es_start]
    es_df = train_pool[train_pool["date"] >= es_start]

    if fit_train_df.empty or es_df.empty:
        raise RuntimeError("Empty train or early-stopping frame")

    pred_df = panel[panel["date"] == as_of].dropna(subset=FEATURE_COLUMNS).copy()
    if pred_df.empty:
        raise RuntimeError(f"No prediction rows for as_of={as_of.date()}")

    pred_df["stock_code"] = pred_df["stock_code"].astype(str).str.zfill(6)

    results = []
    score_table = pd.DataFrame({"stock_code": pred_df["stock_code"]})

    for name, config in VARIANTS.items():
        print(f"      training {name}")

        model = train_model(fit_train_df, es_df, config)
        raw_scores = model.predict(pred_df[FEATURE_COLUMNS])
        scores = pd.Series(raw_scores, index=pred_df["stock_code"])

        portfolio_return, benchmark_return, excess_return, weights = evaluate_scores(
            scores=scores,
            prices=prices,
            index_df=index_df,
            val_start=val_start,
            val_end=val_end,
            top_k=config["top_k"],
        )

        pred_with_target = pred_df.copy()
        pred_with_target["score"] = raw_scores
        pred_with_target = pred_with_target.dropna(subset=[TARGET_COLUMN])

        ic = rank_ic(
            pred_with_target[TARGET_COLUMN].to_numpy(),
            pred_with_target["score"].to_numpy(),
            pred_with_target["date"].to_numpy(),
        )

        results.append(
            {
                "model": name,
                "train_end": train_end.date(),
                "as_of": as_of.date(),
                "val_start": val_start.date(),
                "val_end": val_end.date(),
                "rank_ic_asof": ic,
                "portfolio_return": portfolio_return,
                "benchmark_return": benchmark_return,
                "excess_return": excess_return,
                "top_k": config["top_k"],
                "max_weight": weights.max(),
                "min_weight": weights.min(),
                "n_names": len(weights),
            }
        )

        score_table[name] = raw_scores

    # Equal-weight ensemble. Standardize each model's scores first so one model's
    # score scale does not dominate.
    score_table = score_table.set_index("stock_code")
    ensemble_score = pd.Series(0.0, index=score_table.index)

    # for name in VARIANTS:
    #     s = score_table[name]
    #     s = (s - s.mean()) / (s.std() + 1e-12)
    #     ensemble_score += s / len(VARIANTS)

    ensemble_weights = {
    # "xgb_aggressive_top30": 0.40,
    # "xgb_stable_top50": 0.25,
    # "xgb_conservative_top75": 0.20,
    # "ridge": 0.15, # apparently ridge is performing way better than xgb

    # "xgb_aggressive_top30": 0.25,
    # "xgb_stable_top50": 0.20,
    # "xgb_conservative_top75": 0.20,
    # "ridge": 0.35,

    # "xgb_aggressive_top30": 0.30,
    # "xgb_stable_top50": 0.15,
    # "xgb_conservative_top75": 0.15,
    # "ridge": 0.40,

    # "xgb_aggressive_top30": 0.20,
    # "xgb_conservative_top75": 0.15,
    # "xgb_stable_top50": 0,
    # "ridge": 0.65,

    # "xgb_aggressive_top30": 0.15,
    # "xgb_stable_top50": 0,
    # "xgb_conservative_top75": 0.20,
    # "ridge": 0.65,

    "xgb_aggressive_top30": 0.30,
    "xgb_stable_top50": 0.10,
    "xgb_conservative_top75": 0.10,
    "ridge": 0.50,
}

    for name in VARIANTS:
        s = score_table[name]
        s = (s - s.mean()) / (s.std() + 1e-12)
        ensemble_score += ensemble_weights[name] * s

    portfolio_return, benchmark_return, excess_return, weights = evaluate_scores(
        scores=ensemble_score,
        prices=prices,
        index_df=index_df,
        val_start=val_start,
        val_end=val_end,
        top_k=ENSEMBLE_TOP_K,
    )

    results.append(
        {
            # "model": "ENSEMBLE_EQUAL_3XGB",
            "model": "ENSEMBLE_WEIGHTED_3XGB",
            "train_end": train_end.date(),
            "as_of": as_of.date(),
            "val_start": val_start.date(),
            "val_end": val_end.date(),
            "rank_ic_asof": np.nan,
            "portfolio_return": portfolio_return,
            "benchmark_return": benchmark_return,
            "excess_return": excess_return,
            "top_k": ENSEMBLE_TOP_K,
            "max_weight": weights.max(),
            "min_weight": weights.min(),
            "n_names": len(weights),
        }
    )

    return results


def summarize_results(out: pd.DataFrame) -> pd.DataFrame:
    return (
        out.groupby("model")
        .agg(
            mean_excess=("excess_return", "mean"),
            median_excess=("excess_return", "median"),
            std_excess=("excess_return", "std"),
            worst_excess=("excess_return", "min"),
            best_excess=("excess_return", "max"),
            win_rate=("excess_return", lambda x: (x > 0).mean()),
            mean_ic=("rank_ic_asof", "mean"),
        )
        .sort_values("mean_excess", ascending=False)
    )


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
        f"dates {prices['date'].min().date()} to {prices['date'].max().date()}"
    )

    print(">> Building features")
    panel = build_features(prices)
    panel["stock_code"] = panel["stock_code"].astype(str).str.zfill(6)

    splits = make_splits(panel["date"].unique())
    print(f">> Running {len(splits)} CV splits")

    all_results = []

    for i, (train_end, as_of, val_start, val_end) in enumerate(splits, start=1):
        print(
            f"\nSplit {i}: train <= {train_end.date()}, "
            f"as_of={as_of.date()}, eval {val_start.date()} to {val_end.date()}"
        )

        split_results = evaluate_split(
            panel=panel,
            prices=prices,
            index_df=index_df,
            train_end=train_end,
            as_of=as_of,
            val_start=val_start,
            val_end=val_end,
        )
        all_results.extend(split_results)

        split_df = pd.DataFrame(split_results)
        print(split_df[["model", "excess_return", "portfolio_return", "benchmark_return", "top_k"]])

    out = pd.DataFrame(all_results)
    summary = summarize_results(out)

    print("\n=== SUMMARY ===")
    print(summary)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = RESULTS_DIR / f"xgb_ensemble_starter_detail_{timestamp}.csv"
    summary_path = RESULTS_DIR / f"xgb_ensemble_starter_summary_{timestamp}.csv"

    out.to_csv(detail_path, index=False)
    summary.to_csv(summary_path)

    print(f"\n>> Wrote detail results: {detail_path}")
    print(f">> Wrote summary results: {summary_path}")


if __name__ == "__main__":
    main()
