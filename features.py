"""
Feature engineering for the CSI500 stock-selection baseline.

A small set of classic technical features + cross-sectional ranks.  Students are
encouraged to extend this (add fundamentals, industry dummies, alternative data,
better cross-sectional normalization, etc.).

The target is the 5-trading-day forward return on the forward-adjusted close,
i.e. what the portfolio earns if you hold a $1 position from close(t) to close(t+5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# columns used downstream by the baseline

# momentum features - capture trend
# ret_1d, ret_5d -> fast signals, 20d, 60d -> slower
# stocks that went up recently often keep going up, but not always, so we want multiple horizons

# volatility features - measure how noisy the stock is
# high vol -> risky/unstable, low vol -> stable, but also less likely to have big gains
# momentum works better on low value stocks 

# volume features - measure trading activity
# volume_z_20d = (volume - mean) / std, turnover_ma_20d = 20-day moving average of turnover
# high volume spike = attention/news/liquidity
# breakout + high volume = strong signal

# mean reversion features - capture tendency to revert to mean
# close_over_ma20, close_over_ma60 = how far above/below the 20/60-day moving average (how far price is from trend)
# price / moving average - 1
# 0 -> above trend bullish, < 0 -> below trend bearish, but extreme values can also indicate overbought/oversold conditions
# models can think too far above MA -> might revert or above MA with strong momentum -> might keep going up

# RSI - momentum osillator that measures speed and change of price movements
# rsi_14 = 100 - 100 / (1 + RS), where RS = average gain over 14 days / average loss over 14 days
# measures overbought/oversold conditions, typically 70+ = overbought (potentially bearish), 30- = oversold (potentially bullish)
# used for mean reversion strategies, but can also indicate strong momentum if RSI is rising and above 50

# cross sectional ranks 
# ret_5d_ranlk, ret_20d_rank, vol_20d_rank = daily cross-sectional rank of ret_5d, ret_20d, vol_20d
# instead of raw values, rank stocks against each other per day, captures relative strength/volatility
# rank 0 = lowest value (most negative return or lowest volatility), rank 1 = highest value (most positive return or highest volatility)

# target_5d - what we're trying to predict - the 5-day forward return on the forward-adjusted close

FEATURE_COLUMNS = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d", 
    "vol_20d", "volume_z_20d", "turnover_ma_20d",
    "close_over_ma20", "close_over_ma60", "rsi_14",
    "ret_5d_rank", "ret_20d_rank", "vol_20d_rank",
]
TARGET_COLUMN = "target_5d"
FORWARD_HORIZON = 5


def _per_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features that only depend on a single stock's time series."""
    df = df.sort_values("date").copy()
    close = df["close"]

    df["ret_1d"] = close.pct_change(1)
    df["ret_5d"] = close.pct_change(5)
    df["ret_10d"] = close.pct_change(10)
    df["ret_20d"] = close.pct_change(20)
    df["ret_60d"] = close.pct_change(60)

    df["vol_20d"] = df["ret_1d"].rolling(20).std()

    vol = df["volume"].astype(float)
    vol_mean = vol.rolling(20).mean()
    vol_std = vol.rolling(20).std().replace(0, np.nan)
    df["volume_z_20d"] = (vol - vol_mean) / vol_std

    if "turnover" in df.columns:
        df["turnover_ma_20d"] = df["turnover"].astype(float).rolling(20).mean()
    else:
        df["turnover_ma_20d"] = np.nan

    df["close_over_ma20"] = close / close.rolling(20).mean() - 1.0
    df["close_over_ma60"] = close / close.rolling(60).mean() - 1.0

    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    rs = up / down
    df["rsi_14"] = 100 - 100 / (1 + rs)

    df[TARGET_COLUMN] = close.shift(-FORWARD_HORIZON) / close - 1.0
    return df


def _cross_sectional_ranks(panel: pd.DataFrame) -> pd.DataFrame:
    """Daily cross-sectional rank of selected features (values in [0, 1])."""
    for base in ["ret_5d", "ret_20d", "vol_20d"]:
        panel[f"{base}_rank"] = (
            panel.groupby("date")[base].rank(method="average", pct=True)
        )
    return panel


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Build a (date, stock_code) panel of features + target.

    Parameters
    ----------
    prices : DataFrame with columns [date, stock_code, open, close, high, low,
             volume, amount, turnover?]

    Returns
    -------
    DataFrame with FEATURE_COLUMNS and TARGET_COLUMN populated.  Rows where any
    feature is NaN (typically the first ~60 days per stock) are kept so callers
    can decide how to handle them.
    """
    required = {"date", "stock_code", "close", "volume"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices is missing required columns: {missing}")

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    panel = (
        prices.groupby("stock_code", group_keys=False)
        .apply(_per_stock_features)
        .reset_index(drop=True)
    )
    panel = _cross_sectional_ranks(panel)
    return panel


def training_frame(panel: pd.DataFrame, min_date=None, max_date=None) -> pd.DataFrame:
    """Rows usable for supervised training: all features present AND target present.

    The target for date t uses close(t+5), so rows within the last 5 trading
    days of the panel are dropped automatically (target is NaN there).
    """
    df = panel.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    if min_date is not None:
        df = df[df["date"] >= pd.Timestamp(min_date)]
    if max_date is not None:
        df = df[df["date"] <= pd.Timestamp(max_date)]
    return df


def prediction_frame(panel: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """Rows for a single prediction date (defaults to the latest date)."""
    if as_of is None:
        as_of = panel["date"].max()
    as_of = pd.Timestamp(as_of)
    df = panel[panel["date"] == as_of].dropna(subset=FEATURE_COLUMNS).copy()
    return df
