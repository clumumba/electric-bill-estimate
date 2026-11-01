from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from .config import PROCESSED_DIR, load_params


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["hour"] = frame.index.hour
    frame["day_of_week"] = frame.index.dayofweek
    frame["day_of_month"] = frame.index.day
    frame["month"] = frame.index.month
    frame["is_weekend"] = frame["day_of_week"].isin([5, 6]).astype(int)
    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24)
    frame["month_sin"] = np.sin(2 * np.pi * frame["month"] / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame["month"] / 12)
    return frame


def add_lag_features(df: pd.DataFrame, target_column: str, lags: list[int]) -> pd.DataFrame:
    frame = df.copy()
    for lag in lags:
        frame[f"lag_{lag}"] = frame[target_column].shift(lag)
    return frame


def _coerce_raw_dataframe(raw_data: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(raw_data, pd.DataFrame):
        return raw_data.copy()

    df = pd.read_csv(raw_data, index_col="timestamp")
    df.index = pd.to_datetime(df.index)
    return df.sort_index().copy()


def build_feature_matrix(raw_data: str | Path | pd.DataFrame, target_column: str | None = None, lags: list[int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str]:
    params = load_params()
    target_column = target_column or params.get("target_column", "MT_001")
    lags = lags if lags is not None else params.get("features", {}).get("lags", [1, 2, 3, 6, 12, 24, 48])

    df = _coerce_raw_dataframe(raw_data)
    from .data_validation import check_dataset
    check_dataset(df, target_column, int(params.get("billing", {}).get("interval_minutes", 15)))
    if not lags or any(not isinstance(lag, int) or isinstance(lag, bool) or lag <= 0 for lag in lags) or len(set(lags)) != len(lags):
        raise ValueError("Lags must be distinct positive integers.")
    df = create_time_features(df)
    df = add_lag_features(df, target_column, lags)

    # Other customers' readings at the prediction timestamp are leakage: they are
    # unavailable when the target customer's future consumption is forecast.
    feature_columns = [
        *params.get("features", {}).get("time", [
            "hour", "day_of_week", "day_of_month", "month", "is_weekend",
            "hour_sin", "hour_cos", "month_sin", "month_cos",
        ]),
        *[f"lag_{lag}" for lag in lags],
    ]
    X = df[feature_columns]
    y = df[target_column]
    dataset = pd.concat([X, y], axis=1).dropna()
    dataset.index.name = "timestamp"
    dataset = dataset.reset_index()

    test_size = float(params.get("model", {}).get("test_size", 0.2))
    if not 0 < test_size < 1:
        raise ValueError("model.test_size must be between 0 and 1.")
    split_at = int(len(dataset) * (1 - test_size))
    if split_at == 0 or split_at == len(dataset):
        raise ValueError("Dataset is too small for the configured chronological split.")
    train = dataset.iloc[:split_at].copy()
    test = dataset.iloc[split_at:].copy()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    feature_schema = PROCESSED_DIR / "feature_schema.json"
    with feature_schema.open("w", encoding="utf-8") as file:
        json.dump({
            "target_column": target_column,
            "feature_columns": feature_columns,
            "forecast_horizon_steps": 1,
            "target_unit": "kW",
            "interval_minutes": int(params.get("billing", {}).get("interval_minutes", 15)),
        }, file, indent=2)

    return train, test, feature_columns, target_column


def prepare_data(raw_data: str | Path | pd.DataFrame, target_column: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str]:
    return build_feature_matrix(raw_data, target_column=target_column)
