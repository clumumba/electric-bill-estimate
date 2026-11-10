from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import VALIDATION_DIR, load_params


def dataset_issues(df: pd.DataFrame, target_column: str, interval_minutes: int) -> list[str]:
    issues = []
    if df.empty:
        return ["Dataset is empty."]
    if target_column not in df.select_dtypes(include="number").columns:
        issues.append(f"Numeric target '{target_column}' is missing.")
    if not isinstance(df.index, pd.DatetimeIndex):
        issues.append("The index must be a datetime index.")
    else:
        if df.index.hasnans or df.index.has_duplicates or not df.index.is_monotonic_increasing:
            issues.append("Timestamps must be valid, unique, and increasing.")
        if interval_minutes <= 0 or not (df.index.to_series().diff().iloc[1:] == pd.Timedelta(minutes=interval_minutes)).all():
            issues.append("Timestamps must follow the configured interval without gaps.")
    numeric = df.select_dtypes(include="number")
    if len(numeric.columns) != len(df.columns):
        issues.append("All reading columns must be numeric.")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        issues.append("Readings must be finite and non-null.")
    if (numeric < 0).any().any():
        issues.append("Readings must be non-negative.")
    return issues

def check_dataset(df: pd.DataFrame, target_column: str, interval_minutes: int) -> None:
    issues = dataset_issues(df, target_column, interval_minutes)
    if issues:
        raise ValueError(f"Dataset validation failed: {issues}")

def validate_dataset(df: pd.DataFrame, target_column: str | None = None) -> dict:
    params = load_params()
    target_column = target_column or params.get("target_column", "MT_001")
    interval = int(params.get("billing", {}).get("interval_minutes", 15))
    issues = dataset_issues(df, target_column, interval)
    report = {
        "status": "failed" if issues else "passed", "issues": issues,
        "summary": {"rows": len(df), "columns": len(df.columns), "target_column": target_column},
    }
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    (VALIDATION_DIR / "validation_report.json").write_text(json.dumps(report, indent=2))
    if issues:
        raise ValueError(f"Dataset validation failed: {issues}")
    return report
