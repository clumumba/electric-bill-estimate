from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PROCESSED_DIR, RAW_DATA_PATH

# Load the raw dataset from a CSV file and normalize its structure.
def load_raw_dataset(raw_path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}")

    df = pd.read_csv(path, sep=';', decimal=',', thousands='.', encoding='latin1')
    if df.empty:
        raise ValueError(f"Raw dataset at {path} is empty.")

    if df.columns[0].startswith("Unnamed"):
        df = df.rename(columns={df.columns[0]: "timestamp"})

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Source contains invalid timestamps.")
    df = df.set_index("timestamp").sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")

    if df.empty:
        raise ValueError(f"No valid timestamp rows were found in {path}.")

    return df

# Load the raw dataset and save a snapshot to the processed directory.
def ingest(raw_path: str | Path = RAW_DATA_PATH, snapshot_path: str | Path | None = None) -> pd.DataFrame:
    df = load_raw_dataset(raw_path)
    snapshot = Path(snapshot_path) if snapshot_path else PROCESSED_DIR / "raw_snapshot.csv"
    snapshot.parent.mkdir(parents=True, exist_ok=True)

    snapshot_str = str(snapshot)
    try:
        df.to_csv(snapshot_str, index=True)
    except OSError:
        temp_path = snapshot.with_suffix(f"{snapshot.suffix}.tmp")
        df.to_csv(str(temp_path), index=True)
        temp_path.replace(snapshot)

    return df
