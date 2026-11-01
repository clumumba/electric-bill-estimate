from __future__ import annotations

import json
from pathlib import Path

from .artifacts import checked_predictions, load_bundle
import pandas as pd

from .config import MODEL_DIR


def predict(input_path: str | Path, feature_schema_path: str | Path, model_path: str | Path = MODEL_DIR / "latest_model.joblib", output_path: str | Path | None = None) -> pd.DataFrame:
    input_df = pd.read_csv(input_path)
    with Path(feature_schema_path).open("r", encoding="utf-8") as file:
        schema = json.load(file)

    feature_columns = schema["feature_columns"]
    target_column = schema["target_column"]
    bundle = load_bundle(model_path, feature_schema_path)
    if bundle["schema"] != schema:
        raise ValueError("Requested schema differs from the model bundle.")
    model = bundle["model"]

    missing = [col for col in feature_columns if col not in input_df.columns]
    if missing:
        raise ValueError(f"Missing features for prediction: {missing}")

    predictions = checked_predictions(model, input_df[feature_columns])
    output = input_df.copy()
    output["prediction"] = predictions
    output["actual"] = output.get(target_column, None)

    result_path = Path(output_path) if output_path else Path(input_path).with_name("predictions.csv")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(result_path, index=False)
    return output
