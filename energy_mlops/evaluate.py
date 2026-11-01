from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .artifacts import checked_predictions, file_digest, load_bundle
from .config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MODEL_DIR, REPORTS_DIR, load_params


def evaluate_model(test_path: str | Path, feature_schema_path: str | Path,
                   model_path: str | Path = MODEL_DIR / "candidate_model.joblib",
                   tracking_uri: str = MLFLOW_TRACKING_URI) -> dict:
    # Hash before loading; a concurrent replacement cannot be promoted using this report.
    digest = file_digest(model_path)
    bundle = load_bundle(model_path, feature_schema_path)
    schema = json.loads(Path(feature_schema_path).read_text())
    if bundle["schema"] != schema:
        raise ValueError("Evaluation schema differs from the candidate model.")
    test_df = pd.read_csv(test_path)
    if len(test_df) < 2:
        raise ValueError("Evaluation requires at least two holdout rows.")
    y_test = test_df[schema["target_column"]]
    predictions = checked_predictions(bundle["model"], test_df[schema["feature_columns"]])
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions, force_finite=False)),
        "target_column": schema["target_column"], "test_rows": len(test_df),
        "model_version": bundle.get("model_version", "legacy"),
        "model_sha256": digest, "test_data_sha256": file_digest(test_path),
        "evaluation_mode": "rolling_one_step_observed_lags",
    }
    interval = schema.get("interval_minutes", 15)
    baseline_name = f"lag_{1440 // interval}"
    if 1440 % interval or baseline_name not in test_df:
        raise ValueError("Evaluation requires a one-day seasonal lag baseline.")
    baseline = test_df[baseline_name]
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline)))
    metrics["seasonal_naive_rmse"] = baseline_rmse
    metrics["seasonal_naive_mae"] = float(mean_absolute_error(y_test, baseline))
    # A perfect baseline cannot be improved on; a worse model must fail.
    improvement = ((baseline_rmse - metrics["rmse"]) / baseline_rmse
                   if baseline_rmse > 0 else (0.0 if metrics["rmse"] == 0 else -1.0))
    metrics["rmse_improvement_over_baseline"] = improvement
    metrics["rolling_windows"] = [
        {"start": str(test_df.iloc[ix[0]].get("timestamp", ix[0])),
         "end": str(test_df.iloc[ix[-1]].get("timestamp", ix[-1])),
         "rmse": float(np.sqrt(mean_squared_error(y_test.iloc[ix], predictions[ix]))),
         "mae": float(mean_absolute_error(y_test.iloc[ix], predictions[ix]))}
        for ix in np.array_split(np.arange(len(test_df)), 4) if len(ix)
    ]
    params = load_params().get("model", {})
    finite = all(np.isfinite(metrics[key]) for key in ("rmse", "mae", "r2", "rmse_improvement_over_baseline"))
    passed = (finite and metrics["r2"] >= float(params.get("minimum_r2", 0.7))
              and improvement >= float(params.get("minimum_baseline_improvement", 0.0)))
    metrics["status"] = "passed" if passed else "failed"
    if not finite:
        metrics["r2"] = None
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    if not passed:
        raise ValueError("Model quality thresholds not met; serving model remains unchanged.")
    if file_digest(model_path) != digest:
        raise ValueError("Candidate changed during evaluation; retry with serialized release jobs.")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="evaluation", nested=True):
        mlflow.log_metrics({key: metrics[key] for key in ("rmse", "mae", "r2")})
        mlflow.set_tag("model_version", metrics["model_version"])
        mlflow.log_artifact(str(REPORTS_DIR / "evaluation_metrics.json"))
    return metrics
