from __future__ import annotations

import json
import os
import tempfile
import subprocess
from importlib.metadata import version

from .artifacts import file_digest
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_DIR,
    REPORTS_DIR,
    load_params,
)


def train_model(train_path: str | Path, feature_schema_path: str | Path, tracking_uri: str = MLFLOW_TRACKING_URI) -> dict:
    train_df = pd.read_csv(train_path)
    with Path(feature_schema_path).open("r", encoding="utf-8") as file:
        schema = json.load(file)

    feature_columns = schema["feature_columns"]
    target_column = schema["target_column"]
    X_train = train_df[feature_columns]
    y_train = train_df[target_column]

    model_params = load_params().get("model", {})
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=int(model_params.get("n_estimators", 200)),
                    max_depth=int(model_params.get("max_depth", 12)),
                    random_state=int(model_params.get("random_state", 42)),
                    n_jobs=int(model_params.get("n_jobs", 2)),
                ),
            ),
        ]
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="electricity-demand-rf") as run:
        model.fit(X_train, y_train)

        metrics = {
            "train_rows": int(len(train_df)),
            "feature_count": int(len(feature_columns)),
            "target_column": target_column,
        }
        mlflow.log_params({
            "target_column": target_column,
            "feature_count": len(feature_columns),
            "n_estimators": int(model_params.get("n_estimators", 200)),
            "max_depth": int(model_params.get("max_depth", 12)),
            "train_rows": len(train_df),
        })

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / "candidate_model.joblib"
        model_version = sha256(
            json.dumps(schema, sort_keys=True).encode("utf-8")
            + run.info.run_id.encode("utf-8")
        ).hexdigest()[:12]
        try:
            revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                stderr=subprocess.DEVNULL, text=True, timeout=5,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            revision = "unavailable"
        bundle = {
            "model": model,
            "provenance": {
                "git_revision": revision,
                "training_data_sha256": file_digest(train_path),
                "source_sha256": {p.name: file_digest(p) for p in Path(__file__).parent.glob("*.py")},
                "parameters": load_params(),
                "dependencies": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib")},
            },
            "schema": schema,
            "model_version": model_version,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        fd, temporary_name = tempfile.mkstemp(dir=MODEL_DIR, suffix=".joblib.tmp")
        os.close(fd)
        try:
            joblib.dump(bundle, temporary_name)
            os.replace(temporary_name, model_path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        mlflow.log_artifact(str(model_path), artifact_path="model")

        run_id = run.info.run_id
        metrics["mlflow_run_id"] = run_id
        metrics["model_version"] = model_version

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORTS_DIR / "train_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return metrics
