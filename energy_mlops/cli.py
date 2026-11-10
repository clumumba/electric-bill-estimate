# Command-line interface for the electricity demand forecasting MLOps pipeline.
# Provides a command-line interface for ingesting, validating, preprocessing, training, evaluating, promoting, and rolling back models.
# by running python cli.py <command>(--ingest, --validate, --preprocess, --train, --evaluate, --promote, --rollback)
from __future__ import annotations

import argparse

import pandas as pd

from .config import MODEL_DIR, REPORTS_DIR, PROCESSED_DIR, RAW_DATA_PATH
from .artifacts import promote_model, rollback_model
from .data_ingestion import ingest
from .data_preprocessing import prepare_data
from .data_validation import validate_dataset
from .evaluate import evaluate_model
from .train import train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MLOps pipeline for electricity demand forecasting.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Load and persist the upstream dataset.")
    ingest_parser.set_defaults(func=lambda: ingest(RAW_DATA_PATH))

    validate_parser = subparsers.add_parser("validate", help="Validate the ingested dataset.")
    validate_parser.set_defaults(func=lambda: validate_dataset(pd.read_csv(
        PROCESSED_DIR / "raw_snapshot.csv", index_col="timestamp", parse_dates=True
    )))

    preprocess_parser = subparsers.add_parser("preprocess", help="Create train/test splits and feature schemas.")
    preprocess_parser.set_defaults(func=lambda: prepare_data(PROCESSED_DIR / "raw_snapshot.csv"))

    train_parser = subparsers.add_parser("train", help="Train the forecasting model and log to MLflow.")
    train_parser.set_defaults(func=lambda: train_model(PROCESSED_DIR / "train.csv", PROCESSED_DIR / "feature_schema.json"))

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the trained model on holdout data.")
    evaluate_parser.set_defaults(func=lambda: evaluate_model(PROCESSED_DIR / "test.csv", PROCESSED_DIR / "feature_schema.json"))

    promote = subparsers.add_parser("promote", help="Publish the evaluated candidate.")
    promote.set_defaults(func=lambda: promote_model(
        MODEL_DIR / "candidate_model.joblib", REPORTS_DIR / "evaluation_metrics.json",
        MODEL_DIR / "latest_model.joblib"))
    rollback = subparsers.add_parser("rollback", help="Restore the previous serving model.")
    rollback.set_defaults(func=lambda: rollback_model(MODEL_DIR / "latest_model.joblib"))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = args.func()
    if result is not None:
        print(result)


if __name__ == "__main__":
    main()
