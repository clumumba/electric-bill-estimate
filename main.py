from __future__ import annotations

import argparse

from energy_mlops.config import MODEL_DIR, REPORTS_DIR, PROCESSED_DIR, RAW_DATA_PATH
from energy_mlops.artifacts import promote_model
from energy_mlops.data_ingestion import ingest
from energy_mlops.data_preprocessing import prepare_data
from energy_mlops.data_validation import validate_dataset
from energy_mlops.evaluate import evaluate_model
from energy_mlops.train import train_model


def run_pipeline() -> dict:
    raw_df = ingest(RAW_DATA_PATH)
    validate_dataset(raw_df)
    _, _, _, _ = prepare_data(raw_df)
    train_result = train_model(PROCESSED_DIR / "train.csv", PROCESSED_DIR / "feature_schema.json")
    evaluate_result = evaluate_model(PROCESSED_DIR / "test.csv", PROCESSED_DIR / "feature_schema.json")
    promotion = promote_model(MODEL_DIR / "candidate_model.joblib",
                              REPORTS_DIR / "evaluation_metrics.json",
                              MODEL_DIR / "latest_model.joblib")
    return {"train": train_result, "evaluation": evaluate_result, "promotion": promotion}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and run the electricity demand MLOps pipeline.")
    parser.add_argument("--run-pipeline", action="store_true", help="Run the full training pipeline.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.run_pipeline:
        print(run_pipeline())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
