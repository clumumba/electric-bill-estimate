from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "LD2011_2014.txt"
PROCESSED_DIR = DATA_DIR / "processed"
VALIDATION_DIR = DATA_DIR / "validation"
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(BASE_DIR / "models")))
REPORTS_DIR = BASE_DIR / "reports"
MLRUNS_DIR = BASE_DIR / "mlruns"


def load_params() -> dict:
    params_path = BASE_DIR / "params.yaml"
    if not params_path.exists():
        return {}
    with params_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def get_mlflow_tracking_uri() -> str:
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri

    return "sqlite:///" + (BASE_DIR / "mlflow.db").as_posix()


def get_mlflow_experiment_name() -> str:
    env_name = os.getenv("MLFLOW_EXPERIMENT_NAME")
    if env_name:
        return env_name

    return load_params().get("mlflow", {}).get("experiment_name", "electricity-demand-forecast")


MLFLOW_TRACKING_URI = get_mlflow_tracking_uri()
MLFLOW_EXPERIMENT_NAME = get_mlflow_experiment_name()
os.environ.setdefault("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", MLFLOW_EXPERIMENT_NAME)
