"""Shared loading and atomic publication of trusted, local model artifacts."""
from __future__ import annotations

import json
import os
import tempfile
from hashlib import sha256
from pathlib import Path

import joblib
import numpy as np


def file_digest(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bundle(path: str | Path, schema_path: str | Path | None = None) -> dict:
    # Joblib is executable serialization. Only load artifacts from trusted storage.
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        if schema_path is None:
            raise ValueError("Legacy models require an explicit feature schema.")
        artifact = {"model": artifact, "schema": json.loads(Path(schema_path).read_text()),
                    "model_version": "legacy"}
    schema = artifact.get("schema", {})
    columns = schema.get("feature_columns")
    if (not isinstance(columns, list) or not columns
            or not all(isinstance(name, str) for name in columns)
            or len(set(columns)) != len(columns)
            or not isinstance(schema.get("target_column"), str)
            or schema["target_column"] in columns
            or not callable(getattr(artifact.get("model"), "predict", None))):
        raise ValueError("Invalid model bundle or feature schema.")
    interval = schema.get("interval_minutes", 15)
    if not isinstance(interval, int) or isinstance(interval, bool) or interval <= 0:
        raise ValueError("interval_minutes must be a positive integer.")
    learned = getattr(artifact["model"], "feature_names_in_", None)
    if learned is not None and list(learned) != columns:
        raise ValueError("Model and feature schema do not match.")
    return artifact


def checked_predictions(model, frame) -> np.ndarray:
    values = np.asarray(model.predict(frame), dtype=float)
    if values.shape != (len(frame),) or not np.isfinite(values).all():
        raise ValueError("Model returned invalid predictions.")
    return np.maximum(values, 0)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=destination.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as output, source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        os.replace(name, destination)
    finally:
        Path(name).unlink(missing_ok=True)


def promote_model(candidate_path: str | Path, report_path: str | Path,
                  destination: str | Path) -> dict:
    """Publish only the exact candidate whose evaluation passed; keep rollback bytes.

    Release jobs must be serialized by the deployment system.
    """
    candidate, destination = Path(candidate_path), Path(destination)
    report = json.loads(Path(report_path).read_text())
    # Snapshot first so evaluation checks and publication refer to the same bytes.
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
        snapshot = Path(directory) / "candidate.joblib"
        atomic_copy(candidate, snapshot)
        bundle = load_bundle(snapshot)
        if (report.get("status") != "passed"
                or report.get("model_sha256") != file_digest(snapshot)
                or report.get("model_version") != bundle.get("model_version")):
            raise ValueError("Promotion requires a passing evaluation of this exact candidate.")
        if destination.exists():
            if file_digest(destination) == file_digest(snapshot):
                return {"model_version": bundle["model_version"], "status": "already_active"}
            atomic_copy(destination, destination.with_name("previous_model.joblib"))
        atomic_copy(snapshot, destination)
    return {"model_version": bundle["model_version"], "status": "promoted"}


def rollback_model(destination: str | Path) -> dict:
    destination = Path(destination)
    previous = destination.with_name("previous_model.joblib")
    bundle = load_bundle(previous)
    atomic_copy(previous, destination)
    return {"model_version": bundle.get("model_version", "unknown"), "status": "rolled_back"}
