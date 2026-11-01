import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from energy_mlops import data_preprocessing as prep, data_validation as validation
from energy_mlops import evaluate as evaluation, train as training
from energy_mlops.artifacts import file_digest, promote_model, rollback_model
from energy_mlops.predict import predict


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    models = tmp_path / "models"
    reports = tmp_path / "reports"
    params = {"target_column": "customer", "features": {"lags": [1, 96], "time": ["hour"]},
              "model": {"test_size": .2, "n_estimators": 5, "max_depth": 3,
                        "minimum_r2": -100, "minimum_baseline_improvement": -100},
              "billing": {"interval_minutes": 15}}
    for module in (prep, validation, evaluation, training):
        monkeypatch.setattr(module, "load_params", lambda: params)
    monkeypatch.setattr(prep, "PROCESSED_DIR", processed)
    monkeypatch.setattr(validation, "VALIDATION_DIR", reports)
    monkeypatch.setattr(training, "MODEL_DIR", models)
    monkeypatch.setattr(training, "REPORTS_DIR", reports)
    monkeypatch.setattr(evaluation, "REPORTS_DIR", reports)
    frame = pd.DataFrame({"customer": 10 + np.sin(np.arange(600) / 15), "other": np.ones(600)},
                         index=pd.date_range("2025-01-01", periods=600, freq="15min", name="timestamp"))
    return processed, models, reports, params, frame


def test_snapshot_chronology_and_no_customer_leakage(workspace, tmp_path):
    processed, _, _, _, frame = workspace
    snapshot = tmp_path / "snapshot.csv"
    frame.to_csv(snapshot)
    train, test, columns, target = prep.prepare_data(snapshot)
    assert target == "customer"
    assert "other" not in columns
    assert train.timestamp.max() < test.timestamp.min()
    assert test.iloc[0].lag_1 == pytest.approx(frame.loc[test.iloc[0].timestamp - pd.Timedelta(minutes=15), target])
    assert json.loads((processed / "feature_schema.json").read_text())["target_unit"] == "kW"


@pytest.mark.parametrize("fault", ["gap", "duplicate", "infinity", "negative", "unsorted"])
def test_invalid_readings_fail(workspace, fault):
    *_, frame = workspace
    if fault == "gap": frame = frame.drop(frame.index[3])
    if fault == "duplicate": frame = pd.concat([frame, frame.iloc[:1]])
    if fault == "infinity": frame.iloc[0, 0] = np.inf
    if fault == "negative": frame.iloc[0, 0] = -1
    if fault == "unsorted": frame = frame.iloc[::-1]
    with pytest.raises(ValueError): prep.prepare_data(frame)


def test_train_evaluate_promote_predict_and_rollback(workspace, tmp_path):
    processed, models, reports, params, frame = workspace
    prep.prepare_data(frame)
    schema = processed / "feature_schema.json"
    uri = "sqlite:///" + (tmp_path / "tracking.db").as_posix()
    training.train_model(processed / "train.csv", schema, tracking_uri=uri)
    candidate = models / "candidate_model.joblib"
    serving = models / "latest_model.joblib"
    assert candidate.exists() and not serving.exists()
    evaluation.evaluate_model(processed / "test.csv", schema, candidate, tracking_uri=uri)
    report = reports / "evaluation_metrics.json"
    promote_model(candidate, report, serving)
    first = serving.read_bytes()
    result = predict(processed / "test.csv", schema, serving, tmp_path / "predictions.csv")
    assert np.isfinite(result.prediction).all()
    params["model"]["minimum_r2"] = 2
    training.train_model(processed / "train.csv", schema, tracking_uri=uri)
    with pytest.raises(ValueError, match="quality"):
        evaluation.evaluate_model(processed / "test.csv", schema, candidate, tracking_uri=uri)
    with pytest.raises(ValueError, match="passing evaluation"):
        promote_model(candidate, report, serving)
    assert serving.read_bytes() == first
    params["model"]["minimum_r2"] = -100
    evaluation.evaluate_model(processed / "test.csv", schema, candidate, tracking_uri=uri)
    promote_model(candidate, report, serving)
    assert serving.read_bytes() != first
    assert promote_model(candidate, report, serving)["status"] == "already_active"
    assert (models / "previous_model.joblib").read_bytes() == first
    rollback_model(serving)
    assert serving.read_bytes() == first


def test_promotion_rejects_changed_candidate(tmp_path):
    candidate = tmp_path / "candidate.joblib"
    schema = {"feature_columns": ["hour"], "target_column": "load"}
    model = DummyRegressor().fit(pd.DataFrame({"hour": [1, 2]}), [2, 3])
    joblib.dump({"model": model, "schema": schema, "model_version": "one"}, candidate)
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"status": "passed", "model_version": "one", "model_sha256": file_digest(candidate)}))
    joblib.dump({"model": model, "schema": schema, "model_version": "two"}, candidate)
    with pytest.raises(ValueError): promote_model(candidate, report, tmp_path / "latest.joblib")
    assert not (tmp_path / "latest.joblib").exists()


def test_evaluation_requires_seasonal_baseline(workspace, tmp_path):
    processed, models, reports, params, frame = workspace
    params["features"]["lags"] = [1]
    prep.prepare_data(frame)
    uri = "sqlite:///" + (tmp_path / "tracking.db").as_posix()
    schema = processed / "feature_schema.json"
    training.train_model(processed / "train.csv", schema, tracking_uri=uri)
    with pytest.raises(ValueError, match="seasonal lag"):
        evaluation.evaluate_model(processed / "test.csv", schema, models / "candidate_model.joblib", tracking_uri=uri)
    assert not (models / "latest_model.joblib").exists()


def test_nonfinite_predictions_rejected():
    from energy_mlops.artifacts import checked_predictions
    class BrokenModel:
        def predict(self, frame): return [np.nan]
    with pytest.raises(ValueError, match="invalid predictions"):
        checked_predictions(BrokenModel(), pd.DataFrame({"hour": [1]}))
