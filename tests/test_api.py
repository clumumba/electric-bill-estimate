import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyRegressor

from energy_mlops.api import create_app


@pytest.fixture
def artifact(tmp_path):
    model = DummyRegressor(strategy="constant", constant=4).fit(pd.DataFrame({"hour": [0, 1]}), [4, 4])
    path = tmp_path / "model.joblib"
    joblib.dump({"model": model, "schema": {"feature_columns": ["hour"], "target_column": "load",
                 "target_unit": "kW", "forecast_horizon_steps": 1, "interval_minutes": 15},
                 "model_version": "test-v1"}, path)
    return path


@pytest.fixture
def client(artifact, monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.delenv("API_KEY_FILE", raising=False)
    with TestClient(create_app(artifact)) as client:
        yield client


HEADERS = {"X-API-Key": "test-key"}


def test_auth_health_metrics_and_prediction(client):
    assert client.get("/livez").status_code == 200
    assert client.get("/readyz").status_code == 200
    assert client.get("/v1/model/schema").status_code == 401
    assert client.get("/metrics").status_code == 401
    result = client.post("/v1/predictions", headers=HEADERS, json={"features": {"hour": 1}})
    assert result.status_code == 200
    assert result.json() == {"prediction": 4, "model_version": "test-v1"}
    assert result.headers["x-request-id"]
    metrics = client.get("/metrics", headers=HEADERS)
    assert "predictions_total 1.0" in metrics.text
    assert "http_request_duration_seconds" in metrics.text
    assert 'model_info{version="test-v1"} 1.0' in metrics.text


@pytest.mark.parametrize("payload", [
    {"features": {"wrong": 1}}, {"features": {"hour": 1, "extra": 2}},
    {"features": {"hour": "Infinity"}}, {"features": {"hour": "NaN"}},
    {"features": {"hour": 1}, "extra": 2},
])
def test_invalid_inputs(client, payload):
    assert client.post("/v1/predictions", headers=HEADERS, json=payload).status_code == 422


def test_literal_nan_returns_validation_error(client):
    response = client.post("/v1/predictions", headers={**HEADERS, "Content-Type": "application/json"},
                           content='{"features":{"hour":NaN}}')
    assert response.status_code == 422


def test_request_and_batch_limits(client):
    assert client.post("/v1/predictions", headers=HEADERS, content=b"x" * (1048576 + 1)).status_code == 413
    assert client.post("/v1/predictions:batch", headers=HEADERS,
                       json={"instances": [{"features": {"hour": 1}}] * 1001}).status_code == 422


def bill_payload():
    return {"instances": [{"features": {"hour": 0}}, {"features": {"hour": 1}}],
            "timestamps": ["2025-01-01T00:00:00+00:00", "2025-01-01T00:15:00+00:00"],
            "tariff": {"energy_rate_per_kwh": "0.18", "fixed_charge": "12", "tax_rate": "0.05", "currency": "usd"}}


def test_bill_units_rounding_and_explicit_fixed_charge(client):
    payload = bill_payload()
    result = client.post("/v1/bill-estimates", headers=HEADERS, json=payload).json()
    assert result["predicted_kwh"] == 2
    assert result["variable_charge"] == .36
    assert result["fixed_charge"] == 0
    assert result["estimated_bill"] == .38
    assert result["period_end"] == "2025-01-01T00:30:00+00:00"
    payload["include_fixed_charge"] = True
    result = client.post("/v1/bill-estimates", headers=HEADERS, json=payload).json()
    assert result["estimated_bill"] == 12.98
    assert result["currency"] == "USD"


@pytest.mark.parametrize("timestamps", [
    ["2025-01-01T00:00:00+00:00"],
    ["2025-01-01T00:00:00+00:00"] * 2,
    ["2025-01-01T00:00:00+00:00", "2025-01-01T00:30:00+00:00"],
    ["2025-01-01T00:00:00", "2025-01-01T00:15:00"],
])
def test_bill_rejects_missing_duplicate_gapped_or_naive_timestamps(client, timestamps):
    payload = bill_payload()
    payload["timestamps"] = timestamps
    assert client.post("/v1/bill-estimates", headers=HEADERS, json=payload).status_code == 422


def test_nonfinite_tariff_rejected(client):
    payload = bill_payload()
    payload["tariff"]["fixed_charge"] = "Infinity"
    assert client.post("/v1/bill-estimates", headers=HEADERS, json=payload).status_code == 422


def test_production_requires_key_and_hides_docs(artifact, monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("API_KEY_FILE", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Production requires"):
        create_app(artifact)
    monkeypatch.setenv("API_KEY", "x" * 32)
    with TestClient(create_app(artifact)) as client:
        assert client.get("/docs", headers={"X-API-Key": "x" * 32}).status_code == 404


def test_bad_model_does_not_leak_paths(client, artifact):
    artifact.write_bytes(b"corrupted")
    result = client.get("/readyz")
    assert result.status_code == 503
    assert str(artifact) not in result.text
    assert client.get("/livez").status_code == 200


def test_legacy_model_rejected_at_startup(artifact):
    joblib.dump(DummyRegressor().fit([[1]], [2]), artifact)
    with pytest.raises(RuntimeError, match="unavailable or invalid"):
        with TestClient(create_app(artifact)):
            pass


def test_replacement_model_is_loaded(client, artifact):
    bundle = joblib.load(artifact)
    bundle["model_version"] = "test-v2"
    bundle["model"] = DummyRegressor(strategy="constant", constant=8).fit(pd.DataFrame({"hour": [0]}), [8])
    replacement = artifact.with_suffix(".tmp")
    joblib.dump(bundle, replacement)
    replacement.replace(artifact)
    result = client.post("/v1/predictions", headers=HEADERS, json={"features": {"hour": 1}})
    assert result.json() == {"prediction": 8, "model_version": "test-v2"}
