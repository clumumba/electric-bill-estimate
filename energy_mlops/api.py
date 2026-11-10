"""Authenticated, bounded inference for rolling one-step electricity forecasts."""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.exceptions import RequestValidationError
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator

from .artifacts import checked_predictions, load_bundle
from .config import MODEL_DIR, load_params
from .operations import Metrics, RequestControls

logger = logging.getLogger("uvicorn.error")


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, FiniteFloat] = Field(..., min_length=1, max_length=256)


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instances: list[PredictionRequest] = Field(..., min_length=1, max_length=1000)


class Tariff(BaseModel):
    model_config = ConfigDict(extra="forbid")
    energy_rate_per_kwh: Decimal = Field(..., ge=0, le=10000, allow_inf_nan=False)
    fixed_charge: Decimal = Field(Decimal("0"), ge=0, le=1000000, allow_inf_nan=False)
    tax_rate: Decimal = Field(Decimal("0"), ge=0, le=1, allow_inf_nan=False)
    currency: str = Field("USD", pattern=r"^[A-Za-z]{3}$")


class BillEstimateRequest(BatchPredictionRequest):
    tariff: Tariff | None = None
    timestamps: list[datetime] = Field(..., min_length=1, max_length=1000)
    include_fixed_charge: bool = False

    @field_validator("timestamps")
    @classmethod
    def require_timezone(cls, values):
        if any(value.utcoffset() is None for value in values):
            raise ValueError("Billing timestamps must include a timezone offset.")
        return values


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    feature_columns: list[str]
    target_column: str
    signature: tuple[int, int, int]
    model_version: str
    interval_minutes: int
    target_unit: str


class ModelStore:
    """Validate each replacement completely before swapping the in-memory bundle."""
    def __init__(self, model_path: Path, metrics: Metrics):
        self.model_path = model_path
        self.metrics = metrics
        self._bundle: ModelBundle | None = None
        self._lock = Lock()

    def get(self) -> ModelBundle:
        with self._lock:
            try:
                stat = self.model_path.stat()
                signature = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
                if self._bundle is not None and self._bundle.signature == signature:
                    return self._bundle
                artifact = load_bundle(self.model_path)
                schema = artifact["schema"]
                if schema.get("target_unit") != "kW" or schema.get("forecast_horizon_steps") != 1:
                    raise ValueError("Serving requires a current one-step kW model bundle; retrain and evaluate first.")
                model = artifact["model"]
                # Bound tree parallelism under concurrent HTTP requests.
                if hasattr(model, "get_params"):
                    options = {name: 1 for name in model.get_params() if name == "n_jobs" or name.endswith("__n_jobs")}
                    model.set_params(**options)
                bundle = ModelBundle(model, schema["feature_columns"], schema["target_column"],
                                     signature, artifact["model_version"], schema["interval_minutes"], schema["target_unit"])
            except Exception as exc:
                # Do not disclose filesystem paths or serializer internals to callers.
                logger.exception("Model artifact could not be loaded")
                raise RuntimeError("Model is unavailable or invalid.") from exc
            self._bundle = bundle
            self.metrics.model.clear()
            self.metrics.model.labels(bundle.model_version).set(1)
            return bundle


def _validated_frame(records, bundle):
    expected = set(bundle.feature_columns)
    for index, record in enumerate(records):
        missing, unknown = sorted(expected - record.keys()), sorted(record.keys() - expected)
        if missing or unknown:
            problems = []
            if missing:
                problems.append(f"Add the required feature(s): {', '.join(missing)}.")
            if unknown:
                problems.append(f"Remove unrecognized feature(s): {', '.join(unknown)}.")
            raise HTTPException(422, {
                "message": f"Instance {index + 1}: {' '.join(problems)}",
                "instance": index,
                "missing_features": missing,
                "unknown_features": unknown,
                "hint": "Use GET /v1/model/schema to see the exact feature names required for every instance.",
            })
    return pd.DataFrame(records, columns=bundle.feature_columns)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_store.get()
    yield


def create_app(model_path: Path | None = None) -> FastAPI:
    environment = os.getenv("APP_ENV", "development")
    if environment not in {"development", "test", "production"}:
        raise RuntimeError("APP_ENV must be development, test, or production.")
    production = environment == "production"
    key_file = os.getenv("API_KEY_FILE")
    api_key = Path(key_file).read_text().strip() if key_file else os.getenv("API_KEY", "")
    if production and len(api_key) < 32:
        raise RuntimeError("Production requires an API_KEY or API_KEY_FILE containing at least 32 characters.")
    body_limit = int(os.getenv("MAX_BODY_BYTES", "1048576"))
    concurrency = int(os.getenv("MAX_CONCURRENT_REQUESTS", "8"))
    if body_limit <= 0 or concurrency <= 0:
        raise RuntimeError("Request limits must be positive.")
    defaults = load_params().get("billing", {})
    default_tariff = Tariff(energy_rate_per_kwh=defaults.get("energy_rate_per_kwh", .18),
                            fixed_charge=defaults.get("fixed_charge", 0), tax_rate=defaults.get("tax_rate", 0),
                            currency=defaults.get("currency", "USD"))
    app = FastAPI(title="Electricity Demand Forecast API", version="2.0.0", lifespan=lifespan,
                  docs_url=None if production else "/docs", redoc_url=None,
                  openapi_url=None if production else "/openapi.json")
    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        errors = [{key: error[key] for key in ("loc", "msg", "type")} for error in exc.errors()]
        tips = []
        for error in errors:
            loc = error["loc"]
            field = str(loc[-1]) if loc else "request"
            if field == "features":
                tips.append("Each instance needs a 'features' object with numeric values.")
            elif field == "timestamps":
                tips.append("Provide one timezone-aware timestamp per instance, in order, at the model's interval cadence.")
            elif field == "instances":
                tips.append("Provide an 'instances' array containing one feature object per forecast interval.")
            elif field in {"energy_rate_per_kwh", "fixed_charge", "tax_rate", "currency"}:
                tips.append("Check the tariff fields: rates and charges must be nonnegative, tax_rate must be 0–1, and currency must be a three-letter code.")
        return JSONResponse(status_code=422, content={
            "message": "The request could not be used. Check the field errors and correct the request body.",
            "errors": errors,
            "hints": list(dict.fromkeys(tips)),
        })

    metrics = Metrics()
    app.state.metrics = metrics
    app.state.model_store = ModelStore(model_path or MODEL_DIR / "latest_model.joblib", metrics)
    app.add_middleware(RequestControls, api_key=api_key, max_body_bytes=body_limit,
                       max_concurrency=concurrency, metrics=metrics)

    def bundle_for(request):
        try:
            return request.app.state.model_store.get()
        except RuntimeError as exc:
            raise HTTPException(503, "Model is unavailable or invalid.") from exc

    def infer(records, bundle):
        frame = _validated_frame(records, bundle)
        try:
            values = checked_predictions(bundle.model, frame)
        except Exception as exc:
            logger.exception("Prediction failed for model %s", bundle.model_version)
            raise HTTPException(503, "Prediction is temporarily unavailable.") from exc
        metrics.predictions.inc(len(values))
        return values.tolist()

    @app.get("/livez", tags=["operations"])
    def live():
        return {"status": "alive"}

    @app.get("/healthz", tags=["operations"])
    @app.get("/readyz", tags=["operations"])
    def ready(request: Request):
        bundle_for(request)
        return {"status": "ready"}

    @app.get("/metrics", tags=["operations"])
    def prometheus_metrics():
        return Response(generate_latest(metrics.registry), headers={"Content-Type": CONTENT_TYPE_LATEST})

    @app.get("/v1/model/schema", tags=["model"])
    def schema(request: Request):
        bundle = bundle_for(request)
        return {"target_column": bundle.target_column, "feature_columns": bundle.feature_columns,
                "model_version": bundle.model_version, "interval_minutes": bundle.interval_minutes,
                "target_unit": bundle.target_unit, "forecast_horizon_steps": 1}

    @app.post("/v1/predictions", tags=["predictions"])
    def predict(payload: PredictionRequest, request: Request):
        bundle = bundle_for(request)
        return {"prediction": infer([payload.features], bundle)[0], "model_version": bundle.model_version}

    @app.post("/v1/predictions:batch", tags=["predictions"])
    def predict_batch(payload: BatchPredictionRequest, request: Request):
        bundle = bundle_for(request)
        return {"predictions": infer([item.features for item in payload.instances], bundle),
                "model_version": bundle.model_version}

    @app.post("/v1/bill-estimates", tags=["billing"])
    def estimate_bill(payload: BillEstimateRequest, request: Request):
        bundle = bundle_for(request)
        timestamps = payload.timestamps
        if len(timestamps) != len(payload.instances) or any(
                later - earlier != timedelta(minutes=bundle.interval_minutes)
                for earlier, later in zip(timestamps, timestamps[1:])):
            raise HTTPException(422, {
                "message": "The timestamps do not match the forecast intervals.",
                "hint": f"Provide exactly {len(payload.instances)} timezone-aware timestamps, one per instance, ordered {bundle.interval_minutes} minutes apart.",
                "instances": len(payload.instances),
                "timestamps": len(timestamps),
                "interval_minutes": bundle.interval_minutes,
            })
        tariff = payload.tariff or default_tariff
        predictions = infer([item.features for item in payload.instances], bundle)
        kwh = sum((Decimal(str(value)) for value in predictions), Decimal("0")) * Decimal(bundle.interval_minutes) / 60
        variable = (kwh * tariff.energy_rate_per_kwh).quantize(Decimal(".01"), rounding=ROUND_HALF_UP)
        fixed = (tariff.fixed_charge if payload.include_fixed_charge else Decimal("0")).quantize(Decimal(".01"), rounding=ROUND_HALF_UP)
        tax = ((variable + fixed) * tariff.tax_rate).quantize(Decimal(".01"), rounding=ROUND_HALF_UP)
        return {"predicted_kwh": float(kwh.quantize(Decimal(".0001"), rounding=ROUND_HALF_UP)),
                "variable_charge": float(variable), "fixed_charge": float(fixed), "tax": float(tax),
                "estimated_bill": float(variable + fixed + tax), "currency": tariff.currency.upper(),
                "model_version": bundle.model_version, "period_start": timestamps[0].isoformat(),
                "period_end": (timestamps[-1] + timedelta(minutes=bundle.interval_minutes)).isoformat()}

    return app


app = create_app()
