# Electricity demand forecasting

A rolling **one-step (15-minute)** demand model with DVC training, MLflow experiment
tracking, explicit model promotion, and a FastAPI service. Readings are assumed to
be **kW**; interval energy is `kW × interval_minutes / 60`. Verify source units before
using a different dataset. This is not a multi-day or monthly forecasting model.
The included source covers 2011–2014; historical holdout results do not establish
accuracy on current production demand. Validate recent representative data before launch.

## 1. Set up the Linux environment

Use Linux with Python 3.12 and its `venv` module installed.
From the outer `elec-bill/` directory, enter the application directory. If you
are already beside `main.py`, skip the first command. Run the remaining examples
from this directory.

```bash
cd elec
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python main.py --help
```

The single `requirements.txt` install includes serving, DVC training, MLflow,
and test dependencies. The same file is used by Docker and CI.
The help command loads the training imports without starting a training run. To
run the project's tests, use `python -m pytest -q`. In a new terminal, return to
this directory and run `source .venv/bin/activate` again.

`requirements.txt` is a unified exact-version list for serving, development,
training, and tests. It also supplies the dependencies tracked by the DVC training
stage. There are no separate serving, training, or development requirements files.

If `import yaml` fails, the package name is **PyYAML**. Check which environment
is running the app with `python -c 'import sys; print(sys.executable)'`. It should
point into `.venv/bin/`. If the install failed, resolve the first pip error and
rerun the install before trying `main.py` again.

## 2. Train, evaluate, and promote

Provide `data/raw/LD2011_2014.txt` from your authorized data source or configure a
DVC remote and use `dvc pull`. No shared data remote is configured in this checkout.
The source uses semicolons and decimal commas; ingested snapshots use standard CSV.
Invalid timestamps, non-finite/negative readings, duplicate intervals, and cadence
gaps fail validation. Missing intervals must be investigated and repaired upstream.

MLflow defaults to a local SQLite database (`mlflow.db`). Set
`MLFLOW_TRACKING_URI` only if you want to use another tracking store.

```bash
dvc repro
cat reports/evaluation_metrics.json
```

Stages run in order: ingestion → validation → preprocessing → candidate training
→ evaluation. DVC tracks code, parameters, and data dependencies. `dvc repro`
does **not** change the serving model. Existing `dvc.lock` and generated artifacts
in older checkouts must be refreshed by this command; do not trust old metrics as
validation of the new code.

With the defaults in `params.yaml`, preprocessing turns the calendar and earlier
readings of meter `MT_001` into inputs. For 15-minute readings, `lag_1` is the
previous interval and `lag_96` is the same time yesterday. A 200-tree random
forest trains on the oldest 80% of usable rows; evaluation uses the newest 20%.
The default gates require R² of at least 0.7 and RMSE no worse than using yesterday's
reading. See `energy_mlops/data_preprocessing.py`, `train.py`, and `evaluate.py`
in the same package for the implementation.

During evaluation each prediction uses the actual readings available before that
interval. The report also breaks errors down across four time windows. These
results measure one-interval forecasts; they do not establish month-ahead accuracy.

```bash
python -m energy_mlops.cli promote
```

Only if you later need to restore a previous published model:

```bash
python -m energy_mlops.cli rollback
```

Promotion verifies the candidate version and SHA256 against its passing evaluation
report, saves `previous_model.joblib`, then atomically replaces `latest_model.joblib`.
Serialize release jobs: concurrent trainers/promoters are not supported. Restrict
write access to the model directory and evaluation reports. Joblib deserialization
executes code; artifacts must come from trusted training infrastructure.

`python main.py --run-pipeline` runs the same steps and promotes on success. Use
`dvc repro` followed by explicit promotion when reviewing candidates before release.
Artifacts record training-data and source hashes, parameters, dependency versions,
and Git revision where available. Keep the approved artifacts and reports in backed-up
versioned storage. Only one previous version is retained in the local model directory. Re-promoting
the active candidate is a no-op and preserves the rollback backup. Legacy artifacts
from before this migration need their original application/schema to restore; the
rollback command supports current bundled releases.

## 3. Serve the approved model

Old standalone models or bundles without units/horizon metadata are rejected by
the API. Retrain and evaluate first. Do not copy stale artifacts into a release.

```bash
export API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export APP_ENV=production
```

Choose one way to start the API. To run in the active Python environment:

```bash
python -m uvicorn energy_mlops.api:app --host 127.0.0.1 --port 8000 --workers 1
```

Alternatively, with Docker Engine and the Compose plugin installed, use the same
exported key and start the container. The container's user (UID 10001) must be
able to read the promoted file at `models/latest_model.joblib` on the bind mount.

```bash
docker compose up --build -d
```

Production requires a key of at least 32 characters. `API_KEY_FILE` may point to a
mounted secret and takes precedence over `API_KEY`. Never commit keys. Compose
publishes only to loopback, runs as a non-root user with a read-only filesystem,
and mounts approved models read-only. The image installs the unified
`requirements.txt`, so it includes training and test packages as well as serving
packages. It contains the application modules and parameters, and excludes datasets,
`.env`, caches, and models.
Set `MODEL_DIR` when serving artifacts outside the default `models/` directory.

Every endpoint except health probes requires `X-API-Key` when a key is configured.
Interactive documentation is disabled in production. Development without a key is
intended for loopback-only use.

- `GET /livez`: process responds.
- `GET /readyz` and `/healthz`: a valid model is available.
- `GET /v1/model/schema`: feature order, model version, units, cadence, and horizon.
- `POST /v1/predictions`: `{"features": {"feature_name": 1.0, ...}}`.
- `POST /v1/predictions:batch`: `{"instances": [{"features": {...}}, ...]}`.
- `POST /v1/bill-estimates`: estimate energy and charges for consecutive intervals.
- `GET /metrics`: Prometheus request counts, latency histograms, successful prediction
  counts, and active model version. Protect the scrape credential like an API key.

For a local Uvicorn run, open `http://127.0.0.1:8000/docs`, expand `POST /v1/bill-estimates`,
and choose **Try it out**. The example below matches this project's current schema;
check `GET /v1/model/schema` after replacing the model. Each instance must include
every feature name exactly once, with a numeric value, and no extra keys. If an API
key is configured, add `X-API-Key` in the docs' **Authorize** control.

Callers must construct calendar features and lags using the same formulas and
observed history as training. Feature construction from raw history is not yet an
API endpoint. A 422 response includes a message and hints; feature errors list the
missing and unrecognized names, while timestamp errors report the required interval
cadence. Use those details to correct the submitted JSON and try again.
Requests are limited to 1 MiB, 1,000 instances, and 256 features per instance; body
limits also apply to chunked requests. `MAX_BODY_BYTES` and
`MAX_CONCURRENT_REQUESTS` (default 8) configure per-process limits. Scale using one
worker per container so metrics and limits remain per-replica. Put TLS termination,
per-client rate limits, and header/body timeouts at your ingress before public access.

## 4. Billing estimates and operations

Billing requires one timezone-aware timestamp per predicted interval. Timestamps
represent interval starts and must be ordered, unique, and contiguous. This is a
**breaking change** for callers that previously supplied only `instances`.

```json
{
  "instances": [
    {
      "features": {
        "hour": 0,
        "day_of_week": 3,
        "day_of_month": 1,
        "month": 1,
        "is_weekend": 0,
        "hour_sin": 0,
        "hour_cos": 1,
        "month_sin": 0.5,
        "month_cos": 0.8660254,
        "lag_1": 0,
        "lag_2": 0,
        "lag_3": 0,
        "lag_4": 0,
        "lag_8": 0,
        "lag_12": 0,
        "lag_24": 0,
        "lag_48": 0,
        "lag_96": 0,
        "lag_192": 0,
        "lag_672": 0
      }
    }
  ],
  "timestamps": ["2026-01-01T00:00:00+00:00"],
  "tariff": {
    "energy_rate_per_kwh": "0.18",
    "fixed_charge": "12.00",
    "tax_rate": "0.05",
    "currency": "USD"
  },
  "include_fixed_charge": true
}
```

The calendar values correspond to midnight on January 1, 2026 (Thursday;
`day_of_week` uses Monday=0). The lag values are zero only to demonstrate the JSON
shape; replace them with observed target readings in kW. `lag_1` is one 15-minute
interval before the timestamp, `lag_4` is one hour before, and `lag_96` is one day
before. Construct all calendar and lag features with the same formulas and history
used during training. The API does not derive these features from raw usage. This
example estimates one 15-minute interval, not a monthly bill. Charges use
Decimal arithmetic and round half-up to two decimal places; tax applies to the
rounded variable and included fixed charges. Fixed charges are excluded unless
explicitly requested, to avoid silently charging them for every batch. This endpoint
does not track invoices, deduplicate requests across batches, or validate utility
pricing rules. It supports a simple flat tariff with two-decimal currency amounts.
A batch of one-step forecasts is not a validated monthly forecast: future lag
values must not be filled with observations unavailable at forecast time.

Before public deployment:

1. Regenerate real-data artifacts and review the quality gates. Benchmark the actual
   model under representative concurrency and set resource limits from measurements.
2. Configure TLS ingress, credential rotation, client rate limits, and slow-request
   timeouts. Keep MLflow private; run it separately from the serving application.
3. Scrape each replica's metrics; alert on readiness failures, sustained 5xx rates,
   latency, and resource saturation. Request logs include a generated request ID,
   normalized route, status, and duration, without feature values or credentials.
4. Monitor production feature drift and delayed-label forecast error; establish
   retraining triggers. This repository exposes service metrics, not a drift service.
5. Back up approved artifacts, DVC data, and MLflow metadata. Test rollback and
   restore in staging. A corrupt replacement makes readiness fail until repaired
   or rolled back, while liveness remains independent.

CI runs regression tests, a synthetic clean-checkout DVC pipeline, dependency
consistency checks, and a Docker build/startup smoke test. The synthetic model is
only a test fixture, never a release candidate. CI runs when this application
repository is pushed to a GitHub repository with Actions enabled.

