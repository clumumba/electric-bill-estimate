"""Create a synthetic model ONLY for the CI container startup test."""
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
model = DummyRegressor(strategy="constant", constant=4).fit(pd.DataFrame({"hour": [0, 1]}), [4, 4])
joblib.dump({"model": model, "model_version": "ci-synthetic-only", "schema": {
    "feature_columns": ["hour"], "target_column": "load", "interval_minutes": 15,
    "target_unit": "kW", "forecast_horizon_steps": 1,
}}, path)
