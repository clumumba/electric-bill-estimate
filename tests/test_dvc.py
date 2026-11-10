"""Exercise the actual CLI and DVC graph in a clean, disposable checkout."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml


def test_clean_checkout_dvc_repro_and_parameter_invalidation(tmp_path):
    source = Path(__file__).resolve().parents[1]
    shutil.copytree(source / "energy_mlops", tmp_path / "energy_mlops", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("dvc.yaml", ".dvcignore", "requirements.txt"):
        shutil.copy(source / name, tmp_path / name)
    params = yaml.safe_load((source / "params.yaml").read_text())
    params["model"].update(n_estimators=3, max_depth=3, minimum_r2=-100, minimum_baseline_improvement=-100)
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(params))
    raw = tmp_path / "data/raw/LD2011_2014.txt"
    raw.parent.mkdir(parents=True)
    pd.DataFrame({"MT_001": 10 + np.sin(np.arange(1000) / 15)},
                 index=pd.date_range("2025-01-01", periods=1000, freq="15min")).to_csv(raw, sep=";", decimal=",")
    env = {**os.environ, "MLFLOW_TRACKING_URI": "sqlite:///" + (tmp_path / "mlflow.db").as_posix(),
           "DVC_NO_ANALYTICS": "true", "PYTHONDONTWRITEBYTECODE": "1"}
    # DVC's stage commands invoke 'python'; ensure the tested interpreter wins.
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")

    def run(*args):
        result = subprocess.run([sys.executable, "-m", *args], cwd=tmp_path, env=env,
                                text=True, capture_output=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    run("dvc", "init", "--no-scm")
    run("dvc", "repro")
    assert (tmp_path / "models/candidate_model.joblib").exists()
    assert not (tmp_path / "models/latest_model.joblib").exists()
    run("energy_mlops.cli", "promote")
    assert (tmp_path / "models/latest_model.joblib").exists()
    assert json.loads(run("dvc", "status", "--json")) == {}
    params["model"]["n_estimators"] += 1
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(params))
    assert "train" in json.loads(run("dvc", "status", "--json"))
    module = tmp_path / "energy_mlops/data_preprocessing.py"
    module.write_text(module.read_text() + "\n# Source changes must invalidate cached stages.\n")
    assert "preprocess" in json.loads(run("dvc", "status", "--json"))
