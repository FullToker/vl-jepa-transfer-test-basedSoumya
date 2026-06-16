from __future__ import annotations

import os
from pathlib import Path


def setup_mlflow(env_file: str | Path = ".env/mlflow.env") -> None:
    """Load MLflow env vars and configure tracking URI."""
    env_path = Path(env_file)
    if not env_path.exists():
        raise FileNotFoundError(f"MLflow env file not found: {env_path}")

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            if "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

    import mlflow
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
