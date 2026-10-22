"""MLflow set-up shared by the training examples: a local tracking store and model registry.

Author: Yves Schillings.

MLflow keeps the information: runs, parameters, metrics, curves, model files and
numbered model versions. It does not decide what goes live. The release step
(19_model_release.py) checks the gate, a named owner decides, and the service
(18_model_serving.py) loads only the version that decision names.

By default everything stays on this machine:
  * runs, metrics and the model registry in mlflow/mlflow.db (SQLite),
  * model files and charts in mlflow/artifacts/.
Set MLFLOW_TRACKING_URI to use a tracking server instead (see deploy/one-server).

Browse and compare the runs:
    mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
"""
import os
from importlib import metadata
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from fictional_data import ROOT

STORE = ROOT / "mlflow"
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")   # keeps run logs readable
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")              # no usage data leaves the machine


def tracking_uri(store=STORE):
    return os.environ.get("MLFLOW_TRACKING_URI") or "sqlite:///" + (Path(store) / "mlflow.db").as_posix()


def use_experiment(name, store=STORE):
    """Point MLflow at the store and select the experiment, creating it on first use."""
    uri = tracking_uri(store)
    if uri.startswith("sqlite:"):
        Path(store).mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        local = (Path(store) / "artifacts").as_uri() if uri.startswith("sqlite:") else None
        experiment_id = mlflow.create_experiment(name, artifact_location=local)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return MlflowClient(uri)


def pinned(*packages):
    """Exact versions of the packages a saved model needs, written next to the model by MLflow."""
    return [f"{p}=={metadata.version(p)}" for p in packages]
