"""Serve the approved document-triage model with FastAPI.

Author: Yves Schillings.

Run, in this order:
    python examples/01_machine_learning.py   trains and registers model versions in MLflow
    python examples/19_model_release.py      the gate and a named owner decide
    python examples/18_model_serving.py      serves the approved version on 127.0.0.1:8800

The service never decides what goes live. At start-up it:
  1. reads the approval record (release_records.py), kept apart from the model files,
  2. takes the live version from it, or refuses to start: pending, denied and
     withdrawn versions are never served,
  3. refuses a version whose recorded test score is below the agreed gate,
  4. downloads that version from the MLflow registry and compares the digest of
     all its model files with the digest in the approval record,
  5. loads the model only when the digests match.
A model file replaced together with its MLflow metadata is still refused, because the
reference digest lives in the approval record, not next to the model.

Rollback: the release step withdraws the live version (19_model_release.py).
At its next start the service serves the previous approved version.
"""
import tempfile

import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fictional_data import sha256_dir
from mlflow_tracking import STORE, tracking_uri
from release_records import live_entry, read_record

MODEL_NAME = "document-triage"


class Document(BaseModel):
    text: str = Field(min_length=1, max_length=20000)


class RefusedModel(RuntimeError):
    """The service refuses to load this model."""


def load_approved(model_name=MODEL_NAME, approvals=None, store=STORE):
    """Load the live version named by the approval record, after every check has passed."""
    record = read_record(model_name, approvals)
    entry = live_entry(record)
    if entry is None:
        raise RefusedModel(f"No approved version of {model_name}. Pending, denied and withdrawn versions "
                           "are never served.")
    if entry["test_macro_f1"] < record["acceptance"]:
        raise RefusedModel(f"Version {entry['version']} is below the gate {record['acceptance']}.")
    mlflow.set_tracking_uri(tracking_uri(store))
    with tempfile.TemporaryDirectory() as tmp:
        local = mlflow.artifacts.download_artifacts(f"models:/{model_name}/{entry['version']}", dst_path=tmp)
        if sha256_dir(local) != entry["model_sha256"]:
            raise RefusedModel(f"Version {entry['version']}: the model files do not match the approved digest.")
        model = mlflow.sklearn.load_model(local)
    return model, entry


def create_app(model_name=MODEL_NAME, approvals=None, store=STORE):
    model, entry = load_approved(model_name, approvals, store)
    app = FastAPI(title="Document triage (fictional example)")

    @app.get("/health")
    def health():
        return {"model": model_name, "version": entry["version"], "approved_by": entry["owner"],
                "approved_utc": entry["decided_utc"], "model_sha256": entry["model_sha256"][:16]}

    @app.post("/classify")
    def classify(doc: Document):
        if not doc.text.strip():
            raise HTTPException(422, "Empty document")
        proba = model.predict_proba([doc.text])[0]
        best = int(proba.argmax())
        conf = float(proba[best])
        return {"label": str(model.classes_[best]), "confidence": round(conf, 4),
                "requires_review": conf < entry["review_threshold"], "model_version": entry["version"]}

    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="127.0.0.1", port=8800)
