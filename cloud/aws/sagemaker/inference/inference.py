"""Real-time endpoint handler for the document-triage classifier on SageMaker AI (step 7).

Author: Yves Schillings.

STATUS: prepared, not deployed. It follows the DJL Serving Python mode used by the
SageMaker Python SDK v3 script-mode examples: one handle(inputs) function. The endpoint
container unpacks model.tar.gz into /opt/ml/model; the MLflow model saved by
train_triage.py is in its model/ folder.

Request: {"texts": ["...", "..."]}. Answer: a label, a confidence and a review flag per
text, exactly like the local service of example 18 and the Azure scoring script.
"""
import os
from pathlib import Path

import mlflow.sklearn

REVIEW_THRESHOLD = 0.50
MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
_model = None


def load_model(model_dir=MODEL_DIR):
    """Loaded once, on the first (empty) warm-up request."""
    global _model
    if _model is None:
        _model = mlflow.sklearn.load_model(str(Path(model_dir) / "model"))
    return _model


def predict(texts, model=None):
    model = model or load_model()
    proba = model.predict_proba(texts)
    labels = model.classes_[proba.argmax(axis=1)]
    return [{"label": str(lab), "confidence": round(float(p), 4), "requires_review": bool(p < REVIEW_THRESHOLD)}
            for lab, p in zip(labels, proba.max(axis=1))]


def handle(inputs):
    """DJL Serving entry point. An empty request only loads the model."""
    from djl_python import Output
    load_model()
    if inputs.is_empty():
        return None
    texts = inputs.get_as_json()["texts"]
    return Output().add_as_json(predict(texts))
