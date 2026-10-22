"""Scoring script of the Azure Machine Learning online deployment (step 7).

Author: Yves Schillings.

STATUS: prepared. tests in cloud/tests run init() and run() locally on a model saved by
train_triage.py. In Azure, the managed online deployment calls them.

Azure Machine Learning calls init() once when the deployment starts and run() for every
request. The model is the registered MLflow model of the approved training job: Azure
Machine Learning mounts it and sets AZUREML_MODEL_DIR. The answer is the same as the
local service of example 18: a label, a confidence and a review flag.
"""
import json
import os
from pathlib import Path

import mlflow.sklearn

REVIEW_THRESHOLD = 0.50          # below this confidence a person types the document
model = None


def find_model_dir(root):
    """The MLmodel file marks the MLflow model folder, directly or one level below."""
    root = Path(root)
    if (root / "MLmodel").exists():
        return root
    for path in sorted(root.rglob("MLmodel")):
        return path.parent
    raise FileNotFoundError(f"no MLflow model under {root}")


def init():
    global model
    model = mlflow.sklearn.load_model(str(find_model_dir(os.environ["AZUREML_MODEL_DIR"])))


def run(raw_data):
    """Request: {"texts": ["...", "..."]}. Answer: one label, confidence and review flag per text."""
    texts = json.loads(raw_data)["texts"]
    if not isinstance(texts, list) or not all(isinstance(t, str) and t.strip() for t in texts):
        return {"error": "texts must be a list of non-empty strings"}
    proba = model.predict_proba(texts)
    labels = model.classes_[proba.argmax(axis=1)]
    return [{"label": str(lab), "confidence": round(float(p), 4), "requires_review": bool(p < REVIEW_THRESHOLD)}
            for lab, p in zip(labels, proba.max(axis=1))]
