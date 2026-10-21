"""Serve the trained document-triage pipeline with FastAPI.

Author: Yves Schillings.

Run training first:  python examples/01_machine_learning.py
Then:                python examples/18_model_serving.py   (serves on 127.0.0.1:8800)

The service loads ONE file: the fitted scikit-learn Pipeline (TF-IDF vocabulary,
IDF weights and logistic coefficients together). Before loading it, the service
checks that the file's SHA-256 matches the manifest written at training time.
A joblib file can execute code when it is loaded, so an unverified file is refused.
Training and serving are separate processes. The artifact is their only link.
"""
import json
import os
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fictional_data import ARTIFACTS, sha256_file


class Document(BaseModel):
    text: str = Field(min_length=1, max_length=20000)


def load_verified(artifact_dir=ARTIFACTS, name="document_triage"):
    manifest = json.loads((Path(artifact_dir) / f"{name}.manifest.json").read_text(encoding="utf-8"))
    artifact = Path(artifact_dir) / manifest["artifact"]
    if sha256_file(artifact) != manifest["artifact_sha256"]:
        raise RuntimeError("Artifact checksum does not match its manifest: refusing to load it.")
    return joblib.load(artifact), manifest


def create_app(artifact_dir=None):
    model, manifest = load_verified(artifact_dir or os.environ.get("TRIAGE_ARTIFACT_DIR", ARTIFACTS))
    app = FastAPI(title="Document triage (fictional example)")

    @app.get("/health")
    def health():
        return {"model": manifest["model_name"], "artifact_sha256": manifest["artifact_sha256"][:16],
                "release_decision": manifest["release_decision"]}

    @app.post("/classify")
    def classify(doc: Document):
        if not doc.text.strip():
            raise HTTPException(422, "Empty document")
        proba = model.predict_proba([doc.text])[0]
        best = int(proba.argmax())
        conf = float(proba[best])
        return {"label": str(model.classes_[best]), "confidence": round(conf, 4),
                "requires_review": conf < manifest["review_threshold"],
                "model_sha256": manifest["artifact_sha256"][:16]}

    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="127.0.0.1", port=8800)
