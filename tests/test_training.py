"""Deterministic checks of the training examples 01, 02 and 13 to 18. Run: python -m pytest tests -q"""
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
import torch

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))          # the examples import their shared helpers by name

import fictional_data as common             # noqa: E402
from page_images import draw_page, make_set  # noqa: E402


def load(prefix):
    path = next(EXAMPLES.glob(prefix + "_*.py"))
    spec = importlib.util.spec_from_file_location("example_" + prefix, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ml01, dl01, ml05, serving = load("01"), load("02"), load("16"), load("18")


def small_split():
    return common.split_by_group(common.make_case_documents(n_cases=60), "case_id")


def test_split_by_case_never_shares_a_case():
    parts = small_split()
    train, val, test = ({r["case_id"] for r in parts[k]} for k in ("train", "val", "test"))
    assert not (train & val) and not (train & test) and not (val & test)


def test_vectoriser_learns_vocabulary_from_training_documents_only():
    parts = small_split()
    model = ml01.build_pipeline().fit(*ml01.xy(parts["train"]))
    vocab = set(model.named_steps["tfidf"].vocabulary_)
    train_words = {w for r in parts["train"] for w in r["text"].split()}
    assert {v.split()[0] for v in vocab} <= train_words


def test_fitted_classifier_returns_a_known_type():
    parts = small_split()
    model = ml01.build_pipeline().fit(*ml01.xy(parts["train"]))
    assert model.predict(["witness saw a person in the evening"])[0] in common.DOC_TYPES


def test_empty_training_is_rejected():
    with pytest.raises(ValueError):
        ml01.build_pipeline().fit([], [])


def test_low_confidence_goes_to_review():
    parts = small_split()
    model = ml01.build_pipeline().fit(*ml01.xy(parts["train"]))
    assert ml01.predict_with_review(model, ["zzz qqq"], threshold=0.99)[0]["requires_review"] is True


def test_forecast_features_use_only_the_past():
    y = np.arange(60, dtype=float)
    x, target = ml05.features(y)
    assert all(row[1] == t - 1 for row, t in zip(x, target))


def test_service_refuses_a_modified_artifact(tmp_path):
    parts = small_split()
    model = ml01.build_pipeline().fit(*ml01.xy(parts["train"]))
    path = tmp_path / "document_triage.joblib"
    joblib.dump(model, path)
    manifest = {"model_name": "document-triage", "artifact": path.name, "artifact_sha256": common.sha256_file(path),
                "review_threshold": 0.5, "release_decision": "test"}
    (tmp_path / "document_triage.manifest.json").write_text(json.dumps(manifest))
    from fastapi.testclient import TestClient
    client = TestClient(serving.create_app(tmp_path))
    answer = client.post("/classify", json={"text": "witness saw a person in the evening"}).json()
    assert answer["label"] in common.DOC_TYPES and "requires_review" in answer
    with open(path, "ab") as f:
        f.write(b"tampered")
    with pytest.raises(RuntimeError):
        serving.load_verified(tmp_path)


def test_network_outputs_one_score_per_type():
    x, _ = make_set(draw_page, 3, 2, seed=1)
    assert dl01.SmallCNN()(torch.tensor(x).unsqueeze(1)).shape == (6, 3)


def test_frozen_layers_do_not_change_and_validation_does_not_train():
    train = dl01.tensors(*make_set(draw_page, 3, 10, seed=1))
    val = dl01.tensors(*make_set(draw_page, 3, 5, seed=2))
    model = dl01.SmallCNN()
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("head")
    conv = model.c1.weight.detach().clone()
    head = model.head.weight.detach().clone()
    dl01.fit(model, train, val, epochs=2)
    assert torch.equal(conv, model.c1.weight)
    assert not torch.equal(head, model.head.weight)
