"""Deterministic checks of the training examples 01, 02 and 13 to 19. Run: python -m pytest tests -q"""
import copy
import importlib.util
import os
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import mlflow
import mlflow.sklearn
import numpy as np
import pytest
import torch

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))          # the examples import their shared helpers by name

import fictional_data as common             # noqa: E402
import mlflow_tracking                      # noqa: E402
from page_images import draw_page, make_set  # noqa: E402
from release_records import append_decision  # noqa: E402


def load(prefix):
    path = next(EXAMPLES.glob(prefix + "_*.py"))
    spec = importlib.util.spec_from_file_location("example_" + prefix, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ml01, dl01, ml05, serving, release = load("01"), load("02"), load("16"), load("18"), load("19")
OWNER = "test owner"


def small_split():
    return common.split_by_group(common.make_case_documents(n_cases=60), "case_id")


def test_split_by_case_never_shares_a_case():
    parts = small_split()
    train, val, test = ({r["case_id"] for r in parts[k]} for k in ("train", "val", "test"))
    assert not (train & val) and not (train & test) and not (val & test)


def test_final_test_set_is_outside_every_training_pool():
    final = {r["case_id"] for r in common.make_final_test_set()}
    for n in ml01.POOLS:
        assert not final & {r["case_id"] for r in common.make_case_documents(n_cases=n)}


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


def test_keyword_rule_falls_back_to_the_default_type():
    assert ml01.keyword_rule("dear colleague, regards", "incident_report") == "correspondence"
    assert ml01.keyword_rule("nothing typical here", "incident_report") == "incident_report"


def test_forecast_features_use_only_the_past():
    y = np.arange(60, dtype=float)
    x, target = ml05.features(y)
    assert all(row[1] == t - 1 for row, t in zip(x, target))


# --------------------------------------------------------------------------- release gate and serving
@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    """A throw-away MLflow store with five versions. Their test scores are fixed for the test."""
    os.environ.pop("MLFLOW_TRACKING_URI", None)
    store = tmp_path_factory.mktemp("mlflow")
    client = mlflow_tracking.use_experiment("test-document-triage", store=store)
    parts = small_split()
    model = ml01.build_pipeline().fit(*ml01.xy(parts["train"]))
    for score in (0.80, 0.90, 0.91, 0.92, 0.93):
        with mlflow.start_run():
            mlflow.log_params({"review_threshold": 0.5})
            mlflow.log_metric("test_macro_f1", score)
            mlflow.sklearn.log_model(model, name="model", registered_model_name=serving.MODEL_NAME,
                                     pip_requirements=[])
    return client, store


def model_dir(client, version):
    model_id = client.get_model_version(serving.MODEL_NAME, version).source.rsplit("/", 1)[-1]   # models:/m-...
    return Path(url2pathname(urlparse(mlflow.get_logged_model(model_id).artifact_location).path))


def test_release_refuses_to_approve_below_the_gate(registry, tmp_path):
    client, _ = registry
    with pytest.raises(release.GateFailed):
        release.decide(client, "1", "approved", OWNER, "test", tmp_path, update_alias=False)


@pytest.mark.parametrize("decision", ["pending", "denied"])
def test_service_never_serves_a_pending_or_denied_version(registry, tmp_path, decision):
    client, store = registry
    release.decide(client, "2", decision, OWNER, "test", tmp_path, update_alias=False)
    with pytest.raises(serving.RefusedModel):
        serving.load_approved(approvals=tmp_path, store=store)


def test_service_refuses_an_approval_below_the_gate(registry, tmp_path):
    _, store = registry
    append_decision(serving.MODEL_NAME, {"version": "1", "decision": "approved", "test_macro_f1": 0.80,
                                         "model_sha256": "x", "review_threshold": 0.5, "owner": OWNER},
                    acceptance=0.85, directory=tmp_path)
    with pytest.raises(serving.RefusedModel):
        serving.load_approved(approvals=tmp_path, store=store)


def test_service_serves_the_approved_version_and_rolls_back(registry, tmp_path):
    client, store = registry
    release.decide(client, "2", "approved", OWNER, "test", tmp_path)
    release.decide(client, "3", "approved", OWNER, "test", tmp_path)
    from fastapi.testclient import TestClient
    api = TestClient(serving.create_app(approvals=tmp_path, store=store))
    assert api.get("/health").json()["version"] == "3"
    answer = api.post("/classify", json={"text": "witness saw a person in the evening"}).json()
    assert answer["label"] in common.DOC_TYPES and answer["model_version"] == "3"
    assert str(client.get_model_version_by_alias(serving.MODEL_NAME, release.ALIAS).version) == "3"
    withdrawn, live = release.rollback(client, OWNER, "test", tmp_path)
    assert (withdrawn, live["version"]) == ("3", "2")
    assert serving.load_approved(approvals=tmp_path, store=store)[1]["version"] == "2"
    assert str(client.get_model_version_by_alias(serving.MODEL_NAME, release.ALIAS).version) == "2"


def test_service_refuses_a_tampered_model_file(registry, tmp_path):
    client, store = registry
    release.decide(client, "4", "approved", OWNER, "test", tmp_path, update_alias=False)
    target = next(p for p in model_dir(client, "4").iterdir() if p.name.startswith("model."))
    with open(target, "ab") as f:
        f.write(b"tampered")
    with pytest.raises(serving.RefusedModel):
        serving.load_approved(approvals=tmp_path, store=store)


def test_service_refuses_a_model_replaced_together_with_its_metadata(registry, tmp_path):
    client, store = registry
    release.decide(client, "5", "approved", OWNER, "test", tmp_path, update_alias=False)
    source, target = model_dir(client, "2"), model_dir(client, "5")
    for f in source.iterdir():                         # every file swapped, self-consistent again
        if f.is_file():
            shutil.copy(f, target / f.name)
    with pytest.raises(serving.RefusedModel):
        serving.load_approved(approvals=tmp_path, store=store)


# --------------------------------------------------------------------------- deep learning
def test_network_outputs_one_score_per_type():
    x, _ = make_set(draw_page, 3, 2, seed=1)
    assert dl01.SmallCNN()(torch.tensor(x).unsqueeze(1)).shape == (6, 3)


def test_evaluation_changes_no_weight():
    model = dl01.SmallCNN()
    before = copy.deepcopy(model.state_dict())
    x, y = dl01.tensors(*make_set(draw_page, 3, 5, seed=2))
    dl01.evaluate(model, x, y)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())


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
