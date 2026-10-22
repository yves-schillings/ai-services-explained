"""Cloud entry script: train the document-triage classifier inside a training job.

Author: Yves Schillings.

The same script runs in three places, with the same code and the same data:
  * locally, for a check before any cloud spending:
        python cloud/shared/stage.py
        python cloud/build/src/train_triage.py --data cloud/data/triage --output cloud/outputs/triage
  * as an Azure Machine Learning command job (cloud/azure/job_triage.yml),
  * as a SageMaker AI training job (cloud/aws/sagemaker/launch_training.py).

What it does, in the order of the walkthrough:
  4. it runs where the job runs. Inputs are the three JSON Lines files of
     cloud/shared/prepare_data.py: train, val and the fixed final test set;
  5. it chooses the settings on validation, fits on training, scores the final test
     set once, and logs parameters, data version, curve, metrics and the model to
     MLflow. Inside Azure Machine Learning the MLflow tracking URI is already set.
     On SageMaker AI the launcher passes the tracking server ARN in
     MLFLOW_TRACKING_URI. Locally, MLflow writes to the output folder;
  6. it writes gate.json: does the test macro-F1 reach the agreed 0,85? The script
     does NOT register or deploy anything. Registration and deployment happen only
     after the gate passes and a named owner approves (cloud/shared/approval.py).

Outputs in --output: model/ (MLflow model format), metrics.json, gate.json.
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

from fictional_data import DOC_TYPES, sha256_json  # noqa: E402
from machine_learning import (ACCEPTANCE_MACRO_F1, build_pipeline, predict_with_review,  # noqa: E402
                              select_hyperparameters, xy)

EXPERIMENT = "document-triage-cloud"


def pinned(*packages):
    """Exact versions written next to the model, instead of guessing them from the environment."""
    from importlib import metadata
    return [f"{p}=={metadata.version(p).split('+')[0]}" for p in packages]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def default_dir(env, fallback):
    """SageMaker AI passes folders in environment variables. Other places use the arguments."""
    return os.environ.get(env, fallback)


def train(data_dir, output_dir, threshold=ACCEPTANCE_MACRO_F1):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    sets = {name: read_jsonl(data_dir / f"{name}.jsonl") for name in ("train", "val", "test")}
    shared = {r["case_id"] for r in sets["train"]} & {r["case_id"] for r in sets["test"]}
    assert not shared, "a case appears in training and in the final test set"

    best, curve = select_hyperparameters(sets["train"], sets["val"])          # VALIDATION chooses
    cw = None if best["class_weight"] == "none" else best["class_weight"]
    model = build_pipeline(best["C"], cw).fit(*xy(sets["train"]))            # TRAINING learns
    x_test, y_test = xy(sets["test"])
    pred = model.predict(x_test)                                              # the final test, once
    review = predict_with_review(model, x_test)
    metrics = {"val_macro_f1": round(best["val_macro_f1"], 4),
               "test_macro_f1": round(f1_score(y_test, pred, average="macro"), 4),
               "test_accuracy": round(accuracy_score(y_test, pred), 4),
               "share_sent_to_review": round(sum(p["requires_review"] for p in review) / len(review), 4)}
    gate = {"metric": "test_macro_f1", "value": metrics["test_macro_f1"], "threshold": threshold,
            "passes": metrics["test_macro_f1"] >= threshold,
            "note": "Passing the gate makes the model a candidate. A named owner still approves."}

    with mlflow.start_run(run_name="document-triage-cloud"):
        mlflow.log_params({"C": best["C"], "class_weight": best["class_weight"], "ngram_range": "1-2",
                           "min_df": 2, "documents_train": len(sets["train"]), "documents_val": len(sets["val"]),
                           "documents_test": len(sets["test"]), "acceptance_macro_f1": threshold})
        mlflow.set_tags({"data_version": sha256_json(sets["train"] + sets["val"])[:16],
                         "final_test_version": sha256_json(sets["test"])[:16],
                         "gate": "passed" if gate["passes"] else "failed", "labels": ",".join(DOC_TYPES)})
        rows = [c for c in curve if c["class_weight"] == best["class_weight"]]
        for step, point in enumerate(rows):                                   # the validation curve
            mlflow.log_metric("train_macro_f1_by_C", point["train_macro_f1"], step=step)
            mlflow.log_metric("val_macro_f1_by_C", point["val_macro_f1"], step=step)
        mlflow.log_metrics(metrics)
        mlflow.log_dict(gate, "gate.json")
        requirements = pinned("scikit-learn", "numpy", "scipy", "joblib")
        mlflow.sklearn.log_model(sk_model=model, name="model", pip_requirements=requirements)   # not registered

    output_dir.mkdir(parents=True, exist_ok=True)
    mlflow.sklearn.save_model(model, output_dir / "model", pip_requirements=requirements)   # job output folder
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    return metrics, gate


def local_tracking(output):
    """Outside a cloud job: a SQLite store and an artifact folder inside the output folder."""
    output.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///" + (output / "mlflow.db").resolve().as_posix())
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(output / "artifacts").resolve().as_uri())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", default=default_dir("SM_CHANNEL_TRAINING", "cloud/data/triage"),
                   help="folder with train.jsonl, val.jsonl and test.jsonl")
    p.add_argument("--output", default=default_dir("SM_MODEL_DIR", "cloud/outputs/triage"),
                   help="folder for the model, metrics.json and gate.json")
    a = p.parse_args(argv)
    if not os.environ.get("MLFLOW_TRACKING_URI") and "AZUREML_RUN_ID" not in os.environ:
        local_tracking(Path(a.output))                                       # local check only
    mlflow.set_experiment(EXPERIMENT)
    metrics, gate = train(a.data, a.output)
    print("Metrics:", metrics)
    print("Gate:", "passed" if gate["passes"] else "failed", f"({gate['value']} against {gate['threshold']})")


if __name__ == "__main__":
    main()
