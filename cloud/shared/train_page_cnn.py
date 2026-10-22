"""Cloud entry script: train the page-layout CNN inside a training job, on CPU or GPU.

Author: Yves Schillings.

The same script runs locally, as an Azure Machine Learning command job on a GPU
cluster (cloud/azure/job_page_cnn_gpu.yml) and as a SageMaker AI training job:
    python cloud/shared/stage.py
    python cloud/build/src/train_page_cnn.py --data cloud/data/pages --output cloud/outputs/pages

It reuses SmallCNN, fit() and test_scores() of examples/02_deep_learning.py, with the
same pages, the same seeds and the same early stopping. fit() keeps the CPU loop of
example 02: this small network trains on a CPU in seconds, so a CPU cluster is the
right choice for it. The GPU job definition shows how to request a GPU for a larger
network, which would also move the model and each batch to the device. Another
machine or thread count can change the last digits: the numbers of one run are the
evidence for that run only.

Outputs in --output: model/ (MLflow PyTorch model), page_layout_cnn.pt, metrics.json, gate.json.
The script logs to MLflow and does not register or deploy anything.
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
import mlflow.pytorch  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from deep_learning import SmallCNN, fit, n_params, tensors, test_scores  # noqa: E402

EXPERIMENT = "page-layout-cloud"
MIN_CLEAN_ACCURACY = 0.95       # fictional gate for this example, agreed before training


def load(data_dir, name):
    d = np.load(Path(data_dir) / f"{name}.npz")
    return tensors(d["x"], d["y"])


def save_cnn(save, model, example, **where):
    """Recent MLflow saves a traced graph (pt2) that loads without this code. Older clients pickle the model."""
    from importlib import metadata
    requirements = [f"{p}=={metadata.version(p).split('+')[0]}" for p in ("torch", "numpy")]
    try:
        return save(model.eval(), input_example=example, serialization_format="pt2",
                    pip_requirements=requirements, **where)
    except TypeError:                                                        # MLflow before the pt2 format
        return save(model.eval(), input_example=example, pip_requirements=requirements, **where)


def train(data_dir, output_dir, threshold=MIN_CLEAN_ACCURACY, epochs=30, seed=0):
    sets = {name: load(data_dir, name) for name in ("train", "val", "test_clean", "test_hard")}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    model = SmallCNN()
    history = fit(model, sets["train"], sets["val"], epochs=epochs, seed=seed)  # CPU loop of example 02
    clean, hard = test_scores(model, *sets["test_clean"]), test_scores(model, *sets["test_hard"])
    metrics = {"test_clean_accuracy": clean["accuracy"], "test_hard_accuracy": hard["accuracy"],
               "epochs_run": len(history), "parameters": n_params(model)}
    gate = {"metric": "test_clean_accuracy", "value": clean["accuracy"], "threshold": threshold,
            "passes": clean["accuracy"] >= threshold,
            "note": "Passing the gate makes the model a candidate. A named owner still approves."}

    with mlflow.start_run(run_name="page-layout-cnn-cloud"):
        mlflow.log_params({"architecture": "SmallCNN, 3 convolution layers", "epochs_max": epochs,
                           "seed": seed, "device_available": device, "pages_train": len(sets["train"][1]),
                           "pages_val": len(sets["val"][1]), "min_clean_accuracy": threshold})
        for h in history:                                                    # the training curves
            mlflow.log_metrics({"train_loss": h["train_loss"], "val_loss": h["val_loss"],
                                "val_accuracy": h["val_accuracy"]}, step=h["epoch"])
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.log_dict(gate, "gate.json")
        example = sets["test_clean"][0][:2].numpy()                        # 2 pages: batch size stays free
        save_cnn(mlflow.pytorch.log_model, model, example, name="model")    # logged, not registered

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "page_layout_cnn.pt")
    save_cnn(mlflow.pytorch.save_model, model, example, path=str(output_dir / "model"))
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
    p.add_argument("--data", default=os.environ.get("SM_CHANNEL_TRAINING", "cloud/data/pages"))
    p.add_argument("--output", default=os.environ.get("SM_MODEL_DIR", "cloud/outputs/pages"))
    p.add_argument("--epochs", type=int, default=30)
    a = p.parse_args(argv)
    if not os.environ.get("MLFLOW_TRACKING_URI") and "AZUREML_RUN_ID" not in os.environ:
        local_tracking(Path(a.output))                                       # local check only
    mlflow.set_experiment(EXPERIMENT)
    metrics, gate = train(a.data, a.output, epochs=a.epochs)
    print("Metrics:", metrics)
    print("Gate:", "passed" if gate["passes"] else "failed", f"({gate['value']} against {gate['threshold']})")


if __name__ == "__main__":
    main()
