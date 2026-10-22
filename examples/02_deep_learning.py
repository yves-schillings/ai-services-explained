"""Deep learning: train a neural network to recognise the layout of a scanned page.

Author: Yves Schillings.

Run:  python examples/02_deep_learning.py          (about one minute on a laptop CPU)

Need: propose the type of a scanned page from its image, before any text is read.
Data: fictional 32 x 32 page images (page_images.py), three types.

Part 1 trains two networks with separate training, validation and test sets:
  * an MLP, which sees the pixels as one flat list,
  * a CNN, whose convolution filters look for lines, boxes and stamps anywhere.
Each epoch logs the training loss, the validation loss and accuracy, and how far
the first-layer weights moved. Early stopping keeps the epoch with the lowest
validation loss. The test sets are used once, at the end.

Part 2 adapts a network pretrained on generic page pieces, with only 8 training
+ 8 validation pages per type: training from scratch (every weight learns),
feature extraction (frozen convolutions, only the dense layer and the head learn)
and full fine-tuning (every weight learns, from the pretrained values). For each
seed the three modes start from the same initial head weights, and the two
adapted modes from the same pretrained layers. Five seeds also vary which pages
are labelled, and the results report the mean and the spread.

Reproducibility: the run fixes the number of CPU threads (results can differ with
another thread count), requests deterministic algorithms and records the versions.

MLflow records each training run: parameters, per-epoch curves, test scores and,
for the CNN, the saved model as a new version of "page-layout-cnn".
"""
import copy
import time

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fictional_data import RESULTS, run_environment, save_json
from mlflow_tracking import pinned, use_experiment
from page_images import GENERIC_PIECES, SIZE, draw_generic, draw_page, make_set

REVIEW_THRESHOLD = 0.9
N_THREADS = 4                    # fixed and recorded: another thread count can change the results
SEEDS = (0, 1, 2, 3, 4)          # adaptation runs, to measure the spread
MODEL_NAME = "page-layout-cnn"
EXPERIMENT = "page-layout"


class MLP(nn.Module):
    """Fully connected layers on the flattened pixels."""

    def __init__(self, n_out=3):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(SIZE * SIZE, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_out))

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    """Three convolution layers (edges, then shapes, then layouts) and a classifier head."""

    def __init__(self, n_out=3):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 4 * 4, 64)
        self.head = nn.Linear(64, n_out)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = F.max_pool2d(F.relu(self.c3(x)), 2)
        return self.head(F.relu(self.fc(x.flatten(1))))


def tensors(x, y):
    return torch.tensor(x).unsqueeze(1), torch.tensor(y)


def n_params(model, trainable=False):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable))


@torch.no_grad()
def evaluate(model, x, y):
    model.eval()
    logits = model(x)
    loss = F.cross_entropy(logits, y).item()
    proba = F.softmax(logits, dim=1)
    conf, pred = proba.max(dim=1)
    return loss, pred.numpy(), conf.numpy()


def fit(model, train, val, epochs=30, lr=2e-3, batch=32, patience=5, seed=0):
    """The training loop: forward pass, loss, backpropagation, optimiser step, for every batch."""
    torch.manual_seed(seed)
    x, y = train
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    first = next(model.parameters())
    start = first.detach().clone()
    history, best, best_loss, waited = [], None, float("inf"), 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(y))
        total = 0.0
        for i in range(0, len(y), batch):                 # one batch = 32 pages
            b = order[i:i + batch]
            logits = model(x[b])                          # 1 forward pass
            loss = F.cross_entropy(logits, y[b])          # 2 loss against the true type
            opt.zero_grad()
            loss.backward()                               # 3 gradients by backpropagation
            opt.step()                                    # 4 optimiser updates the weights
            total += loss.item() * len(b)
        val_loss, val_pred, _ = evaluate(model, *val)     # validation never updates weights
        history.append({"epoch": epoch, "train_loss": round(total / len(y), 4), "val_loss": round(val_loss, 4),
                        "val_accuracy": round(float((val_pred == val[1].numpy()).mean()), 4),
                        "first_layer_change": round(float((first.detach() - start).norm()), 4)})
        if val_loss < best_loss - 1e-4:                   # early stopping on validation loss
            best_loss, best, waited = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(best)                           # keep the best epoch
    return history


def test_scores(model, x, y):
    _, pred, conf = evaluate(model, x, y)
    y = y.numpy()
    keep = conf >= REVIEW_THRESHOLD
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    return {"accuracy": round(float((pred == y).mean()), 4),
            "share_kept_at_0_9": round(float(keep.mean()), 4),
            "accuracy_on_kept": round(float((pred[keep] == y[keep]).mean()), 4) if keep.any() else None,
            "confusion_matrix": cm.tolist()}


def part1_train_from_scratch():
    train = tensors(*make_set(draw_page, 3, 150, seed=1))
    val = tensors(*make_set(draw_page, 3, 60, seed=2))
    test_clean = tensors(*make_set(draw_page, 3, 200, seed=3))
    test_hard = tensors(*make_set(draw_page, 3, 200, seed=4, hard=True))
    out = {"sizes": {"train": len(train[1]), "validation": len(val[1]), "test_clean": len(test_clean[1]),
                     "test_hard": len(test_hard[1])}}
    models = {}
    for name, cls in (("mlp", MLP), ("cnn", SmallCNN)):
        torch.manual_seed(0)
        model = cls()
        hist = fit(model, train, val)
        best_epoch = min(hist, key=lambda h: h["val_loss"])["epoch"]
        out[name] = {"parameters": n_params(model), "epochs_run": len(hist), "best_epoch": best_epoch,
                     "history": hist, "test_clean": test_scores(model, *test_clean),
                     "test_hard": test_scores(model, *test_hard)}
        models[name] = model
        print(f"{name.upper()}: {out[name]['parameters']:,} weights, stopped after {len(hist)} epochs, "
              f"best epoch {best_epoch}. Clean test {out[name]['test_clean']['accuracy']:.3f}, "
              f"hard test {out[name]['test_hard']['accuracy']:.3f}")
    return out, models["cnn"]


def part2_adapt_pretrained():
    """Few labels: reuse what a network learned on generic page pieces. 8 training + 8 validation pages per type."""
    pre_train = tensors(*make_set(draw_generic, len(GENERIC_PIECES), 400, seed=10, hard=False))
    pre_val = tensors(*make_set(draw_generic, len(GENERIC_PIECES), 60, seed=11))
    torch.manual_seed(0)
    base = SmallCNN(n_out=len(GENERIC_PIECES))
    fit(base, pre_train, pre_val, epochs=8)
    pretrained = {k: v for k, v in base.state_dict().items() if not k.startswith("head")}
    test_clean = tensors(*make_set(draw_page, 3, 200, seed=3))
    modes = ("from_scratch", "feature_extraction", "full_fine_tuning")
    per_seed, trainable = [], {}
    for seed in SEEDS:
        few = tensors(*make_set(draw_page, 3, 8, seed=100 + seed))
        few_val = tensors(*make_set(draw_page, 3, 8, seed=200 + seed))
        torch.manual_seed(seed)
        start = SmallCNN(3)                           # one random start, shared by the three modes
        scratch = copy.deepcopy(start)
        extract = copy.deepcopy(start)
        extract.load_state_dict(pretrained, strict=False)
        tuned = copy.deepcopy(extract)                # same pretrained layers, same head start
        fit(scratch, few, few_val, epochs=60, patience=10, seed=seed)
        for name, p in extract.named_parameters():   # feature extraction: convolutions stay frozen
            p.requires_grad = name.startswith("head") or name.startswith("fc")
        frozen_before = extract.c1.weight.detach().clone()
        fit(extract, few, few_val, epochs=60, lr=3e-3, patience=10, seed=seed)
        assert torch.equal(frozen_before, extract.c1.weight), "frozen weights must not change"
        fit(tuned, few, few_val, epochs=60, lr=5e-4, patience=10, seed=seed)  # every weight may change
        row = {"seed": seed}
        for name, m in zip(modes, (scratch, extract, tuned)):
            row[name] = test_scores(m, *test_clean)["accuracy"]
            trainable[name] = (n_params(m, trainable=True),
                               sorted({k.split(".")[0] for k, p in m.named_parameters() if p.requires_grad}))
        row["best"] = max(modes, key=lambda k: row[k])
        per_seed.append(row)
    out = {"labelled_pages_per_type": {"training": 8, "validation": 8}, "pretraining_images": len(pre_train[1]),
           "test_pages": len(test_clean[1]), "seeds": list(SEEDS), "per_seed": per_seed}
    for name in modes:
        acc = np.array([r[name] for r in per_seed])
        out[name] = {"trainable_parameters": trainable[name][0], "layers_that_learn": trainable[name][1],
                     "test_clean_accuracy_mean": round(float(acc.mean()), 4),
                     "test_clean_accuracy_std": round(float(acc.std(ddof=1)), 4),
                     "test_clean_accuracy_min": round(float(acc.min()), 4),
                     "test_clean_accuracy_max": round(float(acc.max()), 4)}
    out["best_mode_count"] = {name: sum(r["best"] == name for r in per_seed) for name in modes}
    print("Few labels, clean test accuracy over", len(SEEDS), "seeds:",
          {k: f"{out[k]['test_clean_accuracy_mean']:.3f} +/- {out[k]['test_clean_accuracy_std']:.3f}" for k in modes},
          "- best mode per seed:", out["best_mode_count"])
    return out


def log_part1(part1, cnn):
    """One MLflow run per network, with its curves. The CNN is registered as a new model version."""
    version = None
    for name in ("mlp", "cnn"):
        r = part1[name]
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"network": name.upper(), "weights": r["parameters"], "max_epochs": 30, "lr": 2e-3,
                               "batch": 32, "patience": 5, "seed": 0, "threads": N_THREADS,
                               "train_pages": part1["sizes"]["train"], "validation_pages": part1["sizes"]["validation"]})
            for h in r["history"]:                    # the curves, one step per epoch
                mlflow.log_metrics({"train_loss": h["train_loss"], "val_loss": h["val_loss"],
                                    "val_accuracy": h["val_accuracy"], "first_layer_change": h["first_layer_change"]},
                                   step=h["epoch"])
            mlflow.log_metrics({"test_clean_accuracy": r["test_clean"]["accuracy"],
                                "test_hard_accuracy": r["test_hard"]["accuracy"], "best_epoch": r["best_epoch"],
                                "epochs_run": r["epochs_run"]})
            if name == "cnn":
                # Saved as a traced graph (pt2): loading it needs neither this file nor its classes.
                # An example batch of 2 pages keeps the batch size free in the saved graph.
                example = tensors(*make_set(draw_page, 3, 1, seed=6))[0][:2].numpy()
                info = mlflow.pytorch.log_model(cnn.eval(), name="model", registered_model_name=MODEL_NAME,
                                                input_example=example, serialization_format="pt2",
                                                pip_requirements=pinned("torch", "numpy"), await_registration_for=0)
                version = str(info.registered_model_version)
    return version


def log_part2(part2):
    with mlflow.start_run(run_name="adaptation-8-labels"):
        mlflow.log_params({"labelled_pages_per_type": "8 training + 8 validation", "seeds": len(SEEDS),
                           "threads": N_THREADS, "pretraining_images": part2["pretraining_images"]})
        for r in part2["per_seed"]:                   # one step per seed, to see the spread
            mlflow.log_metrics({k: r[k] for k in ("from_scratch", "feature_extraction", "full_fine_tuning")},
                               step=r["seed"])
        for k in ("from_scratch", "feature_extraction", "full_fine_tuning"):
            mlflow.log_metrics({f"{k}_mean": part2[k]["test_clean_accuracy_mean"],
                                f"{k}_std": part2[k]["test_clean_accuracy_std"]})


def main():
    torch.set_num_threads(N_THREADS)
    torch.use_deterministic_algorithms(True, warn_only=True)
    t0 = time.time()
    use_experiment(EXPERIMENT)
    part1, cnn = part1_train_from_scratch()
    version = log_part1(part1, cnn)
    reloaded = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}/{version}")
    x = tensors(*make_set(draw_page, 3, 20, seed=5))[0]
    with torch.no_grad():
        same = bool(torch.equal(cnn.eval()(x).argmax(1), reloaded(x).argmax(1)))
    print(f"CNN registered as version {version} of {MODEL_NAME}. Reloaded from the registry, same predictions: {same}")
    part2 = part2_adapt_pretrained()
    log_part2(part2)
    result = {"device": "cpu", "threads": N_THREADS, "deterministic_algorithms": True,
              "seconds": round(time.time() - t0, 1), "registered_version": version,
              "reloaded_model_same_predictions": same, "environment": run_environment(N_THREADS),
              "command": "python examples/02_deep_learning.py", "part1": part1, "part2": part2}
    save_json("02_deep_learning.json", result)
    plot(part1, part2)
    print(f"Done in {result['seconds']} s with {N_THREADS} CPU threads.")
    return result


def plot(part1, part2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for name, style in (("mlp", "--"), ("cnn", "-")):
        h = part1[name]["history"]
        e = [r["epoch"] for r in h]
        ax[0].plot(e, [r["train_loss"] for r in h], style, label=f"{name.upper()} training loss")
        ax[0].plot(e, [r["val_loss"] for r in h], style, marker="o", markersize=3,
                   label=f"{name.upper()} validation loss")
        ax[1].plot(e, [r["val_accuracy"] for r in h], style, marker="o", markersize=3,
                   label=f"{name.upper()} validation accuracy")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("cross-entropy loss")
    ax[0].set_title("Loss per epoch (measured run)")
    ax[0].legend(fontsize=8)
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("accuracy")
    ax[1].set_title("Validation accuracy per epoch")
    ax[1].legend(fontsize=8)
    names = ["MLP\nclean", "MLP\nhard", "CNN\nclean", "CNN\nhard", "8 labels\nscratch", "8 labels\nfrozen",
             "8 labels\nfine-tuned"]
    modes = ("from_scratch", "feature_extraction", "full_fine_tuning")
    vals = [part1["mlp"]["test_clean"]["accuracy"], part1["mlp"]["test_hard"]["accuracy"],
            part1["cnn"]["test_clean"]["accuracy"], part1["cnn"]["test_hard"]["accuracy"]] + \
           [part2[m]["test_clean_accuracy_mean"] for m in modes]
    errs = [0, 0, 0, 0] + [part2[m]["test_clean_accuracy_std"] for m in modes]
    ax[2].bar(range(len(vals)), vals, yerr=errs, capsize=4,
              color=["#9aa5b1", "#9aa5b1", "#0072c7", "#0072c7", "#d73285", "#008e48", "#e66a00"])
    for i, v in enumerate(vals):
        ax[2].text(i, v + errs[i] + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax[2].set_xticks(range(len(vals)), names, fontsize=7)
    ax[2].set_ylim(0, 1.15)
    ax[2].set_title(f"Test accuracy, used once (8 labels: mean and spread of {len(SEEDS)} seeds)")
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "02_deep_learning.png", dpi=130)


if __name__ == "__main__":
    main()
