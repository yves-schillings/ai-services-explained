"""Deep learning: train a neural network to recognise the layout of a scanned page.

Author: Yves Schillings.

Run:  python examples/02_deep_learning.py          (about one minute on a laptop CPU)

Need: propose the type of a scanned page from its image, before any text is read.
Data: fictional 32 x 32 page images (training/page_images.py), three types.

Part 1 trains two networks with separate training, validation and test sets:
  * an MLP, which sees the pixels as one flat list,
  * a CNN, whose convolution filters look for lines, boxes and stamps anywhere.
Each epoch logs the training loss, the validation loss and accuracy, and how far
the first-layer weights moved. Early stopping keeps the epoch with the lowest
validation loss. The test sets are used once, at the end.

Part 2 adapts a network pretrained on generic page pieces, with only 8 labelled
pages per type: feature extraction (frozen convolutions) versus full fine-tuning
versus training from scratch.
"""
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fictional_data import ARTIFACTS, RESULTS, save_json, sha256_file, write_manifest
from page_images import GENERIC_PIECES, PAGE_TYPES, SIZE, draw_generic, draw_page, make_set

ARTIFACT = ARTIFACTS / "page_layout_cnn.pt"
MANIFEST = ARTIFACTS / "page_layout_cnn.manifest.json"
REVIEW_THRESHOLD = 0.9


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
    """Few labels: reuse what a network learned on generic page pieces."""
    pre_train = tensors(*make_set(draw_generic, len(GENERIC_PIECES), 400, seed=10, hard=False))
    pre_val = tensors(*make_set(draw_generic, len(GENERIC_PIECES), 60, seed=11))
    torch.manual_seed(0)
    base = SmallCNN(n_out=len(GENERIC_PIECES))
    fit(base, pre_train, pre_val, epochs=8)
    few = tensors(*make_set(draw_page, 3, 8, seed=12))
    few_val = tensors(*make_set(draw_page, 3, 8, seed=13))
    test_clean = tensors(*make_set(draw_page, 3, 200, seed=3))
    out = {"labelled_pages_per_type": 8, "pretraining_images": len(pre_train[1])}

    def from_base():
        m = SmallCNN(3)
        m.load_state_dict({k: v for k, v in base.state_dict().items() if not k.startswith("head")}, strict=False)
        return m

    torch.manual_seed(0)
    scratch = SmallCNN(3)
    fit(scratch, few, few_val, epochs=60, patience=10)
    extract = from_base()
    for name, p in extract.named_parameters():       # feature extraction: convolutions stay frozen
        p.requires_grad = name.startswith("head") or name.startswith("fc")
    frozen_before = extract.c1.weight.detach().clone()
    fit(extract, few, few_val, epochs=60, lr=3e-3, patience=10)
    assert torch.equal(frozen_before, extract.c1.weight), "frozen weights must not change"
    tuned = from_base()                               # full fine-tuning: every weight may change
    fit(tuned, few, few_val, epochs=60, lr=5e-4, patience=10)
    for name, m in (("from_scratch", scratch), ("feature_extraction", extract), ("full_fine_tuning", tuned)):
        out[name] = {"trainable_parameters": n_params(m, trainable=True),
                     "test_clean_accuracy": test_scores(m, *test_clean)["accuracy"]}
    print("Few labels, clean test accuracy:", {k: v["test_clean_accuracy"] for k, v in out.items()
                                               if isinstance(v, dict)})
    return out


def main():
    t0 = time.time()
    part1, cnn = part1_train_from_scratch()
    part2 = part2_adapt_pretrained()
    ARTIFACTS.mkdir(exist_ok=True)
    torch.save(cnn.state_dict(), ARTIFACT)
    write_manifest(MANIFEST, model_name="page-layout-cnn", artifact=ARTIFACT.name,
                   artifact_sha256=sha256_file(ARTIFACT), architecture="SmallCNN, 3 convolution layers",
                   input="1 x 32 x 32 grey image, ink = 1", classes=PAGE_TYPES,
                   metrics={"test_clean": part1["cnn"]["test_clean"]["accuracy"],
                            "test_hard": part1["cnn"]["test_hard"]["accuracy"]},
                   review_threshold=REVIEW_THRESHOLD, owner="document intake service owner (fictional)",
                   release_decision="pending human review")
    result = {"device": "cpu", "torch": torch.__version__, "seconds": round(time.time() - t0, 1),
              "part1": part1, "part2": part2}
    save_json("02_deep_learning.json", result)
    plot(part1, part2)
    print(f"Done in {result['seconds']} s. Artifact {ARTIFACT.name} and manifest written.")
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
    vals = [part1["mlp"]["test_clean"]["accuracy"], part1["mlp"]["test_hard"]["accuracy"],
            part1["cnn"]["test_clean"]["accuracy"], part1["cnn"]["test_hard"]["accuracy"],
            part2["from_scratch"]["test_clean_accuracy"], part2["feature_extraction"]["test_clean_accuracy"],
            part2["full_fine_tuning"]["test_clean_accuracy"]]
    ax[2].bar(range(len(vals)), vals, color=["#9aa5b1", "#9aa5b1", "#0072c7", "#0072c7", "#d73285", "#008e48",
                                             "#e66a00"])
    for i, v in enumerate(vals):
        ax[2].text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax[2].set_xticks(range(len(vals)), names, fontsize=7)
    ax[2].set_ylim(0, 1.1)
    ax[2].set_title("Test accuracy, used once")
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "02_deep_learning.png", dpi=130)


if __name__ == "__main__":
    main()
