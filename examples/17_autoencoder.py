"""Representation learning: an autoencoder flags pages that look like none of the known layouts.

Author: Yves Schillings.

Run:  python examples/17_autoencoder.py

Need: before classifying a page, the intake service wants to notice pages that
are unlike anything the classifier was trained on, and send them to a person.
An autoencoder learns to compress a page into 16 numbers and rebuild it. It is
trained only on usual pages, without labels. A page it rebuilds badly is unusual.
The threshold is set on validation pages (95th percentile of their error).
"""
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fictional_data import RESULTS, save_json
from page_images import draw_page, draw_unusual, make_set, scan


class AutoEncoder(nn.Module):
    def __init__(self, code=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
                                 nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                                 nn.Flatten(), nn.Linear(32 * 8 * 8, code))
        self.dec = nn.Sequential(nn.Linear(code, 32 * 8 * 8), nn.ReLU(), nn.Unflatten(1, (32, 8, 8)),
                                 nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
                                 nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1), nn.Sigmoid())

    def forward(self, x):
        return self.dec(self.enc(x))


@torch.no_grad()
def errors(model, x):
    model.eval()
    return ((model(x) - x) ** 2).mean(dim=(1, 2, 3)).numpy()


def main():
    torch.manual_seed(0)
    x_train = torch.tensor(make_set(draw_page, 3, 300, seed=21)[0]).unsqueeze(1)
    x_val = torch.tensor(make_set(draw_page, 3, 60, seed=22)[0]).unsqueeze(1)
    x_usual = torch.tensor(make_set(draw_page, 3, 100, seed=23)[0]).unsqueeze(1)
    rng = random.Random(24)
    x_unusual = torch.tensor(np.stack([scan(draw_unusual(rng), rng) for _ in range(60)])).unsqueeze(1)
    model = AutoEncoder()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    history = []
    for epoch in range(1, 21):
        model.train()
        order = torch.randperm(len(x_train))
        for i in range(0, len(order), 32):
            b = x_train[order[i:i + 32]]
            loss = F.mse_loss(model(b), b)          # the target is the input itself: no labels
            opt.zero_grad()
            loss.backward()
            opt.step()
        history.append({"epoch": epoch, "val_reconstruction_error": round(float(errors(model, x_val).mean()), 5)})
    threshold = float(np.percentile(errors(model, x_val), 95))
    e_usual, e_unusual = errors(model, x_usual), errors(model, x_unusual)
    result = {"threshold_95th_percentile": round(threshold, 5),
              "usual_pages_flagged": round(float((e_usual > threshold).mean()), 3),
              "unusual_pages_flagged": round(float((e_unusual > threshold).mean()), 3),
              "history": history}
    print("Threshold:", result["threshold_95th_percentile"], "| usual pages flagged:",
          result["usual_pages_flagged"], "| unusual pages flagged:", result["unusual_pages_flagged"])
    print("Action: a flagged page goes to a person before any automatic classification.")
    save_json("17_autoencoder.json", result)
    plot(e_usual, e_unusual, threshold)
    return result


def plot(e_usual, e_unusual, threshold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(e_usual, bins=30, alpha=0.7, label="usual pages (test)")
    ax.hist(e_unusual, bins=30, alpha=0.7, label="unusual pages")
    ax.axvline(threshold, color="crimson", linestyle="--", label="threshold from validation")
    ax.set_xlabel("reconstruction error")
    ax.set_ylabel("pages")
    ax.set_title("Autoencoder: unusual pages rebuild badly")
    ax.legend(fontsize=8)
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "17_autoencoder.png", dpi=130)


if __name__ == "__main__":
    main()
