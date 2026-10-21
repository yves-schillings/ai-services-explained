"""Anomaly detection: flag unusual processing activity for a person to review.

Author: Yves Schillings.

Run:  python examples/15_anomaly_detection.py

Need: an operations team wants a short daily list of unusual activity in a
processing platform, such as very large exports at night. There are no labels,
so an unsupervised Isolation Forest learns what usual activity looks like.

An anomaly is a statistical oddity. It is never evidence of wrongdoing.
The fictional data contains a few injected unusual records, so the example can
measure how many of them reach the top of the review list. Real data has no
such answer key.
"""
import numpy as np
from sklearn.ensemble import IsolationForest

from fictional_data import RESULTS, save_json

FEATURES = ["hour", "documents_opened", "export_mb", "failed_logins"]


def make_activity(n=5000, n_unusual=50, seed=3):
    rng = np.random.default_rng(seed)
    hour = np.clip(rng.normal(13, 2.5, n), 6, 20)
    opened = rng.poisson(25, n)
    export = rng.gamma(2.0, 4.0, n)
    failed = rng.poisson(0.2, n)
    x = np.column_stack([hour, opened, export, failed]).astype(float)
    unusual = np.column_stack([rng.uniform(0, 5, n_unusual), rng.poisson(120, n_unusual),
                               rng.gamma(9.0, 20.0, n_unusual), rng.poisson(3, n_unusual)])
    x = np.vstack([x, unusual])
    truth = np.r_[np.zeros(n), np.ones(n_unusual)]
    order = rng.permutation(len(x))
    return x[order], truth[order]


def main():
    x, truth = make_activity()
    half = len(x) // 2
    model = IsolationForest(n_estimators=200, contamination="auto", random_state=0).fit(x[:half])
    score = -model.score_samples(x[half:])           # higher means more unusual
    t = truth[half:]
    order = np.argsort(-score)
    result = {"records_scored": int(len(t)), "unusual_present": int(t.sum())}
    for k in (25, 50, 100):
        result[f"precision_at_{k}"] = round(float(t[order[:k]].mean()), 3)
        result[f"recall_at_{k}"] = round(float(t[order[:k]].sum() / t.sum()), 3)
    top = x[half:][order[:3]]
    result["top_three"] = [dict(zip(FEATURES, map(lambda v: round(float(v), 1), r))) for r in top]
    print("Fitted on", half, "records without labels. Scored", len(t), "new records.")
    print("Review list quality on the injected records:", {k: v for k, v in result.items() if "at" in k})
    print("Top three for review:", result["top_three"])
    print("Action: an operator reviews the list. An anomaly is not evidence and triggers no automatic step.")
    save_json("15_anomaly_detection.json", result)
    plot(score, t)
    return result


def plot(score, t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(score[t == 0], bins=60, alpha=0.7, label="usual records")
    ax.hist(score[t == 1], bins=20, alpha=0.8, label="injected unusual records")
    ax.set_xlabel("anomaly score (higher is more unusual)")
    ax.set_ylabel("records")
    ax.set_title("Isolation Forest scores on new records")
    ax.legend()
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "15_anomaly_detection.png", dpi=130)


if __name__ == "__main__":
    main()
