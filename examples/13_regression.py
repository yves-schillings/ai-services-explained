"""Regression: predict how long a processing job will take, from fictional workload records.

Author: Yves Schillings.

Run:  python examples/13_regression.py

Need: a planner wants an estimate of processing minutes per job to schedule the
queue and size the workers. The target is a number, so this is regression.
Inputs per job: pages, share of scanned pages, audio minutes, language, priority.

The example compares a constant baseline, a linear model and gradient-boosted trees,
and shows underfitting and overfitting with the depth of the trees.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from fictional_data import RESULTS, save_json

FEATURES = ["pages", "scanned_share", "audio_minutes", "language", "priority"]


def make_jobs(n=3000, seed=5):
    """Fictional jobs. Duration grows with pages, more for scanned pages, and with audio length."""
    rng = np.random.default_rng(seed)
    pages = rng.integers(1, 300, n)
    scanned = rng.beta(2, 3, n)
    audio = rng.choice([0, 0, 0, 5, 15, 30, 60, 120], n) * rng.uniform(0.5, 1.5, n)
    language = rng.choice(["fr", "nl", "en"], n, p=[0.45, 0.45, 0.10])
    priority = rng.choice(["normal", "urgent"], n, p=[0.8, 0.2])
    lang_cost = np.select([language == "nl", language == "en"], [1.15, 0.9], 1.0)
    minutes = (2 + 0.08 * pages * (1 + 3 * scanned ** 2) + 0.6 * audio * lang_cost
               + np.where(priority == "urgent", -3, 0))
    minutes = minutes * rng.lognormal(0, 0.15, n)          # unexplained variation
    rows = [{"pages": int(p), "scanned_share": round(float(s), 3), "audio_minutes": round(float(a), 1),
             "language": str(l), "priority": str(q), "minutes": round(float(m), 2)}
            for p, s, a, l, q, m in zip(pages, scanned, audio, language, priority, minutes)]
    return rows


def matrix(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    return df[FEATURES], df["minutes"].to_numpy()


def preprocess():
    # The encoder learns the language and priority categories from the training set only.
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["language", "priority"])],
                             remainder="passthrough")


def scores(y, p):
    return {"mae_minutes": round(float(mean_absolute_error(y, p)), 2),
            "rmse_minutes": round(float(np.sqrt(mean_squared_error(y, p))), 2),
            "r2": round(float(r2_score(y, p)), 3)}


def main():
    rows = make_jobs()
    print("Example job:", rows[0])
    rng = np.random.default_rng(1)
    idx = rng.permutation(len(rows))       # jobs are independent here, so a random split is valid
    tr, va, te = idx[:2100], idx[2100:2550], idx[2550:]
    x_tr, y_tr = matrix([rows[i] for i in tr])
    x_va, y_va = matrix([rows[i] for i in va])
    x_te, y_te = matrix([rows[i] for i in te])

    baseline = np.full(len(y_va), y_tr.mean())
    linear = Pipeline([("prep", preprocess()), ("model", LinearRegression())]).fit(x_tr, y_tr)
    depth_curve = []
    for depth in (1, 2, 3, 4, 6, 8, 10):
        gbr = Pipeline([("prep", preprocess()),
                        ("model", GradientBoostingRegressor(max_depth=depth, n_estimators=200,
                                                            learning_rate=0.05, random_state=0))]).fit(x_tr, y_tr)
        depth_curve.append({"max_depth": depth,
                            "train_mae": scores(y_tr, gbr.predict(x_tr))["mae_minutes"],
                            "val_mae": scores(y_va, gbr.predict(x_va))["mae_minutes"]})
    best_depth = min(depth_curve, key=lambda r: r["val_mae"])["max_depth"]
    print("Validation MAE by tree depth:", depth_curve, "-> chosen depth", best_depth)
    final = Pipeline([("prep", preprocess()),
                      ("model", GradientBoostingRegressor(max_depth=best_depth, n_estimators=200,
                                                          learning_rate=0.05, random_state=0))]).fit(x_tr, y_tr)
    result = {"validation": {"constant_baseline": scores(y_va, baseline), "linear": scores(y_va, linear.predict(x_va)),
                             "gradient_boosting": scores(y_va, final.predict(x_va))},
              "test_once": {"gradient_boosting": scores(y_te, final.predict(x_te))},
              "depth_curve": depth_curve, "chosen_depth": best_depth}
    names = list(final.named_steps["prep"].get_feature_names_out())
    imp = final.named_steps["model"].feature_importances_
    result["feature_importance"] = {n.split("__")[-1]: round(float(v), 3)
                                    for n, v in sorted(zip(names, imp), key=lambda t: -t[1])}
    print("Validation scores:", result["validation"])
    print("Final test (once):", result["test_once"])
    print("Feature importance:", result["feature_importance"])
    print("Limit: the estimate supports planning. It never decides a priority for a person or a case.")
    save_json("13_regression.json", result)
    plot(depth_curve, y_te, final.predict(x_te))
    return result


def plot(curve, y, p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot([r["max_depth"] for r in curve], [r["train_mae"] for r in curve], "o-", label="training set")
    ax[0].plot([r["max_depth"] for r in curve], [r["val_mae"] for r in curve], "s-", label="validation set")
    ax[0].set_xlabel("tree depth (model capacity)")
    ax[0].set_ylabel("mean absolute error (minutes)")
    ax[0].set_title("Underfitting on the left, overfitting on the right")
    ax[0].legend()
    ax[1].scatter(y, p, s=6, alpha=0.5)
    lim = [0, max(y.max(), p.max())]
    ax[1].plot(lim, lim, "k--", linewidth=1)
    ax[1].set_xlabel("actual minutes (test set)")
    ax[1].set_ylabel("predicted minutes")
    ax[1].set_title("Final test: predicted versus actual")
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "13_regression.png", dpi=130)


if __name__ == "__main__":
    main()
