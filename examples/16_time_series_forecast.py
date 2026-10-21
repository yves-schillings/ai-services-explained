"""Time-dependent prediction: forecast tomorrow's processing workload with a temporal split.

Author: Yves Schillings.

Run:  python examples/16_time_series_forecast.py

Need: a service manager plans staff and compute for the next day. The data is a
daily series, so the evaluation must respect time: train on the past, validate on
the following weeks, test on the most recent weeks. Every feature of day d must be
known before day d. A centred moving average breaks that rule and looks excellent.
"""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from fictional_data import RESULTS, save_json


def make_series(days=730, seed=4):
    rng = np.random.default_rng(seed)
    t = np.arange(days)
    weekly = np.array([1.15, 1.1, 1.05, 1.0, 0.95, 0.45, 0.35])[t % 7]
    trend = 1 + 0.0006 * t
    yearly = 1 + 0.12 * np.sin(2 * np.pi * t / 365)
    level = 400 * weekly * trend * yearly
    shocks = np.where(rng.random(days) < 0.02, rng.uniform(1.3, 1.8, days), 1.0)
    return np.round(level * shocks * rng.lognormal(0, 0.06, days)).astype(float)


def features(y):
    """Each row describes day d using only information known before day d."""
    rows, target = [], []
    for d in range(28, len(y)):
        rows.append([d % 7, y[d - 1], y[d - 7], y[d - 14], y[d - 7:d].mean(), y[d - 28:d].mean()])
        target.append(y[d])
    return np.array(rows), np.array(target)


def main():
    y = make_series()
    x, target = features(y)
    n = len(target)
    tr, va = int(n * 0.70), int(n * 0.85)       # past | following weeks | most recent weeks
    model = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0)
    model.fit(x[:tr], target[:tr])
    seasonal_naive_val = x[tr:va, 2]             # same weekday last week
    res = {"days": int(len(y)), "train_days": tr, "val_days": va - tr, "test_days": n - va,
           "validation_mae": {"same_day_last_week": round(float(mean_absolute_error(target[tr:va],
                                                                                    seasonal_naive_val)), 1),
                              "gradient_boosting": round(float(mean_absolute_error(target[tr:va],
                                                                                   model.predict(x[tr:va]))), 1)}}
    model.fit(x[:va], target[:va])               # refit on train + validation before the final test
    pred_test = model.predict(x[va:])
    res["test_mae_once"] = {"same_day_last_week": round(float(mean_absolute_error(target[va:], x[va:, 2])), 1),
                            "gradient_boosting": round(float(mean_absolute_error(target[va:], pred_test)), 1)}
    # A classic leak: a centred 7-day mean also contains the day being predicted.
    # It looks excellent in testing and is impossible to compute in production.
    leaky_x = np.column_stack([x, [y[d - 3:d + 4].mean() for d in range(28, len(y))]])
    leaky = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0)
    leaky.fit(leaky_x[:va], target[:va])
    res["leaky_feature_test_mae"] = round(float(mean_absolute_error(target[va:], leaky.predict(leaky_x[va:]))), 1)
    print("Validation MAE (documents per day):", res["validation_mae"])
    print("Final test MAE, most recent weeks:", res["test_mae_once"])
    print("Test MAE with a leaky centred mean (impossible in production):", res["leaky_feature_test_mae"])
    print("Action: the forecast supports planning. A manager adjusts it for known events.")
    save_json("16_time_series_forecast.json", res)
    plot(target[va:], pred_test, x[va:, 2])
    return res


def plot(actual, pred, naive):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(actual, label="actual", linewidth=1.5)
    ax.plot(pred, label="gradient boosting forecast", linewidth=1.2)
    ax.plot(naive, label="same day last week", linewidth=1, alpha=0.6)
    ax.set_xlabel("day in the final test period")
    ax.set_ylabel("documents per day")
    ax.set_title("Temporal test: train on the past, test on the most recent weeks")
    ax.legend(fontsize=8)
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "16_time_series_forecast.png", dpi=130)


if __name__ == "__main__":
    main()
