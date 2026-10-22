"""Train, evaluate, register and monitor a document-triage classifier.

Author: Yves Schillings.

Run:  python examples/01_machine_learning.py
Then: python examples/19_model_release.py    (the gate and a named owner decide)
      python examples/18_model_serving.py    (serves the approved version)

The model proposes the type of each incoming document so that it reaches the
right reviewer. It is a scikit-learn Pipeline with two fitted steps:
  * TfidfVectorizer learns the vocabulary and one IDF weight per word,
  * LogisticRegression learns one coefficient per word and document type.
Only the training set changes those learned values. The validation set chooses
the hyperparameters. The final test set is used once per scenario, at the end.

Data (all fictional, see fictional_data.py):
  * three labelled pools of 100, 200 and 400 cases, each pool containing the
    previous one. Each pool is split by case: 80 % training, 20 % validation;
  * ONE fixed final test set of 100 other cases, generated separately. It is
    never used for training or tuning, so every scenario is judged on exactly
    the same documents.

Two baselines show what the model must beat: always proposing the most frequent
type, and a keyword rule a person could write without any training.

MLflow records each scenario as a run (parameters, data version, metrics,
curves, the saved model and its exact dependencies) and registers the model as
a new version of "document-triage". MLflow keeps the information. It does not
approve anything: that is the release step (19_model_release.py).
"""
import json
from collections import Counter

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from fictional_data import (DOC_TYPES, FINAL_TEST_CASES, RESULTS, make_case_documents, make_final_test_set,
                             run_environment, save_json, sha256_json, split_at_random, split_by_group)
from mlflow_tracking import pinned, use_experiment

ACCEPTANCE_MACRO_F1 = 0.85     # agreed with the business owner before training
REVIEW_THRESHOLD = 0.50        # below this confidence a person types the document
GRID_C = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
POOLS = (100, 200, 400)        # labelled cases; each pool contains the previous one
MODEL_NAME = "document-triage"
EXPERIMENT = "document-triage"
# Fictional cost assumptions, to agree with the owner: a person checks and routes one document
# in 2 minutes; a misrouted document that nobody checked costs 20 minutes when it is found later.
REVIEW_MINUTES = 2.0
ERROR_MINUTES = 20.0
# A rule a person could write from the typical words of each type, without any training.
KEYWORDS = {"meeting_minutes": ["worker", "speaker", "decides"],
            "incident_report": ["operator", "parking", "incident"],
            "item_inventory": ["extinguisher", "shelved", "inventory"],
            "correspondence": ["dear", "regards", "letter"]}


def build_pipeline(c=1.0, class_weight=None):
    """The whole fitted object that is saved and served: preprocessing and model together."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(C=c, class_weight=class_weight, max_iter=2000)),
    ])


def xy(rows):
    return [r["text"] for r in rows], [r["label"] for r in rows]


def macro_f1(model, rows):
    x, y = xy(rows)
    return f1_score(y, model.predict(x), average="macro")


def select_hyperparameters(train, val):
    """Fit one candidate per setting on TRAIN, score it on VALIDATION. Nothing touches TEST here."""
    curve = []
    for cw in (None, "balanced"):
        for c in GRID_C:                                     # 0.03 ... 30
            m = build_pipeline(c, cw).fit(*xy(train))        # learns on TRAIN
            curve.append({"C": c, "class_weight": cw or "none", "train_macro_f1": macro_f1(m, train),
                          "val_macro_f1": macro_f1(m, val)})  # judged on VALIDATION
    best = max(curve, key=lambda r: (round(r["val_macro_f1"], 4), -r["C"]))   # ties: smaller C
    return best, curve


def predict_with_review(model, texts, threshold=REVIEW_THRESHOLD):
    """Inference as the service does it: a label, a confidence and a review flag."""
    proba = model.predict_proba(texts)
    labels = model.classes_[proba.argmax(axis=1)]
    conf = proba.max(axis=1)
    return [{"label": str(lab), "confidence": round(float(p), 4), "requires_review": bool(p < threshold)}
            for lab, p in zip(labels, conf)]


def leakage_experiment(docs, best):
    """Inside one pool, the same pipeline with two ways of splitting it into train, validation and test.
    The random split lets case words leak into its test part. The fixed final test set is not used here."""
    out = {}
    cw = None if best["class_weight"] == "none" else best["class_weight"]
    for name, parts in (("split_by_case", split_by_group(docs, "case_id")),
                        ("random_split", split_at_random(docs))):
        m = build_pipeline(best["C"], cw)
        m.fit(*xy(parts["train"]))
        out[name] = round(macro_f1(m, parts["test"]), 4)
    # Second leakage: fitting the vectoriser on all documents, including the test set.
    parts = split_by_group(docs, "case_id")
    leaky_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(
        [r["text"] for r in docs])                    # wrong: test documents shape the vocabulary
    clf = LogisticRegression(C=best["C"], class_weight=cw, max_iter=2000).fit(
        leaky_vec.transform(xy(parts["train"])[0]), xy(parts["train"])[1])
    x_test, y_test = xy(parts["test"])
    out["vectoriser_fitted_on_all_data"] = round(f1_score(y_test, clf.predict(leaky_vec.transform(x_test)),
                                                          average="macro"), 4)
    return out


def drifted_batch(seed=99, n=200):
    """After deployment: a new source writes documents with words the model never saw."""
    rng = np.random.default_rng(seed)
    new_words = ["portal", "ticket", "upload", "form", "submission", "online", "reference", "status"]
    docs = make_case_documents(n_cases=40, seed=seed)[:n]
    for d in docs:
        extra = list(rng.choice(new_words, size=6))
        words = d["text"].split()[: max(3, len(d["text"].split()) // 3)] + extra
        d["text"] = " ".join(words)
    return docs


def keyword_rule(text, default):
    """The type whose keywords appear most often in the text, or the default type."""
    words = set(text.split())
    hits = {t: len(words & set(k)) for t, k in KEYWORDS.items()}
    best = max(hits, key=hits.get)
    return best if hits[best] else default


def baselines(train, final_test):
    """What the model must beat, scored on the same final test set."""
    x, y = xy(train)
    x_test, y_test = xy(final_test)
    dummy = DummyClassifier(strategy="most_frequent").fit(x, y)
    default = str(dummy.predict(["any"])[0])
    rule = [keyword_rule(t, default) for t in x_test]
    return {"most_frequent_type": round(f1_score(y_test, dummy.predict(x_test), average="macro",
                                                 zero_division=0), 4),
            "keyword_rule": round(f1_score(y_test, rule, average="macro", zero_division=0), 4)}


def macro_f1_interval(y_true, y_pred, cases, n=1000, seed=5):
    """95 % interval of the test macro-F1. Whole cases are resampled, because documents of one case are alike."""
    rng = np.random.default_rng(seed)
    groups = {}
    for i, c in enumerate(cases):
        groups.setdefault(c, []).append(i)
    groups = list(groups.values())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    scores = []
    for _ in range(n):
        idx = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        scores.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
    return [round(float(np.percentile(scores, 2.5)), 4), round(float(np.percentile(scores, 97.5)), 4)]


def review_cost(model, x, y, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    """Minutes per 1.000 documents: people check the uncertain ones, errors among the others cost later."""
    proba = model.predict_proba(x)
    labels = model.classes_[proba.argmax(axis=1)]
    conf = proba.max(axis=1)
    wrong = labels != np.array(y)
    rows = []
    for t in thresholds:
        review = conf < t
        errors = int((wrong & ~review).sum())
        minutes = REVIEW_MINUTES * review.sum() + ERROR_MINUTES * errors
        rows.append({"threshold": t, "share_sent_to_review": round(float(review.mean()), 4),
                     "errors_not_reviewed": errors, "minutes_per_1000": round(float(minutes) / len(y) * 1000, 1)})
    return {"review_minutes": REVIEW_MINUTES, "error_minutes": ERROR_MINUTES, "documents": len(y),
            "everything_by_hand_minutes_per_1000": REVIEW_MINUTES * 1000,
            "no_review_minutes_per_1000": round(ERROR_MINUTES * float(wrong.sum()) / len(y) * 1000, 1),
            "by_threshold": rows}


def train_and_evaluate(docs, final_test):
    """Steps 3 to 7 for one labelled pool: split by case, tune on validation, test once, check leakage."""
    parts = split_by_group(docs, "case_id", fractions=(0.8, 0.2, 0.0))  # no case in two sets
    assert not parts["test"], "a pool holds training and validation cases only"
    sizes = {"train": len(parts["train"]), "val": len(parts["val"]), "final_test": len(final_test)}
    shared = {r["case_id"] for r in docs} & {r["case_id"] for r in final_test}
    best, curve = select_hyperparameters(parts["train"], parts["val"])
    cw = None if best["class_weight"] == "none" else best["class_weight"]
    model = build_pipeline(best["C"], cw).fit(*xy(parts["train"]))  # learns on TRAIN only
    x_test, y_test = xy(final_test)
    pred = model.predict(x_test)                                    # the test set, used once
    test = {"accuracy": round(accuracy_score(y_test, pred), 4),
            "macro_f1": round(f1_score(y_test, pred, average="macro"), 4),
            "macro_f1_interval_95": macro_f1_interval(y_test, pred, [r["case_id"] for r in final_test]),
            "per_class_f1": {k: round(v["f1-score"], 4) for k, v in
                             classification_report(y_test, pred, output_dict=True).items() if k in DOC_TYPES},
            "confusion_matrix": confusion_matrix(y_test, pred, labels=DOC_TYPES).tolist(),
            "labels": DOC_TYPES}
    review = predict_with_review(model, x_test)
    kept = [(p, y) for p, y in zip(review, y_test) if not p["requires_review"]]
    test["share_sent_to_review"] = round(1 - len(kept) / len(review), 4)
    test["accuracy_on_kept"] = round(sum(p["label"] == y for p, y in kept) / max(1, len(kept)), 4)
    test["passes_acceptance"] = test["macro_f1"] >= ACCEPTANCE_MACRO_F1
    leak = leakage_experiment(docs, best)
    return {"model": model, "parts": parts, "sizes": sizes, "shared_cases": len(shared), "best": best,
            "curve": curve, "test": test, "leakage": leak, "baselines": baselines(parts["train"], final_test),
            "review_cost": review_cost(model, x_test, y_test),
            "learned": {"vocabulary_terms": len(model.named_steps["tfidf"].vocabulary_),
                        "logistic_coefficients": int(model.named_steps["clf"].coef_.size
                                                     + model.named_steps["clf"].intercept_.size)}}


def log_to_mlflow(n_cases, r, docs, final_test):
    """Record one scenario as an MLflow run and register its model as a new version."""
    with mlflow.start_run(run_name=f"pool-{n_cases}-cases") as run:
        mlflow.log_params({"labelled_cases": n_cases, "documents_train": r["sizes"]["train"],
                           "documents_val": r["sizes"]["val"], "split": "by case, 80/20, seed 11",
                           "final_test_cases": FINAL_TEST_CASES, "C": r["best"]["C"],
                           "class_weight": r["best"]["class_weight"], "ngram_range": "1-2", "min_df": 2,
                           "review_threshold": REVIEW_THRESHOLD, "acceptance_macro_f1": ACCEPTANCE_MACRO_F1})
        mlflow.set_tags({"data_version": sha256_json(docs)[:16], "final_test_version": sha256_json(final_test)[:16],
                         "gate": "passed" if r["test"]["passes_acceptance"] else "failed"})
        rows = [c for c in r["curve"] if c["class_weight"] == r["best"]["class_weight"]]
        for step, point in enumerate(rows):                          # the validation curve, one step per C
            mlflow.log_metric("train_macro_f1_by_C", point["train_macro_f1"], step=step)
            mlflow.log_metric("val_macro_f1_by_C", point["val_macro_f1"], step=step)
        mlflow.log_metrics({"val_macro_f1": r["best"]["val_macro_f1"], "test_macro_f1": r["test"]["macro_f1"],
                            "test_accuracy": r["test"]["accuracy"],
                            "share_sent_to_review": r["test"]["share_sent_to_review"],
                            "accuracy_on_kept": r["test"]["accuracy_on_kept"],
                            "baseline_most_frequent_macro_f1": r["baselines"]["most_frequent_type"],
                            "baseline_keyword_rule_macro_f1": r["baselines"]["keyword_rule"],
                            "leak_split_by_case_macro_f1": r["leakage"]["split_by_case"],
                            "leak_random_split_macro_f1": r["leakage"]["random_split"]})
        mlflow.log_dict({"labels": DOC_TYPES, "confusion_matrix": r["test"]["confusion_matrix"]},
                        "confusion_matrix.json")
        info = mlflow.sklearn.log_model(sk_model=r["model"], name="model", registered_model_name=MODEL_NAME,
                                        pip_requirements=pinned("scikit-learn", "numpy", "scipy", "joblib"),
                                        await_registration_for=0)
    return run.info.run_id, str(info.registered_model_version)


def main():
    print("1. Need: propose a document type. Acceptance: macro-F1 >=", ACCEPTANCE_MACRO_F1,
          "on the final test set, and low-confidence documents go to a person.")
    client = use_experiment(EXPERIMENT)
    final_test = make_final_test_set()
    print("2. Fixed final test set:", len(final_test), "documents in", FINAL_TEST_CASES,
          "cases, never used for training or tuning.")
    runs = {}
    for n_cases in POOLS:
        docs = make_case_documents(n_cases=n_cases)
        if n_cases == POOLS[0]:
            print("   Example document:", json.dumps(docs[0]))
            print("   Class mix:", dict(Counter(d["label"] for d in docs)))
        r = train_and_evaluate(docs, final_test)
        r["run_id"], r["registered_version"] = log_to_mlflow(n_cases, r, docs, final_test)
        client.set_model_version_tag(MODEL_NAME, r["registered_version"], "labelled_cases", str(n_cases))
        client.set_model_version_tag(MODEL_NAME, r["registered_version"], "gate",
                                     "passed" if r["test"]["passes_acceptance"] else "failed")
        r["docs"] = docs
        runs[n_cases] = r
        print(f"--- pool of {n_cases} labelled cases ({len(docs)} documents)")
        print("3. Split by case:", r["sizes"], "- cases shared with the final test set:", r["shared_cases"])
        print("4-5. Hyperparameters chosen on validation:", r["best"])
        print("   Learned parameters:", r["learned"])
        print("6. Final test, used once:", {k: v for k, v in r["test"].items()
                                           if k not in ("confusion_matrix", "labels", "per_class_f1")})
        print("   Baselines on the same test set (macro-F1):", r["baselines"])
        print("7. Leakage check inside the pool, test macro-F1:", r["leakage"])
        print("8. MLflow run", r["run_id"][:8], "registered as version", r["registered_version"], "of", MODEL_NAME,
              "- gate", "passed" if r["test"]["passes_acceptance"] else "failed")

    final = runs[POOLS[-1]]
    test = final["test"]
    print("   Review cost with the largest pool (minutes per 1.000 documents):",
          {row["threshold"]: row["minutes_per_1000"] for row in final["review_cost"]["by_threshold"]},
          "- everything by hand:", final["review_cost"]["everything_by_hand_minutes_per_1000"])

    loaded = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{final['registered_version']}")
    sample = predict_with_review(loaded, ["worker saw a person near the hall in the evening"])
    print("9. Inference from the registered version:", sample)

    x_test = xy(final_test)[0]
    drift = drifted_batch()
    on_test = predict_with_review(loaded, x_test)
    on_new = predict_with_review(loaded, [d["text"] for d in drift])
    mon = {"mean_confidence_test": round(float(np.mean([p["confidence"] for p in on_test])), 4),
           "mean_confidence_new_batch": round(float(np.mean([p["confidence"] for p in on_new])), 4),
           "review_share_test": round(float(np.mean([p["requires_review"] for p in on_test])), 4),
           "review_share_new_batch": round(float(np.mean([p["requires_review"] for p in on_new])), 4)}
    # Rule agreed with the owner: investigate when the review share rises by more than 10 points.
    mon["investigate"] = mon["review_share_new_batch"] - mon["review_share_test"] > 0.10
    print("10. Monitoring after deployment:", mon)

    summary = {str(n): {k: r[k] for k in ("sizes", "best", "curve", "test", "leakage", "learned", "baselines",
                                          "review_cost", "run_id", "registered_version")}
               for n, r in runs.items()}
    save_json("01_machine_learning.json", {
        "model_name": MODEL_NAME, "acceptance_macro_f1": ACCEPTANCE_MACRO_F1, "review_threshold": REVIEW_THRESHOLD,
        "final_test": {"cases": FINAL_TEST_CASES, "documents": len(final_test),
                       "version": sha256_json(final_test)[:16]},
        "runs": summary, "monitoring": mon, "environment": run_environment(),
        "command": "python examples/01_machine_learning.py"})
    plot(runs)
    return test


def plot(runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    last = runs[POOLS[-1]]
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.6))
    ax = ax.ravel()
    rows = [r for r in last["curve"] if r["class_weight"] == last["best"]["class_weight"]]
    ax[0].semilogx([r["C"] for r in rows], [r["train_macro_f1"] for r in rows], "o-", label="training set")
    ax[0].semilogx([r["C"] for r in rows], [r["val_macro_f1"] for r in rows], "s-", label="validation set")
    ax[0].set_xlabel("C (inverse regularisation strength)")
    ax[0].set_ylabel("macro-F1")
    ax[0].set_title(f"Choose C on validation data ({POOLS[-1]} cases)")
    ax[0].legend()
    scores = [runs[n]["test"]["macro_f1"] for n in POOLS]
    low = [s - runs[n]["test"]["macro_f1_interval_95"][0] for s, n in zip(scores, POOLS)]
    high = [runs[n]["test"]["macro_f1_interval_95"][1] - s for s, n in zip(scores, POOLS)]
    ax[1].errorbar(range(len(POOLS)), scores, yerr=[low, high], fmt="o-", capsize=5, label="model, 95 % interval")
    ax[1].axhline(ACCEPTANCE_MACRO_F1, color="crimson", linestyle="--", label="acceptance 0.85")
    ax[1].axhline(last["baselines"]["keyword_rule"], color="#e66a00", linestyle=":", label="keyword rule")
    ax[1].axhline(last["baselines"]["most_frequent_type"], color="grey", linestyle=":", label="most frequent type")
    for i, v in enumerate(scores):
        ax[1].text(i + 0.08, v, f"{v:.3f}", fontsize=8, va="center")
    ax[1].set_xticks(range(len(POOLS)), [f"{n} cases" for n in POOLS])
    ax[1].set_ylim(0, 1.0)
    ax[1].set_ylabel("macro-F1 on the fixed final test set")
    ax[1].set_title("More labelled cases, same final test set")
    ax[1].legend(fontsize=8, loc="lower right")
    leak_pools = [n for n in POOLS if n != POOLS[0]]
    honest = [runs[n]["leakage"]["split_by_case"] for n in leak_pools]
    leaky = [runs[n]["leakage"]["random_split"] for n in leak_pools]
    xs = np.arange(len(leak_pools))
    ax[2].bar(xs - 0.18, honest, 0.36, label="split by case (honest)")
    ax[2].bar(xs + 0.18, leaky, 0.36, label="random split (leaky)")
    for x, v in list(zip(xs - 0.18, honest)) + list(zip(xs + 0.18, leaky)):
        ax[2].text(x, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax[2].set_xticks(xs, [f"{n} cases" for n in leak_pools])
    ax[2].set_ylim(0.7, 1.0)
    ax[2].set_ylabel("test macro-F1 inside the pool")
    ax[2].set_title("Leakage inflates the score")
    ax[2].legend(fontsize=8, loc="upper left")
    cm = np.array(last["test"]["confusion_matrix"])
    ax[3].imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax[3].text(j, i, cm[i, j], ha="center", va="center")
    short = [t.replace("_", " ") for t in DOC_TYPES]
    ax[3].set_xticks(range(len(short)), short, fontsize=7, rotation=20)
    ax[3].set_yticks(range(len(short)), short, fontsize=7)
    ax[3].set_xlabel("predicted")
    ax[3].set_ylabel("true")
    ax[3].set_title(f"Final test confusion matrix ({POOLS[-1]} cases)")
    cost = last["review_cost"]
    ts = [r["threshold"] for r in cost["by_threshold"]]
    ax[4].plot(ts, [r["minutes_per_1000"] for r in cost["by_threshold"]], "o-", label="model + review")
    ax[4].axhline(cost["everything_by_hand_minutes_per_1000"], color="grey", linestyle="--", label="everything by hand")
    ax[4].set_xlabel("review threshold (confidence)")
    ax[4].set_ylabel("minutes per 1.000 documents")
    ax[4].set_title("Cost of review and of unchecked errors")
    ax[4].legend(fontsize=8)
    ax[5].plot(ts, [r["share_sent_to_review"] for r in cost["by_threshold"]], "o-", label="share sent to review")
    ax5 = ax[5].twinx()
    ax5.plot(ts, [r["errors_not_reviewed"] for r in cost["by_threshold"]], "s--", color="#d73285",
             label="errors nobody checked")
    ax[5].set_xlabel("review threshold (confidence)")
    ax[5].set_ylabel("share sent to review")
    ax5.set_ylabel(f"errors among {cost['documents']} test documents")
    ax[5].set_title("Review share versus unchecked errors")
    ax[5].legend(fontsize=8, loc="center left")
    ax5.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "01_machine_learning.png", dpi=130)


if __name__ == "__main__":
    main()
