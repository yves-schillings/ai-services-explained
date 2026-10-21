"""Train, evaluate, package and monitor a document-triage classifier.

Author: Yves Schillings.

Run:  python examples/01_machine_learning.py

The model proposes the type of each incoming document so that it reaches the
right reviewer. It is a scikit-learn Pipeline with two fitted steps:
  * TfidfVectorizer learns the vocabulary and one IDF weight per word,
  * LogisticRegression learns one coefficient per word and document type.
Only the training set changes those learned values. The validation set chooses
the hyperparameters. The test set is used once, at the end.

All data is fictional (see training/common.py). The scores show the mechanism,
not production quality.
"""
import json
from collections import Counter

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from fictional_data import (ARTIFACTS, DOC_TYPES, RESULTS, make_case_documents, save_json, sha256_file,
                             sha256_json, split_at_random, split_by_group, write_manifest)

ACCEPTANCE_MACRO_F1 = 0.85     # agreed with the business owner before training
REVIEW_THRESHOLD = 0.50        # below this confidence a person types the document
GRID_C = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
ARTIFACT = ARTIFACTS / "document_triage.joblib"
MANIFEST = ARTIFACTS / "document_triage.manifest.json"


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
    """Same pipeline, two ways of splitting. The random split lets case words leak into the test set."""
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


def train_and_evaluate(docs):
    """Steps 3 to 7 for one dataset: split by case, tune on validation, test once, check leakage."""
    parts = split_by_group(docs, "case_id")                         # no case in two sets
    sizes = {k: len(v) for k, v in parts.items()}
    shared = {r["case_id"] for r in parts["train"]} & {r["case_id"] for r in parts["test"]}
    best, curve = select_hyperparameters(parts["train"], parts["val"])
    cw = None if best["class_weight"] == "none" else best["class_weight"]
    model = build_pipeline(best["C"], cw).fit(*xy(parts["train"]))  # learns on TRAIN only
    x_test, y_test = xy(parts["test"])
    pred = model.predict(x_test)                                    # the test set, used once
    test = {"accuracy": round(accuracy_score(y_test, pred), 4),
            "macro_f1": round(f1_score(y_test, pred, average="macro"), 4),
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
            "curve": curve, "test": test, "leakage": leak,
            "learned": {"vocabulary_terms": len(model.named_steps["tfidf"].vocabulary_),
                        "logistic_coefficients": int(model.named_steps["clf"].coef_.size
                                                     + model.named_steps["clf"].intercept_.size)}}


def main():
    print("1. Need: propose a document type. Acceptance: macro-F1 >=", ACCEPTANCE_MACRO_F1,
          "on the final test set, and low-confidence documents go to a person.")
    runs = {}
    for n_cases in (200, 400):
        docs = make_case_documents(n_cases=n_cases)
        if n_cases == 200:
            print("2. Data:", len(docs), "fictional documents in", n_cases, "cases. Example:", json.dumps(docs[0]))
            print("   Class mix:", dict(Counter(d["label"] for d in docs)))
        r = train_and_evaluate(docs)
        r["docs"] = docs
        runs[n_cases] = r
        print(f"--- {n_cases} labelled cases ({len(docs)} documents)")
        print("3. Split by case:", r["sizes"], "- cases shared between train and test:", r["shared_cases"])
        print("4-5. Hyperparameters chosen on validation:", r["best"])
        print("   Learned parameters:", r["learned"])
        print("6. Final test, used once:", {k: v for k, v in r["test"].items() if k != "confusion_matrix"})
        print("7. Leakage check, test macro-F1:", r["leakage"])
        print("   Release decision:", "candidate for release" if r["test"]["passes_acceptance"] else
              "refused: below the acceptance criterion, more labelled cases needed")

    final = runs[400]
    model, test, best, docs = final["model"], final["test"], final["best"], final["docs"]
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(model, ARTIFACT)                      # the whole fitted pipeline
    manifest = write_manifest(
        MANIFEST, model_name="document-triage", artifact=ARTIFACT.name, artifact_sha256=sha256_file(ARTIFACT),
        data_version=sha256_json(docs)[:16], labelled_cases=400, split="by case, 70/15/15, seed 11",
        hyperparameters={"C": best["C"], "class_weight": best["class_weight"], "ngram_range": [1, 2],
                         "min_df": 2}, metrics={"val_macro_f1": round(best["val_macro_f1"], 4), **test},
        review_threshold=REVIEW_THRESHOLD, acceptance_macro_f1=ACCEPTANCE_MACRO_F1,
        owner="document intake service owner (fictional)", release_decision="pending human review")
    print("8. Artifact:", ARTIFACT.name, manifest["artifact_sha256"][:16], "... manifest written.")

    loaded = joblib.load(ARTIFACT)                    # what the service loads at start-up
    sample = predict_with_review(loaded, ["witness saw a person near the harbour in the evening"])
    print("9. Inference from the saved artifact:", sample)

    x_test = xy(final["parts"]["test"])[0]
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

    summary = {str(n): {k: r[k] for k in ("sizes", "best", "curve", "test", "leakage", "learned")}
               for n, r in runs.items()}
    save_json("01_machine_learning.json", {"runs": summary, "monitoring": mon,
                                            "artifact_sha256": manifest["artifact_sha256"]})
    plot(final["curve"], test, runs)
    return test


def plot(curve, test, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in curve if r["class_weight"] == runs[400]["best"]["class_weight"]]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].semilogx([r["C"] for r in rows], [r["train_macro_f1"] for r in rows], "o-", label="training set")
    ax[0].semilogx([r["C"] for r in rows], [r["val_macro_f1"] for r in rows], "s-", label="validation set")
    ax[0].set_xlabel("C (inverse regularisation strength)")
    ax[0].set_ylabel("macro-F1")
    ax[0].set_title("Choose C on validation data (400 cases)")
    ax[0].legend()
    honest = [runs[n]["leakage"]["split_by_case"] for n in (200, 400)]
    leaky = [runs[n]["leakage"]["random_split"] for n in (200, 400)]
    xs = np.arange(2)
    ax[1].bar(xs - 0.18, honest, 0.36, label="split by case (honest)")
    ax[1].bar(xs + 0.18, leaky, 0.36, label="random split (leaky)")
    ax[1].axhline(ACCEPTANCE_MACRO_F1, color="crimson", linestyle="--", label="acceptance 0.85")
    for x, v in zip(xs - 0.18, honest):
        ax[1].text(x, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    for x, v in zip(xs + 0.18, leaky):
        ax[1].text(x, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax[1].set_xticks(xs, ["200 cases", "400 cases"])
    ax[1].set_ylim(0.7, 1.0)
    ax[1].set_ylabel("test macro-F1")
    ax[1].set_title("Leakage inflates the score")
    ax[1].legend(fontsize=8, loc="upper left")
    cm = np.array(test["confusion_matrix"])
    ax[2].imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax[2].text(j, i, cm[i, j], ha="center", va="center")
    short = [t.replace("_", " ") for t in DOC_TYPES]
    ax[2].set_xticks(range(len(short)), short, fontsize=7, rotation=20)
    ax[2].set_yticks(range(len(short)), short, fontsize=7)
    ax[2].set_xlabel("predicted")
    ax[2].set_ylabel("true")
    ax[2].set_title("Final test confusion matrix (400 cases)")
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "01_machine_learning.png", dpi=130)


if __name__ == "__main__":
    main()
