"""Train a tiny text classifier on fictional service tickets.

Run: python "01_machine_learning.py"
This is a real fitted multinomial naive Bayes model, not keyword rules.
The tiny holdout illustrates evaluation, not an enterprise accuracy claim.
"""
import math, re, json
from collections import Counter

def words(text):
    return re.findall(r"[a-z]+", text.lower())

def train(rows):
    counts, totals, prior, vocabulary = {}, {}, Counter(), set()
    for text, label in rows:
        prior[label] += 1
        counts.setdefault(label, Counter()).update(words(text))
        vocabulary.update(words(text))
    for label in prior:
        totals[label] = sum(counts[label].values())
    return counts, totals, prior, vocabulary

def predict(model, text):
    counts, totals, prior, vocabulary = model
    if not vocabulary:
        raise ValueError("Empty training data")
    tokens = [word for word in words(text) if word in vocabulary]
    scores = {label: math.log(n / sum(prior.values())) + sum(
        math.log((counts[label][word] + 1) / (totals[label] + len(vocabulary)))
        for word in tokens) for label, n in prior.items()}
    return max(scores, key=scores.get)

TRAIN = [("invoice payment amount", "billing"),
         ("incorrect invoice payment", "billing"),
         ("refund invoice amount", "billing"),
         ("network outage connection", "technical"),
         ("connection failure router", "technical"),
         ("network router outage", "technical")]
TEST = [("invoice refund", "billing"), ("router connection", "technical")]
if __name__ == "__main__":
    model = train(TRAIN)
    predictions = [predict(model, text) for text, _ in TEST]
    print(json.dumps({"fictional_holdout_predictions": predictions,
                      "correct": sum(p == y for p, (_, y) in zip(predictions, TEST)),
                      "holdout_size": len(TEST), "production_validated": False}))
