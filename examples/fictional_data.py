"""Shared helpers for the training examples: fictional data, splits and run evidence.

Author: Yves Schillings.

Every record produced here is invented. Nothing comes from a real organisation,
case or person. The generators are deterministic for a given seed, so a rerun
reproduces the same data and the same results.
"""
import hashlib
import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# ---------------------------------------------------------------------------
# Fictional case documents for text classification and clustering
# ---------------------------------------------------------------------------
DOC_TYPES = ["meeting_minutes", "incident_report", "item_inventory", "correspondence"]

# Typical words of each document type. Several words are shared between types,
# so the task is not trivially separable.
TYPE_WORDS = {
    "meeting_minutes": ["worker", "saw", "heard", "speaker", "decides", "remembers", "noticed",
                        "describes", "around", "person", "evening", "asked", "answered", "signed"],
    "incident_report": ["operator", "parking", "attended", "incident", "scaffold", "reported", "damage",
                        "response", "arrived", "unit", "time", "location", "observed", "signed"],
    "item_inventory": ["item", "extinguisher", "shelved", "bag", "photograph", "recording", "inventory",
                       "cupboard", "reference", "label", "stored", "collected", "number", "location"],
    "correspondence": ["dear", "colleague", "regards", "meeting", "please", "reply", "letter",
                       "request", "appointment", "thanks", "email", "schedule", "asked", "time"],
}
COMMON_WORDS = ["the", "and", "with", "near", "file", "case", "page", "date", "street", "report",
                "copy", "note", "team", "office", "document", "review"]
# Case-specific words: invented place and object names that recur inside one case.
CASE_WORDS = ["hall", "bluevan", "mill", "orchard", "tunnel", "kiosk", "bridge", "garage", "depot",
              "quarry", "station", "market", "chapel", "canal", "silo", "ferry", "arcade", "pier",
              "foundry", "warehouse", "terrace", "viaduct", "granary", "lighthouse"]

# Imbalanced class mix, as in real document intake.
TYPE_WEIGHTS = [0.34, 0.33, 0.23, 0.10]


def make_case_documents(n_cases=200, seed=7, prefix="case"):
    """Return a list of dicts: {doc_id, case_id, text, label}.

    Each case comes mostly from one source, so most of its documents share a type
    and the case's own place words. A random split therefore leaks case words
    between training and test data. A split by case closes this route.
    """
    rng = random.Random(seed)
    docs = []
    for c in range(n_cases):
        case_id = f"{prefix}-{c:03d}"
        dominant = rng.choices(DOC_TYPES, weights=TYPE_WEIGHTS)[0]
        case_terms = rng.sample(CASE_WORDS, 2)
        for d in range(rng.randint(4, 9)):
            label = dominant if rng.random() < 0.9 else rng.choices(DOC_TYPES, weights=TYPE_WEIGHTS)[0]
            n_type = rng.randint(3, 7)
            words = rng.sample(TYPE_WORDS[label], n_type)
            # Noise: words borrowed from another type make the task realistic.
            other = rng.choice([t for t in DOC_TYPES if t != label])
            words += rng.sample(TYPE_WORDS[other], rng.randint(1, 3))
            words += rng.sample(COMMON_WORDS, rng.randint(4, 8))
            words += case_terms * rng.randint(3, 5)
            rng.shuffle(words)
            docs.append({"doc_id": f"{case_id}-d{d}", "case_id": case_id, "text": " ".join(words),
                         "label": label})
    return docs


# The final test set: generated once with its own seed and its own case names. It is never
# part of a training pool, so every scenario is judged on exactly the same documents.
FINAL_TEST_CASES = 100
FINAL_TEST_SEED = 23


def make_final_test_set():
    return make_case_documents(n_cases=FINAL_TEST_CASES, seed=FINAL_TEST_SEED, prefix="test-case")


def split_by_group(rows, key, fractions=(0.7, 0.15, 0.15), seed=11):
    """Split rows into train, validation and test sets so that a group never spans two sets."""
    groups = sorted({r[key] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n = len(groups)
    a = int(n * fractions[0])
    b = a + int(n * fractions[1])
    names = {g: "train" for g in groups[:a]}
    names.update({g: "val" for g in groups[a:b]})
    names.update({g: "test" for g in groups[b:]})
    out = {"train": [], "val": [], "test": []}
    for r in rows:
        out[names[r[key]]].append(r)
    return out


def split_at_random(rows, fractions=(0.7, 0.15, 0.15), seed=11):
    """The wrong split for grouped data: documents of one case land in several sets."""
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n = len(rows)
    a = int(n * fractions[0])
    b = a + int(n * fractions[1])
    return {"train": rows[:a], "val": rows[a:b], "test": rows[b:]}


# ---------------------------------------------------------------------------
# Evidence for a release decision
# ---------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()


def code_revision():
    """The commit that produced a result, marked when the examples differ from that commit."""
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                      stderr=subprocess.DEVNULL, text=True).strip()
        changed = subprocess.check_output(["git", "status", "--porcelain", "--", "examples"], cwd=ROOT,
                                          stderr=subprocess.DEVNULL, text=True).strip()
        return rev + ("+local-changes" if changed else "")
    except Exception:
        return "not-a-git-checkout"


def sha256_dir(path):
    """One digest for a whole model directory: every file name and every file content."""
    h = hashlib.sha256()
    root = Path(path)
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(f.relative_to(root).as_posix().encode("utf-8"))
        h.update(sha256_file(f).encode("ascii"))
    return h.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_environment(threads=None):
    """What a reader needs to reproduce a run: versions, threads and the code revision."""
    versions = {}
    for package in ("numpy", "scikit-learn", "torch", "mlflow"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "platform": platform.platform(terse=True),
            "cpu_count": os.cpu_count(), "threads": threads, "packages": versions,
            "code_revision": code_revision()}


def save_json(name, obj):
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return path
