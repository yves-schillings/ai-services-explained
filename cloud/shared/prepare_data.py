"""Steps 1 and 2 of every cloud walkthrough: prepare the fictional data and split it.

Author: Yves Schillings.

Run:  python cloud/shared/prepare_data.py

The data is exactly the data of the local examples, so a cloud run can be compared
with the local run on the same documents and the same pages:
  * document triage (example 01): the 400-case labelled pool, split by case into
    training (80 %) and validation (20 %), plus the fixed final test set of
    100 other cases that is never used for training or tuning;
  * page layout (example 02): 150 training, 60 validation and 200 clean plus
    200 degraded test pages per type, with the seeds of example 02.

Output (ignored by git, recreated on demand):
  cloud/data/triage/train.jsonl, val.jsonl, test.jsonl
  cloud/data/pages/train.npz, val.npz, test_clean.npz, test_hard.npz
  cloud/data/manifest.json   counts and SHA-256 of every file (the data version)

Only fictional text and synthetic images are written. Nothing leaves this machine.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

CLOUD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLOUD.parent / "examples"))

from fictional_data import make_case_documents, make_final_test_set, split_by_group  # noqa: E402
from page_images import draw_page, make_set  # noqa: E402

DATA = CLOUD / "data"
POOL_CASES = 400                       # the largest labelled pool of example 01
PAGE_SETS = {"train": (150, 1, False), "val": (60, 2, False),           # per type, seed, degraded
             "test_clean": (200, 3, False), "test_hard": (200, 4, True)}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps({"case_id": r["case_id"], "text": r["text"], "label": r["label"]}) + "\n")


def prepare_triage(out=DATA / "triage"):
    """Step 1: fictional documents with their labels. Step 2: split by case, never by document."""
    pool = make_case_documents(n_cases=POOL_CASES)
    parts = split_by_group(pool, "case_id", fractions=(0.8, 0.2, 0.0))     # the split of example 01
    final_test = make_final_test_set()
    sets = {"train": parts["train"], "val": parts["val"], "test": final_test}
    cases = {k: {r["case_id"] for r in v} for k, v in sets.items()}
    assert not (cases["train"] & cases["val"]) and not (cases["train"] | cases["val"]) & cases["test"]
    for name, rows in sets.items():
        write_jsonl(out / f"{name}.jsonl", rows)
    return {name: len(rows) for name, rows in sets.items()}


def prepare_pages(out=DATA / "pages"):
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, (per_type, seed, hard) in PAGE_SETS.items():
        x, y = make_set(draw_page, 3, per_type, seed=seed, hard=hard)
        np.savez_compressed(out / f"{name}.npz", x=x.astype(np.float32), y=y.astype(np.int64))
        counts[name] = int(len(y))
    return counts


def main():
    counts = {"triage": prepare_triage(), "pages": prepare_pages()}
    files = sorted(p for p in DATA.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest = {"counts": counts,
                "files": {p.relative_to(DATA).as_posix(): sha256_file(p) for p in files}}
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Prepared:", json.dumps(counts))
    print("Data version (SHA-256 of each file) in", DATA / "manifest.json")


if __name__ == "__main__":
    main()
