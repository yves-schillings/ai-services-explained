"""The approval record: the trusted reference that says which model version may go live.

Author: Yves Schillings.

Only the release step (19_model_release.py) writes this record, after the
validation code has checked the gate and a named owner has decided. The service
(18_model_serving.py) only reads it.

The record lives apart from the model files, in its own directory
(TRIAGE_APPROVALS_DIR, by default approvals/). That separation matters:
a model replaced together with its manifest or MLflow metadata stays
self-consistent, so a checksum next to the model proves nothing. A checksum
proves integrity only against a reference that is protected on its own. In
production that directory belongs to the release owners, and the service mounts
it read-only.

Decisions are appended, never edited. For each version the last decision counts:
  approved   the version may go live,
  denied     the owner or the gate refused it,
  pending    it waits for a decision,
  withdrawn  it was live and was taken back (rollback).
The live version is the most recently approved version that is still approved.
"""
import json
import os
from pathlib import Path

from fictional_data import ROOT, utc_now

APPROVALS = ROOT / "approvals"
DECISIONS = ("approved", "denied", "pending", "withdrawn")


def approvals_dir():
    return Path(os.environ.get("TRIAGE_APPROVALS_DIR") or APPROVALS)


def record_path(model_name, directory=None):
    return Path(directory or approvals_dir()) / f"{model_name}.json"


def read_record(model_name, directory=None):
    path = record_path(model_name, directory)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def append_decision(model_name, entry, acceptance, directory=None):
    """Add one decision. Called by the release step only."""
    if entry["decision"] not in DECISIONS:
        raise ValueError(f"Unknown decision: {entry['decision']}")
    path = record_path(model_name, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = read_record(model_name, directory) or {"model_name": model_name, "acceptance": acceptance,
                                                    "decisions": []}
    record["decisions"].append({**entry, "decided_utc": utc_now()})
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def status_by_version(record):
    status = {}
    for d in record["decisions"]:
        status[d["version"]] = d
    return status


def live_entry(record):
    """The decision entry of the version that may serve, or None."""
    if not record:
        return None
    status = status_by_version(record)
    for d in reversed(record["decisions"]):
        if d["decision"] == "approved" and status[d["version"]]["decision"] == "approved":
            return d
    return None
