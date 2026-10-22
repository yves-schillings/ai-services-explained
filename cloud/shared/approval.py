"""Step 6 of every cloud walkthrough: apply the quality gate, then record a named owner's decision.

Author: Yves Schillings.

The cloud services keep the evidence: Azure Machine Learning and SageMaker AI both
show the MLflow run, its metrics and its model. They do not decide what goes live.
This module is the decision step that the submit scripts call after the training
job: the gate is checked by code, the approval is written by a person, and only an
approved candidate is registered and deployed.

The approval record is a small JSON file kept outside the job outputs, for example
in the repository that holds the release decisions, never next to the model files.
"""
import json
from datetime import datetime, timezone
from pathlib import Path


def check_gate(gate):
    """gate is the gate.json written by the training job."""
    for key in ("metric", "value", "threshold", "passes"):
        if key not in gate:
            raise ValueError(f"gate.json misses {key}")
    if bool(gate["value"] >= gate["threshold"]) != bool(gate["passes"]):
        raise ValueError("gate.json is inconsistent: value, threshold and passes disagree")
    return bool(gate["passes"])


def record_decision(path, model_name, job_name, gate, owner, decision, reason):
    """Write the decision of a named owner. Only 'approved' allows registration and deployment."""
    if decision not in ("approved", "denied", "pending"):
        raise ValueError("decision must be approved, denied or pending")
    if decision == "approved" and not check_gate(gate):
        raise ValueError("a model below the gate cannot be approved")
    if not owner.strip():
        raise ValueError("an approval needs a named owner")
    record = {"model_name": model_name, "training_job": job_name, "gate": gate, "decision": decision,
              "owner": owner, "reason": reason, "decided_at": datetime.now(timezone.utc).isoformat()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def approved(path, job_name):
    """True only if the record exists, names this job and says approved."""
    p = Path(path)
    if not p.exists():
        return False
    record = json.loads(p.read_text(encoding="utf-8"))
    return record.get("training_job") == job_name and record.get("decision") == "approved"
