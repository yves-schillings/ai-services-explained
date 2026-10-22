"""Release step: the gate and a named owner decide which model version may go live.

Author: Yves Schillings.

Run after training:  python examples/19_model_release.py
Then:                python examples/18_model_serving.py
Rollback:            python examples/19_model_release.py --rollback "reason"

MLflow keeps the information about every version that 01_machine_learning.py
registered: its run, parameters, metrics and model files. This step adds what
MLflow does not decide:
  1. the validation code checks the agreed gate from the run's metrics and
     refuses to approve a version below it,
  2. a named owner records a decision in the approval record (release_records.py),
     together with the digest of the version's model files,
  3. only then the MLflow alias "live" points at the approved version, so that
     people browsing MLflow see which version serves. The alias is for information.
     The service trusts the approval record, not the alias.

The demonstration ends with a rollback drill on a copy of the record: the owner
withdraws the live version, and the previous approved version becomes live.
"""
import json
import shutil
import tempfile

import mlflow

from fictional_data import RESULTS, save_json, sha256_dir
from mlflow_tracking import use_experiment
from release_records import append_decision, approvals_dir, live_entry, read_record

MODEL_NAME = "document-triage"
EXPERIMENT = "document-triage"
# The release step keeps its own copy of the gate agreed with the business owner,
# so the training code cannot lower it.
GATE_METRIC = "test_macro_f1"
GATE_MINIMUM = 0.85
OWNER = "document intake service owner (fictional)"
ALIAS = "live"


class GateFailed(ValueError):
    """The validation code refuses to approve a version below the agreed gate."""


def evidence(client, version, model_name=MODEL_NAME):
    """What the decision rests on: the run's metrics and the digest of the version's model files."""
    mv = client.get_model_version(model_name, version)
    run = client.get_run(mv.run_id)
    with tempfile.TemporaryDirectory() as tmp:
        local = mlflow.artifacts.download_artifacts(f"models:/{model_name}/{version}", dst_path=tmp)
        digest = sha256_dir(local)
    return {"version": str(version), "run_id": mv.run_id, "model_sha256": digest,
            GATE_METRIC: run.data.metrics[GATE_METRIC],
            "review_threshold": float(run.data.params["review_threshold"]),
            "labelled_cases": run.data.params.get("labelled_cases")}


def move_alias(client, record, model_name=MODEL_NAME):
    """Point the MLflow alias at the live version of the record, or remove it."""
    entry = live_entry(record)
    if entry:
        client.set_registered_model_alias(model_name, ALIAS, entry["version"])
    else:
        try:
            client.delete_registered_model_alias(model_name, ALIAS)
        except mlflow.exceptions.MlflowException:
            pass


def decide(client, version, decision, owner, reason, directory=None, model_name=MODEL_NAME, update_alias=True):
    """Record one owner's decision. Approval is refused in code when the gate is not met."""
    ev = evidence(client, version, model_name)
    if decision == "approved" and ev[GATE_METRIC] < GATE_MINIMUM:
        raise GateFailed(f"Version {version}: {GATE_METRIC} {ev[GATE_METRIC]:.4f} is below the gate "
                         f"{GATE_MINIMUM}. It cannot be approved.")
    record = append_decision(model_name, {**ev, "decision": decision, "owner": owner, "reason": reason},
                             acceptance=GATE_MINIMUM, directory=directory)
    if update_alias:
        move_alias(client, record, model_name)
    return record


def rollback(client, owner, reason, directory=None, model_name=MODEL_NAME, update_alias=True):
    """Withdraw the live version. The previous approved version becomes live."""
    entry = live_entry(read_record(model_name, directory))
    if entry is None:
        raise RuntimeError("Nothing is live, so there is nothing to roll back.")
    record = decide(client, entry["version"], "withdrawn", owner, reason, directory, model_name, update_alias)
    return entry["version"], live_entry(record)


def main():
    client = use_experiment(EXPERIMENT)
    trained = json.loads((RESULTS / "01_machine_learning.json").read_text(encoding="utf-8"))["runs"]
    directory = approvals_dir()
    steps = []
    for n_cases, run in trained.items():
        version = run["registered_version"]
        try:
            decide(client, version, "approved", OWNER, f"Gate met with {n_cases} labelled cases.", directory)
            steps.append({"version": version, "labelled_cases": n_cases, "decision": "approved"})
        except GateFailed as refused:
            print("   Validation code:", refused)
            decide(client, version, "denied", OWNER, "Below the agreed gate.", directory)
            steps.append({"version": version, "labelled_cases": n_cases, "decision": "denied"})
        print(f"Version {version} ({n_cases} cases): {steps[-1]['decision']}")
    live = live_entry(read_record(MODEL_NAME, directory))
    print("Live version:", live["version"], "- MLflow alias", ALIAS, "->",
          client.get_model_version_by_alias(MODEL_NAME, ALIAS).version)

    # Rollback drill on a copy of the record: the live decision itself is not changed.
    with tempfile.TemporaryDirectory() as drill:
        shutil.copy(directory / f"{MODEL_NAME}.json", drill)
        withdrawn, previous = rollback(client, OWNER, "Rollback drill.", drill, update_alias=False)
    print(f"Rollback drill: version {withdrawn} withdrawn, version {previous['version']} becomes live.")
    save_json("19_model_release.json", {
        "gate": {"metric": GATE_METRIC, "minimum": GATE_MINIMUM}, "owner": OWNER, "decisions": steps,
        "live_version": live["version"], "live_model_sha256": live["model_sha256"],
        "rollback_drill": {"withdrawn": withdrawn, "live_after_rollback": previous["version"]},
        "approval_record": "approvals/document-triage.json (kept apart from the model files)",
        "command": "python examples/19_model_release.py"})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Release decisions for the document-triage model.")
    parser.add_argument("--rollback", metavar="REASON",
                        help="withdraw the live version; the previous approved version becomes live")
    args = parser.parse_args()
    if args.rollback:
        withdrawn, now_live = rollback(use_experiment(EXPERIMENT), OWNER, args.rollback)
        print(f"Version {withdrawn} withdrawn. Live version: {now_live['version'] if now_live else 'none'}.")
    else:
        main()
