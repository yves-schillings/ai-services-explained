"""Train, approve and deploy the document-triage classifier with Azure Machine Learning.

Author: Yves Schillings.

STATUS: prepared, not executed against Azure. Without --submit, --decide, --deploy or
--download the script only builds the job definition and prints it: no Azure call,
no cost. Every other mode needs a signed-in identity and creates billed resources.

The seven steps of the walkthrough:
  1. prepare the fictional documents and their labels   cloud/shared/prepare_data.py
  2. split them by case into training, validation and test (same script)
  3. store the data and define the software environment  build_job(): the data folder is
     uploaded to the workspace storage, environment/conda.yml defines the packages
  4. submit the script to Azure Machine Learning         --submit
  5. log parameters, metrics, curves and the model       train_triage.py logs to MLflow;
     inside the job the MLflow tracking URI points to the workspace automatically
  6. apply the quality gate and record the approval      --decide (gate by code, decision
     by a named owner, kept in approvals/ outside the job outputs)
  7. deploy the approved model and call its endpoint      --deploy, then call_endpoint.py

Commands (from the repository root):
  python cloud/shared/prepare_data.py && python cloud/shared/stage.py
  python cloud/azure/submit.py                                  dry run, no Azure call
  python cloud/azure/submit.py --submit                         step 4, billed
  python cloud/azure/submit.py --decide JOB --owner "A. Owner" --decision approved --reason "gate met"
  python cloud/azure/submit.py --deploy JOB                     step 7, billed while it runs
  python cloud/azure/submit.py --download document-triage:1     model files for local hosting

Workspace: set AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP and AZUREML_WORKSPACE.
Sign in with az login or a managed identity (DefaultAzureCredential).
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

CLOUD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLOUD / "shared"))

from approval import approved, record_decision  # noqa: E402

AZURE = CLOUD / "azure"
MODEL_NAME = "document-triage"
ENDPOINT = "document-triage"
IMAGE = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest"
APPROVALS = CLOUD.parent / "approvals" / "cloud"


def local(path):
    """A path relative to the working folder: on Windows an absolute path such as E:/... would be
    read as an asset reference with the scheme E. Run the script from the repository root."""
    return "./" + Path(os.path.relpath(path)).as_posix()


def build_job():
    """Steps 3 and 4 as a definition: code, data, environment and compute. Builds nothing in Azure."""
    from azure.ai.ml import Input, Output, command
    from azure.ai.ml.entities import Environment

    return command(
        code=local(CLOUD / "build" / "src"),                                # only the staged training code
        command="python train_triage.py --data ${{inputs.data}} --output ${{outputs.result}}",
        inputs={"data": Input(type="uri_folder", path=local(CLOUD / "data" / "triage"))},
        outputs={"result": Output(type="uri_folder", mode="upload")},
        environment=Environment(image=IMAGE, conda_file=str(AZURE / "environment" / "conda.yml")),
        compute="cpu-cluster",
        experiment_name="document-triage-cloud",
        display_name="document-triage-cloud",
        tags={"example": "01", "data": "fictional"},
    )


def connect():
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential

    return MLClient(DefaultAzureCredential(), os.environ["AZURE_SUBSCRIPTION_ID"],
                    os.environ["AZURE_RESOURCE_GROUP"], os.environ["AZUREML_WORKSPACE"])


def submit(ml_client):
    """Step 4. The job uploads the code and the data, then runs train_triage.py on cpu-cluster."""
    job = ml_client.jobs.create_or_update(build_job())
    print("Submitted", job.name, job.studio_url)
    ml_client.jobs.stream(job.name)                                          # step 5 happens in the job
    return job.name


def job_result(ml_client, job_name):
    """Read gate.json and metrics.json from the job output, never from the job's own claim."""
    with tempfile.TemporaryDirectory() as tmp:
        ml_client.jobs.download(job_name, output_name="result", download_path=tmp)
        folder = next(Path(tmp).rglob("gate.json")).parent
        return (json.loads((folder / "gate.json").read_text(encoding="utf-8")),
                json.loads((folder / "metrics.json").read_text(encoding="utf-8")))


def decide(ml_client, job_name, owner, decision, reason):
    """Step 6. The gate is checked by code. The decision is written by a named person."""
    gate, metrics = job_result(ml_client, job_name)
    print("Metrics of", job_name, metrics)
    record = record_decision(APPROVALS / f"{job_name}.json", MODEL_NAME, job_name, gate, owner, decision, reason)
    print("Decision recorded:", record["decision"], "by", record["owner"])


def deploy(ml_client, job_name):
    """Step 7. Register the approved model as a new version, deploy it and route the traffic to it."""
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import (CodeConfiguration, Environment, ManagedOnlineDeployment,
                                      ManagedOnlineEndpoint, Model)

    if not approved(APPROVALS / f"{job_name}.json", job_name):
        raise SystemExit(f"{job_name} has no approval record that says approved: nothing is registered")
    model = ml_client.models.create_or_update(Model(
        path=f"azureml://jobs/{job_name}/outputs/result/paths/model/", name=MODEL_NAME,
        type=AssetTypes.MLFLOW_MODEL, tags={"training_job": job_name, "approval": "approvals/cloud"}))
    ml_client.online_endpoints.begin_create_or_update(
        ManagedOnlineEndpoint(name=ENDPOINT, auth_mode="aad_token")).result()
    ml_client.online_deployments.begin_create_or_update(ManagedOnlineDeployment(
        name="blue", endpoint_name=ENDPOINT, model=model,
        environment=Environment(image=IMAGE, conda_file=str(AZURE / "environment" / "conda.yml")),
        code_configuration=CodeConfiguration(code=local(AZURE / "scoring"), scoring_script="score.py"),
        instance_type="Standard_DS3_v2", instance_count=1)).result()
    endpoint = ml_client.online_endpoints.get(ENDPOINT)
    endpoint.traffic = {"blue": 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    answer = ml_client.online_endpoints.invoke(endpoint_name=ENDPOINT, deployment_name="blue",
                                               request_file=local(AZURE / "sample_request.json"))
    print("Deployed", MODEL_NAME, "version", model.version, "- test answer:", answer)
    print("Delete the endpoint after the test: az ml online-endpoint delete -n", ENDPOINT)


def download(ml_client, name_version, path="downloaded_models"):
    """Local hosting: the registered model is a normal MLflow folder that can be downloaded."""
    name, version = name_version.split(":")
    ml_client.models.download(name=name, version=version, download_path=path)
    print("Downloaded to", path, "- load it with mlflow.sklearn.load_model on your own server")


def main(argv=None):
    p = argparse.ArgumentParser(description="Azure Machine Learning walkthrough (prepared)")
    p.add_argument("--submit", action="store_true", help="step 4: submit the training job (billed)")
    p.add_argument("--decide", metavar="JOB", help="step 6: record the owner's decision for a finished job")
    p.add_argument("--owner", default="")
    p.add_argument("--decision", choices=("approved", "denied", "pending"), default="pending")
    p.add_argument("--reason", default="")
    p.add_argument("--deploy", metavar="JOB", help="step 7: register and deploy an approved job (billed)")
    p.add_argument("--download", metavar="NAME:VERSION", help="download a registered model")
    a = p.parse_args(argv)
    if not (a.submit or a.decide or a.deploy or a.download):
        print("Dry run: the job definition below is built locally. Nothing is sent to Azure.")
        print(build_job())
        return
    ml_client = connect()
    if a.submit:
        submit(ml_client)
    if a.decide:
        decide(ml_client, a.decide, a.owner, a.decision, a.reason)
    if a.deploy:
        deploy(ml_client, a.deploy)
    if a.download:
        download(ml_client, a.download)


if __name__ == "__main__":
    main()
