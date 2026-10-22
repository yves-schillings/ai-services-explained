"""Train the document classifier or the page CNN with SageMaker AI, track it in MLflow, deploy it.

Author: Yves Schillings.

STATUS: prepared, not executed against AWS. Without --submit, --decide or --deploy the
script only prints the plan: no AWS call, no cost. The other modes need AWS credentials,
an S3 bucket, an IAM execution role and, for MLflow, a SageMaker AI managed MLflow
tracking server. They start billed resources.

SageMaker AI runs YOUR OWN training code (scikit-learn or PyTorch) in a container on an
instance it starts for the job. The same entry scripts as Azure Machine Learning are used:
cloud/build/src/train_triage.py and train_page_cnn.py (see cloud/shared/stage.py).

Walkthrough:
  1-2. python cloud/shared/prepare_data.py   fictional data, split by case
  3.   upload the data folder to S3           (--submit does it)
  4.   start a training job (ModelTrainer, SageMaker Python SDK v3)
  5.   the script logs to MLflow: the job receives the tracking server ARN in
       MLFLOW_TRACKING_URI and the sagemaker-mlflow plugin signs the calls with the
       job's role. SageMaker AI writes /opt/ml/model to S3 as model.tar.gz
  6.   --decide: read gate.json from model.tar.gz, record the named owner's decision
  7.   --deploy: deploy the approved model to a real-time endpoint

Commands (from the repository root):
  python cloud/shared/prepare_data.py && python cloud/shared/stage.py
  python cloud/aws/sagemaker/launch_training.py                       plan only
  python cloud/aws/sagemaker/launch_training.py --submit triage        billed
  python cloud/aws/sagemaker/launch_training.py --decide JOB --owner "A. Owner" --decision approved
  python cloud/aws/sagemaker/launch_training.py --deploy JOB           billed while it runs

Settings (environment variables): SAGEMAKER_ROLE_ARN, SAGEMAKER_BUCKET,
MLFLOW_TRACKING_ARN, SAGEMAKER_INFERENCE_IMAGE, AWS_REGION.
"""
import argparse
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path

CLOUD = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CLOUD / "shared"))

from approval import approved, record_decision  # noqa: E402

APPROVALS = CLOUD.parent / "approvals" / "cloud"
JOBS = {
    # name: entry script, data folder, instance type. ml.m5.large is a CPU instance: both
    # models of this article train on CPU in seconds to minutes.
    "triage": ("train_triage.py", CLOUD / "data" / "triage", "ml.m5.large"),
    "pages": ("train_page_cnn.py", CLOUD / "data" / "pages", "ml.m5.large"),
}
PYTORCH = {"framework": "pytorch", "version": "2.8", "py_version": "py312", "image_scope": "training"}


def plan(job="triage"):
    """What the job would do, as plain data. Used by the dry run and by the offline tests."""
    entry, data, instance = JOBS[job]
    return {"entry_script": entry, "source_dir": str(CLOUD / "build" / "src"), "requirements": "requirements.txt",
            "data_folder": str(data), "s3_data": f"s3://<SAGEMAKER_BUCKET>/ai-services-explained/{job}/data",
            "s3_output": f"s3://<SAGEMAKER_BUCKET>/ai-services-explained/{job}/output",
            "instance_type": instance, "instance_count": 1, "max_runtime_seconds": 3600,
            "image": PYTORCH, "channel": "training",
            "environment": {"MLFLOW_TRACKING_URI": "<MLFLOW_TRACKING_ARN>",
                            "MLFLOW_EXPERIMENT_NAME": f"{job}-sagemaker"}}


def submit(job):
    """Steps 3 to 5. Upload the data, start the job, wait for it. Returns the job name."""
    from sagemaker.core import image_uris
    from sagemaker.core.helper.session_helper import Session
    from sagemaker.train import ModelTrainer
    from sagemaker.train.configs import Compute, InputData, OutputDataConfig, SourceCode, StoppingCondition

    p, bucket = plan(job), os.environ["SAGEMAKER_BUCKET"]
    session = Session()
    s3_data = session.upload_data(p["data_folder"], bucket=bucket, key_prefix=f"ai-services-explained/{job}/data")
    image = image_uris.retrieve(region=session.boto_region_name, instance_type=p["instance_type"], **PYTORCH)
    trainer = ModelTrainer(
        training_image=image,
        source_code=SourceCode(source_dir=p["source_dir"], entry_script=p["entry_script"],
                               requirements=p["requirements"]),
        compute=Compute(instance_type=p["instance_type"], instance_count=1),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=p["max_runtime_seconds"]),
        output_data_config=OutputDataConfig(s3_output_path=p["s3_output"].replace("<SAGEMAKER_BUCKET>", bucket)),
        environment={"MLFLOW_TRACKING_URI": os.environ["MLFLOW_TRACKING_ARN"],
                     "MLFLOW_EXPERIMENT_NAME": p["environment"]["MLFLOW_EXPERIMENT_NAME"]},
        role=os.environ["SAGEMAKER_ROLE_ARN"], base_job_name=f"ai-services-{job}", sagemaker_session=session)
    trainer.train(input_data_config=[InputData(channel_name=p["channel"], data_source=s3_data)], wait=True)
    name = trainer._latest_training_job.training_job_name             # set by train() in SDK v3
    print("Training job finished:", name)
    return name


def model_archive(job_name):
    """S3 location of model.tar.gz: SageMaker AI packs /opt/ml/model at the end of the job."""
    import boto3
    desc = boto3.client("sagemaker").describe_training_job(TrainingJobName=job_name)
    return desc["ModelArtifacts"]["S3ModelArtifacts"]


def read_gate(job_name):
    import boto3
    bucket, key = model_archive(job_name)[len("s3://"):].split("/", 1)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.tar.gz"
        boto3.client("s3").download_file(bucket, key, str(path))
        with tarfile.open(path) as tar:
            gate = json.load(tar.extractfile("gate.json"))
            metrics = json.load(tar.extractfile("metrics.json"))
    return gate, metrics


def decide(job_name, owner, decision, reason):
    """Step 6: the gate by code, the decision by a named person."""
    gate, metrics = read_gate(job_name)
    print("Metrics of", job_name, metrics)
    record = record_decision(APPROVALS / f"{job_name}.json", "document-triage", job_name, gate, owner, decision,
                             reason)
    print("Decision recorded:", record["decision"], "by", record["owner"])


def deploy(job_name):
    """Step 7: only an approved job reaches a real-time endpoint.

    The endpoint runs inference/inference.py in a prebuilt AWS Deep Learning Container with
    DJL Serving, the pattern of the SageMaker Python SDK v3 script-mode examples. Choose the
    CPU inference image in the list of available Deep Learning Containers and set
    SAGEMAKER_INFERENCE_IMAGE to its URI.
    """
    from sagemaker.core.helper.session_helper import Session
    from sagemaker.serve import ModelBuilder, ModelServer
    from sagemaker.serve.mode.function_pointers import Mode
    from sagemaker.train.configs import SourceCode

    if not approved(APPROVALS / f"{job_name}.json", job_name):
        raise SystemExit(f"{job_name} has no approval record that says approved: nothing is deployed")
    builder = ModelBuilder(
        image_uri=os.environ["SAGEMAKER_INFERENCE_IMAGE"], model_server=ModelServer.DJL_SERVING,
        source_code=SourceCode(source_dir=str(Path(__file__).parent / "inference"), entry_script="inference.py",
                               requirements="requirements.txt"),
        s3_model_data_url=model_archive(job_name), env_vars={"OPTION_ENTRYPOINT": "code/inference.py"},
        role_arn=os.environ["SAGEMAKER_ROLE_ARN"], sagemaker_session=Session())
    builder.build(model_name="document-triage", mode=Mode.SAGEMAKER_ENDPOINT)
    predictor = builder.deploy(endpoint_name="document-triage", initial_instance_count=1,
                               instance_type="ml.m5.large")
    print("Endpoint ready:", predictor.endpoint_name, "- delete it after the test, it is billed per hour")


def main(argv=None):
    p = argparse.ArgumentParser(description="SageMaker AI walkthrough (prepared)")
    p.add_argument("--submit", choices=sorted(JOBS), help="steps 3 to 5 for one model (billed)")
    p.add_argument("--decide", metavar="JOB")
    p.add_argument("--owner", default="")
    p.add_argument("--decision", choices=("approved", "denied", "pending"), default="pending")
    p.add_argument("--reason", default="")
    p.add_argument("--deploy", metavar="JOB")
    a = p.parse_args(argv)
    if not (a.submit or a.decide or a.deploy):
        print("Plan only. Nothing is sent to AWS.")
        for job in JOBS:
            print(json.dumps(plan(job), indent=2))
        return
    if a.submit:
        submit(a.submit)
    if a.decide:
        decide(a.decide, a.owner, a.decision, a.reason)
    if a.deploy:
        deploy(a.deploy)


if __name__ == "__main__":
    main()
