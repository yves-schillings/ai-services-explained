"""Bedrock model customization (fine-tuning) job for Meta Llama 3.2 1B Instruct.

Author: Yves Schillings.

STATUS: prepared, not executed. Without --submit the script builds the request and
validates it offline against the Bedrock API model shipped with botocore: no AWS call,
no cost. --submit starts a billed job in us-west-2.

What this path is, and what it is not:
  * Bedrock fine-tunes a SUPPORTED foundation model that AWS hosts, within the limits
    of that model and Region. Meta Llama 3.2 1B Instruct is supported in us-west-2.
  * It does NOT run your own scikit-learn or PyTorch program. For that, use SageMaker AI
    (cloud/aws/sagemaker) or Azure Machine Learning (cloud/azure).
  * The custom model is used through Bedrock: a custom model deployment for on-demand
    inference, or Provisioned Throughput. The Bedrock documentation describes no way to
    download the fine-tuned weights, so plan local hosting from the open base model
    (LoRA on your own GPU) instead.

Billed: tokens processed during training (records x tokens x epochs), monthly storage of
the custom model, and inference on the deployment.

Run (from the repository root):
  python cloud/shared/prepare_data.py && python cloud/aws/bedrock/prepare_dataset.py
  python cloud/aws/bedrock/customization_job.py                  offline validation only
  aws s3 cp cloud/data/bedrock s3://<BUCKET>/ai-services-explained/bedrock/ --recursive
  python cloud/aws/bedrock/customization_job.py --submit         billed

Settings: BEDROCK_ROLE_ARN (service role that reads and writes the S3 prefix), BEDROCK_BUCKET.
"""
import argparse
import json
import os

REGION = "us-west-2"
BASE_MODEL = "meta.llama3-2-1b-instruct-v1:0:128k"
PREFIX = "ai-services-explained/bedrock"
# Llama 3.2 limits on Bedrock: epochCount 1-10 (default 5), batchSize 1, learningRate 5e-6 to 0.1.
HYPERPARAMETERS = {"epochCount": "2", "batchSize": "1", "learningRate": "0.0001"}


def request(bucket="<BEDROCK_BUCKET>", role_arn="arn:aws:iam::123456789012:role/<BEDROCK_ROLE>",
            job_name="document-type-llama32-1b", model_name="document-type-llama32-1b"):
    """The CreateModelCustomizationJob request, as plain data."""
    return {"jobName": job_name,
            "customModelName": model_name,
            "roleArn": role_arn,
            "baseModelIdentifier": BASE_MODEL,
            "customizationType": "FINE_TUNING",
            "trainingDataConfig": {"s3Uri": f"s3://{bucket}/{PREFIX}/train.jsonl"},
            "validationDataConfig": {"validators": [{"s3Uri": f"s3://{bucket}/{PREFIX}/validation.jsonl"}]},
            "outputDataConfig": {"s3Uri": f"s3://{bucket}/{PREFIX}/output/"},
            "hyperParameters": HYPERPARAMETERS,
            "jobTags": [{"key": "data", "value": "fictional"}]}


def validate_offline(params):
    """Check the request against the Bedrock API model in botocore. No network call."""
    import botocore.session
    from botocore.validate import ParamValidator

    op = botocore.session.get_session().get_service_model("bedrock").operation_model("CreateModelCustomizationJob")
    report = ParamValidator().validate(params, op.input_shape)
    if report.has_errors():
        raise ValueError(report.generate_report())
    return True


def submit(params):
    import boto3
    response = boto3.client("bedrock", region_name=REGION).create_model_customization_job(**params)
    print("Job started:", response["jobArn"])
    print("Follow it: aws bedrock get-model-customization-job --job-identifier", response["jobArn"])


def main(argv=None):
    p = argparse.ArgumentParser(description="Bedrock fine-tuning job (prepared)")
    p.add_argument("--submit", action="store_true", help="start the billed customization job")
    a = p.parse_args(argv)
    if a.submit:
        params = request(os.environ["BEDROCK_BUCKET"], os.environ["BEDROCK_ROLE_ARN"])
        validate_offline(params)
        submit(params)
    else:
        params = request()
        validate_offline(params)
        print("Request valid against the Bedrock API model. Nothing is sent to AWS.")
        print(json.dumps(params, indent=2))


if __name__ == "__main__":
    main()
