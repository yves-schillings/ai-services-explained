# Training on a local server, Azure or AWS

Author: Yves Schillings.

**Status: prepared.** No job has run in Azure or AWS, and no cloud resource has been created. The offline checks
listed below ran on a laptop CPU. Every cloud command creates billed resources, so it waits for an explicit decision
to spend.

The same two models as the local examples are trained, on the same fictional data, with the same quality gates:

- **The document classifier** of `examples/01_machine_learning.py`. It is a scikit-learn pipeline and its gate is a
  test macro-F1 of 0,85.
- **The page-layout CNN** of `examples/02_deep_learning.py`. It is a PyTorch network, and this folder uses a
  fictional gate of 0,95 clean test accuracy.

## Four options

| Option | Who runs the training code | Data sent and destination | Getting the model back for local hosting |
| --- | --- | --- | --- |
| **Local server + MLflow** (`examples/`) | Your own Python on your own machine | Nothing leaves the machine | The model is already local |
| **Azure Machine Learning** (`azure/`) | Your own script, as a command job on a CPU or GPU cluster | Fictional JSON Lines and page arrays go to the workspace storage account in the workspace region | Download the registered MLflow model with `az ml model download` or `ml_client.models.download` |
| **Amazon SageMaker AI** (`aws/sagemaker/`) | Your own script, as a training job in a container | The data goes to an S3 bucket in the chosen Region. SageMaker AI writes `model.tar.gz` back to S3 | Download `model.tar.gz` from S3 and load its `model/` folder with MLflow |
| **Amazon Bedrock** (`aws/bedrock/`) | AWS fine-tunes a supported foundation model. Your own scikit-learn or PyTorch code does not run | The JSON Lines records go to S3 in us-west-2 and Bedrock reads them | The Bedrock documentation describes using the custom model through Bedrock, not downloading its weights. For local hosting, adapt the open base model yourself (LoRA) |

AWS is the provider. SageMaker AI and Bedrock are two different AWS services:
- **SageMaker AI** runs and deploys your own models.
- **Bedrock** gives access to foundation models and fine-tunes the supported ones.
- **Bedrock AgentCore**, a third service, runs agents that use models. It trains nothing.

## The walkthrough, the same for every cloud option

1. **Prepare the data.** Fictional documents and their labels, written by `shared/prepare_data.py`.
2. **Split by case.** The pool is split into training and validation, and the fixed final test set is never used for
   training. The same script does this.
3. **Store the data and define the software environment.** The data is uploaded with the job. The environment is in
   `azure/environment/conda.yml` or `shared/job_requirements.txt`.
4. **Submit the job.** `shared/stage.py` copies only the training code. Azure Machine Learning and SageMaker AI then
   run `train_triage.py` or `train_page_cnn.py`.
5. **Log to MLflow.** The job records the parameters, data version, metrics, curves and model:
   - in Azure, to the workspace (the tracking URI is set automatically)
   - in AWS, to a SageMaker AI managed MLflow tracking server (its ARN is passed in `MLFLOW_TRACKING_URI`).
6. **Gate and approval.** `shared/approval.py` checks the gate by code. A named owner then writes the decision, and
   it is kept outside the job outputs. The training job itself registers and deploys nothing.
7. **Deploy and call.** Only an approved job is registered and deployed. The application then calls the endpoint
   (`azure/call_endpoint.py`).

## What was executed and what stays prepared

| Executed offline, on a laptop CPU | Prepared, not executed |
| --- | --- |
| Data preparation and the split by case: 2.044 training, 544 validation and 684 final test documents. Pages: 450 training, 180 validation, 600 clean and 600 degraded test pages | Creating any Azure or AWS resource: workspace, cluster, bucket, role, tracking server |
| `train_triage.py` locally, with SQLite MLflow: validation macro-F1 0,9198, test macro-F1 0,9184, gate passed, 17,7 % sent to review | Submitting the Azure Machine Learning job and the SageMaker AI job |
| `train_page_cnn.py` locally on CPU: training, held-out evaluation, MLflow artifact output and the clean-accuracy gate were exercised | GPU cloud jobs and MLflow tracking to the Azure workspace or a SageMaker AI tracking server |
| The Azure scoring script and the SageMaker AI handler, both answering on that model | Registration, endpoints and their calls |
| The Azure job built by the SDK and the YAML loaded by `load_job` (azure-ai-ml 1.35.2) | The Bedrock customization job and its inference |
| The SageMaker AI SDK v3 classes, and the PyTorch training image resolved to `pytorch-training:2.8-cpu-py312` (sagemaker 3.26.0) | |
| The Bedrock request validated against the API model of botocore 1.43.117, the JSON Lines records checked, and the before/after rules with a stand-in client | |

Run the offline checks:

```powershell
python -m pytest cloud/tests -q
```

Tests that need `azure-ai-ml`, `sagemaker` or `boto3` skip themselves when the package is missing. Install them in a
separate environment to run all the checks.

## Cost elements (no prices: they change by Region and over time)

- **Azure Machine Learning:**
  - compute cluster minutes while a node runs
  - environment image builds and the container registry
  - storage
  - online endpoint instance hours while it is deployed.
- **SageMaker AI:**
  - training instance seconds
  - S3 storage
  - the managed MLflow tracking server while it runs
  - endpoint instance hours.
- **Bedrock:**
  - training tokens multiplied by the epochs
  - monthly storage of the custom model
  - inference tokens, or Provisioned Throughput hours.

The two models here train in seconds. The local run of `train_triage.py` takes about one minute, mostly for logging
the model to MLflow. In the cloud, starting a node and building an environment image take longer than the training
itself.
