# Amazon SageMaker AI: train your own model, track it in MLflow, deploy it

Author: Yves Schillings. **Status: prepared, not executed in AWS.**

SageMaker AI runs your own training code in a container on an instance that it starts for the job. Here that code is
the same entry script as for Azure Machine Learning: `cloud/build/src/train_triage.py` or `train_page_cnn.py`.

## Files

| File | Role |
| --- | --- |
| `launch_training.py` | Steps 3 to 7 with the SageMaker Python SDK v3 (`ModelTrainer`, `ModelBuilder`). Without arguments it prints the plan only |
| `inference/inference.py`, `inference/requirements.txt` | DJL Serving handler of the real-time endpoint. It returns the same answer as the local service |
| `../../shared/job_requirements.txt` | Packages installed in the training job (staged as `requirements.txt`) |

## Run

```powershell
python cloud/shared/prepare_data.py; python cloud/shared/stage.py
python cloud/aws/sagemaker/launch_training.py                 # plan only, no AWS call
python cloud/aws/sagemaker/launch_training.py --submit triage
python cloud/aws/sagemaker/launch_training.py --decide <JOB> --owner "<Name>" --decision approved
python cloud/aws/sagemaker/launch_training.py --deploy <JOB>
```

Settings:
- `SAGEMAKER_ROLE_ARN`
- `SAGEMAKER_BUCKET`
- `MLFLOW_TRACKING_ARN`
- `SAGEMAKER_INFERENCE_IMAGE`, a CPU inference Deep Learning Container with DJL Serving
- `AWS_REGION`.

## What to know

- **Permissions:**
  - an execution role that reads and writes the S3 prefix, writes logs and calls the MLflow tracking server
  - your own rights to start training jobs and endpoints.
- **Data sent:**
  - the fictional files go to your S3 bucket in your Region
  - SageMaker AI writes `/opt/ml/model` back to S3 as `model.tar.gz`
  - MLflow metadata goes to the tracking server.
- **Hardware and duration:**
  - one `ml.m5.large` CPU instance with the `pytorch-training:2.8-cpu-py312` image
  - the training takes seconds, and the container start takes a few minutes.
- **Cost elements:**
  - training instance seconds
  - S3
  - the managed MLflow tracking server while it runs
  - endpoint instance hours until the endpoint is deleted.
- **MLflow:** the `sagemaker-mlflow` plugin accepts the tracking server ARN as tracking URI and signs the calls with
  the job's role. The client version must match the server version, for example `mlflow==3.0.0` for a 3.0.x server.
- **Local hosting:** download `model.tar.gz` from the S3 location in `ModelArtifacts`, extract it, and load the
  `model/` folder with `mlflow.sklearn.load_model`.

## Documentation used (checked 22 October 2026)

- [SageMaker Python SDK v3: model training](https://sagemaker.readthedocs.io/en/stable/training/index.html)
- [Migration from SDK v2 to v3](https://github.com/aws/sagemaker-python-sdk/blob/master/migration.md)
- [Script mode in SDK v3, with a scikit-learn model deployed through ModelBuilder](https://aws.amazon.com/blogs/machine-learning/bring-your-own-model-with-amazon-sagemaker-ai-script-mode-in-sdk-v3/)
- [Integrate MLflow with your environment (tracking server ARN, client versions)](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-track-experiments.html)
