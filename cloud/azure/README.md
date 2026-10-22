# Azure Machine Learning: train, approve, deploy

Author: Yves Schillings. **Status: prepared, not executed in Azure.**

## Files

| File | Role |
| --- | --- |
| `job_triage.yml` | CLI v2 command job that trains the document classifier on `cpu-cluster` |
| `job_page_cnn_gpu.yml` | Optional GPU variant for the CNN. This small CNN trains on CPU, and the file shows how a larger network asks for a GPU |
| `compute_cpu.yml`, `compute_gpu.yml` | Compute clusters with `min_instances: 0`, so no node runs between jobs. Delete them after use: the training guide says a cluster bills while it exists |
| `environment/conda.yml` | Packages of the jobs and of the deployment |
| `submit.py` | Python SDK v2 (`azure-ai-ml`) version of the seven steps. Without arguments it is a dry run |
| `scoring/score.py`, `endpoint.yml`, `deployment.yml` | Managed online endpoint with Microsoft Entra ID authentication, and the scoring script |
| `call_endpoint.py`, `sample_request.json` | How the application calls the endpoint |

## Run

```powershell
python cloud/shared/prepare_data.py; python cloud/shared/stage.py
python cloud/azure/submit.py                     # dry run: builds the job locally, no Azure call
az ml compute create -f cloud/azure/compute_cpu.yml
az ml job create -f cloud/azure/job_triage.yml   # or: python cloud/azure/submit.py --submit
python cloud/azure/submit.py --decide <JOB> --owner "<Name>" --decision approved --reason "gate met"
python cloud/azure/submit.py --deploy <JOB>
python cloud/azure/call_endpoint.py "witness declares she saw a person near the harbour"
az ml online-endpoint delete -n document-triage --yes
az ml compute delete -n cpu-cluster --yes
```

## What to know

- **Permissions:**
  - a workspace role such as AzureML Data Scientist for jobs, MLflow tracking and models
  - rights to create compute and endpoints for the first run.
- **Data sent:** the fictional JSON Lines files (about 0,7 MB) go to the workspace storage account. The document
  texts sent to the endpoint go to the workspace region.
- **Hardware and duration:**
  - the job runs on one Standard_DS3_v2 CPU node
  - the training takes seconds, while the node start and the first environment image build take minutes.
- **Cost elements:**
  - cluster node minutes
  - image builds and the container registry
  - storage
  - endpoint instance hours. The deployment runs, and is billed, until it is deleted.
- **MLflow:**
  - inside the job the tracking URI points to the workspace automatically
  - `azureml-mlflow` 1.62 accepts MLflow clients up to 3.15, so the environment pins `mlflow-skinny==3.15.0`.
- **Registration:** the model is registered from the named job output with
  `azureml://jobs/<job>/outputs/result/paths/model/`, as an MLflow model, and only after the approval record says
  approved.
- **Local hosting:** `ml_client.models.download(name, version)` returns the MLflow folder.
  `mlflow.sklearn.load_model` loads it on your own server.

## Documentation used (checked 22 October 2026)

- [Train models (command jobs, SDK and CLI)](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-train-model)
- [Command job YAML schema](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-job-command)
- [Configure MLflow for Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-mlflow-configure-tracking)
- [Register and work with models](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-models)
- [Deploy to online endpoints](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints)
