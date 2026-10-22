# Azure Machine Learning: train, approve, deploy

Author: Yves Schillings. **Status: prepared, not executed in Azure.**

## Files

| File | Role |
| --- | --- |
| `job_triage.yml` | CLI v2 command job that trains the document classifier on `cpu-cluster` |
| `job_page_cnn_gpu.yml` | Prepared GPU variant for the CNN. It explicitly uses CUDA for the model and batches and fails if CUDA is unavailable. The small example only needs CPU |
| `compute_cpu.yml`, `compute_gpu.yml` | Compute clusters with `min_instances: 0`. After the idle timeout, nodes are deallocated, avoiding VM compute charges while idle. Storage and other provisioned resources can still incur charges |
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
  - no VM compute charges after the cluster scales to zero. This does not remove storage or other provisioned-resource charges
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
- **CNN export:** GPU training exports the traced MLflow model for CPU inference. On a machine with CUDA,
  load that export with `mlflow.pyfunc.load_model(path, model_config={"device": "cpu"})`.
  A traced `pt2` graph does not move its weights between CPU and CUDA at load time. GPU serving needs a
  separate CUDA export, or a reconstruction of `SmallCNN` from the saved `page_layout_cnn.pt` state dictionary.

## Documentation used (checked 22 October 2026)

- [Train models (command jobs, SDK and CLI)](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-train-model)
- [Compute clusters, scaling to zero and avoiding idle VM charges](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster?view=azureml-api-2)
- [Command job YAML schema](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-job-command)
- [Configure MLflow for Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-mlflow-configure-tracking)
- [Register and work with models](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-models)
- [Deploy to online endpoints](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints)
