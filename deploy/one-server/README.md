# One server: model server, MLflow, PostgreSQL and the application on one GPU machine

Author: Yves Schillings.

Status: **executed**. The stack below was started, used and stopped on one Windows
workstation with Docker Desktop and one NVIDIA GeForce RTX 2080 Ti (11 GB). The full
log, with every command, its output and its duration, is in
[`results/one_server_test_log.txt`](../../results/one_server_test_log.txt).

## What runs where

| Container | Image | Role | Data it keeps |
| --- | --- | --- | --- |
| `ollama` | `ollama/ollama:0.40.8` | Model server for the LLM and the embedding model. The only container that receives the GPU | `ollama-models` volume |
| `mlflow` | built from `Dockerfile` (MLflow 3.17.2) | Tracking server and model registry, artifacts served through the server | `mlflow-artifacts` volume |
| `postgres` | `pgvector/pgvector:pg16` (PostgreSQL 16.15) | Metadata database of MLflow, and vector index for retrieval, in two databases with two users | `postgres-data` volume |
| `app` | built from `Dockerfile` | FastAPI application: the approved triage model (`/classify`) and questions answered by the LLM over retrieved passages (`/ask`) | reads `approvals` read-only |
| `jobs` | built from `Dockerfile` | One-off jobs: training (`01_machine_learning.py`) and release decisions (`19_model_release.py`) | writes `approvals`, `job-results` |

Two networks separate the traffic. `data` connects PostgreSQL to MLflow and the
application only. `apps` connects the application, MLflow, the jobs and the model server.
The model server cannot reach PostgreSQL. Every container except the model server has
CPU and memory limits. The image runs as an unprivileged user.

## Run it

```bash
cd deploy/one-server
docker compose build
docker compose up -d postgres mlflow ollama
docker compose exec ollama ollama pull qwen2.5:0.5b      # Apache 2.0, 397 MB, into the volume
docker compose exec ollama ollama pull all-minilm         # Apache 2.0, 45 MB, embeddings
docker compose run --rm jobs python examples/01_machine_learning.py
docker compose run --rm jobs python examples/19_model_release.py
docker compose up -d app                                  # http://127.0.0.1:8800/docs
docker compose run --rm jobs python examples/19_model_release.py --rollback "reason"
docker compose restart app                                # serves the previous approved version
docker compose down                                       # add -v to delete the volumes
```

The MLflow interface is on http://127.0.0.1:5055. `bash test_stack.sh` runs the whole
test and writes the log. The credentials in `docker-compose.yml` are local test values:
replace them before any other use.

## What the test showed

Run of 22 October 2026: Docker 29.8.2, Docker Compose 5.5.1, NVIDIA driver 617.14.
Durations are from the log. The models were already in their volume, because an
earlier run had pulled them.

| Step | What happened | Duration |
| --- | --- | --- |
| Build the image | Cached rebuild (the first build took 3 min 16 s) | 2.8 s |
| Start PostgreSQL, MLflow and the model server | All running, then MLflow answers | 7.9 s + 12.6 s |
| GPU per container | The RTX 2080 Ti is visible in `ollama`. The `jobs` container has no GPU device | 3.0 s |
| Models in the volume | `qwen2.5:0.5b` (397 MB) and `all-minilm` (45 MB) | 2.1 s + 1.9 s |
| Application before any approval | It stops with `RefusedModel: No approved version of document-triage` | 28.9 s |
| Training job | Three MLflow runs, versions 1 to 3 registered. Test macro-F1 0.820 (gate failed), 0.888 and 0.918 | 39.9 s |
| Release job | Version 1 denied by the validation code, versions 2 and 3 approved. Version 3 is live | 5.4 s |
| Application start | `/health` shows version 3. `/classify` returns `witness_statement`, confidence 0.8928 | 39.6 s to ready |
| First LLM call after start | The model loads into GPU memory | 139.1 s |
| Two questions (`/ask`) | pgvector returns passages P1 and P6, then P4 and P6. The LLM answers at 152 and 149 tokens per second. Both models are fully in GPU memory | 3.9 s and 0.3 s |
| Rollback | Version 3 withdrawn, application restarted, `/health` shows version 2 | 4.8 s + 3.0 s + 4.7 s |
| Network separation | The model server cannot resolve `postgres` | 4.8 s |
| Database separation | The retrieval user gets `permission denied for database "mlflow"`. This refusal is the expected result (exit code 2) | 0.6 s |
| Stop | `docker compose down`. No container of the stack keeps running | 6.7 s |

What the test also taught:

- **Threads must match the CPU limit.** In a container limited to 4 CPUs, the training
  job first took 772.6 s, because the numeric libraries started more threads than the
  container could run. With `OMP_NUM_THREADS=4` and `OPENBLAS_NUM_THREADS=4` it took 39.9 s.
- **The first LLM call after a start is slow.** It took 139.1 s, then each answer took
  under 4 s. The likely cause is that the GPU code is prepared for this card at the
  first call. That cause was not investigated further.
- **Answer quality is not tested.** The second answer of the 0.5-billion-parameter model
  does not cite its passage. The test proves the path, not the quality.
- **The planned model could not be pulled.** `qwen3:0.6b` failed with `Error: EOF` on
  Ollama 0.40.8, so the test used `qwen2.5:0.5b` (Apache 2.0). The cause was not investigated.
- **Port 5000 was already in use** on the test machine, so MLflow is published on
  127.0.0.1:5055 (`MLFLOW_PORT`).

Not tested here:

- vLLM (Ollama was used);
- MinIO (a named volume holds the artifacts);
- TLS;
- user login and audit;
- backups;
- several projects side by side.

## What one server does well, and where it stops

Good for a pilot, a small team, learning, and fictional or approved data.

- **Containers isolate processes. The separation between projects still has to be
  configured.** Here it is done by hand: two networks, separate volumes, one database
  user per database, CPU and memory limits, a read-only approval volume and an
  unprivileged user. A second project would need its own set, and nothing enforces quotas
  between projects.
- **Versions and rollback do not need a platform.** The MLflow registry keeps the
  versions, the approval record names the live one, and the deployment procedure
  (restart the application) applies the decision. KServe is not required for that.
- **GPUs are assigned per container.** Compose reserves the GPU for the model server
  only. Sharing one GPU between several model servers, or scaling them, stays manual.
- **One server stays a single point of failure.** There is no high availability.
  Installing Kubernetes on that same machine does not remove it.
- Access control, audit, updates and backups must be built by hand.

MLflow is not needed just to call a pretrained LLM: the `/ask` path uses no MLflow at
all. MLflow earns its place for experiments, evaluations and model versions, as in the
training and release jobs.

## When to move to a platform

Several projects that need isolation, shared GPU pools, serving with versions and
rollback across machines (for example KServe), repeatable training pipelines
(Kubeflow Pipelines or Data Science Pipelines), and central identity, quotas and
monitoring. Kubernetes, or Red Hat OpenShift with OpenShift AI, provides these. This
repository does not test a platform.
