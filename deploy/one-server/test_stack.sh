#!/bin/bash
# End-to-end test of the one-server stack. Author: Yves Schillings.
# Run from deploy/one-server:  bash test_stack.sh
# Writes every command, its output and its duration to results/one_server_test_log.txt.
# Pulls two small open models into the ollama-models volume (never into the repository).
set -u
LOG=../../results/one_server_test_log.txt
: > "$LOG"
say() { echo "$*" | tee -a "$LOG"; }
step() {                                   # step "title" command...
  local title=$1; shift
  say ""; say "### $title"; say "\$ $*"
  local t0=$(date +%s%N)
  "$@" 2>&1 | tail -n 40 | tee -a "$LOG"
  local code=${PIPESTATUS[0]}
  local ms=$(( ($(date +%s%N) - t0) / 1000000 ))
  say "(exit $code, $((ms / 1000)).$(( (ms % 1000) / 100 )) s)"
}
wait_for() {                               # wait_for url seconds
  for _ in $(seq 1 "$2"); do curl -fs "$1" > /dev/null 2>&1 && return 0; sleep 1; done; return 1
}
json() { curl -s "$@"; echo; }

say "One-server stack test, $(date -u +%Y-%m-%dT%H:%M:%SZ)"
step "Versions" bash -c "docker version --format 'Docker {{.Server.Version}}'; docker compose version; \
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"
step "Build the image (examples and application code)" docker compose build
step "Clean start: remove the test data volumes, keep the model volume" bash -c   "docker compose --profile jobs down --remove-orphans; docker volume rm ai-one-server_postgres-data ai-one-server_mlflow-artifacts    ai-one-server_approvals ai-one-server_job-results 2>&1 | sed 's/^/removed or absent: /'"
step "Start PostgreSQL, MLflow and the model server" docker compose up -d postgres mlflow ollama
step "Wait for the MLflow tracking server" wait_for http://127.0.0.1:5055/health 120
step "The GPU is assigned to the model-server container only" bash -c \
  "docker compose exec -T ollama nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; \
   docker compose run --rm --no-deps -T jobs sh -c 'ls /dev/nvidia* 2>/dev/null || echo no GPU device in the jobs container'"
step "Pull a small LLM (qwen2.5:0.5b, Apache 2.0) into the model volume" docker compose exec -T ollama ollama pull qwen2.5:0.5b
step "Pull a small embedding model (all-minilm, Apache 2.0)" docker compose exec -T ollama ollama pull all-minilm
step "Models in the volume" docker compose exec -T ollama ollama list
step "The application refuses to start without an approved version" bash -c \
  "docker compose up -d app; sleep 25; docker compose ps -a app --format '{{.Name}} {{.State}} {{.Status}}'; \
   docker compose logs app | grep -E 'RefusedModel|No approved' | tail -2"
step "Training job: three runs logged to MLflow, three versions registered" \
  docker compose run --rm -T jobs python examples/01_machine_learning.py
step "Release job: the gate and the named owner decide" docker compose run --rm -T jobs python examples/19_model_release.py
step "Start the application again" bash -c "docker compose up -d --force-recreate app"
step "Wait for the application" wait_for http://127.0.0.1:8800/health 180
step "Health: the approved version is served" json http://127.0.0.1:8800/health
step "Classify a document" json -X POST http://127.0.0.1:8800/classify -H 'Content-Type: application/json' \
  -d '{"text": "witness saw a person near the harbour in the evening"}'
step "Warm-up: load the LLM into GPU memory (the first call also prepares the GPU kernels)" \
  docker compose exec -T ollama ollama run qwen2.5:0.5b "Reply with OK."
step "Ask a question: pgvector retrieval, then the LLM on the GPU" json -X POST http://127.0.0.1:8800/ask \
  -H 'Content-Type: application/json' -d '{"question": "What happens when the type is proposed with low confidence?"}'
step "Ask a second question" json -X POST http://127.0.0.1:8800/ask \
  -H 'Content-Type: application/json' -d '{"question": "How do we go back to the previous model version?"}'
step "Where the LLM runs (size_vram > 0 means GPU memory)" json http://127.0.0.1:8800/models
step "Rollback: the owner withdraws the live version" \
  docker compose run --rm -T jobs python examples/19_model_release.py --rollback "Rollback test of the one-server stack"
step "Restart the application" bash -c "docker compose restart app"
step "Wait for the application" wait_for http://127.0.0.1:8800/health 180
step "Health after rollback: the previous approved version is served" json http://127.0.0.1:8800/health
step "Separation that must be configured: the model server cannot reach PostgreSQL" \
  docker compose exec -T ollama sh -c "getent hosts postgres || echo 'postgres is not reachable from the model-server network'"
step "Separation that must be configured: the retrieval user cannot open the MLflow database" \
  docker compose exec -T postgres psql -U rag -d mlflow -c "select 1"
step "Stop the stack (volumes kept: models, metadata, artifacts, approvals)" docker compose down
step "No container of the stack is running" bash -c "docker ps --filter name=ai-one-server --format '{{.Names}}' | wc -l"
say ""; say "End of test."
