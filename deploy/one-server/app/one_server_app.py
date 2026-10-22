"""The application of the one-server stack: the approved triage model and a local LLM over retrieved passages.

Author: Yves Schillings.

It runs in the "app" container of deploy/one-server/docker-compose.yml:
  * /classify  serves the approved document-triage version, loaded exactly as in
               examples/18_model_serving.py (approval record, gate, digest check),
  * /ask       finds the two closest fictional passages in PostgreSQL + pgvector and
               asks the model server (Ollama) to answer from those passages only,
  * /health    shows the approved version and the models known to the model server.
A pretrained LLM needs no MLflow to be called. MLflow serves the trained triage model,
its experiments and its versions.
"""
import importlib.util
import os
from pathlib import Path

import httpx
import psycopg
from pydantic import BaseModel, Field

EXAMPLES = Path(os.environ.get("PYTHONPATH", "/app/examples").split(os.pathsep)[0])
_spec = importlib.util.spec_from_file_location("model_serving", EXAMPLES / "18_model_serving.py")
serving = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(serving)

OLLAMA = os.environ.get("OLLAMA_URL", "http://ollama:11434")
LLM = os.environ.get("LLM_MODEL", "qwen2.5:0.5b")
EMBEDDER = os.environ.get("EMBEDDING_MODEL", "all-minilm")
DATABASE = os.environ["RAG_DATABASE_URL"]

# Fictional passages about the document intake procedure.
PASSAGES = {
    "P1": "A document whose type is proposed with a confidence below 0.5 goes to a reviewer before routing.",
    "P2": "Witness statements are routed to the case officer within one working day.",
    "P3": "Evidence inventories must list the seal number of every exhibit bag.",
    "P4": "A new model version goes live only after the release owner approves it in the approval record.",
    "P5": "Correspondence is filed with the case and does not change the case status.",
    "P6": "If a live model misbehaves, the release owner withdraws it and the previous approved version serves again.",
}


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)


def embed(texts):
    r = httpx.post(f"{OLLAMA}/api/embed", json={"model": EMBEDDER, "input": texts}, timeout=120)
    r.raise_for_status()
    return r.json()["embeddings"]


def vector(values):
    return "[" + ",".join(f"{v:.6f}" for v in values) + "]"


def index_passages():
    """Create the vector table on first start and embed the passages once."""
    with psycopg.connect(DATABASE) as db:
        dim = len(embed(["dimension check"])[0])
        db.execute(f"CREATE TABLE IF NOT EXISTS passages (id text PRIMARY KEY, body text, embedding vector({dim}))")
        if db.execute("SELECT count(*) FROM passages").fetchone()[0] == 0:
            for (pid, body), emb in zip(PASSAGES.items(), embed(list(PASSAGES.values()))):
                db.execute("INSERT INTO passages VALUES (%s, %s, %s::vector)", (pid, body, vector(emb)))


app = serving.create_app()
index_passages()


@app.post("/ask")
def ask(q: Question):
    with psycopg.connect(DATABASE) as db:
        rows = db.execute("SELECT id, body FROM passages ORDER BY embedding <=> %s::vector LIMIT 2",
                          (vector(embed([q.question])[0]),)).fetchall()
    context = "\n".join(f"[{pid}] {body}" for pid, body in rows)
    messages = [{"role": "system", "content": "Answer in one sentence, only from the passages, and cite the "
                                              "passage identifier. If the passages do not answer, say so."},
                {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {q.question}"}]
    request = {"model": LLM, "messages": messages, "stream": False,
               "options": {"temperature": 0, "num_predict": 120}}
    if LLM.startswith("qwen3"):
        request["think"] = False                     # answer directly, without a reasoning trace
    r = httpx.post(f"{OLLAMA}/api/chat", json=request, timeout=300)
    r.raise_for_status()
    out = r.json()
    return {"answer": out["message"]["content"].strip(), "passages": [pid for pid, _ in rows], "model": LLM,
            "generated_tokens": out.get("eval_count"),
            "tokens_per_second": round(out["eval_count"] / (out["eval_duration"] / 1e9), 1)
            if out.get("eval_duration") else None}


@app.get("/models")
def models():
    return httpx.get(f"{OLLAMA}/api/ps", timeout=30).json()
