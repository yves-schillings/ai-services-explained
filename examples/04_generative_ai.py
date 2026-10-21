"""Generate a structured draft from a fictional incident note.

Run: python "04_generative_ai.py" --model YOUR_LOCAL_MODEL
JSON structure validation does not establish factual accuracy. Review is mandatory.
"""

import json
from urllib.request import Request, build_opener, ProxyHandler

def local_api(path, payload):
    """Call only a loopback Ollama server. No proxy and no model download."""
    if path not in {"chat", "embed"}:
        raise ValueError("Unsupported endpoint")
    request = Request("http://127.0.0.1:11434/api/" + path,
                      json.dumps(payload).encode(),
                      {"Content-Type": "application/json"})
    # Disable inherited proxy settings for this explicitly local example.
    with build_opener(ProxyHandler({})).open(request, timeout=120) as response:
        return json.load(response)

def chat(model, messages, schema=None):
    payload = {"model": model, "messages": messages, "stream": False,
               "options": {"temperature": 0, "num_predict": 300}}
    if schema is not None:
        payload["format"] = schema
    return local_api("chat", payload)["message"]["content"]

SCHEMA = {"type": "object", "properties": {
    "summary": {"type": "string"}, "next_action": {"type": "string"}},
    "required": ["summary", "next_action"], "additionalProperties": False}

def validate(value):
    if set(value) != {"summary", "next_action"}:
        raise ValueError("Unexpected output fields")
    if not all(isinstance(x, str) and x.strip() for x in value.values()):
        raise ValueError("Expected nonempty strings")
    return value

if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--model", required=True); a=p.parse_args()
    note="Fictional example: server LAB-01 stopped at 10:00. Restarted at 10:12. Cause unknown."
    answer=chat(a.model, [{"role":"system", "content":"Draft only from the note. Do not invent a cause. Return JSON."},
                          {"role":"user", "content":note}], SCHEMA)
    print(json.dumps({"draft":validate(json.loads(answer)), "human_review_required":True}, indent=2))
