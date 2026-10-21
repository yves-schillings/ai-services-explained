"""Ask a preinstalled LOCAL language model a question.

Run: python "03_language_model - Codex.py" --model YOUR_LOCAL_MODEL
Provision and license the model separately. This script never downloads weights.
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

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--model", required=True)
    args = p.parse_args()
    print(chat(args.model, [{"role": "system", "content": "Explain clearly. Say when unsure."},
                           {"role": "user", "content": "Explain the difference between training and inference."}]))
