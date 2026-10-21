"""Draft a translation with a separately provisioned local language model.
Run: python "11_translation - Codex.py" --model YOUR_LOCAL_MODEL
Numbers, negation and domain terms require a bilingual reviewer.
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

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--model",required=True);a=p.parse_args()
    print(chat(a.model,[{"role":"system","content":"Translate French to Dutch. Preserve LAB-01, 10:12 and negation. Output only the translation."},
                        {"role":"user","content":"Le serveur LAB-01 a redémarré à 10:12. La cause n’est pas confirmée."}]))
