"""Ask a separately provisioned local vision-language model about an image.

Run: python "09_multimodal - Codex.py" sample.png --model YOUR_LOCAL_VISION_MODEL
The selected model must actually support images. No OCR accuracy claim.
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

from pathlib import Path
import base64

def image_message(path):
    path=Path(path).resolve(strict=True)
    if path.suffix.lower() not in {".png",".jpg",".jpeg"}:raise ValueError("Unsupported image")
    if path.stat().st_size>10*1024*1024:raise ValueError("Image exceeds example limit")
    return {"role":"user","content":"Describe the visible layout. Do not infer identities or hidden facts. Say what is unreadable.",
            "images":[base64.b64encode(path.read_bytes()).decode("ascii")]}

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("image");p.add_argument("--model",required=True);a=p.parse_args()
    print(chat(a.model,[image_message(a.image)]))
