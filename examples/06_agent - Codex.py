"""A bounded local-model agent can read one fictional ticket and then stop.

Run: python "06_agent - Codex.py" --model YOUR_LOCAL_MODEL
No shell, no arbitrary file access, no write tool. Test actor is fixed by the app.
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

SCHEMA={"type":"object","properties":{"tool":{"type":"string","enum":["read_ticket","finish"]},
 "ticket_id":{"type":"string"}},"required":["tool","ticket_id"],"additionalProperties":False}
TICKETS={"T1":{"owner":"alice","status":"Awaiting operator review"}}

def execute(action, actor):
    if set(action)!={"tool","ticket_id"}:raise ValueError("Invalid tool arguments")
    if action["tool"]=="finish":return "finished"
    if action["tool"]!="read_ticket":raise PermissionError("Tool not allowed")
    ticket=TICKETS.get(action["ticket_id"])
    if not ticket or ticket["owner"]!=actor:raise PermissionError("Ticket not permitted")
    return {"ticket_id":action["ticket_id"],"status":ticket["status"]}

def run_agent(propose, actor="alice", max_steps=3):
    observations=[]
    for _ in range(max_steps):
        action=propose(observations)
        result=execute(action,actor)  # Application enforces rights after every proposal.
        observations.append(result)
        if result=="finished":return observations
    raise RuntimeError("Agent step limit reached")

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--model",required=True);a=p.parse_args()
    def propose(observations):
        return json.loads(chat(a.model,[{"role":"system","content":"Choose read_ticket T1 once, then finish. Return tool and ticket_id as JSON."},
          {"role":"user","content":json.dumps({"task":"Check T1 status","observations":observations})}],SCHEMA))
    print(json.dumps(run_agent(propose),indent=2))
