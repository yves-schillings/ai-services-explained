"""Mask obvious email addresses in a fictional note.

Run: python "12_redaction - Codex.py"
This is a baseline control, NOT an anonymisation system or permission to export data.
Names, indirect identifiers and re-identification remain outside this detector.
"""
import re,json
EMAIL=re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

def redact(text):
    return EMAIL.sub("[EMAIL REDACTED]",text)

if __name__=="__main__":
    print(json.dumps({"text":redact("Fictional contact: learner@example.org"),
                      "anonymisation_established":False}))
