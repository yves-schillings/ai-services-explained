"""How the application calls the deployed model (step 7). STATUS: prepared.

Author: Yves Schillings.

The endpoint uses Microsoft Entra ID authentication (auth_mode aad_token in
endpoint.yml): the application gets a token for its own identity and sends it with each
request. No shared key is stored in the application.

  set AZUREML_SCORING_URI=https://document-triage.<region>.inference.ml.azure.com/score
  python cloud/azure/call_endpoint.py "worker remembers she saw a person near the hall"

Data sent: the document text, to the endpoint in the workspace region. The answer is a
label, a confidence and a review flag. A person types the documents flagged for review.
"""
import json
import os
import sys
from urllib.request import Request, urlopen

SCOPE = "https://ml.azure.com/.default"


def classify(texts, scoring_uri=None, token=None):
    scoring_uri = scoring_uri or os.environ["AZUREML_SCORING_URI"]
    if token is None:
        from azure.identity import DefaultAzureCredential
        token = DefaultAzureCredential().get_token(SCOPE).token
    request = Request(scoring_uri, data=json.dumps({"texts": texts}).encode(),
                      headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


if __name__ == "__main__":
    for answer in classify(sys.argv[1:] or ["worker remembers she saw a person near the hall"]):
        print(answer)
