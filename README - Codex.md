# AI Services Explained

GitHub repository: https://github.com/yves-schillings/ai-services-explained
Twelve commented Python examples accompany the article in preparation.

The examples use fictional data. Start with examples 01 and 02, then the negative
tests. They never download model weights or call an external hosted model API.
Examples 03 to 06 and 09 to 11 require a model already installed in a local Ollama
server. The HTTP client is fixed to 127.0.0.1 and disables inherited proxies.
This does not establish the security or offline configuration of that server.

## Execution

Python 3.11 or later. For the neural-network example install NumPy. For speech,
install faster-whisper and provision its CTranslate2 model files separately.
OCR requires the Tesseract executable and selected language data on PATH.
Check upstream package licences and model licences before deployment.

Run a filename in quotes, for example:

```powershell
python "examples/01_machine_learning - Codex.py"
python "examples/02_deep_learning - Codex.py"
python "tests - Codex.py"
```

Every example starts with its command, prerequisites and scope. Read the comments
before enabling a model call. These are educational applications, not a complete
authentication service, CPE controller or production deployment bundle.

## Reading order

01 fitted classification, 02 neural learning, 03 local language inference,
04 controlled draft generation, 05 permitted semantic retrieval and generation,
06 bounded tool use, 07 OCR, 08 transcription, 09 multimodal interpretation,
10 embeddings, 11 translation and 12 elementary redaction.

The article has one business function, application flow and infrastructure diagram
for each main family. Article publication and GitHub publication remain pending
approval of the exact article document. No client names or client data are included.

## Verification

22 deterministic tests passed on 21 October 2026. The fitted classifier and toy
neural network were executed. Other tests cover access filtering, schema refusal,
agent tool permissions and bounded execution, with mocked model or OCR calls.
Real language, embedding, vision, OCR and speech quality was not evaluated.

## Examples

- [01_machine_learning - Codex.py](examples/01_machine_learning%20-%20Codex.py)
- [02_deep_learning - Codex.py](examples/02_deep_learning%20-%20Codex.py)
- [03_language_model - Codex.py](examples/03_language_model%20-%20Codex.py)
- [04_generative_ai - Codex.py](examples/04_generative_ai%20-%20Codex.py)
- [05_rag - Codex.py](examples/05_rag%20-%20Codex.py)
- [06_agent - Codex.py](examples/06_agent%20-%20Codex.py)
- [07_vision_ocr - Codex.py](examples/07_vision_ocr%20-%20Codex.py)
- [08_speech - Codex.py](examples/08_speech%20-%20Codex.py)
- [09_multimodal - Codex.py](examples/09_multimodal%20-%20Codex.py)
- [10_embeddings - Codex.py](examples/10_embeddings%20-%20Codex.py)
- [11_translation - Codex.py](examples/11_translation%20-%20Codex.py)
- [12_redaction - Codex.py](examples/12_redaction%20-%20Codex.py)

