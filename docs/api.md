# API usage

Interactive schema: <http://127.0.0.1:8000/docs> locally, or port 8001 for Compose.

| Route | Method | Purpose |
|---|---|---|
| `/health` | GET | Runtime/default-model readiness |
| `/models` | GET | Saved classifier options and recorded benchmark metrics |
| `/classify` | POST | Classify English text with a selected model |
| `/extract` | POST | Copy named source sections with offsets |
| `/documents` | POST | Analyze pasted text and return an opaque document handle |
| `/documents/upload` | POST | Analyze multipart PDF/DOCX/HTML/text upload |
| `/documents/{id}` | GET | Read the active analyzed document |
| `/documents/{id}` | DELETE | Clear its handle/runtime index reference |
| `/ask` | POST | Ask the supplied document and return source evidence/fallback |

```python
import requests

base = "http://127.0.0.1:8000"
text = open("data/examples/phase2_demo_leaflet.txt", encoding="utf-8").read()
document = requests.post(base + "/documents", json={"text": text, "source_name": "Example leaflet"}).json()
answer = requests.post(base + "/ask", json={
    "document_id": document["document_id"],
    "question": "How long can this product take for full effect?",
}).json()
print(answer["answer"])
print(answer["citations"])
requests.delete(base + "/documents/" + document["document_id"])
```

File upload uses multipart field `file`; optional form field `model` selects one
of `text_cnn`, `random_forest`, `logistic_regression`, `linear_svm`, or `bert_mini`.
Default is TextCNN. All model outputs describe the six source-derived categories.

Limits and one-hour document-handle lifetime are in `configs/phase5.json`.
Missing/expired handles return 404, invalid inputs 422, and absent model artifacts
503. Dedicated missing extraction fields remain empty. The source UUID is an
opaque demo handle, not an authenticated production identity.
