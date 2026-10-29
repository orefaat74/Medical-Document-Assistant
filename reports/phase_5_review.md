# Phase 5 — Complete capstone application and handoff

## Application

The project now provides a FastAPI backend and Streamlit interface for:

- Uploading text-based PDF, DOCX, TXT/MD, or HTML, or pasting text.
- Classifying the document with the validation-selected TextCNN or another saved model.
- Inspecting dosage, warnings, contraindications, side effects, uses, and interactions
  copied from source headings with exact character offsets.
- Asking explicit questions about the current document, with quoted evidence or
  the missing-answer fallback.
- Clearing/replacing documents and resetting the conversation.

The runtime caches model instances and uses separate opaque API document handles.
Each handle refers to one canonical source. Vector indexes are separate from
training artifacts; clearing a final reference deletes its runtime index. The
application evaluates English text and does not claim OCR, Arabic QA, calibrated
clinical confidence, or expert-adjudicated category labels.

## Open and review

- Temporary HTTPS demo: <https://stories-acknowledged-cord-connector.trycloudflare.com>
- Local Streamlit: <http://127.0.0.1:8501>
- Local FastAPI docs: <http://127.0.0.1:8000/docs>
- Docker Streamlit: <http://127.0.0.1:7860>
- Docker FastAPI docs: <http://127.0.0.1:8001/docs>

The temporary URL depends on the local app and Cloudflare tunnel remaining active.
The user selected temporary-demo/deployment handoff and the source package. A
persistent Hugging Face Space and a GitHub repository were therefore not published.
The deployment script and source files are ready for the owner's account-backed
publication step. See `../docs/deployment.md`.

## Verification evidence

- `phase5/integration_results.json`: **14 real local API/browser checks passed**.
- `phase5/container_integration_results.json`: **9 real Linux-container API checks passed**.
- `phase5/public_demo_verification.json`: **5 HTTPS/browser/websocket/chat checks passed**.
- `phase5/container_build.json`: Docker image built successfully; approximately **1.71 GB**.
- Container reported **healthy** under Docker Compose.
- Full software regression suite: **80 tests passed**, including 12 ingestion/API tests.
- Dependency consistency: `pip check` passed.
- Architecture, slide PDF/PPTX, speaker notes, screenshots, and source handoff were generated.

These checks exercised real trained classification, exact source extraction,
positive/missing QA, PDF and DOCX reading, model/data lifecycle, and browser upload
and chat. They supplement the earlier classifier/QA artifact checks rather than
establishing medical reliability.

## Measured example response times

| Workflow | Observed local warm response | Observed container response |
|---|---:|---:|
| TextCNN classification | 8.48 ms | 1,370.16 ms, first model load |
| Source extraction | 18.88 ms | 5.64 ms |
| Grounded full-effect answer | 360.14 ms | 4,314.30 ms, first QA load |
| Missing-answer fallback | 26.07 ms | 31.54 ms |
| Text-based PDF upload | 15.58 ms | 17.49 ms |
| DOCX upload | 18.81 ms | 23.09 ms |

These are individual example measurements, not load-test statistics. Model caching
changes cold-versus-warm latency. Full records are in the integration JSON files.

## Final model comparison

| Model | Validation macro F1 | Test accuracy | Test macro F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.8355 | 97.66% | 0.9799 |
| Linear SVM | 0.8565 | 95.61% | 0.9622 |
| Random Forest | 0.8671 | 97.08% | 0.9751 |
| TextCNN — application default | 0.9972 | 98.54% | 0.9876 |
| Compact fine-tuned BERT | 0.9944 | 95.32% | 0.9597 |

The common source-derived corpus contains 2,471 labels and six targets, with
ingredient-aware record splits. Context budgets and model capacities differ.
The QA benchmark uses 20 evaluation cases across three leaflets; the initial
frequency/strength mismatch and post-correction regression result are both
preserved. See the Phase 1–4 reports for full method/limits.

## Deliverables

- `../streamlit_app.py` — upload/paste, source results, model comparison, evidence chat.
- `../src/medical_doc_assistant/api.py` — documented FastAPI endpoints.
- `../Dockerfile` and `../docker-compose.yml` — verified container packaging.
- `../scripts/deploy_huggingface.py` — persistent Docker Space staging/deployment.
- `../docs/architecture.md`, `.png`, `.svg` — architecture description and visuals.
- `../deliverables/Final_Presentation.pptx` and `.pdf` — eight slides.
- `../deliverables/Speaker_Notes.md` — 741 words, approximately 6–7 minutes including the demo.
- `../deliverables/Source_Code_Handoff.zip` — repository-ready source/config/tests/docs.
- `phase5/screenshots/` — actual local/public browser evidence.

## Remaining scope and operation

Persistent hosting/repository publication was deferred by the chosen handoff.
The English-text app and current trained artifacts are usable locally and in the
verified Docker image. Arabic model evaluation, OCR/handwriting, broader independent
QA cases, and production account/identity operations are documented extensions.

Data provided by the U.S. Food and Drug Administration (<https://open.fda.gov>).
Pretrained model credits/licenses are in `../docs/phase4_model_cards.md`.

As specified in the original plan, this is informational only and is not medical
advice. Source quotations do not independently certify the source's clinical claims.
