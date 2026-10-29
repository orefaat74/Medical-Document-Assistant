# Phase 4 verification

Verified on **2026-10-28**, Python **3.10.6**, PyTorch **2.8.0+cpu**. The pinned NLP
versions are in `requirements-phase4.txt`; source model revisions/licenses and
file hashes are in `model_assets.json`.

| Check | Result |
|---|---|
| Compact BERT experiment | Two learning-rate candidates fine-tuned; validation-selected model refit and evaluated |
| Full software suite | **68 tests passed** |
| Integrated artifact verification | **54 checks passed** |
| Review notebook | **16 code cells executed successfully**, outputs saved and schema validated |
| BERT CLI inference | Real development example classified as `acid_reducer` |
| Interactive QA CLI | Supported full-effect question quoted the source; missing retail-price question returned fallback |
| Dependency consistency | `pip check`: no broken requirements |

## Recorded results

- BERT base: Google four-layer BERT-mini, hidden size 256.
- BERT test accuracy / macro F1: **95.32% / 0.9597**.
- TextCNN comparison: **98.54% / 0.9876**; the BERT result is retained as measured.
- QA corpus: **48 cases / 7 real source documents**.
- Corrected QA evaluation: **20 cases / 3 documents**, 11 answerable and 9 absent facts.
- Missing-answer fallback: **9 / 9**.
- Positive reference evidence coverage / source-citation support: **11 / 11** returned positive answers.

The initial QA result covered 10 of 11 reference facts: one frequency question
returned a strength description. A generic temporal-answer check corrected it;
the original policy-v1 files are preserved in `rag/archive/policy_v1/`. Updated
scores are regression results after that software correction, not a fresh unseen
test estimate. The benchmark is narrow and template-heavy; it is not a claim of
clinical validation or universal medical QA accuracy.

## Verification coverage

Checks cover classifier source/split identity, pinned model files, learning-rate
selection, best validation epochs, fixed refit length, complete BERT weights,
source predictions/metrics/confusion counts, and bounded WordPiece coverage.
QA checks cover real copied source texts, literal positive reference answers,
document-group isolation, recorded policy/data hashes, independent normalized
FAISS indexes, source-citation offsets, required fallback outputs, and the absence
of rejected generated text in responses.

The software tests exercise real BERT encoder gradient flow, window padding,
checkpoint/tokenizer round trips, complete chunking despite prior tokenizer
truncation, document isolation, checked LLM span acceptance, rejection of invented
values, explicit source-section quoting, temporal answer types, and index tampering.

## Reproduction commands

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[ml,dl,nlp,notebooks]" -c "requirements-phase4.txt"
.\.venv\Scripts\python.exe scripts/fetch_phase4_models.py
.\.venv\Scripts\python.exe -m medical_doc_assistant phase4
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/execute_notebook.py notebooks/04_bert_and_grounded_qa.ipynb
.\.venv\Scripts\python.exe -m medical_doc_assistant classify-bert --file "data/examples/phase2_demo_leaflet.txt"
.\.venv\Scripts\python.exe -m medical_doc_assistant chat --file "data/examples/phase2_demo_leaflet.txt"
.\.venv\Scripts\python.exe -m pip check
```

Completed classifier/evaluation artifacts are reused when their hashes and
settings match. For the original long BERT training step, the project used
`scripts/train_bert_phase4.py`, which records progress/status; classifier training
was not repeated while the independent retrieval/QA code was built.

As required by the project plan, the product is informational only and is not
medical advice.

**Phase 4 is complete. Phase 5 awaits your review.**
