# Medical Document Assistant

An AI Engineering final project based on [Final_Project_Plan.pdf](Final_Project_Plan.pdf):
classify medicine documents, retain their clinical sections, and eventually answer
questions using only the uploaded document. The course plan covers five phases.

## Final project — complete capstone application

The five course phases now form a working **FastAPI + Streamlit application**:
upload or paste a document, view its category and source sections, and ask questions
with quoted evidence. Text-based PDF, DOCX, TXT/MD, and HTML uploads are supported.
The evaluated scope is English text; the app is informational only, not medical advice.

The current document experience also provides:

- Document-bound follow-up questions such as `dosage` → `how often?` →
  `what about children?`; changing documents clears that context.
- A source-backed overview for identity, form, strength, route, uses, dosage, and
  key restrictions, including a warning for inconsistent medicine names.
- Suggested questions generated only from topics detected in the current document.
- Evidence labels with PDF page numbers, DOCX paragraph/table locations, and a
  **Jump to source** action that highlights the exact cited text.

Classifier confidence, persistent saved workspaces/report exports, expanded QA
benchmarking, and loading-performance work are intentionally deferred.

### Open the application

- **Temporary live demo:** <https://stories-acknowledged-cord-connector.trycloudflare.com>
- Local frontend: <http://127.0.0.1:8501>
- Local API/Swagger: <http://127.0.0.1:8000/docs>
- Docker frontend: <http://127.0.0.1:7860>
- Docker API/Swagger: <http://127.0.0.1:8001/docs>

The public URL is a temporary Cloudflare review tunnel. It depends on this local
application and tunnel remaining active. The agreed handoff includes the Docker
image, persistent Hugging Face deployment script, and source package.

```powershell
# Local launch with the supplied virtual environment:
.\.venv\Scripts\python.exe scripts/run_app.py

# Container launch:
docker compose up --build -d
```

### Capstone deliverables

- [`reports/phase_5_review.md`](reports/phase_5_review.md) — final app, verification, timings, deployment status, and handoff.
- [`docs/architecture.md`](docs/architecture.md) and [`docs/architecture.svg`](docs/architecture.svg) — serving/training architecture.
- [`docs/deployment.md`](docs/deployment.md) — local, Docker, temporary-demo, and persistent-hosting instructions.
- [`deliverables/Final_Presentation.pptx`](deliverables/Final_Presentation.pptx) and [`Final_Presentation.pdf`](deliverables/Final_Presentation.pdf) — eight slides.
- [`deliverables/Speaker_Notes.md`](deliverables/Speaker_Notes.md) — approximately 6–7 minutes including the live demo.
- `deliverables/Source_Code_Handoff.zip` — repository-ready source, tests, docs, and configuration.
- `streamlit_app.py`, `src/medical_doc_assistant/api.py`, `Dockerfile`, and `docker-compose.yml` — complete application runtime.

The source package and temporary-demo handoff were selected instead of publishing
to a hosting/GitHub account. Persistent hosting can be performed with
`scripts/deploy_huggingface.py` after a local Hugging Face login.

The initial snapshot (2026-10-28) contains 3,572 raw records and **2,471 cleaned
English medicine labels from openFDA**. The downloaded snapshot, evaluation split,
ingredient reference, and trained models are available locally.

The six starting categories are antibacterial, anti-inflammatory (NSAID),
corticosteroid, antifungal, acid reducer, and antihypertensive. Exact mappings from
`openfda.pharm_class_epc` are in [`configs/phase1.json`](configs/phase1.json).
These are a bounded classification task, not an exhaustive taxonomy of medicines.

### Recorded classifier results

| Model | Validation macro F1 | Test accuracy | Test macro F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.8355 | 97.66% | 0.9799 |
| Linear SVM | 0.8565 | 95.61% | 0.9622 |
| **Random Forest — selected** | **0.8671** | **97.08%** | **0.9751** |
| **PyTorch TextCNN — selected deep model** | **0.9972** | **98.54%** | **0.9876** |
| Fine-tuned BERT-mini | 0.9944 | 95.32% | 0.9597 |

BERT-mini uses four encoder layers, hidden size 256, and two 256-token windows.
It did not exceed the TextCNN on this snapshot. All measured results are retained.

### Grounded QA and Phase 4 reference

- MiniLM sentence embeddings with one FAISS index per uploaded/pasted document.
- Source-preserving chunk offsets and persisted index/model checksums.
- Answerability checks, broad-section retrieval, and checked local FLAN-T5 generation.
- Source quotation + citation offsets, or **“I cannot find that in the document.”**
- Conversational questions and short topics such as `dosage`, `side effects`,
  `how often should I take it?`, and `can I take it if I'm pregnant?`.
- Concise source excerpts with explicit yes/no, conditional, or uncertain wording
  when supported; age limits, dose limits, exceptions, and uncertainty are retained.
- Source context expands under each answer. To see a complete section, ask
  `show the full dosage section`. The **Price example** checks for a stated price;
  an absent price correctly produces the missing-answer response and explanation.
- Larger upload fixtures: [`upload_examples/qa_tests/`](upload_examples/qa_tests/).
  Two clearly fictional test leaflets contain 2,052 and 1,048 words and are excluded
  from training. See [`docs/simple_question_qa.md`](docs/simple_question_qa.md) for
  examples and the policy-v5 verification results.
- English source-linked QA benchmark: **48 cases / 7 documents**, including
  **20 evaluation cases / 3 documents**. All 11 answerable cases had reference
  evidence covered and all 9 missing-answer cases returned the fallback under the
  recorded policy-v2 correction. This is a small template-based regression set, not clinical
  validation; the initial frequency-answer mismatch and correction are preserved.

- [`reports/phase_4_review.md`](reports/phase_4_review.md) — BERT comparison, QA architecture, examples, and benchmark limits.
- [`notebooks/04_bert_and_grounded_qa.ipynb`](notebooks/04_bert_and_grounded_qa.ipynb) — executed classifier/retrieval/QA walkthrough.
- [`docs/phase4_model_cards.md`](docs/phase4_model_cards.md) — model roles, source licenses, usage, and scope.
- [`reports/phase4/verification.md`](reports/phase4/verification.md) — 68 software tests, 54 artifact checks, and 16 notebook cells.
- `models/phase4/bert/model.pt` + `tokenizer/` — standalone fine-tuned classifier.
- `data/evaluation/phase4_qa_cases.json` — authored questions with literal FDA reference answers.

### Phase 3 reference results

Random Forest is the preselected classical benchmark. The CNN improves its test
macro F1 by **0.0125**. Neural settings were chosen on validation before test
evaluation: learning rate **0.003**, batch size **32**, dropout **0.5**, and a
**23-epoch** final train/validation refit. The CNN reads up to eight contiguous
256-token passages per document; the report includes text-coverage diagnostics.

- [`reports/phase_3_review.md`](reports/phase_3_review.md) — learning curves, comparison, coverage, matrices, and errors.
- [`notebooks/03_pytorch_text_classifier.ipynb`](notebooks/03_pytorch_text_classifier.ipynb) — executed Dataset/DataLoader, model, and inference walkthrough.
- [`docs/neural_model_card.md`](docs/neural_model_card.md) — neural checkpoint usage and scope.
- [`reports/phase3/verification.md`](reports/phase3/verification.md) — 53 software tests, 95 experiment checks, 16 notebook cells, and CLI verification.
- `models/phase3/text_cnn.pt` — standalone neural state, vocabulary, architecture, and provenance.

### Phase 2 reference results

The fixed split contains **1,762 training**, **367 validation**, and **342 test**
documents. Related labels stay in the same partition. Hyperparameters use
three-fold training-only, group-aware CV. After validation selection, all frozen
candidates are refit on train + validation and evaluated on test. The report also
includes group-balanced metrics, class supports, and source-linked errors.

- [`reports/phase_2_review.md`](reports/phase_2_review.md) — model comparison, protocol, charts, and errors.
- [`notebooks/02_baseline_classification.ipynb`](notebooks/02_baseline_classification.ipynb) — executed comparison and real-document inference notebook.
- [`docs/baseline_model_card.md`](docs/baseline_model_card.md) — selected model usage and scope.
- [`reports/phase2/verification.md`](reports/phase2/verification.md) — 38 software tests, 75 experiment checks, and notebook/CLI verification.
- `models/phase2/benchmark.joblib` — complete selected TF-IDF + classifier pipeline.
- `models/phase2/logistic_regression.joblib`, `models/phase2/linear_svm.joblib`,
  `models/phase2/random_forest.joblib` — all three families.
- `data/splits/phase2_split.csv` — frozen record/group/partition/CV assignments for later models.

### Phase 1 reference artifacts

- [`reports/phase_1_review.md`](reports/phase_1_review.md) — results, charts, and review points.
- [`notebooks/01_data_collection_and_eda.ipynb`](notebooks/01_data_collection_and_eda.ipynb) — executed EDA notebook.
- `data/processed/medicine_labels.parquet` — full cleaned dataset (recommended).
- `data/processed/medicine_labels.csv.gz` — the same dataset as compressed CSV.
- `data/processed/review_sample.csv` — two full documents per class for inspection.
- [`docs/data_card.md`](docs/data_card.md) — data source, schema, labeling rules, and limitations.
- `data/raw/openfda_manifest.json` — query settings, retrieval dates, source metadata, and checksum.
- [`reports/verification.md`](reports/verification.md) — completed pipeline, test, and notebook checks.

## Run locally

Use **Python 3.10** with the supplied dependency constraints to match the verified
environment. The project already contains a Python 3.10 virtual environment.

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[ml,dl,nlp,web,notebooks]" -c "requirements-phase5.txt"
.\.venv\Scripts\python.exe scripts/run_app.py
```

If `.venv`, data, and models are already present, run only the last command. It
launches the API and web interface with the supplied model artifacts. Source
training/evaluation commands remain available below. The first model preparation
downloads pinned public runtimes; inference can then run offline.

`requirements-phase2.txt` includes the Phase 1 constraints and the ML dependencies.
`requirements-phase3.txt` adds PyTorch and `requirements-phase4.txt` adds the NLP
and FAISS dependencies. The initial local models use CPU with four
threads; CUDA is supported when a compatible PyTorch wheel/device is available.

### Linux / macOS / Colab

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[ml,dl,nlp,web,notebooks]" -c requirements-phase5.txt
python scripts/run_app.py
```

### Individual steps

Use the virtual environment's Python, or activate it before these commands:

```bash
python -m medical_doc_assistant collect
python -m medical_doc_assistant clean
python -m medical_doc_assistant eda
python -m medical_doc_assistant validate
python -m medical_doc_assistant prepare-phase2
python -m medical_doc_assistant phase2
python -m medical_doc_assistant validate-phase2
python -m medical_doc_assistant phase3
python -m medical_doc_assistant validate-phase3
python -m medical_doc_assistant phase4
python -m medical_doc_assistant validate-phase4
python -m unittest discover -s tests -v
python scripts/execute_notebook.py
python scripts/execute_notebook.py notebooks/02_baseline_classification.ipynb
python scripts/execute_notebook.py notebooks/03_pytorch_text_classifier.ipynb
python scripts/execute_notebook.py notebooks/04_bert_and_grounded_qa.ipynb
python -m jupyterlab notebooks/04_bert_and_grounded_qa.ipynb
```

### Ask questions about a real source document

```powershell
.\.venv\Scripts\python.exe -m medical_doc_assistant ask --file "data/examples/phase2_demo_leaflet.txt" --question "How long can this product take for full effect?"
.\.venv\Scripts\python.exe -m medical_doc_assistant chat --file "data/examples/phase2_demo_leaflet.txt"
```

The first command returns a source quotation and citation. In chat, type `/quit`
to exit. Questions explicitly refer to the same document; personal history is not
inferred. `index-document --file ...` prints a saved document ID, and
`ask --document-id <sha256> --question ...` reuses that isolated index.

### Try the saved classifiers

The executed notebook saves a real source-backed example with provenance in
`data/examples/phase2_demo_leaflet.json`:

```powershell
.\.venv\Scripts\python.exe -m medical_doc_assistant classify-neural --file "data/examples/phase2_demo_leaflet.txt"
.\.venv\Scripts\python.exe -m medical_doc_assistant classify-bert --file "data/examples/phase2_demo_leaflet.txt"
```

Expected category for the supplied example: `acid_reducer`. Use `--file` with your
own UTF-8 leaflet text, or `--text` for pasted text. Python usage is described in
the neural model card. Current inference predicts one of the six trained categories.
The classical benchmark remains available with the `classify` command.

### Data collection and reproducibility

The Phase 1 downloader collects up to 600 raw labels per class by default, with
100 labels per request, and reuses the verified local snapshot.

To collect a larger or smaller new snapshot:

```bash
python -m medical_doc_assistant phase1 --per-class 1000 --refresh
```

`--refresh` replaces the downloaded snapshot. Without it, changed collection
settings raise an error rather than silently reusing different data. Back up a
snapshot before replacing it if you want to preserve an experiment.

Phase 2 results are bound to the saved dataset, split, reference, and config
checksums. Keep those artifacts together for reproduction and later model
comparisons. A new corpus or different training settings should have a separate
experiment copy/root so the recorded benchmark remains available.

An API key is optional for the small download used here; the current endpoint
accepts unauthenticated requests within its published limits. For larger use,
obtain a free key at <https://open.fda.gov/apis/authentication/> and set it as an
environment variable:

```powershell
$env:OPENFDA_API_KEY = "your-key"
```

The key is read from the environment and excluded from manifests and logs.

## Data pipeline

```text
openFDA human drug labels + pharmacologic-class metadata
                  ↓
cached raw JSONL + collection manifest
                  ↓
encoding / HTML cleanup + section normalization
                  ↓
explicit EPC mapping + exclusion audit + deduplication
                  ↓
clean Parquet / CSV + shared-ingredient leakage groups
                  ↓
EDA tables, charts, report, and notebook
```

The cleaner preserves numbers, units, inequalities, case, and negation. Dosage,
warnings, contraindications, and adverse reactions are copied from existing
source sections; missing sections stay empty. `sections_json` also retains
readable table text. These fields are source-section preparation for later
information extraction, rather than a trained fact extractor.

Only `document_text` is classifier input. Class metadata, drug identifiers, source
URLs, and group IDs are kept in separate columns. Phase 2 grouping preserves the
Phase 1 ingredient components and adds exact RxNorm ingredient relationships and
a thresholded near-duplicate audit. Later models should reuse
`benchmark_group_id`, `split`, and `cv_fold` from `data/splits/phase2_split.csv`.

## Project structure

```text
configs/                 Reviewable class definitions and collection settings
src/medical_doc_assistant/
  collection.py          Rate-limited, retrying openFDA downloader
  cleaning.py            Reusable text/section cleanup and dataset builder
  labels.py              Exact pharmacologic-class mapping
  eda.py                 Exploratory analysis and report generation
  validation.py          Dataset integrity checks
  rxnorm.py              Cached exact ingredient relationships
  splitting.py           Duplicate-aware partitions and training CV folds
  baseline.py            TF-IDF training, selection, persistence, inference
  baseline_evaluation.py  Document and group-balanced classifier metrics
  phase2_validation.py   Split, feature-fit, artifact, and metric checks
  phase2_reporting.py    Baseline comparison charts and review report
  neural_text.py         Tokenizer, vocabulary, contiguous windows, Dataset
  neural_model.py        Padding-aware TextCNN and standalone checkpoint IO
  deep_learning.py       Tuning, early stopping, fixed refit, evaluation
  phase3_validation.py   Neural isolation, checkpoint, loss, metric checks
  phase3_reporting.py    Learning curves, coverage, and comparison reports
  hf_assets.py           Pinned public model downloads and source/license hashes
  bert_model.py          WordPiece windows, compact BERT, checkpoint inference
  bert_training.py       BERT validation selection and final refit
  document_index.py      Full-source chunks and per-document FAISS persistence
  grounded_qa.py         Checked source-quoted retrieval and local generation
  qa_cases.py            Source-derived MedQuAD-style reference cases
  rag_evaluation.py      QA answerability/evidence/fallback evaluation
  phase4_validation.py   Classifier, index, model, and citation checks
  phase4_reporting.py    BERT + QA charts and review report
  ingestion.py           PDF/DOCX/text reading and source-offset extraction
  serving.py             Cached models, opaque document handles, source QA
  api.py                 FastAPI classify/extract/upload/ask routes
  pipeline.py            CLI for data, model training, and classification
data/raw/                Real source snapshot and provenance manifest
data/processed/          Clean data and a small review sample
data/reference/          Source-backed ingredient mapping cache
data/splits/             Frozen Phase 2 evaluation assignments
data/examples/           Real demo text and source provenance
data/evaluation/         Real QA source documents and reference pairs
data/indexes/            Independent source/vector document indexes
models/phase2/           Complete classifier pipeline artifacts
models/phase3/           Selected neural state and validation trial checkpoints
models/phase4/           BERT checkpoint/tokenizer and pinned QA runtime assets
reports/                 EDA outputs, exclusion audit, and verification results
notebooks/               Executed EDA and model comparison notebooks
scripts/                 Notebook execution helper
tests/                   Data, grouping, CV leakage, metrics, persistence checks
docs/                    Data/model cards and phase status
```

Large raw/processed files, trained binaries, caches, and `.venv` are ignored by
Git. They are present locally after the pipelines run. Keep manifests and frozen
snapshots with the experiment for reproducibility.

## Phase checkpoints

| Phase | Course output | Status |
|---|---|---|
| 1 | Data + EDA pipeline | Complete — approved for continuation |
| 2 | TF-IDF baseline classifiers | Complete — approved for continuation |
| 3 | PyTorch text classifier | Complete — approved for continuation |
| 4 | BERT + grounded RAG chatbot | Complete — approved for continuation |
| 5 | FastAPI, Streamlit, deployment, presentation | Complete — temporary demo and source handoff |

English is the starting corpus, as recommended in the plan. Arabic is an
extension after English works; translated examples must carry explicit synthetic
provenance. Each phase has a review checkpoint before the next phase begins.

## Sources and use

- Main corpus: <https://open.fda.gov/apis/drug/label/>
- Alternative larger source: <https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm>
- Terms: <https://open.fda.gov/terms/>
- API limits: <https://open.fda.gov/apis/authentication/>
- Ingredient reference: <https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html>
- RxNorm terms: <https://lhncbc.nlm.nih.gov/RxNav/TermsofService.html>
- Phase 4 pretrained model cards/revisions/licenses: `reports/phase4/model_assets.json`.
- deepset answerability model (CC BY 4.0): <https://huggingface.co/deepset/minilm-uncased-squad2>.

Data provided by the U.S. Food and Drug Administration (<https://open.fda.gov>).
openFDA describes its data as generally public domain / CC0 unless otherwise
noted; third-party rights may apply to some materials. See the source terms and
the data card for the recorded provenance.

This product uses publicly available data from the U.S. National Library of
Medicine (NLM), National Institutes of Health, Department of Health and Human
Services; NLM is not responsible for the product and does not endorse or recommend
this or any other product.

As specified in the project plan, this project is **informational only and is not
medical advice**. Source labels are not independently verified by this project.
