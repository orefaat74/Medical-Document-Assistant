# Phase checkpoints

The original scope is in `Final_Project_Plan.pdf`. Work follows the five phases
in that document, with a user-review checkpoint after each phase.

| Phase | Deliverable | State |
|---|---|---|
| 1 — Python for AI | Real data, reusable cleaning, EDA, notebook | Complete — approved for continuation |
| 2 — Machine Learning | TF-IDF + Logistic Regression, Linear SVM, Random Forest | Complete — approved for continuation |
| 3 — Deep Learning | PyTorch tokenizer, Dataset/DataLoader, TextCNN | Complete — approved for continuation |
| 4 — NLP and LLMs | Fine-tuned BERT-mini + source-quoted FAISS QA | Complete — approved for continuation |
| 5 — Capstone | FastAPI, Streamlit, Docker, demo, presentation | Complete — temporary demo and source handoff selected |

## Final capstone handoff

Completed on 2026-10-28: FastAPI endpoints, a Streamlit upload/results/chat UI,
text-based PDF/DOCX/HTML/text ingestion, source-offset extraction, cached trained
models, opaque document handles, and document-specific grounded QA.

The Docker image built and the Linux container passed the real API workflow.
The local browser workflow and public HTTPS websocket/chat workflow were verified.
The temporary review URL is recorded in `reports/phase5/live_demo.json`.

The user selected **temporary demo and deployment handoff** and **source package**
instead of account-backed hosting/repository publication. Docker/Compose and a
Hugging Face Spaces deployment script are included. The final presentation,
speaker notes, architecture visuals, and source ZIP are in `deliverables/`.

Review `reports/phase_5_review.md`, `docs/deployment.md`, and the slide deck.

## Phase 4 reference

Completed on 2026-10-28. The pinned compact Google BERT was fine-tuned on the
frozen classifier corpus. A local MiniLM/FAISS/SQuAD2/FLAN-T5 document assistant
was implemented with exact source offsets and the required missing-answer fallback.

- BERT-mini test accuracy / macro F1: **95.32% / 0.9597**.
- BERT did not beat the TextCNN (**98.54% / 0.9876**); all five classifier results
  are retained for the final comparison.
- QA references: **48 source-linked cases / 7 documents**.
- Corrected evaluation: **11 answerable / 9 missing** cases across **3 documents**;
  source support and reference evidence covered in all returned answers.
- Initial evaluation preserved: one frequency answer quoted strength instead;
  a general answer-type guard was added, with the updated result recorded as a
  post-correction regression evaluation, not a fresh untouched test.
- Verification: **68 software tests**, **54 artifact checks**, and **16 executed
  notebook cells**; classifier, source-answer, and missing-answer CLI checks passed.
- Evaluated language: English. Arabic remains an explicitly scoped extension.

Review `reports/phase_4_review.md`, `notebooks/04_bert_and_grounded_qa.ipynb`,
`docs/phase4_model_cards.md`, and `reports/phase4/verification.md`.

Phase 4 was approved for continuation by the user's request to start the last phase.

## Phase 3 reference

Completed on 2026-10-28. A PyTorch multi-window TextCNN was trained from scratch
and eight learning-rate/batch-size/dropout candidates were compared on validation.
The selected configuration is **0.003 / 32 / 0.5**, with a **23-epoch** final refit
on 2,129 train/validation documents.

- Frozen source/split: the same **2,471 documents** and **100 components** as Phase 2.
- Validation macro F1: **0.9972**.
- Test accuracy / macro F1: **98.54% / 0.9876**.
- Group-balanced test macro F1: **0.9859**.
- Macro-F1 improvement over the Random Forest benchmark: **0.0125**.
- Context budget: up to eight contiguous 256-token windows per document.
- Checkpoint: `models/phase3/text_cnn.pt`.
- Verification: **53 software tests**, **95 experiment checks**, **16 executed
  notebook code cells**, and successful real-document CLI inference.

Review:

1. `reports/phase_3_review.md` — comparison, learning curves, text coverage, errors.
2. `notebooks/03_pytorch_text_classifier.ipynb` — executed PyTorch walkthrough.
3. `docs/neural_model_card.md` — checkpoint usage and scope.
4. `reports/phase3/verification.md` — checks and reproduction commands.

Phase 3 was approved for continuation by the user's request to start Phase 4.

## Phase 2 reference

Completed on 2026-10-28. All three TF-IDF families were tuned with three-fold,
group-separated training CV and compared on validation. Random Forest won
validation macro F1 (**0.8671**) and was selected before test evaluation.

- Frozen split: **1,762 train / 367 validation / 342 test** documents.
- Ingredient-aware benchmark components: **100**.
- Selected baseline test accuracy / macro F1: **97.08% / 0.9751**.
- Selected group-balanced test macro F1: **0.9688**.
- All three final pipelines saved; `models/phase2/benchmark.joblib` is the selection.
- Verification: **38 software tests**, **75 experiment integrity checks**, and
  **15 notebook code cells** passed; the real-document CLI example also succeeded.

Review:

1. `reports/phase_2_review.md` — comparison, split, CV variation, matrices, errors.
2. `notebooks/02_baseline_classification.ipynb` — executed walkthrough.
3. `docs/baseline_model_card.md` — input/output scope and model usage.
4. `reports/phase2/verification.md` — checks and reproducibility commands.

Phase 2 was approved for continuation by the user's request to start Phase 3.

## Phase 1 reference

Completed on 2026-10-28: 3,572 raw openFDA records → 2,471 cleaned English labels,
six target classes, CSV/Parquet exports, five EDA charts, and an executed notebook.
Verification passed: 20 software tests, 21 dataset/export checks, and 13 notebook
code cells. Details are in `reports/verification.md`.

1. Read `reports/phase_1_review.md`.
2. Confirm the class scheme in `configs/phase1.json`.
3. Inspect the full source examples in `data/processed/review_sample.csv`.
4. Open `notebooks/01_data_collection_and_eda.ipynb`.
5. Phase 1 was approved for continuation by the user's request to start Phase 2.

The initial corpus uses real openFDA labels in English. The original plan's Arabic
extension will be addressed after the English system works.
