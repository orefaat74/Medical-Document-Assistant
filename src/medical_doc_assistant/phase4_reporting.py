"""Generate the BERT + grounded RAG review checkpoint from recorded results."""

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from medical_doc_assistant.rxnorm import NLM_ATTRIBUTION
from medical_doc_assistant.storage import write_json


def _table(frame: pd.DataFrame) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", " / ").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(cell, frame.columns)) + " |", "| " + " | ".join("---" for _ in frame.columns) + " |",
                      *("| " + " | ".join(map(cell, row)) + " |" for row in frame.itertuples(index=False, name=None))])


def generate_phase4_report(root: Path, config: dict[str, Any], verification: dict[str, Any], definitions: list[dict[str, Any]]) -> None:
    directory = root / "reports" / "phase4"
    figures = directory / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    bert = json.loads((directory / "bert" / "run_manifest.json").read_text(encoding="utf-8"))
    qa = json.loads((directory / "rag" / "evaluation_metrics.json").read_text(encoding="utf-8"))
    initial = json.loads((directory / "rag" / "archive" / "policy_v1" / "evaluation_metrics.json").read_text(encoding="utf-8"))
    selected = next(row for row in bert["selection"]["results"] if row["trial_id"] == bert["selection"]["selected_trial"])
    comparison = pd.read_csv(directory / "bert" / "model_comparison.csv")
    history = pd.read_csv(directory / "bert" / (selected["trial_id"] + "_history.csv"))
    names = {definition["label"]: definition["display_name"] for definition in definitions}
    sns.set_theme(style="whitegrid")

    def save(fig: plt.Figure, filename: str) -> None:
        fig.savefig(figures / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, field, title in zip(axes, ["validation_macro_f1", "test_macro_f1"], ["Validation macro F1", "Frozen-test macro F1"]):
        bars = ax.bar(comparison["display_name"], comparison[field], color=["#bc735c" if model == "bert_mini" else "#5674a5" for model in comparison["model"]])
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.set(title=title, ylabel="Macro F1", ylim=(0, 1.08))
        ax.tick_params(axis="x", labelrotation=20, labelsize=8)
    fig.tight_layout()
    save(fig, "classifier_comparison.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="Training")
    axes[0].plot(history["epoch"], history["validation_loss"], label="Validation")
    axes[0].legend()
    axes[0].set(title="Selected BERT trial: weighted loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(history["epoch"], history["validation_macro_f1"], color="#658c73")
    axes[1].axvline(selected["best_epoch"], linestyle="--", color="#bc735c")
    axes[1].set(title="Selected BERT trial: macro F1", xlabel="Epoch", ylabel="Validation macro F1", ylim=(0, 1))
    save(fig, "bert_learning_curves.png")

    matrix = pd.read_csv(directory / "bert" / "confusion_matrix.csv", index_col=0)
    matrix.index = matrix.columns = [names[label] for label in matrix.index]
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set(title="Fine-tuned BERT-mini test confusion matrix", xlabel="Prediction", ylabel="Source target")
    ax.tick_params(axis="x", labelrotation=40, labelsize=9)
    ax.tick_params(axis="y", labelrotation=0, labelsize=9)
    save(fig, "bert_confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    values = [qa["answerability_accuracy"], qa["correct_fallback_rate"], qa["source_support_rate"], qa["positive_reference_evidence_coverage"]]
    bars = ax.bar(["Answerability", "Missing-answer fallback", "Source support", "Reference evidence"], values, color="#658c73")
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set(title="Recorded QA evaluation (20 cases / 3 documents)", ylabel="Rate", ylim=(0, 1.08))
    save(fig, "qa_evaluation.png")

    table = comparison[["display_name", "validation_macro_f1", "test_accuracy", "test_macro_precision", "test_macro_recall", "test_macro_f1", "test_group_balanced_macro_f1"]].round(4)
    table = table.rename(columns={"display_name": "Model", "validation_macro_f1": "Validation F1", "test_accuracy": "Test accuracy",
                                 "test_macro_precision": "Test precision", "test_macro_recall": "Test recall", "test_macro_f1": "Test macro F1",
                                 "test_group_balanced_macro_f1": "Group-balanced F1"})
    source_table = pd.DataFrame([{ "Role": role, "Model": model["repo"], "Revision": model["revision"][:12], "License": model["license"] }
                                for role, model in config["models"].items()])
    summary = {"bert_validation_macro_f1": bert["validation_metrics"]["macro_f1"], "bert_test_accuracy": bert["test_metrics"]["accuracy"],
               "bert_test_macro_f1": bert["test_metrics"]["macro_f1"], "bert_selected_trial": selected["trial_id"],
               "bert_refit_epochs": bert["refit_epochs"], "qa_evaluation": qa, "artifact_checks_passed": len(verification["checks"])}
    write_json(directory / "summary.json", summary)
    report = f"""# Phase 4 — BERT + grounded document QA review

## Result

Phase 4 delivers **fine-tuned BERT-mini** and a **local, source-quoted RAG assistant**.
The classifier checkpoint, semantic embeddings, per-document FAISS storage,
answerability model, and local instruction model are available in the project.

- BERT validation macro F1: **{bert['validation_metrics']['macro_f1']:.4f}**.
- BERT test accuracy / macro F1: **{bert['test_metrics']['accuracy']:.4f} / {bert['test_metrics']['macro_f1']:.4f}**.
- Selected BERT rate / fixed refit epochs: **{selected['learning_rate']} / {bert['refit_epochs']}**.
- QA evaluation: **{qa['cases']} cases** across **{qa['source_documents']} source documents**.
- Missing-answer cases correctly declined: **{qa['missing_answer_cases'] - qa['false_answers_to_missing_questions']} / {qa['missing_answer_cases']}**.
- Answerable reference evidence covered: **{qa['positive_reference_evidence_coverage']:.2%}**.
- Returned answer/citation source support: **{qa['source_support_rate']:.2%}**.
- Artifact verification: **{len(verification['checks'])} checks passed**.

## Classifier comparison

{_table(table)}

![Classifier comparison](phase4/figures/classifier_comparison.png)

The compact BERT did not outperform the TextCNN on this test snapshot. The
comparison retains all measured results rather than replacing earlier models.
BERT uses **4 layers / hidden size 256**, with up to **two 256-token windows**
(508 content WordPieces). CNN uses up to 2,048 regex tokens; TF-IDF uses full text.
Model capacity and context budgets differ and are explicitly part of the comparison.

![BERT learning curves](phase4/figures/bert_learning_curves.png)

![BERT confusion matrix](phase4/figures/bert_confusion_matrix.png)

Both learning-rate candidates were fine-tuned using training text/labels and
validation selection with early stopping. The selected epoch count was fixed,
then the model was freshly refit on train + validation before test inference.
All five model families share the same frozen corpus, class labels, and record
split. The pretrained tokenizer is fixed; it is not fitted to test text.

## Grounded QA architecture

```text
One uploaded/pasted document
 → source-preserving WordPiece chunks (180 tokens, overlap 40)
 → sentence-transformer MiniLM embeddings (384 dimensions, normalized)
 → independent document FAISS IndexFlatIP
 → semantic top-4 retrieval + explicit-section retrieval for broad FAQ questions
 → question-focus and SQuAD2 answerability checks
 → local FLAN-T5-small answer-span generation
 → literal-source/answer-type verification
 → source quotation + exact citation offsets, or missing-answer fallback
```

Indexes are isolated by a SHA-256 document ID. Each stored index includes the
canonical cleaned text, chunk offsets, vectors, FAISS data, model revision, and
file checksums. Retrieval never searches another uploaded document's index.
Chunking explicitly disables any leftover tokenizer truncation before processing
the full source; every document's tail remains indexable.

Broad requests about named leaflet sections return the explicit source section
or an excerpt. Specific questions use the answerability model's checked span.
The local LLM receives retrieved source passages. Its answer span is accepted
only when it matches the validated source span; otherwise a grounded extractive
quote is used. Rejected generated text is not exposed as an answer or metadata.
The returned quote includes nearby source-clause context to retain conditions.

Missing or unsupported evidence returns exactly:

> I cannot find that in the document.

No answer is inferred from the drug classifier, model world knowledge, or another
document. Current responses are source quotations, not individualized medical
recommendations or unconstrained summaries.

## QA benchmark and measured limits

The **48 authored MedQuAD-style pairs** use **7 real source-linked FDA leaflets**:
8 development cases, 20 calibration cases, and 20 evaluation cases. Positive
reference answers are exact source sections/spans; no medical facts were invented.
These are project-authored source-linked templates, not the original MedQuAD
dataset. Evaluation ingredient components differ from development/calibration.

![QA evaluation](phase4/figures/qa_evaluation.png)

- Answerable cases: **{qa['answerable_cases']}**; missing-answer cases: **{qa['missing_answer_cases']}**.
- Answerability accuracy: **{qa['answerability_accuracy']:.2%}**.
- Reference-evidence coverage: **{qa['positive_reference_evidence_coverage']:.2%}**.
- Missing-answer fallback rate: **{qa['correct_fallback_rate']:.2%}**.
- Literal answer/citation support: **{qa['source_support_rate']:.2%}**.

The initial evaluation had reference coverage **{initial['positive_reference_evidence_coverage']:.2%}**:
one frequency question received a supported but irrelevant strength description.
A general temporal-answer guard was added and the evaluation rerun under policy
version 2. Original results are retained in `phase4/rag/archive/policy_v1/`.
The updated score is therefore a documented post-correction regression result,
not a fresh untouched test estimate. No BERT weights/hyperparameters were changed
using these QA findings.

This small benchmark is dominated by broad section FAQs and simple span queries.
It checks source grounding, document isolation, and missing-answer behavior; it
does not establish clinical reliability or performance on all medical questions.
Open text without recognized headings relies on the answerability/semantic path
and may correctly decline questions. Source texts themselves are not independently
medically adjudicated by the project.

## Pinned model runtimes and attribution

{_table(source_table)}

Model cards, immutable revisions, asset hashes, and licenses are recorded in
`phase4/model_assets.json`. The QA model is by **deepset**, distributed under
**CC BY 4.0**; see <https://huggingface.co/deepset/minilm-uncased-squad2>.
The Google BERT/FLAN and sentence-transformer assets are Apache 2.0 model releases.

## Try the components

```powershell
.\\.venv\\Scripts\\python.exe -m medical_doc_assistant classify-bert --file "data/examples/phase2_demo_leaflet.txt"
.\\.venv\\Scripts\\python.exe -m medical_doc_assistant ask --file "data/examples/phase2_demo_leaflet.txt" --question "How long can this product take for full effect?"
.\\.venv\\Scripts\\python.exe -m medical_doc_assistant chat --file "data/examples/phase2_demo_leaflet.txt"
```

`index-document --file ...` persists an index and prints its document ID.
`ask --document-id <sha256> --question ...` reuses that document. The interactive
chat asks explicit questions about the same document; it does not infer facts
from an implicit personal history.

## Deliverables and review checkpoint

- `../models/phase4/bert/model.pt` + `tokenizer/` — standalone fine-tuned classifier
- `phase4/bert/model_comparison.csv` — all five measured classifier results
- `../data/indexes/<document_sha256>/` — independent persisted vector indexes
- `../data/evaluation/phase4_qa_cases.json` — source-linked QA references
- `phase4/rag/evaluation_responses.json` — questions, answers, offsets, and passages
- `phase4/model_assets.json` — model revisions/licenses/file hashes
- `phase4/validation.json` — classifier/data/index/grounding checks
- `../notebooks/04_bert_and_grounded_qa.ipynb` — executed review notebook
- `../docs/phase4_model_cards.md` — component usage and scope

Review the classifier comparison, model-size/context tradeoffs, QA examples,
literal evidence, fallback behavior, and benchmark limits. The next phase is
FastAPI/Streamlit, Docker/deployment, and capstone documentation, after review.
English is the evaluated path; Arabic remains an explicitly scoped extension.

Data provided by the U.S. Food and Drug Administration (<https://open.fda.gov>).

{NLM_ATTRIBUTION}

As required by the original plan, this project is informational only and is not
medical advice.
"""
    (root / "reports" / "phase_4_review.md").write_text(report, encoding="utf-8")
    cards = f"""# Phase 4 component cards

## Fine-tuned BERT-mini classifier

- Base: `{config['models']['bert']['repo']}` at `{config['models']['bert']['revision']}` (Apache 2.0).
- Shape: 4 encoder layers, hidden size 256; document-level mean/max CLS pooling.
- Input: two contiguous windows of at most 256 tokens, 508 content WordPieces.
- Labels: {', '.join(bert['class_labels'])}.
- Validation macro F1: {bert['validation_metrics']['macro_f1']:.4f}.
- Test accuracy / macro F1: {bert['test_metrics']['accuracy']:.4f} / {bert['test_metrics']['macro_f1']:.4f}.
- Final fit: {bert['fitted_documents']} train/validation documents, {bert['refit_epochs']} fixed epochs.
- Artifact: `models/phase4/bert/model.pt` plus its adjacent `tokenizer/`.

Load with `load_bert_checkpoint` or classify with `classify_bert_text`. It is a
closed-set English label classifier. Its output is an automatic-metadata-derived
course category, not a diagnosis, expert-confirmed indication, or recommendation.
The compact model/context budget did not exceed the TextCNN on this snapshot.

## Document-grounded QA

- Embeddings: `{config['models']['embedding']['repo']}`, normalized 384-dimensional vectors.
- Vector store: one FAISS IndexFlatIP per canonical document hash.
- Answerability: `{config['models']['qa']['repo']}` by deepset (CC BY 4.0).
- Local instruction model: `{config['models']['generation']['repo']}` (Apache 2.0).
- Output: verbatim source quotation with source offsets, or the required missing-answer fallback.
- Policy: question coverage, answerability/null margin, literal span checks, explicit source sections for broad FAQs, and a general frequency-answer guard.

Use `DocumentAssistant(root, config).index_document(text)` and
`assistant.answer(index, question)` for application integration. Inference is
local after public model assets are downloaded. Saved indexes verify the source
text and model revision before loading. Classifier predictions do not restrict
the QA evidence source.

The 20-case/3-document QA result is a small source-template regression benchmark.
The initial frequency-answer mismatch and corrected policy results are both
recorded. It is not a clinical validation, a universal medical QA accuracy claim,
or evidence for Arabic QA. Rejected LLM generations are not displayed as answers.

## Provenance and limitations

Source FDA texts can be outdated, incomplete, or unadjudicated; quoting them does
not certify medical accuracy. Model scope is English text, explicit document
questions, and source-derived categories. Rich clinical reasoning, implicit
personal context, and unsupported facts may receive a fallback.

deepset model attribution: <https://huggingface.co/deepset/minilm-uncased-squad2>,
CC BY 4.0. All four model revision/license records are in
`reports/phase4/model_assets.json`.

Data provided by the U.S. Food and Drug Administration (<https://open.fda.gov>).

{NLM_ATTRIBUTION}

As stated in the project plan, the project is informational only and is not
medical advice.
"""
    (root / "docs" / "phase4_model_cards.md").write_text(cards, encoding="utf-8")
