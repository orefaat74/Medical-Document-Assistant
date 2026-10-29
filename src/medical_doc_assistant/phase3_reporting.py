"""Render learning curves, context coverage, and the Phase 3 review checkpoint."""

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
    def value(item: Any) -> str:
        return str(item).replace("|", " / ").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(value, frame.columns)) + " |",
                      "| " + " | ".join("---" for _ in frame.columns) + " |",
                      *("| " + " | ".join(map(value, row)) + " |" for row in frame.itertuples(index=False, name=None))])


def generate_phase3_report(root: Path, manifest: dict[str, Any], verification: dict[str, Any], definitions: list[dict[str, Any]]) -> None:
    report_dir = root / "reports" / "phase3"
    figures = report_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    comparison = pd.read_csv(report_dir / "model_comparison.csv")
    trials = pd.read_csv(report_dir / "trial_comparison.csv")
    coverage = pd.read_csv(report_dir / "token_coverage.csv")
    per_class = pd.read_csv(report_dir / "per_class_test_metrics.csv")
    matrix = pd.read_csv(report_dir / "confusion_matrix.csv", index_col=0)
    selected = next(result for result in manifest["selection"]["results"] if result["trial_id"] == manifest["selected_trial"])
    history = pd.read_csv(root / selected["history_path"])
    display_names = {item["label"]: item["display_name"] for item in definitions}
    sns.set_theme(style="whitegrid")

    def save(fig: plt.Figure, name: str) -> None:
        fig.savefig(figures / name, dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="Training (online)")
    axes[0].plot(history["epoch"], history["validation_loss"], label="Validation")
    axes[0].set(title="Selected trial: weighted cross-entropy", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["validation_macro_f1"], color="#658c73")
    axes[1].axvline(selected["best_epoch"], color="#bc735c", linestyle="--", label="Selected checkpoint")
    axes[1].set(title="Selected trial: validation macro F1", xlabel="Epoch", ylabel="Macro F1", ylim=(0, 1))
    axes[1].legend()
    fig.tight_layout()
    save(fig, "learning_curves.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(trials["trial_id"], trials["validation_macro_f1"],
                  color=["#658c73" if trial == manifest["selected_trial"] else "#5674a5" for trial in trials["trial_id"]])
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set(title="Predeclared neural hyperparameter trials", xlabel="Trial", ylabel="Validation macro F1", ylim=(0, 1.08))
    save(fig, "trial_comparison.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["#658c73" if phase == 3 else "#5674a5" for phase in comparison["phase"]]
    for ax, field, title in zip(axes, ["validation_macro_f1", "test_macro_f1"], ["Validation macro F1", "Frozen test macro F1"]):
        bars = ax.bar(comparison["display_name"], comparison[field], color=colors)
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.set(title=title, ylabel="Macro F1", ylim=(0, 1.08))
        ax.tick_params(axis="x", labelrotation=15)
    fig.tight_layout()
    save(fig, "baseline_comparison.png")

    readable = matrix.copy()
    readable.index = readable.columns = [display_names[label] for label in matrix.index]
    for normalized, name in ((False, "confusion_matrix.png"), (True, "confusion_matrix_normalized.png")):
        values = readable.div(readable.sum(axis=1), axis=0) if normalized else readable
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(values, annot=True, fmt=".2f" if normalized else "d", cmap="Blues", cbar=False, ax=ax)
        ax.set(title="TextCNN — " + ("recall-normalized" if normalized else "count") + " test confusion matrix",
               xlabel="Predicted class", ylabel="Source target class")
        ax.tick_params(axis="x", labelrotation=40, labelsize=9)
        ax.tick_params(axis="y", labelrotation=0, labelsize=9)
        save(fig, name)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=coverage, x="retention_ratio", y="split", order=["train", "validation", "test"], color="#92a6be", ax=ax)
    ax.set(title="Fraction of source regex tokens seen by the CNN", xlabel="Retained token fraction", ylabel="", xlim=(0, 1.02))
    save(fig, "token_coverage.png")

    projection = pd.read_csv(report_dir / "training_embedding_projection.csv")
    projection["category"] = projection["class_label"].map(display_names)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=projection, x="pc1", y="pc2", hue="category", s=15, alpha=.5, ax=ax)
    ax.set(title="CNN training-document representations (PCA, exploratory)", xlabel="PC1", ylabel="PC2")
    save(fig, "training_embedding_projection.png")

    comparison_table = comparison[["display_name", "validation_macro_f1", "test_accuracy", "test_macro_precision", "test_macro_recall", "test_macro_f1",
                                   "test_group_balanced_macro_f1"]].round(4).rename(columns={
        "display_name": "Model", "validation_macro_f1": "Validation F1", "test_accuracy": "Test accuracy", "test_macro_precision": "Test precision",
        "test_macro_recall": "Test recall", "test_macro_f1": "Test macro F1", "test_group_balanced_macro_f1": "Group-balanced test F1"})
    coverage_summary = coverage.groupby("split").agg(documents=("record_id", "size"), mean_tokens=("token_count", "mean"),
        mean_seen=("encoded_tokens", "mean"), mean_retention=("retention_ratio", "mean"), mean_unknown=("unknown_ratio", "mean")).reindex(
            ["train", "validation", "test"]).round(4).reset_index()
    class_table = per_class[["class_label", "precision", "recall", "f1", "support", "groups"]].copy()
    class_table["class_label"] = class_table["class_label"].map(display_names)
    delta = manifest["test_macro_f1_difference_from_benchmark"]
    outcome = "higher" if delta > 0 else "lower" if delta < 0 else "equal"
    summary = {"selected_trial": manifest["selected_trial"], "parameters": selected["parameters"],
               "validation_macro_f1": manifest["validation_metrics"]["macro_f1"], "test_accuracy": manifest["test_metrics"]["accuracy"],
               "test_macro_f1": manifest["test_metrics"]["macro_f1"],
               "test_group_balanced_macro_f1": manifest["test_metrics"]["group_balanced_macro_f1"],
               "test_macro_f1_difference_from_classical_benchmark": delta, "refit_epochs": manifest["refit_epochs"],
               "verification_checks_passed": len(verification["checks"]), "coverage": coverage_summary.to_dict("records")}
    write_json(report_dir / "summary.json", summary)
    report = f"""# Phase 3 — PyTorch text classifier review

## Result

A **multi-window TextCNN** was trained from scratch with a learned word embedding,
PyTorch Dataset/DataLoader, class-weighted loss, hyperparameter tuning, and early
stopping. The selected deep model is saved in `../models/phase3/text_cnn.pt`.

- Selected trial: **{manifest['selected_trial']}**; parameters: `{selected['parameters']}`.
- Validation macro F1: **{manifest['validation_metrics']['macro_f1']:.4f}**.
- Frozen test accuracy / macro F1: **{manifest['test_metrics']['accuracy']:.4f} / {manifest['test_metrics']['macro_f1']:.4f}**.
- Group-balanced test macro F1: **{manifest['test_metrics']['group_balanced_macro_f1']:.4f}**.
- Final fixed-epoch refit: **{manifest['refit_epochs']} epochs**, **{manifest['fitted_documents']} train/validation documents**.
- Learnable parameters: **{manifest['model_parameters']:,}**.
- Experiment integrity: **{len(verification['checks'])} checks passed**.

## Comparison with classical baselines

{_table(comparison_table)}

![Baseline comparison](phase3/figures/baseline_comparison.png)

The CNN's test macro F1 is **{abs(delta):.4f} {outcome}** than the preselected
Random Forest benchmark. All classical results are retained. A deep model is not
assumed to outperform TF-IDF on this small, source-derived corpus; the comparison
is reported from the actual recorded predictions.

## Architecture and text handling

```text
Clinical document text
 → fixed Unicode-aware word tokenizer
 → frequency vocabulary (PAD=0, UNK=1)
 → up to 8 contiguous windows × 256 tokens
 → learned 64-dimensional word embeddings
 → Conv1D banks with widths 3, 4, 5 and 32 filters each
 → padding-masked max/mean pooling across document windows
 → pooled embedding mean + dropout + six-class linear head
```

Convolutions are applied within each contiguous source window; nonadjacent
passages are not concatenated into artificial phrases. Long labels include the
first and last windows plus evenly spaced middle windows. Labels at most 2,048
regex tokens long are retained in full. PAD-touching convolution windows are
excluded from pooling, and empty chunks cannot affect logits.

The vocabulary is learned from training text only during tuning, then relearned
on train + validation for the final refit. Raw clinical text remains the source;
normalization/token sampling is an input representation. No metadata, identifiers,
source URLs, ingredient fields, or test text are included in vocabulary fitting.

## Context coverage

{_table(coverage_summary)}

![Token coverage](phase3/figures/token_coverage.png)

These are regex-token counts, which differ from Phase 1 whitespace word counts.
The CNN has a **2,048-token document budget**; TF-IDF processes full document text.
This input-budget difference is part of the model comparison and can affect
results. Unknown ratios describe encoded tokens missing from the fitted vocabulary.
The source corpus and frozen group-separated split are the same for every model.

## Tuning and early stopping

{_table(trials[['trial_id', 'learning_rate', 'batch_size', 'dropout', 'best_epoch', 'epochs_run', 'validation_macro_f1']].round(4))}

![Trial comparison](phase3/figures/trial_comparison.png)

Eight combinations of learning rate, batch size, and dropout were declared before
training. AdamW uses class-weighted cross-entropy, weight decay, and gradient
clipping. The limit is 24 epochs with patience 5 and minimum meaningful validation
macro-F1 improvement 0.0001. Checkpoints use the highest validation macro F1, with
lower validation loss as the tie-break. Small/tied checkpoint updates do not reset
early-stopping patience unless the primary metric meaningfully improves.

![Learning curves](phase3/figures/learning_curves.png)

The best trial was selected on the 367-document validation partition. Its selected
epoch count was then fixed for a fresh train + validation refit, matching the
classical models' fit partition. Test inference occurred only after this selection
and refit. Trial/checkpoint records, loss curves, and token vocabularies are saved
under `phase3/`. Completed trials can be reused after an interrupted run.

## Class results and source-linked errors

{_table(class_table.round(4))}

![Confusion matrix](phase3/figures/confusion_matrix.png)

![Normalized confusion matrix](phase3/figures/confusion_matrix_normalized.png)

Errors and source links are in [`phase3/test_errors.csv`](phase3/test_errors.csv).
The test set has **15 independent components**, with limited ingredient coverage
in some classes. Targets come from pharmacologic-class metadata and are not
expert-adjudicated clinical indications.

## Optional representation experiment

![Training representation PCA](phase3/figures/training_embedding_projection.png)

PCA visualizes the final CNN's pooled representations for **training documents
only**. It is an exploratory illustration of learned features, not a held-out
performance estimate or a medical clustering result.

## Hardware and reproduction

Training device: `{manifest['environment']['device']}`; PyTorch:
`{manifest['environment']['torch']}`; CPU threads: `{manifest['environment']['cpu_threads']}`.
The initial run uses a compact CPU model because the local GPU was occupied.
Seeds and deterministic PyTorch settings are recorded. CUDA execution is supported
when a compatible PyTorch installation/device is available.

## Deliverables and review checkpoint

- `../models/phase3/text_cnn.pt` — standalone model state, vocabulary, architecture, and provenance
- `../models/phase3/trials/*.pt` — best validation checkpoint for every candidate
- `phase3/trial_comparison.csv` and `trial_*_history.csv` — tuning and learning curves
- `phase3/model_selection.json` — pre-test selection and fixed refit epoch count
- `phase3/token_coverage.csv` — context-budget and vocabulary coverage by source record
- `phase3/model_comparison.csv` — all three classical families plus the CNN
- `phase3/predictions_test.csv` — recorded predictions/logits/source links
- `phase3/validation.json` — data-isolation, checkpoint, loss, and metric checks
- `../notebooks/03_pytorch_text_classifier.ipynb` — executed review notebook
- `../docs/neural_model_card.md` — usage and model scope

Review the actual model comparison, learning curves, text-coverage tradeoff,
class-level errors, and checkpoint usage. The next course phase is BERT fine-tuning
and a document-grounded RAG chatbot, after your review.

Data provided by the U.S. Food and Drug Administration (<https://open.fda.gov>).

{NLM_ATTRIBUTION}

As specified in the original plan, the project is informational only and is not
medical advice. The classifier is a closed-set English medicine-label model.
"""
    (root / "reports" / "phase_3_review.md").write_text(report, encoding="utf-8")
    card = f"""# Neural model card — Phase 3 TextCNN

- Architecture: learned word embedding + padding-aware multi-window 1D-CNN.
- Checkpoint: `models/phase3/text_cnn.pt` (state dictionaries and primitive metadata).
- Selected trial / parameters: `{manifest['selected_trial']}` / `{selected['parameters']}`.
- Validation macro F1: **{manifest['validation_metrics']['macro_f1']:.4f}**.
- Test accuracy / macro F1: **{manifest['test_metrics']['accuracy']:.4f} / {manifest['test_metrics']['macro_f1']:.4f}**.
- Final fit: **{manifest['fitted_documents']}** train/validation documents, **{manifest['refit_epochs']}** preselected epochs.
- Vocabulary: **{manifest['vocabulary_sizes']['refit']}** tokens; PAD=0, UNK=1.
- Input budget: eight contiguous windows of 256 tokens, up to 2,048 tokens per label.
- Initial runtime: PyTorch `{manifest['environment']['torch']}`, `{manifest['environment']['device']}`.

## Use

```powershell
.\\.venv\\Scripts\\python.exe -m medical_doc_assistant classify-neural --file "data/examples/phase2_demo_leaflet.txt"
```

```python
from pathlib import Path
from medical_doc_assistant.neural_model import classify_neural_text

text = Path("data/examples/phase2_demo_leaflet.txt").read_text(encoding="utf-8")
result = classify_neural_text(Path("models/phase3/text_cnn.pt"), text)
print(result["predicted_class"])
```

The checkpoint includes architecture parameters, label order, the token vocabulary,
encoding settings, and source/split provenance. Loading reconstructs the model
from its state dictionary with `weights_only=True`; inference uses evaluation
mode with dropout disabled.

## Scope and comparison

Outputs are the six EPC-derived course categories: {', '.join(manifest['class_labels'])}.
This is a closed-set English medicine-label classifier. It does not provide an
expert-confirmed indication, diagnosis, medical recommendation, or calibrated
confidence. Other document types, unknown categories, and Arabic are later extensions.

Training-only vocabulary fitting and group-separated validation govern tuning.
Final vocabulary/model fitting use train + validation, excluding test. The same
record split as Phase 2 is used for evaluation. TF-IDF uses full text whereas the
CNN samples a bounded context; coverage and actual results are recorded in
`reports/phase_3_review.md`. Source labels are automatic metadata targets.

Data provided by the U.S. Food and Drug Administration (<https://open.fda.gov>).

{NLM_ATTRIBUTION}

As stated in the project plan, the project is informational only and is not
medical advice.
"""
    (root / "docs" / "neural_model_card.md").write_text(card, encoding="utf-8")
