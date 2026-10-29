"""Render the classical-model review checkpoint from recorded experiment outputs."""

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

    return "\n".join([
        "| " + " | ".join(map(cell, frame.columns)) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
        *("| " + " | ".join(map(cell, row)) + " |" for row in frame.itertuples(index=False, name=None)),
    ])


def generate_phase2_report(
    root: Path, manifest: dict[str, Any], verification: dict[str, Any], definitions: list[dict[str, Any]], model_names: dict[str, str]
) -> None:
    report_dir = root / "reports" / "phase2"
    figures = report_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    display = {item["label"]: item["display_name"] for item in definitions}
    labels = manifest["class_labels"]
    selected = manifest["selected_model"]
    comparison = pd.read_csv(report_dir / "model_comparison.csv")
    best = comparison.loc[comparison["model"] == selected].iloc[0]
    split_manifest = json.loads((report_dir / "split_manifest.json").read_text(encoding="utf-8"))
    assignments = pd.read_csv(root / "data" / "splits" / "phase2_split.csv")
    balance = pd.read_csv(report_dir / "split_class_balance.csv")
    cv = pd.read_csv(report_dir / "cv_results.csv")
    cv_balance = pd.read_csv(report_dir / "cv_class_balance.csv")
    errors = pd.read_csv(report_dir / "test_errors.csv")
    sns.set_theme(style="whitegrid")

    def save(fig: plt.Figure, filename: str) -> None:
        fig.savefig(figures / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    sizes = balance.pivot(index="class_label", columns="split", values="documents").reindex(labels)[["train", "validation", "test"]]
    sizes.index = [display[label] for label in sizes.index]
    fig, ax = plt.subplots(figsize=(11, 5))
    sizes.plot.barh(ax=ax, color=["#5674a5", "#bc735c", "#658c73"])
    ax.set(title="Frozen group-separated train / validation / test split", xlabel="Documents", ylabel="")
    ax.invert_yaxis()
    save(fig, "split_class_balance.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = ["#658c73" if name == selected else "#5674a5" for name in comparison["model"]]
    for ax, column, title in zip(axes, ["validation_macro_f1", "test_macro_f1"], ["Validation macro F1 (model selection)", "Held-out test macro F1"]):
        bars = ax.bar(comparison["display_name"], comparison[column], color=colors)
        ax.set(title=title, ylabel="Macro F1", ylim=(0, 1.08))
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.tick_params(axis="x", labelrotation=12)
    fig.tight_layout()
    save(fig, "model_comparison.png")

    fold_scores = []
    for row in manifest["comparison"]:
        best_cv = cv.loc[(cv["model"] == row["model"]) & (cv["parameters"] == row["best_parameters"])].iloc[0]
        fold_scores.append({"Model": model_names[row["model"]],
                            **{f"Fold {fold + 1}": best_cv[f"fold_{fold}_macro_f1"] for fold in range(manifest["config"]["cv_folds"])}})
    fig, ax = plt.subplots(figsize=(10, 4))
    pd.DataFrame(fold_scores).set_index("Model").plot.bar(ax=ax)
    ax.set(title="Training-only group-aware CV: selected hyperparameters", ylabel="Macro F1", ylim=(0, 1.08), xlabel="")
    ax.tick_params(axis="x", labelrotation=0)
    save(fig, "cv_fold_scores.png")

    for model in model_names:
        matrix = pd.read_csv(report_dir / ("confusion_matrix_" + model + ".csv"), index_col=0)
        readable = matrix.copy()
        readable.index = readable.columns = [display[label] for label in labels]
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(readable, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set(title=model_names[model] + " — held-out test confusion matrix", xlabel="Predicted class", ylabel="Source target class")
        ax.tick_params(axis="x", labelrotation=40, labelsize=9)
        ax.tick_params(axis="y", labelrotation=0, labelsize=9)
        save(fig, "confusion_matrix_" + model + ".png")
        if model == selected:
            normalized = readable.div(readable.sum(axis=1), axis=0)
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(normalized, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, cbar=False, ax=ax)
            ax.set(title="Selected baseline — recall-normalized test confusion matrix", xlabel="Predicted class", ylabel="Source target class")
            ax.tick_params(axis="x", labelrotation=40, labelsize=9)
            ax.tick_params(axis="y", labelrotation=0, labelsize=9)
            save(fig, "selected_confusion_normalized.png")

    main_table = pd.DataFrame({
        "Model": comparison["display_name"],
        "CV macro F1 ± SD": [f"{mean:.4f} ± {std:.4f}" for mean, std in zip(comparison["cv_macro_f1_mean"], comparison["cv_macro_f1_std"])],
        "Validation F1": comparison["validation_macro_f1"].round(4),
        "Test accuracy": comparison["test_accuracy"].round(4),
        "Test precision": comparison["test_macro_precision"].round(4),
        "Test recall": comparison["test_macro_recall"].round(4),
        "Test macro F1": comparison["test_macro_f1"].round(4),
        "Benchmark": ["Selected" if model == selected else "" for model in comparison["model"]],
    })
    grouped_table = comparison[["display_name", "test_group_balanced_accuracy", "test_group_balanced_macro_f1"]].rename(
        columns={"display_name": "Model", "test_group_balanced_accuracy": "Group-balanced test accuracy",
                 "test_group_balanced_macro_f1": "Group-balanced test macro F1"}).round(4)
    partitions = pd.DataFrame([
        {"Partition": name, "Documents": manifest["partition_documents"][name],
         "Groups": manifest["partition_groups"][name],
         "% of corpus": round(100 * manifest["partition_documents"][name] / split_manifest["documents"], 2)}
        for name in ("train", "validation", "test")
    ])
    per_class = pd.read_csv(report_dir / "per_class_test_metrics.csv")
    selected_classes = per_class.loc[per_class["model"] == selected, ["class_label", "precision", "recall", "f1", "support", "groups"]].copy()
    selected_classes["class_label"] = selected_classes["class_label"].map(display)
    selected_classes = selected_classes.rename(columns={"class_label": "Class", "support": "Test documents", "groups": "Test groups"}).round(4)
    selected_errors = errors.loc[errors["model"] == selected]
    error_table = selected_errors[["generic_names", "true_class", "predicted_class"]].head(10).rename(
        columns={"generic_names": "Source generic names", "true_class": "Source target", "predicted_class": "Prediction"})
    error_text = _table(error_table) if len(error_table) else "No selected-model errors were observed in this held-out partition."
    audit = split_manifest["grouping_audit"]
    summary = {
        "selected_model": selected, "selected_display_name": model_names[selected],
        "documents": split_manifest["documents"], "benchmark_groups": audit["benchmark_groups"],
        "partition_documents": manifest["partition_documents"], "partition_groups": manifest["partition_groups"],
        "selected_validation_macro_f1": float(best["validation_macro_f1"]),
        "selected_test_accuracy": float(best["test_accuracy"]), "selected_test_macro_f1": float(best["test_macro_f1"]),
        "selected_test_group_balanced_macro_f1": float(best["test_group_balanced_macro_f1"]),
        "selected_test_errors": len(selected_errors), "verification_checks_passed": len(verification["checks"]),
        "test_used_for_selection": False,
    }
    write_json(report_dir / "summary.json", summary)
    report = f"""# Phase 2 — Baseline classifier review

## Result

**{model_names[selected]}** is the selected benchmark, chosen using validation
macro F1 before test evaluation. The standalone pipeline is saved at
`../models/phase2/benchmark.joblib`.

- Validation macro F1: **{best['validation_macro_f1']:.4f}**
- Held-out test accuracy: **{best['test_accuracy']:.4f}**
- Held-out test macro F1: **{best['test_macro_f1']:.4f}**
- Group-balanced test macro F1: **{best['test_group_balanced_macro_f1']:.4f}**
- Integrity verification: **{len(verification['checks'])} checks passed**

All three planned families were trained: Logistic Regression, Linear SVM, and
Random Forest. Precision, recall, and F1 below are macro averages over the six
fixed target labels. Standard deviation is across the training CV folds.

## Model comparison

{_table(main_table)}

![Model comparison](phase2/figures/model_comparison.png)

The most-frequent-class dummy reference has test accuracy
**{manifest['dummy_test_metrics']['accuracy']:.4f}** and macro F1
**{manifest['dummy_test_metrics']['macro_f1']:.4f}**. This is a reference value,
separate from the three trained classifier families.

## Frozen evaluation data

{_table(partitions)}

![Split class balance](phase2/figures/split_class_balance.png)

The approved **{split_manifest['documents']:,}-document** Phase 1 dataset is used in
full. Record assignments are saved in `../data/splits/phase2_split.csv` for reuse
by later course models. Targets are approximately 70% / 15% / 15%; exact row
stratification is constrained by whole ingredient components. All six classes
appear in each partition and in each training CV fit/validation fold.

Ingredient grouping preserves all Phase 1 components and adds exact/normalized
RxNorm ingredient (IN) relationships. **{audit['rxnorm_names_matched']} of
{audit['rxnorm_names_total']} substance names** have an unambiguous relationship
in the cached RxNorm `{audit['rxnorm_version']}` release. Components changed from
**{audit['phase1_groups']} to {audit['groups_after_rxnorm']}**. Ambiguous/unmapped
names retain source-based grouping; no approximate name match is guessed.

A fixed, vocabulary-free hashed word-5-gram audit found
**{audit['near_duplicate_pairs']} additional cross-component pairs** at cosine
similarity ≥ {audit['similarity_threshold']} and word-count ratio ≥ 0.9. Qualifying
pairs are grouped before splitting. This thresholded audit is not a guarantee
that every possible semantic near-duplicate or chemical synonym is resolved.
The audit representation is used only for grouping, not classifier features.

Group allocation minimizes class-document and class-group balance deviations
subject to class coverage. At least three groups per training class permit
three-fold CV; holdouts retain at least two groups per class when inventory allows.
Solver settings, actual constraints, hashes, and assignments are recorded in
[`phase2/split_manifest.json`](phase2/split_manifest.json).

## Training and selection protocol

1. Freeze partitions using data/ingredient relationships, independently of model scores.
2. Fit TF-IDF and classifiers inside each of **three group-separated training CV folds**.
3. Select each family's hyperparameters by training CV macro F1.
4. Compare tuned families on the separate validation partition; select by macro F1.
5. Persist the selection, then refit each frozen candidate on train + validation.
6. Evaluate the refitted candidates on the held-out test partition for the comparison.

`document_text` is the sole feature input. Vocabulary and IDF are learned within
CV fit folds, then relearned on train + validation for final artifacts. Metadata,
IDs, group assignments, validation text during CV, and test text during fitting
are excluded. Class weights address remaining class-size imbalance. The selected
baseline remains the validation winner even if another family has a higher test
score. Test results are not used to change hyperparameters or the winner.

TF-IDF uses unigrams/bigrams, sublinear term frequency, `min_df=2`, `max_df=0.99`,
and up to 30,000 features. Clinical source text is retained without truncation.
Actual parameters and all grid/fold results are in
[`phase2/cv_results.csv`](phase2/cv_results.csv).

![CV fold scores](phase2/figures/cv_fold_scores.png)

Large connected ingredient components make CV fold sizes and class supports
uneven. Fold-to-fold variation reflects this difficult held-out-family setting;
CV standard deviation is not a confidence interval. Class/fold support is
recorded in [`phase2/cv_class_balance.csv`](phase2/cv_class_balance.csv).

## Selected baseline: class-level test results

{_table(selected_classes)}

![Selected confusion matrix](phase2/figures/confusion_matrix_{selected}.png)

![Normalized confusion matrix](phase2/figures/selected_confusion_normalized.png)

All family confusion matrices are available under `phase2/figures/` and as CSV.
The normalized matrix shows recall per source target class.

## Repeated-label weighting and errors

{_table(grouped_table)}

Group-balanced metrics give each test ingredient component total weight one
(each of its documents receives weight `1 / group_size`). This supplements the
standard document-level metrics when many labels describe one ingredient family.
The test set has only **{manifest['partition_groups']['test']} independent
components**; these results describe this snapshot, not all medicines or clinical
practice. Some classes have few independent test ingredients.

The selected baseline made **{len(selected_errors)}** test-document errors.
Example errors, when present:

{error_text}

Full error rows and source links are in
[`phase2/test_errors.csv`](phase2/test_errors.csv). Positive linear coefficients
or global forest feature importances are in
[`phase2/top_features.csv`](phase2/top_features.csv). Genuine drug/class mentions
in the clinical document remain text features; EPC label metadata is separate.

## Deliverables and review

- `../models/phase2/benchmark.joblib` — selected complete TF-IDF/classifier pipeline
- `../models/phase2/*.joblib` — all three refitted model families
- `../data/splits/phase2_split.csv` — frozen partition and CV assignments
- `../data/reference/rxnorm_ingredient_map.json` — source-backed ingredient mapping cache
- `phase2/model_comparison.csv` — CV, validation, test, and group-balanced metrics
- `phase2/predictions_test.csv` — reproducible held-out predictions
- `phase2/validation.json` — split, feature-fit, selection, artifact, and metric checks
- `../notebooks/02_baseline_classification.ipynb` — guided executed comparison notebook
- `../docs/baseline_model_card.md` — usage, scope, and limitations

Review the selected baseline, split/group coverage, CV variation, confusion
matrices, and source-linked errors. The next course phase is a PyTorch LSTM or
1D-CNN, after your review. Later models should reuse this split and tune with
training/validation data to preserve the benchmark comparison.

## Sources

Data provided by the U.S. Food and Drug Administration (<https://open.fda.gov>).
RxNorm reference terms: <https://lhncbc.nlm.nih.gov/RxNav/TermsofService.html>.

{NLM_ATTRIBUTION}

As specified in the original plan, the project is informational only and is not
medical advice. Current targets are operational EPC-derived categories for
English medicine labels; Arabic, other document types, and out-of-scope handling
remain later extensions.
"""
    (root / "reports" / "phase_2_review.md").write_text(report, encoding="utf-8")
    model_card = f"""# Baseline model card — Phase 2

## Model and benchmark

- Selected family: **{model_names[selected]}**.
- Artifact: `models/phase2/benchmark.joblib`, schema version 1.
- Selection: validation macro F1, before held-out test evaluation.
- Hyperparameters: `{best['best_parameters']}`.
- Final fit: **{sum(manifest['partition_documents'][name] for name in ('train', 'validation')):,}** train/validation documents.
- Test: **{manifest['partition_documents']['test']}** documents in **{manifest['partition_groups']['test']}** components.
- Accuracy / macro F1: **{best['test_accuracy']:.4f} / {best['test_macro_f1']:.4f}**.
- Group-balanced macro F1: **{best['test_group_balanced_macro_f1']:.4f}**.
- Dataset and split checksums: `reports/phase2/run_manifest.json`.

## Inputs and outputs

Input is English medicine-label text. The stored pipeline includes TF-IDF and the
classifier; no external vocabulary or training cache is required. The helper
normalizes pasted HTML/encoding with the Phase 1 cleaner. Output is one of:

{', '.join('`' + label + '`' for label in labels)}.

This is a closed-set category classifier. Its output is an EPC-derived course
category, not an expert-confirmed indication, diagnosis, or recommendation.
Prescriptions, health articles, Arabic text, and unknown categories are not
represented by this initial classifier benchmark. No calibrated confidence or
unknown-document rejection is claimed by this phase.

## Use

```powershell
.\\.venv\\Scripts\\python.exe -m medical_doc_assistant classify --file "data/examples/phase2_demo_leaflet.txt"
```

```python
from pathlib import Path
from medical_doc_assistant.baseline import classify_text

leaflet_text = Path("data/examples/phase2_demo_leaflet.txt").read_text(encoding="utf-8")
result = classify_text(Path("models/phase2/benchmark.joblib"), leaflet_text)
print(result["predicted_class"])
```

For batch inference, load the bundle with `load_baseline` and call
`bundle["pipeline"].predict(list_of_clean_texts)`. The artifact metadata records
the label list, feature column, data/split hashes, and fitted partitions.

## Evaluation limits

The corpus is a selected recent openFDA snapshot with automatic metadata labels.
Ingredient/near-duplicate grouping makes held-out evaluation more independent
than a row split, but it cannot establish coverage of every chemical synonym or
semantic paraphrase. Some test classes have only two independent components.
CV folds are uneven because a large connected component spans several targets.
Standard document metrics and component-balanced metrics are both reported.

All three model configurations were frozen using training CV, the selected
family was chosen on validation, and final models were refit on train + validation
before test prediction. The full protocol, matrices, and errors are in
`reports/phase_2_review.md`. Future course models should keep the same test split
and tune on training/validation data.

## Sources

Data provided by the U.S. Food and Drug Administration (<https://open.fda.gov>).
Ingredient reference: NLM RxNorm API, cached release `{audit['rxnorm_version']}`.

{NLM_ATTRIBUTION}

As stated in the project plan, the project is informational only and is not
medical advice.
"""
    (root / "docs" / "baseline_model_card.md").write_text(model_card, encoding="utf-8")
