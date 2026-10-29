"""Reproducible exploratory analysis for the first project checkpoint."""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from medical_doc_assistant.cleaning import FACT_FIELDS
from medical_doc_assistant.storage import utc_now, write_json

# Deliberately simple English stop words; medical terms are not removed or stemmed.
STOP_WORDS = frozenset("""
a about above after again against all also am an and any are as at be because been
before being below between both but by can could did do does doing down during
each few for from further had has have having he her here hers herself him himself
his how i if in into is it its itself just may me more most my myself no nor not
now of off on once only or other our ours ourselves out over own same she should
so some such than that the their theirs them themselves then there these they
this those through to too under until up very was we were what when where which
while who whom why will with would you your yours yourself yourselves
""".split())
TOKEN = re.compile(r"\b[a-z][a-z'-]{2,}\b")


def _markdown_table(frame: pd.DataFrame) -> str:
    def escape(value: Any) -> str:
        return str(value).replace("|", " / ").replace("\n", " ")

    lines = ["| " + " | ".join(map(escape, frame.columns)) + " |",
             "| " + " | ".join("---" for _ in frame.columns) + " |"]
    lines.extend("| " + " | ".join(escape(value) for value in row) + " |"
                 for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def generate_eda(config: dict[str, Any], frame: pd.DataFrame, raw_dir: Path, report_dir: Path) -> dict[str, Any]:
    figures = report_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    definitions = config["classes"]
    labels = [item["label"] for item in definitions]
    display_names = {item["label"]: item["display_name"] for item in definitions}
    balance = frame.groupby("class_label").agg(
        documents=("record_id", "size"),
        unique_ingredient_sets=("ingredient_key", "nunique"),
        leakage_groups=("leakage_group_id", "nunique"),
        median_words=("word_count", "median"),
    ).reindex(labels).reset_index()
    balance.insert(1, "display_name", balance["class_label"].map(display_names))
    balance["percent"] = (100 * balance["documents"] / len(frame)).round(2)
    balance.to_csv(report_dir / "class_balance.csv", index=False)

    missing = frame.isna()
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        missing[column] = missing[column] | frame[column].isin(["", "[]", "{}"])
    missing_table = pd.DataFrame({
        "field": missing.columns, "missing_count": missing.sum().values,
        "missing_percent": (100 * missing.mean()).round(2).values,
    }).sort_values(["missing_count", "field"], ascending=[False, True])
    missing_table.to_csv(report_dir / "missing_values.csv", index=False)
    section_missing = missing_table.set_index("field").loc[list(FACT_FIELDS)].reset_index()
    section_missing.to_csv(report_dir / "section_missingness.csv", index=False)

    lengths = frame.groupby("class_label")[["word_count", "character_count"]].describe(
        percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    )
    lengths.columns = ["_".join(column) for column in lengths.columns]
    lengths.to_csv(report_dir / "text_lengths_by_class.csv")

    global_terms: Counter[str] = Counter()
    document_frequency: Counter[str] = Counter()
    by_class: dict[str, Counter[str]] = defaultdict(Counter)
    for row in frame[["class_label", "document_text"]].itertuples(index=False):
        tokens = [token for token in TOKEN.findall(row.document_text.lower()) if token not in STOP_WORDS]
        counts = Counter(tokens)
        global_terms.update(counts)
        by_class[row.class_label].update(counts)
        document_frequency.update(counts.keys())
    terms = pd.DataFrame([
        {"rank": rank, "term": term, "count": count,
         "document_frequency": document_frequency[term],
         "document_frequency_percent": round(100 * document_frequency[term] / len(frame), 2)}
        for rank, (term, count) in enumerate(global_terms.most_common(30), 1)
    ])
    terms.to_csv(report_dir / "common_terms.csv", index=False)
    pd.DataFrame([
        {"class_label": label, "rank": rank, "term": term, "count": count}
        for label in labels for rank, (term, count) in enumerate(by_class[label].most_common(15), 1)
    ]).to_csv(report_dir / "common_terms_by_class.csv", index=False)

    groups = frame.groupby("leakage_group_id").agg(
        documents=("record_id", "size"),
        class_labels=("class_label", lambda values: ", ".join(sorted(set(values)))),
        example_generic_names=("generic_names", "first"),
    ).sort_values(["documents", "leakage_group_id"], ascending=[False, True]).reset_index()
    groups.to_csv(report_dir / "leakage_groups.csv", index=False)

    sns.set_theme(style="whitegrid", context="notebook")

    def save(fig: plt.Figure, name: str) -> None:
        fig.savefig(figures / name, dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=balance, x="documents", y="display_name", hue="display_name", palette="deep", legend=False, ax=ax)
    ax.set(title="Clean documents per class", xlabel="Documents", ylabel="")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3)
    save(fig, "class_balance.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, field, title in zip(axes, ["word_count", "character_count"], ["Document words", "Document characters"]):
        values = frame[field]
        bins = np.geomspace(values.min(), max(values.max(), values.min() * 1.1), 35)
        ax.hist(values, bins=bins, color="#5674a5", edgecolor="white")
        ax.set_xscale("log")
        ax.set(title=title, xlabel="Length (log scale)", ylabel="Documents")
    fig.tight_layout()
    save(fig, "text_length_distribution.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_frame = frame[["class_label", "word_count"]].copy()
    plot_frame["category"] = plot_frame["class_label"].map(display_names)
    sns.boxplot(data=plot_frame, x="word_count", y="category", hue="category",
                order=[display_names[label] for label in labels], palette="deep", showfliers=False, legend=False, ax=ax)
    ax.set_xscale("log")
    ax.set(title="Document length by class (outliers hidden)", xlabel="Words (log scale)", ylabel="")
    save(fig, "text_length_by_class.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=section_missing, x="missing_percent", y="field", color="#bc735c", ax=ax)
    ax.set(title="Missing source sections in the cleaned dataset", xlabel="Missing (%)", ylabel="", xlim=(0, 100))
    save(fig, "section_missingness.png")

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=terms.head(20), x="count", y="term", color="#658c73", ax=ax)
    ax.set(title="Most common terms (simple English stop words removed)", xlabel="Occurrences", ylabel="")
    save(fig, "common_terms.png")

    manifest = json.loads((raw_dir / "openfda_manifest.json").read_text(encoding="utf-8"))
    cleaning = json.loads((report_dir / "cleaning_summary.json").read_text(encoding="utf-8"))
    validation = json.loads((report_dir / "validation.json").read_text(encoding="utf-8"))
    summary = {
        "created_at_utc": utc_now(),
        "documents": len(frame),
        "classes": len(balance),
        "class_balance": balance.to_dict("records"),
        "unique_ingredient_sets": int(frame["ingredient_key"].nunique()),
        "leakage_groups": int(frame["leakage_group_id"].nunique()),
        "groups_with_multiple_documents": int((groups["documents"] > 1).sum()),
        "largest_group_documents": int(groups["documents"].max()),
        "class_imbalance_ratio": round(float(balance["documents"].max() / balance["documents"].min()), 3),
        "word_count": {key: float(value) for key, value in frame["word_count"].describe(
            percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).items()},
        "character_count": {key: float(value) for key, value in frame["character_count"].describe().items()},
        "section_missingness": section_missing.to_dict("records"),
        "synthetic_documents": int(frame["is_synthetic"].sum()),
        "source_retrieved_at_utc": manifest["retrieved_at_utc"],
        "api_last_updated": sorted({date for query in manifest["queries"] for date in query["api_last_updated"]}),
        "dataset_validation_passed": validation["all_passed"],
    }
    write_json(report_dir / "eda_summary.json", summary)
    review_balance = balance[["display_name", "documents", "percent", "leakage_groups", "median_words"]].rename(
        columns={"display_name": "Class", "documents": "Documents", "percent": "%",
                 "leakage_groups": "Leakage groups", "median_words": "Median words"})
    exclusion_table = pd.DataFrame([
        {"Reason": reason, "Records": count} for reason, count in cleaning["exclusion_reasons"].items()
    ], columns=["Reason", "Records"])
    source_table = pd.DataFrame([
        {"Class": display_names[query["class_label"]], "API matches": query["available_records"],
         "Downloaded": query["retrieved_records"]} for query in manifest["queries"]
    ])
    missing_review = section_missing.rename(columns={"field": "Source section column",
                                                     "missing_count": "Missing", "missing_percent": "Missing %"})
    report = f"""# Phase 1 — Review checkpoint

## Result

**{len(frame):,} real medicine-label documents across {len(balance)} classes** are ready
for review. The data was downloaded from openFDA, not generated. There are
{summary['synthetic_documents']} synthetic documents.

- Raw records: **{cleaning['raw_records']:,}**
- Retained: **{len(frame):,}**; excluded with recorded reasons: **{cleaning['excluded_records']:,}**
- Unique ingredient sets: **{summary['unique_ingredient_sets']}**
- Shared-ingredient leakage groups: **{summary['leakage_groups']}**
- Median document length: **{int(frame['word_count'].median()):,} words**
- Length range: **{int(frame['word_count'].min()):,}–{int(frame['word_count'].max()):,} words**
- Dataset integrity checks: **{'passed' if validation['all_passed'] else 'failed'}**
- Retrieved at: `{manifest['retrieved_at_utc']}`
- API snapshot last updated: `{', '.join(summary['api_last_updated'])}`

## Class balance

{_markdown_table(review_balance)}

![Class balance](figures/class_balance.png)

The largest/smallest class ratio is **{summary['class_imbalance_ratio']:.2f}** after
cleaning. Download caps intentionally sample selected classes; this distribution
does not estimate the distribution of medicines in openFDA or clinical practice.

## Source collection

{_markdown_table(source_table)}

Each class uses the exact EPC terms in [`../configs/phase1.json`](../configs/phase1.json),
limited to human prescription/OTC products. Results are ordered by
`effective_time:desc` and capped per class. Counts can overlap when a combination
product matches more than one query; raw IDs are deduplicated during collection.
Recent-label ordering can overrepresent repackagers, popular ingredients, and
recent updates. The cached raw snapshot and checksums preserve this experiment.

## Cleaning and exclusions

{_markdown_table(exclusion_table)}

Encoding and HTML are normalized while retaining numbers, units, inequalities,
negation, and medical wording. Source section boundaries are retained. Narrative
arrays are deduplicated when they repeat a complete subsection. Readable table
text is saved in `sections_json`, with a table fallback when a narrative section
is absent. Tables still require source inspection before interpreting doses.

The newest available version of an SPL set is retained. Identical normalized
documents are deduplicated. Texts with conflicting class labels and combinations
that match multiple selected categories are excluded. Unmapped extra EPCs remain
in metadata; the target is the unique match among the six chosen categories.
The full exclusion audit is in [`excluded_records.csv`](excluded_records.csv).

## Missing sections

{_markdown_table(missing_review)}

![Section missingness](figures/section_missingness.png)

Empty means the section is missing from the source record, not that the drug has
no warnings, contraindications, or side effects. Some OTC labels include these
facts in a general warnings section instead of the separate prescription fields.
No missing medical facts are invented. Raw-field missingness is also recorded in
[`cleaning_summary.json`](cleaning_summary.json).

## Text length and frequent terms

![Text lengths](figures/text_length_distribution.png)

![Lengths by class](figures/text_length_by_class.png)

![Common terms](figures/common_terms.png)

Lengths vary substantially between compact OTC leaflets and long prescription
labels. Future models will need an explicit text-length strategy. Frequency
tables use lowercase English words of at least three characters and a small
stop-word list; this EDA does not change the saved document text. See
[`common_terms_by_class.csv`](common_terms_by_class.csv) for class-specific terms.

## Duplicate-aware evaluation preparation

**{summary['groups_with_multiple_documents']} groups contain multiple documents**;
the largest contains **{summary['largest_group_documents']}**. Exact-text
deduplication alone does not remove near-identical labels from different
manufacturers. `leakage_group_id` connects records sharing a source ingredient
name or UNII, including combination products. Phase 2 should keep these groups
disjoint in training, validation, and testing. Salt synonyms and additional
near-duplicate relationships should be reviewed before committing the split.

Groups can span several classes through combination-product links. These related
documents should share the same train/validation/test partition. See
[`leakage_groups.csv`](leakage_groups.csv) for group sizes and class coverage.

There are only {summary['leakage_groups']} groups, so the independent evaluation
units are fewer than the document count. These labels cover selected drug classes;
prescriptions and health articles are not yet represented in the training corpus.

## What to review

1. Confirm the six starting categories in the class mapping.
2. Inspect `../data/processed/review_sample.csv` and its source links.
3. Review class balance, missing sections, and ingredient-group counts.
4. Open `../notebooks/01_data_collection_and_eda.ipynb` for the guided analysis.
5. Approve Phase 1 or request changes before Phase 2 starts.

## Deliverables

- `../data/raw/openfda_labels.jsonl.gz` — original API records
- `../data/raw/openfda_manifest.json` — source and query provenance
- `../data/processed/medicine_labels.parquet` — canonical clean dataset
- `../data/processed/medicine_labels.csv.gz` — CSV equivalent
- `../data/processed/review_sample.csv` — two complete documents per class
- `../src/medical_doc_assistant/cleaning.py` — reusable cleaning module
- `../notebooks/01_data_collection_and_eda.ipynb` — EDA notebook
- `validation.json` — dataset and export checks, with environment versions
- `../docs/data_card.md` — schema, terms, and limitations

## Scope checkpoint

This is the end of **Phase 1: Data + EDA pipeline**. The next planned course phase
is the TF-IDF baseline comparison and held-out evaluation, subject to your review.
Arabic is a later extension, consistent with the original plan's English-first
approach.

Data provided by the U.S. Food and Drug Administration (<https://open.fda.gov>).
Source terms: <https://open.fda.gov/terms/>. As stated in the plan, this project is
informational only and is not medical advice.
"""
    (report_dir / "phase_1_review.md").write_text(report, encoding="utf-8")
    return summary
