# Data card — Phase 1 medicine-label corpus

## Purpose

A real English document corpus for the Medical Document Assistant course project.
Phase 1 prepares data for pharmacologic-category classification and later
document-grounded retrieval. Current dataset statistics are in
[`../reports/phase_1_review.md`](../reports/phase_1_review.md).

## Source and provenance

- **Source:** openFDA Drug Label API, <https://api.fda.gov/drug/label.json>.
- **Documentation:** <https://open.fda.gov/apis/drug/label/>.
- **Underlying material:** manufacturer/distributor Structured Product Labeling
  (SPL) submissions, with openFDA-harmonized metadata.
- **Scope:** records with a selected `openfda.pharm_class_epc` and a human
  prescription or human OTC `openfda.product_type`.
- **Download strategy:** up to 600 records per selected class, in pages of 100,
  ordered by `effective_time:desc`. Exact searches are saved in the manifest.
- **Snapshot:** `data/raw/openfda_labels.jsonl.gz` contains original API records,
  deduplicated by record ID across class queries. Records are not rewritten into
  invented leaflets.
- **Dates/checksum:** `data/raw/openfda_manifest.json` records retrieval time, API
  update dates, available/retrieved counts, settings, and the raw SHA-256.
- **Language:** `en` is a corpus-scope annotation based on US label submissions,
  not automatic language detection. Arabic text/translation is a later extension.
- **Synthetic data:** none in this training corpus. Fictional software-test
  fixtures are isolated in `tests/fixtures/qa/` and `upload_examples/qa_tests/`
  and never included in the training data pipeline.

Stable source links use SPL set IDs to open DailyMed. A live source page can be
updated after collection; the local raw snapshot remains the recorded source for
this experiment. `effective_time` is the label's effective date, not the download
date or a clinical event date.

## Terms and attribution

Source terms: <https://open.fda.gov/terms/> (consulted 2026-10-28).
openFDA describes its content/data as generally public domain with a CC0 1.0
dedication unless otherwise noted, and explains that some private-party materials
may have separate rights. Those exceptions remain governed by the source terms;
the project does not assert new ownership or a blanket license over third-party
label text.

**Attribution:** Data provided by the U.S. Food and Drug Administration
(<https://open.fda.gov>).

The API documentation states that labeling is reformatted and is not
independently verified by FDA through this service. Records are not guaranteed to
represent currently distributed or approved labeling.

## Class labeling

Exact, case-insensitive EPC matching uses
[`../configs/phase1.json`](../configs/phase1.json). Target labels are derived from
metadata, not predicted from document keywords.

| Target | Included established pharmacologic families |
|---|---|
| `antibacterial` | Explicit antibacterial/antimicrobial families, including penicillins, cephalosporins, macrolides, tetracyclines, nitrofurans, and antimycobacterials |
| `anti_inflammatory_nsaid` | Nonsteroidal Anti-inflammatory Drug |
| `corticosteroid` | Corticosteroid |
| `antifungal` | Explicit antifungal families such as azoles, polyenes, echinocandins, and allylamines |
| `acid_reducer` | Proton Pump Inhibitor and Histamine-2 Receptor Antagonist |
| `antihypertensive` | Selected ACE inhibitors, ARBs, beta blockers, and calcium-channel blockers |

These are practical course categories, not a comprehensive WHO ATC ontology or
expert-adjudicated indications. A drug can have several clinical uses; an EPC
family does not establish the condition for which a specific patient uses it.
For example, beta blockers have uses beyond hypertension, and some antimicrobial
families also treat non-bacterial infections. NSAIDs and corticosteroids are
separate operational targets even though both can reduce inflammation.

- A unique match among selected targets yields one class label.
- Multiple EPCs within the same target are allowed.
- More than one selected target is ambiguous and the document is excluded.
- Additional unselected EPCs are preserved in metadata.
- Missing/unmapped EPCs are excluded; no class is guessed from drug names or text.
- Counts and excluded record IDs/reasons are recorded in the cleaning audit.

The corpus does not include an `other` class. Later inference needs a policy for
out-of-scope documents. This initial classifier corpus contains medicine labels;
prescriptions and health articles will require separate data/handling.

## Cleaning and section handling

1. Repair encoding glitches with `ftfy` and normalize Unicode to NFC.
2. Decode HTML entities and remove markup while preserving paragraph/table-cell
   boundaries; strip script/style content and unusable control characters.
3. Keep case, numbers, units, inequalities, negation, and original medical wording.
4. Normalize selected clinical source sections and retain readable table variants.
5. Remove repeated/contained strings within a source section array.
6. Build `document_text` from human-readable section headings and source text.
   Table text is used as a fallback when narrative text is missing.
7. Populate fact-section columns only from their corresponding source fields.
   These are prepared source passages, not generated structured medical facts.
8. Exclude documents below 80 whitespace-separated words and documents missing
   stable source IDs or usable ingredient metadata.
9. Retain the latest available effective date/version for an SPL set, then remove
   identical case/whitespace-normalized text. Identical text with conflicting
   selected labels is excluded entirely.

No stemming, stop-word removal, translation, dose interpretation, or truncation
is applied to saved classifier/RAG text. EDA uses a separate simple English
token-frequency view. Source tables can be complex; the original raw HTML remains
available for later table-aware processing.

## Schema

Parquet is the canonical format. CSV is gzip-compressed to keep a full-text corpus
manageable. Both exports have one row per retained document.

| Column(s) | Type | Meaning |
|---|---|---|
| `record_id`, `set_id` | string | openFDA record ID and SPL set ID |
| `effective_time`, `version` | string / integer | Source effective date (`YYYYMMDD` when present) and version |
| `class_label` | string | One selected EPC-derived target |
| `language`, `is_synthetic`, `source` | string / boolean | `en`, `False`, `openFDA` |
| `source_url` | string | DailyMed link for this SPL set |
| `brand_names`, `generic_names`, `substance_names`, `unii_codes` | JSON-array string | Source drug/ingredient identifiers |
| `pharm_class_epc`, `matched_pharm_class_epc` | JSON-array string | All source EPCs and selected matching EPCs |
| `ingredient_key`, `ingredient_key_source` | string | Exact ingredient-set key; prefers UNIIs, then substance names, then generic-name fallback |
| `leakage_group_id` | string | Connected component of shared ingredient identifiers/names |
| `document_text`, `document_hash` | string | Clinical input text and SHA-256 of case/whitespace-normalized text |
| `word_count`, `character_count`, `section_count` | integer | Descriptive text/section sizes |
| `sections_json` | JSON-object string | Cleaned source sections, including separate readable table variants |
| `indications_text`, `dosage_text`, `warnings_text`, `contraindications_text`, `side_effects_text` | string | Source-section passages; empty when no corresponding source field exists |

JSON strings keep both export formats consistent. Read them with `json.loads`.
When reading CSV, use `keep_default_na=False` to preserve intentionally empty
sections and an explicit string dtype for `effective_time`.

```python
import json
import pandas as pd

df = pd.read_parquet("data/processed/medicine_labels.parquet")
sections = json.loads(df.iloc[0]["sections_json"])

# Equivalent CSV import:
df_csv = pd.read_csv(
    "data/processed/medicine_labels.csv.gz",
    keep_default_na=False,
    dtype={"effective_time": str},
)
```

## Bias, missingness, and evaluation readiness

- This is a selected, capped, recent-label sample. Its class distribution is not
  representative of prescribing frequency or the complete openFDA corpus.
- EPC harmonization is missing for many openFDA records. This corpus measures
  coverage within the labeled subset, not completeness of all drug labels.
- Prescription labels can be much longer than OTC labels. EDA reports lengths by
  class so later truncation/chunking choices can be explicit.
- An empty source section is not evidence that a medicine has no corresponding
  risk or effect. Some facts may appear in other sections of the same document.
- Labels for different manufacturers/repackagers often share wording. Exact
  deduplication leaves near-duplicates. `leakage_group_id` connects shared source
  UNIIs and ingredient/generic names, including combination-product bridges.
- Phase 2 should split by these groups and verify disjointness; random row splits
  could significantly overestimate generalization. Salt synonyms, name variants,
  unharmonized ingredients, and remaining cross-group near-duplicates need review.
- Do not use class metadata, ingredient names as separate metadata features,
  IDs, URLs, hashes, or group IDs as classifier inputs. Genuine drug/class mentions
  in the source document remain legitimate text.
- Groups, not row count alone, determine the number of independent held-out
  evaluation units. Some classes have only a limited set of unique ingredients.

Dataset/export verification results and installed library versions are in
`reports/validation.json`. Every raw record is accounted for by a retained row or
an exclusion reason in `reports/excluded_records.csv`.

## Phase 2 evaluation metadata

Phase 2 uses the full 2,471-document corpus with additional evaluation metadata
in `data/splits/phase2_split.csv`. Its `benchmark_group_id` combines the source
ingredient components with exact/normalized RxNorm ingredient (IN) relationships
and the thresholded text-duplicate audit. Use this frozen group ID and the saved
`split`/`cv_fold` assignments for later model comparisons.

```python
assignments = pd.read_csv("data/splits/phase2_split.csv")
model_data = df.merge(
    assignments[["record_id", "benchmark_group_id", "split", "cv_fold"]],
    on="record_id",
    validate="one_to_one",
)
```

The initial benchmark has **100 components**, with **1,762 train**, **367
validation**, and **342 test** documents. The cached ingredient reference is in
`data/reference/rxnorm_ingredient_map.json`; it records source responses and the
RxNorm `05-Oct-2026` release. A unique exact/normalized name match and a unique IN
relationship are required; ambiguous names retain source-based grouping. RxNorm
is used for evaluation grouping, while target labels remain the EPC-derived
categories described above.

Reference documentation and terms:
<https://lhncbc.nlm.nih.gov/RxNav/APIs/api-RxNorm.getRelatedByType.html> and
<https://lhncbc.nlm.nih.gov/RxNav/TermsofService.html> (consulted 2026-10-28).

This product uses publicly available data from the U.S. National Library of
Medicine (NLM), National Institutes of Health, Department of Health and Human
Services; NLM is not responsible for the product and does not endorse or recommend
this or any other product.

## Phase 3 neural representation

The neural classifier uses the same saved corpus and record/group split. A fixed
Unicode-aware tokenizer generates lowercase word/number/unit tokens. During
tuning, its frequency vocabulary fits training text only; the final refit fits
the vocabulary/model on train + validation, excluding test.

Documents are represented by up to **eight contiguous 256-token windows**.
Documents of at most 2,048 regex tokens are fully retained. Longer documents use
first/last passages plus evenly spaced middle passages; convolution windows never
cross gaps between those samples. PAD=0 and UNK=1 are reserved indices.

Coverage by source record is stored in `reports/phase3/token_coverage.csv`. Token
counts differ from Phase 1 whitespace counts. The CNN's bounded context and
TF-IDF's full-text input are explicit model-comparison differences. Corpus labels
remain the same source-derived targets.

The checkpoint includes the final vocabulary, architecture, and source provenance.
See `docs/neural_model_card.md` and `reports/phase_3_review.md` for results and
scope. An exploratory PCA visualization uses training-document representations
only and is not an independent evaluation or medical clustering claim.

## Phase 4 question-answering references

`data/evaluation/phase4_qa_cases.json` contains **48 project-authored MedQuAD-style
question/answer pairs** over **7 real source documents**. These are not the
original MedQuAD corpus. Positive reference text is copied from Phase 1 cleaned
source sections/spans; no medical answer facts are generated or translated.
The selected references use single-substance leaflets with clear source fields.

Partitions are development (8 cases), calibration (20), and evaluation (20).
Ingredient components for evaluation differ from the development/calibration
components. Questions include broad section FAQs, simple span queries, and facts
absent from the document. Sources, URLs, hashes, question-authoring markers, and
expected-answerability fields are recorded with each case.

The initial evaluation exposed one supported but wrong answer type (strength for
frequency). A general temporal-answer guard corrected it. Initial results remain
in `reports/phase4/rag/archive/policy_v1/`; updated figures describe a documented
post-correction regression check, not an untouched test or clinical validation.

The QA runtime indexes only its supplied canonical document. Quotation offsets
refer to that cleaned text. The model path is English; unsupported evidence uses
the missing-answer phrase. Model asset revisions/licenses are in
`reports/phase4/model_assets.json`; the deepset SQuAD2 answerability model is
CC BY 4.0 (<https://huggingface.co/deepset/minilm-uncased-squad2>).

## Where to obtain additional data

- **More of the same source:** increase `--per-class` and use `--refresh`.
- **Larger offline label corpus:** DailyMed SPL downloads,
  <https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm>.
  Full releases are multi-part, multi-gigabyte archives; XML parsing and class
  mapping would be an additional ingestion path.
- **Alternative class mapping:** RxClass/RxNorm,
  <https://lhncbc.nlm.nih.gov/RxNav/APIs/RxClassAPIs.html>.
- **Later chatbot evaluation:** MedQuAD,
  <https://github.com/abachaa/MedQuAD>. General medical QA is not automatically
  answerable from one uploaded leaflet; evaluation cases must match the document.
- **Arabic extension:** source-backed Arabic leaflets or carefully reviewed
  translated material, with explicit original-source and synthetic/translation
  provenance. No Arabic corpus is included in Phase 1.

As required by the original plan, the project is informational only and is not
medical advice.
