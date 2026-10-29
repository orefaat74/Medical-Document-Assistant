# Baseline model card — Phase 2

## Model and benchmark

- Selected family: **Random Forest**.
- Artifact: `models/phase2/benchmark.joblib`, schema version 1.
- Selection: validation macro F1, before held-out test evaluation.
- Hyperparameters: `{"classifier__max_depth": 40, "classifier__min_samples_leaf": 2}`.
- Final fit: **2,129** train/validation documents.
- Test: **342** documents in **15** components.
- Accuracy / macro F1: **0.9708 / 0.9751**.
- Group-balanced macro F1: **0.9688**.
- Dataset and split checksums: `reports/phase2/run_manifest.json`.

## Inputs and outputs

Input is English medicine-label text. The stored pipeline includes TF-IDF and the
classifier; no external vocabulary or training cache is required. The helper
normalizes pasted HTML/encoding with the Phase 1 cleaner. Output is one of:

`antibacterial`, `anti_inflammatory_nsaid`, `corticosteroid`, `antifungal`, `acid_reducer`, `antihypertensive`.

This is a closed-set category classifier. Its output is an EPC-derived course
category, not an expert-confirmed indication, diagnosis, or recommendation.
Prescriptions, health articles, Arabic text, and unknown categories are not
represented by this initial classifier benchmark. No calibrated confidence or
unknown-document rejection is claimed by this phase.

## Use

```powershell
.\.venv\Scripts\python.exe -m medical_doc_assistant classify --file "data/examples/phase2_demo_leaflet.txt"
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
Ingredient reference: NLM RxNorm API, cached release `05-Oct-2026`.

This product uses publicly available data from the U.S. National Library of Medicine (NLM), National Institutes of Health, Department of Health and Human Services; NLM is not responsible for the product and does not endorse or recommend this or any other product.

As stated in the project plan, the project is informational only and is not
medical advice.
