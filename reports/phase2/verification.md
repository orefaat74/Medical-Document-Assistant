# Phase 2 verification

Verified on **2026-10-28**, using Python **3.10.6** and the versions recorded in
`run_manifest.json` / `requirements-phase2.txt`.

| Check | Result |
|---|---|
| Full Phase 2 experiment | Completed: all three families trained, selected, refit, saved, and evaluated |
| Full software regression suite | **38 tests passed**, including 18 new Phase 2 checks |
| Experiment integrity | **75 checks passed** |
| Comparison notebook | **15 code cells executed successfully**, outputs saved, notebook schema validated |
| Command-line inference | Real source-backed example classified as `acid_reducer` |
| Dependency consistency | `pip check`: no broken requirements |
| Chart inspection | Split balance, model comparison, CV-fold scores, and selected raw/normalized confusion matrices reviewed |

## Experiment and selection

- Data: **2,471** real Phase 1 openFDA label documents, **6** target classes.
- Frozen partitions: **1,762 train / 367 validation / 342 test**.
- Benchmark ingredient components: **100**, including **15** test components.
- Cross-validation: **3** whole-group training folds, **10** grid configurations.
- Selected model: **Random Forest**, by validation macro F1 **0.8671**.
- Selected model test accuracy / macro F1: **97.08% / 0.9751**.
- Selected group-balanced test macro F1: **0.9688**.
- Final artifact refit: **2,129** train/validation documents.
- Benchmark artifact: `../../models/phase2/benchmark.joblib`.

The model-selection snapshot was written before test prediction. Logistic
Regression has a slightly higher document-level test score; the benchmark remains
the pre-test validation selection. All three results are retained in the model
comparison table.

## What is verified

Split checks cover whole-component isolation, preservation of Phase 1 groups,
source-name/UNII/RxNorm ingredient links, qualifying duplicate pairs, complete
class coverage, minimum independent groups, and CV isolation.

Artifact checks verify source/split/reference/config/output hashes, standalone
TF-IDF/classifier pipelines, fitted partitions, frozen parameters, all-class
support, and the selected benchmark alias. Final vocabulary/IDF are independently
compared with a fit using train + validation only, excluding every test text.

Prediction checks compare source truth and group IDs, complete validation/test
record coverage, all reported document/group-balanced metrics, and each family
confusion matrix. The software tests additionally observe separate TF-IDF fits
inside CV, reject holdout rows for tuning, test serialization for all families,
reject ambiguous ingredient names, and check group-weight invariance.

## Commands used

From the project root in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[ml,notebooks]" -c "requirements-phase2.txt"
.\.venv\Scripts\python.exe scripts/collect_rxnorm_reference.py
.\.venv\Scripts\python.exe -m medical_doc_assistant prepare-phase2
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m medical_doc_assistant phase2
.\.venv\Scripts\python.exe scripts/execute_notebook.py notebooks/02_baseline_classification.ipynb
.\.venv\Scripts\python.exe -m medical_doc_assistant classify --file "data/examples/phase2_demo_leaflet.txt"
.\.venv\Scripts\python.exe -m pip check
```

Detailed results are in `validation.json`, `split_manifest.json`,
`model_selection.json`, and `run_manifest.json`. The notebook's example metadata
records the source ID/URL, target, synthetic flag, and dataset checksum.

The checks establish experiment consistency against the source-derived labels;
they do not independently adjudicate medical claims. Dataset coverage and model
scope are documented in `../../docs/data_card.md` and
`../../docs/baseline_model_card.md`.

**Phase 2 is complete. Phase 3 awaits your review.**
