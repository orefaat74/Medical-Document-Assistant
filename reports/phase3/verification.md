# Phase 3 verification

Verified on **2026-10-28** with Python **3.10.6**, PyTorch **2.8.0+cpu**, and the
common dependency constraints from earlier phases.

| Check | Result |
|---|---|
| Complete neural experiment | Eight candidates trained with early stopping; selected model refit and evaluated |
| Full software regression suite | **53 tests passed**, including 15 neural checks |
| Final neural-specific rerun | **15 tests passed** |
| Experiment integrity | **95 checks passed** |
| Review notebook | **16 code cells executed successfully**, outputs saved and schema validated |
| CLI inference | Real source-backed example classified as `acid_reducer`; all 598 tokens retained |
| Dependency consistency | `pip check`: no broken requirements |
| Chart review | Learning curves, model comparison, coverage, and test confusion matrices inspected |

## Selected experiment

- Model: multi-window PyTorch TextCNN, learned word embeddings, 1,306,214 parameters.
- Candidate: `trial_08`, learning rate **0.003**, batch size **32**, dropout **0.5**.
- Validation macro F1: **0.9972**.
- Fixed final refit: **23 epochs**, **2,129** train/validation documents.
- Test: **342** documents in **15** components.
- Accuracy / macro F1: **98.54% / 0.9876**.
- Group-balanced macro F1: **0.9859**.
- Random Forest benchmark macro F1: **0.9751**; difference **+0.0125**.
- Checkpoint: `../../models/phase3/text_cnn.pt`.

The same source corpus and record split as Phase 2 are used. Tuning and epoch
selection use validation only. Final vocabulary/model fitting use train +
validation. Test inference follows selection and the fixed-epoch refit.

## What is checked

Data checks preserve source/split hashes, ingredient isolation, class coverage,
and the recorded classical comparison. Vocabulary token counts/indices are
independently regenerated from the allowed fitting partitions to check holdout
isolation. Trial-plan parameters, learning histories, checkpoint hashes, best
validation epochs, and fixed refit length are verified.

Checkpoint checks cover tensor finiteness, PAD row, architecture/vocabulary/label
dimensions, complete inference settings, and padding invariance. Saved logits
reproduce predictions, source-record truth matches, and document/group-balanced
metrics, class-weighted CE, and confusion counts are recomputed from the outputs.
Software tests additionally demonstrate actual learning on isolated non-medical
fixtures and checkpoint logit preservation after loading.

## Commands used

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[ml,dl,notebooks]" -c "requirements-phase3.txt"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m medical_doc_assistant phase3
.\.venv\Scripts\python.exe scripts/execute_notebook.py notebooks/03_pytorch_text_classifier.ipynb
.\.venv\Scripts\python.exe -m medical_doc_assistant classify-neural --file "data/examples/phase2_demo_leaflet.txt"
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_neural_classifier.py" -v
.\.venv\Scripts\python.exe -m pip check
```

Detailed provenance is in `run_manifest.json`, `model_selection.json`,
`trial_plan.json`, and `validation.json`. Text-budget diagnostics are in
`token_coverage.csv`. The test estimates apply to this source-derived snapshot;
the labels and medical claims are not independently adjudicated by this project.

**Phase 3 is complete. Phase 4 awaits user review.**
