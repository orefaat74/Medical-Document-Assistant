# Phase 1 verification

Verified on **2026-10-28** with the local Python **3.10.6** virtual environment.

| Check | Result |
|---|---|
| Full Phase 1 pipeline | Completed successfully using the verified openFDA snapshot |
| Software regression tests | **20 passed** |
| Dataset and export integrity | **21 checks passed** |
| EDA notebook | **13 code cells executed successfully**, outputs saved, notebook schema validated |
| Dependency consistency | `pip check`: no broken requirements |
| Chart review | Class balance, lengths, section missingness, and common-term charts inspected |

## Data snapshot

- Source: openFDA human prescription/OTC label API.
- Retrieved: `2026-10-28T11:05:52+00:00`.
- API last updated: `2026-10-27`.
- Raw records: **3,572**.
- Clean documents: **2,471** in **6** categories.
- Exclusions: **1,101**, with record-level reasons in `excluded_records.csv`.
- Synthetic training documents: **0**.
- Raw checksum and query provenance: `../data/raw/openfda_manifest.json`.
- Export checksums and byte sizes: `cleaning_summary.json`.
- Detailed integrity results and environment versions: `validation.json`.

## Commands used

From the project root in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m medical_doc_assistant collect
.\.venv\Scripts\python.exe -m medical_doc_assistant phase1
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/execute_notebook.py
.\.venv\Scripts\python.exe -m pip check
```

The software tests cover clinical text/negation/unit preservation, readable HTML
tables, missing-section behavior, metadata-based labeling, ambiguous combination
products, latest-version/text deduplication, conflicting labels, connected
ingredient groups, multi-chunk CSV round trips, rate-limit retries, API errors,
cache verification, and source-query scope.

Dataset checks recompute document lengths and normalized hashes, verify class
mapping and source metadata, check source-section JSON and ingredient grouping,
compare every CSV row's identifiers/class/full text/hashes/counts/flags with the
Parquet data, verify export checksums, and reconcile the exclusion ledger.

The notebook includes actual tables, plotted results, and a real source excerpt.
This verification establishes software and dataset consistency; it does not
independently adjudicate the source's medical claims.

**Phase 1 is complete. The next phase awaits your review.**
