# Files to upload to the Medical Document Assistant

Start with **`pdf/01_Esomeprazole.pdf`**. It is a real medicine-label example and
works with the application's upload, section extraction, classification, and chat.

## How to use

1. Open the app and choose **Upload file**.
2. Select one PDF from this folder's `pdf/` subfolder.
3. Click **Analyze document**.
4. Review the category and source sections, then ask a question about that document.

The `docx/` and `txt/` versions contain the same source text. You can upload them,
or open a TXT file and paste its contents into **Paste text**.

## Included real medicine labels

| Medicine | Suggested upload | Source-derived category |
|---|---|---|
| Esomeprazole | `pdf/01_Esomeprazole.pdf` | `acid_reducer` |
| Aspirin | `pdf/02_Aspirin.pdf` | `anti_inflammatory_nsaid` |
| Nystatin | `pdf/03_Nystatin.pdf` | `antifungal` |
| Erythromycin | `pdf/04_Erythromycin.pdf` | `antibacterial` |
| Desoximetasone | `pdf/05_Desoximetasone.pdf` | `corticosteroid` |
| Pindolol | `pdf/06_Pindolol.pdf` | `antihypertensive` |

## Questions to try

For **Esomeprazole**:

- How long can this product take for full effect?
- What does the document say about dosage and administration?
- What does the document say about pregnancy or breast-feeding?
- What is the retail price? (Tests the missing-answer fallback.)

For the other files:

- What does the document say about warnings?
- What does the document say about contraindications?
- What does the document say about adverse reactions and side effects?
- According to the dosage directions, how often is this product used?

Some files lack a dedicated section. Missing source text stays empty, and a model
may decline a question when it cannot establish sufficient evidence.

## Provenance

These are upload-ready renderings of the project's **real cleaned openFDA labels**,
with the same medical text in every format. They are not official DailyMed PDF
layouts. Source IDs/URLs and file checksums are in `manifest.json`; no patient
names, invented prescriptions, or fabricated medical instructions were added.

Data provided by the U.S. Food and Drug Administration (<https://open.fda.gov>).
Use source terms: <https://open.fda.gov/terms/>. This project is informational only
and is not medical advice.

## Training data

The full model-training corpus is `../data/processed/medicine_labels.parquet` or
`../data/processed/medicine_labels.csv.gz`. For app testing, choose the individual
leaflet files above. All 18 files were parsed with the application's ingestion
code and checked against the source text.
