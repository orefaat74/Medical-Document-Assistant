# Presentation speaker notes (approximately 6–7 minutes including demo)

## Slide 1: Medical Document Assistant

Introduce the project as an educational document assistant. A user uploads or pastes text, sees an automatically predicted medicine category and source sections, then asks explicit questions with evidence. The main engineering objective is an end-to-end system: real data, reproducible model comparisons, retrieval, verified answers, and a usable interface. Emphasize that the app quotes a supplied source rather than recommending treatment. The evaluated language is English; Arabic and image OCR are scoped extensions. This presentation follows the five course phases and finishes with a live example.

## Slide 2: Architecture and application flow

Walk through the diagram from the user's file to its canonical text. FastAPI assigns an opaque handle, so the frontend keeps its questions attached to the current document. Classification and extraction are separate from QA: a wrong category does not change which source document is searched. Retrieval uses a separate FAISS index for each text. A local instruction model proposes an answer span, but a verifier returns only a supported quotation or the fallback. CPU models are cached, and shared tokenizer inference is serialized. The Docker container supervises the two web processes.

## Slide 3: Real data and fair evaluation

Explain that no medical training documents were invented. We downloaded real structured labels from openFDA, preserved the clinical sections, normalized HTML and encoding, and removed identical documents and ambiguous target combinations. Labels are automatic pharmacologic-class metadata targets, not expert-confirmed diagnoses or indications. Repackagers often supply similar text, so random row splitting would overestimate performance. The project freezes ingredient-aware components, with exact RxNorm links and duplicate checks, before training. Every model uses the same record assignment. The source manifests and hashes make the comparison auditable and reproducible.

## Slide 4: Classifier comparison

Describe the three TF-IDF baselines, the learned-embedding TextCNN, and the compact BERT. Hyperparameters use training and validation, with test results used for reporting. TextCNN was the validation-selected default and obtained the strongest recorded test result. The compact four-layer BERT did not beat it. Capacity and input budgets differ: TF-IDF reads full text, the CNN samples up to 2,048 regex tokens, and BERT samples 508 content WordPieces. All measured outcomes are retained. This is a comparison on a small source-derived snapshot, not a universal medical accuracy claim.

## Slide 5: Grounded QA and honest testing

Explain the answerability and grounding checks before showing a response. Broad section FAQs can retrieve explicit source headings; specific queries use semantic retrieval and a SQuAD2 model. Generated text is accepted only as a checked span, with source clause context included. The initial QA evaluation exposed an important bug: a frequency question received a strength description. A generic temporal-answer check corrected that, and both the initial and updated results are preserved. The updated small template-based benchmark passed its source-evidence and fallback checks. It is a regression check, not clinical validation or a fresh untouched medical QA estimate.

## Slide 6: Live document demonstration

Use about one minute for the live demonstration. Load the built-in real leaflet and point to the predicted category, document word count, and exact dosage section. Ask the full-effect question and open its source-evidence expander to show the quote and character positions. Then ask for retail price: this fact is absent, and the app should return the missing-answer phrase. Finally clear or replace the document to show conversation isolation. If network access is unavailable, use the captured browser screenshots in the report folder. Avoid interpreting the leaflet as individual medical advice; this demo shows source handling.

## Slide 7: Verification and deployment

Summarize verification at two levels. Software tests cover data isolation, padding, serialization, typed answer checks, ingestion, and API document handling. Real integration exercises the trained models, PDF and DOCX uploads, and browser chat flows. Docker packaging includes CPU PyTorch and immutable public model downloads. A deployment script stages the app for Hugging Face Spaces using the owner's local login; credentials are not placed in source. A temporary Cloudflare link can expose the running local app for review, but depends on that machine and tunnel remaining active. Persistent hosting is a separate account-backed publication step.

## Slide 8: Conclusion and next improvements

Close with the engineering lessons. Strong TF-IDF baselines matter; a larger pretrained model is not automatically better under a compact runtime and short context. Related-document splitting and source hashes improve the validity of a model comparison. Source support alone is not enough for QA: the answer must also match the requested type, and negative questions need evaluation. The final application is usable and modular, but the source labels and clinical claims remain unadjudicated. Future work should broaden independent QA references, add tested Arabic and OCR support, and operate a persistent deployment with appropriate identity and storage design. Invite questions about the architecture or evaluation.

