# Phase 4 component cards

## Fine-tuned BERT-mini classifier

- Base: `google/bert_uncased_L-4_H-256_A-4` at `387825ce42dbb39b87911cdf8e383ee3b25184f8` (Apache 2.0).
- Shape: 4 encoder layers, hidden size 256; document-level mean/max CLS pooling.
- Input: two contiguous windows of at most 256 tokens, 508 content WordPieces.
- Labels: antibacterial, anti_inflammatory_nsaid, corticosteroid, antifungal, acid_reducer, antihypertensive.
- Validation macro F1: 0.9944.
- Test accuracy / macro F1: 0.9532 / 0.9597.
- Final fit: 2129 train/validation documents, 3 fixed epochs.
- Artifact: `models/phase4/bert/model.pt` plus its adjacent `tokenizer/`.

Load with `load_bert_checkpoint` or classify with `classify_bert_text`. It is a
closed-set English label classifier. Its output is an automatic-metadata-derived
course category, not a diagnosis, expert-confirmed indication, or recommendation.
The compact model/context budget did not exceed the TextCNN on this snapshot.

## Document-grounded QA

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`, normalized 384-dimensional vectors.
- Vector store: one FAISS IndexFlatIP per canonical document hash.
- Answerability: `deepset/minilm-uncased-squad2` by deepset (CC BY 4.0).
- Local instruction model: `google/flan-t5-small` (Apache 2.0).
- Output: verbatim source quotation with source offsets, or the required missing-answer fallback.
- Policy: question coverage, answerability/null margin, literal span checks, explicit source sections for broad FAQs, and a general frequency-answer guard.

Use `DocumentAssistant(root, config).index_document(text)` and
`assistant.answer(index, question)` for application integration. Inference is
local after public model assets are downloaded. Saved indexes verify the source
text and model revision before loading. Classifier predictions do not restrict
the QA evidence source.

The 20-case/3-document QA result is a small source-template regression benchmark.
The initial frequency-answer mismatch and corrected policy results are both
recorded. It is not a clinical validation, a universal medical QA accuracy claim,
or evidence for Arabic QA. Rejected LLM generations are not displayed as answers.

## Provenance and limitations

Source FDA texts can be outdated, incomplete, or unadjudicated; quoting them does
not certify medical accuracy. Model scope is English text, explicit document
questions, and source-derived categories. Rich clinical reasoning, implicit
personal context, and unsupported facts may receive a fallback.

deepset model attribution: <https://huggingface.co/deepset/minilm-uncased-squad2>,
CC BY 4.0. All four model revision/license records are in
`reports/phase4/model_assets.json`.

Data provided by the U.S. Food and Drug Administration (<https://open.fda.gov>).

This product uses publicly available data from the U.S. National Library of Medicine (NLM), National Institutes of Health, Department of Health and Human Services; NLM is not responsible for the product and does not endorse or recommend this or any other product.

As stated in the project plan, the project is informational only and is not
medical advice.
