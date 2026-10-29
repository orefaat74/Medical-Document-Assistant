# Simple document questions and concise answers

The document assistant recognizes short topic requests and conversational wording.
Its answer is grounded in the current upload, with original quotations and exact
character offsets available under **Source evidence**.

## Questions to try

| Question | What the assistant looks for |
|---|---|
| `dosage` / `dose` / `directions` | Important dosing instructions and their conditions |
| `how much should I take?` | The stated amount or starting instructions |
| `how often should I take it?` | Frequency, rather than a strength or treatment duration |
| `how long should I take it?` | A stated course duration and any exception |
| `maximum dosage` | The limit together with its age/quantity conditions |
| `dosage for children` | Child-specific instructions or an explicit lack of an established regimen |
| `can I take it if I'm pregnant?` | Pregnancy instructions, restrictions, conditions, and uncertainty |
| `can I breastfeed while taking it?` | Nursing/breastfeeding information |
| `can I crush the tablet?` | Explicit crushing/administration instructions |
| `side effects`, `warnings`, `storage`, `price` | Brief, relevant source statements |
| `show the full dosage section` | The complete source section when explicitly requested |

Straight and curly apostrophes, `I'm` / `I am` / `im`, and `I have pregnancy`
are handled without treating the conversational words as missing document facts.

## Answer behavior

- Ordinary topic questions receive a few relevant source statements rather than
  an entire research/background paragraph.
- Explicit source permission or prohibition can produce **Yes** or **No**, scoped
  to what the document says.
- `only if`, `unless`, and other source conditions remain **conditional**.
- Missing human safety evidence remains **uncertain**, rather than being turned
  into permission based on animal findings.
- Age limits, decimal quantities, units, maxima, and exceptions remain attached
  to the instructions. Standard frequency abbreviations such as `b.i.d.` are
  displayed as `twice a day`; evidence retains the original abbreviation.
- A specific absent drug or patient group is not answered from unrelated text in
  another section. Answers from one upload are not reused for another upload.

## Price example and missing facts

The former **Missing fact** suggestion is labeled **Price example**. It asks for
the document's retail price. A price is returned if it is actually stated.
Otherwise the answer remains **“I cannot find that in the document.”**, with an
explanation that no retail price was found. Personal insurance numbers, telephone
numbers, and expiration dates likewise require actual source values.

## Larger upload tests

`upload_examples/qa_tests/` contains TXT and DOCX copies of two **fictional
software-test leaflets**, generated from `tests/fixtures/qa/`:

- `long_patient_leaflet.docx`: **2,052 words / 13,437 canonical characters**;
  long background passages, adult/child directions, conditional pregnancy use,
  a nursing restriction, administration instructions, and a stated test price.
- `long_uncertain_leaflet.docx`: **1,048 words / 6,894 canonical characters**;
  long research background, uncertain human pregnancy evidence, and missing doses
  and price.

Their product names, doses, studies, and prices are invented test data, excluded
from classifier training. The real FDA examples remain in `upload_examples/docx/`.

## Verification — QA policy v6

- **116 software tests passed**; many tests exercise multiple question variants.
- **39 API checks passed on the native service** and **39 on Docker**.
- **Five Streamlit interaction checks passed with each backend**: Pregnancy,
  Dosage, Price example, typed pregnancy wording, and typed `dosage`.
- Follow-up chat, the source-backed overview, document-aware suggestions, DOCX
  paragraph/table locations, PDF page locations, and highlighted jump-to-source
  navigation have dedicated regression coverage.
- Real-file checks include `06_Pindolol.docx` and the edited
  `05_Desoximetasone.docx`; long-fixture checks exercise relevant facts beyond the
  previous 1,800-character section excerpt limit.
- Checks assert answerability, important information, short answer length,
  conditional/uncertain interpretation, current policy, and original citation
  offsets. Evidence: `reports/phase5/simple_qa_verification.json`.

The edited Desoximetasone file's added pregnancy paragraph names **pindolol**;
these checks verify reading that supplied text, not correcting its drug attribution.
This is focused software regression evidence; the original policy-v2 QA benchmark
remains historical, and these checks are not clinical validation.

```powershell
# Recreate only the additional test uploads:
.\.venv\Scripts\python.exe scripts/build_qa_test_documents.py

# Software regression suite:
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Running native + Docker APIs, actual suggestion buttons, and typed chat:
.\.venv\Scripts\python.exe scripts/verify_simple_qa.py --ui
```

After a backend restart, refresh the browser, upload the file again, and click
**Analyze document** before asking a question.
