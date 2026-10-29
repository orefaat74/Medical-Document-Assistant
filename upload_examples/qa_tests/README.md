# Larger question-answering test documents

These **fictional software-test leaflets** are generated from `tests/fixtures/qa/`.
Their names, doses, studies, and prices are invented test facts and are excluded
from classifier training. The six real FDA example documents remain available
in `upload_examples/docx/`.

Generate the TXT/DOCX files with:

```powershell
.\.venv\Scripts\python.exe scripts/build_qa_test_documents.py
```

Upload `long_patient_leaflet.docx` and try:

- `can I take the medicine if I'm pregnant?` — conditional use and uncertain human evidence.
- `dosage` — adult amount/frequency, maximum, and the child restriction.
- `how often should I take it?` — frequency, not the treatment-course duration.
- `how long should I take it?` — the course duration and its exception.
- `can I crush the tablet?` — explicit source prohibition.
- `can I take it with food?` — source-supported permission with the adult condition.
- `can I breastfeed?` — the nursing restriction.
- `price` — the stated fictional price.
- `what is my insurance policy number?` — missing information.

Upload `long_uncertain_leaflet.docx` and ask about pregnancy, dosage, or price.
It supplies pregnancy uncertainty but deliberately has no dosing regimen or price.
The long research/background passages test finding important information near the
end of sections and returning concise answers with exact source evidence.

For the complete text of a section, explicitly ask `show the full dosage section`.
