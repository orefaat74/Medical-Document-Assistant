"""Check the generated presentation, source archive, and final verification evidence."""

import json
import zipfile
from pathlib import Path

from pptx import Presentation
from pypdf import PdfReader

from medical_doc_assistant.storage import file_sha256, write_json


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    deliverables = root / "deliverables"
    results = []
    with zipfile.ZipFile(deliverables / "Source_Code_Handoff.zip") as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert "src/medical_doc_assistant/api.py" in names and "streamlit_app.py" in names
        assert not any(name.startswith((".runtime/", ".venv/", ".cache/")) for name in names)
        results.append({"name": "source_archive", "passed": True, "files": len(names)})
    deck = Presentation(deliverables / "Final_Presentation.pptx")
    pdf = PdfReader(deliverables / "Final_Presentation.pdf")
    assert len(deck.slides) == len(pdf.pages) == 8
    results.append({"name": "presentation", "passed": True, "slides": 8})
    for filename in ("integration_results.json", "container_integration_results.json", "public_demo_verification.json"):
        report = json.loads((root / "reports" / "phase5" / filename).read_text(encoding="utf-8"))
        assert report["all_passed"]
        results.append({"name": filename, "passed": True, "checks": len(report["checks"])})
    report = {"all_passed": True, "checks": results,
              "deliverable_sha256": {path.name: file_sha256(path) for path in deliverables.iterdir() if path.is_file()}}
    write_json(root / "reports" / "phase5" / "handoff_verification.json", report)
    print(json.dumps(report, indent=2))
