"""Inspect ingestion/section detection for a supplied document without editing it."""

import argparse
import json
import re
import requests
from pathlib import Path

from docx import Document

from medical_doc_assistant.grounded_qa import broad_section_passage, question_coverage
from medical_doc_assistant.ingestion import read_document_file, source_sections


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    parser.add_argument("--api", help="Also reproduce the uploaded-document QA workflow against this API URL")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    settings = json.loads((root / "configs" / "phase5.json").read_text(encoding="utf-8"))
    if args.file.suffix.lower() == ".docx":
        document = Document(args.file)
        print("DOCX pregnancy paragraphs:")
        for index, paragraph in enumerate(document.paragraphs):
            if re.search(r"pregnan|breast|lactat", paragraph.text, re.IGNORECASE):
                print(json.dumps({"paragraph": index, "style": paragraph.style.name, "text": paragraph.text}, ensure_ascii=False))
    text = read_document_file(args.file.name, args.file.read_bytes(), settings)
    print("Detected source headings:", [section["heading"] for section in source_sections(text)])
    print("Canonical document tail:", json.dumps(text[-1800:]))
    for question in ("What does the document say about pregnancy?", "Can I use this while pregnant?", "Is it safe during pregnancy?", "What is the pregnancy information?"):
        print(json.dumps({"question": question, "coverage": question_coverage(question, text), "section": broad_section_passage(text, question)}))
    if args.api:
        response = requests.post(args.api.rstrip("/") + "/documents/upload", files={"file": (args.file.name, args.file.read_bytes())}, timeout=180)
        response.raise_for_status()
        document = response.json()
        print("Fresh upload:", json.dumps({"document_id": document["document_id"], "pregnancy_present": "Pregnancy" in document["text"]}))
        for question in ("What does the document say about pregnancy?", "Can I use this while pregnant?", "Is it safe during pregnancy?", "What is the pregnancy information?"):
            result = requests.post(args.api.rstrip("/") + "/ask", json={"document_id": document["document_id"], "question": question}, timeout=180)
            result.raise_for_status()
            data = result.json()
            print("API result:", json.dumps({"question": question, "status": data["status"], "answer": data["answer"], "question_coverage": data.get("question_coverage")}))
        requests.delete(args.api.rstrip("/") + "/documents/" + document["document_id"], timeout=15)
