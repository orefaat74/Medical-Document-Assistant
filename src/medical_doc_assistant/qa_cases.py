"""Source-derived MedQuAD-style QA pairs; medical answer text is never invented."""

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from medical_doc_assistant.deep_learning import load_frozen_inputs
from medical_doc_assistant.storage import file_sha256, write_json

SECTION_QUESTIONS = {
    "dosage_and_administration": "What does the document say about dosage and administration?",
    "contraindications": "What does the document say about contraindications?",
    "do_not_use": "What does the document say about when not to use this product?",
    "adverse_reactions": "What does the document say about adverse reactions and side effects?",
    "pregnancy_or_breast_feeding": "What does the document say about pregnancy or breast-feeding?",
    "drug_interactions": "What does the document say about drug interactions?",
    "warnings": "What does the document say about warnings?",
}
MISSING_QUESTIONS = [
    "What is the retail price stated in this document?",
    "What is my prescribing doctor's telephone number?",
    "What is my insurance policy number?",
]


def source_pairs(row: pd.Series) -> list[dict[str, Any]]:
    sections = json.loads(row["sections_json"])
    pairs = []
    for field, question in SECTION_QUESTIONS.items():
        value = sections.get(field, "")
        if value and 15 <= len(value) <= 1500:
            pairs.append({"question": question, "expected_answer": value, "answerable": True,
                          "case_type": "section", "source_field": field})
    dosage = sections.get("dosage_and_administration", "")
    frequency = re.search(r"\b(?:twice daily|once daily|once a day|twice a day|two times a day|three times a day|three times daily)\b", dosage, re.IGNORECASE)
    if frequency:
        pairs.append({"question": "According to the dosage directions, how often is this product used?",
                      "expected_answer": frequency.group(0), "answerable": True, "case_type": "specific_span", "source_field": "dosage_and_administration"})
    for question in MISSING_QUESTIONS:
        pairs.append({"question": question, "expected_answer": "", "answerable": False, "case_type": "missing", "source_field": ""})
    return pairs


def prepare_qa_cases(root: Path, refresh: bool = False) -> dict[str, Any]:
    path = root / "data" / "evaluation" / "phase4_qa_cases.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    frame, baseline, _ = load_frozen_inputs(root)
    source_dir = root / "data" / "evaluation" / "phase4_sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    demo = json.loads((root / "data" / "examples" / "phase2_demo_leaflet.json").read_text(encoding="utf-8"))
    demo_row = frame.loc[frame["record_id"] == demo["record_id"]].iloc[0]
    documents = [(demo_row, "development")]
    used_groups = {demo_row["benchmark_group_id"]}
    for index, label in enumerate(baseline["class_labels"]):
        candidates = frame.loc[frame["class_label"] == label].sort_values(["word_count", "record_id"])
        for _, row in candidates.iterrows():
            if row["benchmark_group_id"] in used_groups or row["word_count"] < 200:
                continue
            if re.search(r"nosode|homeopathic", row["generic_names"], re.IGNORECASE):
                continue
            if len(json.loads(row["substance_names"])) != 1:
                continue
            pairs = source_pairs(row)
            if sum(pair["answerable"] for pair in pairs) >= 2:
                documents.append((row, "calibration" if index % 2 == 0 else "evaluation"))
                used_groups.add(row["benchmark_group_id"])
                break
    cases, sources = [], []
    for row, partition in documents:
        filename = row["record_id"] + ".txt"
        (source_dir / filename).write_text(row["document_text"], encoding="utf-8")
        source = {"record_id": row["record_id"], "benchmark_group_id": row["benchmark_group_id"], "partition": partition,
                  "generic_names": row["generic_names"], "source_url": row["source_url"],
                  "file": "data/evaluation/phase4_sources/" + filename, "sha256": file_sha256(source_dir / filename)}
        sources.append(source)
        for pair in source_pairs(row):
            cases.append({"case_id": f"qa_{len(cases):04d}", **pair, "partition": partition, "source_record_id": row["record_id"],
                          "source_file": source["file"], "source_url": row["source_url"], "benchmark_group_id": row["benchmark_group_id"],
                          "question_is_authored": True, "answer_is_source_quote": pair["answerable"]})
    payload = {"schema_version": 1, "format": "MedQuAD-style source-linked question/answer pairs",
               "dataset_sha256": baseline["dataset_sha256"], "sources": sources, "cases": cases,
               "description": "Questions are authored templates over selected single-substance leaflets with clear source fields; positive medical answers are exact cleaned openFDA section/span text. No synthetic medical facts.",
               "attribution": "Data provided by the U.S. Food and Drug Administration (https://open.fda.gov)."}
    write_json(path, payload)
    return payload
