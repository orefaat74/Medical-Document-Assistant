"""Exercise the real local retrieval/QA/generation pipeline on a development leaflet."""

import json
import logging
from pathlib import Path

from medical_doc_assistant.grounded_qa import DocumentAssistant
from medical_doc_assistant.hf_assets import load_phase4_config
from medical_doc_assistant.storage import write_json


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = Path(__file__).resolve().parents[1]
    assistant = DocumentAssistant(root, load_phase4_config(root / "configs" / "phase4.json"))
    index = assistant.index_document((root / "data" / "examples" / "phase2_demo_leaflet.txt").read_text(encoding="utf-8"), "development esomeprazole leaflet")
    results = []
    for question in (
        "How long can this product take for full effect?",
        "How often should an adult take this product?",
        "How long is one course of treatment?",
        "How should the capsule be swallowed?",
        "What does the document say about pregnancy or breast-feeding?",
        "What is the price of the product?",
        "What is the expiration date?",
    ):
        result = assistant.answer(index, question)
        candidates = assistant.answerability.candidates(question, result["retrieved_passages"])
        result["development_candidate_diagnostics"] = [{"span": item["span"], "span_score": item["span_score"],
                                                       "answer_margin": item["answer_margin"],
                                                       "retrieval_score": item["passage"]["retrieval_score"]} for item in candidates]
        results.append(result)
        print(json.dumps({key: result.get(key) for key in ("question", "status", "answer", "extracted_span", "span_score", "answer_margin", "question_coverage", "answer_mode")}, ensure_ascii=False))
    write_json(root / "reports" / "phase4" / "rag" / "development_checks.json", results)
