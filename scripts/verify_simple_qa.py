"""Assert conversational QA on real DOCX uploads and long fictional fixtures."""

import argparse
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import requests

from medical_doc_assistant.grounded_qa import MISSING_ANSWER, RAG_POLICY_VERSION
from medical_doc_assistant.rag_evaluation import response_supported
from medical_doc_assistant.storage import utc_now, write_json

ROOT = Path(__file__).resolve().parents[1]


def cases() -> list[dict]:
    pregnancy = ("can I take the medicine if I have pregnancy?", "can I take the medicine if I'm pregnant?",
                 "I'm pregnant, can I take this medicine?", "Is it safe while I'm pregnant?", "pregnancy")
    matrix = []
    for name in ("06_Pindolol.docx", "05_Desoximetasone.docx"):
        for question in pregnancy:
            matrix.append({"file": "upload_examples/docx/" + name, "question": question,
                           "status": "answered", "contains": ["only if", "potential risk"], "maximum_chars": 550,
                           "polarity": "conditional" if question != "pregnancy" else None})
    for question, contains in (("dosage", ["5 mg", "twice a day"]), ("how much should I take?", ["5 mg"]),
                               ("how often should I take it?", ["twice a day"]),
                               ("Can I breastfeed while taking it?", ["nursing should not be undertaken"])):
        matrix.append({"file": "upload_examples/docx/06_Pindolol.docx", "question": question,
                       "status": "answered", "contains": contains, "maximum_chars": 650})
    for question, contains in (("dosage", ["twice daily"]), ("how much should I use?", ["thin film"]),
                               ("how often should I use it?", ["twice daily"])):
        matrix.append({"file": "upload_examples/docx/05_Desoximetasone.docx", "question": question,
                       "status": "answered", "contains": contains, "maximum_chars": 650})
    long_cases = (("dosage", ["one 5 mg tablet", "do not exceed 10 mg", "18 years and older"]),
                  ("how much should I take?", ["one 5 mg tablet"]),
                  ("how often should I take it?", ["twice daily"]),
                  ("how long should I take it?", ["7 days only", "unless"]),
                  ("dosage for children", ["dosing has not been established"]),
                  ("can I take it if I'm pregnant?", ["only if", "no adequate controlled human studies"]),
                  ("can I take it with food?", ["yes, according to the document", "with food"]),
                  ("can I crush the tablet?", ["no, according to the document", "do not crush"]),
                  ("can I breastfeed?", ["no, according to the document", "nursing should not be undertaken"]),
                  ("side effects", ["dizziness, nausea, and fatigue"]),
                  ("storage", ["20 to 25 degrees"]), ("price", ["$24.95"]))
    for question, contains in long_cases:
        matrix.append({"file": "upload_examples/qa_tests/long_patient_leaflet.docx", "question": question,
                       "status": "answered", "contains": contains, "maximum_chars": 600})
    for filename in ("upload_examples/docx/06_Pindolol.docx", "upload_examples/qa_tests/long_uncertain_leaflet.docx"):
        for question in ("What is the retail price stated in the document?", "What is my insurance policy number?", "expiration date"):
            matrix.append({"file": filename, "question": question, "status": "not_found"})
    matrix.append({"file": "upload_examples/qa_tests/long_uncertain_leaflet.docx",
                   "question": "can I take it if I'm pregnant?", "status": "answered", "polarity": "uncertain",
                   "contains": ["not been established"], "maximum_chars": 550})
    for question in ("dosage", "how often should I take it?"):
        matrix.append({"file": "upload_examples/qa_tests/long_uncertain_leaflet.docx", "question": question, "status": "not_found"})
    matrix.append({"file": "upload_examples/qa_tests/long_patient_leaflet.docx", "question": "amoxicillin dosage", "status": "not_found"})
    return matrix


def wait_for_api(base: str) -> None:
    deadline = time.monotonic() + 45
    while True:
        try:
            requests.get(base + "/health", timeout=2).raise_for_status()
            return
        except requests.RequestException:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.5)


def check_api(base: str, check_ui: bool) -> dict:
    wait_for_api(base)
    documents, results, ui_results = {}, [], []
    try:
        for case in cases():
            if case["file"] not in documents:
                path = ROOT / case["file"]
                upload = requests.post(base + "/documents/upload", files={"file": (path.name, path.read_bytes())}, timeout=90)
                upload.raise_for_status()
                documents[case["file"]] = upload.json()
            document = documents[case["file"]]
            response = requests.post(base + "/ask", json={"document_id": document["document_id"], "question": case["question"]}, timeout=120)
            response.raise_for_status()
            result = response.json()
            errors = []
            if result["status"] != case["status"]:
                errors.append("wrong answerability")
            if result.get("policy_version") != RAG_POLICY_VERSION:
                errors.append("stale policy")
            if not response_supported(result, document["text"]):
                errors.append("unsupported answer/citation")
            if result["status"] == "not_found" and result["answer"] != MISSING_ANSWER:
                errors.append("incorrect missing-answer phrase")
            if result["status"] == "answered":
                if any(term.lower() not in result["answer"].lower() for term in case.get("contains", [])):
                    errors.append("important information missing")
                if len(result["answer"]) > case.get("maximum_chars", 1000):
                    errors.append("answer too long")
                if "polarity" in case and result.get("answer_polarity") != case["polarity"]:
                    errors.append("wrong yes/no/conditional interpretation")
            row = {"file": case["file"], "question": case["question"], "expected_status": case["status"],
                   "status": result["status"], "answer": result["answer"], "answer_chars": len(result["answer"]),
                   "answer_polarity": result.get("answer_polarity"), "latency_ms": result.get("latency_ms"),
                   "source_supported": response_supported(result, document["text"]), "passed": not errors, "errors": errors}
            results.append(row)
            if errors:
                print(json.dumps({"api": base, **row}), flush=True)
        if check_ui:
            from streamlit.testing.v1 import AppTest
            with patch.dict(os.environ, {"MEDICAL_API_URL": base}):
                app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=120)
                app.session_state["document"] = documents["upload_examples/docx/06_Pindolol.docx"]
                app.run()
                for label, expected in (("Pregnancy", "only if"), ("Dosage", "5 mg"), ("Price example", MISSING_ANSWER)):
                    next(button for button in app.button if button.label == label).click().run()
                    assert not app.exception, str(app.exception)
                    message = app.session_state["messages"][-1]
                    assert expected in message["content"], message["content"]
                    ui_results.append({"button": label, "passed": True})
                for question, expected in (("can I take it if I'm pregnant?", "only if"), ("dosage", "5 mg")):
                    app.chat_input[0].set_value(question).run()
                    assert not app.exception, str(app.exception)
                    assert expected in app.session_state["messages"][-1]["content"]
                    ui_results.append({"typed_question": question, "passed": True})
        summary = {"api": base, "policy_version": RAG_POLICY_VERSION, "checks": len(results),
                   "passed": sum(row["passed"] for row in results), "failed": sum(not row["passed"] for row in results),
                   "ui_checks": ui_results, "results": results}
        print(json.dumps({key: value for key, value in summary.items() if key != "results"}), flush=True)
        return summary
    finally:
        for document in documents.values():
            requests.delete(base + "/documents/" + document["document_id"], timeout=15).raise_for_status()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", action="append", help="API to verify; repeat for local and Docker")
    parser.add_argument("--ui", action="store_true", help="Also click actual Streamlit buttons and type questions")
    args = parser.parse_args()
    results = [check_api(base.rstrip("/"), args.ui) for base in (args.api or ["http://127.0.0.1:8000", "http://127.0.0.1:8001"])]
    report = {"created_at_utc": utc_now(), "policy_version": RAG_POLICY_VERSION, "apis": results,
              "scope": "Focused software regression over two real DOCX uploads and two long fictional test leaflets; not training or clinical evaluation"}
    write_json(ROOT / "reports" / "phase5" / "simple_qa_verification.json", report)
    if any(result["failed"] for result in results):
        raise SystemExit("Some simple-question checks failed; see reports/phase5/simple_qa_verification.json")


if __name__ == "__main__":
    main()
