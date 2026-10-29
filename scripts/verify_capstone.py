"""Exercise the real API and browser workflows and save capstone verification evidence."""

import argparse
import html
import io
import json
import os
import time
from pathlib import Path

import requests
from docx import Document
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from medical_doc_assistant.grounded_qa import MISSING_ANSWER
from medical_doc_assistant.storage import write_json


def source_pdf(text: str) -> bytes:
    buffer = io.BytesIO()
    document = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = []
    for line in text.splitlines():
        if line.strip():
            elements.append(Paragraph(html.escape(line), styles["BodyText"]))
        else:
            elements.append(Spacer(1, 6))
    document.build(elements)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--frontend", default="http://127.0.0.1:8501")
    parser.add_argument("--skip-browser", action="store_true")
    parser.add_argument("--output", default="reports/phase5/integration_results.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    results, records = [], []

    def record(name: str, passed: bool, details: str) -> None:
        results.append({"name": name, "passed": bool(passed), "details": details})
        if not passed:
            raise AssertionError(name + ": " + details)

    def request(method: str, path: str, **kwargs):
        start = time.perf_counter()
        response = requests.request(method, args.api.rstrip("/") + path, timeout=240, **kwargs)
        response.raise_for_status()
        return response.json(), round(1000 * (time.perf_counter() - start), 2)

    text = (root / "data" / "examples" / "phase2_demo_leaflet.txt").read_text(encoding="utf-8")
    health, latency = request("GET", "/health")
    record("api_health", health["status"] == "ok" and health["default_model_available"], f"{latency} ms")
    classification, latency = request("POST", "/classify", json={"text": text})
    record("real_classifier", classification["predicted_class"] == "acid_reducer", f"{latency} ms; {classification['model']}")
    extraction, latency = request("POST", "/extract", json={"text": text})
    record("source_section_extraction", extraction["fields"]["dosage"]["present"] and extraction["fields"]["warnings"]["present"], f"{latency} ms")
    document, latency = request("POST", "/documents", json={"text": text, "source_name": "Real FDA demo"})
    records.append(document["document_id"])
    record("paste_document", document["classification"]["predicted_class"] == "acid_reducer", f"{latency} ms")
    answer, latency = request("POST", "/ask", json={"document_id": document["document_id"], "question": "How long can this product take for full effect?"})
    supported = all(document["text"][citation["char_start"]:citation["char_end"]] == citation["quote"] for citation in answer["citations"])
    record("real_grounded_answer", answer["status"] == "answered" and "1 to 4 days" in answer["answer"] and supported, f"{latency} ms; {answer['answer_mode']}")
    missing, latency = request("POST", "/ask", json={"document_id": document["document_id"], "question": "What is the retail price?"})
    record("real_missing_answer", missing["answer"] == MISSING_ANSWER and missing["citations"] == [], f"{latency} ms")
    pdf = source_pdf(text)
    (root / "data" / "examples" / "demo_leaflet.pdf").write_bytes(pdf)
    uploaded, latency = request("POST", "/documents/upload", files={"file": ("demo_leaflet.pdf", pdf, "application/pdf")})
    records.append(uploaded["document_id"])
    record("text_pdf_upload", uploaded["words"] > 100 and uploaded["classification"]["predicted_class"] == "acid_reducer", f"{latency} ms")
    docx = Document()
    for line in text.splitlines():
        docx.add_paragraph(line)
    buffer = io.BytesIO()
    docx.save(buffer)
    uploaded_docx, latency = request("POST", "/documents/upload", files={"file": ("demo_leaflet.docx", buffer.getvalue(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")})
    records.append(uploaded_docx["document_id"])
    record("docx_upload", uploaded_docx["extraction"]["fields"]["dosage"]["present"], f"{latency} ms")
    for identifier in records:
        request("DELETE", "/documents/" + identifier)
    response = requests.post(args.api + "/ask", json={"document_id": records[0], "question": "What is the dose?"}, timeout=10)
    record("cleared_document_isolation", response.status_code == 404, "Cleared IDs cannot query another stored document")
    if not args.skip_browser:
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(root / ".cache" / "playwright")
        from playwright.sync_api import sync_playwright
        screenshots = root / "reports" / "phase5" / "screenshots"
        screenshots.mkdir(parents=True, exist_ok=True)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1100})
            page.goto(args.frontend, wait_until="domcontentloaded")
            page.get_by_role("heading", name="Medical Document Assistant", exact=True).wait_for(timeout=60000)
            page.get_by_role("button", name="Use example leaflet", exact=True).click()
            page.get_by_text("Acid reducer", exact=True).wait_for(timeout=120000)
            page.screenshot(path=str(screenshots / "document_results.png"), full_page=True)
            record("browser_example_results", True, "Upload/example results rendered with category and source sections")
            chat = page.get_by_placeholder("Ask a question about the current document")
            chat.fill("How long can this product take for full effect?")
            chat.press("Enter")
            page.get_by_label("Chat message from assistant").get_by_text("may take 1 to 4 days for full effect 14-Day Course of Treatment", exact=True).first.wait_for(timeout=120000)
            page.get_by_text("Source evidence · chunk_0000", exact=False).first.click()
            page.get_by_label("Chat message from assistant").get_by_test_id("stText").first.wait_for(state="visible", timeout=30000)
            page.screenshot(path=str(screenshots / "grounded_chat.png"), full_page=True)
            record("browser_grounded_chat", True, "Question, source quotation, and citation expander rendered")
            chat = page.get_by_placeholder("Ask a question about the current document")
            chat.fill("What is the retail price?")
            chat.press("Enter")
            page.get_by_text(MISSING_ANSWER, exact=True).wait_for(timeout=120000)
            page.screenshot(path=str(screenshots / "missing_answer.png"), full_page=True)
            record("browser_missing_answer", True, "Missing-answer fallback displayed")
            page.get_by_role("button", name="Clear document", exact=True).click()
            page.get_by_role("heading", name="Start with a document", exact=True).wait_for(timeout=30000)
            record("browser_clear_document", True, "Document and conversation reset")
            uploader = page.locator('input[type="file"]')
            uploader.set_input_files({"name": "demo_leaflet.pdf", "mimeType": "application/pdf", "buffer": pdf})
            page.get_by_role("button", name="Analyze document", exact=True).click()
            page.get_by_text("Acid reducer", exact=True).wait_for(timeout=120000)
            record("browser_pdf_upload", True, "PDF upload through Streamlit and FastAPI completed")
            page.screenshot(path=str(screenshots / "pdf_upload.png"), full_page=True)
            browser.close()
    report = {"all_passed": all(result["passed"] for result in results), "api_url": args.api, "frontend_url": args.frontend, "checks": results}
    write_json(root / args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
