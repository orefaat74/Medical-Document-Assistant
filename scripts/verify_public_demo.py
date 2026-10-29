"""Verify the temporary HTTPS frontend, websocket session, and grounded chat workflow."""

import json
import os
from pathlib import Path

from medical_doc_assistant.grounded_qa import MISSING_ANSWER
from medical_doc_assistant.storage import write_json


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    demo = json.loads((root / "reports" / "phase5" / "live_demo.json").read_text(encoding="utf-8"))
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(root / ".cache" / "playwright")
    from playwright.sync_api import sync_playwright
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        page.goto(demo["url"], wait_until="domcontentloaded")
        page.get_by_role("heading", name="Medical Document Assistant", exact=True).wait_for(timeout=90000)
        page.get_by_role("button", name="Use example leaflet", exact=True).click()
        page.get_by_text("Acid reducer", exact=True).wait_for(timeout=120000)
        chat = page.get_by_placeholder("Ask a question about the current document")
        chat.fill("How long can this product take for full effect?")
        chat.press("Enter")
        page.get_by_label("Chat message from assistant").get_by_text("may take 1 to 4 days for full effect 14-Day Course of Treatment", exact=True).first.wait_for(timeout=120000)
        chat = page.get_by_placeholder("Ask a question about the current document")
        chat.fill("What is the retail price?")
        chat.press("Enter")
        page.get_by_text(MISSING_ANSWER, exact=True).wait_for(timeout=120000)
        page.screenshot(path=str(root / "reports" / "phase5" / "screenshots" / "public_demo.png"), full_page=True)
        browser.close()
    result = {"all_passed": True, "url": demo["url"], "temporary": True,
              "checks": ["HTTPS frontend loaded", "Streamlit websocket session connected", "Real example classified", "Source-quoted answer displayed", "Missing-answer fallback displayed"]}
    write_json(root / "reports" / "phase5" / "public_demo_verification.json", result)
    print(json.dumps(result, indent=2))
