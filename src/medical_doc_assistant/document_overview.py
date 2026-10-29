"""Source-backed identity cards, document summaries, and answerable suggestions."""

import re
from pathlib import Path
from typing import Any

from medical_doc_assistant.cleaning import SECTION_FIELDS
from medical_doc_assistant.qa_language import concise_citations, question_intent, render_source_answer, source_passages
from medical_doc_assistant.source_navigation import enrich_citations

FORM = re.compile(r"\b(?:tablets?|capsules?|cream|ointment|gel|solution|suspension|injection|inhaler|powder|drops?)\b", re.IGNORECASE)
STRENGTH = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg/mL|mcg/mL|mg|mcg|g|%)", re.IGNORECASE)
ROUTES = [(r"\b(?:topical|affected skin|external use)\b", "Topical / external"),
          (r"\b(?:by mouth|oral|swallow)\b", "Oral"), (r"\b(?:intravenous|intramuscular|injection)\b", "Injection"),
          (r"\b(?:inhale|inhalation)\b", "Inhalation"), (r"\b(?:ophthalmic|into the eye)\b", "Ophthalmic")]


def _fact(document: str, value: str, start: int, end: int, blocks: list[dict], sections: list[dict]) -> dict:
    citation = {"chunk_id": "identity", "char_start": start, "char_end": end,
                "quote": document[start:end], "passage": document[start:end]}
    return {"present": True, "text": value, "citations": enrich_citations([citation], blocks, sections)}


def _missing() -> dict:
    return {"present": False, "text": "Not explicitly stated", "citations": []}


def _topic_fact(text: str, question: str, blocks: list[dict], sections: list[dict]) -> dict:
    citations = concise_citations(source_passages(text, question_intent(question), sections), question_intent(question))
    return {"present": bool(citations), "text": render_source_answer(citations) if citations else "Not explicitly stated",
            "citations": enrich_citations(citations, blocks, sections)}


def build_overview(text: str, source_name: str, blocks: list[dict], sections: list[dict]) -> dict[str, Any]:
    identity = {key: _missing() for key in ("medicine_name", "active_ingredient", "form", "strength", "route")}
    explicit = next((section for section in sections if section["field"] == "medicine_name"), None)
    if explicit:
        value = explicit["text"].splitlines()[0]
        identity["medicine_name"] = _fact(text, value, explicit["char_start"], explicit["char_start"] + len(value), blocks, sections)
    else:
        first = text.splitlines()[0]
        headings = {field.replace("_", " ") for field in SECTION_FIELDS}
        if len(first) < 100 and first.lower().rstrip(":") not in headings and not re.search(r"\b(?:software|fictional|test|warnings|dosage|indications)\b", first, re.IGNORECASE):
            identity["medicine_name"] = _fact(text, first, 0, len(first), blocks, sections)
    ingredient = next((section for section in sections if section["field"] == "active_ingredient"), None)
    if ingredient:
        line = ingredient["text"].splitlines()[0]
        identity["active_ingredient"] = _fact(text, line, ingredient["char_start"], ingredient["char_start"] + len(line), blocks, sections)
    descriptions = [section for section in sections if section["field"] in {"description", "dosage_forms_and_strengths", "active_ingredient"}]
    hint = re.sub(r"^\d+[_ -]*", "", Path(source_name).stem).replace("_", " ")
    if not identity["medicine_name"]["present"] and re.fullmatch(r"[A-Za-z][A-Za-z -]{3,60}", hint):
        pattern = re.compile(r"\b(" + re.escape(hint) + r")\b\s*,?\s*(?:tablets?|capsules?|cream|ointment|gel|is|should|may|contains|magnesium|sodium|calcium)\b", re.IGNORECASE)
        for section in [*descriptions, *[s for s in sections if s["field"] == "dosage_and_administration"]]:
            match = pattern.search(section["text"])
            if match:
                identity["medicine_name"] = _fact(text, match.group(1), section["char_start"] + match.start(1), section["char_start"] + match.end(1), blocks, sections)
                break
    for section in descriptions:
        for key, pattern in (("form", FORM), ("strength", STRENGTH)):
            match = pattern.search(section["text"])
            if match and not identity[key]["present"]:
                identity[key] = _fact(text, match.group(), section["char_start"] + match.start(), section["char_start"] + match.end(), blocks, sections)
    if not identity["active_ingredient"]["present"]:
        for section in descriptions:
            match = re.search(r"\b(?:active ingredient(?: is)?|contains?(?: the active ingredient)?)\s*[:–-]?\s*([^.;\n]{3,140})", section["text"], re.IGNORECASE)
            if match:
                identity["active_ingredient"] = _fact(text, match.group(1), section["char_start"] + match.start(1), section["char_start"] + match.end(1), blocks, sections)
                break
    for section in [*source_passages(text, question_intent("dosage"), sections), *descriptions]:
        for pattern, label in ROUTES:
            match = re.search(pattern, section["text"], re.IGNORECASE)
            if match:
                identity["route"] = _fact(text, label, section["char_start"] + match.start(), section["char_start"] + match.end(), blocks, sections)
                break
        if identity["route"]["present"]:
            break
    summary = {key: _topic_fact(text, question, blocks, sections) for key, question in
               (("uses", "uses"), ("dosage", "dosage"), ("warnings", "warnings"),
                ("pregnancy", "pregnancy"), ("breastfeeding", "breastfeeding"))}
    # A filename is an identity hint, not an invented source fact.
    name = identity["medicine_name"]["text"] if identity["medicine_name"]["present"] else hint
    issues = []
    for section in sections:
        if section["field"] != "pregnancy":
            continue
        match = re.search(r"\b([A-Za-z][A-Za-z-]{3,30}),\s+as with any drug,\s+should\b", section["text"], re.IGNORECASE)
        if match and name and match.group(1).lower() not in name.lower() and re.search(r"\b" + re.escape(hint) + r"\b", text, re.IGNORECASE):
            fact = _fact(text, match.group(1), section["char_start"] + match.start(1), section["char_start"] + match.end(1), blocks, sections)
            issues.append({"kind": "medicine_name_mismatch", "message": "The pregnancy passage names " + match.group(1) + " while the document is identified as " + name + ".",
                           "citations": fact["citations"]})
    return {"identity": identity, "source_name_hint": hint, "summary": summary, "consistency_issues": issues,
            "method": "source excerpts and explicit identity patterns"}


def suggested_questions(text: str, sections: list[dict], overview: dict) -> list[dict]:
    topical = overview["identity"]["route"].get("text") == "Topical / external"
    candidates = [("dosage", "Dosage", "dosage"), ("pregnancy", "Pregnancy", "pregnancy"),
                  ("uses", "Uses", "what is this medicine used for?"), ("warnings", "Warnings", "warnings"),
                  ("side_effects", "Side effects", "side effects"), ("breastfeeding", "Breastfeeding", "can I breastfeed while taking it?"),
                  ("children", "Children", "what does the document say about use in children?"),
                  ("administration", "How to apply" if topical else "With food?", "how should I apply it?" if topical else "can I take it with food?"),
                  ("administration", "Crush tablets?", "can I crush the tablet?"), ("storage", "Storage", "storage"), ("price", "Price", "price")]
    suggestions = []
    for topic, label, question in candidates:
        if topical and label == "Crush tablets?":
            continue
        intent = question_intent(question)
        if concise_citations(source_passages(text, intent, sections), intent):
            suggestions.append({"id": topic + "_" + str(len(suggestions)), "label": label, "question": question,
                                "topic": topic, "source_supported": True})
    return suggestions[:8]
