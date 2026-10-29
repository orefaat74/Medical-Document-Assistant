"""Resolve short follow-ups using only the current document's conversation."""

import re
from typing import Any

from medical_doc_assistant.qa_language import normalize_question, question_intent

TOPIC_NAMES = {"dosage": "dosage", "pregnancy": "pregnancy", "breastfeeding": "breastfeeding",
               "warnings": "warnings", "side_effects": "side effects", "administration": "administration",
               "uses": "uses", "interactions": "interactions", "children": "use in children"}


def resolve_follow_up(question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = normalize_question(question).strip(" ?.! ")
    previous = next((turn for turn in reversed(history) if turn.get("response", {}).get("status") == "answered"
                     and turn["response"].get("question_topic")), None)
    topic = previous["response"]["question_topic"] if previous else None
    resolved, clarification = question.strip(), None
    if re.fullmatch(r"(?:and )?(?:how often|how frequently|frequency)", normalized):
        resolved = "How often should this medicine be used?"
    elif re.fullmatch(r"(?:and )?(?:how long|for how long|duration)", normalized):
        resolved = "How long is the treatment course?"
    elif re.fullmatch(r"(?:and )?(?:how much|what amount|how many)", normalized):
        resolved = "How much of this medicine should be used?"
    elif re.search(r"^(?:and |what about |how about |for )?(?:for )?(?:children|kids|a child|child)\b", normalized):
        resolved = "What dosage instructions are stated for children?" if topic == "dosage" else "What does the document say about use in children?"
    elif re.fullmatch(r"(?:and )?(?:is it safe|is that safe|can i take it|can i use it|is it okay|is it ok)", normalized):
        if topic in {"pregnancy", "breastfeeding"}:
            resolved = "Can I use this medicine during " + TOPIC_NAMES[topic] + "?"
        else:
            clarification = "Which situation do you mean: pregnancy, breastfeeding, taking it with food, or another condition?"
    elif re.fullmatch(r"(?:and )?(?:why|why is that|what is the reason)", normalized):
        if topic:
            resolved = "What reason does the document give about " + TOPIC_NAMES.get(topic, topic) + "?"
        else:
            clarification = "What would you like the document to explain? Ask about a specific instruction or restriction."
    elif re.fullmatch(r"(?:and )?(?:tell me more|more details|explain that|explain it)", normalized):
        if topic:
            resolved = "Show the full " + TOPIC_NAMES.get(topic, topic) + " section."
        else:
            clarification = "Which section would you like to explore: dosage, warnings, pregnancy, or another topic?"
    elif not normalized:
        clarification = "Please ask a question about the current document."
    return {"question": question.strip(), "resolved_question": resolved,
            "context_used": previous is not None and resolved != question.strip(),
            "context_topic": topic if previous and resolved != question.strip() else None,
            "clarification": clarification}
