"""Understand short leaflet questions and compose concise answers from source spans."""

import re
from dataclasses import dataclass
from typing import Any

from medical_doc_assistant.ingestion import source_sections

FREQUENCY = re.compile(
    r"\b(?:once|twice|daily|hourly|weekly|monthly|as needed|[a-z]+ times (?:a|per) day)\b"
    r"|\bevery\s+(?:\d+(?:\s*(?:to|[-–])\s*\d+)?|one|two|four|six|eight|twelve)\s+(?:hours?|days?)\b"
    r"|\b(?:b\.i\.d|t\.i\.d|q\.i\.d|q\.d)\.?", re.IGNORECASE,
)
DOSE_AMOUNT = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?|tablets?|capsules?|puffs?|drops?)\b", re.IGNORECASE)
DOSE_ACTION = re.compile(r"\b(?:take|apply|administer|swallow|inhale|instill|recommended|initial|starting|dose|dosage)\b", re.IGNORECASE)
CONDITION = re.compile(r"\bonly if\b|\bunless\b|\bonly (?:under|on|with|when|after)\b|\bconsult\b|\bask (?:a |your )?doctor\b", re.IGNORECASE)
UNCERTAINTY = re.compile(
    r"\bno (?:adequate|controlled|human|clinical|safety|reliable|available)\b"
    r"|\bnot (?:been )?(?:established|studied|known|determined)\b"
    r"|\b(?:insufficient|inadequate)\b|without a definitive safety finding", re.IGNORECASE,
)
NEGATIVE_USE = re.compile(
    r"\b(?:do|must|should|shall) not\s+(?:(?:be|ever|normally|generally)\s+)?(?:use|used|take|taken|crush|chew|swallow|undertaken|breastfeed)\b"
    r"|\b(?:use|used|take|taken|nursing|breastfeeding)\b.{0,45}\b(?:not recommended|should not|must not|contraindicated)\b"
    r"|\bcontraindicated\b|\bavoid\b.{0,45}\b(?:use|pregnancy|pregnant|breastfeeding|crushing|chewing)\b",
    re.IGNORECASE,
)
POSITIVE_USE = re.compile(
    r"\b(?:may|can) (?:be )?(?:used|taken|use|take)\b"
    r"|\b(?:use|taking) is (?:permitted|allowed)\b|\bsafe (?:for use |to use )?(?:during|in) pregnancy\b",
    re.IGNORECASE,
)
TOPIC_FIELDS = {
    "pregnancy": ("pregnancy", "pregnancy_or_breast_feeding", "use_in_specific_populations", "warnings", "warnings_and_cautions", "precautions"),
    "breastfeeding": ("nursing_mothers", "pregnancy_or_breast_feeding", "use_in_specific_populations", "warnings", "precautions"),
    "dosage": ("dosage_and_administration", "instructions_for_use", "information_for_patients"),
    "administration": ("dosage_and_administration", "instructions_for_use", "information_for_patients", "warnings", "do_not_use"),
    "warnings": ("boxed_warning", "warnings_and_cautions", "warnings", "precautions", "do_not_use"),
    "contraindications": ("contraindications", "do_not_use"),
    "side_effects": ("adverse_reactions",),
    "uses": ("indications_and_usage", "purpose"),
    "interactions": ("drug_interactions", "ask_doctor_or_pharmacist"),
    "storage": ("storage_and_handling", "information_for_patients", "description"),
    "price": ("price",),
    "expiry": (), "phone": (), "insurance": (),
    "children": ("pediatric_use", "dosage_and_administration", "warnings", "precautions", "do_not_use"),
}
TOPIC_MARKERS = {
    "pregnancy": r"\bpregnan|\bfetus\b|\bfetal\b",
    "breastfeeding": r"\bbreast.?feed|\bnursing\b|\bhuman milk\b|\bbreast milk\b",
    "dosage": r"\bdos(?:e|age|ing)\b|\b(?:take|apply|administer|swallow|inhale|instill)\b",
    "administration": r"\b(?:food|meal|stomach|crush|chew|swallow|water)\b",
    "warnings": r"\b(?:warning|caution|risk|not|avoid|stop|doctor)\b",
    "contraindications": r"\b(?:contraindicat|not|avoid|allerg|hypersensitiv)",
    "side_effects": r"\b(?:side effect|adverse|reaction|common|reported|include)\b",
    "uses": r"\b(?:indicated|treat|treatment|relie|use|purpose)\b",
    "interactions": r"\b(?:interact|concomitant|together|combination|other medicine|avoid|not)\b",
    "storage": r"\b(?:store|storage|keep|temperature)\b",
    "price": r"\b(?:price|cost)\b.{0,100}(?:[$€£]|\b(?:USD|EUR|GBP)\b)|(?:[$€£]|\b(?:USD|EUR|GBP)\b).{0,100}\b(?:price|cost)\b",
    "children": r"\b(?:child|children|pediatric|paediatric|under \d+)\b",
}


@dataclass(frozen=True)
class QuestionIntent:
    normalized: str
    topic: str | None = None
    detail: str = "summary"
    yes_no: bool = False
    full_section: bool = False


def normalize_question(question: str) -> str:
    """Expand conversational wording without discarding negation or named subjects."""
    text = question.strip().lower().replace("’", "'").replace("‘", "'")
    contractions = {"i'm": "i am", "i've": "i have", "i'd": "i would", "i'll": "i will",
                    "it's": "it is", "that's": "that is", "what's": "what is", "there's": "there is",
                    "you're": "you are", "can't": "cannot", "won't": "will not", "isn't": "is not",
                    "aren't": "are not", "don't": "do not", "doesn't": "does not", "shouldn't": "should not"}
    for contraction, expanded in contractions.items():
        text = re.sub(r"\b" + re.escape(contraction) + r"\b", expanded, text)
    text = re.sub(r"\b(doctor|prescriber|physician)'s\b", r"\1", text)
    text = re.sub(r"\bim\b", "i am", text)
    text = re.sub(r"\b(?:expecting (?:a baby|a child)|preggers)\b", "pregnant", text)
    text = re.sub(r"\bhave pregnancy\b", "am pregnant", text)
    text = re.sub(r"\bbreast[- ]feeding\b", "breastfeeding", text)
    for typo, correction in {"dosge": "dosage", "doseage": "dosage", "pregnacy": "pregnancy",
                             "pregnent": "pregnant", "pregancy": "pregnancy", "warnigns": "warnings"}.items():
        text = re.sub(r"\b" + typo + r"\b", correction, text)
    return " ".join(text.split())


def question_intent(question: str) -> QuestionIntent:
    text = normalize_question(question)
    topic, detail = None, "summary"
    if re.search(r"\b(?:price|cost|expensive)\b|how much is (?:it|this|the medicine)", text):
        topic = "price"
    elif re.search(r"\b(?:expiry|expiration|expires|expire)\b|use[- ]by date", text):
        topic = "expiry"
    elif re.search(r"\b(?:telephone|phone)\b", text):
        topic = "phone"
    elif re.search(r"\binsurance\b.*\b(?:number|policy|id)\b", text):
        topic = "insurance"
    elif re.search(r"\b(?:dosage|dosing|dose|directions|administration)\b|how much.*(?:take|use)|how many.*(?:take|use|tablets?|pills?|capsules?)", text):
        topic = "dosage"
        if re.search(r"\b(?:maximum|max|limit)\b", text):
            detail = "maximum"
        elif re.search(r"how often|\bfrequency\b|times (?:a|per) day", text):
            detail = "frequency"
        elif re.search(r"how long.*(?:course|treatment|use|take)|\bduration\b", text) and not re.search(r"full effect|work|response", text):
            detail = "duration"
        elif re.search(r"how much|how many|\b(?:initial|starting) dose\b", text):
            detail = "amount"
    elif re.search(r"how often.*(?:take|use)|\bfrequency\b", text):
        topic, detail = "dosage", "frequency"
    elif re.search(r"how long.*(?:course|treatment|use|take)|\bduration\b", text) and not re.search(r"full effect|work|response", text):
        topic, detail = "dosage", "duration"
    elif re.search(r"\bpregnan", text):
        topic = "pregnancy"
    elif re.search(r"\bbreastfeed|\bnursing\b", text):
        topic = "breastfeeding"
    elif re.search(r"\b(?:food|meal|stomach|crush|chew|swallow|water|apply)\b|how (?:do|should|can) i (?:take|use)", text):
        topic = "administration"
    elif re.search(r"\b(?:side effects?|adverse reactions?)\b", text):
        topic = "side_effects"
    elif re.search(r"\bcontraindicat|when.*not.*(?:use|take)|who.*(?:should|must).*not", text):
        topic = "contraindications"
    elif re.search(r"\b(?:warning|warnings|precaution|precautions|caution)\b", text):
        topic = "warnings"
    elif re.search(r"\binteract|other medicines?|take.*together", text):
        topic = "interactions"
    elif re.search(r"\b(?:uses|indications|purpose)\b|what.*(?:used for|treat)|what is (?:it|this|the medicine) for", text):
        topic = "uses"
    elif re.search(r"\b(?:storage|store)\b|where.*keep", text):
        topic = "storage"
    elif re.search(r"\b(?:children|kids|child|pediatric|paediatric)\b", text):
        topic = "children"
    if topic and re.search(r"\b(?:reason|why)\b", text):
        detail = "reason"
    yes_no = bool(re.search(r"\b(?:can|may|should) (?:i|we|you|this|the medicine)\b|\bis .*\b(?:safe|allowed|okay|ok)\b", text))
    # A negative question changes yes/no logic; quote the source rather than flip it.
    yes_no = yes_no and not bool(re.search(r"\b(?:not|cannot|never)\b", text))
    full = bool(re.search(r"\b(?:full|entire|complete) (?:source |text of the )?(?:section|paragraph|text|dosage|pregnancy|breastfeeding|warnings|details)\b", text))
    return QuestionIntent(text, topic, detail, yes_no, full)


def source_passages(document: str, intent: QuestionIntent, sections: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Use shared heading recognition and retain complete sections, including their tail."""
    fields = TOPIC_FIELDS.get(intent.topic, ())
    if intent.topic == "pregnancy" and re.search(r"pregnan(?:cy|t) or breastfeeding|breastfeeding or pregnan", intent.normalized):
        fields = (*fields, "nursing_mothers")
    sections = source_sections(document) if sections is None else sections
    passages = []
    for field in fields:
        for section in sections:
            if section["field"] != field:
                continue
            passages.append({"chunk_id": "section_" + field, "text": section["text"],
                             "char_start": section["char_start"], "char_end": section["char_end"],
                             "heading": section["heading"], "retrieval_score": None,
                             "source_locator": "explicit_section_heading"})
    return passages


def _sentence_ranges(text: str) -> tuple[tuple[int, int], ...]:
    # Decimal points and internal periods in b.i.d. are not sentence boundaries.
    boundaries = [0] + [match.end() for match in re.finditer(r"\n|•|(?<=[.!?])\s+(?=[A-Z])", text)] + [len(text)]
    spans = []
    for left, right in zip(boundaries, boundaries[1:]):
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right:
            spans.append((left, right))
    return tuple(spans)


def sentence_spans(passage: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"chunk_id": passage["chunk_id"], "char_start": passage["char_start"] + left,
             "char_end": passage["char_start"] + right, "quote": passage["text"][left:right], "passage": passage["text"]}
            for left, right in _sentence_ranges(passage["text"])]


def evidence_polarity(text: str, topic: str | None) -> str | None:
    """An absence of harm in animal studies is never an affirmative permission."""
    if topic not in {"pregnancy", "breastfeeding", "administration"}:
        return None
    for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
        if NEGATIVE_USE.search(sentence) and not re.search(r"\b(?:unless|only if|only when|until)\b", sentence, re.IGNORECASE):
            return "no"
    if CONDITION.search(text):
        return "conditional"
    if NEGATIVE_USE.search(text) and re.search(r"\buntil\b", text, re.IGNORECASE):
        return "conditional"
    if NEGATIVE_USE.search(text):
        return "no"
    if UNCERTAINTY.search(text):
        return "uncertain"
    if POSITIVE_USE.search(text):
        return "yes"
    if topic == "administration" and re.search(r"\b(?:take|swallow|use)\b.{0,100}\b(?:with food|with water|with a meal)\b", text, re.IGNORECASE):
        return "yes"
    if topic in {"pregnancy", "breastfeeding"}:
        return "uncertain"
    return None


def _sentence_score(quote: str, intent: QuestionIntent, specific_terms: set[str], dedicated: bool = False) -> float:
    text = quote.lower()
    matching_text = re.sub(r"\bchildren\b", "child", normalize_question(quote))
    matching_text = re.sub(r"\b(?:doctor|physician|prescribing)\b", "prescriber", matching_text)
    if specific_terms and not all(re.search(r"\b" + re.escape(term) + r"(?:s)?\b", matching_text) for term in specific_terms):
        return -1
    topic = intent.topic
    if intent.detail == "reason":
        return 15 if re.search(r"\b(?:because|since|due to|as a result)\b", text) else -1
    if topic == "dosage":
        regimen = bool(DOSE_ACTION.search(text) and (DOSE_AMOUNT.search(text) or FREQUENCY.search(text)))
        if intent.detail == "frequency":
            return 10 + int(regimen) if FREQUENCY.search(text) and DOSE_ACTION.search(text) else -1
        if intent.detail == "duration":
            return 10 if re.search(r"\b(?:for|course|duration|continue|longer than)\b.{0,45}\b\d+\s*(?:days?|weeks?)\b", text) and not re.search(r"\bevery\b|response|effect", text) else -1
        if intent.detail == "maximum":
            return 10 if re.search(r"maximum|do not exceed|no more than|not more than", text) and DOSE_AMOUNT.search(text) else -1
        if not regimen:
            if re.search(r"\b(?:children|child|pediatric|adults?)\b", text) and re.search(r"not (?:been )?established|do not|not recommended", text):
                return 8
            return 4 if re.search(r"dosage.*individualized|dose.*individualized", text) else -1
        score = 8 + 4 * bool(re.search(r"initial|starting|recommended|\badults?\b", text))
        score += 2 * bool(FREQUENCY.search(text))
        score -= 5 * bool(re.search(r"adjust|increase|increments?|response|effect", text))
        return score
    if topic in {"pregnancy", "breastfeeding"}:
        marker = bool(re.search(TOPIC_MARKERS[topic], text))
        if topic == "pregnancy" and re.search(r"pregnan(?:cy|t) or breastfeeding|breastfeeding or pregnan", intent.normalized):
            marker = marker or bool(re.search(TOPIC_MARKERS["breastfeeding"], text))
        if not marker and not dedicated:
            return -1
        # A short directive within a dedicated section may inherit the topic heading.
        directive = bool(CONDITION.search(text) or NEGATIVE_USE.search(text) or POSITIVE_USE.search(text))
        uncertain = bool(UNCERTAINTY.search(text))
        score = 5 * marker + 10 * directive + 7 * uncertain
        if re.search(r"\b(?:rats?|rabbits?|mice|animals?|teratogenic|category [abc])\b", text) and not directive and not uncertain:
            score -= 8
        return max(0, score) if dedicated else score if score > 0 else -1
    if topic == "administration":
        focus = re.findall(r"\b(?:food|meal|stomach|crush|chew|swallow|water|apply)\b", intent.normalized)
        if focus and not any(re.search(r"\b" + word + r"(?:ing)?\b", text) for word in focus):
            return -1
        return 10 if re.search(r"\b(?:take|apply|swallow|administer|inhale|instill|crush|chew)\b", text) or NEGATIVE_USE.search(text) else -1
    if topic == "price":
        money = bool(re.search(r"(?:[$€£]\s*\d|\b\d+(?:\.\d+)?\s*(?:USD|EUR|GBP)\b)", text, re.IGNORECASE))
        return 10 if money and (dedicated or re.search(TOPIC_MARKERS[topic], text, re.IGNORECASE)) else -1
    if topic == "expiry":
        return 10 if re.search(r"\b(?:expiry|expiration|expires|use[- ]by)\b", text) and re.search(r"\b\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}\b", text) else -1
    if topic == "phone":
        return 10 if re.search(r"\b(?:phone|telephone)\b", text) and re.search(r"\+?\d[\d ()-]{7,}\d", text) else -1
    if topic == "insurance":
        return 10 if re.search(r"\binsurance\b", text) and re.search(r"\b(?:policy|number|id)\s*(?:number\s*)?[:#]\s*[a-z0-9-]*\d[a-z0-9-]*", text) else -1
    if topic == "children":
        return 10 if re.search(TOPIC_MARKERS[topic], text, re.IGNORECASE) else -1
    marker = TOPIC_MARKERS.get(topic, "")
    if not marker:
        return -1
    score = 3 * bool(re.search(marker, text))
    if topic in {"warnings", "contraindications", "interactions"}:
        score += 6 * bool(re.search(r"\b(?:not|avoid|stop|consult|contraindicated|allergic|risk)\b", text))
    if topic == "uses":
        score += 6 * bool(re.search(r"\b(?:indicated|used to|used for|relief|treatment of)\b", text))
    return score if score else 3 if dedicated else -1


def _shorten_pregnancy_quote(citation: dict[str, Any]) -> list[dict[str, Any]]:
    text = citation["quote"]
    condition = re.search(r"\bonly if\b", text, re.IGNORECASE)
    if len(text) <= 260 or not condition or not UNCERTAINTY.search(text):
        return [citation]
    # Retain both the uncertainty and the complete condition from a long research preamble.
    uncertainty = re.search(r"\b(?:there are )?no adequate.{0,100}?studies in pregnant women", text, re.IGNORECASE)
    if not uncertainty or re.search(r"\b(?:if|unless)\b", text[:condition.start()]):
        return [citation]
    excerpts = []
    for left, right in ((uncertainty.start(), uncertainty.end()), (condition.start(), len(text))):
        excerpts.append({**citation, "quote": text[left:right], "char_start": citation["char_start"] + left,
                         "char_end": citation["char_start"] + right})
    return excerpts


def concise_citations(passages: list[dict[str, Any]], intent: QuestionIntent,
                      specific_terms: set[str] | None = None) -> list[dict[str, Any]]:
    candidates = []
    for order, passage in enumerate(passages):
        primary = {"pregnancy": {"pregnancy"}, "breastfeeding": {"nursing_mothers"},
                   "uses": {"indications_and_usage", "purpose"}, "side_effects": {"adverse_reactions"},
                   "warnings": {"warnings", "warnings_and_cautions", "boxed_warning"}, "price": {"price"}}
        dedicated = passage["chunk_id"] in {"section_" + field for field in primary.get(intent.topic, {str(intent.topic)})}
        for citation in sentence_spans(passage):
            score = _sentence_score(citation["quote"], intent, specific_terms or set(), dedicated)
            if score >= 0:
                candidates.append((score, -order, citation))
    if not candidates:
        return []
    ranked = sorted(candidates, key=lambda item: (-item[0], -item[1], item[2]["char_start"]))
    selected = [ranked[0][2]]
    if intent.detail == "summary" and intent.topic != "price":
        for score, _, citation in ranked[1:]:
            if citation["quote"] in {item["quote"] for item in selected}:
                continue
            if intent.topic == "dosage":
                # Keep another population's regimen or a required limit/context, not titration history.
                if not re.search(r"\b(?:children|pediatric|adults?|do not exceed|individualized)\b", citation["quote"], re.IGNORECASE):
                    continue
            elif score < ranked[0][0] - 4:
                continue
            selected.append(citation)
            if len(selected) == (3 if intent.topic in {"warnings", "side_effects", "dosage"} else 2):
                break
    # If a population/age label is on its own preceding line, retain it with the instruction.
    if intent.topic in {"dosage", "administration"}:
        for citation in list(selected):
            passage = next(p for p in passages if p["char_start"] <= citation["char_start"] < p["char_end"])
            prior = [item for item in sentence_spans(passage) if item["char_end"] <= citation["char_start"]]
            if prior:
                label = prior[-1]
                if len(label["quote"].split()) <= 14 and re.search(r"\b(?:adults?|children|aged|years|under|over)\b", label["quote"], re.IGNORECASE) and not DOSE_ACTION.search(label["quote"]):
                    selected.append(label)
    selected.sort(key=lambda item: item["char_start"])
    if intent.topic == "pregnancy":
        selected = [excerpt for citation in selected for excerpt in _shorten_pregnancy_quote(citation)]
    unique = {}
    for citation in selected:
        unique[(citation["char_start"], citation["char_end"])] = citation
    return list(unique.values())


def render_source_answer(citations: list[dict[str, Any]], polarity: str | None = None) -> str:
    """Only source quotations, fixed evidence-backed prefixes, and standard dose abbreviations."""
    quotes = []
    for citation in citations:
        text = citation["quote"]
        for pattern, expansion in ((r"\bb\.i\.d\.?", "twice a day"), (r"\bt\.i\.d\.?", "three times a day"),
                                   (r"\bq\.i\.d\.?", "four times a day"), (r"\bq\.d\.?", "once a day")):
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        quotes.append(text)
    body = quotes[0] if len(quotes) == 1 else "\n".join("- " + quote for quote in quotes)
    prefixes = {"yes": "Yes, according to the document", "no": "No, according to the document",
                "conditional": "Only under the conditions stated in the document",
                "uncertain": "The document does not establish a definite yes or no"}
    if polarity in prefixes:
        return prefixes[polarity] + (" — " if len(quotes) == 1 else ":\n") + body
    return body


def concise_response_supported(response: dict[str, Any], document: str) -> bool:
    citations = response.get("citations", [])
    if not citations or any(not 0 <= citation["char_start"] < citation["char_end"] <= len(document)
                            or document[citation["char_start"]:citation["char_end"]] != citation["quote"] for citation in citations):
        return False
    intent = question_intent(response.get("resolved_question", response["question"]))
    polarity = evidence_polarity(" ".join(citation["quote"] for citation in citations), intent.topic) if intent.yes_no else None
    return response.get("answer_polarity") == polarity and response["answer"] == render_source_answer(citations, polarity)
