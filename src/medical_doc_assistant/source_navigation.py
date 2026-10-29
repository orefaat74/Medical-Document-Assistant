"""Human-readable source locations and safely highlighted canonical source text."""

import html
from typing import Any


def location_label(block: dict[str, Any]) -> str:
    if block.get("page"):
        return "PDF page " + str(block["page"])
    if block.get("paragraph"):
        return "DOCX paragraph " + str(block["paragraph"])
    if block.get("table"):
        return "DOCX table " + str(block["table"]) + ", row " + str(block["row"])
    return "Text line " + str(block.get("line", 1))


def locate_span(blocks: list[dict[str, Any]], start: int, end: int) -> list[dict[str, Any]]:
    return [{key: value for key, value in block.items() if key != "text"} | {"label": location_label(block)}
            for block in blocks if block["char_start"] < end and block["char_end"] > start]


def enrich_citations(citations: list[dict[str, Any]], blocks: list[dict[str, Any]],
                     sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for citation in citations:
        start, end = citation["char_start"], citation["char_end"]
        locations = locate_span(blocks, start, end)
        headings = list(dict.fromkeys(section["heading"] for section in sections
                                     if section["char_start"] < end and section["char_end"] > start))
        labels = list(dict.fromkeys(location["label"] for location in locations))
        enriched.append({**citation, "locations": locations, "section_headings": headings,
                         "source_label": " · ".join([*labels, *headings]) or "Canonical source text"})
    return enriched


def highlighted_text(text: str, start: int = 0, end: int | None = None,
                     highlights: list[tuple[int, int]] | None = None) -> str:
    end = len(text) if end is None else end
    if not 0 <= start <= end <= len(text):
        raise ValueError("Source display range is outside the document.")
    ranges = sorted((max(start, left), min(end, right)) for left, right in highlights or []
                    if 0 <= left < right <= len(text) and left < end and right > start)
    merged = []
    for left, right in ranges:
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
        else:
            merged.append((left, right))
    parts, cursor = [], start
    for left, right in merged:
        parts.append(html.escape(text[cursor:left]))
        parts.append("<mark>" + html.escape(text[left:right]) + "</mark>")
        cursor = right
    parts.append(html.escape(text[cursor:end]))
    return '<div class="source-text" style="white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.65">' + "".join(parts) + "</div>"
