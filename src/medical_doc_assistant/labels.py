"""An explicit, reviewable label scheme based on openFDA EPC metadata."""

import json
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Expected phase-1 config schema_version 1.")
    if not config.get("classes"):
        raise ValueError("At least one class definition is required.")
    labels = [item["label"] for item in config["classes"]]
    if len(labels) != len(set(labels)):
        raise ValueError("Class labels must be unique.")
    epc_mapping(config)
    for field in ("per_class", "page_size", "min_document_words"):
        if not isinstance(config.get(field), int) or config[field] <= 0:
            raise ValueError(f"{field} must be a positive integer.")
    if config["page_size"] > 1000 or config["per_class"] > 25000:
        raise ValueError("Collection settings exceed openFDA pagination limits.")
    return config


def epc_mapping(config: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in config["classes"]:
        if not item["epc_terms"]:
            raise ValueError(f"No EPC terms for {item['label']}.")
        for term in item["epc_terms"]:
            key = term.strip().casefold()
            if not key.endswith(" [epc]"):
                raise ValueError(f"Expected an established pharmacologic class: {term}")
            if key in mapping:
                raise ValueError(f"EPC term appears more than once: {term}")
            mapping[key] = item["label"]
    return mapping


def match_classes(terms: list[str], mapping: dict[str, str]) -> tuple[list[str], list[str]]:
    """Return all target matches; callers must reject cross-class combinations."""
    matched = [term for term in terms if term.strip().casefold() in mapping]
    labels = sorted({mapping[term.strip().casefold()] for term in matched})
    return labels, matched


def build_search(epc_terms: list[str]) -> str:
    """Use exact metadata terms, never heuristics on the medical document text."""
    clauses = []
    for term in epc_terms:
        if '"' in term or "\\" in term:
            raise ValueError("EPC terms cannot contain query escape characters.")
        clauses.append(f'openfda.pharm_class_epc.exact:"{term}"')
    human = '(openfda.product_type.exact:"HUMAN PRESCRIPTION DRUG" OR '
    human += 'openfda.product_type.exact:"HUMAN OTC DRUG")'
    return "(" + " OR ".join(clauses) + ") AND " + human
