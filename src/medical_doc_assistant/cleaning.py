"""Normalize source sections while preserving clinical wording and provenance."""

import hashlib
import html
import json
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pandas as pd
from ftfy import fix_text

from medical_doc_assistant.labels import epc_mapping, match_classes
from medical_doc_assistant.storage import file_sha256, read_records, utc_now, write_json

SECTION_FIELDS = (
    "indications_and_usage", "purpose", "dosage_and_administration",
    "dosage_forms_and_strengths", "contraindications", "do_not_use",
    "boxed_warning", "warnings_and_cautions", "warnings", "precautions",
    "general_precautions", "ask_doctor", "ask_doctor_or_pharmacist",
    "when_using", "stop_use", "pregnancy_or_breast_feeding", "adverse_reactions",
    "drug_interactions", "use_in_specific_populations", "pregnancy",
    "nursing_mothers", "pediatric_use", "geriatric_use", "overdosage",
    "description", "clinical_pharmacology", "mechanism_of_action",
    "information_for_patients", "patient_medication_information", "spl_medguide",
    "instructions_for_use", "keep_out_of_reach_of_children",
)
FACT_FIELDS = {
    "indications_text": ("indications_and_usage", "purpose"),
    "dosage_text": ("dosage_and_administration",),
    "warnings_text": (
        "boxed_warning", "warnings_and_cautions", "warnings", "precautions",
        "general_precautions", "ask_doctor", "ask_doctor_or_pharmacist",
        "when_using", "stop_use", "pregnancy_or_breast_feeding",
        "keep_out_of_reach_of_children",
    ),
    "contraindications_text": ("contraindications", "do_not_use"),
    "side_effects_text": ("adverse_reactions",),
}
HTML_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9]*\b[^>]*>")
HUMAN_PRODUCTS = {"HUMAN PRESCRIPTION DRUG", "HUMAN OTC DRUG"}


class _ReadableHTML(HTMLParser):
    """Keep paragraph and table-cell boundaries instead of joining adjacent doses."""

    block_tags = {"p", "div", "br", "li", "tr", "table", "paragraph", "item", "list", "caption"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self.hidden_depth += 1
        if self.hidden_depth:
            return
        if tag in self.block_tags:
            self.parts.append("\n")
        elif tag in {"td", "th"} and self.parts and not self.parts[-1].endswith("\n"):
            self.parts.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self.hidden_depth:
            self.hidden_depth -= 1
            return
        if not self.hidden_depth and tag in self.block_tags:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if not self.hidden_depth:
            self.parts.append(data)


def string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item.strip()]
    return []


def clean_text(value: str) -> str:
    """Repair encoding/HTML without lowercasing, stripping units, or changing facts."""
    text = fix_text(value, unescape_html=False, normalization="NFC")
    text = html.unescape(text).replace("\r\n", "\n").replace("\r", "\n")
    if HTML_TAG.search(text):
        parser = _ReadableHTML()
        parser.feed(text)
        parser.close()
        text = "".join(parser.parts)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    lines = [" ".join(line.split()).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _join_nonredundant(chunks: list[str]) -> str:
    unique = list(dict.fromkeys(chunk for chunk in chunks if chunk))
    # SPL sometimes repeats a subsection after the entire section in the same array.
    maximal = [chunk for chunk in unique if not any(chunk in other and chunk != other for other in unique)]
    return "\n\n".join(maximal)


def extract_sections(record: dict[str, Any]) -> dict[str, str]:
    sections: dict[str, str] = {}
    for field in SECTION_FIELDS:
        for source_field in (field, field + "_table"):
            cleaned = _join_nonredundant([clean_text(text) for text in string_list(record.get(source_field))])
            if cleaned:
                sections[source_field] = cleaned
    return sections


def section_content(sections: dict[str, str], field: str) -> str:
    # Structured tables remain in sections_json; avoid repeating them in narrative text.
    return sections.get(field) or sections.get(field + "_table", "")


def document_hash(text: str) -> str:
    canonical = " ".join(text.casefold().split())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _names(value: Any) -> list[str]:
    return sorted({clean_text(name) for name in string_list(value) if clean_text(name)})


def clean_record(
    record: dict[str, Any], mapping: dict[str, str], min_words: int
) -> tuple[dict[str, Any] | None, str]:
    record_id = record.get("id")
    set_id = record.get("set_id")
    if not record_id or not set_id:
        return None, "missing_record_or_set_id"
    metadata = record.get("openfda") or {}
    if not HUMAN_PRODUCTS.intersection(string_list(metadata.get("product_type"))):
        return None, "missing_or_non_human_product_type"
    epc_terms = _names(metadata.get("pharm_class_epc"))
    if not epc_terms:
        return None, "missing_pharm_class_epc"
    labels, matched = match_classes(epc_terms, mapping)
    if not labels:
        return None, "unmapped_pharm_class_epc"
    if len(labels) > 1:
        return None, "ambiguous_target_classes"

    substances = _names(metadata.get("substance_name"))
    generic_names = _names(metadata.get("generic_name"))
    unii_codes = _names(metadata.get("unii"))
    if unii_codes:
        ingredient_key = "unii:" + "|".join(unii_codes)
        ingredient_key_source = "unii"
    elif substances:
        ingredient_key = "substance:" + "|".join(name.casefold() for name in substances)
        ingredient_key_source = "substance_name"
    elif generic_names:
        ingredient_key = "generic:" + "|".join(name.casefold() for name in generic_names)
        ingredient_key_source = "generic_name_fallback"
    else:
        return None, "missing_ingredient_metadata"

    sections = extract_sections(record)
    document = "\n\n".join(
        field.replace("_", " ").title() + "\n" + section_content(sections, field)
        for field in SECTION_FIELDS if section_content(sections, field)
    )
    word_count = len(document.split())
    if word_count < min_words:
        return None, "short_or_empty_document"
    version = str(record.get("version", "0"))
    row = {
        "record_id": record_id,
        "set_id": set_id,
        "effective_time": str(record.get("effective_time", "")),
        "version": int(version) if version.isdecimal() else 0,
        "class_label": labels[0],
        "language": "en",
        "is_synthetic": False,
        "source": "openFDA",
        "source_url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}",
        "brand_names": json.dumps(_names(metadata.get("brand_name")), ensure_ascii=False),
        "generic_names": json.dumps(generic_names, ensure_ascii=False),
        "substance_names": json.dumps(substances, ensure_ascii=False),
        "unii_codes": json.dumps(unii_codes),
        "pharm_class_epc": json.dumps(epc_terms, ensure_ascii=False),
        "matched_pharm_class_epc": json.dumps(matched, ensure_ascii=False),
        "ingredient_key": ingredient_key,
        "ingredient_key_source": ingredient_key_source,
        "document_text": document,
        "document_hash": document_hash(document),
        "word_count": word_count,
        "character_count": len(document),
        "section_count": sum(bool(section_content(sections, field)) for field in SECTION_FIELDS),
        "sections_json": json.dumps(sections, ensure_ascii=False),
    }
    for column, source_fields in FACT_FIELDS.items():
        row[column] = _join_nonredundant([section_content(sections, field) for field in source_fields])
    return row, "kept"


def ingredient_tokens(row: dict[str, Any]) -> set[str]:
    tokens = {"unii:" + code.casefold() for code in json.loads(row["unii_codes"])}
    # A shared name also links records when one record lacks a harmonized UNII.
    for column in ("substance_names", "generic_names"):
        tokens.update("name:" + name.casefold() for name in json.loads(row[column]))
    return tokens


def add_leakage_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Connect labels that share an ingredient, including combination products."""
    rows = frame.to_dict("records")
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    owners: dict[str, int] = {}
    for index, row in enumerate(rows):
        tokens = ingredient_tokens(row) | {"set:" + row["set_id"], "text:" + row["document_hash"]}
        for token in sorted(tokens):
            if token in owners:
                first, second = find(index), find(owners[token])
                parent[max(first, second)] = min(first, second)
            else:
                owners[token] = index
    members: dict[int, list[str]] = {}
    for index, row in enumerate(rows):
        members.setdefault(find(index), []).append(row["record_id"])
    group_ids = {
        root: "group_" + hashlib.sha256("|".join(sorted(ids)).encode("utf-8")).hexdigest()[:16]
        for root, ids in members.items()
    }
    result = frame.copy()
    result["leakage_group_id"] = [group_ids[find(index)] for index in range(len(rows))]
    return result


def clean_dataset(config: dict[str, Any], raw_path: Path, output_dir: Path, report_dir: Path) -> pd.DataFrame:
    mapping = epc_mapping(config)
    rows = []
    rejected = []
    reasons: Counter[str] = Counter()
    missing_before: Counter[str] = Counter()
    raw_count = 0

    def reject(record: dict[str, Any], reason: str) -> None:
        reasons[reason] += 1
        rejected.append({"record_id": record.get("record_id", record.get("id", "")),
                         "set_id": record.get("set_id", ""), "reason": reason})

    for record in read_records(raw_path):
        raw_count += 1
        metadata = record.get("openfda") or {}
        for field in SECTION_FIELDS:
            if not string_list(record.get(field)) and not string_list(record.get(field + "_table")):
                missing_before[field] += 1
        for field in ("pharm_class_epc", "substance_name", "generic_name", "unii"):
            if not string_list(metadata.get(field)):
                missing_before["openfda." + field] += 1
        row, reason = clean_record(record, mapping, config["min_document_words"])
        if row is None:
            reject(record, reason)
        else:
            rows.append(row)
    if not rows:
        raise ValueError("No usable documents survived cleaning. See the raw data and class mapping.")
    frame = pd.DataFrame(rows).sort_values(
        ["effective_time", "version", "record_id"], ascending=[False, False, True], kind="stable"
    )
    for key, reason in (("record_id", "duplicate_record_id"), ("set_id", "older_version_of_set")):
        duplicate = frame.duplicated(key, keep="first")
        for row in frame.loc[duplicate].to_dict("records"):
            reject(row, reason)
        frame = frame.loc[~duplicate].copy()
    conflicts = frame.groupby("document_hash")["class_label"].nunique()
    conflicting = frame["document_hash"].isin(conflicts[conflicts > 1].index)
    for row in frame.loc[conflicting].to_dict("records"):
        reject(row, "identical_text_conflicting_classes")
    frame = frame.loc[~conflicting].copy()
    duplicate = frame.duplicated("document_hash", keep="first")
    for row in frame.loc[duplicate].to_dict("records"):
        reject(row, "duplicate_normalized_text")
    frame = frame.loc[~duplicate].sort_values(["class_label", "record_id"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("No documents remain after deduplication.")
    frame = add_leakage_groups(frame)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "medicine_labels.parquet"
    csv_path = output_dir / "medicine_labels.csv.gz"
    frame.to_parquet(parquet_path, index=False, compression="zstd")
    frame.to_csv(csv_path, index=False, compression={"method": "gzip", "mtime": 0})
    preview_columns = ["record_id", "class_label", "generic_names", "word_count",
                       "source_url", "leakage_group_id", "document_text"]
    frame.groupby("class_label", group_keys=False).head(2)[preview_columns].to_csv(
        output_dir / "review_sample.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(rejected, columns=["record_id", "set_id", "reason"]).to_csv(
        report_dir / "excluded_records.csv", index=False
    )
    summary = {
        "created_at_utc": utc_now(),
        "schema_version": 1,
        "raw_sha256": file_sha256(raw_path),
        "cleaning_config": config,
        "raw_records": raw_count,
        "clean_records": len(frame),
        "excluded_records": sum(reasons.values()),
        "exclusion_reasons": dict(sorted(reasons.items())),
        "missing_source_fields": {field: {"count": missing_before[field],
                                          "percent": round(100 * missing_before[field] / raw_count, 2)}
                                  for field in sorted(set(SECTION_FIELDS) | {
                                      "openfda.pharm_class_epc", "openfda.substance_name",
                                      "openfda.generic_name", "openfda.unii"})},
        "outputs": {path.name: {"sha256": file_sha256(path), "bytes": path.stat().st_size}
                    for path in (parquet_path, csv_path)},
    }
    write_json(report_dir / "cleaning_summary.json", summary)
    return frame
