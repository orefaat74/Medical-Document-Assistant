"""Integrity checks for data that will feed later training and retrieval phases."""

import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from medical_doc_assistant.cleaning import HTML_TAG, document_hash, ingredient_tokens
from medical_doc_assistant.labels import epc_mapping, match_classes
from medical_doc_assistant.storage import file_sha256, utc_now, write_json

REQUIRED_COLUMNS = {
    "record_id", "set_id", "class_label", "language", "is_synthetic", "source",
    "source_url", "generic_names", "substance_names", "unii_codes", "pharm_class_epc",
    "ingredient_key", "leakage_group_id", "document_text", "document_hash",
    "word_count", "character_count", "sections_json", "dosage_text", "warnings_text",
    "contraindications_text", "side_effects_text", "indications_text",
}


def validate_dataset(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("nonempty_dataset", not frame.empty, f"{len(frame):,} documents")
    missing = REQUIRED_COLUMNS - set(frame.columns)
    check("required_columns", not missing, "Missing: " + ", ".join(sorted(missing)) if missing else "Schema present")
    if not frame.empty and not missing:
        expected_classes = {item["label"] for item in config["classes"]}
        check("configured_classes_present", set(frame["class_label"]) == expected_classes,
              ", ".join(sorted(frame["class_label"].unique())))
        for column in ("record_id", "set_id", "document_hash"):
            check("unique_" + column, not frame[column].duplicated().any(),
                  f"{int(frame[column].duplicated().sum())} duplicate values")
        check("minimum_document_length", (frame["word_count"] >= config["min_document_words"]).all(),
              f"Minimum {int(frame['word_count'].min())} words")
        check("stored_word_counts", frame["document_text"].map(lambda text: len(text.split())).eq(frame["word_count"]).all(),
              "Word counts recomputed from exported text")
        check("stored_character_counts", frame["document_text"].str.len().eq(frame["character_count"]).all(),
              "Character counts recomputed from exported text")
        check("stored_document_hashes", frame["document_text"].map(document_hash).eq(frame["document_hash"]).all(),
              "SHA-256 of whitespace/case-normalized text")
        check("no_html_tags_in_document", not frame["document_text"].str.contains(HTML_TAG).any(),
              "Literal numeric inequalities are permitted")
        check("real_source_provenance", frame["source"].eq("openFDA").all() and frame["is_synthetic"].eq(False).all(),
              "All rows identify openFDA and is_synthetic=False")
        check("english_scope_metadata", frame["language"].eq("en").all(),
              "Corpus scope annotation; this is not automatic language detection")
        check("source_links", frame["source_url"].eq(
            "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=" + frame["set_id"]
        ).all(), "DailyMed URLs correspond to SPL set IDs")

        mapping = epc_mapping(config)
        mapping_ok = True
        sections_ok = True
        ingredient_groups_ok = True
        owners: dict[str, str] = {}
        for row in frame.to_dict("records"):
            try:
                labels, _ = match_classes(json.loads(row["pharm_class_epc"]), mapping)
                mapping_ok &= labels == [row["class_label"]]
                sections = json.loads(row["sections_json"])
                sections_ok &= isinstance(sections, dict) and bool(sections)
                sections_ok &= all(isinstance(value, str) and bool(value.strip()) for value in sections.values())
                tokens = ingredient_tokens(row)
                ingredient_groups_ok &= bool(tokens) and bool(row["ingredient_key"])
                for token in tokens:
                    if token in owners and owners[token] != row["leakage_group_id"]:
                        ingredient_groups_ok = False
                    owners[token] = row["leakage_group_id"]
            except (TypeError, ValueError, AttributeError):
                mapping_ok = sections_ok = ingredient_groups_ok = False
        check("labels_match_epc_metadata", mapping_ok, "Exactly one selected target class per document")
        check("valid_source_sections", sections_ok, "Section JSON has nonempty, readable strings")
        check("shared_ingredients_stay_grouped", ingredient_groups_ok,
              "Shared source ingredient names/UNIIs do not cross leakage groups")

    return {
        "created_at_utc": utc_now(),
        "record_count": len(frame),
        "all_passed": all(item["passed"] for item in checks),
        "checks": checks,
        "environment": {
            "python": sys.version.split()[0],
            **{package: importlib.metadata.version(package)
               for package in ("numpy", "pandas", "requests", "ftfy", "matplotlib", "seaborn", "pyarrow")},
        },
    }


def validate_exports(
    frame: pd.DataFrame, config: dict[str, Any], output_dir: Path, report_dir: Path
) -> dict[str, Any]:
    report = validate_dataset(frame, config)
    columns = ["record_id", "document_hash", "document_text", "class_label",
               "leakage_group_id", "word_count", "is_synthetic"]
    offset = 0
    csv_matches = True
    csv_path = output_dir / "medicine_labels.csv.gz"
    for chunk in pd.read_csv(csv_path, usecols=columns, keep_default_na=False, chunksize=100):
        chunk = chunk[columns].reset_index(drop=True)
        expected = frame.iloc[offset:offset + len(chunk)][columns].reset_index(drop=True)
        csv_matches &= len(expected) == len(chunk) and chunk.eq(expected).all().all()
        offset += len(chunk)
    report["checks"].append({
        "name": "csv_parquet_round_trip", "passed": bool(csv_matches and offset == len(frame)),
        "detail": f"Compared identifiers, class, full document text, hashes, counts, and flags for {offset:,} CSV rows",
    })
    cleaning_summary = json.loads((report_dir / "cleaning_summary.json").read_text(encoding="utf-8"))
    for filename, metadata in cleaning_summary["outputs"].items():
        report["checks"].append({
            "name": "checksum_" + filename,
            "passed": file_sha256(output_dir / filename) == metadata["sha256"],
            "detail": "Export matches the cleaning manifest",
        })
    report["checks"].append({
        "name": "exclusion_audit_totals",
        "passed": cleaning_summary["raw_records"] == len(frame) + cleaning_summary["excluded_records"],
        "detail": "Every raw record is retained or has an exclusion reason",
    })
    report["all_passed"] = all(check["passed"] for check in report["checks"])
    write_json(report_dir / "validation.json", report)
    if not report["all_passed"]:
        failed = [check["name"] for check in report["checks"] if not check["passed"]]
        raise ValueError("Dataset validation failed: " + ", ".join(failed))
    return report
