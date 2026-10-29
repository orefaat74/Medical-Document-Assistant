"""Fabricated software fixtures only: these are never added to the real dataset."""

import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from medical_doc_assistant.cleaning import add_leakage_groups, clean_dataset, clean_record, clean_text
from medical_doc_assistant.collection import OpenFDAClient, collect_data
from medical_doc_assistant.labels import build_search, epc_mapping
from medical_doc_assistant.storage import read_records
from medical_doc_assistant.validation import validate_dataset, validate_exports

ROOT = Path(__file__).resolve().parents[1]
PPI = "Proton Pump Inhibitor [EPC]"
STEROID = "Corticosteroid [EPC]"
CONFIG = {
    "schema_version": 1, "per_class": 1, "page_size": 1, "min_document_words": 1,
    "classes": [
        {"label": "acid_reducer", "display_name": "Acid reducer", "epc_terms": [PPI]},
        {"label": "corticosteroid", "display_name": "Corticosteroid", "epc_terms": [STEROID]},
    ],
}
MAPPING = epc_mapping(CONFIG)


def source_record(identifier: str, *, epc: list[str] | None = None, names: list[str] | None = None) -> dict:
    ingredients = names or ["Example ingredient"]
    return {
        "id": identifier, "set_id": "set-" + identifier, "effective_time": "20260101", "version": "1",
        "openfda": {
            "pharm_class_epc": epc if epc is not None else [PPI],
            "product_type": ["HUMAN PRESCRIPTION DRUG"],
            "generic_name": [" + ".join(ingredients)], "substance_name": ingredients,
            "unii": ["TEST-" + name.upper() for name in ingredients],
        },
        "indications_and_usage": [f"Fabricated software fixture {identifier}. This text is only for a unit test."],
    }


def cleaned(identifier: str, **kwargs: object) -> dict:
    row, reason = clean_record(source_record(identifier, **kwargs), MAPPING, 1)
    if row is None:
        raise AssertionError(reason)
    return row


class TextCleaningTests(unittest.TestCase):
    def test_preserves_doses_units_inequalities_and_negation(self) -> None:
        result = clean_text("<p>Do NOT exceed 2.5 mg; use 5 µg only if age &lt; 12. Limit ≥ 4 mL.</p>")
        self.assertEqual(result, "Do NOT exceed 2.5 mg; use 5 µg only if age < 12. Limit ≥ 4 mL.")

    def test_repairs_mojibake(self) -> None:
        self.assertEqual(clean_text("Childrenâ€™s dose: 5 Âµg."), "Children's dose: 5 µg.")

    def test_preserves_table_cell_and_row_boundaries(self) -> None:
        table = "<table><tr><th>Age</th><th>Dose</th></tr><tr><td>Adult</td><td>2.5 mg</td></tr></table>"
        self.assertEqual(clean_text(table), "Age | Dose\nAdult | 2.5 mg")

    def test_removes_script_and_style_contents(self) -> None:
        self.assertEqual(clean_text("<style>.hidden { color:red; }</style><p>Keep text.</p><script>bad()</script>"), "Keep text.")

    def test_collapses_duplicate_source_subsections(self) -> None:
        record = source_record("subsections")
        record["warnings"] = ["Full warning. Repeated subsection.", "Repeated subsection."]
        row, _ = clean_record(record, MAPPING, 1)
        self.assertEqual(row["warnings_text"], "Full warning. Repeated subsection.")

    def test_missing_medical_sections_stay_empty(self) -> None:
        record = source_record("missing")
        record["warnings"] = ["A warning about possible side effects."]
        row, _ = clean_record(record, MAPPING, 1)
        self.assertEqual(row["dosage_text"], "")
        self.assertEqual(row["contraindications_text"], "")
        self.assertEqual(row["side_effects_text"], "")

    def test_table_is_retained_and_used_when_narrative_is_missing(self) -> None:
        record = source_record("table")
        record["dosage_and_administration_table"] = ["<table><tr><td>Example</td><td>2.5 mg</td></tr></table>"]
        row, _ = clean_record(record, MAPPING, 1)
        self.assertEqual(row["dosage_text"], "Example | 2.5 mg")
        self.assertIn("dosage_and_administration_table", json.loads(row["sections_json"]))


class LabelingTests(unittest.TestCase):
    def test_rejects_cross_class_combinations(self) -> None:
        row, reason = clean_record(source_record("combo", epc=[PPI, STEROID]), MAPPING, 1)
        self.assertIsNone(row)
        self.assertEqual(reason, "ambiguous_target_classes")

    def test_does_not_guess_class_from_document_text(self) -> None:
        record = source_record("unknown", epc=["Unmapped class [EPC]"])
        record["indications_and_usage"] = ["This passage mentions a proton pump inhibitor."]
        row, reason = clean_record(record, MAPPING, 1)
        self.assertIsNone(row)
        self.assertEqual(reason, "unmapped_pharm_class_epc")

    def test_class_metadata_is_separate_from_input_text(self) -> None:
        row = cleaned("metadata")
        self.assertNotIn(PPI, row["document_text"])
        self.assertIn(PPI, json.loads(row["pharm_class_epc"]))

    def test_source_query_excludes_animal_products(self) -> None:
        query = build_search([PPI])
        self.assertIn('openfda.product_type.exact:"HUMAN PRESCRIPTION DRUG"', query)
        self.assertIn('openfda.product_type.exact:"HUMAN OTC DRUG"', query)


class DatasetTests(unittest.TestCase):
    def test_latest_set_and_unique_text_are_retained_with_complete_audit(self) -> None:
        old = source_record("old")
        old["set_id"] = "shared-set"
        old["effective_time"] = "20200101"
        new = source_record("new")
        new["set_id"] = "shared-set"
        new["version"] = "2"
        duplicate = source_record("duplicate")
        duplicate["indications_and_usage"] = new["indications_and_usage"]
        missing = source_record("missing-epc", epc=[])
        combo = source_record("ambiguous", epc=[PPI, STEROID])
        steroid = source_record("steroid", epc=[STEROID], names=["Different ingredient"])
        records = [old, new, duplicate, missing, combo, steroid]
        with tempfile.TemporaryDirectory(prefix=".test-", dir=ROOT) as directory:
            root = Path(directory)
            raw = root / "raw.jsonl.gz"
            with gzip.open(raw, "wt", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")
            frame = clean_dataset(CONFIG, raw, root / "processed", root / "reports")
            self.assertEqual(set(frame["record_id"]), {"new", "steroid"})
            audit = json.loads((root / "reports" / "cleaning_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["raw_records"], audit["clean_records"] + audit["excluded_records"])
            self.assertEqual(audit["exclusion_reasons"]["older_version_of_set"], 1)
            self.assertEqual(audit["exclusion_reasons"]["duplicate_normalized_text"], 1)
            self.assertTrue(validate_dataset(frame, CONFIG)["all_passed"])

    def test_identical_text_with_conflicting_labels_is_excluded_entirely(self) -> None:
        first = source_record("conflict-acid")
        second = source_record("conflict-steroid", epc=[STEROID])
        second["indications_and_usage"] = first["indications_and_usage"]
        retained_acid = source_record("retained-acid")
        retained_steroid = source_record("retained-steroid", epc=[STEROID], names=["Different ingredient"])
        with tempfile.TemporaryDirectory(prefix=".test-", dir=ROOT) as directory:
            root = Path(directory)
            raw = root / "raw.jsonl.gz"
            with gzip.open(raw, "wt", encoding="utf-8") as handle:
                for record in (first, second, retained_acid, retained_steroid):
                    handle.write(json.dumps(record) + "\n")
            frame = clean_dataset(CONFIG, raw, root / "processed", root / "reports")
            self.assertEqual(set(frame["record_id"]), {"retained-acid", "retained-steroid"})
            audit = json.loads((root / "reports" / "cleaning_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["exclusion_reasons"]["identical_text_conflicting_classes"], 2)

    def test_csv_round_trip_across_multiple_reader_chunks(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".test-", dir=ROOT) as directory:
            root = Path(directory)
            raw = root / "raw.jsonl.gz"
            with gzip.open(raw, "wt", encoding="utf-8") as handle:
                for index in range(125):
                    record = source_record(f"row-{index}", epc=[PPI if index % 2 else STEROID])
                    handle.write(json.dumps(record) + "\n")
            frame = clean_dataset(CONFIG, raw, root / "processed", root / "reports")
            report = validate_exports(frame, CONFIG, root / "processed", root / "reports")
            self.assertTrue(report["all_passed"])
            self.assertEqual(report["record_count"], 125)

    def test_combinations_bridge_shared_ingredient_groups(self) -> None:
        frame = pd.DataFrame([
            cleaned("alpha", names=["Alpha"]), cleaned("combo", names=["Alpha", "Beta"]),
            cleaned("beta", names=["Beta"]), cleaned("gamma", names=["Gamma"]),
        ])
        grouped = add_leakage_groups(frame).set_index("record_id")
        self.assertEqual(grouped.loc["alpha", "leakage_group_id"], grouped.loc["combo", "leakage_group_id"])
        self.assertEqual(grouped.loc["beta", "leakage_group_id"], grouped.loc["combo", "leakage_group_id"])
        self.assertNotEqual(grouped.loc["gamma", "leakage_group_id"], grouped.loc["combo", "leakage_group_id"])
        shuffled = add_leakage_groups(frame.sample(frac=1, random_state=42)).set_index("record_id")
        pd.testing.assert_series_equal(grouped["leakage_group_id"].sort_index(), shuffled["leakage_group_id"].sort_index())

    def test_validation_detects_broken_shared_ingredient_group(self) -> None:
        frame = add_leakage_groups(pd.DataFrame([
            cleaned("one", names=["Alpha"]), cleaned("two", epc=[STEROID], names=["Alpha"]),
        ]))
        frame.loc[1, "leakage_group_id"] = "deliberately-broken-group"
        report = validate_dataset(frame, CONFIG)
        failed = {check["name"] for check in report["checks"] if not check["passed"]}
        self.assertIn("shared_ingredients_stay_grouped", failed)


class CollectionTests(unittest.TestCase):
    def test_transient_rate_limit_is_retried(self) -> None:
        client = OpenFDAClient(request_interval=0)
        limited = Mock(status_code=429, headers={"Retry-After": "0"})
        success = Mock(status_code=200, ok=True)
        success.json.return_value = {"results": [{"id": "one"}]}
        with patch.object(client.session, "get", side_effect=[limited, success]) as get:
            with patch("medical_doc_assistant.collection.time.sleep"):
                self.assertEqual(client.fetch({"limit": 1}), success.json.return_value)
        self.assertEqual(get.call_count, 2)
        client.close()

    def test_api_key_is_not_exposed_in_error_message(self) -> None:
        client = OpenFDAClient(api_key="not-a-real-key", request_interval=0)
        response = Mock(status_code=403, ok=False)
        with patch.object(client.session, "get", return_value=response):
            with self.assertRaises(RuntimeError) as error:
                client.fetch({"limit": 1})
        self.assertNotIn("not-a-real-key", str(error.exception))
        client.close()

    def test_empty_search_returns_empty_results(self) -> None:
        client = OpenFDAClient(request_interval=0)
        response = Mock(status_code=404)
        response.json.return_value = {"error": {"code": "NOT_FOUND"}}
        with patch.object(client.session, "get", return_value=response):
            self.assertEqual(client.fetch({"limit": 1})["results"], [])
        client.close()

    def test_verified_cache_deduplicates_ids_and_rejects_setting_changes(self) -> None:
        payload = {"meta": {"results": {"total": 1}, "last_updated": "2026-01-01"},
                   "results": [source_record("shared")]}
        with tempfile.TemporaryDirectory(prefix=".test-", dir=ROOT) as directory:
            root = Path(directory)
            raw, manifest_path = root / "raw.jsonl.gz", root / "manifest.json"
            with patch("medical_doc_assistant.collection.OpenFDAClient") as factory:
                factory.return_value.fetch.return_value = copy.deepcopy(payload)
                manifest = collect_data(CONFIG, raw, manifest_path)
                self.assertEqual(manifest["unique_raw_records"], 1)
                self.assertEqual(len(list(read_records(raw))), 1)
            with patch("medical_doc_assistant.collection.OpenFDAClient") as factory:
                self.assertEqual(collect_data(CONFIG, raw, manifest_path), manifest)
                factory.assert_not_called()
                with self.assertRaisesRegex(ValueError, "settings differ"):
                    collect_data(CONFIG, raw, manifest_path, per_class=2)
                with raw.open("ab") as handle:
                    handle.write(b"tampered")
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    collect_data(CONFIG, raw, manifest_path)


if __name__ == "__main__":
    unittest.main()
