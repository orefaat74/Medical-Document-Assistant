"""Synthetic software fixtures for Phase 2 leakage and partitioning invariants."""

import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from medical_doc_assistant.rxnorm import RxNormClient
from medical_doc_assistant.splitting import build_benchmark_groups, create_partitions, training_cv_indices

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs" / "phase2.json").read_text(encoding="utf-8"))
REFERENCE = {
    "rxnorm_version": "test-only",
    "entries": {
        "ALPHA": {"status": "matched", "ingredient_rxcui": "1", "ingredient_name": "alpha"},
        "ALPHA SALT": {"status": "matched", "ingredient_rxcui": "1", "ingredient_name": "alpha"},
        "BETA": {"status": "matched", "ingredient_rxcui": "2", "ingredient_name": "beta"},
        "UNKNOWN": {"status": "not_found"},
    },
}


def audit_row(identifier: str, name: str, text: str, group: str | None = None) -> dict:
    return {"record_id": identifier, "class_label": "software_fixture", "substance_names": json.dumps([name]),
            "leakage_group_id": group or identifier, "document_text": text, "word_count": len(text.split())}


def partition_frame(groups_per_class: int = 6) -> pd.DataFrame:
    return pd.DataFrame([
        {"record_id": f"{label}-{group}-{copy}", "class_label": label, "document_hash": f"hash-{label}-{group}-{copy}",
         "leakage_group_id": f"source-{label}-{group}", "benchmark_group_id": f"benchmark-{label}-{group}"}
        for label in ("a", "b") for group in range(groups_per_class) for copy in range(2)
    ])


class GroupingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = copy.deepcopy(CONFIG)
        self.config["near_duplicate_audit"].update(ngram_size=2, hash_features=65536, chunk_size=2)

    def test_source_salt_names_share_rxnorm_ingredient_group(self) -> None:
        frame = pd.DataFrame([
            audit_row("one", "ALPHA", "green meadow flowers blossom beside the winding river in spring"),
            audit_row("two", "ALPHA SALT", "small factory engines rotate quickly while workers inspect metal parts"),
            audit_row("three", "BETA", "distant planets orbit bright stars beyond the vast northern sky"),
        ])
        identifiers, _, audit = build_benchmark_groups(frame, REFERENCE, self.config)
        self.assertEqual(identifiers[0], identifiers[1])
        self.assertNotEqual(identifiers[0], identifiers[2])
        self.assertEqual(audit["groups_after_rxnorm"], 2)

    def test_existing_phase1_groups_are_never_fragmented(self) -> None:
        frame = pd.DataFrame([
            audit_row("one", "ALPHA", "green meadow flowers blossom beside the winding river", "shared-source"),
            audit_row("two", "BETA", "distant planets orbit stars beyond the northern sky", "shared-source"),
        ])
        identifiers, _, _ = build_benchmark_groups(frame, REFERENCE, self.config)
        self.assertEqual(identifiers[0], identifiers[1])

    def test_near_identical_text_connects_otherwise_unrelated_groups(self) -> None:
        repeated = "the green meadow contains flowers trees streams birds fields paths hills ponds and quiet woodland trails"
        frame = pd.DataFrame([
            audit_row("one", "UNKNOWN", repeated), audit_row("two", "BETA", repeated),
            audit_row("three", "ALPHA", "engineers build digital processors circuits switches networks cables routers servers memory devices and power supplies"),
        ])
        identifiers, pairs, audit = build_benchmark_groups(frame, REFERENCE, self.config)
        self.assertEqual(identifiers[0], identifiers[1])
        self.assertNotEqual(identifiers[0], identifiers[2])
        self.assertEqual(audit["near_duplicate_pairs"], 1)
        self.assertEqual(set(pairs.iloc[0][["left_record_id", "right_record_id"]]), {"one", "two"})

    def test_group_identifiers_are_stable_under_input_reordering(self) -> None:
        frame = pd.DataFrame([
            audit_row("one", "ALPHA", "green meadow flowers blossom beside the winding river"),
            audit_row("two", "ALPHA SALT", "small engines rotate while workers inspect metal parts"),
            audit_row("three", "BETA", "distant planets orbit stars beyond the northern sky"),
        ])
        original, _, _ = build_benchmark_groups(frame, REFERENCE, self.config)
        shuffled = frame.sample(frac=1, random_state=17).reset_index(drop=True)
        reordered, _, _ = build_benchmark_groups(shuffled, REFERENCE, self.config)
        self.assertEqual(dict(zip(frame.record_id, original)), dict(zip(shuffled.record_id, reordered)))


class PartitionTests(unittest.TestCase):
    def test_whole_groups_and_all_classes_are_present_in_each_partition_and_cv_fold(self) -> None:
        frame = partition_frame()
        assignments, details = create_partitions(frame, CONFIG, ["a", "b"])
        self.assertTrue(assignments.groupby("benchmark_group_id")["split"].nunique().eq(1).all())
        self.assertEqual(set(assignments.record_id), set(frame.record_id))
        for name in ("train", "validation", "test"):
            self.assertEqual(set(assignments.loc[assignments.split == name, "class_label"]), {"a", "b"})
        training = assignments.loc[assignments.split == "train"].reset_index(drop=True)
        for fitting, validation in training_cv_indices(training, CONFIG["cv_folds"]):
            self.assertFalse(set(training.iloc[fitting].benchmark_group_id) & set(training.iloc[validation].benchmark_group_id))
            self.assertEqual(set(training.iloc[fitting].class_label), {"a", "b"})
            self.assertEqual(set(training.iloc[validation].class_label), {"a", "b"})
        self.assertEqual(details["minimum_groups_per_class"]["train"]["a"], 3)
        self.assertTrue(assignments.loc[assignments.split != "train", "cv_fold"].eq(-1).all())

    def test_group_coverage_failure_is_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "independent groups"):
            create_partitions(partition_frame(3), CONFIG, ["a", "b"])

    def test_cv_rejects_validation_or_test_rows(self) -> None:
        frame = partition_frame()
        frame["split"], frame["cv_fold"] = "test", -1
        with self.assertRaisesRegex(ValueError, "only the training"):
            training_cv_indices(frame, 3)


class RxNormMatchingTests(unittest.TestCase):
    def test_only_one_exact_normalized_name_and_one_ingredient_are_accepted(self) -> None:
        client = RxNormClient()
        with patch.object(client, "fetch", side_effect=[
            {"idGroup": {"rxnormId": ["100"]}},
            {"relatedGroup": {"conceptGroup": [{"tty": "IN", "conceptProperties": [{"rxcui": "1", "name": "alpha"}]}]}},
        ]) as fetch:
            result = client.resolve("ALPHA SALT", "test-only")
        self.assertEqual(result["ingredient_rxcui"], "1")
        self.assertEqual(fetch.call_args_list[0].args[2]["search"], 2)

    def test_ambiguous_name_is_not_guessed(self) -> None:
        client = RxNormClient()
        with patch.object(client, "fetch", return_value={"idGroup": {"rxnormId": ["100", "200"]}}) as fetch:
            result = client.resolve("UNKNOWN", "test-only")
        self.assertEqual(result["status"], "ambiguous_name")
        self.assertEqual(fetch.call_count, 1)
        self.assertNotIn("ingredient_rxcui", result)

    def test_missing_ingredient_stays_unmapped(self) -> None:
        client = RxNormClient()
        with patch.object(client, "fetch", side_effect=[
            {"idGroup": {"rxnormId": ["100"]}},
            {"relatedGroup": {"conceptGroup": [{"tty": "IN", "conceptProperties": None}]}},
        ]):
            result = client.resolve("UNKNOWN", "test-only")
        self.assertEqual(result["status"], "ambiguous_or_missing_ingredient")
        self.assertNotIn("ingredient_rxcui", result)


if __name__ == "__main__":
    unittest.main()
