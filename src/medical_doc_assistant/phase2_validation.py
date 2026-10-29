"""Verify frozen partitions, model selection, saved features, and reported metrics."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from medical_doc_assistant.baseline_evaluation import classification_metrics, confusion_table
from medical_doc_assistant.cleaning import ingredient_tokens
from medical_doc_assistant.rxnorm import canonical_ingredient_tokens
from medical_doc_assistant.splitting import config_digest, training_cv_indices
from medical_doc_assistant.storage import file_sha256, utc_now, write_json


def validate_splits(
    frame: pd.DataFrame, manifest: dict[str, Any], config: dict[str, Any],
    reference: dict[str, Any], pairs: pd.DataFrame,
) -> dict[str, Any]:
    checks = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    labels = set(manifest["class_labels"])
    check("all_documents_accounted_for", len(frame) == manifest["documents"] and not frame["record_id"].duplicated().any(),
          f"{len(frame):,} unique source records")
    check("three_partitions", set(frame["split"]) == {"train", "validation", "test"}, "Train, validation, and test are nonempty")
    check("benchmark_groups_disjoint", frame.groupby("benchmark_group_id")["split"].nunique().eq(1).all(),
          "No benchmark group crosses a partition")
    check("phase1_groups_preserved", frame.groupby("leakage_group_id")["benchmark_group_id"].nunique().eq(1).all(),
          "No approved Phase 1 ingredient component is fragmented")
    check("all_classes_in_each_partition", all(set(subset["class_label"]) == labels for _, subset in frame.groupby("split")),
          "Every held-out partition contains all configured targets")
    minimum_ok = True
    for name, requirements in manifest["minimum_groups_per_class"].items():
        subset = frame.loc[frame["split"] == name]
        for label, minimum in requirements.items():
            minimum_ok &= subset.loc[subset["class_label"] == label, "benchmark_group_id"].nunique() >= minimum
    check("minimum_independent_class_groups", minimum_ok, "Recorded per-class training/holdout constraints are satisfied")
    token_groups = {}
    ingredients_ok = True
    columns = ["substance_names", "generic_names", "unii_codes", "benchmark_group_id"]
    for row in frame[columns].to_dict("records"):
        for token in ingredient_tokens(row) | canonical_ingredient_tokens(row, reference):
            if token in token_groups and token_groups[token] != row["benchmark_group_id"]:
                ingredients_ok = False
            token_groups[token] = row["benchmark_group_id"]
    check("source_and_rxnorm_ingredients_grouped", ingredients_ok,
          "Shared source names/UNIIs and unambiguous RxNorm IN relationships remain in one component")
    group_by_record = frame.set_index("record_id")["benchmark_group_id"]
    near_ok = all(left in group_by_record and right in group_by_record and group_by_record[left] == group_by_record[right]
                  for left, right in pairs[["left_record_id", "right_record_id"]].itertuples(index=False, name=None))
    check("audited_near_duplicates_grouped", near_ok, f"{len(pairs)} qualifying cross-ingredient text pairs")
    training = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    check("training_fold_assignments", set(training["cv_fold"]) == set(range(config["cv_folds"])) and
          training.groupby("benchmark_group_id")["cv_fold"].nunique().eq(1).all(), "One validation fold per training group")
    check("holdout_rows_excluded_from_cv", frame.loc[frame["split"] != "train", "cv_fold"].eq(-1).all(),
          "Validation/test rows are outside hyperparameter CV")
    for fold, (fitting, validation) in enumerate(training_cv_indices(training, config["cv_folds"])):
        fit, held = training.iloc[fitting], training.iloc[validation]
        check(f"cv_fold_{fold}_disjoint", not set(fit["benchmark_group_id"]) & set(held["benchmark_group_id"]),
              f"{len(fit)} fit documents; {len(held)} validation documents")
        check(f"cv_fold_{fold}_all_classes", set(fit["class_label"]) == labels and set(held["class_label"]) == labels,
              "All target classes in the fit and validation subset")
    return {"created_at_utc": utc_now(), "all_passed": all(item["passed"] for item in checks), "checks": checks}


def validate_phase2(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    from medical_doc_assistant.baseline import MODEL_NAMES, load_baseline, select_baseline

    report_dir = root / "reports" / "phase2"
    manifest = json.loads((report_dir / "run_manifest.json").read_text(encoding="utf-8"))
    split_manifest = json.loads((report_dir / "split_manifest.json").read_text(encoding="utf-8"))
    dataset = root / "data" / "processed" / "medicine_labels.parquet"
    reference_path = root / "data" / "reference" / "rxnorm_ingredient_map.json"
    split_path = root / "data" / "splits" / "phase2_split.csv"
    frame = pd.read_parquet(dataset).merge(
        pd.read_csv(split_path)[["record_id", "benchmark_group_id", "split", "cv_fold"]],
        on="record_id", validate="one_to_one",
    )
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    report = validate_splits(frame, split_manifest, config, reference, pd.read_csv(report_dir / "near_duplicate_pairs.csv"))

    def check(name: str, passed: bool, detail: str) -> None:
        report["checks"].append({"name": name, "passed": bool(passed), "detail": detail})

    check("approved_input_checksum", file_sha256(dataset) == manifest["dataset_sha256"] == split_manifest["dataset_sha256"],
          "The benchmark uses the approved Phase 1 Parquet snapshot")
    check("frozen_split_checksum", file_sha256(split_path) == manifest["split_sha256"] == split_manifest["split_sha256"],
          "Evaluation record assignments match both manifests")
    check("rxnorm_reference_checksum", file_sha256(reference_path) == split_manifest["reference_sha256"], "Ingredient reference is frozen")
    check("near_duplicate_audit_checksum", file_sha256(report_dir / "near_duplicate_pairs.csv") == split_manifest["near_duplicate_pairs_sha256"],
          "Text-pair audit matches the split manifest")
    check("run_configuration_checksum", config_digest(config) == manifest["config_sha256"], "Recorded training parameters match the configuration")
    labels = manifest["class_labels"]
    selection = json.loads((report_dir / "model_selection.json").read_text(encoding="utf-8"))
    selection_rows = selection["validation_results"]
    check("baseline_selected_on_validation", not selection["test_used_for_selection"] and
          select_baseline(selection_rows) == manifest["selected_model"] == selection["selected_model"],
          "Validation macro F1, with the predefined CV/family tie-break")
    check("selection_snapshot_contains_no_test_metrics", all(not any(key.startswith("test_") for key in row) for row in selection_rows),
          "The saved selection snapshot predates all test metrics")
    check("selection_precedes_test_evaluation", datetime.fromisoformat(selection["selected_at_utc"]) <=
          datetime.fromisoformat(manifest["test_evaluation_started_at_utc"]), "Selection was persisted before test prediction")

    bundles = {model: load_baseline(root / metadata["path"]) for model, metadata in manifest["models"].items()}
    development = frame.loc[frame["split"] != "test"]
    reference_vectorizer = clone(next(iter(bundles.values()))["pipeline"].named_steps["tfidf"])
    reference_vectorizer.fit(development["document_text"])
    comparison = {row["model"]: row for row in manifest["comparison"]}
    for model, bundle in bundles.items():
        metadata = manifest["models"][model]
        pipeline = bundle["pipeline"]
        check(model + "_artifact_checksum", file_sha256(root / metadata["path"]) == metadata["sha256"], "Saved classifier matches its manifest")
        check(model + "_standalone_text_pipeline", isinstance(pipeline, Pipeline) and list(pipeline.named_steps) == ["tfidf", "classifier"] and
              pipeline.memory is None and bundle["feature_column"] == "document_text", "A complete TF-IDF + classifier pipeline with no training-cache dependency")
        check(model + "_training_provenance", bundle["dataset_sha256"] == manifest["dataset_sha256"] and
              bundle["split_sha256"] == manifest["split_sha256"] and bundle["config_sha256"] == manifest["config_sha256"] and
              bundle["fitted_partitions"] == ["train", "validation"] and bundle["fitted_documents"] == len(development),
              "Final refit uses train + validation only")
        vectorizer = pipeline.named_steps["tfidf"]
        check(model + "_train_validation_only_tfidf", vectorizer.vocabulary_ == reference_vectorizer.vocabulary_ and
              np.allclose(vectorizer.idf_, reference_vectorizer.idf_), "Vocabulary and IDF independently match a fit that excludes all test text")
        params = json.loads(comparison[model]["best_parameters"])
        check(model + "_frozen_parameters", all(pipeline.get_params()[key] == value for key, value in params.items()),
              "Final model keeps the training-CV-selected hyperparameters")
        check(model + "_class_coverage", set(pipeline.classes_) == set(labels), "Classifier supports all six recorded target labels")

    test_predictions = pd.read_csv(report_dir / "predictions_test.csv")
    validation_predictions = pd.read_csv(report_dir / "predictions_validation.csv")
    truth = frame.set_index("record_id")
    for partition, predictions in (("validation", validation_predictions), ("test", test_predictions)):
        expected_ids = set(frame.loc[frame["split"] == partition, "record_id"])
        for model in MODEL_NAMES:
            subset = predictions.loc[predictions["model"] == model]
            source_rows = truth.loc[subset["record_id"]]
            check(model + "_" + partition + "_prediction_coverage", len(subset) == len(expected_ids) and set(subset["record_id"]) == expected_ids and
                  subset["split"].eq(partition).all() and np.array_equal(subset["true_class"], source_rows["class_label"]) and
                  np.array_equal(subset["benchmark_group_id"], source_rows["benchmark_group_id"]), "Every partition record has the correct source truth and group")
            metrics = classification_metrics(subset["true_class"], subset["predicted_class"], labels, subset["benchmark_group_id"])
            check(model + "_" + partition + "_reported_metrics", all(np.isclose(comparison[model][partition + "_" + key], value, atol=1e-12)
                                                                      for key, value in metrics.items()), "All reported metrics recomputed from saved predictions")
            if partition == "test":
                matrix = pd.read_csv(report_dir / ("confusion_matrix_" + model + ".csv"), index_col=0)
                check(model + "_confusion_matrix", matrix.equals(confusion_table(subset, labels)),
                      "All matrix entries and class supports match saved test predictions")

    benchmark = load_baseline(root / manifest["benchmark_artifact"]["path"])
    check("benchmark_artifact_matches_selection", benchmark["model_name"] == manifest["selected_model"] and
          file_sha256(root / manifest["benchmark_artifact"]["path"]) == manifest["benchmark_artifact"]["sha256"],
          "The reusable benchmark is the pre-test-selected model")
    probe = development["document_text"].head(8)
    check("benchmark_alias_predictions", np.array_equal(benchmark["pipeline"].predict(probe),
          bundles[manifest["selected_model"]]["pipeline"].predict(probe)), "Benchmark alias and family artifact predict identically after loading")
    for filename, checksum in manifest["output_checksums"].items():
        check("output_checksum_" + filename, file_sha256(report_dir / filename) == checksum, "CSV output matches the completed run")
    report["all_passed"] = all(item["passed"] for item in report["checks"])
    report["selected_model"] = manifest["selected_model"]
    write_json(report_dir / "validation.json", report)
    if not report["all_passed"]:
        failed = [item["name"] for item in report["checks"] if not item["passed"]]
        raise ValueError("Phase 2 validation failed: " + ", ".join(failed))
    return report
