"""Verify neural data isolation, early stopping, checkpoints, and reported results."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from medical_doc_assistant.baseline_evaluation import classification_metrics, confusion_table
from medical_doc_assistant.deep_learning import balanced_weights, load_frozen_inputs, select_trial
from medical_doc_assistant.neural_model import load_checkpoint
from medical_doc_assistant.neural_text import MedicineTextDataset, fit_vocabulary
from medical_doc_assistant.phase2_validation import validate_splits
from medical_doc_assistant.splitting import config_digest
from medical_doc_assistant.storage import file_sha256, utc_now, write_json


def validate_phase3(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    report_dir = root / "reports" / "phase3"
    manifest = json.loads((report_dir / "run_manifest.json").read_text(encoding="utf-8"))
    frame, baseline, split = load_frozen_inputs(root)
    labels = baseline["class_labels"]
    reference = json.loads((root / "data" / "reference" / "rxnorm_ingredient_map.json").read_text(encoding="utf-8"))
    report = validate_splits(frame, split, baseline["config"], reference,
                             pd.read_csv(root / "reports" / "phase2" / "near_duplicate_pairs.csv"))

    def check(name: str, passed: bool, detail: str) -> None:
        report["checks"].append({"name": name, "passed": bool(passed), "detail": detail})

    check("phase3_config_checksum", config_digest(config) == manifest["config_sha256"], "Recorded neural settings match the configuration")
    check("same_dataset_and_split_as_baselines", manifest["dataset_sha256"] == baseline["dataset_sha256"] and
          manifest["split_sha256"] == baseline["split_sha256"], "Identical source documents and record-level evaluation assignments")
    check("baseline_manifest_preserved", file_sha256(root / "reports" / "phase2" / "run_manifest.json") == manifest["baseline_manifest_sha256"],
          "Classical comparison is the recorded Phase 2 experiment")
    training = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    development = frame.loc[frame["split"] != "test"].reset_index(drop=True)
    tuning_vocabulary = json.loads((report_dir / "tuning_vocabulary.json").read_text(encoding="utf-8"))
    refit_vocabulary = json.loads((report_dir / "refit_vocabulary.json").read_text(encoding="utf-8"))
    expected_training = fit_vocabulary(training, config["vocabulary"], {"train"}).to_dict()
    expected_development = fit_vocabulary(development, config["vocabulary"], {"train", "validation"}).to_dict()
    check("tuning_vocabulary_training_only", tuning_vocabulary == expected_training, "Recomputed word counts/indices using training text only")
    check("refit_vocabulary_excludes_test", refit_vocabulary == expected_development, "Recomputed final vocabulary using train + validation only")
    selection = json.loads((report_dir / "model_selection.json").read_text(encoding="utf-8"))
    chosen = select_trial(selection["results"])
    check("selection_validation_only", not selection["test_used_for_selection"] and chosen["trial_id"] == manifest["selected_trial"],
          "Validation macro F1, lower validation loss, then fixed trial order")
    check("selection_precedes_test", datetime.fromisoformat(selection["selected_at_utc"]) <=
          datetime.fromisoformat(manifest["test_evaluation_started_at_utc"]), "Hyperparameters/epoch count persisted before test evaluation")
    check("selection_has_no_test_metrics", all(not any(key.startswith("test_") for key in result) for result in selection["results"]),
          "Tuning records contain only validation selection results")
    plan = json.loads((report_dir / "trial_plan.json").read_text(encoding="utf-8"))
    planned = {trial["trial_id"]: trial for trial in plan["trials"]}
    check("all_planned_trials_completed", set(planned) == {result["trial_id"] for result in selection["results"]},
          f"{len(planned)} fixed learning-rate/batch-size/dropout combinations")
    for result in selection["results"]:
        name = result["trial_id"]
        history = pd.read_csv(root / result["history_path"])
        check(name + "_plan_parameters", result["parameters"] == planned[name]["parameters"], "Candidate matches the pre-training grid")
        check(name + "_checkpoint_history_hashes", file_sha256(root / result["checkpoint_path"]) == result["checkpoint_sha256"] and
              file_sha256(root / result["history_path"]) == result["history_sha256"], "Best checkpoint and full learning history are frozen")
        peak = history["validation_macro_f1"].max()
        best = history.loc[np.isclose(history["validation_macro_f1"], peak, atol=1e-12, rtol=0)].sort_values("validation_loss").iloc[0]
        check(name + "_best_epoch", int(best["epoch"]) == result["best_epoch"] and
              np.isclose(best["validation_macro_f1"], result["validation_metrics"]["macro_f1"]), "Checkpoint corresponds to the best validation epoch")
        check(name + "_bounded_training", len(history) == result["epochs_run"] and result["epochs_run"] <= config["training"]["max_epochs"],
              "Recorded epochs respect the configured limit")
    refit_history = pd.read_csv(report_dir / "refit_history.csv")
    check("fixed_selected_epoch_refit", len(refit_history) == manifest["refit_epochs"] == selection["refit_epochs"] == chosen["best_epoch"] and
          not any(column.startswith("validation_") or column.startswith("test_") for column in refit_history),
          "Final train/validation refit uses a preselected epoch count without holdout monitoring")
    path = root / manifest["checkpoint"]["path"]
    bundle = load_checkpoint(path)
    model, vocabulary = bundle["model"], bundle["vocabulary_object"]
    check("final_checkpoint_checksum", file_sha256(path) == manifest["checkpoint"]["sha256"], "Standalone neural state/vocabulary checkpoint matches the manifest")
    check("final_checkpoint_provenance", bundle["provenance"]["dataset_sha256"] == manifest["dataset_sha256"] and
          bundle["provenance"]["split_sha256"] == manifest["split_sha256"] and
          bundle["provenance"]["fitted_partitions"] == ["train", "validation"] and
          bundle["provenance"]["fitted_documents"] == len(development), "Same benchmark data; final fit excludes test labels/text")
    check("checkpoint_vocabulary_matches_refit", vocabulary.to_dict() == refit_vocabulary, "Saved token indices match the final vocabulary file")
    check("classifier_and_embedding_dimensions", model.classifier.out_features == len(labels) and
          model.embedding.num_embeddings == len(vocabulary.tokens) and bundle["class_labels"] == labels,
          "Vocabulary rows and label indices match the model architecture")
    check("finite_state_and_zero_padding_row", all(torch.isfinite(value).all() for value in model.state_dict().values()) and
          model.embedding.weight[0].eq(0).all(), "Finite learned tensors with PAD embedding fixed at zero")
    probe_data = MedicineTextDataset.from_frame(development.head(2), vocabulary, config["encoding"], labels)
    probe = probe_data.input_ids
    expanded = torch.cat([probe, torch.zeros_like(probe[:, :2])], dim=1)
    with torch.inference_mode():
        logits, padded_logits = model(probe), model(expanded)
    check("checkpoint_padding_invariance", torch.isfinite(logits).all() and torch.allclose(logits, padded_logits, atol=1e-5, rtol=1e-5),
          "Adding all-padding chunks does not change inference logits")
    coverage = pd.read_csv(report_dir / "token_coverage.csv")
    budget = config["encoding"]["chunk_size"] * config["encoding"]["max_chunks"]
    check("encoding_coverage_records", len(coverage) == len(frame) and set(coverage["record_id"]) == set(frame["record_id"]),
          "Text-coverage diagnostics account for every corpus document")
    check("bounded_contiguous_encoding", (coverage["encoded_tokens"] <= budget).all() and
          (coverage["encoded_tokens"] <= coverage["token_count"]).all() and
          coverage["retention_ratio"].between(0, 1).all() and coverage["unknown_ratio"].between(0, 1).all(),
          "Sampled tokens fit the documented context budget; coverage/unknown ratios are valid")
    truth = frame.set_index("record_id")
    for partition, weights_source in (("validation", training), ("test", development)):
        predictions = pd.read_csv(report_dir / ("predictions_" + partition + ".csv"))
        expected = frame.loc[frame["split"] == partition]
        ordered = truth.loc[predictions["record_id"]]
        check(partition + "_source_record_coverage", len(predictions) == len(expected) and set(predictions["record_id"]) == set(expected["record_id"]) and
              np.array_equal(predictions["true_class"], ordered["class_label"]), "Each source holdout record has the recorded target label")
        scores = predictions[["logit__" + label for label in labels]].to_numpy(dtype=np.float32)
        check(partition + "_logit_prediction_consistency", np.isfinite(scores).all() and
              np.array_equal(np.asarray(labels)[scores.argmax(axis=1)], predictions["predicted_class"]),
              "Saved finite logits reproduce the class predictions")
        metrics = classification_metrics(predictions["true_class"], predictions["predicted_class"], labels, predictions["benchmark_group_id"])
        recorded = manifest[partition + "_metrics"]
        check(partition + "_metrics_recomputed", all(np.isclose(recorded[key], value, atol=1e-12) for key, value in metrics.items()),
              "All document and group-balanced metrics match saved predictions")
        lookup = {label: index for index, label in enumerate(labels)}
        fit_targets = torch.tensor([lookup[label] for label in weights_source["class_label"]])
        targets = torch.tensor([lookup[label] for label in predictions["true_class"]])
        weights = balanced_weights(fit_targets, len(labels), torch.device("cpu"))
        loss = F.cross_entropy(torch.from_numpy(scores), targets, weight=weights, reduction="sum") / weights[targets].sum()
        check(partition + "_weighted_loss_recomputed", np.isclose(float(loss), recorded["loss"], atol=1e-5), "Class-weighted CE recomputed with fit-partition weights")
        if partition == "test":
            matrix = pd.read_csv(report_dir / "confusion_matrix.csv", index_col=0)
            check("test_confusion_matrix", matrix.equals(confusion_table(predictions, labels)), "Counts/supports match test predictions")
    comparison = pd.read_csv(report_dir / "model_comparison.csv")
    copied_ok = True
    for row in baseline["comparison"]:
        copied = comparison.loc[comparison["model"] == row["model"]].iloc[0]
        for field in ("validation_macro_f1", "test_accuracy", "test_macro_precision", "test_macro_recall", "test_macro_f1", "test_group_balanced_macro_f1"):
            copied_ok &= np.isclose(copied[field], row[field], atol=1e-12)
    check("classical_comparison_preserved", copied_ok, "All three classical scores are copied from the frozen Phase 2 run")
    for filename, checksum in manifest["output_checksums"].items():
        check("output_checksum_" + filename, file_sha256(report_dir / filename) == checksum, "Recorded experiment CSV is unchanged")
    report["all_passed"] = all(item["passed"] for item in report["checks"])
    report["selected_trial"] = manifest["selected_trial"]
    report["created_at_utc"] = utc_now()
    write_json(report_dir / "validation.json", report)
    if not report["all_passed"]:
        raise ValueError("Phase 3 validation failed: " + ", ".join(item["name"] for item in report["checks"] if not item["passed"]))
    return report
