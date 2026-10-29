"""Verify BERT training provenance and the recorded document-grounded QA evaluation."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from medical_doc_assistant.baseline_evaluation import classification_metrics, confusion_table
from medical_doc_assistant.bert_model import load_bert_checkpoint
from medical_doc_assistant.deep_learning import load_frozen_inputs
from medical_doc_assistant.document_index import DocumentIndex
from medical_doc_assistant.grounded_qa import MISSING_ANSWER, RAG_POLICY_VERSION, normalized_quote
from medical_doc_assistant.rag_evaluation import response_supported
from medical_doc_assistant.splitting import config_digest
from medical_doc_assistant.storage import file_sha256, utc_now, write_json


def validate_phase4(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    directory = root / "reports" / "phase4"
    bert_dir, rag_dir = directory / "bert", directory / "rag"
    manifest = json.loads((bert_dir / "run_manifest.json").read_text(encoding="utf-8"))
    frame, baseline, split = load_frozen_inputs(root)
    labels = baseline["class_labels"]
    checks = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("same_frozen_corpus_and_split", manifest["dataset_sha256"] == split["dataset_sha256"] and manifest["split_sha256"] == split["split_sha256"],
          "All five classifier families use the same source records and group-separated partitions")
    check("phase4_configuration_checksum", manifest["config_sha256"] == config_digest(config), "BERT/QA settings match the experiment")
    check("phase3_comparison_preserved", file_sha256(root / "reports" / "phase3" / "run_manifest.json") == manifest["neural_manifest_sha256"],
          "Earlier model results retain their recorded provenance")
    assets = json.loads((directory / "model_assets.json").read_text(encoding="utf-8"))
    for role, model in assets["models"].items():
        path = root / "models" / "phase4" / "pretrained" / role
        check(role + "_pinned_revision", model["revision"] == config["models"][role]["revision"], "Immutable public model revision")
        check(role + "_asset_hashes", all(file_sha256(path / name) == checksum for name, checksum in model["file_sha256"].items()), "Local model/tokenizer assets match their manifest")
    selected = manifest["selection"]
    winner = max(selected["results"], key=lambda row: (row["validation_metrics"]["macro_f1"], -row["validation_metrics"]["loss"]))
    check("bert_validation_selection", winner["trial_id"] == selected["selected_trial"] and not selected["test_used_for_selection"],
          "Learning rate and epoch count selected using validation only")
    check("bert_selection_before_test", datetime.fromisoformat(selected["selected_at_utc"]) <= datetime.fromisoformat(manifest["test_evaluation_started_at_utc"]),
          "The test inference follows selection and fixed-epoch refit")
    for result in selected["results"]:
        history = pd.read_csv(bert_dir / (result["trial_id"] + "_history.csv"))
        best = history.loc[np.isclose(history["validation_macro_f1"], history["validation_macro_f1"].max(), atol=1e-12, rtol=0)].sort_values("validation_loss").iloc[0]
        check(result["trial_id"] + "_best_epoch", int(best["epoch"]) == result["best_epoch"], "Best validation checkpoint matches the learning history")
        check(result["trial_id"] + "_checkpoint_hash", file_sha256(root / result["checkpoint_path"]) == result["checkpoint_sha256"], "Frozen best trial weights")
    refit = pd.read_csv(bert_dir / "refit_history.csv")
    check("bert_fixed_epoch_refit", len(refit) == selected["refit_epochs"] == manifest["refit_epochs"], "Final fit epoch count fixed before test")
    path = root / manifest["checkpoint"]["path"]
    bundle = load_bert_checkpoint(path)
    check("bert_final_checkpoint_hash", file_sha256(path) == manifest["checkpoint"]["sha256"], "Final model weights match the manifest")
    check("bert_real_encoder_architecture", bundle["bert_config"]["num_hidden_layers"] == 4 and bundle["bert_config"]["hidden_size"] == 256 and
          bundle["class_labels"] == labels, "Four-layer Google BERT-mini with all six target classes")
    check("bert_final_fit_provenance", bundle["provenance"]["fitted_partitions"] == ["train", "validation"] and
          bundle["provenance"]["fitted_documents"] == int(frame["split"].ne("test").sum()), "Final classifier excludes test fitting")
    check("bert_finite_state", all(torch.isfinite(value).all() for value in bundle["model"].state_dict().values()), "All saved trainable weights are finite")
    for partition in ("validation", "test"):
        predictions = pd.read_csv(bert_dir / ("predictions_" + partition + ".csv"))
        expected = frame.loc[frame["split"] == partition].set_index("record_id")
        source = expected.loc[predictions["record_id"]]
        check(partition + "_bert_source_coverage", len(predictions) == len(expected) and set(predictions["record_id"]) == set(expected.index) and
              np.array_equal(predictions["true_class"], source["class_label"]), "Every held-out record retains the correct source target")
        metrics = classification_metrics(predictions["true_class"], predictions["predicted_class"], labels, predictions["benchmark_group_id"])
        check(partition + "_bert_metrics", all(np.isclose(value, manifest[partition + "_metrics"][key], atol=1e-12) for key, value in metrics.items()), "Reported classifier metrics recomputed")
        if partition == "test":
            matrix = pd.read_csv(bert_dir / "confusion_matrix.csv", index_col=0)
            check("bert_test_confusion_matrix", matrix.equals(confusion_table(predictions, labels)), "All class supports/counts match predictions")
    coverage = pd.read_csv(bert_dir / "token_coverage.csv")
    limit = (config["bert"]["max_length"] - 2) * config["bert"]["max_windows"]
    check("bert_bounded_wordpiece_context", len(coverage) == len(frame) and coverage["encoded_tokens"].le(limit).all() and
          coverage["encoded_tokens"].le(coverage["token_count"]).all(), "Pretrained tokenizer and declared contiguous-window budget")
    for filename, checksum in manifest["output_checksums"].items():
        check("bert_output_" + filename, file_sha256(bert_dir / filename) == checksum, "Frozen classifier CSV output")

    cases_path = root / "data" / "evaluation" / "phase4_qa_cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    groups = {partition: {source["benchmark_group_id"] for source in cases["sources"] if source["partition"] == partition}
              for partition in ("development", "calibration", "evaluation")}
    check("qa_document_group_isolation", not groups["evaluation"] & (groups["calibration"] | groups["development"]), "QA evaluation uses different source ingredient components")
    source_map = {source["record_id"]: source for source in cases["sources"]}
    source_frame = frame.set_index("record_id")
    source_ok = all(file_sha256(root / source["file"]) == source["sha256"] and
                    (root / source["file"]).read_text(encoding="utf-8") == source_frame.loc[source["record_id"], "document_text"]
                    for source in cases["sources"])
    check("qa_real_source_documents", source_ok, "QA documents are exact cleaned openFDA texts")
    check("qa_answers_are_source_quotes", all(not case["answerable"] or normalized_quote(case["expected_answer"]) in
          normalized_quote((root / case["source_file"]).read_text(encoding="utf-8")) for case in cases["cases"]),
          "Positive reference answers contain no invented medical text")
    plan = json.loads((rag_dir / "evaluation_plan.json").read_text(encoding="utf-8"))
    check("qa_recorded_policy", plan["policy_version"] == RAG_POLICY_VERSION and plan["config_sha256"] == config_digest(config) and
          plan["qa_cases_sha256"] == file_sha256(cases_path), "Versioned source-grounding policy and QA corpus")
    check("qa_initial_evaluation_preserved", (rag_dir / "archive" / "policy_v1" / "evaluation_metrics.json").exists(), "Initial evaluation retained before the general frequency-answer guard correction")
    case_map = {case["case_id"]: case for case in cases["cases"]}
    for partition in ("calibration", "evaluation"):
        responses = json.loads((rag_dir / (partition + "_responses.json")).read_text(encoding="utf-8"))
        metrics = json.loads((rag_dir / (partition + "_metrics.json")).read_text(encoding="utf-8"))
        expected_cases = [case for case in cases["cases"] if case["partition"] == partition]
        check(partition + "_qa_case_coverage", {item["case_id"] for item in responses} == {case["case_id"] for case in expected_cases}, "All source-linked QA cases are evaluated")
        supported = True
        fallback = True
        indexes = {}
        for item in responses:
            response = item["response"]
            source = source_map[item["source_record_id"]]
            document = (root / source["file"]).read_text(encoding="utf-8")
            index = DocumentIndex.load(root / "data" / "indexes" / response["document_id"])
            indexes[response["document_id"]] = index
            supported &= index.document == document or normalized_quote(index.document) == normalized_quote(document)
            supported &= response_supported(response, index.document)
            supported &= "generated_candidate" not in response
            if response["status"] == "not_found":
                fallback &= response["answer"] == MISSING_ANSWER and not response["citations"]
        check(partition + "_qa_source_grounding", supported, "Every returned answer/citation is an exact source quotation; rejected generation is not exposed")
        check(partition + "_qa_missing_fallback", fallback, "Missing responses use the required phrase with empty citations")
        check(partition + "_qa_vector_indexes", all(index.index.ntotal == len(index.chunks) and index.vectors.shape[1] == 384 and
              np.allclose(np.linalg.norm(index.vectors, axis=1), 1, atol=1e-4) for index in indexes.values()), "Normalized MiniLM embeddings in independent FAISS indexes")
        table = pd.read_csv(rag_dir / (partition + "_results.csv"))
        check(partition + "_qa_metric_consistency", len(table) == metrics["cases"] and np.isclose(table["answerability_correct"].mean(), metrics["answerability_accuracy"]), "Reported QA counts/answerability match saved case rows")
    report = {"created_at_utc": utc_now(), "checks": checks, "all_passed": all(item["passed"] for item in checks)}
    write_json(directory / "validation.json", report)
    if not report["all_passed"]:
        raise ValueError("Phase 4 validation failed: " + ", ".join(item["name"] for item in checks if not item["passed"]))
    return report
