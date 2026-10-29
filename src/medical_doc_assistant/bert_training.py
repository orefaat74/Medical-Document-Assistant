"""Fine-tune the pinned compact BERT and compare on the existing frozen split."""

import json
import logging
import time
from pathlib import Path
from typing import Any

from medical_doc_assistant.hf_assets import ensure_model_assets
import pandas as pd
import torch
from transformers import AutoTokenizer

from medical_doc_assistant.baseline_evaluation import confusion_table, per_class_metrics
from medical_doc_assistant.bert_model import BertDocumentDataset, load_bert_checkpoint, pretrained_document_bert, save_bert_checkpoint
from medical_doc_assistant.deep_learning import EarlyStopping, balanced_weights, configure_runtime, evaluate_model, load_frozen_inputs, make_loader, train_epoch
from medical_doc_assistant.splitting import config_digest
from medical_doc_assistant.storage import file_sha256, utc_now, write_json

LOGGER = logging.getLogger(__name__)


def run_bert_phase4(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    frame, baseline, split = load_frozen_inputs(root)
    neural_run_path = root / "reports" / "phase3" / "run_manifest.json"
    neural = json.loads(neural_run_path.read_text(encoding="utf-8"))
    prior_table = root / "reports" / "phase3" / "model_comparison.csv"
    if file_sha256(prior_table) != neural["output_checksums"][prior_table.name] or neural["split_sha256"] != split["split_sha256"]:
        raise ValueError("The recorded neural comparison does not match the frozen source experiment.")
    model_dir, report_dir = root / "models" / "phase4" / "bert", root / "reports" / "phase4" / "bert"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    run_path = report_dir / "run_manifest.json"
    digest = config_digest(config)
    if run_path.exists():
        previous = json.loads(run_path.read_text(encoding="utf-8"))
        if previous["config_sha256"] != digest or previous["split_sha256"] != split["split_sha256"]:
            raise ValueError("The completed BERT experiment uses different settings.")
        return previous
    source = ensure_model_assets(root, config, ["bert"])["bert"]
    settings, labels = config["bert"], baseline["class_labels"]
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, use_fast=True)
    tokenizer.save_pretrained(model_dir / "tokenizer")
    loader_config = {"random_state": config["random_state"], "training": {"num_workers": 0}}
    training = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    validation = frame.loc[frame["split"] == "validation"].reset_index(drop=True)
    train_data = BertDocumentDataset(training, tokenizer, settings, labels)
    val_data = BertDocumentDataset(validation, tokenizer, settings, labels)
    started_at = utc_now()
    provenance = {"dataset_sha256": split["dataset_sha256"], "split_sha256": split["split_sha256"],
                  "config_sha256": digest, "pretrained": config["models"]["bert"]}
    plan = [{"trial_id": f"bert_{index:02d}", "learning_rate": rate, "batch_size": settings["batch_size"], "dropout": settings["dropout"]}
            for index, rate in enumerate(settings["learning_rates"], 1)]
    plan_path = report_dir / "trial_plan.json"
    plan_payload = {"provenance": provenance, "trials": plan}
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan_payload:
        raise ValueError("The pre-training BERT plan differs from this configuration.")
    write_json(plan_path, plan_payload)
    results = []
    for trial in plan:
        result_path = report_dir / (trial["trial_id"] + ".json")
        checkpoint = model_dir / (trial["trial_id"] + ".pt")
        history_path = report_dir / (trial["trial_id"] + "_history.csv")
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result["provenance"] != provenance or file_sha256(checkpoint) != result["checkpoint_sha256"]:
                raise ValueError("A saved BERT trial does not match its provenance.")
            results.append(result)
            continue
        device = configure_runtime(config)
        model = pretrained_document_bert(source, labels, settings["dropout"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=trial["learning_rate"], weight_decay=settings["weight_decay"])
        weights = balanced_weights(train_data.targets, len(labels), device)
        fit_loader = make_loader(train_data, settings["batch_size"], True, loader_config)
        val_loader = make_loader(val_data, settings["evaluation_batch_size"], False, loader_config)
        stop = EarlyStopping(settings["patience"], settings["minimum_f1_improvement"])
        best, best_epoch, history = None, 0, []
        start = time.perf_counter()
        for epoch in range(1, settings["max_epochs"] + 1):
            trained = train_epoch(model, fit_loader, optimizer, weights, device, settings["gradient_clip_norm"])
            metrics, _ = evaluate_model(model, val_loader, validation, labels, weights, device)
            history.append({"epoch": epoch, "train_loss": trained["loss"], "train_accuracy": trained["accuracy"],
                            "validation_loss": metrics["loss"], "validation_macro_f1": metrics["macro_f1"],
                            "validation_accuracy": metrics["accuracy"]})
            if best is None or metrics["macro_f1"] > best["macro_f1"] + 1e-12 or (abs(metrics["macro_f1"] - best["macro_f1"]) <= 1e-12 and metrics["loss"] < best["loss"]):
                best, best_epoch = metrics, epoch
                save_bert_checkpoint(checkpoint, model, labels, settings, trial,
                                     {**provenance, "fitted_partitions": ["train"], "fitted_documents": len(training), "epoch": epoch})
            pd.DataFrame(history).to_csv(history_path, index=False)
            LOGGER.info("%s epoch %s: training loss %.4f; validation macro F1 %.4f", trial["trial_id"], epoch, trained["loss"], metrics["macro_f1"])
            if stop.update(metrics["macro_f1"]):
                break
        result = {**trial, "selection_partition": "validation", "validation_metrics": best, "best_epoch": best_epoch,
                  "epochs_run": len(history), "training_seconds": time.perf_counter() - start,
                  "checkpoint_path": str(checkpoint.relative_to(root)).replace("\\", "/"),
                  "checkpoint_sha256": file_sha256(checkpoint), "history_sha256": file_sha256(history_path), "provenance": provenance}
        write_json(result_path, result)
        results.append(result)
    chosen = max(results, key=lambda result: (result["validation_metrics"]["macro_f1"], -result["validation_metrics"]["loss"], -plan.index(next(row for row in plan if row["trial_id"] == result["trial_id"]))))
    selection = {"selected_at_utc": utc_now(), "selected_trial": chosen["trial_id"], "results": results,
                 "test_used_for_selection": False, "refit_epochs": chosen["best_epoch"]}
    write_json(report_dir / "model_selection.json", selection)
    pd.DataFrame([{key: value for key, value in result.items() if key in {"trial_id", "learning_rate", "batch_size", "dropout", "best_epoch", "epochs_run", "training_seconds"}} |
                  {"validation_macro_f1": result["validation_metrics"]["macro_f1"]} for result in results]).to_csv(report_dir / "trial_comparison.csv", index=False)
    selected_model = load_bert_checkpoint(root / chosen["checkpoint_path"])
    val_weights = balanced_weights(train_data.targets, len(labels), torch.device("cpu"))
    val_metrics, val_predictions = evaluate_model(selected_model["model"], make_loader(val_data, settings["evaluation_batch_size"], False, loader_config),
                                                 validation, labels, val_weights, torch.device("cpu"))
    val_predictions["model"] = "bert_mini"
    val_predictions.to_csv(report_dir / "predictions_validation.csv", index=False)
    development = frame.loc[frame["split"] != "test"].reset_index(drop=True)
    dev_data = BertDocumentDataset(development, tokenizer, settings, labels)
    device = configure_runtime(config)
    model = pretrained_document_bert(source, labels, settings["dropout"]).to(device)
    weights = balanced_weights(dev_data.targets, len(labels), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=chosen["learning_rate"], weight_decay=settings["weight_decay"])
    dev_loader = make_loader(dev_data, settings["batch_size"], True, loader_config)
    refit = []
    for epoch in range(1, chosen["best_epoch"] + 1):
        metrics = train_epoch(model, dev_loader, optimizer, weights, device, settings["gradient_clip_norm"])
        refit.append({"epoch": epoch, "train_loss": metrics["loss"], "train_accuracy": metrics["accuracy"]})
        LOGGER.info("BERT final refit epoch %s / %s: loss %.4f", epoch, chosen["best_epoch"], metrics["loss"])
    pd.DataFrame(refit).to_csv(report_dir / "refit_history.csv", index=False)
    checkpoint = model_dir / "model.pt"
    save_bert_checkpoint(checkpoint, model, labels, settings, chosen,
                         {**provenance, "fitted_partitions": ["train", "validation"], "fitted_documents": len(development), "epoch": chosen["best_epoch"]})
    loaded = load_bert_checkpoint(checkpoint, device)
    test_started = utc_now()
    test = frame.loc[frame["split"] == "test"].reset_index(drop=True)
    test_data = BertDocumentDataset(test, tokenizer, settings, labels)
    test_metrics, predictions = evaluate_model(loaded["model"], make_loader(test_data, settings["evaluation_batch_size"], False, loader_config),
                                              test, labels, weights, device)
    predictions["model"] = "bert_mini"
    predictions.to_csv(report_dir / "predictions_test.csv", index=False)
    predictions.loc[~predictions["correct"]].to_csv(report_dir / "test_errors.csv", index=False)
    confusion_table(predictions, labels).to_csv(report_dir / "confusion_matrix.csv")
    per_class_metrics(predictions, labels).to_csv(report_dir / "per_class_test_metrics.csv", index=False)
    pd.concat([dev_data.coverage_table(development), test_data.coverage_table(test)], ignore_index=True).to_csv(report_dir / "token_coverage.csv", index=False)
    comparison = pd.read_csv(prior_table)
    row = {"model": "bert_mini", "display_name": "Fine-tuned BERT-mini", "phase": 4, "validation_macro_f1": val_metrics["macro_f1"],
           "test_accuracy": test_metrics["accuracy"], "test_macro_precision": test_metrics["macro_precision"], "test_macro_recall": test_metrics["macro_recall"],
           "test_macro_f1": test_metrics["macro_f1"], "test_group_balanced_macro_f1": test_metrics["group_balanced_macro_f1"], "classical_benchmark": False}
    pd.concat([comparison, pd.DataFrame([row])], ignore_index=True).to_csv(report_dir / "model_comparison.csv", index=False)
    manifest = {"schema_version": 1, "started_at_utc": started_at, "test_evaluation_started_at_utc": test_started, "completed_at_utc": utc_now(),
                "config_sha256": digest, "config": config, "dataset_sha256": split["dataset_sha256"], "split_sha256": split["split_sha256"],
                "neural_manifest_sha256": file_sha256(neural_run_path), "class_labels": labels, "selection": selection,
                "validation_metrics": val_metrics, "test_metrics": test_metrics, "refit_epochs": chosen["best_epoch"], "fitted_documents": len(development),
                "checkpoint": {"path": "models/phase4/bert/model.pt", "sha256": file_sha256(checkpoint)},
                "pretrained_model": config["models"]["bert"], "encoder_layers": model.encoder.config.num_hidden_layers,
                "hidden_size": model.encoder.config.hidden_size, "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "environment": {"torch": str(torch.__version__), "device": str(device), "cpu_threads": torch.get_num_threads()},
                "output_checksums": {path.name: file_sha256(path) for path in report_dir.glob("*.csv")}}
    write_json(run_path, manifest)
    LOGGER.info("Fine-tuned BERT test accuracy %.4f; macro F1 %.4f", test_metrics["accuracy"], test_metrics["macro_f1"])
    return manifest
