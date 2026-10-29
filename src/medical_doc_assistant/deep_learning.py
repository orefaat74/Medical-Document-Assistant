"""Train, tune, refit, and evaluate the Phase 3 PyTorch classifier."""

import importlib.metadata
import itertools
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch.nn import functional as F
from torch.utils.data import DataLoader

from medical_doc_assistant.baseline_evaluation import classification_metrics, confusion_table, per_class_metrics
from medical_doc_assistant.neural_model import TextCNN, build_text_cnn, load_checkpoint, save_checkpoint
from medical_doc_assistant.neural_text import MedicineTextDataset, Vocabulary, fit_vocabulary
from medical_doc_assistant.splitting import config_digest, prepare_phase2
from medical_doc_assistant.storage import file_sha256, utc_now, write_json

LOGGER = logging.getLogger(__name__)


def load_phase3_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config["model"]["architecture"] != "text_cnn":
        raise ValueError("Unsupported Phase 3 configuration or architecture.")
    if config["device"] not in {"cpu", "cuda", "auto"}:
        raise ValueError("Choose cpu, cuda, or auto as the training device.")
    if config["encoding"]["chunk_size"] < max(config["model"]["kernel_sizes"]) or config["encoding"]["max_chunks"] < 1:
        raise ValueError("Invalid convolution/window dimensions.")
    if config["training"]["max_epochs"] < 1 or config["training"]["patience"] < 1:
        raise ValueError("Epoch limit and patience must be positive.")
    grid = config["parameter_grid"]
    if set(grid) != {"learning_rate", "batch_size", "dropout"} or any(not values for values in grid.values()):
        raise ValueError("Define learning-rate, batch-size, and dropout candidates.")
    if any(rate <= 0 for rate in grid["learning_rate"]) or any(size < 1 for size in grid["batch_size"]) or any(not 0 <= dropout < 1 for dropout in grid["dropout"]):
        raise ValueError("Invalid hyperparameter candidates.")
    return config


def configure_runtime(config: dict[str, Any]) -> torch.device:
    seed = config["random_state"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config["cpu_threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this PyTorch installation; use cpu or install a compatible CUDA wheel.")
    return torch.device(device)


def make_loader(dataset: MedicineTextDataset, batch_size: int, shuffle: bool, config: dict[str, Any]) -> DataLoader:
    generator = torch.Generator().manual_seed(config["random_state"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
                      num_workers=config["training"]["num_workers"], drop_last=False)


def balanced_weights(targets: torch.Tensor, classes: int, device: torch.device) -> torch.Tensor:
    counts = torch.bincount(targets, minlength=classes).to(torch.float32)
    if torch.any(counts == 0):
        raise ValueError("The fit partition must contain every target class.")
    return (len(targets) / (classes * counts)).to(device)


def train_epoch(model: TextCNN, loader: DataLoader, optimizer: torch.optim.Optimizer, weights: torch.Tensor,
                device: torch.device, gradient_clip: float) -> dict[str, float]:
    model.train()
    numerator = denominator = 0.0
    correct = total = 0
    for batch in loader:
        inputs, target = batch["input_ids"].to(device), batch["target"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss_sum = F.cross_entropy(logits, target, weight=weights, reduction="sum")
        weight_sum = weights[target].sum()
        loss = loss_sum / weight_sum
        if not torch.isfinite(loss):
            raise ValueError("Neural training produced a nonfinite loss.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        numerator += float(loss_sum.detach())
        denominator += float(weight_sum.detach())
        correct += int(logits.detach().argmax(dim=1).eq(target).sum())
        total += len(target)
    return {"loss": numerator / denominator, "accuracy": correct / total}


def predict_loader(model: TextCNN, loader: DataLoader, device: torch.device, *, features: bool = False) -> np.ndarray:
    model.eval()
    outputs, indices = [], []
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["input_ids"].to(device)
            values = model.forward_features(inputs) if features else model(inputs)
            outputs.append(values.cpu().numpy())
            indices.append(batch["index"].numpy())
    index = np.concatenate(indices)
    if len(set(index.tolist())) != len(loader.dataset):
        raise ValueError("Inference did not cover each source dataset sample exactly once.")
    return np.concatenate(outputs)[np.argsort(index)]


def evaluate_model(model: TextCNN, loader: DataLoader, frame: pd.DataFrame, labels: list[str],
                   weights: torch.Tensor, device: torch.device) -> tuple[dict[str, float], pd.DataFrame]:
    logits = predict_loader(model, loader, device)
    targets = loader.dataset.targets.to(device)
    loss = F.cross_entropy(torch.from_numpy(logits).to(device), targets, weight=weights, reduction="sum") / weights[targets].sum()
    predicted = np.asarray(labels)[logits.argmax(axis=1)]
    metrics = classification_metrics(frame["class_label"], predicted, labels, frame["benchmark_group_id"])
    metrics["loss"] = float(loss)
    columns = ["record_id", "class_label", "benchmark_group_id", "split", "generic_names", "source_url", "word_count"]
    predictions = frame[columns].reset_index(drop=True).rename(columns={"class_label": "true_class"})
    predictions.insert(0, "model", "text_cnn")
    predictions["predicted_class"] = predicted
    predictions["correct"] = predictions["true_class"].eq(predictions["predicted_class"])
    for index, label in enumerate(labels):
        predictions["logit__" + label] = logits[:, index]
    return metrics, predictions


class EarlyStopping:
    def __init__(self, patience: int, minimum_improvement: float):
        self.patience, self.minimum_improvement = patience, minimum_improvement
        self.meaningful_best = -float("inf")
        self.nonimproving_epochs = 0

    def update(self, score: float) -> bool:
        if score > self.meaningful_best + self.minimum_improvement:
            self.meaningful_best, self.nonimproving_epochs = score, 0
        else:
            self.nonimproving_epochs += 1
        return self.nonimproving_epochs >= self.patience


def select_trial(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results or any(result["selection_partition"] != "validation" for result in results):
        raise ValueError("Select neural hyperparameters using validation results only.")
    return max(results, key=lambda result: (result["validation_metrics"]["macro_f1"],
                                           -result["validation_metrics"]["loss"], -result["trial_index"]))


def load_frozen_inputs(root: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    baseline = json.loads((root / "reports" / "phase2" / "run_manifest.json").read_text(encoding="utf-8"))
    comparison = root / "reports" / "phase2" / "model_comparison.csv"
    if file_sha256(comparison) != baseline["output_checksums"][comparison.name]:
        raise ValueError("The recorded classical comparison differs from its manifest.")
    frame, split = prepare_phase2(root, baseline["config"], baseline["class_labels"])
    if split["dataset_sha256"] != baseline["dataset_sha256"] or split["split_sha256"] != baseline["split_sha256"]:
        raise ValueError("Phase 3 must use the frozen Phase 2 corpus and record assignments.")
    return frame, baseline, split


def run_trial(
    root: Path, trial: dict[str, Any], config: dict[str, Any], vocabulary: Vocabulary,
    training: pd.DataFrame, validation: pd.DataFrame, training_data: MedicineTextDataset,
    validation_data: MedicineTextDataset, labels: list[str], provenance: dict[str, Any],
) -> dict[str, Any]:
    if not training["split"].eq("train").all() or not validation["split"].eq("validation").all():
        raise ValueError("Tuning requires the isolated training and validation partitions.")
    report_dir, model_dir = root / "reports" / "phase3", root / "models" / "phase3" / "trials"
    trial_path = report_dir / (trial["trial_id"] + ".json")
    history_path = report_dir / (trial["trial_id"] + "_history.csv")
    checkpoint = model_dir / (trial["trial_id"] + ".pt")
    if trial_path.exists():
        previous = json.loads(trial_path.read_text(encoding="utf-8"))
        if previous["provenance"] != provenance or previous["parameters"] != trial["parameters"]:
            raise ValueError("A completed neural trial uses different data or parameters.")
        if file_sha256(checkpoint) != previous["checkpoint_sha256"] or file_sha256(history_path) != previous["history_sha256"]:
            raise ValueError("A completed trial checkpoint/history checksum does not match.")
        LOGGER.info("Using completed neural trial %s", trial["trial_id"])
        return previous
    device = configure_runtime(config)
    model = build_text_cnn(vocabulary, labels, config["model"], trial["parameters"]["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=trial["parameters"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    weights = balanced_weights(training_data.targets, len(labels), device)
    fit_loader = make_loader(training_data, trial["parameters"]["batch_size"], True, config)
    held_loader = make_loader(validation_data, config["training"]["evaluation_batch_size"], False, config)
    stopper = EarlyStopping(config["training"]["patience"], config["training"]["minimum_f1_improvement"])
    best_metrics, best_epoch, history = None, 0, []
    started = time.perf_counter()
    for epoch in range(1, config["training"]["max_epochs"] + 1):
        begin = time.perf_counter()
        trained = train_epoch(model, fit_loader, optimizer, weights, device, config["training"]["gradient_clip_norm"])
        metrics, _ = evaluate_model(model, held_loader, validation, labels, weights, device)
        history.append({"trial_id": trial["trial_id"], "epoch": epoch, "train_loss": trained["loss"],
                        "train_accuracy": trained["accuracy"], "validation_loss": metrics["loss"],
                        "validation_accuracy": metrics["accuracy"], "validation_macro_f1": metrics["macro_f1"],
                        "validation_group_balanced_macro_f1": metrics["group_balanced_macro_f1"],
                        "epoch_seconds": time.perf_counter() - begin})
        improved = best_metrics is None or metrics["macro_f1"] > best_metrics["macro_f1"] + 1e-12 or (
            abs(metrics["macro_f1"] - best_metrics["macro_f1"]) <= 1e-12 and metrics["loss"] < best_metrics["loss"])
        if improved:
            best_metrics, best_epoch = metrics, epoch
            save_checkpoint(checkpoint, model, vocabulary, labels, config, trial["parameters"],
                            {**provenance, "fitted_partitions": ["train"], "vocabulary_fitted_partitions": ["train"],
                             "fitted_documents": len(training), "epoch": epoch})
        pd.DataFrame(history).to_csv(history_path, index=False)
        LOGGER.info("%s epoch %s: train loss %.4f; validation macro F1 %.4f", trial["trial_id"], epoch, trained["loss"], metrics["macro_f1"])
        if stopper.update(metrics["macro_f1"]):
            break
    result = {**trial, "selection_partition": "validation", "best_epoch": best_epoch,
              "epochs_run": len(history), "stopped_early": len(history) < config["training"]["max_epochs"],
              "validation_metrics": best_metrics, "training_seconds": time.perf_counter() - started,
              "checkpoint_path": str(checkpoint.relative_to(root)).replace("\\", "/"),
              "history_path": str(history_path.relative_to(root)).replace("\\", "/"),
              "checkpoint_sha256": file_sha256(checkpoint), "history_sha256": file_sha256(history_path),
              "provenance": provenance}
    write_json(trial_path, result)
    return result


def run_phase3(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    frame, baseline, split = load_frozen_inputs(root)
    labels = baseline["class_labels"]
    report_dir, model_dir = root / "reports" / "phase3", root / "models" / "phase3"
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    run_path = report_dir / "run_manifest.json"
    digest = config_digest(config)
    if run_path.exists():
        previous = json.loads(run_path.read_text(encoding="utf-8"))
        if previous["config_sha256"] != digest or previous["split_sha256"] != split["split_sha256"]:
            raise ValueError("A completed Phase 3 experiment uses different settings; preserve it before starting another experiment.")
        LOGGER.info("Using the completed Phase 3 TextCNN experiment")
        return previous
    started_at = utc_now()
    device = configure_runtime(config)
    training = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    validation = frame.loc[frame["split"] == "validation"].reset_index(drop=True)
    vocabulary = fit_vocabulary(training, config["vocabulary"], {"train"})
    write_json(report_dir / "tuning_vocabulary.json", vocabulary.to_dict())
    training_data = MedicineTextDataset.from_frame(training, vocabulary, config["encoding"], labels)
    validation_data = MedicineTextDataset.from_frame(validation, vocabulary, config["encoding"], labels)
    pd.concat([training_data.coverage_table(training), validation_data.coverage_table(validation)]).to_csv(
        report_dir / "tuning_token_coverage.csv", index=False)
    provenance = {"dataset_sha256": split["dataset_sha256"], "split_sha256": split["split_sha256"],
                  "config_sha256": digest, "tuning_vocabulary_sha256": config_digest(vocabulary.to_dict())}
    grid = config["parameter_grid"]
    trials = [{"trial_id": f"trial_{index:02d}", "trial_index": index,
               "parameters": {"learning_rate": rate, "batch_size": size, "dropout": dropout}}
              for index, (rate, size, dropout) in enumerate(itertools.product(
                  grid["learning_rate"], grid["batch_size"], grid["dropout"]), 1)]
    plan_path = report_dir / "trial_plan.json"
    plan = {"schema_version": 1, "config_sha256": digest, "provenance": provenance, "trials": trials}
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise ValueError("The recorded pre-training trial plan differs from this experiment.")
    write_json(plan_path, plan)
    results = []
    for trial in trials:
        LOGGER.info("Training %s: %s", trial["trial_id"], trial["parameters"])
        results.append(run_trial(root, trial, config, vocabulary, training, validation, training_data,
                                 validation_data, labels, provenance))
    selected = select_trial(results)
    selection = {"selected_at_utc": utc_now(), "selected_trial": selected["trial_id"],
                 "selection_metric": "validation macro F1", "tie_break": "lower validation loss, then trial order",
                 "test_used_for_selection": False, "results": results, "refit_epochs": selected["best_epoch"]}
    write_json(report_dir / "model_selection.json", selection)
    pd.DataFrame([{**result["parameters"], "trial_id": result["trial_id"], "best_epoch": result["best_epoch"],
                   "epochs_run": result["epochs_run"], "stopped_early": result["stopped_early"],
                   "validation_macro_f1": result["validation_metrics"]["macro_f1"],
                   "validation_accuracy": result["validation_metrics"]["accuracy"],
                   "validation_loss": result["validation_metrics"]["loss"], "training_seconds": result["training_seconds"]}
                  for result in results]).to_csv(report_dir / "trial_comparison.csv", index=False)
    selected_bundle = load_checkpoint(root / selected["checkpoint_path"], device)
    val_weights = balanced_weights(training_data.targets, len(labels), device)
    val_metrics, val_predictions = evaluate_model(selected_bundle["model"], make_loader(validation_data,
        config["training"]["evaluation_batch_size"], False, config), validation, labels, val_weights, device)
    val_predictions.to_csv(report_dir / "predictions_validation.csv", index=False)
    LOGGER.info("Selected %s before test evaluation; refitting for %s fixed epochs", selected["trial_id"], selected["best_epoch"])
    development = frame.loc[frame["split"] != "test"].reset_index(drop=True)
    refit_vocabulary = fit_vocabulary(development, config["vocabulary"], {"train", "validation"})
    write_json(report_dir / "refit_vocabulary.json", refit_vocabulary.to_dict())
    development_data = MedicineTextDataset.from_frame(development, refit_vocabulary, config["encoding"], labels)
    device = configure_runtime(config)
    model = build_text_cnn(refit_vocabulary, labels, config["model"], selected["parameters"]["dropout"]).to(device)
    weights = balanced_weights(development_data.targets, len(labels), device)
    loader = make_loader(development_data, selected["parameters"]["batch_size"], True, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=selected["parameters"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    refit_history = []
    for epoch in range(1, selected["best_epoch"] + 1):
        metrics = train_epoch(model, loader, optimizer, weights, device, config["training"]["gradient_clip_norm"])
        refit_history.append({"epoch": epoch, "train_loss": metrics["loss"], "train_accuracy": metrics["accuracy"]})
        LOGGER.info("Final refit epoch %s / %s: loss %.4f", epoch, selected["best_epoch"], metrics["loss"])
    pd.DataFrame(refit_history).to_csv(report_dir / "refit_history.csv", index=False)
    checkpoint = model_dir / "text_cnn.pt"
    save_checkpoint(checkpoint, model, refit_vocabulary, labels, config, selected["parameters"],
                    {**provenance, "fitted_partitions": ["train", "validation"],
                     "vocabulary_fitted_partitions": ["train", "validation"], "fitted_documents": len(development),
                     "epoch": selected["best_epoch"], "selected_trial": selected["trial_id"]})
    loaded = load_checkpoint(checkpoint, device)
    probe = development_data.input_ids[:3].to(device)
    model.eval()
    with torch.inference_mode():
        if not torch.allclose(model(probe), loaded["model"](probe), atol=1e-6):
            raise ValueError("Saved neural checkpoint logits differ after loading.")
    test_started = utc_now()
    test = frame.loc[frame["split"] == "test"].reset_index(drop=True)
    test_data = MedicineTextDataset.from_frame(test, refit_vocabulary, config["encoding"], labels)
    test_metrics, predictions = evaluate_model(loaded["model"], make_loader(test_data,
        config["training"]["evaluation_batch_size"], False, config), test, labels, weights, device)
    predictions.to_csv(report_dir / "predictions_test.csv", index=False)
    predictions.loc[~predictions["correct"]].to_csv(report_dir / "test_errors.csv", index=False)
    per_class_metrics(predictions, labels).to_csv(report_dir / "per_class_test_metrics.csv", index=False)
    confusion_table(predictions, labels).to_csv(report_dir / "confusion_matrix.csv")
    coverage = pd.concat([development_data.coverage_table(development), test_data.coverage_table(test)], ignore_index=True)
    coverage.to_csv(report_dir / "token_coverage.csv", index=False)
    comparison = [{"model": row["model"], "display_name": row["display_name"], "phase": 2,
                   "validation_macro_f1": row["validation_macro_f1"], "test_accuracy": row["test_accuracy"],
                   "test_macro_precision": row["test_macro_precision"], "test_macro_recall": row["test_macro_recall"],
                   "test_macro_f1": row["test_macro_f1"], "test_group_balanced_macro_f1": row["test_group_balanced_macro_f1"],
                   "classical_benchmark": row["selected_baseline"]} for row in baseline["comparison"]]
    comparison.append({"model": "text_cnn", "display_name": "PyTorch TextCNN", "phase": 3,
                       "validation_macro_f1": val_metrics["macro_f1"], "test_accuracy": test_metrics["accuracy"],
                       "test_macro_precision": test_metrics["macro_precision"], "test_macro_recall": test_metrics["macro_recall"],
                       "test_macro_f1": test_metrics["macro_f1"], "test_group_balanced_macro_f1": test_metrics["group_balanced_macro_f1"],
                       "classical_benchmark": False})
    pd.DataFrame(comparison).to_csv(report_dir / "model_comparison.csv", index=False)
    # Exploratory representation visualization uses training documents only, after selection/evaluation.
    projection_data = MedicineTextDataset.from_frame(training, refit_vocabulary, config["encoding"], labels)
    embeddings = predict_loader(loaded["model"], make_loader(projection_data,
        config["training"]["evaluation_batch_size"], False, config), device, features=True)
    projection = PCA(n_components=2, svd_solver="randomized", random_state=config["random_state"]).fit_transform(embeddings)
    pd.DataFrame({"record_id": training["record_id"], "class_label": training["class_label"],
                  "pc1": projection[:, 0], "pc2": projection[:, 1]}).to_csv(report_dir / "training_embedding_projection.csv", index=False)
    benchmark = next(row for row in baseline["comparison"] if row["selected_baseline"])
    manifest = {"schema_version": 1, "started_at_utc": started_at, "test_evaluation_started_at_utc": test_started,
                "completed_at_utc": utc_now(), "config": config, "config_sha256": digest,
                "dataset_sha256": split["dataset_sha256"], "split_sha256": split["split_sha256"],
                "baseline_manifest_sha256": file_sha256(root / "reports" / "phase2" / "run_manifest.json"),
                "class_labels": labels, "selected_trial": selected["trial_id"], "selection": selection,
                "validation_metrics": val_metrics, "test_metrics": test_metrics, "refit_epochs": selected["best_epoch"],
                "fitted_documents": len(development), "partition_documents": frame["split"].value_counts().to_dict(),
                "partition_groups": frame.groupby("split")["benchmark_group_id"].nunique().to_dict(),
                "vocabulary_sizes": {"tuning": len(vocabulary.tokens), "refit": len(refit_vocabulary.tokens)},
                "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
                "checkpoint": {"path": "models/phase3/text_cnn.pt", "sha256": file_sha256(checkpoint), "bytes": checkpoint.stat().st_size},
                "classical_benchmark": benchmark["model"], "classical_benchmark_test_macro_f1": benchmark["test_macro_f1"],
                "test_macro_f1_difference_from_benchmark": test_metrics["macro_f1"] - benchmark["test_macro_f1"],
                "environment": {"python": sys.version.split()[0], "torch": str(torch.__version__), "device": str(device),
                                "cpu_threads": torch.get_num_threads(), "cuda_available": torch.cuda.is_available(),
                                **{package: importlib.metadata.version(package) for package in ("numpy", "pandas", "scikit-learn")}},
                "output_checksums": {path.name: file_sha256(path) for path in report_dir.glob("*.csv")}}
    write_json(run_path, manifest)
    LOGGER.info("TextCNN held-out accuracy %.4f; macro F1 %.4f", test_metrics["accuracy"], test_metrics["macro_f1"])
    return manifest
