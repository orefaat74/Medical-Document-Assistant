"""TF-IDF baseline comparison with training-only CV and pre-test model selection."""

import importlib.metadata
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from medical_doc_assistant.baseline_evaluation import classification_metrics, confusion_table, per_class_metrics, predictions_table
from medical_doc_assistant.cleaning import clean_text
from medical_doc_assistant.splitting import config_digest, prepare_phase2, training_cv_indices
from medical_doc_assistant.storage import file_sha256, utc_now, write_json

LOGGER = logging.getLogger(__name__)
MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "linear_svm": "Linear SVM",
    "random_forest": "Random Forest",
}


def build_pipeline(model_name: str, config: dict[str, Any], memory: Memory | None = None) -> Pipeline:
    parameters = dict(config["tfidf"])
    parameters["ngram_range"] = tuple(parameters["ngram_range"])
    vectorizer = TfidfVectorizer(**parameters, dtype=np.float32)
    settings = config["training"]
    seed = config["random_state"]
    if model_name == "logistic_regression":
        classifier = LogisticRegression(class_weight="balanced", solver="lbfgs", tol=1e-3,
                                        max_iter=settings["logistic_max_iter"], random_state=seed)
    elif model_name == "linear_svm":
        classifier = LinearSVC(class_weight="balanced", dual="auto", max_iter=settings["svm_max_iter"], random_state=seed)
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=settings["forest_estimators"], class_weight="balanced_subsample", max_features="sqrt",
            n_jobs=settings["estimator_jobs"], random_state=seed,
        )
    else:
        raise ValueError(f"Unknown baseline family: {model_name}")
    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)], memory=memory)


def tune_on_training(
    model_name: str, training: pd.DataFrame, labels: list[str], config: dict[str, Any], memory: Memory | None = None
) -> GridSearchCV:
    cv = training_cv_indices(training, config["cv_folds"])
    for fitting, validation in cv:
        if set(training.iloc[fitting]["class_label"]) != set(labels) or set(training.iloc[validation]["class_label"]) != set(labels):
            raise ValueError("Every training CV fit/validation fold must contain all target classes.")
        if set(training.iloc[fitting]["benchmark_group_id"]) & set(training.iloc[validation]["benchmark_group_id"]):
            raise ValueError("Training CV folds share a benchmark group.")
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "macro_precision": make_scorer(precision_score, labels=labels, average="macro", zero_division=0),
        "macro_recall": make_scorer(recall_score, labels=labels, average="macro", zero_division=0),
        "macro_f1": make_scorer(f1_score, labels=labels, average="macro", zero_division=0),
    }
    search = GridSearchCV(
        build_pipeline(model_name, config, memory), config["parameter_grids"][model_name],
        scoring=scoring, refit="macro_f1", cv=cv, n_jobs=config["training"]["cv_jobs"],
        error_score="raise", return_train_score=False,
    )
    with warnings.catch_warnings(), threadpool_limits(limits=config["training"]["blas_threads"]):
        warnings.simplefilter("error", ConvergenceWarning)
        # Only clinical text is passed to the pipeline. Metadata and holdout text are excluded.
        search.fit(training["document_text"], training["class_label"])
    return search


def select_baseline(validation_rows: list[dict[str, Any]]) -> str:
    if not validation_rows or any(row["selection_partition"] != "validation" for row in validation_rows):
        raise ValueError("Select the benchmark using validation results, before test evaluation.")
    order = {name: index for index, name in enumerate(MODEL_NAMES)}
    best = max(validation_rows, key=lambda row: (row["validation_macro_f1"], row["cv_macro_f1_mean"], -order[row["model"]]))
    return best["model"]


def _cv_table(model: str, search: GridSearchCV, folds: int) -> pd.DataFrame:
    table = pd.DataFrame(search.cv_results_)
    result = pd.DataFrame({"model": model, "parameters": [json.dumps(params, sort_keys=True) for params in table["params"]]})
    result["rank_macro_f1"] = table["rank_test_macro_f1"]
    for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        result[metric + "_mean"] = table["mean_test_" + metric]
        result[metric + "_std"] = table["std_test_" + metric]
    for fold in range(folds):
        result[f"fold_{fold}_macro_f1"] = table[f"split{fold}_test_macro_f1"]
    result["mean_fit_seconds"] = table["mean_fit_time"]
    return result


def _top_features(model: str, pipeline: Pipeline) -> list[dict[str, Any]]:
    terms = pipeline.named_steps["tfidf"].get_feature_names_out()
    classifier = pipeline.named_steps["classifier"]
    rows = []
    if hasattr(classifier, "coef_"):
        coefficients_by_class = classifier.coef_
        if coefficients_by_class.shape[0] == 1 and len(classifier.classes_) == 2:
            coefficients_by_class = np.vstack([-coefficients_by_class[0], coefficients_by_class[0]])
        for label, coefficients in zip(classifier.classes_, coefficients_by_class):
            for index in np.argsort(coefficients)[-15:][::-1]:
                rows.append({"model": model, "class_label": label, "term": terms[index],
                             "weight": float(coefficients[index]), "weight_type": "positive_coefficient"})
    else:
        for index in np.argsort(classifier.feature_importances_)[-30:][::-1]:
            rows.append({"model": model, "class_label": "all_classes", "term": terms[index],
                         "weight": float(classifier.feature_importances_[index]), "weight_type": "global_feature_importance"})
    return rows


def load_baseline(path: Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    if bundle.get("artifact_schema_version") != 1 or bundle.get("feature_column") != "document_text":
        raise ValueError("Unsupported baseline artifact schema or input column.")
    return bundle


def classify_text(path: Path, text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Provide nonempty document text.")
    bundle = load_baseline(path)
    return {"predicted_class": str(bundle["pipeline"].predict([clean_text(text)])[0]),
            "model": bundle["model_name"], "class_labels": bundle["class_labels"]}


def run_phase2(root: Path, config: dict[str, Any], labels: list[str]) -> dict[str, Any]:
    from medical_doc_assistant.phase2_validation import validate_splits

    frame, split_manifest = prepare_phase2(root, config, labels)
    report_dir = root / "reports" / "phase2"
    model_dir = root / "models" / "phase2"
    run_path = report_dir / "run_manifest.json"
    digest = config_digest(config)
    if run_path.exists():
        previous = json.loads(run_path.read_text(encoding="utf-8"))
        if previous["config_sha256"] != digest or previous["split_sha256"] != split_manifest["split_sha256"]:
            raise ValueError("A completed benchmark uses different settings. Preserve its results before starting a new experiment.")
        LOGGER.info("Using the completed Phase 2 benchmark: %s", previous["selected_model"])
        return previous
    split_checks = validate_splits(frame, split_manifest, config,
                                   json.loads((root / "data" / "reference" / "rxnorm_ingredient_map.json").read_text(encoding="utf-8")),
                                   pd.read_csv(report_dir / "near_duplicate_pairs.csv"))
    write_json(report_dir / "split_validation.json", split_checks)
    if not split_checks["all_passed"]:
        raise ValueError("Resolve Phase 2 split validation failures before training.")
    model_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = root / ".cache" / "phase2_tfidf"
    memory = Memory(location=str(cache_dir), verbose=0)
    training = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    validation = frame.loc[frame["split"] == "validation"].reset_index(drop=True)
    comparison_rows, searches, cv_tables, validation_predictions = [], {}, [], []
    started = utc_now()
    for model, display_name in MODEL_NAMES.items():
        LOGGER.info("Training %s with %s group-aware CV folds", display_name, config["cv_folds"])
        begin = time.perf_counter()
        search = tune_on_training(model, training, labels, config, memory)
        elapsed = time.perf_counter() - begin
        predictions = predictions_table(model, search.best_estimator_, validation)
        metrics = classification_metrics(predictions["true_class"], predictions["predicted_class"], labels, predictions["benchmark_group_id"])
        best_index = search.best_index_
        row = {"model": model, "display_name": display_name, "selection_partition": "validation",
               "best_parameters": json.dumps(search.best_params_, sort_keys=True), "search_seconds": elapsed,
               "cv_macro_f1_mean": float(search.cv_results_["mean_test_macro_f1"][best_index]),
               "cv_macro_f1_std": float(search.cv_results_["std_test_macro_f1"][best_index]),
               **{"validation_" + key: value for key, value in metrics.items()}}
        comparison_rows.append(row)
        searches[model] = search
        cv_tables.append(_cv_table(model, search, config["cv_folds"]))
        validation_predictions.append(predictions)
        LOGGER.info("%s: CV macro F1 %.4f; validation macro F1 %.4f", display_name, row["cv_macro_f1_mean"], metrics["macro_f1"])
    selected = select_baseline(comparison_rows)
    # Persist this decision before any test prediction or metric is produced.
    selection = {"selected_at_utc": utc_now(), "selected_model": selected, "selection_metric": "validation macro F1",
                 "tie_break": "training CV macro F1, then Logistic Regression / Linear SVM / Random Forest order",
                 "test_used_for_selection": False, "validation_results": comparison_rows}
    write_json(report_dir / "model_selection.json", selection)
    pd.concat(cv_tables, ignore_index=True).to_csv(report_dir / "cv_results.csv", index=False)
    pd.concat(validation_predictions, ignore_index=True).to_csv(report_dir / "predictions_validation.csv", index=False)
    LOGGER.info("Selected %s before test evaluation; refitting frozen candidates on train + validation", MODEL_NAMES[selected])
    development = frame.loc[frame["split"] != "test"].reset_index(drop=True)
    probe = training["document_text"].head(8)
    artifacts, features = {}, []
    for row in comparison_rows:
        model = row["model"]
        pipeline = clone(searches[model].best_estimator_)
        begin = time.perf_counter()
        with warnings.catch_warnings(), threadpool_limits(limits=config["training"]["blas_threads"]):
            warnings.simplefilter("error", ConvergenceWarning)
            pipeline.fit(development["document_text"], development["class_label"])
        row["final_fit_seconds"] = time.perf_counter() - begin
        row["features"] = len(pipeline.named_steps["tfidf"].vocabulary_)
        # Artifacts are standalone; a machine-specific training cache is not serialized.
        pipeline.memory = None
        bundle = {"artifact_schema_version": 1, "pipeline": pipeline, "model_name": model, "class_labels": labels,
                  "feature_column": "document_text", "fitted_partitions": ["train", "validation"],
                  "fitted_documents": len(development), "dataset_sha256": split_manifest["dataset_sha256"],
                  "split_sha256": split_manifest["split_sha256"], "config_sha256": digest}
        path = model_dir / (model + ".joblib")
        joblib.dump(bundle, path, compress=3)
        loaded = load_baseline(path)
        if not np.array_equal(pipeline.predict(probe), loaded["pipeline"].predict(probe)):
            raise ValueError("Saved baseline predictions differ after loading.")
        artifacts[model] = {"path": str(path.relative_to(root)).replace("\\", "/"),
                            "sha256": file_sha256(path), "bytes": path.stat().st_size}
        features.extend(_top_features(model, pipeline))
        if model == selected:
            joblib.dump(bundle, model_dir / "benchmark.joblib", compress=3)
    test = frame.loc[frame["split"] == "test"].reset_index(drop=True)
    test_started = utc_now()
    test_predictions = []
    for row in comparison_rows:
        model = row["model"]
        predictions = predictions_table(model, load_baseline(model_dir / (model + ".joblib"))["pipeline"], test)
        metrics = classification_metrics(predictions["true_class"], predictions["predicted_class"], labels, predictions["benchmark_group_id"])
        row.update({"test_" + key: value for key, value in metrics.items()})
        row["selected_baseline"] = model == selected
        test_predictions.append(predictions)
        confusion_table(predictions, labels).to_csv(report_dir / ("confusion_matrix_" + model + ".csv"))
        LOGGER.info("%s: held-out test accuracy %.4f; macro F1 %.4f", MODEL_NAMES[model], metrics["accuracy"], metrics["macro_f1"])
    predictions = pd.concat(test_predictions, ignore_index=True)
    predictions.to_csv(report_dir / "predictions_test.csv", index=False)
    predictions.loc[~predictions["correct"]].to_csv(report_dir / "test_errors.csv", index=False)
    per_class_metrics(predictions, labels).to_csv(report_dir / "per_class_test_metrics.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(report_dir / "model_comparison.csv", index=False)
    pd.DataFrame(features).to_csv(report_dir / "top_features.csv", index=False)
    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(development), 1)), development["class_label"])
    dummy_metrics = classification_metrics(test["class_label"], dummy.predict(np.zeros((len(test), 1))), labels, test["benchmark_group_id"])
    manifest = {
        "schema_version": 1, "started_at_utc": started, "test_evaluation_started_at_utc": test_started,
        "completed_at_utc": utc_now(), "selected_model": selected, "selection": selection,
        "dataset_sha256": split_manifest["dataset_sha256"], "split_sha256": split_manifest["split_sha256"],
        "config_sha256": digest, "config": config, "class_labels": labels,
        "partition_documents": frame["split"].value_counts().to_dict(),
        "partition_groups": frame.groupby("split")["benchmark_group_id"].nunique().to_dict(),
        "models": artifacts, "benchmark_artifact": {"path": "models/phase2/benchmark.joblib",
                                                       "sha256": file_sha256(model_dir / "benchmark.joblib")},
        "dummy_test_metrics": dummy_metrics,
        "dummy_predicted_class": str(dummy.classes_[int(np.argmax(dummy.class_prior_))]),
        "comparison": comparison_rows,
        "environment": {"python": sys.version.split()[0],
                        **{package: importlib.metadata.version(package) for package in (
                            "scikit-learn", "scipy", "numpy", "pandas", "joblib", "threadpoolctl")}},
    }
    # Selection rows were extended for reporting; keep the persisted pre-test selection authoritative.
    manifest["selection"] = json.loads((report_dir / "model_selection.json").read_text(encoding="utf-8"))
    manifest["output_checksums"] = {path.name: file_sha256(path) for path in report_dir.glob("*.csv")}
    write_json(run_path, manifest)
    return manifest
