"""Document-level and group-balanced metrics for the frozen baseline benchmark."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


def group_weights(groups: list[str] | np.ndarray | pd.Series) -> np.ndarray:
    series = pd.Series(np.asarray(groups))
    return 1.0 / series.map(series.value_counts()).to_numpy(dtype=float)


def classification_metrics(
    truth: list[str] | np.ndarray | pd.Series,
    predicted: list[str] | np.ndarray | pd.Series,
    labels: list[str],
    groups: list[str] | np.ndarray | pd.Series,
) -> dict[str, float]:
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    if not len(truth) or len(truth) != len(predicted) or len(truth) != len(groups):
        raise ValueError("Metrics require equally sized, nonempty truth, prediction, and group arrays.")
    metrics: dict[str, float] = {}
    for prefix, weights in (("", None), ("group_balanced_", group_weights(groups))):
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth, predicted, labels=labels, average="macro", zero_division=0, sample_weight=weights,
        )
        metrics.update({prefix + "accuracy": float(accuracy_score(truth, predicted, sample_weight=weights)),
                        prefix + "macro_precision": float(precision), prefix + "macro_recall": float(recall),
                        prefix + "macro_f1": float(f1)})
    metrics["weighted_f1"] = float(precision_recall_fscore_support(
        truth, predicted, labels=labels, average="weighted", zero_division=0,
    )[2])
    return metrics


def predictions_table(model_name: str, pipeline: Any, subset: pd.DataFrame) -> pd.DataFrame:
    columns = ["record_id", "class_label", "benchmark_group_id", "split", "generic_names", "source_url", "word_count"]
    result = subset[columns].copy().reset_index(drop=True).rename(columns={"class_label": "true_class"})
    result.insert(0, "model", model_name)
    result["predicted_class"] = pipeline.predict(subset["document_text"])
    result["correct"] = result["true_class"].eq(result["predicted_class"])
    return result


def per_class_metrics(predictions: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    rows = []
    for model, subset in predictions.groupby("model", sort=False):
        report = classification_report(subset["true_class"], subset["predicted_class"], labels=labels,
                                       output_dict=True, zero_division=0)
        for label in labels:
            values = report[label]
            rows.append({"model": model, "class_label": label, "precision": values["precision"],
                         "recall": values["recall"], "f1": values["f1-score"], "support": int(values["support"]),
                         "groups": int(subset.loc[subset["true_class"] == label, "benchmark_group_id"].nunique())})
    return pd.DataFrame(rows)


def confusion_table(predictions: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    matrix = confusion_matrix(predictions["true_class"], predictions["predicted_class"], labels=labels)
    result = pd.DataFrame(matrix, index=labels, columns=labels)
    result.index.name = "true_class"
    return result
