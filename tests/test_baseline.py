"""Synthetic non-medical fixtures for training leakage, metrics, and persistence."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from medical_doc_assistant.baseline import MODEL_NAMES, build_pipeline, classify_text, load_baseline, select_baseline, tune_on_training
from medical_doc_assistant.baseline_evaluation import classification_metrics, group_weights

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs" / "phase2.json").read_text(encoding="utf-8"))


def small_config() -> dict:
    config = copy.deepcopy(CONFIG)
    config["tfidf"].update(ngram_range=[1, 1], min_df=1, max_df=1.0, max_features=200)
    config["training"].update(forest_estimators=12, estimator_jobs=1, cv_jobs=1, blas_threads=1)
    config["parameter_grids"] = {
        "logistic_regression": {"classifier__C": [1.0]},
        "linear_svm": {"classifier__C": [1.0]},
        "random_forest": {"classifier__max_depth": [None], "classifier__min_samples_leaf": [1]},
    }
    return config


def training_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"record_id": f"{label}-{fold}-{index}", "class_label": label, "split": "train", "cv_fold": fold,
         "benchmark_group_id": f"group-{label}-{fold}", "generic_names": '["metadataonlysentinel"]',
         "document_text": f"{topic} software fixture sample{fold} instance{index}"}
        for label, topic in (("a", "amber ruby warm summer sunshine"), ("b", "azure navy cool winter snowfall"))
        for fold in range(3) for index in range(2)
    ])


class TrainingTests(unittest.TestCase):
    def test_tfidf_is_fit_separately_inside_each_cv_fit_fold(self) -> None:
        frame = training_frame()
        observed = []
        original = TfidfVectorizer.fit_transform

        def track_fit(vectorizer: TfidfVectorizer, documents: pd.Series, y: object = None):
            observed.append(set(documents))
            return original(vectorizer, documents, y)

        with patch.object(TfidfVectorizer, "fit_transform", track_fit):
            search = tune_on_training("logistic_regression", frame, ["a", "b"], small_config())
        for fold in range(3):
            self.assertEqual(observed[fold], set(frame.loc[frame.cv_fold != fold, "document_text"]))
            self.assertFalse(observed[fold] & set(frame.loc[frame.cv_fold == fold, "document_text"]))
        self.assertEqual(observed[-1], set(frame.document_text))
        self.assertEqual(len(observed), 4)
        vocabulary = search.best_estimator_.named_steps["tfidf"].vocabulary_
        self.assertNotIn("metadataonlysentinel", vocabulary)
        self.assertNotIn("validationonlysentinel", vocabulary)
        self.assertNotIn("testonlysentinel", vocabulary)
        search.best_estimator_.predict(["ruby warm validationonlysentinel", "navy cool testonlysentinel"])
        self.assertNotIn("testonlysentinel", vocabulary)

    def test_tuning_rejects_mixed_training_and_holdout_rows(self) -> None:
        frame = training_frame()
        frame.loc[0, "split"] = "validation"
        with self.assertRaisesRegex(ValueError, "only the training"):
            tune_on_training("logistic_regression", frame, ["a", "b"], small_config())

    def test_all_three_pipeline_families_survive_serialization(self) -> None:
        frame = training_frame()
        with tempfile.TemporaryDirectory(prefix=".test-", dir=ROOT) as directory:
            for name in MODEL_NAMES:
                with self.subTest(model=name):
                    pipeline = build_pipeline(name, small_config()).fit(frame.document_text, frame.class_label)
                    path = Path(directory) / (name + ".joblib")
                    bundle = {"artifact_schema_version": 1, "feature_column": "document_text", "pipeline": pipeline,
                              "model_name": name, "class_labels": ["a", "b"]}
                    joblib.dump(bundle, path, compress=3)
                    loaded = load_baseline(path)
                    queries = ["amber ruby warm summer", "azure navy cool winter"]
                    np.testing.assert_array_equal(pipeline.predict(queries), loaded["pipeline"].predict(queries))
                    self.assertEqual(classify_text(path, "<p>amber ruby warm summer</p>")["predicted_class"], pipeline.predict([queries[0]])[0])


class SelectionTests(unittest.TestCase):
    def test_test_scores_cannot_change_the_validation_winner(self) -> None:
        rows = [
            {"model": "logistic_regression", "selection_partition": "validation", "validation_macro_f1": .8,
             "cv_macro_f1_mean": .7, "test_macro_f1": .1},
            {"model": "linear_svm", "selection_partition": "validation", "validation_macro_f1": .7,
             "cv_macro_f1_mean": .8, "test_macro_f1": .99},
        ]
        self.assertEqual(select_baseline(rows), "logistic_regression")

    def test_test_partition_is_rejected_for_selection(self) -> None:
        with self.assertRaisesRegex(ValueError, "validation"):
            select_baseline([{"model": "linear_svm", "selection_partition": "test", "validation_macro_f1": 1.0,
                              "cv_macro_f1_mean": 1.0}])

    def test_validation_ties_use_training_cv_then_fixed_family_order(self) -> None:
        rows = [{"model": name, "selection_partition": "validation", "validation_macro_f1": .8, "cv_macro_f1_mean": .7}
                for name in MODEL_NAMES]
        self.assertEqual(select_baseline(rows), "logistic_regression")
        rows[1]["cv_macro_f1_mean"] = .75
        self.assertEqual(select_baseline(rows), "linear_svm")


class GroupMetricTests(unittest.TestCase):
    def test_each_ingredient_group_receives_total_weight_one(self) -> None:
        groups = ["large", "large", "large", "small"]
        weights = group_weights(groups)
        totals = pd.Series(weights).groupby(groups).sum()
        np.testing.assert_allclose(totals, [1, 1])
        metrics = classification_metrics(["a", "a", "a", "b"], ["a", "a", "a", "a"], ["a", "b"], groups)
        self.assertEqual(metrics["accuracy"], .75)
        self.assertEqual(metrics["group_balanced_accuracy"], .5)

    def test_replicating_one_group_does_not_improve_group_balanced_metrics(self) -> None:
        original = classification_metrics(["a", "b"], ["a", "a"], ["a", "b"], ["large", "small"])
        repeated = classification_metrics(["a"] * 6 + ["b"], ["a"] * 7, ["a", "b"], ["large"] * 6 + ["small"])
        self.assertGreater(repeated["accuracy"], original["accuracy"])
        self.assertAlmostEqual(repeated["group_balanced_macro_f1"], original["group_balanced_macro_f1"])


if __name__ == "__main__":
    unittest.main()
