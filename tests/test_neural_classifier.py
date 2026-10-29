"""Synthetic software fixtures for tokenizer isolation, padding, and neural learning."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from medical_doc_assistant.deep_learning import EarlyStopping, balanced_weights, make_loader, predict_loader, select_trial, train_epoch
from medical_doc_assistant.neural_model import build_text_cnn, classify_neural_text, load_checkpoint, save_checkpoint
from medical_doc_assistant.neural_text import MedicineTextDataset, Vocabulary, contiguous_windows, encode_document, fit_vocabulary, tokenize

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs" / "phase3.json").read_text(encoding="utf-8"))


def toy_config() -> dict:
    config = copy.deepcopy(CONFIG)
    config["encoding"] = {"chunk_size": 8, "max_chunks": 2}
    config["model"].update(embedding_dim=12, filters=8, kernel_sizes=[2, 3])
    config["training"]["num_workers"] = 0
    return config


def toy_vocabulary() -> Vocabulary:
    return Vocabulary.fit(["amber warm summer sunshine", "azure cold winter snowfall"] * 4, 64, 1)


class TokenizerTests(unittest.TestCase):
    def test_numbers_units_negation_and_inequalities_are_retained(self) -> None:
        self.assertEqual(tokenize("Do NOT exceed 2.5 mg if age < 12; use 5 µg."),
                         ["do", "not", "exceed", "2.5", "mg", "if", "age", "<", "12", "use", "5", "µg"])

    def test_vocabulary_has_stable_reserved_indices_and_round_trip(self) -> None:
        first = Vocabulary.fit(["beta alpha", "alpha beta"], 10, 1)
        second = Vocabulary.fit(["alpha beta", "beta alpha"], 10, 1)
        self.assertEqual(first.indices["<pad>"], 0)
        self.assertEqual(first.indices["<unk>"], 1)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(Vocabulary.from_dict(first.to_dict()).to_dict(), first.to_dict())

    def test_vocabulary_fitting_rejects_holdout_rows_and_ignores_metadata(self) -> None:
        frame = pd.DataFrame({"document_text": ["amber warm", "azure cold"], "split": ["train", "train"],
                              "generic_names": ["metadataonlysentinel"] * 2})
        vocabulary = fit_vocabulary(frame, {"max_size": 32, "min_frequency": 1}, {"train"})
        self.assertNotIn("metadataonlysentinel", vocabulary.indices)
        encoded, _ = encode_document("validationonlysentinel", vocabulary, {"chunk_size": 4, "max_chunks": 1})
        self.assertEqual(encoded[0, 0], 1)
        self.assertNotIn("validationonlysentinel", vocabulary.indices)
        frame.loc[1, "split"] = "test"
        with self.assertRaisesRegex(ValueError, "holdout"):
            fit_vocabulary(frame, {"max_size": 32, "min_frequency": 1}, {"train"})

    def test_long_windows_are_contiguous_and_include_the_first_and_last_passages(self) -> None:
        tokens = [f"token{index}" for index in range(20)]
        windows = contiguous_windows(tokens, 4, 2)
        self.assertEqual(windows, [(0, tokens[:4]), (16, tokens[-4:])])
        for start, window in windows:
            self.assertEqual(window, tokens[start:start + 4])

    def test_short_documents_are_fully_retained_without_duplicate_tokens(self) -> None:
        tokens = [f"token{index}" for index in range(9)]
        windows = contiguous_windows(tokens, 4, 4)
        self.assertEqual([token for _, window in windows for token in window], tokens)

    def test_dataset_targets_and_input_tensor_do_not_include_record_metadata(self) -> None:
        frame = pd.DataFrame({"document_text": ["amber warm", "azure cold"], "class_label": ["a", "b"],
                              "record_id": ["metadatasonlyone", "metadatasonlytwo"]})
        dataset = MedicineTextDataset.from_frame(frame, toy_vocabulary(), toy_config()["encoding"], ["a", "b"])
        self.assertEqual(tuple(dataset.input_ids.shape), (2, 2, 8))
        self.assertEqual(dataset.targets.tolist(), [0, 1])
        self.assertEqual(dataset[1]["index"], 1)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_padding_cannot_change_logits_even_with_nonzero_biases(self) -> None:
        torch.manual_seed(12)
        model = build_text_cnn(toy_vocabulary(), ["a", "b"], toy_config()["model"], 0.0).eval()
        with torch.no_grad():
            model.embedding.weight[0].fill_(11)
            for convolution in model.convolutions:
                convolution.bias.fill_(9)
        original = torch.tensor([[[2, 3, 4, 0, 0, 0, 0, 0]]])
        padded = torch.zeros((1, 3, 16), dtype=torch.long)
        padded[:, 0, :8] = original[:, 0]
        with torch.inference_mode():
            torch.testing.assert_close(model(original), model(padded), atol=1e-6, rtol=1e-6)

    def test_all_padding_and_short_chunks_produce_finite_logits(self) -> None:
        model = build_text_cnn(toy_vocabulary(), ["a", "b"], toy_config()["model"], 0.0).eval()
        inputs = torch.zeros((2, 2, 8), dtype=torch.long)
        inputs[1, 0, 0] = 2
        self.assertTrue(torch.isfinite(model(inputs)).all())

    def test_checkpoint_preserves_logits_and_complete_inference_settings(self) -> None:
        vocabulary, config = toy_vocabulary(), toy_config()
        model = build_text_cnn(vocabulary, ["a", "b"], config["model"], 0.2).eval()
        dataset = MedicineTextDataset(["amber warm summer"], vocabulary, config["encoding"])
        with tempfile.TemporaryDirectory(prefix=".test-", dir=ROOT) as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, model, vocabulary, ["a", "b"], config,
                            {"dropout": .2, "learning_rate": .001, "batch_size": 2}, {"fixture": "test-only"})
            loaded = load_checkpoint(path)
            torch.testing.assert_close(model(dataset.input_ids), loaded["model"](dataset.input_ids))
            self.assertEqual(loaded["vocabulary_object"].to_dict(), vocabulary.to_dict())
            self.assertIn(classify_neural_text(path, "amber warm summer")["predicted_class"], ["a", "b"])


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_model_actually_learns_a_small_nonmedical_classification_problem(self) -> None:
        torch.manual_seed(17)
        vocabulary, config = toy_vocabulary(), toy_config()
        documents = ["amber warm summer sunshine"] * 4 + ["azure cold winter snowfall"] * 4
        dataset = MedicineTextDataset(documents, vocabulary, config["encoding"], [0] * 4 + [1] * 4)
        model = build_text_cnn(vocabulary, ["a", "b"], config["model"], 0.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.02)
        weights = balanced_weights(dataset.targets, 2, torch.device("cpu"))
        loader = make_loader(dataset, 8, True, config)
        initial = float(F.cross_entropy(model(dataset.input_ids), dataset.targets).detach())
        for _ in range(20):
            train_epoch(model, loader, optimizer, weights, torch.device("cpu"), 1.0)
        model.eval()
        final = float(F.cross_entropy(model(dataset.input_ids), dataset.targets).detach())
        self.assertLess(final, initial * .5)
        self.assertTrue(model(dataset.input_ids).argmax(dim=1).eq(dataset.targets).all())
        self.assertTrue(model.embedding.weight[0].eq(0).all())

    def test_shuffled_prediction_batches_are_restored_to_source_record_order(self) -> None:
        torch.manual_seed(21)
        vocabulary, config = toy_vocabulary(), toy_config()
        dataset = MedicineTextDataset(["amber warm", "azure cold", "summer sunshine", "winter snowfall"],
                                      vocabulary, config["encoding"])
        model = build_text_cnn(vocabulary, ["a", "b"], config["model"], 0.0).eval()
        with torch.inference_mode():
            expected = model(dataset.input_ids).numpy()
        actual = predict_loader(model, make_loader(dataset, 2, True, config), torch.device("cpu"))
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_early_stopping_patience_is_not_reset_by_tied_scores(self) -> None:
        stopper = EarlyStopping(2, .0001)
        self.assertFalse(stopper.update(.5))
        self.assertFalse(stopper.update(.6))
        self.assertFalse(stopper.update(.6))
        self.assertTrue(stopper.update(.599))

    def test_missing_fit_class_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "every target"):
            balanced_weights(torch.tensor([0, 0]), 2, torch.device("cpu"))


class TrialSelectionTests(unittest.TestCase):
    def test_selection_is_validation_based_with_loss_tie_break(self) -> None:
        results = [{"trial_id": "first", "trial_index": 1, "selection_partition": "validation",
                    "validation_metrics": {"macro_f1": .8, "loss": .4}, "test_macro_f1": .1},
                   {"trial_id": "second", "trial_index": 2, "selection_partition": "validation",
                    "validation_metrics": {"macro_f1": .8, "loss": .3}, "test_macro_f1": .99}]
        self.assertEqual(select_trial(results)["trial_id"], "second")
        results[1]["validation_metrics"]["macro_f1"] = .79
        self.assertEqual(select_trial(results)["trial_id"], "first")

    def test_test_partition_cannot_select_hyperparameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "validation"):
            select_trial([{"selection_partition": "test"}])


if __name__ == "__main__":
    unittest.main()
