"""Small non-medical transformer fixtures for window isolation and checkpoint IO."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from medical_doc_assistant.bert_model import DocumentBert, classify_bert_text, load_bert_checkpoint, save_bert_checkpoint, wordpiece_windows

ROOT = Path(__file__).resolve().parents[1]


class BertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def tokenizer(self, directory: Path) -> BertTokenizerFast:
        path = directory / "vocab.txt"
        path.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "amber", "azure", "warm", "cold", "summer", "winter"]), encoding="utf-8")
        return BertTokenizerFast(vocab_file=str(path), do_lower_case=True)

    def model(self) -> DocumentBert:
        config = BertConfig(vocab_size=11, hidden_size=24, intermediate_size=48, num_hidden_layers=1, num_attention_heads=4,
                            hidden_dropout_prob=0, attention_probs_dropout_prob=0, pad_token_id=0)
        return DocumentBert(BertModel(config, add_pooling_layer=False), 2, 0).eval()

    def test_wordpiece_windows_respect_special_tokens_and_full_short_text(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".test-", dir=ROOT) as directory:
            tokenizer = self.tokenizer(Path(directory))
            encoded, stats = wordpiece_windows("amber warm summer", tokenizer, {"max_length": 8, "max_windows": 2})
            self.assertEqual(encoded.shape, (2, 8))
            self.assertEqual(encoded[0, 0], tokenizer.cls_token_id)
            self.assertEqual(encoded[0, 4], tokenizer.sep_token_id)
            self.assertTrue((encoded[1] == tokenizer.pad_token_id).all())
            self.assertEqual(stats["encoded_tokens"], stats["token_count"])

    def test_empty_padding_windows_cannot_change_bert_logits(self) -> None:
        torch.manual_seed(9)
        model = self.model()
        source = torch.tensor([[[2, 5, 7, 3, 0, 0, 0, 0]]])
        padded = torch.cat([source, torch.zeros_like(source)], dim=1)
        with torch.inference_mode():
            torch.testing.assert_close(model(source), model(padded), atol=1e-6, rtol=1e-6)

    def test_encoder_parameters_receive_fine_tuning_gradients(self) -> None:
        model = self.model().train()
        inputs = torch.tensor([[[2, 5, 7, 3, 0, 0, 0, 0]], [[2, 6, 8, 3, 0, 0, 0, 0]]])
        torch.nn.functional.cross_entropy(model(inputs), torch.tensor([0, 1])).backward()
        gradient = model.encoder.embeddings.word_embeddings.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0)

    def test_checkpoint_and_local_tokenizer_preserve_predictions(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".test-", dir=ROOT) as directory:
            root = Path(directory)
            tokenizer = self.tokenizer(root)
            tokenizer.save_pretrained(root / "tokenizer")
            model = self.model()
            checkpoint = root / "model.pt"
            save_bert_checkpoint(checkpoint, model, ["a", "b"], {"max_length": 8, "max_windows": 2}, {"dropout": 0}, {"fixture": True})
            loaded = load_bert_checkpoint(checkpoint)
            encoded, _ = wordpiece_windows("amber warm summer", tokenizer, {"max_length": 8, "max_windows": 2})
            inputs = torch.from_numpy(encoded).unsqueeze(0)
            with torch.inference_mode():
                torch.testing.assert_close(model(inputs), loaded["model"](inputs))
            self.assertIn(classify_bert_text(checkpoint, "amber warm summer")["predicted_class"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
