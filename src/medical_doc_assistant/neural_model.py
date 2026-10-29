"""A padding-aware, multi-window PyTorch TextCNN and standalone checkpoint IO."""

from pathlib import Path
from typing import Any

import torch
from torch import nn

from medical_doc_assistant.cleaning import clean_text
from medical_doc_assistant.neural_text import Vocabulary, encode_document


class TextCNN(nn.Module):
    def __init__(self, vocabulary_size: int, classes: int, embedding_dim: int, filters: int,
                 kernel_sizes: list[int], dropout: float):
        super().__init__()
        self.kernel_sizes = tuple(kernel_sizes)
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)
        self.convolutions = nn.ModuleList(nn.Conv1d(embedding_dim, filters, kernel_size=size) for size in kernel_sizes)
        self.output_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim + 2 * filters * len(kernel_sizes), classes)

    def forward_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 3 or input_ids.shape[-1] < max(self.kernel_sizes):
            raise ValueError("TextCNN expects [batch, chunks, length] with enough length for each kernel.")
        batch, chunks, length = input_ids.shape
        flat = input_ids.reshape(batch * chunks, length)
        token_mask = flat.ne(0)
        embedded = self.embedding_dropout(self.embedding(flat))
        total_tokens = token_mask.reshape(batch, chunks * length).sum(dim=1, keepdim=True).clamp_min(1)
        embedding_mean = (embedded * token_mask.unsqueeze(-1)).reshape(batch, chunks * length, -1).sum(dim=1) / total_tokens
        features = [embedding_mean]
        convolution_input = embedded.transpose(1, 2)
        for size, convolution in zip(self.kernel_sizes, self.convolutions):
            activations = torch.relu(convolution(convolution_input))
            # Windows touching PAD are excluded, including every window of an empty chunk.
            valid = token_mask.unfold(1, size, 1).all(dim=-1)
            width = activations.shape[-1]
            activations = activations.reshape(batch, chunks, -1, width).permute(0, 2, 1, 3).flatten(2)
            valid = valid.reshape(batch, 1, chunks * width)
            maximum = activations.masked_fill(~valid, torch.finfo(activations.dtype).min).amax(dim=-1)
            maximum = torch.where(valid.any(dim=-1), maximum, torch.zeros_like(maximum))
            average = (activations * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)
            features.extend([maximum, average])
        return torch.cat(features, dim=1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.output_dropout(self.forward_features(input_ids)))


def build_text_cnn(vocabulary: Vocabulary, labels: list[str], model_settings: dict[str, Any], dropout: float) -> TextCNN:
    if model_settings["architecture"] != "text_cnn":
        raise ValueError("Unsupported neural architecture.")
    return TextCNN(len(vocabulary.tokens), len(labels), model_settings["embedding_dim"], model_settings["filters"],
                   model_settings["kernel_sizes"], dropout)


def save_checkpoint(path: Path, model: TextCNN, vocabulary: Vocabulary, labels: list[str], config: dict[str, Any],
                    parameters: dict[str, Any], provenance: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"artifact_schema_version": 1, "architecture": "text_cnn", "feature_column": "document_text",
               "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
               "vocabulary": vocabulary.to_dict(), "class_labels": labels, "model_settings": config["model"],
               "encoding_settings": config["encoding"], "parameters": parameters, "provenance": provenance}
    temporary = path.with_name(path.name + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("artifact_schema_version") != 1 or payload.get("architecture") != "text_cnn" or payload.get("feature_column") != "document_text":
        raise ValueError("Unsupported neural checkpoint schema.")
    vocabulary = Vocabulary.from_dict(payload["vocabulary"])
    model = build_text_cnn(vocabulary, payload["class_labels"], payload["model_settings"], payload["parameters"]["dropout"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    return {**payload, "model": model, "vocabulary_object": vocabulary}


def classify_neural_text(path: Path, text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Provide nonempty document text.")
    bundle = load_checkpoint(path)
    identifiers, coverage = encode_document(clean_text(text), bundle["vocabulary_object"], bundle["encoding_settings"])
    with torch.inference_mode():
        index = int(bundle["model"](torch.from_numpy(identifiers).unsqueeze(0)).argmax(dim=1)[0])
    return {"predicted_class": bundle["class_labels"][index], "model": "text_cnn", "class_labels": bundle["class_labels"],
            "token_count": coverage["token_count"], "encoded_tokens": coverage["encoded_tokens"]}
