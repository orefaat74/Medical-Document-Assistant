"""Multi-window BERT document classification and self-contained checkpoint inference."""

from pathlib import Path
from typing import Any

from medical_doc_assistant.hf_assets import ensure_model_assets
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer, BertConfig, BertModel

from medical_doc_assistant.cleaning import clean_text
from medical_doc_assistant.neural_text import MedicineTextDataset


def wordpiece_windows(text: str, tokenizer: Any, settings: dict[str, int]) -> tuple[np.ndarray, dict[str, Any]]:
    pieces = tokenizer.backend_tokenizer.encode(text, add_special_tokens=False).ids
    width, count = settings["max_length"] - 2, settings["max_windows"]
    if len(pieces) <= width * count:
        starts = list(range(0, len(pieces), width))
    elif count == 1:
        starts = [0]
    else:
        starts = np.linspace(0, len(pieces) - width, count, dtype=int).tolist()
    result = np.full((count, settings["max_length"]), tokenizer.pad_token_id, dtype=np.int64)
    seen = 0
    for index, start in enumerate(starts):
        window = pieces[start:start + width]
        sequence = tokenizer.build_inputs_with_special_tokens(window)
        result[index, :len(sequence)] = sequence
        seen += len(window)
    return result, {"token_count": len(pieces), "encoded_tokens": seen, "windows": len(starts),
                    "retention_ratio": seen / len(pieces) if pieces else 1.0}


class BertDocumentDataset(MedicineTextDataset):
    def __init__(self, frame: pd.DataFrame, tokenizer: Any, settings: dict[str, int], labels: list[str]):
        encodings, self.coverage = [], []
        for text in frame["document_text"]:
            encoded, stats = wordpiece_windows(text, tokenizer, settings)
            encodings.append(encoded)
            self.coverage.append(stats)
        if not encodings:
            raise ValueError("BERT requires a nonempty document dataset.")
        self.input_ids = torch.from_numpy(np.stack(encodings))
        mapping = {label: index for index, label in enumerate(labels)}
        self.targets = torch.tensor([mapping[label] for label in frame["class_label"]], dtype=torch.long)


class DocumentBert(nn.Module):
    def __init__(self, encoder: BertModel, classes: int, dropout: float):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * encoder.config.hidden_size, classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, windows, length = input_ids.shape
        flat = input_ids.reshape(batch * windows, length)
        attention = flat.ne(self.encoder.config.pad_token_id or 0)
        active = attention.any(dim=1)
        positions = torch.nonzero(active, as_tuple=False).flatten()
        values = self.classifier.weight.new_zeros((batch * windows, self.encoder.config.hidden_size))
        if len(positions):
            encoded = self.encoder(flat[active], attention_mask=attention[active]).last_hidden_state[:, 0]
            values = values.index_copy(0, positions, encoded)
        values = values.reshape(batch, windows, -1)
        valid = active.reshape(batch, windows, 1)
        average = (values * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        maximum = values.masked_fill(~valid, torch.finfo(values.dtype).min).amax(dim=1)
        maximum = torch.where(valid.any(dim=1), maximum, torch.zeros_like(maximum))
        return self.classifier(self.dropout(torch.cat([average, maximum], dim=1)))


def pretrained_document_bert(path: Path, labels: list[str], dropout: float) -> DocumentBert:
    encoder = BertModel.from_pretrained(path, local_files_only=True, add_pooling_layer=False, attn_implementation="sdpa")
    return DocumentBert(encoder, len(labels), dropout)


def save_bert_checkpoint(path: Path, model: DocumentBert, labels: list[str], settings: dict[str, Any],
                         parameters: dict[str, Any], provenance: dict[str, Any]) -> None:
    model_config = model.encoder.config.to_dict()
    model_config["_name_or_path"] = ""
    payload = {"artifact_schema_version": 1, "architecture": "bert_mini", "class_labels": labels,
               "bert_config": model_config, "encoding_settings": {"max_length": settings["max_length"], "max_windows": settings["max_windows"]},
               "parameters": parameters, "provenance": provenance,
               "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_bert_checkpoint(path: Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("artifact_schema_version") != 1 or payload.get("architecture") != "bert_mini":
        raise ValueError("Unsupported BERT checkpoint schema.")
    encoder = BertModel(BertConfig(**payload["bert_config"]), add_pooling_layer=False)
    model = DocumentBert(encoder, len(payload["class_labels"]), payload["parameters"]["dropout"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(path.parent / "tokenizer", local_files_only=True, use_fast=True)
    return {**payload, "model": model, "tokenizer": tokenizer}


def classify_bert_text(path: Path, text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Provide nonempty document text.")
    bundle = load_bert_checkpoint(path)
    encoded, coverage = wordpiece_windows(clean_text(text), bundle["tokenizer"], bundle["encoding_settings"])
    with torch.inference_mode():
        index = int(bundle["model"](torch.from_numpy(encoded).unsqueeze(0)).argmax(dim=1)[0])
    return {"predicted_class": bundle["class_labels"][index], "model": "bert_mini",
            "class_labels": bundle["class_labels"], "token_count": coverage["token_count"], "encoded_tokens": coverage["encoded_tokens"]}
