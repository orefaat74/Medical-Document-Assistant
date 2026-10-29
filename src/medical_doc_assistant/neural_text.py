"""Deterministic word vocabulary and contiguous multi-window document encoding."""

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

TOKEN_PATTERN = r"[^\W_]+(?:[.'/-][^\W_]+)*|<=|>=|[≤≥<>%]"
TOKENIZER = re.compile(TOKEN_PATTERN, re.UNICODE)
PAD_TOKEN, UNKNOWN_TOKEN = "<pad>", "<unk>"


def tokenize(text: str) -> list[str]:
    """Keep decimal numbers, units, negation, and inequalities in the input view."""
    return TOKENIZER.findall(unicodedata.normalize("NFC", text).lower())


@dataclass
class Vocabulary:
    tokens: tuple[str, ...]
    frequencies: tuple[int, ...]
    document_count: int
    max_size: int
    min_frequency: int
    indices: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.tokens[:2] != (PAD_TOKEN, UNKNOWN_TOKEN) or len(set(self.tokens)) != len(self.tokens):
            raise ValueError("Vocabulary must have unique tokens with PAD=0 and UNK=1.")
        if len(self.tokens) != len(self.frequencies):
            raise ValueError("Vocabulary token/frequency lengths differ.")
        self.indices = {token: index for index, token in enumerate(self.tokens)}

    @classmethod
    def fit(cls, documents: Iterable[str], max_size: int, min_frequency: int) -> "Vocabulary":
        if max_size < 3 or min_frequency < 1:
            raise ValueError("Vocabulary size/frequency settings are invalid.")
        counts: Counter[str] = Counter()
        document_count = 0
        for text in documents:
            counts.update(tokenize(text))
            document_count += 1
        if not document_count:
            raise ValueError("Fit the vocabulary with nonempty training documents.")
        retained = sorted(((token, count) for token, count in counts.items() if count >= min_frequency),
                          key=lambda item: (-item[1], item[0]))[:max_size - 2]
        return cls((PAD_TOKEN, UNKNOWN_TOKEN, *(token for token, _ in retained)),
                   (0, 0, *(count for _, count in retained)), document_count, max_size, min_frequency)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "token_pattern": TOKEN_PATTERN, "normalization": "NFC + lowercase",
                "tokens": list(self.tokens), "frequencies": list(self.frequencies),
                "document_count": self.document_count, "max_size": self.max_size, "min_frequency": self.min_frequency}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Vocabulary":
        if payload.get("schema_version") != 1 or payload.get("token_pattern") != TOKEN_PATTERN:
            raise ValueError("Unsupported vocabulary/tokenizer schema.")
        return cls(tuple(payload["tokens"]), tuple(payload["frequencies"]), int(payload["document_count"]),
                   int(payload["max_size"]), int(payload["min_frequency"]))


def fit_vocabulary(frame: pd.DataFrame, settings: dict[str, Any], allowed_splits: set[str]) -> Vocabulary:
    if not set(frame["split"]) <= allowed_splits:
        raise ValueError("Vocabulary fitting includes a disallowed holdout partition.")
    return Vocabulary.fit(frame["document_text"], settings["max_size"], settings["min_frequency"])


def contiguous_windows(tokens: list[str], chunk_size: int, max_chunks: int) -> list[tuple[int, list[str]]]:
    if chunk_size < 1 or max_chunks < 1:
        raise ValueError("Encoding windows must have positive dimensions.")
    if not tokens:
        return []
    if len(tokens) <= chunk_size * max_chunks:
        starts = range(0, len(tokens), chunk_size)
    elif max_chunks == 1:
        starts = [0]
    else:
        starts = np.linspace(0, len(tokens) - chunk_size, max_chunks, dtype=int).tolist()
    return [(int(start), tokens[start:start + chunk_size]) for start in starts]


def encode_document(text: str, vocabulary: Vocabulary, settings: dict[str, int]) -> tuple[np.ndarray, dict[str, Any]]:
    tokens = tokenize(text)
    windows = contiguous_windows(tokens, settings["chunk_size"], settings["max_chunks"])
    encoded = np.zeros((settings["max_chunks"], settings["chunk_size"]), dtype=np.int64)
    unknown = 0
    retained = 0
    for index, (_, window) in enumerate(windows):
        identifiers = [vocabulary.indices.get(token, 1) for token in window]
        encoded[index, :len(identifiers)] = identifiers
        retained += len(identifiers)
        unknown += identifiers.count(1)
    return encoded, {"token_count": len(tokens), "encoded_tokens": retained, "chunks": len(windows),
                     "retention_ratio": retained / len(tokens) if tokens else 1.0,
                     "unknown_tokens": unknown, "unknown_ratio": unknown / retained if retained else 0.0}


class MedicineTextDataset(Dataset):
    """One source document per sample; metadata is not included in input tensors."""

    def __init__(self, documents: Iterable[str], vocabulary: Vocabulary, settings: dict[str, int],
                 targets: Iterable[int] | None = None):
        encodings, self.coverage = [], []
        for text in documents:
            encoded, stats = encode_document(text, vocabulary, settings)
            encodings.append(encoded)
            self.coverage.append(stats)
        if not encodings:
            raise ValueError("The document dataset cannot be empty.")
        self.input_ids = torch.from_numpy(np.stack(encodings))
        labels = list(targets) if targets is not None else [-1] * len(encodings)
        if len(labels) != len(encodings):
            raise ValueError("Document and target counts differ.")
        self.targets = torch.tensor(labels, dtype=torch.long)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, vocabulary: Vocabulary, settings: dict[str, int], labels: list[str]) -> "MedicineTextDataset":
        label_index = {label: index for index, label in enumerate(labels)}
        return cls(frame["document_text"], vocabulary, settings, [label_index[label] for label in frame["class_label"]])

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        return {"input_ids": self.input_ids[index], "target": self.targets[index], "index": index}

    def coverage_table(self, frame: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([frame[["record_id", "class_label", "split", "benchmark_group_id"]].reset_index(drop=True),
                          pd.DataFrame(self.coverage)], axis=1)
