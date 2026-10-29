"""Per-document FAISS indexes with exact source-text offsets and persisted vectors."""

import hashlib
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from medical_doc_assistant.cleaning import clean_text
from medical_doc_assistant.storage import file_sha256, utc_now, write_json


def document_chunks(text: str, tokenizer: Any, tokens_per_chunk: int, overlap: int) -> list[dict[str, Any]]:
    if not 0 <= overlap < tokens_per_chunk:
        raise ValueError("Chunk overlap must be smaller than the token budget.")
    # Sentence-transformer tokenization can leave a truncation setting on the backend.
    # Chunking must cover the entire source, independently of the embedding batch limit.
    tokenizer.backend_tokenizer.no_truncation()
    tokenizer.backend_tokenizer.no_padding()
    encoded = tokenizer.backend_tokenizer.encode(text, add_special_tokens=False)
    offsets, pieces = encoded.offsets, encoded.tokens
    chunks = []
    start = 0
    while start < len(offsets):
        end = min(len(offsets), start + tokens_per_chunk)
        if end < len(pieces) and pieces[end].startswith("##"):
            boundary = end
            while boundary > start and pieces[boundary].startswith("##"):
                boundary -= 1
            if boundary > start:
                end = boundary
        first, last = offsets[start][0], offsets[end - 1][1]
        chunk_text = text[first:last]
        if chunk_text.strip():
            chunks.append({"chunk_id": f"chunk_{len(chunks):04d}", "char_start": first, "char_end": last,
                           "text": chunk_text, "token_start": start, "token_end": end})
        if end == len(offsets):
            break
        next_start = max(start + 1, end - overlap)
        while next_start > start + 1 and pieces[next_start].startswith("##"):
            next_start -= 1
        start = next_start
    return chunks


class DocumentIndex:
    def __init__(self, document: str, chunks: list[dict[str, Any]], vectors: np.ndarray, metadata: dict[str, Any]):
        self.document, self.chunks, self.metadata = document, chunks, metadata
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if len(chunks) != len(vectors) or not chunks:
            raise ValueError("Index chunks and embedding rows must match and be nonempty.")
        self.index = faiss.IndexFlatIP(self.vectors.shape[1])
        self.index.add(self.vectors)

    @classmethod
    def build(cls, root: Path, text: str, embedder: Any, settings: dict[str, Any], model_specification: dict[str, Any],
              source_name: str = "uploaded document") -> "DocumentIndex":
        document = clean_text(text)
        if not document:
            raise ValueError("Upload or paste a nonempty document before asking questions.")
        identifier = hashlib.sha256(document.encode("utf-8")).hexdigest()
        chunks = document_chunks(document, embedder.tokenizer, settings["chunk_tokens"], settings["overlap_tokens"])
        vectors = embedder.encode([chunk["text"] for chunk in chunks], batch_size=settings["embedding_batch_size"],
                                  convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        metadata = {"schema_version": 1, "document_id": identifier, "source_name": source_name,
                    "created_at_utc": utc_now(), "embedding_model": model_specification,
                    "chunk_tokens": settings["chunk_tokens"], "overlap_tokens": settings["overlap_tokens"], "chunking_version": 2}
        index = cls(document, chunks, vectors, metadata)
        index.save(root / "data" / "indexes" / identifier)
        return index

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "document.txt").write_text(self.document, encoding="utf-8")
        write_json(directory / "chunks.json", self.chunks)
        np.save(directory / "embeddings.npy", self.vectors, allow_pickle=False)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        metadata = {**self.metadata, "dimension": self.vectors.shape[1], "chunks": len(self.chunks),
                    "file_sha256": {name: file_sha256(directory / name) for name in ("document.txt", "chunks.json", "embeddings.npy", "index.faiss")}}
        write_json(directory / "metadata.json", metadata)
        self.metadata = metadata

    @classmethod
    def load(cls, directory: Path) -> "DocumentIndex":
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("schema_version") != 1 or metadata.get("chunking_version") != 2 or any(file_sha256(directory / name) != checksum for name, checksum in metadata["file_sha256"].items()):
            raise ValueError("Document index schema/checksum mismatch.")
        document = (directory / "document.txt").read_text(encoding="utf-8")
        if hashlib.sha256(document.encode("utf-8")).hexdigest() != metadata["document_id"]:
            raise ValueError("The index document ID does not match the stored source text.")
        chunks = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        if any(document[chunk["char_start"]:chunk["char_end"]] != chunk["text"] for chunk in chunks):
            raise ValueError("Chunk offsets do not identify their source passages.")
        vectors = np.load(directory / "embeddings.npy", allow_pickle=False)
        result = cls(document, chunks, vectors, metadata)
        result.index = faiss.read_index(str(directory / "index.faiss"))
        return result

    def retrieve(self, question: str, embedder: Any, limit: int) -> list[dict[str, Any]]:
        vector = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        scores, identifiers = self.index.search(np.ascontiguousarray(vector, dtype=np.float32), min(limit, len(self.chunks)))
        return [{**self.chunks[int(identifier)], "retrieval_score": float(score)}
                for score, identifier in zip(scores[0], identifiers[0]) if identifier >= 0]
