"""Application runtime: models, document-bound chat, overview, and source navigation."""

import hashlib
import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from medical_doc_assistant.conversation import resolve_follow_up
from medical_doc_assistant.document_overview import build_overview, suggested_questions
from medical_doc_assistant.ingestion import extract_source_fields, normalize_document, text_document_details
from medical_doc_assistant.source_navigation import enrich_citations

DISCLAIMER = "Informational only — not medical advice. Answers quote the supplied document."
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATHS = {
    "text_cnn": ("PyTorch TextCNN", "models/phase3/text_cnn.pt"),
    "random_forest": ("Random Forest", "models/phase2/random_forest.joblib"),
    "logistic_regression": ("Logistic Regression", "models/phase2/logistic_regression.joblib"),
    "linear_svm": ("Linear SVM", "models/phase2/linear_svm.joblib"),
    "bert_mini": ("Fine-tuned BERT-mini", "models/phase4/bert/model.pt"),
}


def load_app_settings(root: Path) -> dict[str, Any]:
    settings = json.loads((root / "configs" / "phase5.json").read_text(encoding="utf-8"))
    if settings.get("schema_version") != 1 or settings["default_classifier"] not in MODEL_PATHS:
        raise ValueError("Invalid application configuration.")
    return settings


def language_scope(text: str) -> str:
    letters = re.findall(r"[^\W\d_]", text, re.UNICODE)
    arabic = re.findall(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]", text)
    return "ar" if arabic and len(arabic) / max(1, len(letters)) > .2 else "en"


class ApplicationRuntime:
    def __init__(self, root: Path = PROJECT_ROOT, storage_root: Path | None = None, settings: dict[str, Any] | None = None):
        self.root = root
        self.settings = settings or load_app_settings(root)
        self.storage_root = storage_root or Path(os.getenv("MEDICAL_STORAGE_ROOT", str(root / ".runtime")))
        self.lock = threading.RLock()
        self.models: dict[str, Any] = {}
        self.documents: dict[str, dict[str, Any]] = {}
        self.assistant: Any = None

    def available_models(self) -> list[dict[str, Any]]:
        comparison = self.root / "reports" / "phase4" / "bert" / "model_comparison.csv"
        metrics = {}
        if comparison.exists():
            import pandas as pd
            metrics = {row["model"]: row for row in pd.read_csv(comparison).to_dict("records")}
        return [{"id": name, "name": title, "available": (self.root / path).exists(),
                 "default": name == self.settings["default_classifier"],
                 "validation_macro_f1": metrics.get(name, {}).get("validation_macro_f1"),
                 "test_accuracy": metrics.get(name, {}).get("test_accuracy"),
                 "test_macro_f1": metrics.get(name, {}).get("test_macro_f1")}
                for name, (title, path) in MODEL_PATHS.items()]

    def classify(self, text: str, model_name: str | None = None) -> dict[str, Any]:
        canonical = normalize_document(text, self.settings)
        if language_scope(canonical) != "en":
            return {"status": "unsupported_language", "predicted_class": None, "model": None,
                    "message": "The evaluated classifier supports English medicine-label text.", "disclaimer": DISCLAIMER}
        name = model_name or self.settings["default_classifier"]
        if name not in MODEL_PATHS:
            raise ValueError("Choose one of the available classifier models.")
        begin = time.perf_counter()
        with self.lock:
            import torch
            torch.set_num_threads(self.settings["cpu_threads"])
            path = self.root / MODEL_PATHS[name][1]
            if not path.exists():
                raise FileNotFoundError("The requested trained model is not installed.")
            if name not in self.models:
                if name == "text_cnn":
                    from medical_doc_assistant.neural_model import load_checkpoint
                    self.models[name] = load_checkpoint(path)
                elif name == "bert_mini":
                    from medical_doc_assistant.bert_model import load_bert_checkpoint
                    self.models[name] = load_bert_checkpoint(path)
                else:
                    from medical_doc_assistant.baseline import load_baseline
                    self.models[name] = load_baseline(path)
            bundle = self.models[name]
            coverage = {}
            if name == "text_cnn":
                from medical_doc_assistant.neural_text import encode_document
                encoded, coverage = encode_document(canonical, bundle["vocabulary_object"], bundle["encoding_settings"])
                with torch.inference_mode():
                    index = int(bundle["model"](torch.from_numpy(encoded).unsqueeze(0)).argmax(dim=1)[0])
                predicted = bundle["class_labels"][index]
            elif name == "bert_mini":
                from medical_doc_assistant.bert_model import wordpiece_windows
                encoded, coverage = wordpiece_windows(canonical, bundle["tokenizer"], bundle["encoding_settings"])
                with torch.inference_mode():
                    index = int(bundle["model"](torch.from_numpy(encoded).unsqueeze(0)).argmax(dim=1)[0])
                predicted = bundle["class_labels"][index]
            else:
                predicted = str(bundle["pipeline"].predict([canonical])[0])
        return {"status": "classified", "predicted_class": predicted, "model": name,
                "model_name": MODEL_PATHS[name][0], "coverage": coverage,
                "latency_ms": round(1000 * (time.perf_counter() - begin), 2), "disclaimer": DISCLAIMER}

    def _expire(self) -> None:
        now = time.monotonic()
        expired = [identifier for identifier, record in self.documents.items()
                   if now - record["created_monotonic"] > self.settings["document_ttl_seconds"]]
        for identifier in expired:
            self.delete_document(identifier)

    def _assistant(self):
        with self.lock:
            if self.assistant is None:
                from medical_doc_assistant.grounded_qa import DocumentAssistant
                from medical_doc_assistant.hf_assets import load_phase4_config
                self.assistant = DocumentAssistant(self.root, load_phase4_config(self.root / "configs" / "phase4.json"),
                                                   storage_root=self.storage_root)
        return self.assistant

    def add_document(self, text: str, source_name: str, model_name: str | None = None,
                     details: dict[str, Any] | None = None, original: bytes | None = None) -> dict[str, Any]:
        canonical = normalize_document(text, self.settings)
        details = details or text_document_details(canonical, self.settings)
        if details["text"] != canonical:
            raise ValueError("Source-location metadata does not match the canonical document.")
        classification = self.classify(canonical, model_name)
        extraction = extract_source_fields(canonical)
        overview = build_overview(canonical, source_name, details["source_blocks"], extraction["sections"])
        suggestions = suggested_questions(canonical, extraction["sections"], overview)
        with self.lock:
            self._expire()
            if len(self.documents) >= self.settings["maximum_documents"]:
                raise ValueError("The demo document limit is reached. Clear a document and try again.")
            identifier = str(uuid.uuid4())
            self.documents[identifier] = {
                "document_id": identifier, "source_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                "source_name": source_name[:200] or "uploaded document", "text": canonical,
                "language": language_scope(canonical), "created_monotonic": time.monotonic(),
                "classification": classification, "extraction": extraction, "index": None,
                "source_blocks": details["source_blocks"], "file_type": details["file_type"],
                "overview": overview, "suggestions": suggestions, "history": [],
                "turn_lock": threading.RLock(), "original": original,
            }
            return self.document_view(identifier)

    def document_view(self, identifier: str) -> dict[str, Any]:
        with self.lock:
            self._expire()
            if identifier not in self.documents:
                raise KeyError("Document not found or expired. Upload or paste it again.")
            record = self.documents[identifier]
            keys = ("document_id", "source_hash", "source_name", "text", "language", "classification", "extraction",
                    "overview", "suggestions", "source_blocks", "file_type")
            return {key: record[key] for key in keys} | {
                "characters": len(record["text"]), "words": len(record["text"].split()),
                "index_ready": record["index"] is not None, "conversation": list(record["history"]),
                "disclaimer": DISCLAIMER,
            }

    def delete_document(self, identifier: str) -> bool:
        with self.lock:
            record = self.documents.pop(identifier, None)
            if not record:
                return False
            if not any(other["source_hash"] == record["source_hash"] for other in self.documents.values()):
                directory = self.storage_root / "data" / "indexes" / record["source_hash"]
                if directory.exists():
                    shutil.rmtree(directory)
            return True

    def ask(self, identifier: str, question: str) -> dict[str, Any]:
        if not question.strip() or len(question) > self.settings["maximum_question_characters"]:
            raise ValueError("Use a nonempty question within 1,000 characters.")
        begin = time.perf_counter()
        with self.lock:
            self._expire()
            if identifier not in self.documents:
                raise KeyError("Document not found or expired. Upload or paste it again.")
            record = self.documents[identifier]
        with record["turn_lock"]:
            if record["language"] != "en" or language_scope(question) != "en":
                from medical_doc_assistant.grounded_qa import MISSING_ANSWER
                return {"document_id": identifier, "question": question, "answer": MISSING_ANSWER,
                        "status": "unsupported_language", "citations": [], "disclaimer": DISCLAIMER}
            resolution = resolve_follow_up(question, record["history"][-self.settings.get("conversation_context_turns", 12):])
            if resolution["clarification"]:
                response = {"answer": resolution["clarification"], "status": "clarification_needed",
                            "answer_mode": "clarification", "citations": [], "question_topic": None}
            else:
                assistant = self._assistant()
                resolved = resolution["resolved_question"]
                from medical_doc_assistant.qa_language import question_intent
                if question_intent(resolved).topic:
                    response = assistant.answer_text(record["text"], resolved, record["source_name"])
                else:
                    if record["index"] is None:
                        record["index"] = assistant.index_document(record["text"], record["source_name"])
                    response = assistant.answer(record["index"], resolved)
            response["citations"] = enrich_citations(response.get("citations", []), record["source_blocks"], record["extraction"]["sections"])
            response["retrieved_passage_count"] = len(response.pop("retrieved_passages", []))
            final = {**response, **{key: resolution[key] for key in ("question", "resolved_question", "context_used", "context_topic")},
                     "document_id": identifier, "source_hash": record["source_hash"],
                     "latency_ms": round(1000 * (time.perf_counter() - begin), 2), "disclaimer": DISCLAIMER}
            with self.lock:
                if identifier not in self.documents:
                    raise KeyError("Document was cleared. Upload it again before asking.")
                record["history"].append({"question": question.strip(), "resolved_question": resolution["resolved_question"],
                                          "response": final})
            return final

    def original_document(self, identifier: str) -> tuple[bytes, str, str]:
        self.document_view(identifier)
        record = self.documents[identifier]
        mime = {"pdf": "application/pdf", "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}.get(
            record["file_type"], "text/plain; charset=utf-8")
        return record["original"] or record["text"].encode("utf-8"), mime, record["source_name"]
