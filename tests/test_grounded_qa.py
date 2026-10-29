"""Non-medical fixtures for complete chunking, index isolation, and answer grounding."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from transformers import BertTokenizerFast

from medical_doc_assistant.document_index import DocumentIndex, document_chunks
from medical_doc_assistant.grounded_qa import DocumentAssistant, MISSING_ANSWER, supported_quote
from medical_doc_assistant.rag_evaluation import response_supported

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs" / "phase4.json").read_text(encoding="utf-8"))


class FakeEmbedder:
    def __init__(self, directory: Path):
        path = directory / "vocab.txt"
        path.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "amber", "azure", "widget", "wooden", "cabinet", "stored", "inside", "the", "is", "a", "warnings", "do", "not", "press", "red", "button"]), encoding="utf-8")
        self.tokenizer = BertTokenizerFast(vocab_file=str(path), do_lower_case=True)

    def encode(self, texts, **kwargs):
        vectors = np.array([[float("amber" in text.lower()), float("azure" in text.lower()), .1] for text in texts], dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class FakeAnswerability:
    def __init__(self, answer="wooden cabinet", margin=10):
        self.answer, self.margin = answer, margin

    def candidates(self, question, passages):
        for passage in passages:
            offset = passage["text"].find(self.answer)
            if offset >= 0:
                return [{"span": self.answer, "char_start": passage["char_start"] + offset,
                         "char_end": passage["char_start"] + offset + len(self.answer), "span_score": 1.0,
                         "answer_margin": self.margin, "passage": passage}]
        return []


class FakeGenerator:
    def __init__(self, answer="wooden cabinet"):
        self.output = answer

    def answer(self, question, passages):
        return self.output


class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".test-", dir=ROOT)
        self.root = Path(self.temporary.name)
        self.embedder = FakeEmbedder(self.root)
        self.config = copy.deepcopy(CONFIG)
        self.config["rag"].update(chunk_tokens=20, overlap_tokens=5, minimum_retrieval_score=0.0)
        self.text = "The amber widget is stored inside a wooden cabinet. The azure widget is beside the door."

    def tearDown(self):
        self.temporary.cleanup()

    def assistant(self, generator=None, answerability=None):
        return DocumentAssistant(self.root, self.config, embedder=self.embedder,
                                 answerability=answerability or FakeAnswerability(), generator=generator or FakeGenerator())

    def test_chunking_disables_leftover_tokenizer_truncation_and_covers_the_source_tail(self):
        text = " ".join(["amber widget wooden cabinet"] * 20)
        self.embedder.tokenizer.backend_tokenizer.enable_truncation(max_length=8)
        chunks = document_chunks(text, self.embedder.tokenizer, 12, 3)
        self.assertGreater(len(chunks), 2)
        self.assertEqual(chunks[0]["char_start"], 0)
        self.assertEqual(chunks[-1]["char_end"], len(text))
        self.assertTrue(all(text[chunk["char_start"]:chunk["char_end"]] == chunk["text"] for chunk in chunks))

    def test_index_round_trip_preserves_vectors_source_and_retrieval(self):
        assistant = self.assistant()
        index = assistant.index_document(self.text)
        restored = DocumentIndex.load(self.root / "data" / "indexes" / index.metadata["document_id"])
        np.testing.assert_array_equal(index.vectors, restored.vectors)
        self.assertEqual(index.document, restored.document)
        self.assertEqual(index.retrieve("amber", self.embedder, 1)[0]["text"], restored.retrieve("amber", self.embedder, 1)[0]["text"])

    def test_separate_document_index_cannot_retrieve_other_document_content(self):
        assistant = self.assistant()
        amber = assistant.index_document(self.text)
        azure = assistant.index_document("The azure widget belongs to the blue shelf.")
        self.assertNotEqual(amber.metadata["document_id"], azure.metadata["document_id"])
        self.assertNotIn("wooden cabinet", " ".join(passage["text"] for passage in azure.retrieve("amber", self.embedder, 4)))

    def test_verified_llm_span_returns_exact_source_quote_with_valid_offsets(self):
        assistant = self.assistant()
        index = assistant.index_document(self.text)
        response = assistant.answer(index, "Where is the amber widget stored?")
        self.assertEqual(response["status"], "answered")
        self.assertEqual(response["answer_mode"], "verified_llm_span")
        self.assertTrue(response_supported(response, index.document))
        self.assertIn("wooden cabinet", response["answer"])

    def test_unsupported_generated_values_are_not_exposed_as_answers_or_metadata(self):
        assistant = self.assistant(FakeGenerator("9999 unsupported widgets"))
        index = assistant.index_document(self.text)
        response = assistant.answer(index, "Where is the amber widget stored?")
        self.assertEqual(response["answer_mode"], "grounded_extractive_quote")
        self.assertNotIn("9999", json.dumps(response))
        self.assertTrue(response_supported(response, index.document))

    def test_missing_subject_returns_fallback_even_if_a_span_model_would_guess(self):
        assistant = self.assistant()
        index = assistant.index_document(self.text)
        response = assistant.answer(index, "What is the retail price?")
        self.assertEqual(response["status"], "not_found")
        self.assertEqual(response["answer"], MISSING_ANSWER)
        self.assertEqual(response["citations"], [])

    def test_no_answer_model_margin_causes_fallback(self):
        assistant = self.assistant(answerability=FakeAnswerability(margin=-5))
        index = assistant.index_document(self.text)
        self.assertEqual(assistant.answer(index, "Where is the amber widget stored?")["answer"], MISSING_ANSWER)

    def test_frequency_question_cannot_be_answered_with_a_storage_description(self):
        assistant = self.assistant()
        index = assistant.index_document(self.text)
        response = assistant.answer(index, "How often is the amber widget stored?")
        self.assertEqual(response["status"], "not_found")

    def test_explicit_warning_section_uses_only_the_source_section(self):
        assistant = self.assistant(FakeGenerator("invented instruction"))
        index = assistant.index_document("Warnings\nDo not press the red button.\nPurpose\nThe amber widget is for display.")
        response = assistant.answer(index, "What does the document say about warnings?")
        self.assertEqual(response["answer"], "Do not press the red button.")
        self.assertEqual(response["answer_mode"], "grounded_concise_quote")
        self.assertTrue(response_supported(response, index.document))

    def test_natural_pregnancy_questions_quote_the_present_source_section(self):
        assistant = self.assistant(FakeGenerator("yes it is safe"))
        note = "The uploaded source records a pregnancy note without a definitive safety finding."
        index = assistant.index_document("Warnings\nKeep the amber widget dry.\nPregnancy\n" + note)
        for question in ("What does the document say about pregnancy?", "Is it safe during pregnancy?",
                         "What is the pregnancy information?", "Can I use this while pregnant?"):
            with self.subTest(question=question):
                response = assistant.answer(index, question)
                self.assertEqual(response["status"], "answered")
                self.assertEqual(response["citations"][0]["quote"], note)
                self.assertIn(note, response["answer"])
                self.assertNotIn("yes it is safe", json.dumps(response))
                self.assertTrue(response_supported(response, index.document))

    def test_pregnancy_or_breastfeeding_question_accepts_one_present_topic(self):
        assistant = self.assistant(FakeGenerator("invented advice"))
        for heading, note in (("Pregnancy", "The source contains a pregnancy note only."),
                              ("Pregnancy Or Breast Feeding", "The source contains a breast-feeding note only.")):
            index = assistant.index_document(heading + "\n" + note)
            for question in ("What does the document say about pregnancy or breast-feeding?",
                             "What does the document say about pregnancy or breast feeding?",
                             "What does the document say about breast-feeding or pregnancy?"):
                with self.subTest(heading=heading, question=question):
                    response = assistant.answer(index, question)
                    self.assertEqual(response["status"], "answered")
                    self.assertEqual(response["answer"], note)
                    self.assertEqual(response["question_coverage"], 1.0)
                    self.assertTrue(response_supported(response, index.document))
                    self.assertNotIn("invented advice", json.dumps(response))

    def test_pregnancy_question_with_an_absent_specific_subject_does_not_get_a_generic_answer(self):
        assistant = self.assistant()
        index = assistant.index_document("Pregnancy\nThe uploaded source records a pregnancy note.")
        for question in ("Is amoxicillin safe during pregnancy?", "Is amoxicillin safe during pregnancy or breast-feeding?"):
            with self.subTest(question=question):
                response = assistant.answer(index, question)
                self.assertEqual(response["status"], "not_found")

    def test_source_clause_expansion_does_not_include_next_bullet(self):
        text = "Use the amber widget. • Keep it in the wooden cabinet • Avoid the blue shelf."
        first = text.index("wooden cabinet")
        quote, start, end = supported_quote(text, first, first + len("wooden cabinet"))
        self.assertNotIn("•", quote)
        self.assertEqual(text[start:end], quote)

    def test_tampered_document_index_is_rejected(self):
        assistant = self.assistant()
        index = assistant.index_document(self.text)
        directory = self.root / "data" / "indexes" / index.metadata["document_id"]
        (directory / "document.txt").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            DocumentIndex.load(directory)


if __name__ == "__main__":
    unittest.main()
