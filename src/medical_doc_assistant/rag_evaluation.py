"""Evaluate answerability, exact source support, and reference QA evidence."""

import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from medical_doc_assistant.grounded_qa import DocumentAssistant, MISSING_ANSWER, RAG_POLICY_VERSION, normalized_quote
from medical_doc_assistant.qa_language import concise_response_supported
from medical_doc_assistant.qa_cases import prepare_qa_cases
from medical_doc_assistant.splitting import config_digest
from medical_doc_assistant.storage import file_sha256, utc_now, write_json


def answer_tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"\w+", text.lower()) if token not in {"a", "an", "the"}]


def token_f1(expected: str, actual: str) -> float:
    first, second = Counter(answer_tokens(expected)), Counter(answer_tokens(actual))
    intersection = sum((first & second).values())
    if not first or not second:
        return float(first == second)
    if not intersection:
        return 0.0
    precision, recall = intersection / sum(second.values()), intersection / sum(first.values())
    return 2 * precision * recall / (precision + recall)


def response_supported(response: dict[str, Any], document: str) -> bool:
    if response["status"] == "not_found":
        return response["answer"] == MISSING_ANSWER and response["citations"] == []
    if response.get("answer_style") == "concise":
        return concise_response_supported(response, document)
    return bool(response["citations"]) and all(document[citation["char_start"]:citation["char_end"]] == citation["quote"]
        for citation in response["citations"]) and any(citation["quote"] == response["answer"] for citation in response["citations"])


def evaluate_rag(root: Path, config: dict[str, Any], partition: str = "calibration") -> dict[str, Any]:
    cases = prepare_qa_cases(root)
    selected = [case for case in cases["cases"] if case["partition"] == partition]
    if not selected:
        raise ValueError("The selected QA evaluation partition is empty.")
    directory = root / "reports" / "phase4" / "rag"
    directory.mkdir(parents=True, exist_ok=True)
    cases_sha = file_sha256(root / "data" / "evaluation" / "phase4_qa_cases.json")
    metrics_path = directory / (partition + "_metrics.json")
    if metrics_path.exists():
        cached = json.loads(metrics_path.read_text(encoding="utf-8"))
        if cached.get("config_sha256") == config_digest(config) and cached.get("qa_cases_sha256") == cases_sha and cached.get("policy_version") == RAG_POLICY_VERSION:
            return cached
    if partition == "evaluation":
        plan = {"config_sha256": config_digest(config), "qa_cases_sha256": file_sha256(root / "data" / "evaluation" / "phase4_qa_cases.json"),
                "settings": config["rag"], "models": config["models"], "partition": partition, "policy_version": RAG_POLICY_VERSION}
        path = directory / "evaluation_plan.json"
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            old_version = previous.get("policy_version", 1)
            comparable = {key: value for key, value in previous.items() if key != "policy_version"}
            if comparable != {key: value for key, value in plan.items() if key != "policy_version"}:
                raise ValueError("Preserve the previous held-out QA experiment before changing its data/settings.")
            if old_version != RAG_POLICY_VERSION:
                archive = directory / "archive" / ("policy_v" + str(old_version))
                archive.mkdir(parents=True, exist_ok=True)
                for filename in ("evaluation_plan.json", "evaluation_results.csv", "evaluation_responses.json", "evaluation_metrics.json"):
                    if (directory / filename).exists():
                        shutil.copy2(directory / filename, archive / filename)
        write_json(path, plan)
    assistant = DocumentAssistant(root, config)
    indexes, results, responses = {}, [], []
    source_map = {source["record_id"]: source for source in cases["sources"]}
    for case in selected:
        source = source_map[case["source_record_id"]]
        if file_sha256(root / source["file"]) != source["sha256"]:
            raise ValueError("QA source text differs from its source manifest.")
        if case["source_record_id"] not in indexes:
            indexes[case["source_record_id"]] = assistant.index_document((root / source["file"]).read_text(encoding="utf-8"), source["generic_names"])
        index = indexes[case["source_record_id"]]
        response = assistant.answer(index, case["question"])
        answered = response["status"] == "answered"
        expected = case["expected_answer"]
        returned = response.get("extracted_span", response["answer"]) if answered else ""
        evidence = normalized_quote(expected) in normalized_quote(response["answer"]) if case["answerable"] and answered else False
        row = {"case_id": case["case_id"], "question": case["question"], "case_type": case["case_type"],
               "source_record_id": case["source_record_id"], "expected_answerable": case["answerable"], "answered": answered,
               "answerability_correct": answered == case["answerable"], "source_supported": response_supported(response, index.document),
               "reference_evidence_covered": evidence, "exact_match": normalized_quote(returned) == normalized_quote(expected) if case["answerable"] else not answered,
               "token_f1": token_f1(expected, returned) if case["answerable"] else float(not answered),
               "answer_mode": response["answer_mode"], "answer": response["answer"]}
        results.append(row)
        responses.append({"case_id": case["case_id"], "source_record_id": case["source_record_id"], "response": response})
    table = pd.DataFrame(results)
    positive = table.loc[table["expected_answerable"]]
    negative = table.loc[~table["expected_answerable"]]
    answered = table.loc[table["answered"]]
    tp = int(positive["answered"].sum())
    fp = int(negative["answered"].sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / len(positive) if len(positive) else 0.0
    metrics = {"partition": partition, "cases": len(table), "source_documents": len(indexes),
               "answerable_cases": len(positive), "missing_answer_cases": len(negative),
               "answerability_accuracy": float(table["answerability_correct"].mean()), "answer_precision": precision, "answer_recall": recall,
               "answerability_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
               "correct_fallback_rate": float((~negative["answered"]).mean()) if len(negative) else None,
               "false_answers_to_missing_questions": fp,
               "source_support_rate": float(answered["source_supported"].mean()) if len(answered) else 1.0,
               "positive_reference_evidence_coverage": float(positive["reference_evidence_covered"].mean()),
               "positive_exact_match": float(positive["exact_match"].mean()), "positive_token_f1": float(positive["token_f1"].mean()),
               "generation_verified_answers": int((table["answer_mode"] == "verified_llm_span").sum()),
               "policy_version": RAG_POLICY_VERSION,
               "qa_cases_sha256": cases_sha,
               "created_at_utc": utc_now(), "config_sha256": config_digest(config)}
    table.to_csv(directory / (partition + "_results.csv"), index=False)
    write_json(directory / (partition + "_responses.json"), responses)
    write_json(directory / (partition + "_metrics.json"), metrics)
    return metrics
