"""CLI entry points for phase-by-phase data and model development."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from medical_doc_assistant.cleaning import clean_dataset
from medical_doc_assistant.collection import collect_data
from medical_doc_assistant.labels import load_config
from medical_doc_assistant.validation import validate_exports

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Medical Document Assistant: phased data and model pipelines")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Project root containing configs/phase1.json")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("collect", "Download or reuse a verified openFDA snapshot"),
        ("clean", "Build clean Parquet/CSV from the local raw snapshot"),
        ("eda", "Validate the clean dataset and regenerate EDA reports"),
        ("validate", "Check the clean dataset and CSV/Parquet exports"),
        ("phase1", "Run collection, cleaning, validation, and EDA"),
        ("prepare-phase2", "Freeze ingredient- and duplicate-aware evaluation partitions"),
        ("phase2", "Train, compare, save, and verify the three TF-IDF baseline models"),
        ("validate-phase2", "Verify the frozen baseline experiment and its artifacts"),
        ("classify", "Classify English leaflet text with the saved baseline"),
        ("phase3", "Tune, train, evaluate, and verify the PyTorch TextCNN"),
        ("validate-phase3", "Verify the neural experiment and checkpoint"),
        ("classify-neural", "Classify English leaflet text with the saved PyTorch model"),
        ("phase4", "Run BERT comparison, grounded QA evaluation, and Phase 4 verification"),
        ("validate-phase4", "Verify BERT and document-grounded QA artifacts"),
        ("classify-bert", "Classify text with the fine-tuned compact BERT"),
        ("index-document", "Create a per-document FAISS index for text"),
        ("ask", "Ask a grounded question about one document"),
        ("chat", "Interactive document-grounded question answering"),
    ):
        subparser = commands.add_parser(name, help=help_text)
        if name in {"collect", "phase1"}:
            subparser.add_argument("--per-class", type=int, help="Maximum raw records per class")
            subparser.add_argument("--refresh", action="store_true", help="Replace the cached raw snapshot")
        if name in {"classify", "classify-neural", "classify-bert"}:
            inputs = subparser.add_mutually_exclusive_group(required=True)
            inputs.add_argument("--text", help="Pasted document text")
            inputs.add_argument("--file", type=Path, help="UTF-8 text file")
            subparser.add_argument("--model", type=Path, help="Saved model artifact; defaults to the phase's selected model")
        if name == "phase3":
            subparser.add_argument("--device", choices=["cpu", "cuda", "auto"], help="Override the configured training device")
        if name in {"index-document", "chat"}:
            subparser.add_argument("--file", type=Path, required=True, help="UTF-8 document text file")
        if name == "ask":
            document = subparser.add_mutually_exclusive_group(required=True)
            document.add_argument("--file", type=Path, help="UTF-8 text file to index and question")
            document.add_argument("--document-id", help="SHA-256 document ID from a saved index")
            subparser.add_argument("--question", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = args.root.resolve()
    config = load_config(root / "configs" / "phase1.json")
    if args.command in {"phase4", "validate-phase4", "classify-bert", "index-document", "ask", "chat"}:
        import json
        from medical_doc_assistant.hf_assets import load_phase4_config

        phase4 = load_phase4_config(root / "configs" / "phase4.json")
        if args.command == "classify-bert":
            from medical_doc_assistant.bert_model import classify_bert_text

            text = args.text if args.text is not None else args.file.read_text(encoding="utf-8-sig")
            print(json.dumps(classify_bert_text(args.model or root / "models" / "phase4" / "bert" / "model.pt", text), indent=2))
        elif args.command in {"index-document", "ask", "chat"}:
            from medical_doc_assistant.document_index import DocumentIndex
            from medical_doc_assistant.grounded_qa import DocumentAssistant

            assistant = DocumentAssistant(root, phase4)
            if args.command == "ask" and args.document_id:
                import re
                if not re.fullmatch(r"[0-9a-f]{64}", args.document_id):
                    raise ValueError("Use the 64-character SHA-256 document ID from index-document.")
                index = DocumentIndex.load(root / "data" / "indexes" / args.document_id)
            else:
                index = assistant.index_document(args.file.read_text(encoding="utf-8-sig"), args.file.name)
            if args.command == "index-document":
                print(json.dumps(index.metadata, indent=2))
            elif args.command == "ask":
                print(json.dumps(assistant.answer(index, args.question), indent=2, ensure_ascii=False))
            else:
                print("Document ready. Ask questions; type /quit to stop.")
                while True:
                    question = input("You: ").strip()
                    if question == "/quit":
                        break
                    if question:
                        response = assistant.answer(index, question)
                        print("Assistant: " + response["answer"])
                        for citation in response["citations"]:
                            print("Source: " + citation["chunk_id"] + " — " + citation["quote"])
        else:
            from medical_doc_assistant.phase4_validation import validate_phase4

            if args.command == "phase4":
                from medical_doc_assistant.bert_training import run_bert_phase4
                from medical_doc_assistant.rag_evaluation import evaluate_rag
                from medical_doc_assistant.phase4_reporting import generate_phase4_report

                run_bert_phase4(root, phase4)
                evaluate_rag(root, phase4, "calibration")
                evaluate_rag(root, phase4, "evaluation")
                verification = validate_phase4(root, phase4)
                generate_phase4_report(root, phase4, verification, config["classes"])
                LOGGER.info("Phase 4 ready for review. See reports/phase_4_review.md")
            else:
                verification = validate_phase4(root, phase4)
            LOGGER.info("Passed %s Phase 4 artifact checks", len(verification["checks"]))
        return
    if args.command in {"phase3", "validate-phase3", "classify-neural"}:
        from medical_doc_assistant.deep_learning import load_phase3_config

        phase3 = load_phase3_config(root / "configs" / "phase3.json")
        if args.command == "classify-neural":
            import json
            from medical_doc_assistant.neural_model import classify_neural_text

            text = args.text if args.text is not None else args.file.read_text(encoding="utf-8-sig")
            result = classify_neural_text(args.model or root / "models" / "phase3" / "text_cnn.pt", text)
            print(json.dumps(result, indent=2))
        else:
            from medical_doc_assistant.phase3_validation import validate_phase3

            if args.command == "phase3":
                from medical_doc_assistant.deep_learning import run_phase3
                from medical_doc_assistant.phase3_reporting import generate_phase3_report

                if args.device:
                    phase3["device"] = args.device
                manifest = run_phase3(root, phase3)
                verification = validate_phase3(root, phase3)
                generate_phase3_report(root, manifest, verification, config["classes"])
                LOGGER.info("Phase 3 ready for review. See reports/phase_3_review.md")
            else:
                verification = validate_phase3(root, phase3)
            LOGGER.info("Passed %s Phase 3 experiment checks", len(verification["checks"]))
        return
    if args.command in {"prepare-phase2", "phase2", "validate-phase2", "classify"}:
        from medical_doc_assistant.splitting import load_phase2_config, prepare_phase2

        phase2 = load_phase2_config(root / "configs" / "phase2.json")
        if args.command == "prepare-phase2":
            frame, _ = prepare_phase2(root, phase2, [item["label"] for item in config["classes"]])
            LOGGER.info("Phase 2 partitions: %s", frame["split"].value_counts().to_dict())
        elif args.command == "classify":
            import json
            from medical_doc_assistant.baseline import classify_text

            text = args.text if args.text is not None else args.file.read_text(encoding="utf-8-sig")
            result = classify_text(args.model or root / "models" / "phase2" / "benchmark.joblib", text)
            print(json.dumps(result, indent=2))
        else:
            from medical_doc_assistant.phase2_validation import validate_phase2

            if args.command == "phase2":
                from medical_doc_assistant.baseline import MODEL_NAMES, run_phase2
                from medical_doc_assistant.phase2_reporting import generate_phase2_report

                manifest = run_phase2(root, phase2, [item["label"] for item in config["classes"]])
                verification = validate_phase2(root, phase2)
                generate_phase2_report(root, manifest, verification, config["classes"], MODEL_NAMES)
                LOGGER.info("Phase 2 ready for review: %s. See reports/phase_2_review.md", MODEL_NAMES[manifest["selected_model"]])
            else:
                verification = validate_phase2(root, phase2)
            LOGGER.info("Passed %s Phase 2 integrity checks", len(verification["checks"]))
        return
    raw_dir = root / "data" / "raw"
    output_dir = root / "data" / "processed"
    report_dir = root / "reports"
    raw_path = raw_dir / "openfda_labels.jsonl.gz"
    if args.command in {"collect", "phase1"}:
        collect_data(config, raw_path, raw_dir / "openfda_manifest.json",
                     per_class=args.per_class, refresh=args.refresh)
    if args.command in {"clean", "phase1"}:
        frame = clean_dataset(config, raw_path, output_dir, report_dir)
        LOGGER.info("Saved %s clean documents to %s", len(frame), output_dir)
    if args.command in {"validate", "eda", "phase1"}:
        if args.command != "phase1":
            frame = pd.read_parquet(output_dir / "medicine_labels.parquet")
        validation = validate_exports(frame, config, output_dir, report_dir)
        LOGGER.info("Passed %s dataset/export integrity checks", len(validation["checks"]))
    if args.command in {"eda", "phase1"}:
        from medical_doc_assistant.eda import generate_eda

        summary = generate_eda(config, frame, raw_dir, report_dir)
        LOGGER.info("Phase 1 ready for review: %s documents, %s classes. See reports/phase_1_review.md",
                    summary["documents"], summary["classes"])


if __name__ == "__main__":
    main()
