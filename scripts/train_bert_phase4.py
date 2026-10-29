"""Run the long BERT training step with persistent progress/status for local review."""

import logging
import os
from pathlib import Path

from medical_doc_assistant.bert_training import run_bert_phase4
from medical_doc_assistant.hf_assets import load_phase4_config
from medical_doc_assistant.storage import utc_now, write_json


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    directory = root / "reports" / "phase4" / "bert"
    directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(directory / "training.log", encoding="utf-8"), logging.StreamHandler()])
    status = directory / "process_status.json"
    write_json(status, {"status": "running", "pid": os.getpid(), "started_at_utc": utc_now()})
    try:
        result = run_bert_phase4(root, load_phase4_config(root / "configs" / "phase4.json"))
        write_json(status, {"status": "completed", "pid": os.getpid(), "completed_at_utc": utc_now(), "test_metrics": result["test_metrics"]})
    except Exception as error:
        logging.exception("BERT training failed")
        write_json(status, {"status": "failed", "pid": os.getpid(), "error": str(error), "failed_at_utc": utc_now()})
        raise
