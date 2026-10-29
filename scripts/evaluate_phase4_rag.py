"""Run the source-linked QA calibration or held-out evaluation partition."""

import argparse
import json
from pathlib import Path

from medical_doc_assistant.hf_assets import load_phase4_config
from medical_doc_assistant.rag_evaluation import evaluate_rag


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=["development", "calibration", "evaluation"], default="calibration")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(evaluate_rag(root, load_phase4_config(root / "configs" / "phase4.json"), args.partition), indent=2))
