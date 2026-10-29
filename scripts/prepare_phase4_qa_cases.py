"""Prepare and summarize the independent source-linked medical QA benchmark."""

import json
import argparse
from collections import Counter
from pathlib import Path

from medical_doc_assistant.qa_cases import prepare_qa_cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = prepare_qa_cases(root, refresh=args.refresh)
    print(json.dumps({"sources": data["sources"], "cases": len(data["cases"]),
                      "partitions": dict(Counter(case["partition"] for case in data["cases"]))}, indent=2))
