"""Collect the reproducible salt/ingredient reference used by the Phase 2 split."""

import logging
from pathlib import Path

import pandas as pd

from medical_doc_assistant.rxnorm import collect_ingredient_reference


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = Path(__file__).resolve().parents[1]
    frame = pd.read_parquet(root / "data" / "processed" / "medicine_labels.parquet")
    collect_ingredient_reference(frame, root / "data" / "reference" / "rxnorm_ingredient_map.json")
