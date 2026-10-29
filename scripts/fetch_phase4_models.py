"""Prepare the four pinned local model runtimes used in Phase 4."""

import logging
from pathlib import Path

from medical_doc_assistant.hf_assets import ensure_model_assets, load_phase4_config


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = Path(__file__).resolve().parents[1]
    ensure_model_assets(root, load_phase4_config(root / "configs" / "phase4.json"))
