"""Download pinned public model assets and record file hashes/licensing provenance."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import HfApi, hf_hub_download

from medical_doc_assistant.storage import file_sha256, utc_now, write_json

LOGGER = logging.getLogger(__name__)


def load_phase4_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config["device"] not in {"cpu", "cuda", "auto"}:
        raise ValueError("Unsupported Phase 4 schema/device.")
    if set(config["models"]) != {"bert", "embedding", "qa", "generation"}:
        raise ValueError("Define the four Phase 4 model roles.")
    if config["bert"]["max_length"] < 8 or config["bert"]["max_windows"] < 1:
        raise ValueError("Invalid BERT document-window dimensions.")
    if not 0 <= config["rag"]["overlap_tokens"] < config["rag"]["chunk_tokens"]:
        raise ValueError("Chunk overlap must be smaller than the chunk token limit.")
    return config


def ensure_model_assets(root: Path, config: dict[str, Any], roles: list[str] | None = None) -> dict[str, Path]:
    manifest_path = root / "reports" / "phase4" / "model_assets.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"schema_version": 1, "models": {}}
    locations = {}
    api = HfApi()
    for role in roles or list(config["models"]):
        specification = config["models"][role]
        destination = root / "models" / "phase4" / "pretrained" / role
        existing = manifest["models"].get(role)
        if existing:
            if existing["repo"] != specification["repo"] or existing["revision"] != specification["revision"]:
                raise ValueError(f"The frozen {role} assets use a different model revision.")
            if all((destination / name).exists() and file_sha256(destination / name) == checksum for name, checksum in existing["file_sha256"].items()):
                LOGGER.info("Using pinned local %s model: %s", role, specification["repo"])
                locations[role] = destination
                continue
        info = api.model_info(specification["repo"], revision=specification["revision"])
        if info.sha != specification["revision"]:
            raise ValueError("The provider returned a different model revision.")
        files = [entry.rfilename for entry in info.siblings]
        weight = "model.safetensors" if "model.safetensors" in files else "pytorch_model.bin"
        if weight not in files:
            raise ValueError(f"No supported PyTorch weights found for {role}.")
        wanted = [name for name in files if name == weight or name.endswith((".json", ".txt")) or name in {"README.md", "spiece.model", "tokenizer.model"}]
        wanted = [name for name in wanted if not name.startswith(("onnx/", "openvino/"))]
        destination.mkdir(parents=True, exist_ok=True)
        for name in wanted:
            for attempt in range(4):
                try:
                    hf_hub_download(specification["repo"], name, revision=specification["revision"],
                                    local_dir=destination, cache_dir=root / ".cache" / "huggingface")
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
        manifest["models"][role] = {**specification, "downloaded_at_utc": utc_now(),
                                    "model_card_url": "https://huggingface.co/" + specification["repo"],
                                    "weight_file": weight,
                                    "file_sha256": {name: file_sha256(destination / name) for name in wanted}}
        write_json(manifest_path, manifest)
        locations[role] = destination
        LOGGER.info("Downloaded %s assets at revision %s", role, info.sha)
    return locations
