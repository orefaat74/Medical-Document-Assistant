"""Download a bounded real-data snapshot from the openFDA drug-label API."""

import gzip
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import requests

from medical_doc_assistant.labels import build_search
from medical_doc_assistant.storage import file_sha256, utc_now, write_json

LOGGER = logging.getLogger(__name__)
API_URL = "https://api.fda.gov/drug/label.json"


class OpenFDAClient:
    def __init__(self, api_key: str | None = None, request_interval: float = 0.3):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "medical-doc-assistant-educational/0.1"
        self.api_key = api_key or os.getenv("OPENFDA_API_KEY")
        self.request_interval = request_interval
        self.last_request = 0.0

    def close(self) -> None:
        self.session.close()

    def fetch(self, params: dict[str, Any]) -> dict[str, Any]:
        query = dict(params)
        if self.api_key:
            query["api_key"] = self.api_key
        for attempt in range(5):
            delay = self.request_interval - (time.monotonic() - self.last_request)
            if delay > 0:
                time.sleep(delay)
            try:
                self.last_request = time.monotonic()
                response = self.session.get(API_URL, params=query, timeout=(15, 90))
            except (requests.Timeout, requests.ConnectionError):
                if attempt == 4:
                    raise RuntimeError("openFDA connection failed after five attempts.") from None
                time.sleep(2 ** (attempt + 1))
                continue
            if response.status_code == 404:
                payload = response.json()
                if payload.get("error", {}).get("code") == "NOT_FOUND":
                    return {"meta": {}, "results": []}
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 4:
                    raise RuntimeError(f"openFDA returned HTTP {response.status_code} repeatedly.")
                retry_after = response.headers.get("Retry-After", "")
                seconds = float(retry_after) if retry_after.isdigit() else 2 ** (attempt + 1)
                LOGGER.warning("openFDA HTTP %s; retrying in %ss", response.status_code, seconds)
                time.sleep(seconds)
                continue
            if not response.ok:
                # Do not include response.url: it could contain a user's API key.
                raise RuntimeError(f"openFDA request failed with HTTP {response.status_code}.")
            return response.json()
        raise RuntimeError("openFDA request failed.")


def collect_data(
    config: dict[str, Any],
    raw_path: Path,
    manifest_path: Path,
    *,
    per_class: int | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    target = per_class if per_class is not None else config["per_class"]
    if target < 1 or target > 25000:
        raise ValueError("per_class must be between 1 and 25000.")
    settings = {
        "per_class": target,
        "page_size": config["page_size"],
        "sort": "effective_time:desc",
        "classes": config["classes"],
    }
    if raw_path.exists() and not refresh:
        if not manifest_path.exists():
            raise ValueError("Cached raw data has no manifest. Use --refresh to collect a new snapshot.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["collection_settings"] != settings:
            raise ValueError("Cached collection settings differ. Use --refresh for the requested snapshot.")
        if file_sha256(raw_path) != manifest["raw_sha256"]:
            raise ValueError("Raw snapshot checksum mismatch. Restore it or use --refresh.")
        LOGGER.info("Using cached snapshot: %s records", manifest["unique_raw_records"])
        return manifest

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = raw_path.with_name(raw_path.name + ".partial")
    manifest = {
        "source": "openFDA drug label API",
        "endpoint": API_URL,
        "retrieved_at_utc": utc_now(),
        "is_synthetic": False,
        "language_scope": "English-first US human prescription and OTC labels",
        "collection_settings": settings,
        "queries": [],
        "terms_url": "https://open.fda.gov/terms/",
        "documentation_url": "https://open.fda.gov/apis/drug/label/",
        "attribution": "Data provided by the U.S. Food and Drug Administration (https://open.fda.gov).",
    }
    seen: set[str] = set()
    client = OpenFDAClient()
    try:
        with temporary.open("wb") as file_handle:
            with gzip.GzipFile(filename="", mode="wb", fileobj=file_handle, mtime=0) as compressed:
                for definition in config["classes"]:
                    search = build_search(definition["epc_terms"])
                    retrieved = 0
                    available = None
                    page_updates: set[str] = set()
                    while retrieved < target and (available is None or retrieved < available):
                        limit = min(config["page_size"], target - retrieved)
                        payload = client.fetch({
                            "search": search,
                            "sort": settings["sort"],
                            "skip": retrieved,
                            "limit": limit,
                        })
                        meta = payload.get("meta", {})
                        if meta.get("last_updated"):
                            page_updates.add(meta["last_updated"])
                        records = payload.get("results", [])
                        available = meta.get("results", {}).get("total", 0)
                        if not records:
                            break
                        for row in records:
                            # Cross-class combinations can occur in several queries.
                            key = row.get("id") or json.dumps(row, sort_keys=True, ensure_ascii=False)
                            if key not in seen:
                                seen.add(key)
                                compressed.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
                        retrieved += len(records)
                        LOGGER.info("%s: %s / %s raw labels", definition["label"], retrieved, min(target, available))
                        if len(records) < limit:
                            break
                    manifest["queries"].append({
                        "class_label": definition["label"],
                        "search": search,
                        "available_records": available,
                        "retrieved_records": retrieved,
                        "api_last_updated": sorted(page_updates),
                    })
        if not seen:
            raise RuntimeError("openFDA returned no records for the configured classes.")
        temporary.replace(raw_path)
    finally:
        client.close()
        if temporary.exists():
            temporary.unlink()
    manifest["unique_raw_records"] = len(seen)
    manifest["completed_at_utc"] = utc_now()
    manifest["raw_sha256"] = file_sha256(raw_path)
    manifest["raw_file"] = raw_path.name
    write_json(manifest_path, manifest)
    return manifest
