"""Cached exact RxNorm ingredient relationships for duplicate-aware evaluation."""

import hashlib
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from medical_doc_assistant.storage import utc_now, write_json

LOGGER = logging.getLogger(__name__)
BASE_URL = "https://rxnav.nlm.nih.gov/REST"
NLM_ATTRIBUTION = (
    "This product uses publicly available data from the U.S. National Library of Medicine (NLM), "
    "National Institutes of Health, Department of Health and Human Services; NLM is not responsible "
    "for the product and does not endorse or recommend this or any other product."
)


class RxNormClient:
    """Globally throttle concurrent public requests to at most ten per second."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.last_request = 0.0

    def fetch(self, session: requests.Session, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        for attempt in range(4):
            with self.lock:
                pause = 0.1 - (time.monotonic() - self.last_request)
                if pause > 0:
                    time.sleep(pause)
                self.last_request = time.monotonic()
            try:
                response = session.get(BASE_URL + path, params=params, timeout=(10, 45))
            except (requests.Timeout, requests.ConnectionError):
                if attempt == 3:
                    raise RuntimeError("RxNorm connection failed after four attempts.") from None
                time.sleep(2 ** attempt)
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 3:
                    raise RuntimeError(f"RxNorm repeatedly returned HTTP {response.status_code}.")
                time.sleep(2 ** attempt)
                continue
            if not response.ok:
                raise RuntimeError(f"RxNorm returned HTTP {response.status_code} for {path}.")
            return response.json()
        raise RuntimeError("RxNorm request failed.")

    def resolve(self, name: str, version: str) -> dict[str, Any]:
        with requests.Session() as session:
            session.headers["User-Agent"] = "medical-doc-assistant-educational/0.1"
            # search=2 is exact plus normalized matching; it does not use approximate matching.
            found = self.fetch(session, "/rxcui.json", {"name": name, "search": 2})
            codes = sorted(set((found.get("idGroup") or {}).get("rxnormId") or []))
            result = {
                "source_name": name,
                "retrieved_at_utc": utc_now(),
                "rxnorm_version": version,
                "status": "not_found" if not codes else "ambiguous_name",
                "find_result": found,
            }
            if len(codes) != 1:
                return result
            related = self.fetch(session, f"/rxcui/{codes[0]}/related.json", {"tty": "IN"})
            result["matched_rxcui"] = codes[0]
            result["ingredient_result"] = related
            concepts = [concept for group in (related.get("relatedGroup") or {}).get("conceptGroup", [])
                        if group.get("tty") == "IN" for concept in (group.get("conceptProperties") or [])]
            if len(concepts) == 1:
                result.update(status="matched", ingredient_rxcui=concepts[0]["rxcui"],
                              ingredient_name=concepts[0]["name"])
            else:
                result["status"] = "ambiguous_or_missing_ingredient"
            return result


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    write_json(temporary, payload)
    temporary.replace(path)


def collect_ingredient_reference(frame: pd.DataFrame, path: Path, workers: int = 4) -> dict[str, Any]:
    names = sorted({name for values in frame["substance_names"] for name in json.loads(values)})
    digest = hashlib.sha256(json.dumps(names, ensure_ascii=False).encode("utf-8")).hexdigest()
    if path.exists():
        reference = json.loads(path.read_text(encoding="utf-8"))
        if reference.get("schema_version") != 1:
            raise ValueError("Unsupported RxNorm reference schema.")
        if reference.get("source_names_sha256") != digest:
            raise ValueError("The cached RxNorm reference covers a different corpus. Collect a separate reference for the new corpus.")
        if set(names) <= set(reference["entries"]):
            LOGGER.info("Using cached exact RxNorm reference: %s names", len(names))
            return reference
    else:
        reference = {
            "schema_version": 1,
            "source": "NLM RxNorm API, RXNORM ingredient (IN) relationships",
            "endpoint": BASE_URL,
            "documentation_url": "https://lhncbc.nlm.nih.gov/RxNav/APIs/api-RxNorm.getRelatedByType.html",
            "terms_url": "https://lhncbc.nlm.nih.gov/RxNav/TermsofService.html",
            "attribution": NLM_ATTRIBUTION,
            "source_names_sha256": digest,
            "created_at_utc": utc_now(),
            "entries": {},
        }
    client = RxNormClient()
    with requests.Session() as session:
        version_payload = client.fetch(session, "/version.json")
    version = version_payload["version"]
    if reference.get("rxnorm_version") and reference["rxnorm_version"] != version:
        raise ValueError("The RxNorm release changed during collection. Preserve the old reference and start a fresh one.")
    reference["rxnorm_version"] = version
    reference["version_result"] = version_payload
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = [name for name in names if name not in reference["entries"]]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(client.resolve, name, version): name for name in pending}
        try:
            for future in as_completed(futures):
                name = futures[future]
                reference["entries"][name] = future.result()
                if len(reference["entries"]) % 25 == 0:
                    LOGGER.info("RxNorm ingredient reference: %s / %s names", len(reference["entries"]), len(names))
                    _atomic_json(path, reference)
        finally:
            reference["entries"] = dict(sorted(reference["entries"].items()))
            reference["complete"] = set(names) <= set(reference["entries"])
            reference["updated_at_utc"] = utc_now()
            _atomic_json(path, reference)
    matched = sum(entry["status"] == "matched" for entry in reference["entries"].values())
    LOGGER.info("RxNorm reference complete: %s / %s names have an unambiguous ingredient relationship", matched, len(names))
    return reference


def canonical_ingredient_tokens(row: dict[str, Any], reference: dict[str, Any]) -> set[str]:
    return {
        "rxnorm_in:" + reference["entries"][name]["ingredient_rxcui"]
        for name in json.loads(row["substance_names"])
        if reference["entries"].get(name, {}).get("status") == "matched"
    }
