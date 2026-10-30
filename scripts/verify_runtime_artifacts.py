"""Fail early when a clone is missing trained classifiers or has Git LFS pointers."""

import argparse
import hashlib
import json
from pathlib import Path


def verify(root: Path, manifest_path: Path) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not manifest.get("artifacts"):
        raise ValueError("Invalid runtime artifact manifest.")
    errors = []
    for artifact in manifest["artifacts"]:
        path = root / artifact["path"]
        if not path.is_file():
            errors.append(artifact["path"] + ": missing")
            continue
        size = path.stat().st_size
        if size < 1024 and path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1"):
            errors.append(artifact["path"] + ": Git LFS pointer only")
            continue
        if size != artifact["bytes"]:
            errors.append(artifact["path"] + f": expected {artifact['bytes']} bytes, found {size}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != artifact["sha256"]:
            errors.append(artifact["path"] + ": SHA-256 mismatch")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = args.manifest or root / "configs" / "runtime_artifacts.json"
    errors = verify(root, manifest)
    if errors:
        details = "\n".join("- " + error for error in errors)
        raise SystemExit(
            "Required trained model artifacts are unavailable:\n" + details +
            "\n\nThis repository stores trained classifiers with Git LFS. Run from the repository root:\n"
            "  git lfs install\n  git lfs pull\n  python scripts/verify_runtime_artifacts.py\n"
            "Then rebuild with: docker compose up -d --build\n"
        )
    total = sum(item["bytes"] for item in json.loads(manifest.read_text(encoding="utf-8"))["artifacts"])
    print(f"Verified {total / (1024 * 1024):.1f} MiB of trained runtime artifacts.")


if __name__ == "__main__":
    main()
