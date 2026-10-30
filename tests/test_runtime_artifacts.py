"""Runtime artifacts fail clearly when missing, truncated, or left as LFS pointers."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_runtime_artifacts", ROOT / "scripts" / "verify_runtime_artifacts.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RuntimeArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".test-", dir=ROOT)
        self.root = Path(self.temporary.name)
        self.artifact = self.root / "models" / "model.pt"
        self.manifest = self.root / "manifest.json"

    def tearDown(self):
        self.temporary.cleanup()

    def write_manifest(self, content: bytes):
        self.manifest.write_text(json.dumps({"schema_version": 1, "artifacts": [{
            "path": "models/model.pt", "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}]}), encoding="utf-8")

    def test_valid_artifact_passes(self):
        content = b"trained artifact content"
        self.artifact.parent.mkdir(parents=True)
        self.artifact.write_bytes(content)
        self.write_manifest(content)
        self.assertEqual(MODULE.verify(self.root, self.manifest), [])

    def test_missing_artifact_is_reported(self):
        self.write_manifest(b"expected")
        self.assertEqual(MODULE.verify(self.root, self.manifest), ["models/model.pt: missing"])

    def test_lfs_pointer_is_reported_instead_of_accepted_as_a_model(self):
        pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1000\n"
        self.artifact.parent.mkdir(parents=True)
        self.artifact.write_bytes(pointer)
        self.write_manifest(pointer)
        self.assertEqual(MODULE.verify(self.root, self.manifest), ["models/model.pt: Git LFS pointer only"])

    def test_corrupt_size_is_reported_before_checksum(self):
        content = b"expected model bytes"
        self.artifact.parent.mkdir(parents=True)
        self.artifact.write_bytes(b"short")
        self.write_manifest(content)
        self.assertIn("expected", MODULE.verify(self.root, self.manifest)[0])


if __name__ == "__main__":
    unittest.main()
