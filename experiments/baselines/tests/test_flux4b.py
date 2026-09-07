"""CPU-only validation of the FLUX wrapper; no real model assets are read."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.baselines import fetch_assets
from experiments.baselines.flux4b.fetch_weights import fetch_manifest
from experiments.baselines.flux4b.probe import ORIGINAL_RENDERER_SHA256, source_provenance, validate_bounds


class FluxFetchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.output = self.base / "weights"
        self.payload = b"tiny verified fixture"
        self.manifest = {"repo": "example/repo", "hf_revision": "pinned", "anonymous": True,
                         "bytes": len(self.payload), "license": "test-fixture",
                         "files": [{"path": "model.bin", "bytes": len(self.payload),
                                    "sha256": hashlib.sha256(self.payload).hexdigest(),
                                    "expected_lfs_sha256": None}]}

    def test_download_reuses_safe_helper_and_preserves_flux_manifest_schema(self):
        self.output.mkdir()
        outside = self.base / "preserve.txt"
        outside.write_bytes(b"preserve old partial symlink target")
        (self.output / "model.bin.part").symlink_to(outside)
        def download(command, check):
            self.assertEqual(command[:2], ["curl", "-q"])
            temporary = Path(command[command.index("--output") + 1])
            self.assertNotEqual(temporary.name, "model.bin.part")
            temporary.write_bytes(self.payload)
        with patch.object(fetch_assets.subprocess, "run", side_effect=download):
            fetch_manifest(self.manifest, self.output)
        self.assertEqual(json.loads((self.output / "manifest.json").read_text()), self.manifest)
        self.assertEqual(outside.read_bytes(), b"preserve old partial symlink target")

    def test_verify_missing_is_read_only_and_never_downloads(self):
        with patch.object(fetch_assets.subprocess, "run") as download:
            with self.assertRaises(FileNotFoundError):
                fetch_manifest(self.manifest, self.output, verify_only=True)
            download.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_verify_existing_does_not_write_manifest(self):
        self.output.mkdir()
        (self.output / "model.bin").write_bytes(self.payload)
        old = b'{"existing": "manifest"}\n'
        (self.output / "manifest.json").write_bytes(old)
        timestamp = (self.output / "manifest.json").stat().st_mtime_ns
        with patch.object(fetch_assets.subprocess, "run") as download:
            fetch_manifest(self.manifest, self.output, verify_only=True)
            download.assert_not_called()
        self.assertEqual((self.output / "manifest.json").read_bytes(), old)
        self.assertEqual((self.output / "manifest.json").stat().st_mtime_ns, timestamp)

    def test_mismatched_existing_file_is_never_overwritten(self):
        self.output.mkdir()
        target = self.output / "model.bin"
        target.write_bytes(b"existing user file")
        for verify_only in (False, True):
            with self.subTest(verify_only=verify_only), patch.object(fetch_assets.subprocess, "run") as download:
                with self.assertRaises(FileExistsError):
                    fetch_manifest(self.manifest, self.output, verify_only=verify_only)
                download.assert_not_called()
                self.assertEqual(target.read_bytes(), b"existing user file")

    def test_verify_rejects_symlink_escape(self):
        self.output.mkdir()
        outside = self.base / "outside.bin"
        outside.write_bytes(self.payload)
        (self.output / "model.bin").symlink_to(outside)
        with patch.object(fetch_assets.subprocess, "run") as download:
            with self.assertRaises(ValueError):
                fetch_manifest(self.manifest, self.output, verify_only=True)
            download.assert_not_called()
        self.assertEqual(outside.read_bytes(), self.payload)


class FluxProbeTests(unittest.TestCase):
    def test_rejects_nonpositive_or_oversized_arguments_before_assets(self):
        for values in ((0, 432, 4, 1), (-16, -16, 4, 1), (768, 0, 4, 1),
                       (768, 432, 0, 1), (768, 432, -1, 1), (768, 432, 65, 1),
                       (768, 432, 4, 0), (768, 432, 4, -1), (768, 432, 4, 2**32),
                       (769, 432, 4, 1), (1024, 1024, 4, 1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_bounds(*values)

    def test_valid_defaults_and_bounds(self):
        validate_bounds(768, 432, 4, 20260907)
        validate_bounds(16, 16, 1, 1)
        validate_bounds(432, 768, 64, 2**32 - 1)

    def test_custom_input_is_not_labeled_original_renderer(self):
        self.assertIn("Caller-supplied", source_provenance("0" * 64))
        self.assertIn("verified by SHA-256", source_provenance(ORIGINAL_RENDERER_SHA256))


if __name__ == "__main__":
    unittest.main()
