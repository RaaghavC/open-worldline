"""Download safety regression tests. Network calls are replaced with tiny bytes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


source = Path(__file__).resolve().parents[1] / "fetch_assets.py"
spec = importlib.util.spec_from_file_location("baseline_fetch_assets", source)
fetch_assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_assets)


class FetchTests(unittest.TestCase):
    payload = b"verified test fixture"

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.output = self.base / "assets"
        self.output.mkdir()
        self.artifact = {"path": "model.bin", "size": len(self.payload),
                         "sha256": hashlib.sha256(self.payload).hexdigest()}

    def fetch(self, artifact=None):
        fetch_assets.fetch("example/repo", "pinned-revision", [artifact or self.artifact], self.output)

    def download(self, command, check):
        self.assertIn("--proto-redir", command)
        Path(command[command.index("--output") + 1]).write_bytes(self.payload)

    def test_verified_download_publishes_and_reuses_bytes(self):
        with patch.object(fetch_assets.subprocess, "run", side_effect=self.download) as run:
            self.fetch()
            self.fetch()
            self.assertEqual(run.call_count, 1)
        self.assertEqual((self.output / "model.bin").read_bytes(), self.payload)
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["artifacts"], [self.artifact])

    def test_existing_mismatch_is_preserved(self):
        target = self.output / "model.bin"
        target.write_bytes(b"user file")
        with patch.object(fetch_assets.subprocess, "run") as run:
            with self.assertRaises(FileExistsError):
                self.fetch()
            run.assert_not_called()
        self.assertEqual(target.read_bytes(), b"user file")

    def test_parent_traversal_is_rejected(self):
        artifact = dict(self.artifact, path="../outside.bin")
        with patch.object(fetch_assets.subprocess, "run") as run:
            with self.assertRaises(ValueError):
                self.fetch(artifact)
            run.assert_not_called()

    def test_symlinked_parent_cannot_escape_output(self):
        outside = self.base / "outside"
        outside.mkdir()
        (self.output / "nested").symlink_to(outside, target_is_directory=True)
        with patch.object(fetch_assets.subprocess, "run") as run:
            with self.assertRaises(ValueError):
                self.fetch(dict(self.artifact, path="nested/model.bin"))
            run.assert_not_called()
        self.assertEqual(list(outside.iterdir()), [])

    def test_old_partial_and_manifest_symlinks_do_not_overwrite_targets(self):
        outside = self.base / "preserve.txt"
        outside.write_bytes(b"preserved")
        (self.output / "model.bin.download").symlink_to(outside)
        (self.output / "manifest.json").symlink_to(outside)
        with patch.object(fetch_assets.subprocess, "run", side_effect=self.download):
            self.fetch()
        self.assertEqual(outside.read_bytes(), b"preserved")
        self.assertFalse((self.output / "manifest.json").is_symlink())
        self.assertEqual((self.output / "model.bin").read_bytes(), self.payload)

    def test_failed_hash_cleans_temporary_file(self):
        def corrupt(command, check):
            Path(command[command.index("--output") + 1]).write_bytes(b"corrupt")
        with patch.object(fetch_assets.subprocess, "run", side_effect=corrupt):
            with self.assertRaises(ValueError):
                self.fetch()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_file_created_during_download_is_not_replaced(self):
        def racing_download(command, check):
            self.download(command, check)
            (self.output / "model.bin").write_bytes(b"new user file")
        with patch.object(fetch_assets.subprocess, "run", side_effect=racing_download):
            with self.assertRaises(FileExistsError):
                self.fetch()
        self.assertEqual((self.output / "model.bin").read_bytes(), b"new user file")
        self.assertEqual([p.name for p in self.output.iterdir()], ["model.bin"])


if __name__ == "__main__":
    unittest.main()
