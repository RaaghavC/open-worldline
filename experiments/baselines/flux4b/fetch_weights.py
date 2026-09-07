"""Explicitly download this baseline's external 4-bit weights, or verify them."""
import argparse
import json
from pathlib import Path
import sys

try:
    from ..fetch_assets import fetch
except ImportError:  # Direct execution from any current working directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fetch_assets import fetch


def fetch_manifest(manifest, output, *, verify_only=False):
    if not manifest["anonymous"] or not 0 <= manifest["bytes"] < 8 * 1000**3:
        raise ValueError("Expected the recorded anonymous download below 8 GB")
    if sum(entry["bytes"] for entry in manifest["files"]) != manifest["bytes"]:
        raise ValueError("Manifest byte total does not match its files")
    artifacts = [{"path": entry["path"], "size": entry["bytes"], "sha256": entry["sha256"]}
                 for entry in manifest["files"]]
    fetch(manifest["repo"], manifest["hf_revision"], artifacts, output,
          verify_only=verify_only, manifest_payload=manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true", help="Check local files without downloading.")
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name("weights-manifest.json").read_text())
    fetch_manifest(manifest, args.output, verify_only=args.verify_only)


if __name__ == "__main__":
    main()
